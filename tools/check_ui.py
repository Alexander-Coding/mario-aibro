"""Проверка игры и интерфейса без видимых окон.

    .venv/bin/python -m tools.check_ui            # все проверки
    .venv/bin/python -m tools.check_ui --no-draw  # без невидимого окна (только логика)
    .venv/bin/python -m tools.check_ui --shots DIR  # вдобавок сохранить снимки экранов в DIR

Что проверяется:

1. PlaySession крутит мир фиксированным шагом: прогон уровня 1-1 ботом
   (tools/heuristic_bot) через session.update с dt = 1/60 и с «рваным» dt
   0.005..0.05 даёт тот же world.summary(), что прямой прогон GameWorld.
2. Не больше 4 шагов мира за кадр; фронт прыжка не теряется в коротком кадре.
3. Автопереход ИИ на следующий уровень через 3 с и итоговая таблица
   (манифест на 2 уровня подкладывается подменой core.levels.load_levels).
4. Рекорды пишутся и читаются (временный файл внутри tools/, потом удаляется).
5. Нет модели ИИ — сеанс не падает, управление с клавиатуры.
6. Настоящий контроллер ai.agent (если в ai/models есть модель) играет через сеанс.
7. Дымовой тест отрисовки: arcade.Window(visible=False), on_draw во всех
   состояниях (игра, флаг, итоги, смерть, пауза, панель ИИ, таблица, меню,
   экран уровней). Видимое окно не открывается никогда.

Настоящий records.json не трогается: все сеансы получают Records с путём
во временной папке tools/_check_ui_tmp, которая удаляется в конце.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

from core import levels as core_levels
from core import rules
from core.levels import LevelInfo, make_world
from core.world import FLAGPOLE, Inputs
from records import AI, HUMAN, Records
from tools.heuristic_bot import HeuristicBot
from ui import ai_control
from ui.session import (
    AI_ADVANCE_DELAY,
    ALL_DONE,
    DEAD,
    LEVEL_DONE,
    MAX_STEPS_PER_FRAME,
    PLAYING,
    PlaySession,
)

REPO = Path(__file__).resolve().parent.parent
TMP_DIR = REPO / "tools" / "_check_ui_tmp"

# Страховка от вечного цикла: 1-1 бот проходит за ~2 тыс. кадров
MAX_FRAMES = 20000


# ===============================================================
# ЗАГЛУШКА КОНТРОЛЛЕРА ИИ
# ===============================================================


class BotController:
    """Контроллер по контракту ai.agent (reset/act/overlay) поверх HeuristicBot.

    overlay() отдаёт правдоподобный словарь — чтобы панель ИИ было чем
    заполнить: действие, вероятности, опасность, ожидаемые очки.
    """

    name = "Эвристика (заглушка)"

    def __init__(self):
        self.bot = HeuristicBot()
        self.calls = 0
        self._last: Inputs = Inputs()
        self._danger = 0.0

    def reset(self, world) -> None:
        self.bot.reset()
        self._last = Inputs()

    def act(self, world) -> Inputs:
        self.calls += 1
        self._last = self.bot.act(world)

        player = world.player
        nearest = min(
            (abs(enemy.center_x - player.center_x) for enemy in world.enemies if enemy.is_alive),
            default=900.0,
        )
        self._danger = max(0.0, min(1.0, 1 - nearest / 400))
        return self._last

    def overlay(self) -> dict:
        jump = self._last.jump
        probs = {
            "вправо+прыжок": 0.81 if jump else 0.14,
            "вправо": 0.12 if jump else 0.78,
            "стоять": 0.04,
            "влево": 0.03,
        }
        return {
            "model": self.name,
            "action": "вправо+прыжок" if jump else "вправо",
            "probs": probs,
            "danger": self._danger,
            "expected": 4120.0,
            "confidence": max(probs.values()),
        }


# ===============================================================
# ПОМОЩНИКИ
# ===============================================================


class Report:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append((name, bool(passed), detail))
        mark = "OK  " if passed else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
        return passed

    @property
    def failed(self) -> list[str]:
        return [name for name, passed, _ in self.results if not passed]


def tmp_records(name: str) -> Records:
    """Рекорды во временном файле внутри репозитория (удаляется в конце)."""

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
    return Records(path)


def level_1_1() -> LevelInfo:
    for level in core_levels.load_levels():
        if level.id == "1-1":
            return level
    return LevelInfo(id="1-1", file="custom_level.json", name="1-1")


def direct_run(level: LevelInfo) -> dict:
    """Эталон: GameWorld тем же ботом, без сеанса и аккумулятора."""

    world = make_world(level)
    controller = BotController()
    controller.reset(world)
    while not world.done and world.frame < MAX_FRAMES:
        world.step(controller.act(world))
    return world.summary()


def run_session(session: PlaySession, dt_source) -> tuple[int, float]:
    """Крутит сеанс, пока идёт уровень. Возвращает (кадров окна, секунд «реального» времени)."""

    frames = 0
    elapsed = 0.0
    while session.state == PLAYING and session.world.frame < MAX_FRAMES:
        dt = dt_source()
        session.update(dt)
        frames += 1
        elapsed += dt
    return frames, elapsed


# ===============================================================
# 1. ФИКСИРОВАННЫЙ ШАГ: ИГРА НЕ ЗАВИСИТ ОТ FPS
# ===============================================================


def check_fixed_step(report: Report) -> None:
    level = level_1_1()

    t0 = time.perf_counter()
    reference = direct_run(level)
    direct_s = time.perf_counter() - t0
    report.check(
        "прямой прогон 1-1 ботом завершился",
        reference["frames"] < MAX_FRAMES,
        f"score={reference['score']} won={reference['won']} death={reference['death']} "
        f"frames={reference['frames']} flag={reference['flag_fraction']} ({direct_s:.2f} с)",
    )

    # dt = 1/60: ровно один шаг на кадр
    controller = BotController()
    session = PlaySession(controller=controller, levels=[level], records=tmp_records("fixed"))
    frames, _ = run_session(session, lambda: 1 / 60)
    summary = session.world.summary()
    report.check(
        "сеанс, dt=1/60: summary совпадает с прямым прогоном",
        summary == reference,
        f"кадров окна {frames}, шагов мира {summary['frames']}, вызовов act {controller.calls}",
    )
    report.check(
        "ИИ опрашивается на каждом шаге мира",
        controller.calls == summary["frames"],
        f"act() {controller.calls} раз на {summary['frames']} шагов",
    )
    report.check(
        "после финиша — экран итогов уровня",
        session.state == (LEVEL_DONE if reference["won"] else DEAD),
        f"state={session.state}",
    )

    # Рваный dt 0.005..0.05
    rng = random.Random(20260923)
    controller = BotController()
    session = PlaySession(controller=controller, levels=[level], records=tmp_records("ragged"))
    frames, elapsed = run_session(session, lambda: rng.uniform(0.005, 0.05))
    summary = session.world.summary()
    report.check(
        "сеанс, рваный dt 0.005..0.05: summary совпадает",
        summary == reference,
        f"кадров окна {frames} на {summary['frames']} шагов мира "
        f"(прошло {elapsed:.2f} с, мир отсчитал {summary['frames'] * rules.FIXED_DT:.2f} с)",
    )
    if summary != reference:
        diff = {k: (summary.get(k), reference.get(k)) for k in reference if summary.get(k) != reference.get(k)}
        print("      расхождение:", diff)

    # Остальные уровни манифеста — тем же способом, рваным dt
    others = [info for info in core_levels.load_levels() if info.id != level.id]
    mismatched = []
    lines = []
    for info in others:
        reference = direct_run(info)
        session = PlaySession(controller=BotController(), levels=[info], records=tmp_records(f"lvl-{info.id}"))
        rng = random.Random(info.id)
        run_session(session, lambda: rng.uniform(0.005, 0.05))
        if session.world.summary() != reference:
            mismatched.append(info.id)
        outcome = "победа" if reference["won"] else reference["death"]
        lines.append(f"{info.id}: {reference['score']} ({outcome})")
    if others:
        report.check(
            f"остальные уровни манифеста ({len(others)}): сеанс = прямой прогон",
            not mismatched,
            "; ".join(lines) + (f"; расходятся: {mismatched}" if mismatched else ""),
        )


# ===============================================================
# 2. ШАГИ ЗА КАДР И ФРОНТ ПРЫЖКА
# ===============================================================


def check_steps_and_jump(report: Report) -> None:
    level = level_1_1()

    session = PlaySession(levels=[level], records=tmp_records("steps"))
    frame_before = session.world.frame
    steps = session.update(0.5, Inputs())
    report.check(
        f"не больше {MAX_STEPS_PER_FRAME} шагов мира за кадр",
        steps == MAX_STEPS_PER_FRAME and session.world.frame - frame_before == MAX_STEPS_PER_FRAME,
        f"dt=0.5 → шагов {steps}",
    )
    # Долг времени не копится: следующий обычный кадр — обычное число шагов
    steps_next = session.update(1 / 60, Inputs())
    report.check("хвост длинного кадра не догоняется рывком", steps_next <= 2, f"следующий кадр: {steps_next} шаг(а)")

    # Встаём на землю
    for _ in range(30):
        session.update(1 / 60, Inputs())
    player = session.world.player
    on_ground_y = player.center_y

    # Нажатие пришло в кадре короче шага мира — шага нет, фронт ждёт
    steps = session.update(0.004, Inputs(jump=True))
    pending_kept = session._jump_pending and steps == 0
    session.update(1 / 60, Inputs())
    session.update(1 / 60, Inputs())
    jumped = session.world.player.center_y > on_ground_y + 5
    report.check(
        "фронт прыжка из короткого кадра тратится на первом шаге",
        pending_kept and jumped and not session._jump_pending,
        f"ждал шага: {pending_kept}, подпрыгнул: {jumped} (y {on_ground_y:.0f} → {session.world.player.center_y:.0f})",
    )

    # Удержание прыжка ничего не делает: jump=True только на первом шаге
    session = PlaySession(levels=[level], records=tmp_records("steps2"))
    for _ in range(30):
        session.update(1 / 60, Inputs())
    captured: list[Inputs] = []
    world = session.world
    original_step = world.step

    def spy(inputs):
        captured.append(inputs)
        return original_step(inputs)

    world.step = spy
    session.update(4 / 60, Inputs(right=True, jump=True))
    world.step = original_step
    report.check(
        "в кадре из нескольких шагов прыжок только на первом",
        [i.jump for i in captured] == [True] + [False] * (len(captured) - 1) and all(i.right for i in captured),
        f"шаги: {[('R' if i.right else '-') + ('J' if i.jump else '') for i in captured]}",
    )


# ===============================================================
# 3. АВТОПЕРЕХОД ИИ И ИТОГОВАЯ ТАБЛИЦА
# ===============================================================


TWO_LEVELS = [
    LevelInfo(id="check-1", file="custom_level.json", name="Проверка 1", time_limit=200),
    # 3 секунды: бот не успеет — смерть по времени
    LevelInfo(id="check-2", file="custom_level.json", name="Проверка 2", time_limit=3),
]


def check_auto_advance(report: Report) -> None:
    original = core_levels.load_levels
    core_levels.load_levels = lambda split=None: list(TWO_LEVELS)

    try:
        records = tmp_records("advance")
        controller = BotController()
        session = PlaySession(controller=controller, records=records)

        report.check("манифест подменён: 2 уровня", session.level_count == 2,
                     f"уровни {[level.id for level in session.levels]}")

        run_session(session, lambda: 1 / 60)
        first_state = session.state
        first_score = session.world.score

        # Ждём автопереход: чуть меньше 3 с — ещё на итогах, потом — дальше
        waited = 0.0
        while waited < AI_ADVANCE_DELAY - 0.1:
            session.update(1 / 60)
            waited += 1 / 60
        still_on_results = session.state == first_state and session.index == 0
        while session.index == 0 and waited < AI_ADVANCE_DELAY + 1:
            session.update(1 / 60)
            waited += 1 / 60

        report.check(
            "ИИ сам идёт на следующий уровень через 3 с",
            first_state in (LEVEL_DONE, DEAD) and still_on_results and session.index == 1
            and session.state == PLAYING,
            f"уровень 1: {first_state}, {first_score} очков; переход через {waited:.2f} с",
        )

        run_session(session, lambda: 1 / 60)
        second = session.last_result
        report.check(
            "второй уровень: смерть по времени, причина по-русски",
            session.state == DEAD and second is not None and second.death == "time"
            and second.death_text == "Время вышло",
            f"state={session.state}, death={second.death if second else None}",
        )

        waited = 0.0
        while session.state != ALL_DONE and waited < AI_ADVANCE_DELAY + 1:
            session.update(1 / 60)
            waited += 1 / 60

        rows = session.run_results()
        total = session.total_score
        report.check(
            "после последнего уровня — итоговая таблица с суммой",
            session.state == ALL_DONE and len(rows) == 2 and total == sum(r.score for r in rows),
            f"строк {len(rows)}: " + ", ".join(f"{r.level_id}={r.score}({'победа' if r.won else r.death})" for r in rows)
            + f"; сумма {total}",
        )

        # Рекорды обоих уровней записаны как рекорды ИИ
        reread = Records(records.path)
        report.check(
            "рекорды ИИ записаны по концу каждого уровня",
            reread.best_score("check-1", AI) == rows[0].score
            and reread.best_score("check-2", AI) == rows[1].score
            and reread.get("check-1", HUMAN) is None,
            f"check-1={reread.best_score('check-1', AI)}, check-2={reread.best_score('check-2', AI)}",
        )

        # Человек: Enter — дальше, R — заново
        human = PlaySession(records=tmp_records("human"))
        human.world.player.center_y = -500  # падение в пропасть
        human.update(1 / 60, Inputs())
        dead_state = human.state
        human.update(10.0, Inputs())  # у человека автоперехода нет
        stays = human.state == DEAD
        human.restart_level()
        restarted = human.state == PLAYING and human.index == 0 and human.world.frame == 0
        human.world.player.center_y = -500
        human.update(1 / 60, Inputs())
        human.next_level()
        advanced = human.index == 1 and human.state == PLAYING
        human.world.player.center_y = -500
        human.update(1 / 60, Inputs())
        human.next_level()
        report.check(
            "человек: смерть в яме, R — заново, Enter — дальше, после последнего — таблица",
            dead_state == DEAD and human.last_result is not None and stays and restarted and advanced
            and human.state == ALL_DONE,
            f"причина: {human.results[0].death_text}; ждёт клавишу: {stays}",
        )
    finally:
        core_levels.load_levels = original


# ===============================================================
# 4. РЕКОРДЫ
# ===============================================================


def check_records(report: Report) -> None:
    records = tmp_records("records")
    path = records.path

    first = records.update("1-1", HUMAN, 1200, False, date="2026-09-23T10:00:00")
    worse = records.update("1-1", HUMAN, 900, True)
    better = records.update("1-1", HUMAN, 3400, True, date="2026-09-23T11:00:00")
    same_but_won = records.update("1-1", AI, 500, False) and records.update("1-1", AI, 500, True)

    reread = Records(path)
    human = reread.get("1-1", HUMAN)
    report.check(
        "рекорд: пишется, хуже — не затирает, лучше — затирает",
        first and not worse and better and human == {"score": 3400, "won": True, "date": "2026-09-23T11:00:00"},
        f"в файле: {human}",
    )
    report.check(
        "рекорд: при равном счёте победа лучше смерти",
        same_but_won and reread.get("1-1", AI)["won"] is True,
        f"ИИ: {reread.get('1-1', AI)}",
    )
    report.check("кто набрал больше", reread.leader("1-1") == HUMAN, f"лидер 1-1: {reread.leader('1-1')}")

    path.write_text("{ битый json", encoding="utf-8")
    broken = Records(path)
    report.check("битый файл рекордов не роняет игру", broken.data == {}, "прочитан как пустой")

    report.check(
        "настоящий records.json тестами не трогается",
        all(str(r.path).startswith(str(TMP_DIR)) for r in (records, broken)),
        f"временная папка {TMP_DIR.relative_to(REPO)}",
    )


# ===============================================================
# 5. НЕТ МОДЕЛИ ИИ
# ===============================================================


def check_ai_missing(report: Report) -> None:
    level = level_1_1()

    session = PlaySession(
        ai_mode=True,
        levels=[level],
        records=tmp_records("missing"),
        load_controller=lambda: (None, ai_control.AI_MISSING_TEXT),
    )
    session.update(1 / 60, Inputs(right=True))
    x0 = session.world.player.center_x
    for _ in range(30):
        session.update(1 / 60, Inputs(right=True))
    report.check(
        "нет модели: подсказка на экране, управление с клавиатуры",
        not session.ai_active and session.ai_error == ai_control.AI_MISSING_TEXT
        and session.world.player.center_x > x0 + 50,
        f"ai_error='{session.ai_error}', игрок ушёл вправо на {session.world.player.center_x - x0:.0f} px",
    )

    def broken_loader():
        raise RuntimeError("битый файл модели")

    session = PlaySession(ai_mode=True, levels=[level], records=tmp_records("broken"),
                          load_controller=broken_loader)
    report.check("загрузчик упал — сеанс жив", not session.ai_active and bool(session.ai_error),
                 session.ai_error.split("\n")[-1])

    class Exploding(BotController):
        def act(self, world):
            if world.frame > 10:
                raise ValueError("сломался посреди уровня")
            return super().act(world)

    session = PlaySession(controller=Exploding(), levels=[level], records=tmp_records("explode"))
    for _ in range(40):
        session.update(1 / 60, Inputs(right=True))
    report.check(
        "контроллер упал посреди уровня — управление у клавиатуры",
        not session.ai_active and session.state == PLAYING and "Ошибка ИИ" in (session.ai_error or ""),
        f"шагов мира {session.world.frame}",
    )

    # Настоящий ленивый импорт ai.agent: есть он или нет — не падает
    t0 = time.perf_counter()
    controller, error = ai_control.load_ai_controller()
    report.check(
        "ленивая загрузка ai.agent не бросает исключений",
        (controller is not None) != (error is not None),
        ("модель загружена: " + ai_control.controller_name(controller)) if controller is not None
        else f"модели нет: «{error.splitlines()[0]}»" + (f" / {error.splitlines()[1]}" if "\n" in error else "")
        + f" ({time.perf_counter() - t0:.2f} с)",
    )


# ===============================================================
# 6. НАСТОЯЩИЙ КОНТРОЛЛЕР ai.agent (если есть хоть какая-то модель)
# ===============================================================


def find_real_controller():
    """Контроллер RL-части: обычная модель, иначе самый свежий *.pt в ai/models.

    Возвращает (controller, откуда) или (None, почему нет). Модели может
    не быть вовсе — это не ошибка интерфейса, проверка тогда пропускается.
    """

    controller, error = ai_control.load_ai_controller()
    if controller is not None:
        return controller, "модель по умолчанию"

    models = sorted((REPO / "ai" / "models").glob("*.pt"), key=lambda path: path.stat().st_mtime)
    for path in reversed(models):
        controller, error = ai_control.load_ai_controller(str(path))
        if controller is not None:
            return controller, path.name

    return None, (error or "нет файлов модели").splitlines()[0]


def check_real_controller(report: Report):
    controller, source = find_real_controller()
    if controller is None:
        report.check("настоящий ai.agent: пропущено", True, source)
        return None

    level = level_1_1()
    session = PlaySession(controller=controller, levels=[level], records=tmp_records("real"))
    t0 = time.perf_counter()
    for _ in range(240):
        session.update(1 / 60)
        if session.state != PLAYING:
            break
    elapsed = time.perf_counter() - t0
    info = session.ai_overlay()
    keys = sorted(info) if info else []
    report.check(
        "настоящий ai.agent играет через PlaySession",
        session.ai_active and info is not None and {"model", "action", "probs"} <= set(keys),
        f"{source}: {session.world.frame} шагов за {elapsed:.2f} с "
        f"({elapsed / max(session.world.frame, 1) * 1000:.1f} мс на шаг), overlay: {keys}",
    )
    return controller


# ===============================================================
# 7. ОТРИСОВКА В НЕВИДИМОМ ОКНЕ
# ===============================================================


def make_hidden_window(width: int = 1280, height: int = 720):
    """Невидимое окно arcade. Видимое не создаётся ни при каком исходе."""

    import arcade

    # macOS: приложение без значка в доке и без активации — фокус у пользователя не забираем
    if sys.platform == "darwin":
        try:
            from pyglet.libs.darwin.cocoapy import ObjCClass

            ObjCClass("NSApplication").sharedApplication().setActivationPolicy_(2)
        except Exception:  # noqa: BLE001
            pass

    window = arcade.Window(width, height, "check_ui", visible=False)
    if window.visible:
        window.close()
        raise RuntimeError("окно оказалось видимым — закрыто")
    return window


def check_draw(report: Report, shots_dir: Path | None, real_controller=None) -> None:
    try:
        import arcade

        window = make_hidden_window()
    except Exception as error:  # noqa: BLE001
        report.check("невидимое окно arcade", False, f"создать нельзя: {type(error).__name__}: {error}")
        return

    from game import GameView
    from main_menu import MenuView
    from ui.level_select import LevelSelectView

    drawn: dict[str, float] = {}
    menu_calls: list[str] = []

    def shot(name: str, view) -> None:
        """on_draw вида + (по желанию) снимок в PNG."""

        t0 = time.perf_counter()
        view.on_draw()
        drawn[name] = (time.perf_counter() - t0) * 1000
        if shots_dir is not None:
            shots_dir.mkdir(parents=True, exist_ok=True)
            arcade.get_image(0, 0, window.width, window.height).save(shots_dir / f"{name}.png")

    def fail(name: str) -> None:
        report.check(f"отрисовка: {name}", False, traceback.format_exc(limit=3).strip().splitlines()[-1])
        traceback.print_exc()

    level = level_1_1()

    # --- Игра человеком: движение, прыжок, пауза ---
    try:
        session = PlaySession(levels=[level], records=tmp_records("draw-human"))
        view = GameView(return_to_menu=lambda: menu_calls.append("menu"), session=session)
        window.show_view(view)

        x0 = session.world.player.center_x
        view.on_key_press(arcade.key.D, 0)
        for _ in range(90):
            view.on_update(1 / 60)
        moved = session.world.player.center_x - x0

        view.on_update(1 / 60)
        y0 = session.world.player.center_y
        view.on_key_press(arcade.key.SPACE, 0)
        for _ in range(8):
            view.on_update(1 / 60)
        rose = session.world.player.center_y - y0
        view.on_key_release(arcade.key.D, 0)

        shot("play", view)
        camera_x = view.camera.position.x

        view.on_key_press(arcade.key.ESCAPE, 0)
        frame = session.world.frame
        for _ in range(10):
            view.on_update(1 / 60)
        frozen = session.world.frame == frame and view.paused
        shot("pause", view)
        view.on_key_press(arcade.key.ESCAPE, 0)

        report.check(
            "клавиши: D — вправо, Space — прыжок, Esc — пауза",
            moved > 100 and rose > 20 and frozen and not view.paused,
            f"ушёл вправо на {moved:.0f} px, подпрыгнул на {rose:.0f} px, на паузе мир стоит: {frozen}; "
            f"камера x={camera_x:.0f}",
        )

        # Смерть в яме → экран смерти, R — заново
        session.world.player.center_y = -500
        view.on_update(1 / 60)
        session.state_time = 1.5
        shot("death", view)
        view.on_key_press(arcade.key.R, 0)
        report.check("экран смерти → R — уровень заново", session.state == PLAYING and session.world.frame == 0,
                     f"причина была: {session.results[0].death_text}")

        view.on_key_press(arcade.key.ESCAPE, 0)
        view.on_key_press(arcade.key.ESCAPE, 0)
        session.world.player.center_y = -500
        view.on_update(1 / 60)
        view.on_key_press(arcade.key.ESCAPE, 0)
        report.check("Esc на экране смерти — в меню", menu_calls == ["menu"], f"вызовы: {menu_calls}")
    except Exception:  # noqa: BLE001
        fail("игра человеком")

    # --- ИИ: панель, флаг, итоги уровня, итоговая таблица ---
    original = core_levels.load_levels
    core_levels.load_levels = lambda split=None: list(TWO_LEVELS)
    try:
        session = PlaySession(controller=BotController(), records=tmp_records("draw-ai"))
        view = GameView(return_to_menu=lambda: menu_calls.append("menu"), session=session)
        window.show_view(view)

        for _ in range(200):
            view.on_update(1 / 60)
        shot("ai_panel", view)

        while session.world.phase != FLAGPOLE and session.state == PLAYING:
            view.on_update(1 / 60)
        for _ in range(20):
            view.on_update(1 / 60)
        banner = session.flag_banner
        shot("flag", view)
        report.check(
            "флаг: крупная надпись с высотой и очками",
            banner is not None and banner.points == session.world.flag_bonus,
            f"высота {banner.percent if banner else '?'}%, +{banner.points if banner else '?'}",
        )

        kinds = set()
        while session.state == PLAYING:
            view.on_update(1 / 60)
            kinds.update(p.kind for p in session.popups)
        session.state_time = 2.2
        shot("results", view)
        report.check(
            "всплывающие очки за флаг и бонус за время",
            {"flag", "time_bonus"} <= kinds,
            f"виды всплывашек в конце уровня: {sorted(kinds)}",
        )

        view.on_key_press(arcade.key.ENTER, 0)  # не ждём 3 с
        for _ in range(400):
            view.on_update(1 / 60)
            if session.state != PLAYING:
                break
        for _ in range(int(AI_ADVANCE_DELAY * 60) + 10):
            view.on_update(1 / 60)
        session.state_time = 1.0
        shot("all_done", view)
        report.check("итоговая таблица нарисована", session.state == ALL_DONE,
                     f"сумма {session.total_score}")
    except Exception:  # noqa: BLE001
        fail("режим ИИ")
    finally:
        core_levels.load_levels = original

    # --- Панель ИИ с настоящей моделью (если нашлась) ---
    if real_controller is not None:
        try:
            session = PlaySession(controller=real_controller, levels=[level], records=tmp_records("draw-real"))
            view = GameView(return_to_menu=lambda: None, session=session)
            window.show_view(view)
            for _ in range(150):
                view.on_update(1 / 60)
                if session.state != PLAYING:
                    break
            shot("ai_real", view)
        except Exception:  # noqa: BLE001
            fail("панель ИИ с настоящей моделью")

    # --- Режим ИИ без модели: «Загрузка ИИ…», потом подсказка ---
    try:
        session = PlaySession(
            ai_mode=True,
            levels=[level],
            records=tmp_records("draw-missing"),
            load_controller=lambda: (None, ai_control.AI_MISSING_TEXT),
            defer_ai=True,
        )
        view = GameView(return_to_menu=lambda: None, session=session)
        window.show_view(view)
        view.on_update(1 / 60)
        still_pending = session.ai_pending
        shot("ai_loading", view)
        view.on_update(1 / 60)
        view.on_update(1 / 60)
        shot("ai_missing", view)
        report.check(
            "режим ИИ без модели: «Загрузка…», затем подсказка про обучение",
            still_pending and not session.ai_pending and session.ai_error == ai_control.AI_MISSING_TEXT,
            "модель грузится после первого кадра",
        )
    except Exception:  # noqa: BLE001
        fail("ИИ без модели")

    # --- Меню и экран уровней ---
    try:
        menu = MenuView(open_editor=lambda: None, open_game=lambda **kw: menu_calls.append(("game", kw)),
                        open_levels=lambda: menu_calls.append("levels"))
        window.show_view(menu)
        shot("menu", menu)
        labels = [button.label for button in menu.buttons]
        menu.handle_action("levels")

        records = tmp_records("draw-levels")
        records.update("check-1", HUMAN, 5200, True)
        records.update("check-1", AI, 6100, True)
        records.update("check-2", HUMAN, 800, False)
        chosen: list[dict] = []
        select = LevelSelectView(open_game=lambda **kw: chosen.append(kw), return_to_menu=lambda: None,
                                 records=records, levels=TWO_LEVELS)
        window.show_view(select)
        shot("levels", select)
        select.on_key_press(arcade.key.DOWN, 0)
        select.on_key_press(arcade.key.I, 0)
        select.on_key_press(arcade.key.UP, 0)
        select.on_key_press(arcade.key.ENTER, 0)

        report.check(
            "меню: кнопки автора + «УРОВНИ»",
            labels == ["ИГРАТЬ", "ИИ", "УРОВНИ", "РЕДАКТОР УРОВНЕЙ", "ВЫХОД"] and "levels" in menu_calls,
            f"кнопки: {labels}",
        )
        report.check(
            "экран уровней: I — ИИ с выбранного, Enter — играть",
            chosen == [{"ai_mode": True, "start_index": 1}, {"ai_mode": False, "start_index": 0}],
            f"запуски: {chosen}",
        )
    except Exception:  # noqa: BLE001
        fail("меню и экран уровней")

    report.check(
        "on_draw во всех состояниях без исключений",
        {"play", "pause", "death", "ai_panel", "flag", "results", "all_done", "ai_missing", "menu", "levels"}
        <= set(drawn),
        ", ".join(f"{name} {ms:.1f} мс" for name, ms in drawn.items()),
    )

    window.close()


# ===============================================================
# ЗАПУСК
# ===============================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверка игры и интерфейса без видимых окон")
    parser.add_argument("--no-draw", action="store_true", help="не создавать даже невидимое окно")
    parser.add_argument("--shots", type=Path, default=None, help="куда сохранить PNG-снимки экранов")
    args = parser.parse_args(argv)

    report = Report()
    t0 = time.perf_counter()

    try:
        check_fixed_step(report)
        check_steps_and_jump(report)
        check_auto_advance(report)
        check_records(report)
        check_ai_missing(report)
        real_controller = check_real_controller(report)
        if not args.no_draw:
            check_draw(report, args.shots, real_controller)
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    total = len(report.results)
    failed = report.failed
    print(f"\nИтого: {total - len(failed)}/{total} проверок прошло за {time.perf_counter() - t0:.1f} с")
    if failed:
        print("Не прошли:", "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

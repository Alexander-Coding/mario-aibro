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
8. Трубы и тайник на уровне-словаре (1-1 + своя комната, от карт не
   зависит): «вниз» человека и ИИ доходит до мира, баннер «Секретная
   комната!», удар по «?» — монета, «+100» над блоком и подскок блока, враг —
   «+N» над врагом (цепочка золотом), выход через боковую трубу, итоги
   «собрано из возможного» и они же в рекордах.
9. Редактор: save_map не теряет name/time_limit/warps и прочие ключи.
10. Панель ИИ: у каждого действия из ai.features.ACTION_LABELS (7 с «вниз»)
    свой значок, и все влезают в прежнюю ширину панели.
В отрисовке вдобавок: клавиши S/↓, порядок настоящих вызовов draw() в
трубе (игрок под стенами), тёмный фон и камера в комнате, всплывашки,
итоги с «из»; монета из «?» не закрывает «+100»; надписи экранов смерти
и итогов не вылезают из панели и не наезжают друг на друга — на
настоящих прогонах и на крайних числах (fit_cases).

Настоящий records.json не трогается: все сеансы получают Records с путём
во временной папке tools/_check_ui_tmp, которая удаляется в конце.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import traceback
import types
from pathlib import Path

from core import levels as core_levels
from core import rules
from core.levels import LevelInfo, make_world
from core.world import FLAGPOLE, PIPE_DOWN, PIPE_SIDE, PIPE_UP, Event, GameWorld, Inputs
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
    POPUP_COLORS,
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

    details = {"coins": 3, "coins_total": 20, "secrets": 1, "secrets_total": 2}
    with_details = records.update("2-2", AI, 700, True, details={**details, "junk": "x", "flag": True})
    reread = Records(path)
    old_style = Records(path)
    old_style.data = {}
    path.write_text(json.dumps({"1-1": {"human": {"score": 10, "won": False, "date": ""}}}), encoding="utf-8")
    old_style.reload()
    report.check(
        "рекорд: «собрано из возможного» пишется, старый records.json читается",
        with_details and reread.get("2-2", AI).get("details") == details
        and old_style.get("1-1", HUMAN) == {"score": 10, "won": False, "date": ""},
        f"details={reread.get('2-2', AI).get('details')}; старая запись: {old_style.get('1-1', HUMAN)}",
    )

    path.write_text("{ битый json", encoding="utf-8")
    broken = Records(path)
    report.check("битый файл рекордов не роняет игру", broken.data == {}, "прочитан как пустой")

    # Infinity в JSON читается как float('inf'), а int() от него — OverflowError
    path.write_text('{"1-1": {"human": {"score": 10, "won": false, "details": {"coins": Infinity, "coins_total": 5}}},'
                    ' "1-2": {"human": {"score": Infinity, "won": true}}}', encoding="utf-8")
    try:
        infinite = Records(path)
        survived = infinite.get("1-1", HUMAN)
        shown = f"1-1: {survived}; 1-2: {infinite.get('1-2', HUMAN)}"
    except Exception as error:  # noqa: BLE001
        survived, shown = None, f"упал: {type(error).__name__}: {error}"
    report.check(
        "Infinity в records.json (счёт или details) отбрасывается, а не роняет игру",
        survived is not None and survived.get("details") == {"coins_total": 5}
        and infinite.get("1-2", HUMAN) is None,
        shown,
    )

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
# 8. ТРУБЫ, ТАЙНИК, ВСПЛЫВАШКИ, ИТОГИ «ИЗ ВОЗМОЖНОГО»
# ===============================================================

# Уровень 1-1 + комната: труба-вход и труба-выход на 1-1 (как в эталонной
# проверке ядра), комната собрана здесь же — карты с тайниками пишет
# другая часть, от них проверка не зависит
WARP_LEVEL = LevelInfo(id="check-warp", file="custom_level.json", name="Проверка труб", time_limit=200)
WARP_PIPE = (1872.0, 304.0)
WARP_EXIT = (5872.0, 272.0)


def _tile(x: float, y: float, category: str, resource: str, texture: str) -> dict:
    return {"x": float(x), "y": float(y), "category": category, "resource": resource,
            "texture_path": texture}


def warp_room() -> dict:
    """Комната 20×14 тайлов: пол, стены, потолок, 4 монеты, блок «?», устье
    боковой трубы справа (низ устья = пол), Марио появляется слева."""

    bricks = "resources/Blocks/Bricks/img_1.png"
    tubes = "resources/Blocks/Tubes/"
    objects = []
    for tx in range(20):
        for ty in (0, 1):
            objects.append(_tile(32 * tx + 16, 32 * ty + 16, "Blocks", "Bricks",
                                 "resources/Blocks/Bricks/block_2.png"))
        objects.append(_tile(32 * tx + 16, 32 * 13 + 16, "Blocks", "Bricks", bricks))
    for ty in range(2, 13):
        objects.append(_tile(16, 32 * ty + 16, "Blocks", "Bricks", bricks))
        objects.append(_tile(32 * 19 + 16, 32 * ty + 16, "Blocks", "Bricks", bricks))
    # Устье (tx=17, ty=2..3), горизонтальное тело и вертикальная труба до потолка
    objects += [
        _tile(560, 80, "Blocks", "Tubes", tubes + "img_10.png"),
        _tile(560, 112, "Blocks", "Tubes", tubes + "img_11.png"),
        _tile(592, 80, "Blocks", "Tubes", tubes + "img_9.png"),
        _tile(592, 112, "Blocks", "Tubes", tubes + "img_8.png"),
    ]
    for ty in range(4, 13):
        objects.append(_tile(592, 32 * ty + 16, "Blocks", "Tubes", tubes + "img_2.png"))
    for tx in (5, 6, 7, 8):
        objects.append(_tile(32 * tx + 16, 144, "objects", "Coins", "resources/objects/Coins/img.png"))
    # Низ блока на 3 тайла выше пола — его можно выбить с места
    objects.append(_tile(304, 176, "objects", "ScoreBlock", "resources/objects/ScoreBlock/img_4.png"))
    objects.append(_tile(80, 240, "Entities", "mario", "resources/Entities/mario/idle/Mario.gif"))
    return {"theme": "underground", "objects": objects, "exit": [560, 80]}


def warp_level_data() -> dict:
    data = json.loads((core_levels.MAPS_DIR / "custom_level.json").read_text(encoding="utf-8"))
    data = {key: value for key, value in data.items() if key != "warps"}
    data["warps"] = [{"id": "check", "pipe": list(WARP_PIPE), "exit": list(WARP_EXIT), "room": warp_room()}]
    return data


def warp_session(**kwargs) -> PlaySession:
    data = warp_level_data()
    return PlaySession(
        levels=[WARP_LEVEL],
        world_factory=lambda info: GameWorld(data, name=info.name, time_limit=info.time_limit),
        **kwargs,
    )


def stand_on_pipe(world) -> None:
    """Игрок — прямо над серединой трубы-входа (дальше сам встанет на неё)."""
    warp = world.main.warps[0]
    world.player.center_x = warp.x
    world.player.bottom = warp.top + 2
    world.player.change_x = 0
    world.player.change_y = 0


def log_events(world) -> list:
    """Подменяет world.step: все события копятся в возвращённый список."""

    log: list = []
    original = world.step

    def step(inputs=Inputs()):
        events = original(inputs)
        log.extend(events)
        return events

    world.step = step
    return log


def spy_inputs(world) -> list:
    """Подменяет world.step: все Inputs, дошедшие до мира, копятся в список."""

    captured: list = []
    original = world.step

    def step(inputs=Inputs()):
        captured.append(inputs)
        return original(inputs)

    world.step = step
    return captured


def bump_room_block(session: PlaySession, advance) -> dict:
    """В комнате: встать под блок «?» и ударить головой. advance(inputs) — один кадр.

    Возвращает, что было на экране в кадр удара: монета, «+100», подскок.
    """

    world = session.world
    block = world.blocks[0]
    for _ in range(40):
        advance(Inputs())
    world.player.center_x = block.center_x
    for _ in range(5):
        advance(Inputs())

    hits = world.stats["block"]
    advance(Inputs(jump=True))
    seen: dict = {}
    for _ in range(60):
        if world.stats["block"] > hits:
            seen = {
                "block": block,
                "coins": list(session.block_coins),
                "popups": [p for p in session.popups if p.kind == "block"],
                "bumped": any(bump.block is block for bump in session.bumps),
            }
            break
        advance(Inputs())
    return seen


# Надпись «+100» (15 pt жирным) — примерно столько вниз от центра
POPUP_HALF_HEIGHT = 10.0


def coin_popup_gap(coins: list, popups: list) -> float | None:
    """Наименьший зазор (px) между верхом монеты из «?» и низом «+100» за жизнь монеты.

    Монета и надпись рождаются в один кадр; подъём надписи — как в
    hud.draw_popups (34 px за время жизни). Отрицательный — наезжают.
    """

    from ui import hud
    from ui.session import BLOCK_COIN_TIME, BlockCoin

    if not coins or not popups:
        return None
    coin, popup = coins[0], popups[0]
    gaps = []
    for i in range(61):
        age = BLOCK_COIN_TIME * i / 60
        flying = BlockCoin(x=coin.x, y=coin.y, age=age)
        coin_top = flying.y + flying.height + hud.COIN_SIZE / 2
        popup_bottom = popup.y + 34 * min(age / popup.lifetime, 1.0) - POPUP_HALF_HEIGHT
        gaps.append(popup_bottom - coin_top)
    return min(gaps)


def stomp_nearest_enemy(session: PlaySession, advance) -> list:
    """Ставит игрока над ближайшим живым врагом и ждёт прыжка на него.

    Возвращает события "enemy"/"stomp" за это время.
    """

    world = session.world
    log = log_events(world)
    player = world.player
    enemy = min(
        (enemy for enemy in world.enemies if enemy.is_alive),
        key=lambda enemy: abs(enemy.center_x - player.center_x),
    )
    player.center_x = enemy.center_x
    player.bottom = enemy.top + 30
    player.change_y = 0
    for _ in range(60):
        if any(event.kind in ("enemy", "stomp") for event in log):
            break
        if player.change_y <= 0 and player.bottom > enemy.top:
            player.center_x = enemy.center_x   # враг идёт — держимся над ним
        advance(Inputs())
    del world.step
    return [event for event in log if event.kind in ("enemy", "stomp")]


def check_pipes_and_effects(report: Report) -> None:
    session = warp_session(records=tmp_records("warp"))
    world = session.world
    step = lambda inputs=Inputs(): session.update(1 / 60, inputs)  # noqa: E731

    report.check(
        "уровень-словарь с тайником собран",
        world.totals.get("secrets") == 1 and "check" in world.areas and world.areas["check"].theme == "underground",
        f"totals={world.totals}",
    )

    # «вниз» человека: держится, как вправо/влево, и доходит до мира
    stand_on_pipe(world)
    for _ in range(20):
        step()
    hint = session.pipe_hint
    captured = spy_inputs(world)
    step(Inputs(down=True))
    del world.step
    report.check(
        "человек: Inputs(down=True) доходит до мира, на трубе — подсказка «↓»",
        bool(captured) and captured[0].down and hint and world.phase == PIPE_DOWN,
        f"до мира дошло {captured[:1]}, фаза {world.phase}",
    )

    frames = 0
    while not world.in_room and frames < 300:
        step(Inputs(down=True))
        frames += 1
    banner = session.banner
    report.check(
        "труба → секретная комната: баннер «СЕКРЕТНАЯ КОМНАТА!», тайник 1 из 1",
        world.in_room and banner is not None and banner.title == "СЕКРЕТНАЯ КОМНАТА!"
        and banner.subtitle == "Тайник 1 из 1",
        f"кадров в трубе {frames}, часть {world.area.key}, баннер: {banner.title if banner else None} / "
        f"{banner.subtitle if banner else None}",
    )

    seen = bump_room_block(session, step)
    block = seen.get("block")
    coin_ok = bool(seen.get("coins")) and all(
        abs(c.x - block.center_x) < 1 and abs(c.y - block.center_y) < 1 for c in seen["coins"]
    )
    popup_ok = bool(seen.get("popups")) and all(
        p.text == "+100" and abs(p.x - block.center_x) < 1 and p.y > block.top for p in seen["popups"]
    )
    report.check(
        "удар по «?»: монета вылетает из блока, «+100» над блоком, блок подпрыгивает",
        coin_ok and popup_ok and seen.get("bumped", False),
        f"монет {len(seen.get('coins', []))}, всплывашек {len(seen.get('popups', []))}, "
        f"подскок {seen.get('bumped')}" + (f", блок ({block.center_x:.0f}, {block.center_y:.0f})" if block else ""),
    )

    gap = coin_popup_gap(seen.get("coins", [])[:1], seen.get("popups", [])[:1])
    report.check(
        "удар по «?»: монета не закрывает «+100» за всю свою жизнь",
        gap is not None and gap > 0,
        "наименьший зазор между верхом монеты и низом надписи: "
        + ("—" if gap is None else f"{gap:.1f} px"),
    )

    phases = []
    frames = 0
    while (world.in_room or world.in_pipe) and frames < 600:
        step(Inputs(right=True))
        phases.append(world.phase)
        frames += 1
    exit_x = world.warps[0].exit_x
    report.check(
        "выход: устье боковой трубы → вылез из трубы-выхода на уровне",
        PIPE_SIDE in phases and PIPE_UP in phases and not world.in_room
        and abs(world.player.center_x - exit_x) < 8,
        f"фазы {sorted(set(phases))}, x={world.player.center_x:.0f} (труба-выход {exit_x:.0f})",
    )

    # Враг: очки сразу, «+N» над ним, а не над игроком
    events = stomp_nearest_enemy(session, step)
    event = events[0] if events else None
    popups = [p for p in session.popups if p.kind in ("enemy", "stomp", "chain")]
    report.check(
        "прыжок на врага: «+N» над врагом сразу в кадр прыжка",
        event is not None and any(
            p.text == f"+{event.points}" and abs(p.x - event.info["x"]) < 1
            and abs(p.y - (event.info["y"] + 24)) < 1
            for p in popups
        ),
        f"событие {event.kind if event else None} {event.info if event else ''}; "
        f"всплывашки {[(p.text, round(p.x), round(p.y)) for p in popups]}",
    )

    # Цепочка без приземления: вторая и дальше — золотом, от 1000 — крупно
    before = len(session.popups)
    session._on_event(Event("enemy", 400, {"x": 100.0, "y": 200.0, "chain": 3, "via": "stomp"}))
    session._on_event(Event("stomp", 1000, {"x": 140.0, "y": 200.0, "chain": 6, "via": "stomp"}))
    chain = session.popups[before:]
    report.check(
        "цепочка прыжков: 400 золотом, 1000 — крупной надписью",
        [p.text for p in chain] == ["+400", "+1000"]
        and all(p.color == POPUP_COLORS["chain"] for p in chain)
        and [p.big for p in chain] == [False, True],
        f"{[(p.text, p.kind, p.big) for p in chain]}",
    )

    # Доиграть уровень ботом и проверить итоги
    bot = HeuristicBot()
    bot.reset(world)
    while session.state == PLAYING and world.frame < MAX_FRAMES:
        session.update(1 / 60, bot)
    result = session.last_result
    summary = result.summary if result else {}
    breakdown = result.breakdown if result else {}
    from ui.hud import collected_rows

    rows = collected_rows(result) if result else []
    parts = (breakdown.get("block_points", 0) + breakdown.get("coin_points", 0)
             + breakdown.get("enemy_points", 0) + breakdown.get("flag_points", 0)
             + breakdown.get("time_bonus", 0))
    report.check(
        "итоги: собрано из возможного (монеты, удары по «?», враги, тайники)",
        result is not None and ("тайники", 1, 1) in rows
        and any(name == "удары по «?»" and done >= 1 and total == world.totals["block_hits"]
                for name, done, total in rows)
        and parts == result.score,
        f"{'победа' if result and result.won else (result.death if result else '—')}; "
        + ", ".join(f"{name} {done}/{total}" for name, done, total in rows)
        + f"; строки итогов {parts} = счёт {result.score if result else '—'}",
    )

    record = session.records.get(WARP_LEVEL.id, HUMAN) or {}
    details = record.get("details", {})
    report.check(
        "рекорд хранит «собрано из возможного»",
        details.get("secrets") == 1 and details.get("secrets_total") == 1
        and details.get("block_hits_total") == summary.get("block_hits_total"),
        f"details={details}",
    )

    # ИИ: Inputs контроллера (с down) идут в мир без изменений
    class DownController:
        name = "вниз"

        def __init__(self):
            self.returned: list = []

        def reset(self, world) -> None:
            pass

        def act(self, world) -> Inputs:
            inputs = Inputs(down=True)
            self.returned.append(inputs)
            return inputs

    controller = DownController()
    session = warp_session(controller=controller, records=tmp_records("warp-ai"))
    world = session.world
    stand_on_pipe(world)
    captured = spy_inputs(world)
    for _ in range(40):
        session.update(1 / 60)
        if world.in_pipe or world.in_room:
            break
    del world.step
    report.check(
        "ИИ: Inputs контроллера (down=True) идут в мир как есть — ИИ уходит в трубу",
        bool(captured) and all(a is b for a, b in zip(captured, controller.returned))
        and (world.in_pipe or world.in_room),
        f"шагов {len(captured)}, фаза {world.phase}",
    )


# ===============================================================
# 9. РЕДАКТОР: СОХРАНЕНИЕ НЕ ТЕРЯЕТ КЛЮЧИ УРОВНЯ
# ===============================================================


def check_editor_save(report: Report) -> None:
    import level_editor

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / "editor_level.json"
    original_file = level_editor.MAP_FILE
    level_editor.MAP_FILE = path

    def editor():
        view = level_editor.LevelEditorView.__new__(level_editor.LevelEditorView)
        view.objects = [level_editor.EditorObject("Blocks", "Bricks", 16.0, 16.0, "block_2.png")]
        view.status_text = types.SimpleNamespace(text="")
        return view

    warps = [{"id": "w", "pipe": [1872, 304], "exit": [5872, 272], "room": {"objects": []}}]
    source = {"name": "Проверка", "description": "d", "time_limit": 99, "objects": [{"x": 1}],
              "theme": "overworld", "warps": warps}
    try:
        # Загружен файл: ключи и их порядок — из него
        view = editor()
        view._loaded_template = {key: (None if key == "objects" else value) for key, value in source.items()}
        view.save_map()
        saved = json.loads(path.read_text(encoding="utf-8"))
        loaded_ok = (
            list(saved) == list(source)
            and all(saved[key] == source[key] for key in source if key != "objects")
            and len(saved["objects"]) == 1 and saved["objects"][0]["resource"] == "Bricks"
        )

        # Карту не загружали — лишние ключи берутся из файла на диске
        path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
        view = editor()
        view.save_map()
        saved_disk = json.loads(path.read_text(encoding="utf-8"))
        disk_ok = saved_disk.get("warps") == warps and saved_disk.get("time_limit") == 99

        # Файла нет вовсе — как раньше, только objects
        path.unlink()
        view = editor()
        view.save_map()
        fresh = json.loads(path.read_text(encoding="utf-8"))

        # Загрузили, очистили, нарисовали новое — warps старых труб не переносятся
        path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
        view = editor()
        view.object_sprites = []
        view._cells = {}
        view._invalidate_map_width = lambda: None
        view.get_camera_min_x = lambda: 0.0
        view.load_map()
        loaded_warps = (getattr(view, "_loaded_template", None) or {}).get("warps") == warps
        view.clear_map()
        view.objects = [level_editor.EditorObject("Blocks", "Bricks", 48.0, 16.0, "block_2.png")]
        view.save_map()
        cleared = json.loads(path.read_text(encoding="utf-8"))
    finally:
        level_editor.MAP_FILE = original_file

    report.check(
        "редактор: save_map сохраняет name/time_limit/theme/warps и порядок ключей",
        loaded_ok and disk_ok and list(fresh) == ["objects"],
        f"ключи после сохранения: {list(saved)}; без загрузки: warps {'есть' if disk_ok else 'потеряны'}; "
        f"новый файл: {list(fresh)}",
    )
    report.check(
        "редактор: «Загрузить» → «Очистить» → новая карта — без name/warps прежнего уровня",
        loaded_warps and list(cleared) == ["objects"] and len(cleared["objects"]) == 1,
        f"загружены warps: {loaded_warps}; ключи после сохранения: {list(cleared)}",
    )


# ===============================================================
# 10. ПАНЕЛЬ ИИ: ВСЕ ДЕЙСТВИЯ МОДЕЛИ, СКОЛЬКО БЫ ИХ НИ БЫЛО
# ===============================================================


def seven_action_labels():
    """ai.features.ACTION_LABELS с «вниз» (если владелец RL-части его ещё не добавил)."""

    from ai import features

    labels = tuple(features.ACTION_LABELS)
    return labels if "вниз" in labels else labels + ("вниз",)


def check_ai_actions(report: Report) -> None:
    try:
        from ai import features
    except Exception as error:  # noqa: BLE001
        report.check("панель ИИ: ai.features недоступен — пропущено", True, f"{type(error).__name__}: {error}")
        return

    from ui import hud

    # Не «подписи совпадают с ACTION_LABELS» (action_labels из них и собран),
    # а что у каждого действия модели есть свой значок: новое действие
    # вроде «вправо+бег» иначе показалось бы тремя буквами
    current = list(features.ACTION_LABELS)
    glyphs = [hud.action_glyph(label) for label in current]
    known = set(hud.ACTION_GLYPHS.values())
    report.check(
        "панель ИИ: у каждого действия модели (ai.features.ACTION_LABELS) свой значок-стрелка",
        bool(current) and all(glyph and set(glyph) <= known for glyph in glyphs)
        and len(set(glyphs)) == len(glyphs),
        f"сейчас в модели {len(current)}: "
        + ", ".join(f"{label} {glyph}" for label, glyph in zip(current, glyphs)),
    )

    original = features.ACTION_LABELS
    features.ACTION_LABELS = seven_action_labels()
    try:
        labels = hud.action_labels({"вниз": 0.4})
        glyphs = [hud.action_glyph(label) for label in labels]
        right = 1280 - hud.AI_PANEL_MARGIN
        left = right - hud.AI_PANEL_WIDTH
        columns = hud.action_strip_layout(len(labels), left + hud.AI_PANEL_PAD, right - hud.AI_PANEL_PAD)
    finally:
        features.ACTION_LABELS = original

    widths = [x1 - x0 for x0, x1 in columns]
    report.check(
        "панель ИИ: 7 действий (с «вниз») в прежней ширине панели",
        len(labels) == 7 and "↓" in glyphs and len(set(glyphs)) == 7
        and columns[0][0] >= left + hud.AI_PANEL_PAD - 1e-6
        and columns[-1][1] <= right - hud.AI_PANEL_PAD + 1e-6
        and min(widths) >= 40,
        f"значки {' '.join(glyphs)}; колонка {min(widths):.0f} px в панели {hud.AI_PANEL_WIDTH} px",
    )


# ===============================================================
# 7. ОТРИСОВКА В НЕВИДИМОМ ОКНЕ
# ===============================================================


def fit_result(who: str, won: bool, level_name: str, summary: dict, breakdown: dict,
               record_before: dict | None = None, record_after: dict | None = None):
    """LevelResult для проверки раскладки панелей (крайние случаи чисел)."""

    from ui.session import LevelResult

    summary = {"won": won, "death": None if won else "enemy", **summary}
    return LevelResult(index=0, level_id="1-1", level_name=level_name, who=who, summary=summary,
                       breakdown=breakdown, record_before=record_before or {},
                       record_after=record_after or {})


def fit_cases(level: LevelInfo) -> list[tuple[str, object]]:
    """Экраны смерти и итогов с самыми длинными строками, какие достижимы.

    Из замечания ревьюера: 1-1 без единой находки (603 px в панели 580),
    «монеты 15/19 · удары по «?» 30/41 · враги 10/12 · тайники 1/1» и
    сокровищница «монеты 120/184 · … 40/60 · враги 6/8» (636 px); на
    итогах — «× 10 из 29 · панцирь 12» рядом с «15 400» за врагов.
    """

    world = make_world(level)
    totals = world.totals
    real = {"coins_total": totals["coins"], "block_hits_total": totals["block_hits"],
            "enemies_total": totals["enemies"], "secrets_total": max(totals["secrets"], 1)}

    def breakdown(block, coin, enemy, stomps=0, flag=0, fraction=None, seconds=0):
        return {"block_points": block, "coin_points": coin, "enemy_points": enemy, "stomps": stomps,
                "flag_fraction": fraction, "flag_points": flag, "time_seconds": seconds,
                "time_bonus": seconds * 20}

    rec = {HUMAN: {"score": 123456, "won": True}, AI: {"score": 99999, "won": True}}
    zero = {"score": 0, "coins": 0, "blocks": 0, "enemies": 0, "secrets": 0, **real}
    mid = {"score": 9300, "coins": 15, "coins_total": 19, "blocks": 30, "block_hits_total": 41,
           "enemies": 10, "enemies_total": 12, "secrets": 1, "secrets_total": 1}
    treasury = {"score": 43400, "coins": 120, "coins_total": 184, "blocks": 40, "block_hits_total": 60,
                "enemies": 6, "enemies_total": 8, "secrets": 1, "secrets_total": 2}
    goombas = {"score": 188800, "coins": 110, "coins_total": 999, "blocks": 12, "block_hits_total": 99,
               "enemies": 10, "enemies_total": 29, "secrets": 2, "secrets_total": 2}
    horde = {**goombas, "enemies": 25}
    return [
        ("dead", fit_result(HUMAN, False, level.name, zero, breakdown(0, 0, 0))),
        ("dead", fit_result(AI, False, level.name, mid, breakdown(3000, 3000, 3300), rec, rec)),
        ("dead", fit_result(HUMAN, False, level.name, treasury, breakdown(4000, 24000, 15400), rec, rec)),
        ("res", fit_result(HUMAN, True, level.name, goombas,
                           breakdown(1200, 22000, 15400, 12, 5000, 1.0, 400), rec, rec)),
        ("res", fit_result(AI, True, level.name, horde,
                           breakdown(1200, 22000, 9800, 4, 5000, 1.0, 400), rec, rec)),
        ("res", fit_result(HUMAN, True, level.name, zero, breakdown(0, 0, 0, 0, 100, 0.02, 3))),
    ]


def label_span(label) -> tuple[float, float]:
    """Левый и правый край надписи с учётом привязки."""

    width = label.content_width
    if label.anchor_x == "center":
        return label.x - width / 2, label.x + width / 2
    if label.anchor_x == "right":
        return label.x - width, label.x
    return label.x, label.x + width


def panel_overflow(texts: dict, prefix: str, panel_left: float, panel_right: float) -> list[str]:
    """Надписи экрана (ключи prefix.*), которые вылезают из панели или наезжают друг на друга."""

    margin = 8
    problems = []
    drawn = {key: label for key, label in texts.items()
             if key.startswith(prefix + ".") and not key.endswith("#w") and not key.endswith("#shadow")}
    for key, label in drawn.items():
        a, b = label_span(label)
        if a < panel_left + margin or b > panel_right - margin:
            problems.append(f"{key} «{label.text}» {a:.0f}..{b:.0f} вне {panel_left:.0f}..{panel_right:.0f}")

    def gap(left_key: str, right_key: str) -> None:
        if left_key in drawn and right_key in drawn:
            a = label_span(drawn[left_key])[1]
            b = label_span(drawn[right_key])[0]
            if b - a < margin:
                problems.append(f"«{drawn[left_key].text}» наезжает на «{drawn[right_key].text}» ({b - a:.0f} px)")

    for i in range(10):
        gap(f"res.row{i}.name", f"res.row{i}.detail")
        gap(f"res.row{i}.detail", f"res.row{i}.points")
        for part in ("name", "count", "points"):
            gap(f"dead.col{i}.{part}", f"dead.col{i + 1}.{part}")
    return problems


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


def hud_bar_height() -> float:
    from ui import hud

    return float(hud.HUD_BAR_HEIGHT)


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

    fit_problems: list[str] = []
    fit_checked: list[str] = []

    def check_fit(name: str, view, prefix: str) -> None:
        """Надписи только что нарисованного экрана смерти/итогов — внутри панели."""
        from ui import hud

        panel_w = hud.RESULT_PANEL_WIDTH if prefix == "res" else hud.DEATH_PANEL_WIDTH
        fit_checked.append(name)
        fit_problems.extend(f"{name}: {problem}" for problem in panel_overflow(
            view.texts._texts, prefix, window.width / 2 - panel_w / 2, window.width / 2 + panel_w / 2))

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

        # «Вниз» держится (S или стрелка), прыжок — по-прежнему фронт нажатия
        downs = []
        for key in (arcade.key.S, arcade.key.DOWN):
            view.on_key_press(key, 0)
            downs.append(view._current_inputs().down)
            view.on_key_release(key, 0)
            downs.append(view._current_inputs().down)
        view.on_key_press(arcade.key.DOWN, 0)
        captured = spy_inputs(session.world)
        view.on_update(1 / 60)
        view.on_update(1 / 60)
        del session.world.step
        view.on_key_release(arcade.key.DOWN, 0)
        report.check(
            "клавиши: S и ↓ — «вниз» (Inputs.down) держится и доходит до мира",
            downs == [True, False, True, False] and len(captured) == 2 and all(i.down for i in captured),
            f"нажата/отпущена: {downs}; до мира: {[i.down for i in captured]}",
        )

        # Смерть в яме → экран смерти, R — заново
        session.world.player.center_y = -500
        view.on_update(1 / 60)
        session.state_time = 1.5
        shot("death", view)
        check_fit("death", view, "dead")
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
        check_fit("results", view, "res")
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

    # --- Трубы и тайник: слои, фон, камера, всплывашки, итоги ---
    try:
        from game import THEME_BACKGROUNDS

        session = warp_session(records=tmp_records("draw-warp"))
        view = GameView(return_to_menu=lambda: None, session=session)
        window.show_view(view)
        world = session.world
        width, height = window.width, window.height

        def drawn_layers() -> list[str]:
            """on_draw с записью, в каком порядке на самом деле рисуются стены и игрок.

            Подменяется draw() самих списков спрайтов (стены текущей части
            уровня, спрайт игрока), а не порядок из GameView.world_layers():
            проверяется то, что ушло в отрисовку.
            """
            order: list[str] = []
            spied = [(world.walls, "walls"), (world.player.playerSprite, "player")]
            for sprites, name in spied:
                original_draw = sprites.draw

                def record(*args, _name=name, _draw=original_draw, **kwargs):
                    order.append(_name)
                    return _draw(*args, **kwargs)

                sprites.draw = record
            try:
                view.on_draw()
            finally:
                for sprites, _ in spied:
                    del sprites.draw
            return order

        normal_order = drawn_layers()
        stand_on_pipe(world)
        for _ in range(20):
            view.on_update(1 / 60)
        shot("pipe_hint", view)

        view.on_key_press(arcade.key.DOWN, 0)
        pipe_orders = {}
        for _ in range(12):
            view.on_update(1 / 60)
        pipe_orders[world.phase] = drawn_layers()
        shot("pipe_down", view)

        frames = 0
        while not world.in_room and frames < 300:
            view.on_update(1 / 60)
            frames += 1
        view.on_key_release(arcade.key.DOWN, 0)
        for _ in range(10):
            view.on_update(1 / 60)
        room = world.area
        background = tuple(view.background_color)[:3]
        camera = view.camera.position
        room_cx = (room.left + room.right) / 2
        room_cy = (room.bottom + room.top + hud_bar_height()) / 2
        shot("room_secret", view)
        report.check(
            "комната: тёмный фон, камера по центру комнаты, баннер тайника",
            world.in_room and background == THEME_BACKGROUNDS["underground"]
            and abs(camera.x - room_cx) < 1 and abs(camera.y - room_cy) < 1
            and session.banner is not None,
            f"фон {background}, камера ({camera.x:.0f}, {camera.y:.0f}), центр комнаты ({room_cx:.0f}, {room_cy:.0f}), "
            f"комната {room.right - room.left:.0f}×{room.top - room.bottom:.0f} в окне {width}×{height}",
        )

        # Удар по «?»: монета над блоком видна, блок после отрисовки на месте
        advance = lambda inputs: session.update(1 / 60, inputs)  # noqa: E731
        seen = bump_room_block(session, advance)
        block = seen.get("block")
        block_y = block.center_y if block else None
        shot("block_coin", view)
        report.check(
            "удар по «?» нарисован: монета, «+100», подскок; блок после кадра на месте",
            bool(seen.get("coins")) and bool(seen.get("popups")) and seen.get("bumped")
            and block is not None and block.center_y == block_y == 176.0,
            f"монет {len(seen.get('coins', []))}, всплывашек {len(seen.get('popups', []))}, "
            f"блок y={block_y}",
        )

        view.on_key_press(arcade.key.RIGHT, 0)
        frames = 0
        while world.phase != PIPE_SIDE and frames < 400:
            view.on_update(1 / 60)
            frames += 1
        view.on_update(1 / 60)
        pipe_orders[world.phase] = drawn_layers()
        shot("pipe_side", view)
        while world.phase != PIPE_UP and frames < 800:
            view.on_update(1 / 60)
            frames += 1
        for _ in range(10):
            view.on_update(1 / 60)
        pipe_orders[world.phase] = drawn_layers()
        shot("pipe_up", view)
        while world.in_pipe and frames < 1200:
            view.on_update(1 / 60)
            frames += 1
        view.on_key_release(arcade.key.RIGHT, 0)
        view.on_update(1 / 60)
        back_background = tuple(view.background_color)[:3]
        back_camera = view.camera.position.x
        back_order = drawn_layers()

        def player_under_walls(order):
            return {"player", "walls"} <= set(order) and order.index("player") < order.index("walls")

        def player_over_walls(order):
            return {"player", "walls"} <= set(order) and order.index("player") > order.index("walls")

        report.check(
            "труба: игрок рисуется под стенами (вниз, вбок, вверх), в игре — поверх",
            set(pipe_orders) == {PIPE_DOWN, PIPE_SIDE, PIPE_UP}
            and all(player_under_walls(order) for order in pipe_orders.values())
            and player_over_walls(normal_order) and player_over_walls(back_order),
            f"вызовы draw() спрайтов в трубе: {pipe_orders}; в игре: {normal_order}, после выхода: {back_order}",
        )
        report.check(
            "из комнаты на уровень: снова небо, камера у трубы-выхода",
            not world.in_room and back_background == THEME_BACKGROUNDS["overworld"]
            and abs(back_camera - world.warps[0].exit_x) < width / 2,
            f"фон {back_background}, камера x={back_camera:.0f}, труба-выход x={world.warps[0].exit_x:.0f}",
        )

        # Прыжок на врага — «+N» над ним
        events = stomp_nearest_enemy(session, lambda inputs: view.on_update(1 / 60))
        shot("stomp_popup", view)

        # Итоги уровня: собрано из возможного
        bot = HeuristicBot()
        while session.state == PLAYING and world.frame < MAX_FRAMES:
            session.update(1 / 60, bot)
        view.on_update(1 / 60)
        session.state_time = 2.5
        shot("results_totals", view)
        check_fit("results_totals", view, "res" if session.state == LEVEL_DONE else "dead")
        texts = {key: label.text for key, label in view.texts._texts.items()}
        if session.state == LEVEL_DONE:
            rows = {texts.get(f"res.row{i}.name"): texts.get(f"res.row{i}.detail") for i in range(8)}
            totals_ok = (
                rows.get("Тайники") == "1 из 1"
                and rows.get("Удары по «?»", "").endswith(f"из {world.totals['block_hits']}")
                and f"из {world.totals['enemies']}" in rows.get("Враги", "")
            )
            shown = rows
        else:
            shown = {texts.get(f"dead.col{i}.name"): texts.get(f"dead.col{i}.count") for i in range(4)}
            totals_ok = (
                shown.get("ТАЙНИКИ") == "1 из 1"
                and str(shown.get("УДАРЫ ПО «?»", "")).endswith(f"из {world.totals['block_hits']}")
            )
        report.check(
            "экран итогов: «собрано из возможного», строка «Тайники», прыжок на врага виден",
            totals_ok and bool(events),
            f"{session.state}: {shown}; враг: {[(e.kind, e.points) for e in events]}",
        )
    except Exception:  # noqa: BLE001
        fail("трубы и тайник")

    # --- Надписи на экранах смерти и итогов влезают в панель ---
    try:
        cases = fit_cases(level)
        for i, (prefix, result) in enumerate(cases):
            session = PlaySession(levels=[level], records=tmp_records(f"draw-fit{i}"))
            view = GameView(return_to_menu=lambda: None, session=session)
            window.show_view(view)
            session.results[session.index] = result
            session.state = LEVEL_DONE if prefix == "res" else DEAD
            session.state_time = 3.0
            shot(f"fit{i}_{prefix}", view)
            check_fit(f"fit{i}_{prefix}", view, prefix)
        report.check(
            "экраны смерти и итогов: надписи в панели и не наезжают (настоящие + 1-1 с нулём, сокровищница, 29 врагов)",
            {"death", "results", "results_totals"} <= set(fit_checked)
            and len(fit_checked) == 3 + len(cases) and not fit_problems,
            "; ".join(fit_problems[:4]) if fit_problems else f"экраны: {', '.join(fit_checked)}",
        )
    except Exception:  # noqa: BLE001
        fail("надписи в панелях")

    # --- Панель ИИ на 7 действий (с «вниз») ---
    try:
        from ai import features
        from ui import hud

        seven = seven_action_labels()

        class SevenController(BotController):
            def overlay(self) -> dict:
                info = super().overlay()
                share = 0.5 / (len(seven) - 1)
                info["probs"] = {label: (0.5 if label == "вниз" else share) for label in seven}
                info["action"] = "вниз"
                return info

        original = features.ACTION_LABELS
        features.ACTION_LABELS = seven
        try:
            session = PlaySession(controller=SevenController(), levels=[level], records=tmp_records("draw-ai7"))
            view = GameView(return_to_menu=lambda: None, session=session)
            window.show_view(view)
            for _ in range(60):
                view.on_update(1 / 60)
            shot("ai_panel7", view)
        finally:
            features.ACTION_LABELS = original

        right = window.width - hud.AI_PANEL_MARGIN
        left = right - hud.AI_PANEL_WIDTH
        glyphs = [view.texts._texts.get(f"ai.strip{i}") for i in range(len(seven))]
        spans = [(g.x - g.content_width / 2, g.x + g.content_width / 2) for g in glyphs if g is not None]
        fits = (
            len(spans) == len(seven) == 7
            and all(left < a and b < right for a, b in spans)
            and all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1))
        )
        report.check(
            "панель ИИ на 7 действий нарисована в прежней ширине, значки не наезжают",
            fits and glyphs[-1] is not None and glyphs[-1].text == "↓",
            f"значки {[g.text for g in glyphs if g is not None]}, панель {left:.0f}..{right:.0f}, "
            f"крайние {spans[0][0]:.0f}..{spans[-1][1]:.0f}" if spans else "значков нет",
        )
    except Exception:  # noqa: BLE001
        fail("панель ИИ на 7 действий")

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
        {"play", "pause", "death", "ai_panel", "flag", "results", "all_done", "ai_missing", "menu", "levels",
         "pipe_hint", "pipe_down", "room_secret", "block_coin", "pipe_side", "pipe_up", "stomp_popup",
         "results_totals", "ai_panel7"}
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
        check_pipes_and_effects(report)
        check_editor_save(report)
        check_ai_actions(report)
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

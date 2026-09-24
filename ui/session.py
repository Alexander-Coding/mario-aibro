"""Сеанс игры без отрисовки: уровни по манифесту, фиксированный шаг, итоги.

PlaySession держит список уровней и мир текущего уровня (GameWorld — та
же логика, на которой учится нейросеть), крутит мир фиксированным шагом
rules.FIXED_DT через аккумулятор и решает, какой экран сейчас показывать.
Окна здесь нет, поэтому сеанс проверяется headless (tools/check_ui.py),
а GameView (game.py) только рисует его состояние и передаёт нажатия.

    session = PlaySession(ai_mode=False)
    session.update(delta_time, Inputs(right=True, jump=space_pressed))
    if session.state == LEVEL_DONE:
        session.next_level()
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

from core import levels as core_levels
from core import rules
from core.levels import LevelInfo
from core.world import GameWorld, Inputs, Event
from records import AI, HUMAN, Records
from ui import ai_control

# ---------------------------------------------------------------
# СОСТОЯНИЯ ЭКРАНА
# ---------------------------------------------------------------

PLAYING = "playing"        # идёт уровень (вместе с заставкой флага и замка)
LEVEL_DONE = "level_done"  # уровень пройден — экран итогов
DEAD = "dead"              # смерть — экран с причиной
ALL_DONE = "all_done"      # уровни кончились — итоговая таблица

# ---------------------------------------------------------------
# ЧИСЛА СЕАНСА
# ---------------------------------------------------------------

# Больше шагов мира за один кадр не делаем: на подвисшей машине игра
# замедлится, а не будет догонять время рывками («спираль смерти»).
MAX_STEPS_PER_FRAME = 4

# Через сколько секунд после финиша или смерти ИИ сам идёт дальше
AI_ADVANCE_DELAY = 3.0

# Время жизни всплывающих очков, секунд
POPUP_LIFETIME = 1.0
POPUP_LIFETIME_BIG = 1.6

# Сколько секунд висит крупная надпись о захвате флага
FLAG_BANNER_TIME = 3.0

# Погрешность сравнения аккумулятора с шагом: 1/60 + 1/60 в числах
# с плавающей точкой не всегда ровно 2/60, и без запаса кадр с dt = 1/60
# иногда делал бы 0 шагов, а следующий — 2.
_ACCUMULATOR_EPS = 1e-9

# Причины смерти по-русски
DEATH_CAUSES = {
    "pit": "Упал в пропасть",
    "enemy": "Погиб от врага",
    "time": "Время вышло",
}

# Цвета всплывающих очков по видам событий
POPUP_COLORS = {
    "coin": (255, 214, 64),
    "block": (255, 168, 72),
    "enemy": (250, 250, 250),
    "flag": (120, 240, 120),
    "time_bonus": (110, 220, 255),
    "hit": (255, 90, 90),
}


# ===============================================================
# ДАННЫЕ ДЛЯ ОТРИСОВКИ
# ===============================================================


@dataclass
class Popup:
    """Всплывающая надпись «+100» в мировых координатах."""

    text: str
    x: float
    y: float
    color: tuple
    kind: str = ""
    big: bool = False
    lifetime: float = POPUP_LIFETIME
    age: float = 0.0
    serial: int = 0

    @property
    def alive(self) -> bool:
        return self.age < self.lifetime

    @property
    def progress(self) -> float:
        """0 — только появилась, 1 — исчезла."""
        return min(self.age / max(self.lifetime, 1e-6), 1.0)


@dataclass
class FlagBanner:
    """Крупная надпись при захвате флага: высота в % и очки."""

    fraction: float
    points: int
    age: float = 0.0

    @property
    def percent(self) -> int:
        return int(round(self.fraction * 100))

    @property
    def alive(self) -> bool:
        return self.age < FLAG_BANNER_TIME


@dataclass
class LevelResult:
    """Итог одного уровня для экрана итогов и итоговой таблицы."""

    index: int
    level_id: str
    level_name: str
    who: str                      # "human" | "ai"
    summary: dict                 # world.summary()
    breakdown: dict               # разбивка очков, см. level_breakdown()
    record_before: dict           # {"human": ..., "ai": ...} до этого уровня
    record_after: dict            # {"human": ..., "ai": ...} после записи
    new_record: bool = False

    @property
    def score(self) -> int:
        return int(self.summary["score"])

    @property
    def won(self) -> bool:
        return bool(self.summary["won"])

    @property
    def death(self) -> Optional[str]:
        return self.summary.get("death")

    @property
    def death_text(self) -> str:
        return DEATH_CAUSES.get(self.death or "", "")


def level_breakdown(world: GameWorld) -> dict:
    """Из чего сложились очки уровня: блоки, монеты, враги, флаг, время."""

    stats = world.stats

    return {
        "blocks": int(stats["block"]),
        "block_points": int(stats["block_points"]),
        "coins": int(stats["coin"]),
        "coin_points": int(stats["coin_points"]),
        "enemies": int(stats["enemy"]),
        "enemy_points": int(stats["enemy_points"]),
        "flag_fraction": world.flag_fraction,
        "flag_points": int(world.flag_bonus),
        "time_seconds": int(world.time_left) if world.won else 0,
        "time_bonus": int(world.time_bonus),
        "total": int(world.score),
    }


# ===============================================================
# СЕАНС
# ===============================================================


WorldFactory = Callable[[LevelInfo], GameWorld]
Control = Union[Inputs, Any, None]


class PlaySession:
    """Прохождение уровней подряд — человеком или ИИ.

    ai_mode       — играет нейросеть; controller можно передать готовый
                    (тесты), иначе он загружается лениво через load_controller;
    defer_ai      — не грузить модель в конструкторе: окно сначала нарисует
                    «Загрузка ИИ…», потом позовёт load_ai() (torch грузится секунды);
    levels        — список уровней (по умолчанию core.levels.load_levels());
    start_index   — с какого уровня начать;
    records       — рекорды (по умолчанию records.json в корне).
    """

    def __init__(
        self,
        *,
        ai_mode: bool = False,
        controller: Any = None,
        levels: Optional[Sequence[LevelInfo]] = None,
        start_index: int = 0,
        records: Optional[Records] = None,
        load_controller: Optional[Callable[[], tuple]] = None,
        world_factory: Optional[WorldFactory] = None,
        auto_advance_delay: float = AI_ADVANCE_DELAY,
        defer_ai: bool = False,
    ):
        # Через модуль, а не прямым импортом функции: тесты подменяют
        # core.levels.load_levels своим манифестом
        self.levels: list[LevelInfo] = list(
            levels if levels is not None else core_levels.load_levels()
        )
        if not self.levels:
            self.levels = [LevelInfo(id="1-1", file="custom_level.json", name="1-1")]

        self.records = records if records is not None else Records()
        self.world_factory: WorldFactory = world_factory or core_levels.make_world
        self.auto_advance_delay = float(auto_advance_delay)

        # ИИ
        self.ai_requested = bool(ai_mode or controller is not None)
        self.controller = controller
        self.ai_error: Optional[str] = None
        self._load_controller = load_controller or ai_control.load_ai_controller
        self.ai_pending = self.ai_requested and controller is None

        # Результаты текущего забега по индексам уровней (последняя попытка)
        self.results: dict[int, LevelResult] = {}
        self.start_index = max(0, min(int(start_index), len(self.levels) - 1))

        # Счётчик пересборок мира — камера по нему понимает, что уровень сменился
        self.world_serial = 0
        self._popup_serial = 0

        self.world: Optional[GameWorld] = None
        self._start_level(self.start_index)

        if self.ai_pending and not defer_ai:
            self.load_ai()

    # ========================================================
    # ИИ
    # ========================================================

    def load_ai(self) -> None:
        """Грузит контроллер. Не вышло — остаётся клавиатура и подсказка на экране."""

        if not self.ai_pending:
            return

        self.ai_pending = False

        try:
            controller, error = self._load_controller()
        except Exception as exc:  # noqa: BLE001 — загрузчик тоже не должен ронять игру
            traceback.print_exc()
            controller, error = None, f"{ai_control.AI_MISSING_TEXT}\n({type(exc).__name__}: {exc})"

        self.controller = controller
        self.ai_error = error if controller is None else None

        if controller is not None:
            self._reset_controller()

    @property
    def ai_active(self) -> bool:
        """Сейчас играет нейросеть (контроллер загружен и не сломался)."""
        return self.controller is not None

    @property
    def who(self) -> str:
        return AI if self.ai_active else HUMAN

    def ai_overlay(self) -> Optional[dict]:
        """Что показать на панели ИИ (или None)."""
        return ai_control.safe_overlay(self.controller)

    def ai_model_name(self) -> str:
        return ai_control.controller_name(self.controller)

    def _reset_controller(self) -> None:
        if self.controller is None or not hasattr(self.controller, "reset"):
            return
        try:
            self.controller.reset(self.world)
        except Exception as error:  # noqa: BLE001
            self._disable_ai(error)

    def _disable_ai(self, error: Exception) -> None:
        """Контроллер упал посреди игры — отдаём управление клавиатуре."""

        traceback.print_exc()
        self.controller = None
        self.ai_error = f"Ошибка ИИ: {type(error).__name__}: {error}\nУправление передано клавиатуре"

    # ========================================================
    # УРОВНИ
    # ========================================================

    @property
    def level(self) -> LevelInfo:
        return self.levels[self.index]

    @property
    def level_number(self) -> int:
        return self.index + 1

    @property
    def level_count(self) -> int:
        return len(self.levels)

    @property
    def is_last_level(self) -> bool:
        return self.index >= len(self.levels) - 1

    @property
    def last_result(self) -> Optional[LevelResult]:
        return self.results.get(self.index)

    def _start_level(self, index: int) -> None:
        self.index = index
        self.world = self.world_factory(self.level)
        self.world_serial += 1

        self.state = PLAYING
        self.state_time = 0.0

        self._accumulator = 0.0
        self._jump_pending = False

        self.popups: list[Popup] = []
        self.flag_banner: Optional[FlagBanner] = None

        if not self.ai_pending:
            self._reset_controller()

    def next_level(self) -> None:
        """Следующий уровень или итоговая таблица после последнего."""

        if self.state == ALL_DONE:
            return

        if self.is_last_level:
            self.state = ALL_DONE
            self.state_time = 0.0
            self.popups = []
            self.flag_banner = None
        else:
            self._start_level(self.index + 1)

    def restart_level(self) -> None:
        """Текущий уровень заново. Результат забега перезапишется новой попыткой."""
        self._start_level(self.index)

    def restart_run(self) -> None:
        """Весь забег заново — с того уровня, с которого начинали."""
        self.results = {}
        self._start_level(self.start_index)

    # ========================================================
    # ОБНОВЛЕНИЕ
    # ========================================================

    def update(self, delta_time: float, control: Control = None) -> int:
        """Кадр окна: delta_time реального времени → 0..4 шагов мира.

        control — Inputs человека (right/left удерживаются, jump — фронт
        нажатия с прошлого кадра) или объект с act(world) → Inputs.
        Загруженный ИИ-контроллер сеанса важнее: в режиме ИИ клавиши
        движения не действуют. Возвращает число сделанных шагов мира.
        """

        delta_time = max(0.0, float(delta_time))

        self._age_effects(delta_time)
        self.state_time += delta_time

        if self.state == PLAYING:
            if self.ai_pending:
                return 0  # ждём load_ai(), мир стоит
            return self._advance_world(delta_time, control)

        if self.state in (LEVEL_DONE, DEAD) and self.ai_active:
            if self.state_time >= self.auto_advance_delay:
                self.next_level()

        return 0

    @property
    def advance_in(self) -> Optional[float]:
        """Сколько секунд до автоперехода ИИ (None — автоперехода нет)."""

        if self.state in (LEVEL_DONE, DEAD) and self.ai_active:
            return max(0.0, self.auto_advance_delay - self.state_time)
        return None

    def _advance_world(self, delta_time: float, control: Control) -> int:
        human = control if isinstance(control, Inputs) else None

        # Фронт прыжка копится, пока не случится шаг мира: кадр окна
        # короче 1/60 с шага не делает, и нажатие не должно пропасть
        if human is not None and human.jump and not self.ai_active:
            self._jump_pending = True

        self._accumulator += delta_time
        steps = 0

        while (
            self._accumulator >= rules.FIXED_DT - _ACCUMULATOR_EPS
            and steps < MAX_STEPS_PER_FRAME
        ):
            self._accumulator -= rules.FIXED_DT
            steps += 1

            inputs = self._inputs_for_step(control)
            self._step_world(inputs)

            if self.world.done:
                self._finish_level()
                self._accumulator = 0.0
                break

        # Не успели — лишнее время выбрасываем, а не догоняем следующим кадром
        if self._accumulator > rules.FIXED_DT:
            self._accumulator = rules.FIXED_DT

        return steps

    def _inputs_for_step(self, control: Control) -> Inputs:
        world = self.world

        # ИИ опрашивается на КАЖДОМ шаге мира: решение раз в 4 кадра он
        # принимает сам, а между решениями держит нажатие
        if self.controller is not None:
            try:
                return self.controller.act(world)
            except Exception as error:  # noqa: BLE001
                self._disable_ai(error)

        if control is not None and not isinstance(control, Inputs) and hasattr(control, "act"):
            return control.act(world)

        held = control if isinstance(control, Inputs) else Inputs()
        jump = self._jump_pending
        self._jump_pending = False  # фронт тратится на первом шаге

        return Inputs(right=bool(held.right), left=bool(held.left), jump=jump)

    def _step_world(self, inputs: Inputs) -> None:
        world = self.world

        # Враг приносит очки через полсекунды после прыжка на него, когда
        # его убирают из списка, — запоминаем, кто был, чтобы «+100»
        # всплыло над ним, а не над ушедшим вперёд игроком
        enemies_before = list(world.enemies)

        events = world.step(inputs)

        for event in events:
            self._on_event(event, enemies_before)

    # ========================================================
    # СОБЫТИЯ → ВСПЛЫВАЮЩИЕ ОЧКИ
    # ========================================================

    def _on_event(self, event: Event, enemies_before: list) -> None:
        world = self.world
        player = world.player
        kind = event.kind

        if kind == "coin":
            self._popup(f"+{event.points}", player.center_x, player.top + 12, kind)

        elif kind == "block":
            # Блок «?» над головой: удар засчитан, когда игрок летел вверх сквозь него
            self._popup(f"+{event.points}", player.center_x, player.top + 40, kind)

        elif kind == "enemy":
            x, y = player.center_x, player.top + 12
            for enemy in enemies_before:
                if not enemy.sprite_lists and not enemy.is_alive:
                    x, y = enemy.center_x, enemy.top + 8
                    break
            self._popup(f"+{event.points}", x, y, kind)

        elif kind == "flag":
            fraction = float(event.info.get("fraction", world.flag_fraction or 0.0))
            self.flag_banner = FlagBanner(fraction=fraction, points=event.points)
            self._popup(f"+{event.points}", player.center_x + 24, player.top + 24, kind, big=True)

        elif kind == "time_bonus" and event.points > 0:
            self._popup(
                f"+{event.points}",
                world.geometry.door_x,
                player.center_y + 72,
                kind,
                big=True,
            )

        elif kind == "hit":
            self._popup("−1 жизнь", player.center_x, player.top + 12, kind)

    def _popup(self, text: str, x: float, y: float, kind: str, big: bool = False) -> None:
        self._popup_serial += 1
        self.popups.append(
            Popup(
                text=text,
                x=float(x),
                y=float(y),
                color=POPUP_COLORS.get(kind, (255, 255, 255)),
                kind=kind,
                big=big,
                lifetime=POPUP_LIFETIME_BIG if big else POPUP_LIFETIME,
                serial=self._popup_serial,
            )
        )

    def _age_effects(self, delta_time: float) -> None:
        if self.popups:
            for popup in self.popups:
                popup.age += delta_time
            self.popups = [popup for popup in self.popups if popup.alive]

        if self.flag_banner is not None:
            self.flag_banner.age += delta_time
            if not self.flag_banner.alive:
                self.flag_banner = None

    # ========================================================
    # КОНЕЦ УРОВНЯ
    # ========================================================

    def _finish_level(self) -> None:
        world = self.world
        level = self.level
        who = self.who
        summary = world.summary()

        record_before = self.records.level(level.id)
        try:
            updated = self.records.update(level.id, who, summary["score"], summary["won"])
        except Exception as error:  # noqa: BLE001 — рекорды не должны ронять игру
            print(f"[records] не удалось обновить рекорд: {error}")
            updated = False

        result = LevelResult(
            index=self.index,
            level_id=level.id,
            level_name=level.name,
            who=who,
            summary=summary,
            breakdown=level_breakdown(world),
            record_before=record_before,
            record_after=self.records.level(level.id),
            # Первый в жизни ноль — не «рекорд»
            new_record=bool(updated and summary["score"] > 0),
        )

        self.results[self.index] = result
        self.state = LEVEL_DONE if world.won else DEAD
        self.state_time = 0.0

    # ========================================================
    # ИТОГИ ЗАБЕГА
    # ========================================================

    def run_results(self) -> list[LevelResult]:
        """Результаты забега по порядку уровней."""
        return [self.results[index] for index in sorted(self.results)]

    @property
    def total_score(self) -> int:
        return sum(result.score for result in self.results.values())

    def record_for(self, level_id: Optional[str] = None) -> dict:
        """Рекорды уровня (по умолчанию текущего): {"human": ..., "ai": ...}."""
        return self.records.level(level_id or self.level.id)


__all__ = [
    "PlaySession",
    "Popup",
    "FlagBanner",
    "LevelResult",
    "level_breakdown",
    "PLAYING",
    "LEVEL_DONE",
    "DEAD",
    "ALL_DONE",
    "MAX_STEPS_PER_FRAME",
    "AI_ADVANCE_DELAY",
    "DEATH_CAUSES",
]

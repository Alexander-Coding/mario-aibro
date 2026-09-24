"""Проверка уровня: структура, проходимость, «потолок» очков, прогон бота.

    .venv/bin/python -m tools.level_check maps/level_warmup.json
    .venv/bin/python -m tools.level_check --all            # все уровни манифеста
    .venv/bin/python -m tools.level_check --all --no-bot   # без прогона бота
    .venv/bin/python -m tools.level_check maps/x.json --json

Что проверяется:

(а) Структура. Есть старт; флагшток собран как в 1-1 (основание, шток,
    шар, флаг); есть замок справа от штока; от штока до двери — ровная
    земля без ям и препятствий; все объекты на сетке 32*i + 16; в одной
    клетке не больше одного объекта; враги стоят на твёрдом блоке.

(б) Проходимость. Граф «где игрок может стоять» по тайлам. Рёбра — шаг
    на соседний тайл и полёты: прыжок или сход с края при разной скорости
    по горизонтали. Каждый полёт просчитывается кадр за кадром той же
    физикой, что у игрока в GameWorld (гравитация 0.5, прыжок 15, бег
    4 px/кадр, хитбокс 26×32 плюс по 1 px запаса с боков), с упором в
    стены и потолки. Два режима:
      - «физика» — все полёты, что удаются в симуляции (подъём до 6.8
        тайла, яма до ~6 тайлов на одной высоте): по нему считается потолок;
      - «осторожно» — только полёты с подъёмом ≤ 4 тайлов и сдвигом ≤ 5
        колонок (яма ≤ 4 тайлов): обязательный путь от старта до флагштока
        должен существовать в этом режиме, иначе уровень не проходит проверку.

(в) Прогон tools.heuristic_bot.HeuristicBot — для сведения, не требование.

(г) Потолок очков: монеты и блоки «?», до которых можно добраться и после
    этого дойти до флага; все враги; лучшая доля флагштока, которую даёт
    физика (5000 — если можно схватиться за верх). Отдельно — бонус за
    время при самом быстром пробеге (оценка сверху).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

from tools.level_tiles import (
    ENEMY_KINDS,
    HALF,
    TILE,
    LevelSource,
    level_cells,
    load_level,
    on_grid,
)

# ===============================================================
# ЧИСЛА ФИЗИКИ И ПРАВИЛ
# ===============================================================
# Повторяют core/rules.py и Entities/player.py. Модуль не импортирует ядро,
# чтобы проверка структуры и достижимости шла без arcade; совпадение чисел
# проверяет verify_physics() — прогоном настоящего GameWorld.

GRAVITY = 0.5
JUMP_SPEED = 15.0
RUN_SPEED = 4.0
PLAYER_HALF_W = 13.0  # хитбокс 26×32
PLAYER_HALF_H = 16.0
SOLID_MARGIN = 1.0  # запас с боков при проверке стен
GROUND_PROBE = 5.0  # can_jump(): опора ближе 5 px снизу
DEATH_Y = -100.0
POLE_REACH = TILE / 2 + 1  # core/world.py: _check_flagpole

FLAG_TIERS = ((0.95, 5000), (0.70, 2000), (0.45, 800), (0.20, 400), (0.00, 100))
COIN_POINTS = 100
BLOCK_POINTS = 100
ENEMY_POINTS = 100
TIME_BONUS_PER_SECOND = 20

# Осторожный режим: обязательные прыжки не выше и не дальше этого
SAFE_RISE_TILES = 4
SAFE_SHIFT_TILES = 5  # яма шириной 4 тайла: с края на край — 5 колонок

# Скорости по горизонтали в полёте (px/кадр в среднем). Клавиша даёт
# ровно 4 px/кадр; меньшие скорости — нажатия через кадр (как у игрока,
# который «придерживает» разбег).
JUMP_SPEEDS = (0, 1, 2, 3, 4)
WALK_OFF_SPEEDS = (1, 2, 4)
MAX_FLIGHT_FRAMES = 400
# Старт прыжка с края опоры: центр игрока на столько пикселей за краем.
# В кадре прыжка он ещё сдвинется на 4 px и окажется в 12 px за краем —
# хитбокс (полуширина 13, с запасом 14) всё ещё цепляет блок, прыжок есть.
EDGE_OVERHANG = 8


def flag_points(fraction: float) -> int:
    for threshold, points in FLAG_TIERS:
        if fraction >= threshold:
            return points
    return FLAG_TIERS[-1][1]


def score_block_hits(x: float, y: float) -> int:
    """Число ударов в блоке «?» — та же формула, что в Objects/ScoreBlock.py."""
    return random.Random(f"{float(x):.1f}:{float(y):.1f}").randint(1, 4)


# ===============================================================
# РЕЗУЛЬТАТ ПРОВЕРКИ
# ===============================================================


@dataclass
class Issue:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str

    def __str__(self) -> str:
        mark = {"error": "ОШИБКА", "warning": "внимание", "info": "инфо"}[self.level]
        return f"[{mark}] {self.code}: {self.message}"


@dataclass
class Ceiling:
    coins: int = 0
    coins_total: int = 0
    block_hits: int = 0
    block_hits_total: int = 0
    enemies: int = 0
    flag: int = 0
    flag_fraction: float = 0.0
    time_bonus_max: int = 0

    @property
    def total(self) -> int:
        """Потолок без бонуса за время."""
        return (self.coins + self.block_hits + self.enemies) * 100 + self.flag

    @property
    def total_with_time(self) -> int:
        return self.total + self.time_bonus_max


@dataclass
class LevelReport:
    name: str
    source: str
    width_tiles: int = 0
    time_limit: Optional[float] = None
    issues: list[Issue] = field(default_factory=list)
    safe_path: bool = False
    physics_path: bool = False
    safe_jumps: int = 0  # прыжков на осторожном пути
    ceiling: Ceiling = field(default_factory=Ceiling)
    min_run_seconds: float = 0.0
    bot: Optional[dict] = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def add(self, level: str, code: str, message: str) -> None:
        self.issues.append(Issue(level, code, message))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "ok": self.ok,
            "width_tiles": self.width_tiles,
            "time_limit": self.time_limit,
            "safe_path": self.safe_path,
            "physics_path": self.physics_path,
            "safe_jumps": self.safe_jumps,
            "ceiling": {
                "total": self.ceiling.total,
                "total_with_time": self.ceiling.total_with_time,
                "coins": self.ceiling.coins,
                "coins_total": self.ceiling.coins_total,
                "block_hits": self.ceiling.block_hits,
                "block_hits_total": self.ceiling.block_hits_total,
                "enemies": self.ceiling.enemies,
                "flag": self.ceiling.flag,
                "flag_fraction": round(self.ceiling.flag_fraction, 3),
                "time_bonus_max": self.ceiling.time_bonus_max,
            },
            "min_run_seconds": round(self.min_run_seconds, 1),
            "bot": self.bot,
            "issues": [issue.__dict__ for issue in self.issues],
        }


# ===============================================================
# (а) СТРУКТУРА
# ===============================================================


@dataclass
class Layout:
    """Разобранный уровень: что где лежит (в тайлах)."""

    solid: set[tuple[int, int]]
    coins: dict[tuple[int, int], int]  # клетка → очки
    blocks: dict[tuple[int, int], int]  # клетка → число ударов
    enemies: list[tuple[str, int, int]]
    spawn: Optional[tuple[float, float]]
    pole_x: Optional[float] = None
    pole_bottom: Optional[float] = None
    pole_top: Optional[float] = None
    door_x: Optional[float] = None
    base_ty: Optional[int] = None

    @property
    def pole_tx(self) -> Optional[int]:
        return None if self.pole_x is None else int(self.pole_x // TILE)


def check_structure(data: dict, report: LevelReport) -> Layout:
    cells = level_cells(data)

    solid = {(c.tx, c.ty) for c in cells if c.kind in (
        "ground", "brick", "hard", "stone", "block",
        "tube_top_l", "tube_top_r", "tube_l", "tube_r")}
    coins = {(c.tx, c.ty): COIN_POINTS for c in cells if c.kind == "coin"}
    blocks = {(c.tx, c.ty): score_block_hits(c.x, c.y) for c in cells if c.kind == "question"}
    enemies = [(c.kind, c.tx, c.ty) for c in cells if c.kind in ENEMY_KINDS]

    if not cells:
        report.add("error", "empty", "в уровне нет объектов")
        return Layout(solid, coins, blocks, enemies, None)

    report.width_tiles = max(c.tx for c in cells) - min(0, min(c.tx for c in cells)) + 1

    # --- Сетка 32*i + 16
    off_grid = [c for c in cells if not (on_grid(c.x) and on_grid(c.y))]
    if off_grid:
        sample = ", ".join(f"{c.kind}({c.x:g},{c.y:g})" for c in off_grid[:5])
        report.add("error", "grid", f"{len(off_grid)} объектов не на сетке 32*i+16: {sample}")

    # --- Слева от x = 0 мир считает сплошной стеной
    left = [c for c in cells if c.tx < 0 or c.ty < 0]
    if left:
        report.add("error", "bounds", f"{len(left)} объектов левее x=0 или ниже y=0")

    # --- Не больше одного объекта в клетке
    per_cell = Counter((c.tx, c.ty) for c in cells)
    doubles = [key for key, count in per_cell.items() if count > 1]
    if doubles:
        sample = ", ".join(
            f"({tx},{ty}): " + "+".join(c.kind for c in cells if (c.tx, c.ty) == (tx, ty))
            for tx, ty in sorted(doubles)[:5]
        )
        report.add("error", "overlap", f"{len(doubles)} клеток с двумя объектами: {sample}")

    # --- Старт
    starts = [c for c in cells if c.kind == "mario"]
    spawn = None
    if len(starts) != 1:
        report.add("error", "start", f"точек старта {len(starts)}, нужна ровно одна")
    if starts:
        s = starts[0]
        spawn = (s.x, s.y)
        if (s.tx, s.ty) in solid:
            report.add("error", "start", f"старт внутри блока ({s.tx},{s.ty})")
        elif (s.tx, s.ty - 1) not in solid:
            below = [ty for (tx, ty) in solid if tx == s.tx and ty < s.ty]
            if not below:
                report.add("error", "start", f"под стартом ({s.tx},{s.ty}) пропасть")
            else:
                report.add("warning", "start", f"старт ({s.tx},{s.ty}) висит в воздухе")

    # --- Враги стоят на твёрдом
    for kind, tx, ty in enemies:
        if (tx, ty) in solid:
            report.add("error", "enemy", f"{kind} ({tx},{ty}) внутри блока")
        elif (tx, ty - 1) not in solid:
            report.add("error", "enemy", f"{kind} ({tx},{ty}) не стоит на блоке")

    # --- Монеты и «?» не внутри твёрдого (двойная клетка поймана выше)
    for key in list(coins) + list(blocks):
        if key in solid:
            report.add("error", "overlap", f"объект в твёрдой клетке {key}")

    layout = Layout(solid, coins, blocks, enemies, spawn)

    _check_finish(cells, solid, layout, report)

    top = max(c.ty for c in cells)
    if top > 21:
        report.add("warning", "height", f"объекты до ряда {top}: выше экрана (720 px = 22 ряда)")

    return layout


def _check_finish(cells, solid, layout: Layout, report: LevelReport) -> None:
    """Флагшток и замок как в 1-1."""

    sticks = [c for c in cells if c.kind in ("stick", "ball")]
    flags = [c for c in cells if c.kind == "flag"]
    castle = [c for c in cells if c.kind == "castle" or c.kind.startswith("castle:")]

    if not sticks:
        report.add("error", "pole", "нет флагштока (декорации Flag stick)")
        return

    column = Counter(c.tx for c in sticks).most_common(1)[0][0]
    stray = [c for c in sticks if c.tx != column]
    if stray:
        report.add("error", "pole", f"части штока вне столбца {column}: {len(stray)}")

    shaft = sorted(c.ty for c in sticks if c.tx == column and c.kind == "stick")
    balls = [c.ty for c in sticks if c.tx == column and c.kind == "ball"]
    if len(shaft) < 3:
        report.add("error", "pole", f"шток из {len(shaft)} тайлов, нужно не меньше 3")
        return
    if shaft != list(range(shaft[0], shaft[-1] + 1)):
        report.add("error", "pole", "в штоке разрыв")
    if balls != [shaft[-1] + 1]:
        report.add("error", "pole", "шар (Flag stick/img_8.png) не на верхушке штока")

    base_ty = shaft[0] - 1
    if (column, base_ty) not in solid:
        report.add("error", "pole", f"нет основания штока — блока в ({column},{base_ty})")
    elif (column, base_ty - 1) not in solid:
        report.add("error", "pole", f"основание штока ({column},{base_ty}) висит в воздухе")

    if len(flags) != 1:
        report.add("error", "flag", f"флагов {len(flags)}, нужен один")
    elif flags[0].tx != column - 1 or not (shaft[0] <= flags[0].ty <= shaft[-1]):
        report.add("error", "flag", f"флаг ({flags[0].tx},{flags[0].ty}) не слева у штока")

    # Геометрия как в core/world.build_geometry
    stick_ys = [TILE * ty + HALF for ty in shaft + balls]
    layout.pole_x = column * TILE + HALF
    layout.pole_bottom = min(stick_ys) - HALF
    layout.pole_top = max(stick_ys) + HALF
    layout.base_ty = base_ty

    if not castle:
        report.add("error", "castle", "нет замка (декорации Castle)")
        return

    castle_left = min(c.tx for c in castle) * TILE
    castle_right = (max(c.tx for c in castle) + 1) * TILE
    door_x = (castle_left + castle_right) / 2
    layout.door_x = door_x

    if castle_left <= layout.pole_x:
        report.add("error", "castle", "замок не правее штока")
        return

    # Ровная земля от штока до двери: опора в ряду основания−1, проход свободен
    ground_ty = base_ty - 1
    door_tx = int(door_x // TILE)
    for tx in range(column + 1, door_tx + 1):
        if (tx, ground_ty) not in solid:
            report.add("error", "finish", f"яма между штоком и дверью в столбце {tx}")
            break
        blocked = [ty for ty in range(base_ty, base_ty + 3) if (tx, ty) in solid]
        if blocked:
            report.add("error", "finish", f"препятствие между штоком и дверью в ({tx},{blocked[0]})")
            break


# ===============================================================
# (б) ФИЗИКА ПОЛЁТА ПО ТАЙЛАМ
# ===============================================================


@dataclass
class Flight:
    """Итог одного прыжка или схода с края."""

    end: str  # "land" | "goal" | "death" | "walk" | "timeout"
    node: Optional[tuple[int, int]] = None
    fraction: float = 0.0
    coins: frozenset = frozenset()
    blocks: frozenset = frozenset()
    frames: int = 0


class TilePhysics:
    """Кадр за кадром, как arcade.PhysicsEnginePlatformer со стенами-тайлами.

    Порядок в кадре как в GameWorld: скорость по X из клавиш → гравитация →
    шаг по Y с упором в пол/потолок → шаг по X с упором в стену → can_jump
    (опора в 5 px снизу) → прыжок, если он запрошен и есть опора. Разгона
    нет: клавиша сразу даёт 4 px/кадр. «Заезд на пандус» arcade (подъём до
    4 px при упоре в стену) не повторяется — проверка от этого строже игры.
    Левее x = 0 стены нет: ушедший влево за край земли падает (стеной край
    мира считает только физика врагов).

    Твёрдые клетки хранятся битовыми масками по столбцам: столбец → число,
    где бит ty = 1, если клетка твёрдая. Игрок перекрывает не больше двух
    столбцов и двух рядов, поэтому проверка кадра — пара сдвигов и масок.
    """

    def __init__(self, layout: Layout):
        self.solid = layout.solid
        self.masks = _column_masks(layout.solid)
        self.coin_masks = _column_masks(layout.coins)
        self.block_masks = _column_masks(layout.blocks)
        self.pole_x = layout.pole_x
        self.pole_bottom = layout.pole_bottom
        self.pole_height = (
            max(layout.pole_top - layout.pole_bottom, 1.0) if layout.pole_x is not None else 1.0
        )
        # Ниже верха самого нижнего блока уровня падение уже не остановить
        self.lowest_top = min(((ty + 1) * TILE for _, ty in layout.solid), default=0)

    # -----------------------------------------------------------

    def _rows(self, masks: dict, tx0: int, tx1: int, ty0: int, ty1: int):
        """Нижний и верхний занятые ряды в прямоугольнике клеток (или None)."""
        if ty1 < 0:
            return None
        lo = ty0 if ty0 > 0 else 0
        window = (1 << (ty1 - lo + 1)) - 1
        low = high = None
        for tx in range(tx0, tx1 + 1):
            bits = (masks.get(tx, 0) >> lo) & window
            if bits:
                first = lo + (bits & -bits).bit_length() - 1
                last = lo + bits.bit_length() - 1
                low = first if low is None or first < low else low
                high = last if high is None or last > high else high
        return None if low is None else (low, high)

    def _cols(self, tx0: int, tx1: int, ty0: int, ty1: int):
        """Левый и правый столбцы с твёрдым в прямоугольнике (или None)."""
        if ty1 < 0:
            return None
        lo = ty0 if ty0 > 0 else 0
        window = (1 << (ty1 - lo + 1)) - 1
        masks = self.masks
        cols = [tx for tx in range(tx0, tx1 + 1) if (masks.get(tx, 0) >> lo) & window]
        return (cols[0], cols[-1]) if cols else None

    def standable(self, tx: int, ty: int) -> bool:
        return (tx, ty) not in self.solid and (tx, ty - 1) in self.solid

    def fly(self, tx: int, ty: int, direction: int, speed: int, jump: bool,
            from_edge: bool = False) -> Flight:
        """Полёт с тайла (tx, ty): прыжок (jump) или сход с края.

        direction ±1, speed — средняя скорость 0..4 px/кадр (нажатия
        клавиши распределены равномерно, как у игрока). from_edge — старт
        не из центра тайла, а с самого края опоры (центр за краем на
        EDGE_OVERHANG): так прыгает игрок, добежавший до обрыва.
        """

        hw = PLAYER_HALF_W + SOLID_MARGIN
        hh = PLAYER_HALF_H
        x = float(tx * TILE + HALF)
        if from_edge:
            x = float((tx + 1) * TILE + EDGE_OVERHANG if direction > 0 else tx * TILE - EDGE_OVERHANG)
        y = float(ty * TILE) + hh
        vy = 0.0
        airborne = False
        jump_pending = jump
        walk_limit = TILE + 2 * EDGE_OVERHANG
        start_x = float(tx * TILE + HALF)
        coins: set = set()
        blocks: set = set()
        masks = self.masks
        coin_masks = self.coin_masks
        block_masks = self.block_masks
        pole_x = self.pole_x
        lowest_top = self.lowest_top
        eps = 1e-6

        def done(end, node=None, fraction=0.0, frames=0):
            return Flight(end, node, fraction, frozenset(coins), frozenset(blocks), frames)

        for frame in range(MAX_FLIGHT_FRAMES):
            # Клавиша по X в этом кадре (равномерные нажатия)
            press = (frame + 1) * speed // 4 - frame * speed // 4
            vx = RUN_SPEED * direction * press

            # --- Гравитация и шаг по Y
            vy -= GRAVITY
            y += vy
            tx0 = int((x - hw) // TILE)
            tx1 = int((x + hw - eps) // TILE)
            rows = self._rows(masks, tx0, tx1, int((y - hh) // TILE), int((y + hh - eps) // TILE))
            if rows:
                if vy > 0:
                    y = rows[0] * TILE - hh
                else:
                    y = (rows[1] + 1) * TILE + hh
                vy = 0.0

            ty0 = int((y - hh) // TILE)
            ty1 = int((y + hh - eps) // TILE)

            # --- Шаг по X
            if vx:
                x += vx
                cols = self._cols(int((x - hw) // TILE), int((x + hw - eps) // TILE), ty0, ty1)
                if cols:
                    if vx > 0:
                        x = cols[0] * TILE - hw
                    else:
                        x = (cols[1] + 1) * TILE + hw
                tx0 = int((x - hw) // TILE)
                tx1 = int((x + hw - eps) // TILE)

            # --- Опора (can_jump) и прыжок
            on_ground = self._rows(masks, tx0, tx1, int((y - hh - GROUND_PROBE) // TILE),
                                   int((y - hh - eps) // TILE)) is not None
            if jump_pending:
                jump_pending = False
                if on_ground:
                    vy = JUMP_SPEED
                    airborne = True
                    on_ground = False

            # --- Монеты и удары по «?» (настоящий хитбокс без запаса)
            cx0 = int((x - PLAYER_HALF_W) // TILE)
            cx1 = int((x + PLAYER_HALF_W - eps) // TILE)
            for col in (cx0, cx1) if cx0 != cx1 else (cx0,):
                if col in coin_masks:
                    _collect(coin_masks[col], col, ty0, ty1, coins)
                if vy > 0 and col in block_masks:
                    _collect(block_masks[col], col, ty0, ty1, blocks)

            # --- Флагшток: захват по X, высота — доля штока
            if pole_x is not None and x + PLAYER_HALF_W >= pole_x - POLE_REACH \
                    and x - PLAYER_HALF_W <= pole_x + POLE_REACH:
                height = min(max(y - hh - self.pole_bottom, 0.0), self.pole_height)
                return done("goal", None, height / self.pole_height, frame + 1)

            if y < DEATH_Y or (vy < 0 and y - hh < lowest_top - 1):
                return done("death", frames=frame + 1)

            if not on_ground:
                airborne = True
            elif airborne and vy <= 0:
                return done("land", self._landing_node(x, y - hh), frames=frame + 1)
            elif not jump and abs(x - start_x) >= walk_limit:
                # Сход с края не случился: дошли до соседнего тайла по земле
                return done("walk", frames=frame + 1)

        return done("timeout", frames=MAX_FLIGHT_FRAMES)

    def _landing_node(self, x: float, bottom: float) -> Optional[tuple[int, int]]:
        ty = int(round(bottom / TILE))
        tx = int(x // TILE)
        if self.standable(tx, ty):
            return (tx, ty)
        # Стоим краем хитбокса на соседнем столбце
        for other in (int((x - PLAYER_HALF_W) // TILE), int((x + PLAYER_HALF_W - 1e-6) // TILE)):
            if self.standable(other, ty):
                return (other, ty)
        return None


def _column_masks(cells) -> dict[int, int]:
    masks: dict[int, int] = {}
    for tx, ty in cells:
        if ty >= 0:
            masks[tx] = masks.get(tx, 0) | (1 << ty)
    return masks


def _collect(mask: int, col: int, ty0: int, ty1: int, out: set) -> None:
    """Клетки столбца col из маски, попавшие в ряды ty0..ty1."""
    for ty in range(max(ty0, 0), ty1 + 1):
        if mask >> ty & 1:
            out.add((col, ty))


# ===============================================================
# (б) ГРАФ ДОСТИЖИМОСТИ
# ===============================================================


@dataclass
class Edge:
    target: Optional[tuple[int, int]]  # None — флагшток
    kind: str  # "walk" | "jump" | "drop" | "goal"
    safe: bool
    coins: frozenset = frozenset()
    blocks: frozenset = frozenset()
    fraction: float = 0.0


@dataclass
class Reach:
    nodes: set
    edges: dict  # узел → список Edge
    start: Optional[tuple[int, int]]
    safe_forward: set
    safe_goal: bool
    safe_jumps: int
    forward: set
    backward: set  # узлы, откуда (физикой) можно дойти до флагштока
    goal: bool
    best_fraction: float
    coins: set
    blocks: set


def build_graph(layout: Layout) -> Reach:
    physics = TilePhysics(layout)
    pole_tx = layout.pole_tx

    start = None
    if layout.spawn is not None:
        sx, sy = layout.spawn
        tx = int(sx // TILE)
        ty = int(sy // TILE)
        # Старт в воздухе — падаем до первой опоры
        while ty > 0 and not physics.standable(tx, ty) and (tx, ty - 1) not in layout.solid:
            ty -= 1
        if physics.standable(tx, ty):
            start = (tx, ty)

    edges: dict[tuple[int, int], list[Edge]] = {}

    def expand(node):
        tx, ty = node
        out: list[Edge] = []

        for d in (1, -1):
            nxt = (tx + d, ty)
            if physics.standable(*nxt):
                out.append(Edge(nxt, "walk", True))
            elif nxt not in layout.solid:
                # Край: сход с разной скоростью
                for speed in WALK_OFF_SPEEDS:
                    out.append(_edge(physics.fly(tx, ty, d, speed, False), node, "drop", pole_tx))

        for speed in JUMP_SPEEDS:
            for d in ((1,) if speed == 0 else (1, -1)):
                out.append(_edge(physics.fly(tx, ty, d, speed, True), node, "jump", pole_tx))
                # С края обрыва — дальше на полтайла
                edge = (tx + d, ty)
                if speed and not physics.standable(*edge) and edge not in layout.solid:
                    out.append(_edge(physics.fly(tx, ty, d, speed, True, from_edge=True),
                                     node, "jump", pole_tx))

        # Идём в флагшток по земле
        if pole_tx is not None and tx + 1 == pole_tx and ty == (layout.base_ty or 0):
            out.append(Edge(None, "goal", True, fraction=0.0))

        return [e for e in out if e is not None]

    # --- Прямой обход: все узлы, куда можно попасть
    forward: set = set()
    goal = False
    best_fraction = -1.0
    if start is not None:
        queue = deque([start])
        forward.add(start)
        while queue:
            node = queue.popleft()
            out = edges.setdefault(node, expand(node))
            for e in out:
                if e.target is None:
                    goal = True
                    best_fraction = max(best_fraction, e.fraction)
                elif e.target not in forward:
                    forward.add(e.target)
                    queue.append(e.target)

    # --- Обратный обход: откуда можно дойти до флагштока
    reverse: dict = {}
    to_goal: set = set()
    for node, out in edges.items():
        for e in out:
            if e.target is None:
                to_goal.add(node)
            else:
                reverse.setdefault(e.target, []).append(node)
    backward = set(to_goal)
    queue = deque(to_goal)
    while queue:
        node = queue.popleft()
        for prev in reverse.get(node, ()):
            if prev not in backward:
                backward.add(prev)
                queue.append(prev)

    # --- Осторожный путь: только безопасные рёбра
    safe_forward: set = set()
    safe_goal = False
    safe_jumps = 0
    if start is not None:
        parent = {start: None}
        queue = deque([start])
        safe_forward.add(start)
        goal_from = None
        while queue:
            node = queue.popleft()
            for e in edges.get(node, ()):
                if not e.safe:
                    continue
                if e.target is None:
                    safe_goal = True
                    goal_from = goal_from or node
                elif e.target not in safe_forward:
                    safe_forward.add(e.target)
                    parent[e.target] = node
                    queue.append(e.target)
        # Число прыжков на кратчайшем осторожном пути (для отчёта)
        node = goal_from
        while node is not None and parent.get(node) is not None:
            prev = parent[node]
            if prev[1] != node[1] or abs(prev[0] - node[0]) > 1:
                safe_jumps += 1
            node = prev

    # --- Что можно собрать и всё равно дойти до флага
    coins: set = set()
    blocks: set = set()
    useful = forward & backward
    for node in useful:
        if node in layout.coins:
            coins.add(node)
        for e in edges.get(node, ()):
            if e.target is None or e.target in useful:
                coins |= e.coins
                blocks |= e.blocks

    return Reach(
        nodes=set(edges), edges=edges, start=start,
        safe_forward=safe_forward, safe_goal=safe_goal, safe_jumps=safe_jumps,
        forward=forward, backward=backward, goal=goal,
        best_fraction=max(best_fraction, 0.0), coins=coins, blocks=blocks,
    )


def _edge(flight: Flight, node, kind: str, pole_tx) -> Optional[Edge]:
    tx, ty = node
    if flight.end == "goal":
        safe = pole_tx is not None and pole_tx - tx <= SAFE_SHIFT_TILES + 1
        return Edge(None, "goal", safe, flight.coins, flight.blocks, flight.fraction)
    if flight.end != "land" or flight.node is None or flight.node == node:
        if flight.end == "land" and flight.node == node and (flight.coins or flight.blocks):
            # Прыжок на месте за монетой или по блоку «?»
            return Edge(node, kind, True, flight.coins, flight.blocks)
        return None
    nx, ny = flight.node
    safe = (ny - ty) <= SAFE_RISE_TILES and abs(nx - tx) <= SAFE_SHIFT_TILES
    return Edge(flight.node, kind, safe, flight.coins, flight.blocks)


# ===============================================================
# (в) ПРОГОН БОТА
# ===============================================================


def run_bot(data: dict, name: str, time_limit: Optional[float]) -> dict:
    """HeuristicBot по уровню в GameWorld (без окна)."""

    from core.world import GameWorld
    from tools.heuristic_bot import HeuristicBot

    world = GameWorld(data, name=name, time_limit=time_limit)
    bot = HeuristicBot()
    bot.reset()
    limit = int(world.time_limit * 60) + 3000
    while not world.done and world.frame < limit:
        world.step(bot.act(world))
    return world.summary()


# ===============================================================
# СБОРКА ОТЧЁТА
# ===============================================================


def check_level(level: LevelSource, *, name: Optional[str] = None,
                time_limit: Optional[float] = None, bot: bool = True) -> LevelReport:
    t0 = time.perf_counter()
    data = load_level(level)
    source = level if isinstance(level, str) else (data.get("name") or "<dict>")
    report = LevelReport(name=name or data.get("name") or str(source), source=str(source))
    report.time_limit = time_limit if time_limit is not None else data.get("time_limit")

    if report.time_limit is None:
        report.add("warning", "time", "нет time_limit: будет 200 с по умолчанию")

    layout = check_structure(data, report)

    if layout.spawn is not None and layout.pole_x is not None:
        reach = build_graph(layout)
        report.safe_path = reach.safe_goal
        report.physics_path = reach.goal
        report.safe_jumps = reach.safe_jumps

        if reach.start is None:
            report.add("error", "reach", "старт не на опоре")
        elif not reach.goal:
            report.add("error", "reach", "флагшток недостижим даже по физике")
        elif not reach.safe_goal:
            far = max((n[0] for n in reach.safe_forward), default=0)
            report.add("error", "reach",
                       f"нет осторожного пути (ямы ≤ 4, уступы ≤ 4): дальше столбца {far} не пройти")

        dead_ends = reach.forward - reach.backward
        if dead_ends:
            cols = sorted({n[0] for n in dead_ends})
            report.add("warning", "trap",
                       f"{len(dead_ends)} мест, откуда не выбраться к флагу (столбцы {cols[:8]}…)")

        c = report.ceiling
        c.coins = len(reach.coins)
        c.coins_total = len(layout.coins)
        c.block_hits = sum(layout.blocks[b] for b in reach.blocks)
        c.block_hits_total = sum(layout.blocks.values())
        c.enemies = len(layout.enemies)
        c.flag_fraction = reach.best_fraction if reach.goal else 0.0
        c.flag = flag_points(c.flag_fraction) if reach.goal else 0

        lost_coins = c.coins_total - c.coins
        lost_blocks = len(layout.blocks) - len(reach.blocks)
        if lost_coins or lost_blocks:
            report.add("warning", "unreachable",
                       f"не собрать: монет {lost_coins}, блоков «?» {lost_blocks}")

        # Самый быстрый пробег: от старта до двери на 4 px/кадр
        door = layout.door_x or layout.pole_x
        report.min_run_seconds = max(door - layout.spawn[0], 0.0) / RUN_SPEED / 60
        limit = report.time_limit if report.time_limit is not None else 200.0
        c.time_bonus_max = max(int(limit - report.min_run_seconds), 0) * TIME_BONUS_PER_SECOND
        if limit < report.min_run_seconds * 1.3:
            report.add("warning", "time",
                       f"лимит {limit:g} с при минимальном пробеге {report.min_run_seconds:.0f} с")

    if bot:
        try:
            report.bot = run_bot(data, report.name, report.time_limit)
        except Exception as error:  # бот — только для сведения
            report.add("warning", "bot", f"прогон бота упал: {error!r}")

    report.seconds = time.perf_counter() - t0
    return report


def format_report(report: LevelReport) -> str:
    c = report.ceiling
    lines = [
        f"== {report.source} — «{report.name}»: {report.width_tiles} тайлов, "
        f"лимит {report.time_limit if report.time_limit is not None else '—'} с"
    ]
    structure_errors = [i for i in report.issues if i.level == "error" and i.code != "reach"]
    lines.append("структура: " + ("OK" if not structure_errors else f"{len(structure_errors)} ошибок"))
    lines.append(
        "проходимость: "
        + ("осторожный путь есть" if report.safe_path else "НЕТ осторожного пути")
        + f" (прыжков на нём: {report.safe_jumps}); по физике — "
        + ("флаг достижим" if report.physics_path else "флаг недостижим")
    )
    lines.append(
        f"потолок: {c.total} = монеты {c.coins}/{c.coins_total}×100 + удары «?» "
        f"{c.block_hits}/{c.block_hits_total}×100 + враги {c.enemies}×100 + флаг {c.flag} "
        f"(доля {c.flag_fraction:.2f}); +время до {c.time_bonus_max} "
        f"(пробег ≥ {report.min_run_seconds:.0f} с) → {c.total_with_time}"
    )
    if report.bot is not None:
        b = report.bot
        if b.get("won"):
            lines.append(
                f"бот: прошёл, {b['score']} очков (флаг {b['flag_bonus']}, время {b['time_bonus']}, "
                f"монет {b['coins']}, блоков {b['blocks']}, врагов {b['enemies']})"
            )
        else:
            lines.append(
                f"бот: не прошёл — {b.get('death')}, прогресс {b['progress']:.0%}, {b['score']} очков"
            )
    for issue in report.issues:
        lines.append(f"  {issue}")
    return "\n".join(lines)


def format_table(reports: list[tuple[str, str, LevelReport]]) -> str:
    head = (f"{'id':<14}{'тайлов':>7}{'лимит':>7} {'split':<6}{'стр.':<5}{'путь':<6}"
            f"{'потолок':>9}{'флаг':>6}  бот")
    lines = [head, "-" * len(head)]
    for level_id, split, r in reports:
        b = r.bot
        if b is None:
            bot = "—"
        elif b.get("won"):
            bot = f"прошёл {b['score']}"
        else:
            bot = f"{b.get('death')} {b['progress']:.0%}, {b['score']}"
        structure_ok = not any(i.level == "error" and i.code != "reach" for i in r.issues)
        lines.append(
            f"{level_id:<14}{r.width_tiles:>7}{(r.time_limit or 0):>7g} {split:<6}"
            f"{'OK' if structure_ok else 'ОШ':<5}{'OK' if r.safe_path else 'НЕТ':<6}"
            f"{r.ceiling.total:>9}{r.ceiling.flag:>6}  {bot}"
        )
    return "\n".join(lines)


# ===============================================================
# СВЕРКА ФИЗИКИ С GameWorld
# ===============================================================


def verify_physics() -> list[str]:
    """Сверяет числа модуля с ядром и прыжок симуляции с GameWorld.

    Возвращает список расхождений (пустой — всё сходится).
    """

    from core import rules
    from core.world import GameWorld, Inputs
    from tools.level_tiles import LevelBuilder

    problems = []
    if rules.DEATH_Y != DEATH_Y:
        problems.append(f"DEATH_Y: {rules.DEATH_Y} против {DEATH_Y}")
    if tuple(rules.FLAG_TIERS) != FLAG_TIERS:
        problems.append("FLAG_TIERS разошлись с core/rules.py")
    for ours, theirs in ((COIN_POINTS, rules.COIN_POINTS), (BLOCK_POINTS, rules.BLOCK_POINTS),
                         (ENEMY_POINTS, rules.ENEMY_POINTS),
                         (TIME_BONUS_PER_SECOND, rules.TIME_BONUS_PER_SECOND)):
        if ours != theirs:
            problems.append(f"очки разошлись: {ours} против {theirs}")

    # Прыжок с разбега по ровному полу: высота и дальность
    b = LevelBuilder("сверка").ground(0, 60).start(2).finish(50)
    world = GameWorld(b.to_level())
    world.step(Inputs())
    x0, bottom0 = world.player.center_x, world.player.bottom
    peak, frames = bottom0, 0
    world.step(Inputs(right=True, jump=True))
    while frames < 200:
        world.step(Inputs(right=True))
        frames += 1
        peak = max(peak, world.player.bottom)
        if world.on_ground:
            break
    rise, run = peak - bottom0, world.player.center_x - x0
    if abs(rise - 217.5) > 1.0:
        problems.append(f"высота прыжка {rise:.1f} px, ожидалось 217.5")
    if abs(run - 244) > 8:
        problems.append(f"дальность прыжка {run:.0f} px, ожидалось ~244")
    return problems


def verify_flights(level: LevelSource, samples: int = 300, seed: int = 0) -> dict:
    """Сверка полётов симуляции с GameWorld на конкретном уровне.

    Берёт случайные рёбра графа (узел, скорость, прыжок/сход, с края или из
    центра), ставит игрока в GameWorld в ту же точку, подаёт те же нажатия
    и сравнивает исход: куда приземлился, схватил ли флаг и на какой доле,
    погиб ли. Враги из уровня убираются — они в симуляции не участвуют.
    """

    from core.world import DEAD, FLAGPOLE, GameWorld, Inputs

    data = load_level(level)
    quiet = dict(data)
    quiet["objects"] = [o for o in data.get("objects", [])
                        if str(o.get("resource", "")).lower() not in ENEMY_KINDS]
    layout = check_structure(quiet, LevelReport("сверка", "сверка"))
    reach = build_graph(layout)
    physics = TilePhysics(layout)
    rng = random.Random(seed)

    nodes = sorted(reach.forward)
    world = GameWorld(quiet)
    stats = Counter()
    mismatches = []

    for _ in range(samples):
        tx, ty = rng.choice(nodes)
        jump = rng.random() < 0.7
        direction = rng.choice((1, -1))
        speed = rng.choice(JUMP_SPEEDS if jump else WALK_OFF_SPEEDS)
        edge_cell = (tx + direction, ty)
        from_edge = (jump and speed > 0 and not physics.standable(*edge_cell)
                     and edge_cell not in layout.solid and rng.random() < 0.5)
        if not jump and (physics.standable(*edge_cell) or edge_cell in layout.solid):
            continue

        sim = physics.fly(tx, ty, direction, speed, jump, from_edge=from_edge)
        if sim.end in ("walk", "timeout"):
            continue

        # --- Тот же полёт в настоящем мире
        world.reset()
        player = world.player
        x = float(tx * TILE + HALF)
        if from_edge:
            x = float((tx + 1) * TILE + EDGE_OVERHANG if direction > 0
                      else tx * TILE - EDGE_OVERHANG)
        player.center_x, player.center_y = x, float(ty * TILE) + PLAYER_HALF_H
        player.change_x = player.change_y = 0.0

        real_end, real_node, real_fraction = "timeout", None, 0.0
        airborne = False
        for frame in range(MAX_FLIGHT_FRAMES):
            press = (frame + 1) * speed // 4 - frame * speed // 4
            world.step(Inputs(right=bool(press and direction > 0),
                              left=bool(press and direction < 0),
                              jump=jump and frame == 0))
            if world.phase == FLAGPOLE:
                real_end, real_fraction = "goal", world.flag_fraction or 0.0
                break
            if world.phase == DEAD:
                real_end = "death"
                break
            if not world.on_ground:
                airborne = True
            elif airborne and player.change_y <= 0:
                real_end = "land"
                real_node = physics._landing_node(player.center_x, player.bottom)
                break

        stats["total"] += 1
        same = real_end == sim.end and (
            real_end != "land" or real_node == sim.node
        ) and (real_end != "goal" or flag_points(real_fraction) == flag_points(sim.fraction))
        if same:
            stats["same"] += 1
        else:
            stats[f"{sim.end}→{real_end}"] += 1
            if len(mismatches) < 12:
                mismatches.append(
                    f"({tx},{ty}) {'прыжок' if jump else 'сход'} d={direction} v={speed}"
                    f"{' с края' if from_edge else ''}: симуляция {sim.end} {sim.node or ''}"
                    f"{round(sim.fraction, 2) if sim.end == 'goal' else ''}, игра {real_end} "
                    f"{real_node or ''}{round(real_fraction, 2) if real_end == 'goal' else ''}"
                )

    return {"stats": dict(stats), "mismatches": mismatches}


# ===============================================================
# CLI
# ===============================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверка уровней")
    parser.add_argument("levels", nargs="*", help="пути к JSON уровней")
    parser.add_argument("--all", action="store_true", help="все уровни из maps/levels.json")
    parser.add_argument("--no-bot", action="store_true", help="не гонять HeuristicBot")
    parser.add_argument("--json", action="store_true", help="вывод в JSON")
    parser.add_argument("--verify-physics", action="store_true",
                        help="сверить физику проверки с GameWorld")
    args = parser.parse_args(argv)

    if args.verify_physics:
        problems = verify_physics()
        print("физика сходится" if not problems else "\n".join(problems))
        if not args.levels and not args.all:
            return 1 if problems else 0

    jobs: list[tuple[str, str, str, Optional[str], Optional[float]]] = []
    if args.all:
        from tools.level_tiles import MAPS_DIR
        manifest = json.loads((MAPS_DIR / "levels.json").read_text(encoding="utf-8"))
        for item in manifest.get("levels", []):
            jobs.append((str(item["id"]), str(item.get("split", "test")),
                         str(MAPS_DIR / item["file"]), item.get("name"), item.get("time_limit")))
    for path in args.levels:
        jobs.append((path, "—", path, None, None))

    if not jobs:
        parser.print_help()
        return 2

    reports = []
    for level_id, split, path, name, limit in jobs:
        report = check_level(path, name=name, time_limit=limit, bot=not args.no_bot)
        reports.append((level_id, split, report))
        if not args.json:
            print(format_report(report))
            print()

    if args.json:
        print(json.dumps([dict(id=i, split=s, **r.as_dict()) for i, s, r in reports],
                         ensure_ascii=False, indent=2))
    elif len(reports) > 1:
        print(format_table(reports))

    return 0 if all(r.ok for _, _, r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Проверка уровня: структура, проходимость, «потолок» очков, прогон бота.

    .venv/bin/python -m tools.level_check maps/level_warmup.json
    .venv/bin/python -m tools.level_check --all            # все уровни манифеста
    .venv/bin/python -m tools.level_check --all --no-bot   # без прогона бота
    .venv/bin/python -m tools.level_check --all --no-sim   # без прогона труб в GameWorld
    .venv/bin/python -m tools.level_check --all --thorough # + замер тщательного прохода
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

    Блок «?» твёрдый, как кирпич: он в стенах и обрывает прыжок; удар
    по нему — упор головой снизу (засчитывается блок, под которым больше
    головы). Каждый блок «?» должен выбиваться: под ним пол, низ блока на
    3–4 тайла выше пола (до 7 выбивается, но неудобно), и с этого пола
    можно дойти до флага.

(в) Секретные комнаты («warps»): труба-вход собрана и достижима со
    старта; комната — закрытая коробка, точка появления внутри, устье
    выхода на уровне пола и достижимо от точки появления; труба-выход
    собрана, над ней 2+ тайла свободно, с неё можно дойти до флага.
    Плюс прогон в настоящем GameWorld (без врагов): сценарный бот идёт
    по осторожному пути графа к трубе, держит «вниз», в комнате идёт к
    устью, вылезает из трубы-выхода и стоит на ней в фазе playing.
    Прыжок в устье с зажатым «вправо» — это дошёл (фаза pipe_side);
    приземлился не в ту клетку графа — путь перестраивается оттуда.

(г) Прогон tools.heuristic_bot.HeuristicBot — для сведения, не требование.

(д) Потолок очков: монеты и блоки «?», до которых можно добраться и после
    этого дойти до флага (в комнатах — до устья); все враги; лучшая доля
    флагштока, которую даёт физика (5000 — если можно схватиться за верх).
    Отдельно — бонус за время при самом быстром пробеге (оценка сверху) и
    время тщательного прохождения (путь, все удары «?», монеты, комнаты):
    лимит должен покрывать его с запасом 30 %.

(е) --thorough: тот же сценарный бот проходит уровень тщательно в
    GameWorld (без врагов): слева направо собирает всё, что даёт граф,
    каждый удар по каждому «?», с комнатой и без неё. Замер по таймеру
    уровня; лимит меньше замера — ошибка, меньше замера × 1.3 — внимание.
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
    THOROUGH_MARGIN,
    TILE,
    LevelSource,
    block_hits,
    is_solid,
    level_cells,
    load_level,
    on_grid,
    thorough_seconds,
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


# Число ударов в блоке «?» — та же формула, что в Objects/ScoreBlock.py
score_block_hits = block_hits

# Блок «?» выбивается удобно, если его низ на столько тайлов выше пола
BLOCK_GAP_OK = (3, 4)
# Самый высокий блок, который ещё выбивается: голова стоящего игрока на
# 1 тайл выше пола, прыжок поднимает её ещё на 6.8 — до 7.8 тайла. Низ
# блока на 7 свободных рядах выше пола достаётся, на 8 — уже нет
BLOCK_GAP_MAX = 7


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
    secrets: int = 0

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
    thorough_seconds: float = 0.0
    bot: Optional[dict] = None
    warps: list[dict] = field(default_factory=list)  # итоги прогона труб в GameWorld
    totals: Optional[dict] = None  # world.totals
    tours: list[dict] = field(default_factory=list)  # тщательные проходы в GameWorld
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
                "secrets": self.ceiling.secrets,
            },
            "min_run_seconds": round(self.min_run_seconds, 1),
            "thorough_seconds": round(self.thorough_seconds, 1),
            "totals": self.totals,
            "warps": self.warps,
            "tours": self.tours,
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


def check_structure(data: dict, report: LevelReport, *, room: bool = False) -> Layout:
    """Структура уровня (room=True — секретной комнаты: без финиша, старт в воздухе)."""
    cells = level_cells(data)

    # Блок «?» — твёрдый (core/world.py: Area кладёт его в walls)
    solid = {(c.tx, c.ty) for c in cells if is_solid(c.kind)}
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
            elif not room:  # в комнату игрок падает сверху — так и задумано
                report.add("warning", "start", f"старт ({s.tx},{s.ty}) висит в воздухе")

    # --- Враги стоят на твёрдом
    for kind, tx, ty in enemies:
        if (tx, ty) in solid:
            report.add("error", "enemy", f"{kind} ({tx},{ty}) внутри блока")
        elif (tx, ty - 1) not in solid:
            report.add("error", "enemy", f"{kind} ({tx},{ty}) не стоит на блоке")

    # --- Монеты не внутри твёрдого (двойная клетка поймана выше)
    for key in coins:
        if key in solid:
            report.add("error", "overlap", f"монета в твёрдой клетке {key}")

    layout = Layout(solid, coins, blocks, enemies, spawn)

    if not room:
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

    Блоки «?» твёрдые, удар засчитывается при упоре головой снизу — тому
    блоку, под которым больше головы, если кирпич рядом не накрывает её
    сильнее (core/world.py: GameWorld._head_bump).

    Твёрдые клетки хранятся битовыми масками по столбцам: столбец → число,
    где бит ty = 1, если клетка твёрдая. Игрок перекрывает не больше двух
    столбцов и двух рядов, поэтому проверка кадра — пара сдвигов и масок.
    """

    def __init__(self, layout: Layout):
        self.solid = layout.solid
        self.masks = _column_masks(layout.solid)
        self.coin_masks = _column_masks(layout.coins)
        self.block_cells = set(layout.blocks)
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

    def bumped(self, x: float, row: int) -> Optional[tuple[int, int]]:
        """Блок «?», по которому пришёлся удар головой в ряд row (или None)."""
        eps = 1e-6
        left, right = x - PLAYER_HALF_W, x + PLAYER_HALF_W
        best, best_overlap, other = None, 0.0, 0.0
        for col in range(int(left // TILE), int((right - eps) // TILE) + 1):
            if (col, row) not in self.solid:
                continue
            overlap = min(right, (col + 1) * TILE) - max(left, col * TILE)
            if (col, row) in self.block_cells:
                if overlap > best_overlap:
                    best, best_overlap = (col, row), overlap
            else:
                other = max(other, overlap)
        if best is None or other > best_overlap:
            return None
        return best

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
        block_cells = self.block_cells
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
                    # Упор головой: удар по блоку «?»
                    if block_cells:
                        hit = self.bumped(x, rows[0])
                        if hit is not None:
                            blocks.add(hit)
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

            # --- Монеты (настоящий хитбокс без запаса)
            cx0 = int((x - PLAYER_HALF_W) // TILE)
            cx1 = int((x + PLAYER_HALF_W - eps) // TILE)
            for col in (cx0, cx1) if cx0 != cx1 else (cx0,):
                if col in coin_masks:
                    _collect(coin_masks[col], col, ty0, ty1, coins)

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
    source: Optional[tuple[int, int]] = None
    # Как повторить ребро: направление, скорость 0..4, прыжок, старт с края
    move: tuple = (0, 0, False, False)


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


def build_graph(layout: Layout, goal_nodes: Optional[set] = None) -> Reach:
    """Граф достижимости. goal_nodes — узлы, которые считаются финишем
    вместо флагштока (в комнате — клетка перед устьем выхода)."""
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
                out.append(Edge(nxt, "walk", True, source=node, move=(d, 4, False, False)))
            elif nxt not in layout.solid:
                # Край: сход с разной скоростью
                for speed in WALK_OFF_SPEEDS:
                    out.append(_edge(physics.fly(tx, ty, d, speed, False), node, "drop", pole_tx,
                                     (d, speed, False, False)))

        for speed in JUMP_SPEEDS:
            for d in ((1,) if speed == 0 else (1, -1)):
                out.append(_edge(physics.fly(tx, ty, d, speed, True), node, "jump", pole_tx,
                                 (d, speed, True, False)))
                # С края обрыва — дальше на полтайла
                edge = (tx + d, ty)
                if speed and not physics.standable(*edge) and edge not in layout.solid:
                    out.append(_edge(physics.fly(tx, ty, d, speed, True, from_edge=True),
                                     node, "jump", pole_tx, (d, speed, True, True)))

        # Идём в флагшток по земле
        if pole_tx is not None and tx + 1 == pole_tx and ty == (layout.base_ty or 0):
            out.append(Edge(None, "goal", True, fraction=0.0, source=node))
        if goal_nodes and node in goal_nodes:
            out.append(Edge(None, "goal", True, source=node))

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


def _edge(flight: Flight, node, kind: str, pole_tx, move=(0, 0, False, False)) -> Optional[Edge]:
    tx, ty = node
    if flight.end == "goal":
        safe = pole_tx is not None and pole_tx - tx <= SAFE_SHIFT_TILES + 1
        return Edge(None, "goal", safe, flight.coins, flight.blocks, flight.fraction,
                    source=node, move=move)
    if flight.end != "land" or flight.node is None or flight.node == node:
        if flight.end == "land" and flight.node == node and (flight.coins or flight.blocks):
            # Прыжок на месте за монетой или по блоку «?»
            return Edge(node, kind, True, flight.coins, flight.blocks, source=node, move=move)
        return None
    nx, ny = flight.node
    safe = (ny - ty) <= SAFE_RISE_TILES and abs(nx - tx) <= SAFE_SHIFT_TILES
    return Edge(flight.node, kind, safe, flight.coins, flight.blocks, source=node, move=move)


def find_path(reach: Reach, start, target, safe_only: bool = True) -> Optional[list[Edge]]:
    """Кратчайший путь по рёбрам графа от start до target (узлы) или None.
    Сначала только осторожные рёбра; не нашлось — любые."""
    if start is None or target is None:
        return None
    if start == target:
        return []
    parent: dict = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for e in reach.edges.get(node, ()):
            if e.target is None or e.target in parent or (safe_only and not e.safe):
                continue
            parent[e.target] = e
            if e.target == target:
                path = []
                while parent[target] is not None:
                    path.append(parent[target])
                    target = parent[target].source
                return path[::-1]
            queue.append(e.target)
    return find_path(reach, start, target, safe_only=False) if safe_only else None


# ===============================================================
# (б) БЛОКИ «?»
# ===============================================================


def check_blocks(layout: Layout, reach: Reach, report: LevelReport, where: str = "") -> None:
    """Каждый блок «?» выбивается: под ним пол, низ блока на 3–4 тайла выше
    пола, с этого пола можно уйти дальше (к флагу или к устью комнаты), и
    граф действительно находит удар по нему."""

    useful = reach.forward & reach.backward
    for (tx, ty) in sorted(layout.blocks):
        floor = max((r for (x, r) in layout.solid if x == tx and r < ty), default=None)
        name = f"{where}«?» ({tx},{ty})"
        if floor is None:
            report.add("error", "block", f"{name} над ямой: под ним нет пола, он только обрывает прыжки")
            continue
        gap = ty - floor - 1  # свободных рядов между полом и блоком
        stand = (tx, floor + 1)
        if gap < 1 or gap > BLOCK_GAP_MAX:
            report.add("error", "block", f"{name}: до пола {gap} тайлов — не выбить")
        elif not BLOCK_GAP_OK[0] <= gap <= BLOCK_GAP_OK[1]:
            report.add("warning", "block", f"{name}: низ на {gap} тайла выше пола, удобно 3–4")
        if (tx, ty) not in reach.blocks:
            report.add("error", "block", f"{name} не выбить по графу (пол {stand} недостижим "
                       "или с него не уйти дальше)")
        elif stand not in useful:
            report.add("warning", "block", f"{name} выбивается только с соседнего столбца")

    # Блок «?» над разбегом у края ямы: прыжок с разбега от края (из центра
    # крайнего тайла и с самого края) упирается в него головой, и игрок
    # падает в яму — так на «Сокровищнице» умирал бот
    physics = TilePhysics(layout)
    warned = set()
    for tx, ty in sorted(reach.forward & reach.backward):
        edge = tx + 1
        if (edge, ty) in layout.solid or (edge, ty - 1) in layout.solid:
            continue
        if any((edge, r) in layout.solid for r in range(ty)):
            continue  # ниже есть опора — уступ, а не яма
        for from_edge in (False, True):
            hit = physics.fly(tx, ty, 1, 4, True, from_edge=from_edge).blocks - warned
            for bx, by in sorted(hit):
                warned.add((bx, by))
                report.add("warning", "block", f"{where}«?» ({bx},{by}) обрывает прыжок с края "
                           f"ямы ({tx},{ty})")


# ===============================================================
# (в) СЕКРЕТНЫЕ КОМНАТЫ
# ===============================================================


@dataclass
class RoomCheck:
    key: str
    pipe: tuple[int, int]      # левый верхний тайл трубы-входа
    exit: tuple[int, int]      # левый верхний тайл трубы-выхода
    layout: Optional[Layout] = None
    reach: Optional[Reach] = None
    mouth: Optional[tuple[int, int]] = None  # нижний тайл устья
    entry_node: Optional[tuple[int, int]] = None
    exit_node: Optional[tuple[int, int]] = None  # клетка перед устьем
    seconds: float = 0.0       # оценка тщательного прохода комнаты


def _tile_point(value) -> tuple[int, int]:
    if isinstance(value, dict):
        value = (value["x"], value["y"])
    return int(float(value[0]) // TILE), int(float(value[1]) // TILE)


def _is_closed(layout: Layout, spawn_tile: tuple[int, int]) -> bool:
    """Из точки появления по пустым клеткам не выйти за рамку твёрдых блоков."""
    if not layout.solid:
        return False
    xs = [tx for tx, _ in layout.solid]
    ys = [ty for _, ty in layout.solid]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if spawn_tile in layout.solid:
        return False
    seen = {spawn_tile}
    queue = deque([spawn_tile])
    while queue:
        tx, ty = queue.popleft()
        if not (x0 <= tx <= x1 and y0 <= ty <= y1):
            return False
        for nxt in ((tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)):
            if nxt not in seen and nxt not in layout.solid:
                seen.add(nxt)
                queue.append(nxt)
    return True


def check_warps(data: dict, layout: Layout, reach: Reach, report: LevelReport) -> list[RoomCheck]:
    """Трубы-входы, комнаты и трубы-выходы (формат docs/CONTRACT.md)."""

    cells = {(c.tx, c.ty): c.kind for c in level_cells(data)}
    rooms: list[RoomCheck] = []

    for index, spec in enumerate(data.get("warps") or []):
        key = str(spec.get("id") or f"room{index + 1}")
        try:
            check = RoomCheck(key, _tile_point(spec["pipe"]), _tile_point(spec["exit"]))
            room_data = spec["room"]
        except (KeyError, TypeError, ValueError) as error:
            report.add("error", "warp", f"{key}: запись warps без pipe/exit/room ({error!r})")
            continue
        rooms.append(check)
        name = f"комната {key}"

        # --- Труба-вход: шляпка из двух половин, на ней можно стоять
        px, py = check.pipe
        if cells.get((px, py)) != "tube_top_l" or cells.get((px + 1, py)) != "tube_top_r":
            report.add("error", "warp", f"{name}: в pipe ({px},{py}) нет верха трубы "
                       "(Tubes img.png + img_1.png)")
        entry = [n for n in ((px, py + 1), (px + 1, py + 1)) if n in reach.forward]
        if not entry:
            report.add("error", "warp", f"{name}: на трубу-вход ({px},{py}) не попасть со старта")
        else:
            check.entry_node = entry[0]

        # --- Труба-выход: собрана, над ней свободно, с неё дойти до флага
        ex, ey = check.exit
        if cells.get((ex, ey)) != "tube_top_l" or cells.get((ex + 1, ey)) != "tube_top_r":
            report.add("error", "warp", f"{name}: в exit ({ex},{ey}) нет верха трубы")
        blocked = [(tx, ty) for tx in (ex, ex + 1) for ty in (ey + 1, ey + 2) if (tx, ty) in layout.solid]
        if blocked:
            report.add("error", "warp", f"{name}: над трубой-выходом занято {blocked[0]} — "
                       "нужно 2 свободных тайла")
        if (ex, ey + 1) not in reach.backward and (ex + 1, ey + 1) not in reach.backward:
            report.add("error", "warp", f"{name}: с трубы-выхода ({ex},{ey}) не дойти до флага")
        if (px, py) == (ex, ey):
            report.add("error", "warp", f"{name}: вход и выход — одна труба")
        elif ex < px:
            report.add("warning", "warp", f"{name}: труба-выход левее входа — возврат назад")
        near = [(k, tx) for k, tx, ty in layout.enemies if abs(tx - ex) <= 3 and ty <= ey + 3]
        if near:
            report.add("warning", "warp", f"{name}: враг {near[0][0]} в столбце {near[0][1]} "
                       "рядом с трубой-выходом")

        # --- Комната
        room_report = LevelReport(name, name)
        room_layout = check_structure(room_data, room_report, room=True)
        for issue in room_report.issues:
            report.add(issue.level, issue.code, f"{name}: {issue.message}")
        check.layout = room_layout
        if room_layout.spawn is None:
            continue
        spawn_tile = (int(room_layout.spawn[0] // TILE), int(room_layout.spawn[1] // TILE))
        if not _is_closed(room_layout, spawn_tile):
            report.add("error", "room", f"{name}: не закрытая коробка — из точки появления "
                       "можно выйти за стены")

        room_cells = {(c.tx, c.ty): c.kind for c in level_cells(room_data)}
        if room_data.get("exit") is None:
            report.add("error", "room", f"{name}: нет exit — устья боковой трубы")
            continue
        mx, my = _tile_point(room_data["exit"])
        check.mouth = (mx, my)
        if (mx, my) not in room_layout.solid or (mx, my + 1) not in room_layout.solid:
            report.add("error", "room", f"{name}: устье ({mx},{my}) не из двух твёрдых тайлов")
        elif (room_cells.get((mx, my)), room_cells.get((mx, my + 1))) != ("side_mouth", "side_mouth_top"):
            report.add("warning", "room", f"{name}: устье ({mx},{my}) не из текстур Tubes img_10/img_11")
        if (mx - 1, my - 1) not in room_layout.solid or (mx - 1, my) in room_layout.solid:
            report.add("error", "room", f"{name}: низ устья ({mx},{my}) не на уровне пола перед ним")
            continue
        check.exit_node = (mx - 1, my)

        room_reach = build_graph(room_layout, goal_nodes={check.exit_node})
        check.reach = room_reach
        if room_reach.start is None:
            report.add("error", "room", f"{name}: игрок из точки появления не встаёт на пол")
        elif check.exit_node not in room_reach.forward:
            report.add("error", "room", f"{name}: от точки появления не дойти до устья")
        else:
            check_blocks(room_layout, room_reach, report, where=f"{name}: ")
            lost = len(room_layout.coins) - len(room_reach.coins)
            if lost:
                report.add("warning", "unreachable", f"{name}: не собрать монет {lost}")

        run = abs(mx - spawn_tile[0])
        check.seconds = thorough_seconds(run, sum(room_layout.blocks.values()),
                                         len(room_layout.coins), extra_seconds=1.0)
    return rooms


def _strip_enemies(data: dict) -> dict:
    """Копия уровня без врагов (и в комнатах): сценарий проверяет трубы, а не бой."""

    def clean(objects):
        return [o for o in objects if str(o.get("resource", "")).lower() not in ENEMY_KINDS]

    quiet = dict(data)
    quiet["objects"] = clean(data.get("objects", []))
    quiet["warps"] = [
        dict(w, room=dict(w["room"], objects=clean(w["room"].get("objects", []))))
        for w in data.get("warps") or []
    ]
    return quiet


# Прыжок отрывает от опоры в первом кадре; сход с края из середины тайла
# на самой медленной скорости (1 px/кадр) — через ~30 кадров (16 px до
# края + 13 px полуширины). Стоит дольше — ребро не началось
FLIGHT_START_FRAMES = 45


class ScriptedWalker:
    """Сценарный бот в настоящем GameWorld: повторяет рёбра графа нажатиями.

    Перед каждым ребром подходит к его началу пешком (шаг 4 px, остаток
    меньше шага доводится сдвигом), затем подаёт те же нажатия, что
    симуляция, и сверяет, куда приземлился.

    Цель пути может быть не клеткой, а фазой мира (`finish`): прыжок к
    устью с зажатым «вправо» законно кончается заставкой pipe_side — это
    дошёл, а не сбой. Приземлился не туда (arcade ставит игрока на угол
    блока на ряд выше, чем граф) — путь перестраивается от места
    приземления по тому же графу, не больше REPLANS раз.
    """

    REPLANS = 3

    def __init__(self, world, physics: TilePhysics):
        self.world = world
        self.physics = physics
        self.replans = 0

    def node(self) -> Optional[tuple[int, int]]:
        player = self.world.player
        return self.physics._landing_node(player.center_x, player.bottom)

    def settle(self, limit: int = 90) -> Optional[tuple[int, int]]:
        """Постоять без нажатий, пока игрок не встанет на клетку графа
        (после удара головой о блок он на миг «на опоре» ещё в воздухе)."""
        from core.world import Inputs, PLAYING

        world = self.world
        for _ in range(limit):
            node = self.node()
            if node is not None and world.on_ground:
                return node
            world.step(Inputs())
            if world.phase != PLAYING:
                return None
        return self.node()

    def walk_to(self, x: float, limit: int = 600, finish: tuple = ()) -> bool:
        """Дойти до x. True — дошёл или мир ушёл в фазу из finish."""
        from core.world import Inputs, PLAYING

        world = self.world
        for _ in range(limit):
            dx = x - world.player.center_x
            if abs(dx) < RUN_SPEED:
                world.player.center_x = x
                return True
            world.step(Inputs(right=dx > 0, left=dx < 0))
            if world.phase in finish:
                return True
            if world.phase != PLAYING:
                return False
        return False

    def follow(self, path: list[Edge], reach: Optional[Reach] = None,
               target: Optional[tuple[int, int]] = None, finish: tuple = ()) -> Optional[str]:
        """Пройти путь. None — дошёл (или мир ушёл в фазу из finish), иначе —
        что пошло не так. С reach и target промах приземления не сбой:
        путь до target строится заново от клетки, куда встал игрок."""
        from core.world import FLAGPOLE, Inputs, PLAYING

        world = self.world
        steps = deque(path)
        tries = 0
        while steps:
            e = steps.popleft()
            d, speed, jump, from_edge = e.move
            tx, ty = e.source
            if e.kind == "walk":
                if not self.walk_to(e.target[0] * TILE + HALF, finish=finish):
                    return f"сошёл с пути на шаге {e.source}→{e.target}"
                if world.phase in finish:
                    return None
                continue
            x = float(tx * TILE + HALF)
            if from_edge:
                x = float((tx + 1) * TILE + EDGE_OVERHANG if d > 0 else tx * TILE - EDGE_OVERHANG)
            if not self.walk_to(x, finish=finish):
                return f"не дошёл до начала прыжка {e.source}"
            if world.phase in finish:
                return None
            here = self.node()
            if here == e.source:
                airborne = False
                for frame in range(MAX_FLIGHT_FRAMES):
                    press = (frame + 1) * speed // 4 - frame * speed // 4
                    world.step(Inputs(right=bool(press and d > 0), left=bool(press and d < 0),
                                      jump=jump and frame == 0))
                    if world.phase in finish:
                        return None
                    if world.phase not in (PLAYING, FLAGPOLE):
                        return f"прыжок {e.source}→{e.target}: фаза {world.phase}"
                    if not world.on_ground:
                        airborne = True
                    elif airborne and world.player.change_y <= 0:
                        break
                    elif not airborne and frame >= FLIGHT_START_FRAMES:
                        break  # так и не оторвался от опоры — не идти вслепую дальше
                landed = self.node() or self.settle()
                if landed == e.target:
                    continue
                miss = f"прыжок {e.source}→{e.target}: приземлился в {landed}"
            else:
                # По дороге к началу ребра сошёл с опоры (стоял на углу блока)
                landed = here
                miss = f"к прыжку {e.source}→{e.target} подошёл из {landed}"
            if reach is None or target is None or landed is None:
                return miss
            if tries >= self.REPLANS:
                return f"{miss} (перестроек пути уже {tries})"
            detour = find_path(reach, landed, target)
            if detour is None:
                return f"{miss}, оттуда нет пути к {target}"
            tries += 1
            self.replans += 1
            steps = deque(detour)
        return None

    def enter_pipe(self, warp) -> Optional[str]:
        """Со шляпки трубы-входа: встать на середину, «вниз», дождаться пола комнаты."""
        from core.world import PIPE_DOWN, Inputs, PLAYING

        world = self.world
        if not self.walk_to(warp.x):
            return "не встал на середину трубы"
        for _ in range(10):
            world.step(Inputs(down=True))
            if world.phase != PLAYING:
                break
        if world.phase != PIPE_DOWN:
            return f"«вниз» на трубе не сработал (фаза {world.phase})"
        for _ in range(600):
            world.step(Inputs())
            if world.phase == PLAYING and world.in_room and world.on_ground:
                return None
        return f"не оказался на полу комнаты (фаза {world.phase})"

    def leave_room(self, check: "RoomCheck", warp) -> Optional[str]:
        """Из комнаты к устью, «вправо» в устье, вылезти на трубу-выход."""
        from core.world import PIPE_SIDE, Inputs, PLAYING

        world = self.world
        here = self.node()
        path = find_path(check.reach, here, check.exit_node)
        if path is None:
            return f"в комнате нет пути от {here} к устью"
        problem = self.follow(path, check.reach, check.exit_node, finish=(PIPE_SIDE,))
        if problem is not None:
            return problem
        if world.phase == PLAYING:
            # Последнее ребро — шаг: у устья остановились, дожать «вправо»
            for _ in range(120):
                world.step(Inputs(right=True))
                if world.phase != PLAYING:
                    break
        if world.phase != PIPE_SIDE:
            return f"устье не приняло игрока (фаза {world.phase})"
        for _ in range(600):
            world.step(Inputs())
            if world.phase == PLAYING:
                break
        player = world.player
        if world.in_room or world.phase != PLAYING:
            return f"не вылез в уровень (фаза {world.phase})"
        if abs(player.bottom - warp.exit_top) > 1.0 or abs(player.center_x - warp.exit_x) > 2.0:
            return (f"вылез не на трубу-выход: ({player.center_x:.0f}, {player.bottom:.0f}) "
                    f"вместо ({warp.exit_x:.0f}, {warp.exit_top:.0f})")
        return None


def simulate_warps(data: dict, layout: Layout, reach: Reach, rooms: list[RoomCheck]) -> list[dict]:
    """Прогон каждой трубы в GameWorld: к трубе-входу, «вниз», в комнате
    к устью, «вправо» в устье, вылезти из трубы-выхода и встать на неё."""

    from core.world import PLAYING, GameWorld, Inputs

    quiet = _strip_enemies(data)
    physics = TilePhysics(layout)
    results = []
    for index, check in enumerate(rooms):
        result = {"room": check.key, "ok": False, "message": ""}
        results.append(result)
        if check.entry_node is None or check.reach is None or check.exit_node is None:
            result["message"] = "не проверена: структура с ошибками"
            continue

        world = GameWorld(quiet)
        warp = world.warps[index]
        walker = ScriptedWalker(world, physics)

        # --- К трубе-входу по осторожному пути, «вниз»
        path = find_path(reach, reach.start, check.entry_node)
        problem = ("нет пути до трубы-входа" if path is None
                   else walker.follow(path, reach, check.entry_node))
        if problem is None:
            problem = walker.enter_pipe(warp)
        result["enter_frame"] = world.frame

        # --- По комнате к устью, в трубу и из трубы-выхода
        if problem is None:
            room_walker = ScriptedWalker(world, TilePhysics(check.layout))
            problem = room_walker.leave_room(check, warp)
            walker.replans += room_walker.replans
        if problem is None:
            # Постоять на трубе: не падает, не умирает
            for _ in range(30):
                world.step(Inputs())
            if world.phase != PLAYING or abs(world.player.bottom - warp.exit_top) > 1.0:
                problem = "не удержался на трубе-выходе"

        result.update(
            ok=problem is None, message=problem or "OK", frames=world.frame, replans=walker.replans,
            secrets=world.stats["secret"], coins=world.stats["coin"], blocks=world.stats["block"],
        )
    return results


def world_totals(data: dict) -> dict:
    from core.world import GameWorld

    return dict(GameWorld(data).totals)


# ===============================================================
# (в') ТЩАТЕЛЬНЫЙ ПРОХОД В GameWorld
# ===============================================================
# Замер вместо формулы thorough_seconds: сценарный бот идёт по графу и
# жадно, от ближней цели к дальней, берёт всё, что граф позволяет взять,
# не теряя пути к флагу: монеты, каждый удар по каждому «?», комнату.
# Враги убраны (сценарий не умеет драться), поэтому замер — нижняя
# граница: лимит должен быть не меньше замера × THOROUGH_MARGIN, и запас
# сверх него уходит на врагов и на ошибки агента.

TOUR_TARGETS = 600  # сколько целей (монета/удар) бот пытается взять за проход
TOUR_WINDOW = 12   # ширина окна в столбцах: в нём всё собрать, потом дальше


def _tile_of(sprite) -> tuple[int, int]:
    return int(sprite.center_x // TILE), int(sprite.center_y // TILE)


def _backtrack(parent: dict, node, last: Edge) -> list[Edge]:
    path = [last]
    while parent[node] is not None:
        path.append(parent[node])
        node = parent[node].source
    return path[::-1]


def _nearest_loot(reach: Reach, start, coins: set, hits: set, cols: range,
                  skip: set) -> Optional[list[Edge]]:
    """Кратчайший по числу рёбер путь, последнее ребро которого берёт монету
    или бьёт блок «?» с ударами в столбцах cols. Только через узлы, откуда
    дальше можно уйти (reach.backward)."""

    parent: dict = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for e in reach.edges.get(node, ()):
            if e.target is None or e.target not in reach.backward:
                continue
            loot = [c for c in e.coins if c in coins and c[0] in cols]
            loot += [b for b in e.blocks if b in hits and b[0] in cols]
            if loot and (e.source, e.move, e.target) not in skip:
                return _backtrack(parent, node, e)
            if e.target not in parent:
                parent[e.target] = e
                queue.append(e.target)
    return None


def _goal_path(reach: Reach, start) -> Optional[list[Edge]]:
    """Путь до штока: до первого ребра-финиша, сначала по осторожным рёбрам."""

    for safe_only in (True, False):
        parent: dict = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for e in reach.edges.get(node, ()):
                if safe_only and not e.safe:
                    continue
                if e.target is None:
                    return _backtrack(parent, node, e)
                if e.target not in parent:
                    parent[e.target] = e
                    queue.append(e.target)
    return None


def _sweep(walker: ScriptedWalker, reach: Reach, cols: range) -> Optional[str]:
    """Собрать в текущей части уровня всё, что даёт граф в столбцах cols.

    Слева направо окнами по TOUR_WINDOW столбцов, как идёт игрок: внутри
    окна — ближайшая цель, назад за окно не возвращается. Промах (ребро
    ничего не дало) второй раз не пробуется."""
    from core.world import PLAYING

    world = walker.world
    skip: set = set()  # рёбра, после которых ничего не прибавилось
    targets = 0
    for a in range(cols.start, cols.stop, TOUR_WINDOW):
        window = range(a, min(a + TOUR_WINDOW, cols.stop))
        while targets < TOUR_TARGETS:
            node = walker.node() or walker.settle()
            if node is None:
                return "игрок не на опоре"
            coins = {_tile_of(o) for o in world.objects if type(o).__name__ == "Coin"}
            hits = {_tile_of(b) for b in world.blocks if b.state > 0}
            path = _nearest_loot(reach, node, coins, hits, window, skip)
            if path is None:
                break
            targets += 1
            before = world.stats["coin"] + world.stats["block"]
            last = path[-1]
            problem = walker.follow(path, reach, last.target)
            if world.phase != PLAYING:
                return problem or f"фаза {world.phase}"
            if problem is not None or world.stats["coin"] + world.stats["block"] == before:
                skip.add((last.source, last.move, last.target))
    return None if targets < TOUR_TARGETS else f"не уложился в {TOUR_TARGETS} целей"


def thorough_run(data: dict, layout: Layout, reach: Reach, rooms: list[RoomCheck],
                 use_rooms: bool = True) -> dict:
    """Тщательный проход в GameWorld без врагов: до трубы-входа собрать всё
    слева от неё, комната целиком, после трубы-выхода — всё до флага.
    use_rooms=False — без комнат, весь уровень подряд. Итог — время по
    таймеру уровня (заставки его не тратят) и что собрано."""

    from core.world import FLAGPOLE, PLAYING, GameWorld, Inputs

    world = GameWorld(_strip_enemies(data), time_limit=999)
    walker = ScriptedWalker(world, TilePhysics(layout))
    end = max(tx for tx, _ in layout.solid) + 1 if layout.solid else 0
    left = reach.start[0] if reach.start else 0
    problem = None

    for index, check in enumerate(rooms if use_rooms else []):
        if check.entry_node is None or check.reach is None or check.exit_node is None:
            continue
        if check.pipe[0] < left:
            continue  # труба уже позади
        warp = world.warps[index]
        problem = _sweep(walker, reach, range(left, check.pipe[0] + 2))
        if problem is None:
            path = find_path(reach, walker.node(), check.entry_node)
            problem = ("нет пути до трубы-входа" if path is None
                       else walker.follow(path, reach, check.entry_node))
        if problem is None:
            problem = walker.enter_pipe(warp)
        if problem is None:
            room_walker = ScriptedWalker(world, TilePhysics(check.layout))
            problem = _sweep(room_walker, check.reach, range(-1, 1000))
            if problem is None:
                problem = room_walker.leave_room(check, warp)
            walker.replans += room_walker.replans
        if problem is not None:
            problem = f"комната {check.key}: {problem}"
            break
        left = check.exit[0]

    if problem is None:
        problem = _sweep(walker, reach, range(left, end))
    tries = 0
    while problem is None:
        path = _goal_path(reach, walker.node() or walker.settle())
        if path is None:
            problem = "нет пути до штока"
            break
        miss = walker.follow(path, finish=(FLAGPOLE,))
        if miss is None:
            break
        if world.phase != PLAYING or tries >= ScriptedWalker.REPLANS:
            problem = f"к штоку: {miss}"
            break
        tries += 1  # промах по лестнице к штоку — ещё раз от клетки, где встал
        walker.replans += 1
    seconds = world.time_limit - world.time_left  # на штоке таймер встаёт
    if problem is None:
        while not world.done and world.frame < 60 * 60 * 20:
            world.step(Inputs())
        if not world.won:
            problem = f"не дошёл до замка (фаза {world.phase})"

    s = world.summary()
    return {
        "ok": problem is None, "message": problem or "OK", "rooms": use_rooms,
        "seconds": round(seconds, 1),
        "coins": s["coins"], "coins_total": s["coins_total"],
        "blocks": s["blocks"], "block_hits_total": s["block_hits_total"],
        "secrets": s["secrets"], "replans": walker.replans,
    }


# ===============================================================
# (г) ПРОГОН БОТА
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
                time_limit: Optional[float] = None, bot: bool = True,
                sim: bool = True, thorough: bool = False) -> LevelReport:
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

        check_blocks(layout, reach, report)
        rooms = check_warps(data, layout, reach, report)
        if not rooms:
            report.add("warning", "warp", "нет секретной комнаты (ключ warps)")

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

        # Комнаты — в потолок очков
        for room in rooms:
            if room.layout is None:
                continue
            c.coins_total += len(room.layout.coins)
            c.block_hits_total += sum(room.layout.blocks.values())
            c.enemies += len(room.layout.enemies)
            c.secrets += 1
            if room.reach is not None:
                c.coins += len(room.reach.coins)
                c.block_hits += sum(room.layout.blocks[b] for b in room.reach.blocks)

        # Самый быстрый пробег: от старта до двери на 4 px/кадр
        door = layout.door_x or layout.pole_x
        report.min_run_seconds = max(door - layout.spawn[0], 0.0) / RUN_SPEED / 60
        limit = report.time_limit if report.time_limit is not None else 200.0
        c.time_bonus_max = max(int(limit - report.min_run_seconds), 0) * TIME_BONUS_PER_SECOND
        if limit < report.min_run_seconds * 1.3:
            report.add("warning", "time",
                       f"лимит {limit:g} с при минимальном пробеге {report.min_run_seconds:.0f} с")

        # Тщательное прохождение: весь путь, все удары «?», монеты, комнаты
        report.thorough_seconds = thorough_seconds(
            max(door - layout.spawn[0], 0.0) / TILE, sum(layout.blocks.values()),
            len(layout.coins), sum(room.seconds for room in rooms),
        )
        if limit < report.thorough_seconds * THOROUGH_MARGIN:
            report.add("error", "time",
                       f"лимит {limit:g} с: тщательное прохождение ~{report.thorough_seconds:.0f} с, "
                       f"нужно ≥ {report.thorough_seconds * THOROUGH_MARGIN:.0f} с (запас 30 %)")

        if sim:
            try:
                report.totals = world_totals(data)
                report.warps = simulate_warps(data, layout, reach, rooms)
            except Exception as error:
                report.add("error", "sim", f"прогон в GameWorld упал: {error!r}")
            for result in report.warps:
                if not result["ok"]:
                    report.add("error", "sim", f"труба {result['room']}: {result['message']}")

        if thorough:
            # Замер тщательного прохода: с комнатами и без (весь уровень подряд)
            variants = (True, False) if rooms else (True,)
            try:
                report.tours = [thorough_run(data, layout, reach, rooms, use_rooms=v) for v in variants]
            except Exception as error:
                report.add("error", "tour", f"тщательный проход упал: {error!r}")
            for tour in report.tours:
                label = "с комнатой" if tour["rooms"] and rooms else "без комнаты"
                if not tour["ok"]:
                    report.add("warning", "tour", f"тщательный проход {label}: {tour['message']}")
                elif limit < tour["seconds"]:
                    report.add("error", "time", f"лимит {limit:g} с: тщательный проход {label} "
                               f"занял {tour['seconds']:.0f} с — агент умрёт по таймеру")
                elif limit < tour["seconds"] * THOROUGH_MARGIN:
                    report.add("warning", "time", f"лимит {limit:g} с: тщательный проход {label} "
                               f"{tour['seconds']:.0f} с, запас меньше 30 % (враги не в счёт)")

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
    structure_errors = [i for i in report.issues
                        if i.level == "error" and i.code not in ("reach", "time", "sim")]
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
    lines.append(
        f"тщательно: ~{report.thorough_seconds:.0f} с (путь, все удары «?», монеты, комнаты: "
        f"{c.secrets}); лимит {report.time_limit or 200:g} с = "
        f"×{(report.time_limit or 200) / max(report.thorough_seconds, 1):.2f}"
    )
    if report.totals is not None:
        t = report.totals
        lines.append(f"world.totals: монет {t['coins']}, ударов «?» {t['block_hits']}, "
                     f"врагов {t['enemies']}, секретов {t['secrets']}")
    for w in report.warps:
        lines.append(
            f"труба {w['room']}: " + (
                f"прогон OK — {w['frames']} кадров, в комнате с кадра {w.get('enter_frame')}, "
                f"секретов {w['secrets']}"
                + (f", путь перестроен {w['replans']} раз" if w.get("replans") else "")
                if w["ok"] else f"ОШИБКА — {w['message']}")
        )
    limit = report.time_limit or 200
    for t in report.tours:
        lines.append(
            f"тщательный проход {'с комнатой' if t['rooms'] and report.ceiling.secrets else 'без комнаты'}: "
            + (f"{t['seconds']:.0f} с (лимит ×{limit / max(t['seconds'], 1):.2f}), " if t["ok"]
               else f"СБОЙ — {t['message']}; ")
            + f"монет {t['coins']}/{t['coins_total']}, ударов «?» {t['blocks']}/{t['block_hits_total']}, "
            f"секретов {t['secrets']}"
        )
    if report.bot is not None:
        b = report.bot
        if b.get("won"):
            lines.append(
                f"бот: прошёл, {b['score']} очков = в уровне {b['score'] - b['flag_bonus'] - b['time_bonus']} "
                f"+ флаг {b['flag_bonus']} + время {b['time_bonus']} ({b['time_left']:.0f} с) "
                f"(монет {b['coins']}, блоков {b['blocks']}, врагов {b['enemies']})"
            )
        else:
            lines.append(
                f"бот: не прошёл — {b.get('death')}, прогресс {b['progress']:.0%}, {b['score']} очков"
            )
    for issue in report.issues:
        lines.append(f"  {issue}")
    return "\n".join(lines)


def format_table(reports: list[tuple[str, str, LevelReport]]) -> str:
    head = (f"{'id':<14}{'тайлов':>7}{'лимит':>7}{'тщат.':>6}{'замер':>6} {'split':<6}{'стр.':<5}{'путь':<6}"
            f"{'трубы':<7}{'потолок':>9}{'флаг':>6}  бот")
    lines = [head, "-" * len(head)]
    for level_id, split, r in reports:
        b = r.bot
        if b is None:
            bot = "—"
        elif b.get("won"):
            bot = f"прошёл {b['score']} (в уровне {b['score'] - b['flag_bonus'] - b['time_bonus']})"
        else:
            bot = f"{b.get('death')} {b['progress']:.0%}, {b['score']}"
        structure_ok = not any(i.level == "error" and i.code != "reach" for i in r.issues)
        pipes = f"{sum(w['ok'] for w in r.warps)}/{r.ceiling.secrets}"
        measured = [t["seconds"] for t in r.tours if t["ok"]]
        tour = f"{max(measured):.0f}" if measured else "—"
        lines.append(
            f"{level_id:<14}{r.width_tiles:>7}{(r.time_limit or 0):>7g}{r.thorough_seconds:>6.0f}{tour:>6} "
            f"{split:<6}{'OK' if structure_ok else 'ОШ':<5}{'OK' if r.safe_path else 'НЕТ':<6}"
            f"{pipes:<7}{r.ceiling.total:>9}{r.ceiling.flag:>6}  {bot}"
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
    parser.add_argument("--no-sim", action="store_true",
                        help="не гонять трубы в GameWorld (только граф)")
    parser.add_argument("--thorough", action="store_true",
                        help="замерить тщательный проход в GameWorld (долго: минуты на уровень)")
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
        report = check_level(path, name=name, time_limit=limit, bot=not args.no_bot,
                             sim=not args.no_sim, thorough=args.thorough)
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

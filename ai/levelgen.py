"""Процедурный генератор уровней для обучения нейросети.

    from ai.levelgen import generate_level
    level = generate_level(seed=3, difficulty=0.7)   # словарь уровня
    world = GameWorld(level)                          # сразу в мир

    .venv/bin/python -m ai.levelgen --seed 3 --difficulty 0.7 --ascii
    .venv/bin/python -m ai.levelgen --seed 3 --difficulty 0.7 --out maps/gen_3.json
    .venv/bin/python -m ai.levelgen --seed 3 --check        # проверить один уровень
    .venv/bin/python -m ai.levelgen --check 300 --jobs 12   # 300 уровней, сложность 0..1
    .venv/bin/python -m ai.levelgen --bench 50             # скорость генерации + GameWorld

Уровень собирается из «чанков» слева направо: ровный участок, яма, трубы,
загон между трубами, лестница (иногда с ямой посередине), ряды «?» с
кирпичами как в 1-1, одиночные «?», два яруса блоков, ряд «?» с врагами
наверху, группа врагов, спуск с лестницы на толпу врагов (цепочка прыжков),
парящие платформы над пропастью. В конце — лестница к флагу, флагшток и
замок как в 1-1.

Блок «?» твёрдый, как кирпич (см. docs/CONTRACT.md): удар головой снизу —
+100, пока в нём есть удары. Поэтому каждый «?» стоит над полом так, чтобы
его можно было выбить: низ блока на 3–4 тайла выше опоры, между ними
пусто. Над ямой и у её краёв «?» не бывает — там он обрывал бы прыжок.

Секретная комната (с вероятностью ~0.5, на лёгких уровнях ~0.3): обычная
на вид труба-вход, подземная комната-коробка с монетами, «?» и кирпичами
(на сложности > 0.4 — до двух гумб), устье боковой трубы справа и
труба-выход в 10–30 тайлах дальше по уровню. Комната — награда, а не
срезка: очков (монеты, удары «?», враги) между трубами не больше половины
очков комнаты, иначе вместо очередного чанка — ровная земля и выход.
Обычные трубы остаются частыми, чтобы вход не угадывался по «любая труба».

На основном уровне всегда есть что собрать: не меньше MIN_QUESTIONS «?»,
MIN_ENEMIES врагов и MIN_COINS монет — недостающее добирается чанками
перед финишем. Лимит времени — по ценам ручных уровней
(tools/level_tiles.thorough_seconds: путь, все удары «?», монеты, комната)
плюс секунды на врагов; не ниже политики ручных уровней (time_limit_for,
запас 30 %) и не ниже «лимит по пробегу + весь сбор сверху».

Детерминирован по (seed, difficulty) при неизменном tools/level_tiles.py
(сборщик, дорожки монет, финиш и лимит времени — оттуда): один и тот же
вход — байт в байт тот же уровень; --check печатает дайджест уровней и
хэш level_tiles, сверять надо оба. Проходим по построению: обязательные
прыжки не выше 4 тайлов и через ямы не шире 4 (физика игрока — 6.8 и ~6);
--check это проверяет графом полётов tools/level_check.py (с твёрдыми «?»)
и симуляцией GameWorld: от старта по осторожному пути графа к трубе-входу,
комната, выход и шаг с трубы-выхода.

difficulty 0 — почти ровно: редкие ямы в тайл, трубы в 2–3, одиночные
гумбы, много «?». difficulty 1 — ямы до 4, трубы до 4, группы врагов,
koopa, пропасти с цепочкой платформ, яма перед флагштоком.

Главное правило игры — побеждает тот, у кого больше очков, а не тот, кто
быстрее: генератор кладёт очки, за которыми надо остановиться, — «?» с
несколькими ударами, второй ярус, врагов на блоках (удар снизу), толпы
для цепочки прыжков, монеты по траектории прыжков и секретную комнату.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core import rules
from tools.level_tiles import (
    ENEMY_KINDS,
    GROUND_BOTTOM_TY,
    GROUND_TOP_TY,
    ROOM_STAND_TY,
    SOLID_KINDS,
    STAND_TY,
    TILE,
    LevelBuilder,
    RoomBuilder,
    block_hits,
    center,
    classify,
    thorough_seconds,
    tile_of,
    time_limit_for,
)

GENERATOR_VERSION = 3

# Длина уровня по умолчанию (тайлов от старта до конца замка)
DEFAULT_LENGTH_RANGE = (140, 180)

# Физика игрока (для справки и проверок): прыжок 6.8 тайла вверх, ~7.5 вдаль
SAFE_RISE = 4  # обязательный подъём — не выше
SAFE_PIT = 4  # обязательная яма — не шире

# Ряды блоков «?» над землёй: низ блока на 3 (LOW) или 4 (HIGH) тайла выше
# пола — голова стоящего игрока на 1 тайл выше пола, прыжок достаёт с запасом
LOW = STAND_TY + 3
HIGH = STAND_TY + 4
# Второй ярус — на 4 ряда выше первого: с верха первого ряда низ снова в 3 тайлах
TIER = 4
# Высота низа «?» над опорой, при которой блок честно выбивается (правило
# генератора; docs/CONTRACT.md: «низ блока на 3–4 тайла выше пола»)
BUMP_GAPS = (3, 4)
# Ручные карты проверяются мягче: блок выбивается, если над головой
# стоящего игрока есть хоть тайл и прыжок (6.8 тайла) до него достаёт
BUMP_REACH = (2, 6)
# «?» не ближе стольких столбцов к краю ямы: там он обрывал бы прыжок
PIT_CLEARANCE = 3

# Сколько очков на основном уровне минимум (без комнаты): «?», врагов,
# монет. Недостающее — добором перед финишем (_top_up)
MIN_QUESTIONS = 4
MIN_ENEMIES = 3
MIN_COINS = 10

# Секретная комната
WARP_CHANCE = 0.5
WARP_CHANCE_EASY = 0.3  # при difficulty < WARP_EASY_BELOW
WARP_EASY_BELOW = 0.15
WARP_DISTANCE = (10, 30)  # от трубы-входа до трубы-выхода, тайлов
# Очков между трубами (их пропускает ушедший в комнату) — не больше этой
# доли очков комнаты: комната всегда выгоднее того, что она перепрыгивает
WARP_SKIP_SHARE = 0.5
# К лимиту времени — секунд на врага (в thorough_seconds врагов нет)
ENEMY_SECONDS = 0.5
ROOM_WIDTH = (14, 22)
ROOM_HEIGHT = (8, 11)
ROOM_FLOOR = ROOM_STAND_TY  # ряд, где стоит игрок в комнате (пол — ряды 0–1)
ROOM_ENEMIES_FROM = 0.4  # гумбы в комнате — только сложнее этого


# ===============================================================
# ПАРАМЕТРЫ ПО СЛОЖНОСТИ
# ===============================================================


@dataclass(frozen=True)
class Knobs:
    """Во что превращается difficulty 0..1."""

    difficulty: float
    max_pit: int  # ширина ямы: 1..4
    max_tube: int  # высота трубы: 2..4
    max_stairs: int  # ступеней в лестнице: 2..5
    max_gap: int  # щель между парящими платформами: 2..4
    group: int  # врагов в группе: 1..4
    koopa: float  # доля koopa среди врагов
    risk: float  # вероятность «жадных» монет над ямой
    spacer: tuple[int, int]  # ровная земля между чанками

    @classmethod
    def of(cls, d: float) -> "Knobs":
        return cls(
            difficulty=d,
            max_pit=1 + _round(3 * d),
            max_tube=2 + _round(2 * d),
            max_stairs=2 + _round(3 * d),
            max_gap=2 + _round(2 * d),
            group=1 + _round(3 * d),
            koopa=0.1 + 0.3 * d,
            risk=0.2 + 0.5 * d,
            spacer=(3 - _round(d), 6 - _round(2 * d)),
        )


def _round(value: float) -> int:
    """Округление половин вверх (round() в Python округляет 0.5 к чётному)."""
    return int(value + 0.5)


def _points(cells: dict[tuple[int, int], str], tx0: int = 0, tx1: int = 1 << 30) -> int:
    """Очки за монеты, все удары по «?» и врагов (по 100, без цепочек)
    в столбцах tx0..tx1−1."""
    points = 0
    for (tx, ty), kind in cells.items():
        if not tx0 <= tx < tx1:
            continue
        if kind == "coin":
            points += rules.COIN_POINTS
        elif kind == "question":
            points += rules.BLOCK_POINTS * block_hits(center(tx), center(ty))
        elif kind in ENEMY_KINDS:
            points += rules.ENEMY_POINTS
    return points


def _enemy_kind(rng: random.Random, k: Knobs) -> str:
    return "koopa" if rng.random() < k.koopa else "goomba"


def _block_row(b: LevelBuilder, rng: random.Random, tx0: int, tx1: int, ty: int,
               share: float) -> list[int]:
    """Ряд кирпичей и «?» как в 1-1: share — доля «?», хотя бы один «?».

    Возвращает столбцы, где стоят «?».
    """
    cols = list(range(tx0, tx1 + 1))
    marks = [tx for tx in cols if rng.random() < share]
    if not marks:
        marks = [rng.choice(cols)]
    for tx in cols:
        if tx in marks:
            b.question(tx, ty)
        else:
            b.row(tx, tx, ty)
    return marks


# ===============================================================
# ЧАНКИ
# ===============================================================
# Каждый чанк начинается в столбце x на обычной земле, сам кладёт землю
# под свои столбцы и возвращает первый свободный столбец за собой. На
# обоих краях чанка — ровная земля, поэтому чанки стыкуются в любом порядке.
# Блоки «?» — только в чанках на ровной земле и не ближе 2 столбцов к краю
# чанка: с промежутком это ≥ 4 столбца до любой ямы.
#
# Враги ставятся с явным рядом: LevelBuilder.surface() считает кирпичный
# ряд над головой опорой и поставил бы гумбу «на полку» вместо земли.


def chunk_flat(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Ровный участок; иногда с дорожкой монет над землёй."""
    n = rng.randint(4, 8)
    b.ground(x, x + n - 1)
    if rng.random() < 0.5:
        row = rng.choice((STAND_TY + 1, STAND_TY + 3))
        b.coins(x + 1, x + n - 2, row)
    return x + n


def chunk_pit(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Яма шириной 1..max_pit с монетами над ней (без «?»: он оборвал бы прыжок)."""
    w = rng.randint(1, k.max_pit)
    b.ground(x, x + 1)
    b.ground(x + 2 + w, x + 3 + w)
    edge = x + 1

    roll = rng.random()
    if roll < 0.55:
        # Дорожка по траектории прыжка с края
        b.jump_trail(edge, STAND_TY)
    elif roll < 0.55 + k.risk * 0.6:
        # «Жадные» монеты: выше обычной дорожки, нужен точный прыжок
        b.coins(x + 2, x + 1 + w, STAND_TY + 5)
    return x + 4 + w


def chunk_tube(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Одна труба; монета над ней."""
    h = rng.randint(2, k.max_tube)
    b.ground(x, x + 3)
    b.tube(x + 1, h)
    if rng.random() < 0.5:
        b.coins(x + 1, x + 2, STAND_TY + h + 1)
    return x + 4


def chunk_pen(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Загон: враги между двумя трубами никуда не уходят."""
    inner = rng.randint(5, 8)
    h1 = rng.randint(2, k.max_tube)
    h2 = rng.randint(2, k.max_tube)
    b.ground(x, x + inner + 5)
    b.tube(x + 1, h1)
    b.tube(x + 3 + inner, h2)
    count = max(1, min(k.group, inner // 2))
    for i in range(count):
        b.enemy(_enemy_kind(rng, k), x + 4 + 2 * i, STAND_TY)
    if rng.random() < 0.6:
        b.coins(x + 4, x + 2 + inner, STAND_TY + 4)
    return x + inner + 6


def chunk_stairs(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Лестница вверх и вниз; на сложности — с ямой между ними."""
    steps = rng.randint(2, k.max_stairs)
    top = rng.randint(0, 2)
    with_pit = k.difficulty >= 0.4 and rng.random() < 0.5
    gap = rng.randint(1, min(3, k.max_pit)) if with_pit else 0

    width = 1 + steps + top + gap + steps + 1
    b.ground(x, x + width - 1)
    b.stairs_up(x + 1, steps)
    for i in range(top):
        b.column(x + 1 + steps + i, STAND_TY, STAND_TY + steps - 1)
    down = x + 1 + steps + top + gap
    if gap:
        b.pit(down - gap, down - 1)
    b.stairs_down(down, steps)

    peak = x + steps + top
    if rng.random() < 0.6:
        b.jump_trail(peak, STAND_TY + steps, land_ty=STAND_TY + steps)
    return x + width


def chunk_shelf(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Кирпичная полка над землёй: монеты на ней, «?» в ней и над ней."""
    w = rng.randint(3, 6)
    row = rng.choice((LOW, HIGH))
    n = w + 4
    b.ground(x, x + n - 1)
    # Часть кирпичей полки — «?»: выбиваются с земли
    _block_row(b, rng, x + 2, x + 1 + w, row, share=0.25)
    b.coins(x + 2, x + 1 + w, row + 1)
    if rng.random() < 0.7:
        # «?» над полкой: выбивается, стоя на ней
        b.question(x + 2 + rng.randint(1, w - 2), row + TIER)
    if rng.random() < 0.35 + 0.3 * k.difficulty:
        # Гумба на полке — растоптать или выбить ударом снизу
        b.enemy("goomba", x + 1 + w, row + 1)
    return x + n


def chunk_blocks(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Ряд «?» и кирпичей как в 1-1, иногда «?» над ним; враги под рядом."""
    w = rng.randint(3, 7)
    n = w + 4
    b.ground(x, x + n - 1)
    lo, hi = x + 2, x + 1 + w
    row = LOW if rng.random() < 0.75 else HIGH
    _block_row(b, rng, lo, hi, row, share=0.5)
    if rng.random() < 0.55:
        # Верхний «?» над серединой ряда: бьётся, стоя на кирпичах ряда
        b.question(rng.randint(lo + 1, hi - 1), row + TIER)
    if rng.random() < 0.3 + 0.6 * k.difficulty:
        count = rng.randint(1, min(k.group, 3))
        for i in range(count):
            b.enemy(_enemy_kind(rng, k), min(lo + 1 + 2 * i, x + n - 2), STAND_TY)
    return x + n


def chunk_single(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Одиночные «?» над ровной землёй; монета на блоке — за прыжок на него."""
    n = rng.randint(5, 9)
    b.ground(x, x + n - 1)
    col = x + 2 + rng.randint(0, 1)
    cols = [col]
    if n >= 8 and rng.random() < 0.6:
        cols.append(col + rng.randint(3, 4))
    for tx in cols:
        row = rng.choice((LOW, LOW, HIGH))
        b.question(tx, row)
        if rng.random() < 0.4:
            b.coin(tx, row + 1)
    if rng.random() < 0.4 * k.difficulty:
        b.enemy(_enemy_kind(rng, k), x + n - 2, STAND_TY)
    return x + n


def chunk_tiers(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Два яруса: нижний ряд кирпичей с «?», над его серединой — второй ряд.

    Нижний ряд выбивается с земли, второй — стоя на нижнем (с земли его
    не достать: мешает нижний ряд). Над вторым ярусом — монеты.
    """
    w = rng.randint(7, 10)
    n = w + 4
    b.ground(x, x + n - 1)
    lo, hi = x + 2, x + 1 + w
    _block_row(b, rng, lo, hi, LOW, share=0.3)
    # Второй ярус отступает от краёв нижнего на 2–3 столбца: с края
    # нижнего ряда есть разбег, чтобы запрыгнуть на верх второго
    u0 = lo + rng.randint(2, 3)
    u1 = hi - rng.randint(2, 3)
    _block_row(b, rng, u0, u1, LOW + TIER, share=0.6)
    if rng.random() < 0.6:
        b.coins(u0, u1, LOW + TIER + 1)
    if rng.random() < 0.3 + 0.4 * k.difficulty:
        # Враг на нижнем ярусе — под вторым, удар снизу его сбивает
        b.enemy(_enemy_kind(rng, k), hi, LOW + 1)
    return x + n


def chunk_patrol(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Ряд почти из одних «?» с врагами наверху: удар снизу сбивает врага.

    Иногда по краям ряда — упоры, и враги ходят по ряду, пока их не собьют.
    """
    w = rng.randint(4, 7)
    n = w + 4
    b.ground(x, x + n - 1)
    lo, hi = x + 2, x + 1 + w
    _block_row(b, rng, lo, hi, LOW, share=0.75)
    if rng.random() < 0.5:
        b.put("hard", lo, LOW + 1)
        b.put("hard", hi, LOW + 1)
        first, last = lo + 1, hi - 1
    else:
        first, last = lo, hi
    count = rng.randint(1, max(1, min(2, k.group)))
    for i in range(count):
        tx = last - 2 * i
        if tx >= first:
            b.enemy(_enemy_kind(rng, k), tx, LOW + 1)
    if rng.random() < 0.4:
        b.coins(lo, hi, STAND_TY + 1)
    return x + n


def chunk_enemies(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Группа врагов на ровном; монеты над ними — за прыжок сверху."""
    count = rng.randint(1, k.group)
    n = 4 + 2 * count + rng.randint(0, 3)
    b.ground(x, x + n - 1)
    for i in range(count):
        b.enemy(_enemy_kind(rng, k), x + 3 + 2 * i, STAND_TY)
    if rng.random() < 0.5:
        b.coins(x + 2, x + 2 + 2 * count, STAND_TY + 3)
    return x + n


def chunk_descent(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Площадка на лестнице и толпа врагов внизу — место для цепочки прыжков.

    Враги идут к лестнице и разворачиваются у неё; прыжок с площадки на
    толпу без приземления даёт 100, 200, 400, 500…
    """
    steps = rng.randint(3, 4)
    top = rng.randint(1, 2)
    run = rng.randint(7, 10)
    n = 1 + steps + top + run
    b.ground(x, x + n - 1)
    b.stairs_up(x + 1, steps)
    for i in range(top):
        b.column(x + 1 + steps + i, STAND_TY, STAND_TY + steps - 1)
    edge = x + steps + top  # последний столбец площадки
    count = rng.randint(2, max(2, min(k.group + 1, 4)))
    for i in range(count):
        tx = edge + 3 + 2 * i
        if tx <= x + n - 2:
            b.enemy(_enemy_kind(rng, k), tx, STAND_TY)
    if rng.random() < 0.5:
        b.jump_trail(edge, STAND_TY + steps, land_ty=STAND_TY + 1)
    return x + n


def chunk_sky(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Пропасть с цепочкой парящих кирпичных платформ."""
    span = rng.randint(7, 8 + _round(10 * k.difficulty))
    b.ground(x, x + 1)
    cursor = x + 2  # первый столбец пропасти
    stand = STAND_TY  # ряд, где стоит игрок на предыдущей опоре
    end = x + 2 + span

    while True:
        gap = rng.randint(2, k.max_gap)
        if cursor + gap >= end - 1:
            break
        width = rng.randint(2, 4 - _round(k.difficulty))
        # Ряд кирпичей: подъём к следующей опоре не больше 2, платформы
        # парят (кирпич не ниже ряда 6) и чаще уходят вверх, чем вниз
        row = max(STAND_TY, min(STAND_TY + 5, stand - 1 + rng.randint(-1, 2)))
        start = cursor + gap
        b.row(start, start + width - 1, row)
        if rng.random() < 0.5:
            # Дорожка с края предыдущей опоры
            b.jump_trail(cursor - 1, stand)
        elif rng.random() < k.risk:
            b.coins(start, start + width - 1, row + 4)
        cursor = start + width
        stand = row + 1

    # Посадка на землю: щель не шире max_gap
    land = max(cursor + 2, min(end, cursor + k.max_gap))
    b.ground(land, land + 1)
    return land + 2


# Чанк → вес при difficulty 0 и при 1 (между ними — линейно)
CHUNKS: dict[str, tuple[Callable, float, float]] = {
    "flat": (chunk_flat, 2.0, 0.4),
    "pit": (chunk_pit, 0.6, 2.6),
    "tube": (chunk_tube, 1.4, 1.6),
    "pen": (chunk_pen, 0.3, 1.0),
    "stairs": (chunk_stairs, 0.6, 1.0),
    "shelf": (chunk_shelf, 0.9, 0.7),
    "blocks": (chunk_blocks, 1.3, 1.0),
    "single": (chunk_single, 0.8, 0.5),
    "tiers": (chunk_tiers, 0.5, 0.8),
    "patrol": (chunk_patrol, 0.5, 1.0),
    "enemies": (chunk_enemies, 0.9, 1.5),
    "descent": (chunk_descent, 0.2, 0.8),
    "sky": (chunk_sky, 0.0, 1.1),
}


# Чанки с блоками «?». Если трёх подряд таких не было — следующий из них:
# очки за удары лежат по всему уровню, а не как повезёт со случаем
BLOCK_CHUNKS = ("blocks", "single", "tiers", "patrol", "shelf")
BLOCK_FALLBACK = ("blocks", "blocks", "single", "tiers", "patrol")


def _weights(d: float) -> list[float]:
    return [w0 + (w1 - w0) * d for _, w0, w1 in CHUNKS.values()]


def chunk_coins(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Ровный участок с дорожкой монет низко и горкой над ней (для добора)."""
    n = rng.randint(6, 9)
    b.ground(x, x + n - 1)
    b.coins(x + 1, x + n - 2, STAND_TY + 1)
    b.coins(x + 2, x + n - 3, STAND_TY + 3)
    return x + n


# Добор: чего не хватает → чанк, который это кладёт
TOP_UP: tuple[tuple[str, Callable], ...] = (
    ("question", chunk_blocks),
    ("enemy", chunk_enemies),
    ("coin", chunk_coins),
)


def _top_up(b: LevelBuilder, x: int, rng: random.Random, k: Knobs, used: list[str]) -> int:
    """Перед финишем добрать «?», врагов и монет до MIN_*: на уровне всегда
    есть что выбить, кого растоптать и что собрать (не только добежать)."""
    need = {"question": MIN_QUESTIONS, "enemy": MIN_ENEMIES, "coin": MIN_COINS}
    for _ in range(12):
        counts = Counter(b.cells.values())
        counts["enemy"] = sum(counts[kind] for kind in ENEMY_KINDS)
        short = next(((what, chunk) for what, chunk in TOP_UP if counts[what] < need[what]), None)
        if short is None:
            break
        x = short[1](b, x, rng, k)
        used.append(f"top_up_{short[0]}")
        gap = rng.randint(*k.spacer)
        b.ground(x, x + gap - 1)
        x += gap
    return x


# ===============================================================
# СЕКРЕТНАЯ КОМНАТА
# ===============================================================


def chunk_warp_in(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> tuple[int, tuple[int, int]]:
    """Труба-вход: обычная на вид, над ней пусто. Возвращает (x, левый верхний тайл)."""
    h = rng.randint(2, max(3, k.max_tube))
    b.ground(x, x + 5)
    b.tube(x + 2, h)
    return x + 6, (x + 2, STAND_TY + h - 1)


def chunk_warp_out(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> tuple[int, tuple[int, int]]:
    """Труба-выход: над ней пусто, за ней 4 тайла ровной земли (не в яму)."""
    h = rng.randint(2, 3)
    b.ground(x, x + 7)
    b.tube(x + 2, h)
    return x + 8, (x + 2, STAND_TY + h - 1)


def build_room(rng: random.Random, k: Knobs) -> tuple[dict, dict]:
    """Подземная бонус-комната: закрытая коробка, выход — боковая труба справа.

    Координаты свои (от 0): пол — ряды 0–1, игрок стоит в ряду ROOM_FLOOR,
    стены — столбцы 0 и w−1, потолок — ряд h−1 (RoomBuilder.box). Справа
    внутри — устье боковой трубы (низ устья = верх пола), за ним тело, стык
    и вертикальная труба до потолка (RoomBuilder.side_exit). Возвращает
    (комната для "warps", описание с очками "points" и временем "seconds").
    """

    w = rng.randint(*ROOM_WIDTH)
    h = rng.randint(*ROOM_HEIGHT)
    f = ROOM_FLOOR
    top = h - 2  # верхний свободный ряд
    mouth = w - 5  # столбец устья (RoomBuilder.side_exit)

    r = RoomBuilder("room").box(w, h)
    # Появление — у потолка слева: столбцы 1–2 ничем не заняты до пола
    r.start(2, top - 1)

    # Содержимое — между столбцом 3 и столбцом перед устьем (с зазором)
    c0, c1 = 3, mouth - 2

    # Ковёр монет у пола: нижний ряд собирается на ходу, второй — прыжком
    a = c0 + rng.randint(0, 1)
    z = c1 - rng.randint(0, 1)
    r.coins(a, z, f)
    if rng.random() < 0.7:
        r.coins(a, z, f + 1)

    # Ряд кирпичей и «?» на 3 тайла над полом — бьётся головой с пола.
    # Кончается за 4 столбца до устья: сошедший с ряда вправо падает на пол,
    # а не на боковую трубу
    q1 = mouth - 4
    span = rng.randint(3, min(7, q1 - c0 + 1))
    q0 = rng.randint(c0, q1 - span + 1)
    marks = _block_row(r, rng, q0, q0 + span - 1, f + 3, share=0.5)
    # Над рядом — монеты, если под потолком есть ряд, чтобы на ряд запрыгнуть
    if top >= f + 5:
        r.coins(q0, q0 + span - 1, f + 4)
    # В высокой комнате — ещё ряд монет, за прыжок с ряда блоков
    if top >= f + 7:
        r.coins(max(c0, q0 - 1), min(c1, q0 + span), f + 6)

    enemies = 0
    if k.difficulty > ROOM_ENEMIES_FROM:
        enemies = rng.choice((0, 1, 1, 2))
        for i in range(enemies):
            r.enemy("goomba", c1 - 3 * i, f)

    r.side_exit()
    return r.to_room(), {
        "size": [w, h], "blocks": len(marks), "enemies": enemies,
        "points": _points(r.cells), "seconds": round(r.seconds(), 1),
    }


# ===============================================================
# ТРЕНИРОВКИ
# ===============================================================

DRILL_WIDTH = (24, 36)
DRILL_HEIGHT = (9, 11)
# Сколько решений длится тренировка (15 в секунду): 20 с — и эпизод
# обрезается, а не кончается смертью от таймера
DRILL_STEPS = 300


def build_drill(seed: int, difficulty: float = 0.5) -> dict:
    """Тренировка для обучения: закрытая подземная арена без флага.

    Ряды кирпичей и «?» на 3 тайла над полом (бьются с пола), монеты на
    полу и над рядами, враги на полу. Флага нет, бежать некуда — очки
    дают только удары, монеты и враги. Зачем: дообучение под награду
    collect за 150 тыс. шагов не нашло «встать под блок и прыгать на
    месте» (ударов по «?» ~6 %), на уровне с флагом бег к флагу всегда
    ближе. Здесь этот навык — единственный путь к награде.
    """

    rng = random.Random(f"drill:{seed}")
    k = Knobs.of(difficulty)
    w = rng.randint(*DRILL_WIDTH)
    h = rng.randint(*DRILL_HEIGHT)
    f = ROOM_FLOOR

    r = RoomBuilder("drill").box(w, h)
    r.start(2, f)

    col = 4
    while col < w - 4:
        span = min(rng.randint(2, 6), w - 3 - col)
        if span < 1:
            break
        _block_row(r, rng, col, col + span - 1, f + 3, share=rng.uniform(0.5, 1.0))
        if h - 2 >= f + 5 and rng.random() < 0.5:
            r.coins(col, col + span - 1, f + 4)
        end = min(col + span + rng.randint(2, 5), w - 1)
        if end - 1 >= col + span and rng.random() < 0.6:
            r.coins(col + span, end - 1, f)
        col = end

    for _ in range(rng.randint(1, 2 + round(3 * difficulty))):
        r.enemy(_enemy_kind(rng, k), rng.randint(8, w - 3), f)

    return {
        "name": f"Тренировка {seed}",
        "time_limit": 600,
        "theme": "underground",
        "objects": r.objects(),
        "train_steps": DRILL_STEPS,
    }


# ===============================================================
# ФИНИШ
# ===============================================================


def _finish(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> tuple[int, dict]:
    """Лестница к флагу, флагшток и замок. Возвращает (шток, описание)."""

    roll = rng.random()
    if roll < 0.08:
        # Ровный финиш: без лестницы флаг берётся не выше 0.45 (400 очков)
        b.ground(x, x + 3)
        pole = x + 4
        b.finish(pole)
        return pole, {"stairs": 0}

    # Высота лестницы: 7–9 дают 5000 с разбега, 5–6 — не больше 2000
    height = rng.choice((5, 6)) if roll < 0.2 else rng.choice((7, 8, 8, 9))
    top = rng.randint(1, 2)
    # Сколько ровной земли от лестницы до штока (окно захвата за верх)
    run = rng.randint(4, 5) if height == 7 else rng.randint(3, 6)
    # Яма шириной run−1 ≤ 4: с башни до штока не дальше 6 столбцов
    pit = k.difficulty >= 0.6 and run in (4, 5) and rng.random() < 0.3

    width = height + top - 1 + run
    b.ground(x, x + width)
    pole = b.final_stairs(x, height=height, top_width=top, run=run)
    if pit:
        # Яма между башней и штоком: вниз пути нет, только прыжок к флагу
        b.pit(pole - run, pole - 2)
    if rng.random() < 0.5:
        b.jump_trail(pole - run - 1, until_col=pole - 2)
    return pole, {"stairs": height, "top": top, "run": run, "pit": pit}


# ===============================================================
# ГЕНЕРАТОР
# ===============================================================


def generate_level(seed: int, difficulty: float = 0.5, *,
                   length_tiles: Optional[int] = None) -> dict:
    """Уровень в формате редактора (с name, description, time_limit и warps).

    seed — любое целое; difficulty — 0..1 (обрезается); length_tiles —
    длина от старта до конца замка, по умолчанию 140–180 по seed.
    """

    d = min(max(float(difficulty), 0.0), 1.0)
    d = round(d, 4)
    key = f"mario-levelgen:v{GENERATOR_VERSION}:{int(seed)}:{d:.4f}"
    rng = random.Random(key)
    # Решения про комнату — своим потоком: основной уровень от них не «съезжает»
    wrng = random.Random(key + ":warp")
    k = Knobs.of(d)

    length = int(length_tiles) if length_tiles else rng.randint(*DEFAULT_LENGTH_RANGE)
    length = max(length, 60)

    b = LevelBuilder(
        f"Генерация {seed} · сложность {d:.2f}",
        "",
    )

    # --- Старт: ровная площадка
    b.ground(0, 9)
    b.start(3, STAND_TY)
    x = 10

    # --- Середина: чанки, пока остаётся место под финиш (~30 тайлов)
    finish_room = 30
    x_end = length - finish_room

    # Секретная комната: где поставить вход и как далеко выход. Комната
    # строится сразу — её очки задают, сколько можно положить между трубами
    chance = WARP_CHANCE_EASY if d < WARP_EASY_BELOW else WARP_CHANCE
    warp_on = wrng.random() < chance and x_end >= 16 + WARP_DISTANCE[0] + 10
    entry_at = wrng.randint(16, max(16, x_end - 30)) if warp_on else 0
    exit_after = wrng.randint(*WARP_DISTANCE) if warp_on else 0
    room, room_info = build_room(wrng, k) if warp_on else (None, {})
    budget = WARP_SKIP_SHARE * room_info.get("points", 0)
    pipe: Optional[tuple[int, int]] = None
    exit_pipe: Optional[tuple[int, int]] = None

    def place_exit(x: int) -> int:
        """Ровная земля до минимального расстояния и труба-выход."""
        nonlocal exit_pipe
        # Труба-выход встаёт на x + 2 (chunk_warp_out)
        need = pipe[0] + WARP_DISTANCE[0] - 2
        if x < need:
            b.ground(x, need - 1)
            x = need
        x, exit_pipe = chunk_warp_out(b, x, wrng, k)
        used.append("warp_out")
        return x

    names = list(CHUNKS)
    weights = _weights(d)
    used: list[str] = []
    while x < x_end:
        between = pipe is not None and exit_pipe is None
        if warp_on and pipe is None and x >= entry_at:
            x, pipe = chunk_warp_in(b, x, wrng, k)
            used.append("warp_in")
        elif between and x + 2 - pipe[0] >= exit_after:
            x = place_exit(x)
        else:
            name = rng.choices(names, weights)[0]
            # Две пропасти подряд — перебор даже для d = 1
            if name == "sky" and used and used[-1] == "sky":
                name = "flat"
            if name not in BLOCK_CHUNKS and len(used) >= 3 and \
                    not any(u in BLOCK_CHUNKS for u in used[-3:]):
                name = rng.choice(BLOCK_FALLBACK)
            if between:
                # Между трубами — пробный чанк: если труба-выход уедет дальше
                # WARP_DISTANCE или пропущенных очков станет больше бюджета,
                # чанк откатывается, и сразу ставится выход. Очки — всё правее
                # трубы-входа: дорожка монет может свисать за конец чанка
                saved = dict(b.cells), rng.getstate()
                nxt = CHUNKS[name][0](b, x, rng, k)
                if (nxt + k.spacer[1] + 2 - pipe[0] > WARP_DISTANCE[1]
                        or _points(b.cells, pipe[0] + 2) > budget):
                    b.cells = saved[0]
                    rng.setstate(saved[1])
                    x = place_exit(x)
                else:
                    x = nxt
                    used.append(name)
            else:
                x = CHUNKS[name][0](b, x, rng, k)
                used.append(name)

        gap = rng.randint(*k.spacer)
        b.ground(x, x + gap - 1)
        x += gap

    if pipe and exit_pipe is None:
        # Вход встал поздно: выход сразу за ним, на минимальном расстоянии
        x = place_exit(x)
        b.ground(x, x + 2)
        x += 3

    # --- Добор очков и финиш
    x = _top_up(b, x, rng, k, used)
    pole, finish = _finish(b, x, rng, k)

    # Лимит: тщательное прохождение по ценам ручных уровней (thorough_seconds:
    # путь, 1.2 с на удар «?», 0.2 с на монету, комната) и ENEMY_SECONDS на
    # врага. Не меньше политики ручных уровней (time_limit_for(run, thorough):
    # запас 30 %) и не меньше «лимит по пробегу + весь сбор сверху»: 80 %
    # эпизодов обучения — генерация, и тот, кто выбивает всё, не должен
    # упираться в таймер
    run = b.run_tiles()
    room_on = pipe is not None and exit_pipe is not None
    enemies = sum(kind in ENEMY_KINDS for kind in b.cells.values()) + (room_info["enemies"] if room_on else 0)
    thorough = thorough_seconds(run, b.block_hits_total(), b.coins_total(),
                                (room_info["seconds"] if room_on else 0.0) + ENEMY_SECONDS * enemies)
    collect = thorough - thorough_seconds(run)
    b.time_limit = max(time_limit_for(run, thorough), time_limit_for(run) + int(5 * math.ceil(collect / 5)))
    level = b.to_level()
    counts = {name: used.count(name) for name in dict.fromkeys(used)
              if not name.startswith("warp")}
    level["description"] = (
        f"Процедурный уровень: seed {seed}, сложность {d:.2f}. "
        + ", ".join(f"{name}×{n}" for name, n in counts.items())
    )
    meta = {
        "version": GENERATOR_VERSION,
        "seed": int(seed),
        "difficulty": d,
        "length_tiles": b.width_tiles,
        "chunks": used,
        "finish": finish,
        "warp": None,
    }

    if room_on:
        level["warps"] = [{
            "id": "bonus",
            "pipe": [center(pipe[0]), center(pipe[1])],
            "exit": [center(exit_pipe[0]), center(exit_pipe[1])],
            "room": room,
        }]
        level["description"] += ". Секретная комната"
        meta["warp"] = {
            "pipe": list(pipe),
            "exit": list(exit_pipe),
            "distance": exit_pipe[0] - pipe[0],
            # Очки между трубами: их пропускает ушедший в комнату
            "skipped_points": _points(b.cells, pipe[0] + 2, exit_pipe[0]),
            **room_info,
        }

    meta["thorough_seconds"] = round(thorough, 1)
    level["generator"] = meta
    return level


# ===============================================================
# ASCII
# ===============================================================

ASCII_CHARS = {
    "ground": "#", "brick": "#", "hard": "#", "stone": "#", "block": "#",
    "tube_top_l": "T", "tube_top_r": "T", "tube_l": "T", "tube_r": "T",
    "question": "?", "coin": "o", "flag": "F", "stick": "|", "ball": "*",
    "mario": "M", "goomba": "g", "koopa": "k",
}

ASCII_LEGEND = (
    "# блок  T труба  W труба-вход  E труба-выход  < устье выхода из комнаты  "
    "? блок «?»  o монета  g goomba  k koopa  M старт  | шток  F флаг  C замок"
)


def _ascii_cells(objects: list[dict]) -> dict[tuple[int, int], str]:
    chars: dict[tuple[int, int], str] = {}
    for item in objects:
        if "x" not in item or "y" not in item:
            continue
        kind = classify(item)
        if kind.startswith("castle"):
            char = "C"
        elif str(item.get("resource", "")).lower() == "tubes":
            char = "T"
        else:
            char = ASCII_CHARS.get(kind, ",")
        chars[(tile_of(float(item["x"])), tile_of(float(item["y"])))] = char
    return chars


def _ascii_block(chars: dict[tuple[int, int], str], title: str, width: int) -> list[str]:
    max_tx = max(tx for tx, _ in chars)
    max_ty = max(ty for _, ty in chars)
    lines = [title]
    for start in range(0, max_tx + 1, width):
        stop = min(start + width, max_tx + 1)
        ruler = [" "] * (stop - start)
        for tx in range(start, stop):
            if tx % 10 == 0:
                for i, ch in enumerate(str(tx)):
                    if tx - start + i < len(ruler):
                        ruler[tx - start + i] = ch
        lines.append("")
        lines.append("    " + "".join(ruler))
        for ty in range(max_ty, -1, -1):
            lines.append(f"{ty:>3} " + "".join(chars.get((tx, ty), ".") for tx in range(start, stop)))
    return lines


def render_ascii(level: dict, width: int = 120) -> str:
    """Уровень символами; трубы-входы — W, трубы-выходы — E, комнаты — ниже."""

    chars = _ascii_cells(level["objects"])
    rooms = []
    for index, warp in enumerate(level.get("warps") or []):
        for mark, point in (("W", warp["pipe"]), ("E", warp["exit"])):
            tx, ty = tile_of(float(point[0])), tile_of(float(point[1]))
            chars[(tx, ty)] = chars[(tx + 1, ty)] = mark
        room = warp["room"]
        room_chars = _ascii_cells(room["objects"])
        if room.get("exit") is not None:
            mx, my = tile_of(float(room["exit"][0])), tile_of(float(room["exit"][1]))
            room_chars[(mx, my)] = room_chars[(mx, my + 1)] = "<"
        key = warp.get("id") or f"room{index + 1}"
        rooms.append(_ascii_block(
            room_chars, f"Комната «{key}» ({room.get('theme', 'underground')}): "
            f"вход W на столбце {tile_of(float(warp['pipe'][0]))}, "
            f"выход E на столбце {tile_of(float(warp['exit'][0]))}", width))

    lines = _ascii_block(chars, f"{level.get('name', 'уровень')}: лимит {level.get('time_limit')} с", width)
    for room_lines in rooms:
        lines.append("")
        lines += room_lines
    lines.append("")
    lines.append(ASCII_LEGEND)
    return "\n".join(lines)


# ===============================================================
# ПРОВЕРКА
# ===============================================================

SOLID = set(SOLID_KINDS) | {"question"}


def _cells(objects: list[dict], errors: list[str], where: str) -> dict[tuple[int, int], str]:
    """Клетка → вид; ошибки сетки и двойных клеток пишет в errors."""
    cells: dict[tuple[int, int], str] = {}
    doubles = []
    for item in objects:
        x, y = float(item["x"]), float(item["y"])
        if (x - TILE / 2) % TILE or (y - TILE / 2) % TILE:
            errors.append(f"{where}: объект не на сетке ({x:g}, {y:g})")
        key = (tile_of(x), tile_of(y))
        if key in cells:
            doubles.append(key)
        cells[key] = classify(item)
    if doubles:
        errors.append(f"{where}: {len(doubles)} клеток с двумя объектами, например {doubles[:3]}")
    return cells


def _bump_gap(solid: set, tx: int, ty: int) -> Optional[int]:
    """На сколько тайлов низ блока выше ближайшей опоры под ним (None — пропасть)."""
    for t in range(ty - 1, -1, -1):
        if (tx, t) in solid:
            return ty - (t + 1)
    return None


def _check_blocks(cells: dict, solid: set, errors: list[str], where: str,
                  gaps: tuple[int, int] = BUMP_GAPS) -> list[tuple[int, int, int]]:
    """«?» выбивается: опора под ним на gaps[0]–gaps[1] тайла ниже, между
    ними пусто (генератор — 3–4, ручные карты — BUMP_REACH).

    Возвращает (tx, ty, ряд, где стоит бьющий игрок).
    """
    out = []
    for (tx, ty), kind in cells.items():
        if kind != "question":
            continue
        gap = _bump_gap(solid, tx, ty)
        if gap is None:
            errors.append(f"{where}: «?» ({tx},{ty}) над пропастью")
        elif not gaps[0] <= gap <= gaps[1]:
            errors.append(f"{where}: «?» ({tx},{ty}): низ на {gap} тайлах над опорой, "
                          f"нужно {gaps[0]}–{gaps[1]}")
        else:
            out.append((tx, ty, ty - gap))
    return out


def _check_enemies(cells: dict, solid: set, errors: list[str], where: str) -> None:
    for (tx, ty), kind in cells.items():
        if kind in ("goomba", "koopa") and (tx, ty - 1) not in solid:
            errors.append(f"{where}: {kind} ({tx},{ty}) не стоит на блоке")


def _check_room(room: dict, errors: list[str], gaps: tuple[int, int] = BUMP_GAPS) -> None:
    """Комната — закрытая коробка, старт внутри, устье на уровне пола."""
    cells = _cells(room["objects"], errors, "комната")
    solid = {key for key, kind in cells.items() if kind in SOLID}
    starts = [key for key, kind in cells.items() if kind == "mario"]
    if len(starts) != 1:
        errors.append(f"комната: точек появления {len(starts)}, нужна одна")
        return
    x0 = min(tx for tx, _ in solid) - 1
    x1 = max(tx for tx, _ in solid) + 1
    y0 = min(ty for _, ty in solid) - 1
    y1 = max(ty for _, ty in solid) + 1

    # Заливка от точки появления: до рамки вокруг блоков дойти нельзя
    seen = {starts[0]}
    stack = [starts[0]]
    while stack:
        tx, ty = stack.pop()
        if tx in (x0, x1) or ty in (y0, y1):
            errors.append(f"комната не закрыта: из точки появления видно ({tx},{ty})")
            break
        for nxt in ((tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)):
            if nxt not in seen and nxt not in solid:
                seen.add(nxt)
                stack.append(nxt)

    exit_point = room.get("exit")
    if exit_point is None:
        errors.append("комната: нет exit (устья боковой трубы)")
        return
    mx, my = tile_of(float(exit_point[0])), tile_of(float(exit_point[1]))
    textures = {
        (tile_of(float(o["x"])), tile_of(float(o["y"]))): Path(o["texture_path"]).name
        for o in room["objects"] if o.get("resource") == "Tubes"
    }
    if textures.get((mx, my)) != "img_10.png" or textures.get((mx, my + 1)) != "img_11.png":
        errors.append(f"комната: в ({mx},{my}) нет устья img_10/img_11")
    if (mx, my - 1) not in solid:
        errors.append("комната: под устьем нет пола — низ устья не на уровне пола")
    if (mx - 1, my) in solid or (mx - 1, my - 1) not in solid:
        errors.append("комната: перед устьем нельзя стоять на полу")
    if (mx - 1, my) not in seen:
        errors.append("комната: от точки появления не дойти до устья")
    _check_blocks(cells, solid, errors, "комната", gaps)
    _check_enemies(cells, solid, errors, "комната")


def _check_reach(level: dict, cells: dict, solid: set, bumps: list, errors: list[str],
                 notes: list[str]) -> Optional[dict]:
    """Граф полётов tools/level_check.py, но с твёрдыми «?».

    Флаг достижим осторожным путём (подъёмы ≤ 4, ямы ≤ 4), под каждым «?»
    можно встать и потом дойти до флага, верх трубы-входа достижим, с трубы-
    выхода можно дойти до флага. В итоге под ключом "_graph" — (модуль,
    раскладка, граф) для похода от старта к трубе в _simulate_warp.
    """
    try:
        from tools import level_check as lc
    except Exception as error:  # файл правит другая часть: без графа уровень не проверен
        errors.append(f"граф не построен: tools.level_check не импортируется ({error!r})")
        return None

    main = {key: value for key, value in level.items() if key != "warps"}
    report = lc.LevelReport(name=main.get("name", ""), source="<levelgen>")
    layout = lc.check_structure(main, report)
    for issue in report.issues:
        # Врагов на «?» старая проверка не знает: их проверили выше сами
        if issue.level == "error" and issue.code != "enemy":
            errors.append(f"структура: {issue.code}: {issue.message}")
    layout.solid = set(layout.solid) | {key for key, kind in cells.items() if kind == "question"}
    reach = lc.build_graph(layout)

    if reach.start is None:
        errors.append("граф: старт не на опоре")
        return None
    if not reach.safe_goal:
        far = max((n[0] for n in reach.safe_forward), default=0)
        errors.append(f"граф: нет осторожного пути к флагу с твёрдыми «?» (дальше столбца {far} не пройти)")
    useful = reach.forward & reach.backward

    lost = [(tx, ty) for tx, ty, stand in bumps if (tx, stand) not in useful]
    if lost:
        errors.append(f"граф: под {len(lost)} «?» не встать (или оттуда не дойти до флага): {lost[:4]}")

    for warp in level.get("warps") or []:
        px, py = tile_of(float(warp["pipe"][0])), tile_of(float(warp["pipe"][1]))
        ex, ey = tile_of(float(warp["exit"][0])), tile_of(float(warp["exit"][1]))
        if (px, py + 1) not in useful and (px + 1, py + 1) not in useful:
            errors.append(f"граф: на трубу-вход ({px},{py}) не встать")
        if (ex, ey + 1) not in reach.backward and (ex + 1, ey + 1) not in reach.backward:
            errors.append(f"граф: с трубы-выхода ({ex},{ey}) не дойти до флага")

    return {
        "coins_reach": sum(1 for key in layout.coins if key in reach.coins or key in useful),
        "coins_main": len(layout.coins),
        "_graph": (lc, layout, reach),
    }


def _pits(solid: set, width: int) -> list[int]:
    """Столбцы без земли (ряды 4–5) — ямы и пропасти."""
    return [tx for tx in range(width)
            if not any((tx, ty) in solid for ty in range(GROUND_BOTTOM_TY, GROUND_TOP_TY + 1))]


def _without_main_enemies(level: dict) -> dict:
    """Копия уровня без врагов основного уровня (комнаты — как есть)."""
    quiet = dict(level)
    quiet["objects"] = [o for o in level["objects"]
                        if str(o.get("resource", "")).lower() not in ENEMY_KINDS]
    return quiet


def _simulate_warp(level: dict, graph: Optional[tuple] = None,
                   index: int = 0) -> tuple[bool, str, dict]:
    """GameWorld: от старта к трубе-входу, запрыгнуть, «вниз», пройти комнату
    вправо до устья, вылезти из трубы-выхода и сойти с неё.

    index — какая труба из world.warps. С графом (graph от _check_reach)
    игрок идёт от старта по осторожному пути графа нажатиями
    (tools.level_check.ScriptedWalker), враги основного уровня убраны — как
    в level_check.simulate_warps: проверяется путь, а не бой. Без графа —
    ставится в 2.5 тайла левее трубы (в info["teleport"]). Враги комнаты
    остаются: при враге впереди — прыжок. Смерть — провал.
    """
    from core.world import DEAD, PLAYING, GameWorld, Inputs

    world = GameWorld(_without_main_enemies(level) if graph else level)
    warp = world.warps[index]
    player = world.player
    info = {"frames": 0, "teleport": graph is None}

    def step(inputs) -> None:
        world.step(inputs)
        info["frames"] += 1

    if graph:
        # 0. От старта к трубе-входу по осторожному пути графа
        lc, layout, reach = graph
        px, py = tile_of(warp.x - TILE / 2), tile_of(warp.top - TILE / 2)
        node = next((n for n in ((px, py + 1), (px + 1, py + 1)) if n in reach.forward), None)
        walker = lc.ScriptedWalker(world, lc.TilePhysics(layout))
        here, problem = reach.start, None
        # Модель полётов графа и arcade иногда расходятся на тайл-другой
        # (приземлился на соседний уступ): тогда путь строится заново оттуда
        for _ in range(4):
            path = lc.find_path(reach, here, node) if node else None
            if path is None:
                problem = f"граф: нет пути от {here} к трубе-входу ({px},{py})"
                break
            problem = walker.follow(path)
            landed = walker.node()
            if problem is None or world.phase != PLAYING or landed is None or landed == here:
                break
            info["replans"] = info.get("replans", 0) + 1
            here = landed
        info["frames"] = world.frame
        if problem:
            return False, f"к трубе-входу: {problem}", info
    else:
        # Два тайла левее трубы-входа, на земле
        player.center_x = warp.x - 2.5 * TILE
        player.bottom = STAND_TY * TILE
        player.change_x = player.change_y = 0

    # 1. Запрыгнуть на трубу и нажать «вниз»
    for _ in range(600):
        if world.in_room or world.phase != PLAYING:
            break
        if world.on_ground and world.warp_under_player() is warp:
            step(Inputs(down=True))
            continue
        dx = warp.x - player.center_x
        jump = world.on_ground and player.bottom < warp.top - 4 and abs(dx) < 3 * TILE
        step(Inputs(right=dx > 3, left=dx < -3, jump=jump))
    for _ in range(400):
        if world.in_room and world.phase == PLAYING:
            break
        step(Inputs())
    if not world.in_room:
        return False, f"не ушёл в трубу (фаза {world.phase}, x={player.center_x:.0f})", info

    # 2. Комната: вправо до устья. Запрыгнул на боковую трубу — влево и вниз
    mouth = world.area.exit
    drop = False
    for _ in range(1800):
        if not world.in_room or world.phase == DEAD:
            break
        high = mouth is not None and player.bottom > mouth.bottom + 8
        if high and (drop or player.right > mouth.left + 1):
            drop = True
            step(Inputs(left=True))
            continue
        drop = False
        ahead = any(
            enemy.is_alive and 0 < enemy.center_x - player.center_x < 3 * TILE
            and abs(enemy.center_y - player.center_y) < TILE
            for enemy in world.enemies
        )
        near_mouth = mouth is not None and player.right > mouth.left - 2 * TILE
        step(Inputs(right=True, jump=world.on_ground and ahead and not near_mouth))
    if world.in_room:
        return False, f"не вышел из комнаты (фаза {world.phase}, x={player.center_x:.0f})", info

    # 3. Вылез из трубы-выхода
    for _ in range(400):
        if world.phase == PLAYING:
            break
        step(Inputs())
    if world.phase != PLAYING or world.area is not world.main:
        return False, f"не вылез из трубы (фаза {world.phase})", info
    if abs(player.center_x - warp.exit_x) > 2 or abs(player.bottom - warp.exit_top) > 3:
        return False, (f"вылез не там: ({player.center_x:.0f},{player.bottom:.0f}), "
                       f"труба ({warp.exit_x:.0f},{warp.exit_top:.0f})"), info

    # 4. Сойти с трубы вправо
    x0 = player.center_x
    for _ in range(45):
        step(Inputs(right=True))
    if world.phase == DEAD or player.center_x < x0 + 2 * TILE:
        return False, f"не сошёл с трубы-выхода (x {x0:.0f} → {player.center_x:.0f}, фаза {world.phase})", info

    info["secrets"] = world.stats["secret"]
    return True, "", info


def validate_level(level: dict, *, simulate: bool = True, reach: bool = True,
                   generated: Optional[bool] = None) -> dict:
    """Проверить уровень (сгенерированный или любой с "warps"). Итог — словарь.

    Ошибки (result["errors"]): «?» не выбивается (опора не в 3–4 тайлах под
    ним или под ним не встать), враг не на опоре, нет осторожного пути к
    флагу при твёрдых «?», трубы-входа/выхода нет или над ними занято,
    комната не закрыта или устье не на уровне пола, граф не построен, поход
    от старта в комнату и обратно в GameWorld не удался.

    generated (по умолчанию — есть ли ключ "generator") включает правила
    самого генератора: «?» ровно в 3–4 тайлах над опорой и не у ямы, выход
    в WARP_DISTANCE от входа и без ямы за ним, очков между трубами не больше
    WARP_SKIP_SHARE очков комнаты. Ручные карты (maps/*.json) их не обязаны
    соблюдать: для них «?» — в BUMP_REACH над опорой, над трубами 2 тайла
    свободно; блоки на пути прыжков ловит граф полётов.
    """

    if generated is None:
        generated = "generator" in level
    gaps = BUMP_GAPS if generated else BUMP_REACH
    errors: list[str] = []
    notes: list[str] = []
    cells = _cells(level["objects"], errors, "уровень")
    solid = {key for key, kind in cells.items() if kind in SOLID}
    meta = level.get("generator") or {}

    bumps = _check_blocks(cells, solid, errors, "уровень", gaps)
    _check_enemies(cells, solid, errors, "уровень")

    pole = next((tx for (tx, ty), kind in cells.items() if kind == "stick"),
                max(tx for tx, _ in cells) + 1)
    pits = set(_pits(solid, pole))
    if generated:
        # «?» не у ямы: иначе обрывает прыжок через неё
        for tx, ty, _ in bumps:
            near = [p for p in range(tx - PIT_CLEARANCE, tx + PIT_CLEARANCE + 1) if p in pits]
            if near:
                errors.append(f"уровень: «?» ({tx},{ty}) в {abs(near[0] - tx)} столбцах от ямы")

    warps = level.get("warps") or []
    for warp in warps:
        px, py = tile_of(float(warp["pipe"][0])), tile_of(float(warp["pipe"][1]))
        ex, ey = tile_of(float(warp["exit"][0])), tile_of(float(warp["exit"][1]))
        for name, (tx, ty), free in (("вход", (px, py), 3 if generated else 2), ("выход", (ex, ey), 2)):
            if cells.get((tx, ty)) != "tube_top_l" or cells.get((tx + 1, ty)) != "tube_top_r":
                errors.append(f"труба-{name}: нет верха трубы в ({tx},{ty})")
            busy = [(tx + i, ty + j) for i in (0, 1) for j in range(1, free + 1)
                    if cells.get((tx + i, ty + j)) in SOLID]
            if busy:
                errors.append(f"труба-{name}: над ней занято {busy}")
        _check_room(warp["room"], errors, gaps)
        if not generated:
            continue
        if any(p in pits for p in range(ex + 2, ex + 6)):
            errors.append(f"труба-выход ({ex},{ey}): за ней яма")
        if not WARP_DISTANCE[0] <= ex - px <= WARP_DISTANCE[1]:
            errors.append(f"труба-выход в {ex - px} тайлах от входа, нужно {WARP_DISTANCE}")
        # Комната — награда, а не срезка: пропущенное между трубами дешевле
        skipped = _points(cells, px + 2, ex)
        room_points = _points({(tile_of(float(o["x"])), tile_of(float(o["y"]))): classify(o)
                               for o in warp["room"]["objects"]})
        if skipped > WARP_SKIP_SHARE * room_points:
            errors.append(f"между трубами {skipped} очков, в комнате {room_points}: "
                          f"комната срезает больше, чем даёт")

    result = {
        "warp": bool(warps), "errors": errors, "notes": notes,
        "length": meta.get("length_tiles"), "time_limit": level.get("time_limit"),
    }
    kinds = Counter(cells.values())
    result["main"] = {"questions": kinds["question"], "coins": kinds["coin"],
                      "enemies": sum(kinds[kind] for kind in ENEMY_KINDS)}
    if generated:
        short = [f"{what} {have} < {need}" for what, have, need in (
            ("«?»", kinds["question"], MIN_QUESTIONS),
            ("врагов", result["main"]["enemies"], MIN_ENEMIES),
            ("монет", kinds["coin"], MIN_COINS)) if have < need]
        if short:
            errors.append(f"уровень: мало очков: {', '.join(short)}")
    else:
        notes.append("ручная карта: правила генератора (3–4 тайла под «?», ямы, "
                     "расстояние и очки между трубами, минимум очков) не применялись")
    if meta.get("warp"):
        result["warp_distance"] = meta["warp"]["distance"]
        result["room_size"] = meta["warp"]["size"]
        result["room_points"] = meta["warp"].get("points", 0)
        result["skipped_points"] = meta["warp"].get("skipped_points", 0)

    graph = None
    if reach:
        t0 = time.process_time()
        try:
            extra = _check_reach(level, cells, solid, bumps, errors, notes)
        except Exception as error:  # чужой код посреди правки: без графа уровень не проверен
            errors.append(f"граф упал: {error!r}")
            extra = None
        result["reach_ms"] = (time.process_time() - t0) * 1000
        if extra:
            graph = extra.pop("_graph", None)
            result.update(extra)

    if simulate:
        from core.world import GameWorld

        t0 = time.process_time()
        frames = 0
        for index in range(len(warps)):
            ok, message, info = _simulate_warp(level, graph, index)
            if not ok:
                errors.append(f"симуляция комнаты {index + 1}: {message}")
            frames += info["frames"]
            result["sim_replans"] = result.get("sim_replans", 0) + info.get("replans", 0)
            if info["teleport"] and index == 0:
                notes.append("подход к трубе-входу не проверен (нет графа): игрок поставлен у трубы")
        if warps:
            result["sim_frames"] = frames
        result["totals"] = dict(GameWorld(level).totals)
        result["sim_ms"] = (time.process_time() - t0) * 1000

    # Сколько очков лежит в комнате отдельно
    if warps:
        room_cells = [classify(o) for o in warps[0]["room"]["objects"]]
        result["room_coins"] = room_cells.count("coin")
        result["room_blocks"] = room_cells.count("question")
        result["room_enemies"] = room_cells.count("goomba") + room_cells.count("koopa")

    result["ok"] = not errors
    return result


def check_generated(seed: int, difficulty: float, *, simulate: bool = True,
                    reach: bool = True) -> dict:
    """Сгенерировать уровень и проверить его (validate_level) — для сводки."""

    c0 = time.process_time()
    level = generate_level(seed, difficulty)
    gen_ms = (time.process_time() - c0) * 1000
    result = validate_level(level, simulate=simulate, reach=reach)
    text = json.dumps(level, ensure_ascii=False, sort_keys=True)
    if json.dumps(generate_level(seed, difficulty), ensure_ascii=False, sort_keys=True) != text:
        result["errors"].append("недетерминирован: второй вызов дал другой уровень")
        result["ok"] = False
    result.update(seed=seed, difficulty=difficulty, gen_ms=gen_ms,
                  digest=hashlib.sha256(text.encode("utf-8")).hexdigest())
    return result


def _check_one(job: tuple[int, float, bool, bool]) -> dict:
    seed, difficulty, simulate, reach = job
    try:
        return check_generated(seed, difficulty, simulate=simulate, reach=reach)
    except Exception as error:
        import traceback
        return {"seed": seed, "difficulty": difficulty, "ok": False, "warp": False,
                "errors": [f"исключение: {error!r}", traceback.format_exc(limit=4)],
                "notes": [], "gen_ms": 0.0}


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(int(len(values) * q), len(values) - 1)] if values else 0.0


def check_many(count: int, first_seed: int = 0, difficulty: Optional[float] = None, *,
               jobs: int = 1, simulate: bool = True, reach: bool = True) -> int:
    """--check N: N уровней, сложность по кругу 0.0, 0.1 … 1.0 (или заданная)."""

    tasks = [
        (seed, difficulty if difficulty is not None else round((i % 11) / 10, 1), simulate, reach)
        for i, seed in enumerate(range(first_seed, first_seed + count))
    ]
    t0 = time.perf_counter()
    if jobs > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(jobs) as pool:
            results = pool.map(_check_one, tasks, chunksize=4)
    else:
        results = [_check_one(task) for task in tasks]
    wall = time.perf_counter() - t0

    bad = [r for r in results if not r["ok"]]
    with_warp = [r for r in results if r["warp"]]
    n = len(results)

    def mean(key: str, rows=results, sub: Optional[str] = None) -> float:
        values = [(r[key][sub] if sub else r[key]) for r in rows if key in r]
        return sum(values) / len(values) if values else 0.0

    print(f"уровней: {n}, ок: {n - len(bad)}, с ошибками: {len(bad)}; настенное время {wall:.0f} с, "
          f"процессов {jobs}")
    print(f"с секретной комнатой: {len(with_warp)} ({len(with_warp) / n:.0%}); "
          f"по сложности <0.15: "
          f"{sum(r['warp'] for r in results if r['difficulty'] < 0.15)}/"
          f"{sum(1 for r in results if r['difficulty'] < 0.15)}, ≥0.15: "
          f"{sum(r['warp'] for r in results if r['difficulty'] >= 0.15)}/"
          f"{sum(1 for r in results if r['difficulty'] >= 0.15)}")
    if with_warp:
        dist = [r["warp_distance"] for r in with_warp if "warp_distance" in r]
        print(f"  вход→выход: {min(dist)}–{max(dist)} тайлов, в среднем {sum(dist) / len(dist):.0f}; "
              f"в комнате в среднем монет {mean('room_coins', with_warp):.1f}, "
              f"«?» {mean('room_blocks', with_warp):.1f}, врагов {mean('room_enemies', with_warp):.2f}")
        worse = sum(1 for r in with_warp if r.get("skipped_points", 0) > r.get("room_points", 0))
        print(f"  очки: в комнате в среднем {mean('room_points', with_warp):.0f}, между трубами "
              f"{mean('skipped_points', with_warp):.0f} (макс. доля "
              f"{max(r.get('skipped_points', 0) / max(r.get('room_points', 1), 1) for r in with_warp):.2f}); "
              f"комната беднее пропущенного: {worse}")
        if simulate:
            print(f"  симуляция похода от старта: {sum(1 for r in with_warp if 'sim_frames' in r and r['ok'])}"
                  f"/{len(with_warp)} ок, в среднем {mean('sim_frames', with_warp):.0f} кадров; "
                  f"путь перестроен после неточной посадки на "
                  f"{sum(1 for r in with_warp if r.get('sim_replans'))} уровнях")
        else:
            print("  симуляция похода: не запускалась (--no-sim)")
    mains = [r["main"] for r in results if "main" in r]
    if mains:
        print(f"основной уровень, минимум на уровень: «?» {min(m['questions'] for m in mains)}, "
              f"врагов {min(m['enemies'] for m in mains)}, монет {min(m['coins'] for m in mains)} "
              f"(правило: ≥ {MIN_QUESTIONS}, ≥ {MIN_ENEMIES}, ≥ {MIN_COINS})")
    limits = [r["time_limit"] for r in results if r.get("time_limit")]
    if limits:
        print(f"лимит времени: {min(limits)}–{max(limits)} с, в среднем {sum(limits) / len(limits):.0f}")
    if simulate:
        print(f"на уровень (вместе с комнатой, GameWorld.totals): монет {mean('totals', sub='coins'):.1f}, "
              f"ударов «?» {mean('totals', sub='block_hits'):.1f}, врагов {mean('totals', sub='enemies'):.1f}")
        for lo, hi in ((0.0, 0.3), (0.3, 0.7), (0.7, 1.01)):
            rows = [r for r in results if lo <= r["difficulty"] < hi and "totals" in r]
            if rows:
                print(f"  сложность {lo:.1f}–{min(hi, 1.0):.1f}: монет {mean('totals', rows, 'coins'):.1f}, "
                      f"ударов «?» {mean('totals', rows, 'block_hits'):.1f}, "
                      f"врагов {mean('totals', rows, 'enemies'):.1f}, комнат {sum(r['warp'] for r in rows)}/{len(rows)}")
    digest = hashlib.sha256("".join(r.get("digest", "-") for r in results).encode()).hexdigest()
    tiles = hashlib.sha256(Path(sys.modules[LevelBuilder.__module__].__file__).read_bytes()).hexdigest()
    print(f"дайджест уровней (сверять между машинами): {digest[:16]} "
          f"при tools/level_tiles.py {tiles[:12]}, генератор v{GENERATOR_VERSION}")
    gen = [r["gen_ms"] for r in results]
    print(f"генерация (процессорное): медиана {_percentile(gen, 0.5):.1f} мс, p90 {_percentile(gen, 0.9):.1f} мс, "
          f"макс {max(gen):.1f} мс")
    if reach:
        rows = [r["reach_ms"] for r in results if "reach_ms" in r]
        if rows:
            print(f"граф полётов: медиана {_percentile(rows, 0.5):.0f} мс, макс {max(rows):.0f} мс")
        cov = [r["coins_reach"] / r["coins_main"] for r in results if r.get("coins_main")]
        if cov:
            print(f"монет уровня достижимо графом: в среднем {sum(cov) / len(cov):.1%}")
    notes = Counter(note for r in results for note in r.get("notes", []))
    for note, times in notes.most_common(5):
        print(f"  заметка ×{times}: {note}")
    for r in bad[:20]:
        print(f"ОШИБКА seed {r['seed']} d {r['difficulty']}:")
        for error in r["errors"][:6]:
            print(f"    {error}")
    return 0 if not bad else 1


# ===============================================================
# CLI
# ===============================================================


def _bench(count: int, difficulty: Optional[float]) -> None:
    """Скорость: генерация отдельно и генерация + GameWorld(level).

    Печатает и настенное время, и процессорное: на общей нагруженной машине
    настенное шумит в разы, процессорное — честнее.
    """

    from core.world import GameWorld

    rows = []
    for seed in range(count):
        d = difficulty if difficulty is not None else (seed % 11) / 10
        w0, c0 = time.perf_counter(), time.process_time()
        level = generate_level(seed, d)
        w1, c1 = time.perf_counter(), time.process_time()
        GameWorld(level)
        w2, c2 = time.perf_counter(), time.process_time()
        rows.append(((w1 - w0) * 1000, (c1 - c0) * 1000, (w2 - w0) * 1000, (c2 - c0) * 1000,
                     len(level["objects"]), sum(1 for o in level["objects"]
                                                if o["category"] == "Entities") - 1,
                     bool(level.get("warps"))))

    def stats(index: int) -> str:
        values = sorted(row[index] for row in rows)
        return (f"медиана {values[len(values) // 2]:6.1f} мс, p90 {values[int(len(values) * 0.9)]:6.1f} мс, "
                f"макс {values[-1]:6.1f} мс")

    print(f"уровней: {count}, объектов в среднем {sum(r[4] for r in rows) / count:.0f}, "
          f"врагов в среднем {sum(r[5] for r in rows) / count:.1f}, с комнатой {sum(r[6] for r in rows)}")
    print(f"генерация, настенное:             {stats(0)}")
    print(f"генерация, процессорное:          {stats(1)}")
    print(f"генерация + GameWorld, настенное:  {stats(2)}")
    print(f"генерация + GameWorld, процессорное: {stats(3)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Процедурный уровень для обучения")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=float, default=None,
                        help="0..1 (по умолчанию 0.5; для --check N и --bench — по кругу 0..1)")
    parser.add_argument("--length", type=int, default=None, help="длина в тайлах")
    parser.add_argument("--out", help="записать JSON (например maps/gen_3.json)")
    parser.add_argument("--ascii", action="store_true", help="напечатать уровень и комнату символами")
    parser.add_argument("--check", type=int, nargs="?", const=0, default=None, metavar="N",
                        help="без N — проверить уровень --seed; с N — N уровней подряд от --seed")
    parser.add_argument("--jobs", type=int, default=1, help="с --check N: процессов")
    parser.add_argument("--no-sim", action="store_true", help="с --check: без симуляции GameWorld")
    parser.add_argument("--no-reach", action="store_true", help="с --check: без графа полётов")
    parser.add_argument("--bench", type=int, metavar="N", help="замер скорости на N уровнях")
    args = parser.parse_args(argv)

    if args.bench:
        _bench(args.bench, args.difficulty)
        return 0

    if args.check:
        return check_many(args.check, args.seed, args.difficulty, jobs=args.jobs,
                          simulate=not args.no_sim, reach=not args.no_reach)

    difficulty = 0.5 if args.difficulty is None else args.difficulty
    level = generate_level(args.seed, difficulty, length_tiles=args.length)
    meta = level["generator"]
    print(f"{level['name']}: {meta['length_tiles']} тайлов, {len(level['objects'])} объектов, "
          f"лимит {level['time_limit']} с")
    print("чанки: " + " ".join(meta["chunks"]) + f" | финиш: {meta['finish']}")
    if meta["warp"]:
        print(f"секретная комната: {meta['warp']}")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(level, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"записан {path}")

    if args.ascii:
        print(render_ascii(level))

    if args.check is not None:
        if args.length:
            print("внимание: --check проверяет уровень длины по умолчанию")
        result = check_generated(args.seed, difficulty, simulate=not args.no_sim, reach=not args.no_reach)
        extra = {key: result[key] for key in ("gen_ms", "reach_ms", "sim_ms", "sim_frames", "totals",
                                              "coins_reach", "coins_main") if key in result}
        print(f"проверка: {'ок' if result['ok'] else 'ОШИБКИ'} {extra}")
        for note in result["notes"]:
            print(f"  заметка: {note}")
        for error in result["errors"]:
            print(f"  {error}")
        return 0 if result["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

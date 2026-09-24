"""Процедурный генератор уровней для обучения нейросети.

    from ai.levelgen import generate_level
    level = generate_level(seed=3, difficulty=0.7)   # словарь уровня
    world = GameWorld(level)                          # сразу в мир

    .venv/bin/python -m ai.levelgen --seed 3 --difficulty 0.7 --ascii
    .venv/bin/python -m ai.levelgen --seed 3 --difficulty 0.7 --out maps/gen_3.json
    .venv/bin/python -m ai.levelgen --seed 3 --check      # прогнать tools.level_check
    .venv/bin/python -m ai.levelgen --bench 50           # скорость генерации + GameWorld

Уровень собирается из «чанков» слева направо: ровный участок, яма, трубы,
загон между трубами, лестница (иногда с ямой посередине), кирпичная полка
с монетами и «?», ряд блоков «?», группа врагов, парящие платформы над
пропастью. В конце — лестница к флагу, флагшток и замок как в 1-1.

Детерминирован по (seed, difficulty): один и тот же вход — байт в байт
тот же уровень. Проходим по построению: обязательные прыжки не выше 4
тайлов и через ямы не шире 4 (физика игрока — 6.8 и ~6); это же проверяет
tools/level_check.py симуляцией полёта кадр за кадром.

difficulty 0 — почти ровно: редкие ямы в тайл, трубы в 2, одиночные гумбы.
difficulty 1 — как «Трубы и ямы»: ямы до 4, трубы до 4, группы врагов,
koopa, пропасти с цепочкой платформ, яма перед флагштоком.

Главное правило игры — побеждает тот, у кого больше очков, поэтому
генератор кладёт «риск против очков»: монеты над ямами (дорожкой по
траектории и выше неё), «?» над краем ямы, враги, которых выгоднее
растоптать, и лестница, с которой флаг хватается за верх (5000).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from tools.level_tiles import STAND_TY, LevelBuilder

GENERATOR_VERSION = 1

# Длина уровня по умолчанию (тайлов от старта до конца замка)
DEFAULT_LENGTH_RANGE = (140, 180)

# Физика игрока (для справки и проверок): прыжок 6.8 тайла вверх, ~7.5 вдаль
SAFE_RISE = 4  # обязательный подъём — не выше
SAFE_PIT = 4  # обязательная яма — не шире


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
    risk: float  # вероятность «жадных» монет и «?» над ямой
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
            koopa=0.4 * d,
            risk=0.2 + 0.5 * d,
            spacer=(3 - _round(d), 6 - _round(2 * d)),
        )


def _round(value: float) -> int:
    """Округление половин вверх (round() в Python округляет 0.5 к чётному)."""
    return int(value + 0.5)


# ===============================================================
# ЧАНКИ
# ===============================================================
# Каждый чанк начинается в столбце x на обычной земле, сам кладёт землю
# под свои столбцы и возвращает первый свободный столбец за собой. На
# обоих краях чанка — ровная земля, поэтому чанки стыкуются в любом порядке.


def chunk_flat(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Ровный участок; иногда с дорожкой монет над землёй."""
    n = rng.randint(4, 8)
    b.ground(x, x + n - 1)
    if rng.random() < 0.5:
        row = rng.choice((STAND_TY + 1, STAND_TY + 3))
        b.coins(x + 1, x + n - 2, row)
    return x + n


def chunk_pit(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Яма шириной 1..max_pit с монетами над ней."""
    w = rng.randint(1, k.max_pit)
    b.ground(x, x + 1)
    b.ground(x + 2 + w, x + 3 + w)
    edge = x + 1

    roll = rng.random()
    if roll < 0.55:
        # Дорожка по траектории прыжка с края
        b.jump_trail(edge)
    elif roll < 0.55 + k.risk * 0.6:
        # «Жадные» монеты: выше обычной дорожки, нужен точный прыжок
        b.coins(x + 2, x + 1 + w, STAND_TY + 5)
    if w >= 2 and rng.random() < k.risk * 0.5:
        # Блок «?» над первым столбцом ямы: бьётся прыжком с самого края
        b.question(x + 2, STAND_TY + 3)
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
        kind = "koopa" if rng.random() < k.koopa else "goomba"
        b.enemy(kind, x + 4 + 2 * i)
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
    """Кирпичная полка над землёй: монеты на ней, «?» над ней."""
    w = rng.randint(3, 6)
    row = rng.choice((STAND_TY + 3, STAND_TY + 4))
    n = w + 4
    b.ground(x, x + n - 1)
    b.row(x + 2, x + 1 + w, row)
    b.coins(x + 2, x + 1 + w, row + 1)
    if rng.random() < 0.7:
        b.question(x + 2 + rng.randint(0, w - 1), row + 4)
    if rng.random() < 0.35 + 0.3 * k.difficulty:
        # Гумба на полке — растоптать или спрыгнуть
        b.enemy("goomba", x + 1 + w, row + 1)
    return x + n


def chunk_blocks(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Ряд «?» и кирпичей как в 1-1, один «?» повыше."""
    b.ground(x, x + 8)
    for i in range(5):
        if i % 2 == 0:
            b.question(x + 2 + i, STAND_TY + 3)
        else:
            b.row(x + 2 + i, x + 2 + i, STAND_TY + 3)
    b.question(x + 4, STAND_TY + 7)
    if rng.random() < k.difficulty:
        b.enemy("goomba", x + 5)
    return x + 9


def chunk_enemies(b: LevelBuilder, x: int, rng: random.Random, k: Knobs) -> int:
    """Группа врагов на ровном; монеты над ними — за прыжок сверху."""
    count = rng.randint(1, k.group)
    n = 4 + 2 * count + rng.randint(0, 3)
    b.ground(x, x + n - 1)
    for i in range(count):
        kind = "koopa" if rng.random() < k.koopa else "goomba"
        b.enemy(kind, x + 3 + 2 * i)
    if rng.random() < 0.5:
        b.coins(x + 2, x + 2 + 2 * count, STAND_TY + 3)
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
    "flat": (chunk_flat, 3.0, 0.5),
    "pit": (chunk_pit, 0.6, 3.0),
    "tube": (chunk_tube, 1.5, 2.0),
    "pen": (chunk_pen, 0.2, 1.0),
    "stairs": (chunk_stairs, 0.6, 1.2),
    "shelf": (chunk_shelf, 1.0, 0.8),
    "blocks": (chunk_blocks, 0.8, 0.6),
    "enemies": (chunk_enemies, 0.8, 1.6),
    "sky": (chunk_sky, 0.0, 1.2),
}


def _weights(d: float) -> list[float]:
    return [w0 + (w1 - w0) * d for _, w0, w1 in CHUNKS.values()]


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
    """Уровень в формате редактора (с name, description и time_limit).

    seed — любое целое; difficulty — 0..1 (обрезается); length_tiles —
    длина от старта до конца замка, по умолчанию 140–180 по seed.
    """

    d = min(max(float(difficulty), 0.0), 1.0)
    d = round(d, 4)
    rng = random.Random(f"mario-levelgen:v{GENERATOR_VERSION}:{int(seed)}:{d:.4f}")
    k = Knobs.of(d)

    length = int(length_tiles) if length_tiles else rng.randint(*DEFAULT_LENGTH_RANGE)
    length = max(length, 60)

    b = LevelBuilder(
        f"Генерация {seed} · сложность {d:.2f}",
        "",
    )

    # --- Старт: ровная площадка
    b.ground(0, 9)
    b.start(3)
    x = 10

    # --- Середина: чанки, пока остаётся место под финиш (~30 тайлов)
    finish_room = 30
    names = list(CHUNKS)
    weights = _weights(d)
    used: list[str] = []
    while x < length - finish_room:
        name = rng.choices(names, weights)[0]
        # Две пропасти подряд — перебор даже для d = 1
        if name == "sky" and used and used[-1] == "sky":
            name = "flat"
        x = CHUNKS[name][0](b, x, rng, k)
        used.append(name)

        gap = rng.randint(*k.spacer)
        b.ground(x, x + gap - 1)
        x += gap

    # --- Финиш
    pole, finish = _finish(b, x, rng, k)

    level = b.to_level()
    counts = {name: used.count(name) for name in names if name in used}
    level["description"] = (
        f"Процедурный уровень: seed {seed}, сложность {d:.2f}. "
        + ", ".join(f"{name}×{n}" for name, n in counts.items())
    )
    level["generator"] = {
        "version": GENERATOR_VERSION,
        "seed": int(seed),
        "difficulty": d,
        "length_tiles": b.width_tiles,
        "chunks": used,
        "finish": finish,
    }
    return level


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
                                                if o["category"] == "Entities") - 1))

    def stats(index: int) -> str:
        values = sorted(row[index] for row in rows)
        return (f"медиана {values[len(values) // 2]:6.1f} мс, p90 {values[int(len(values) * 0.9)]:6.1f} мс, "
                f"макс {values[-1]:6.1f} мс")

    print(f"уровней: {count}, объектов в среднем {sum(r[4] for r in rows) / count:.0f}, "
          f"врагов в среднем {sum(r[5] for r in rows) / count:.1f}")
    print(f"генерация, настенное:             {stats(0)}")
    print(f"генерация, процессорное:          {stats(1)}")
    print(f"генерация + GameWorld, настенное:  {stats(2)}")
    print(f"генерация + GameWorld, процессорное: {stats(3)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Процедурный уровень для обучения")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--length", type=int, default=None, help="длина в тайлах")
    parser.add_argument("--out", help="записать JSON (например maps/gen_3.json)")
    parser.add_argument("--ascii", action="store_true", help="напечатать уровень символами")
    parser.add_argument("--check", action="store_true", help="проверить tools.level_check")
    parser.add_argument("--bot", action="store_true", help="с --check: прогнать HeuristicBot")
    parser.add_argument("--bench", type=int, metavar="N", help="замер скорости на N уровнях")
    args = parser.parse_args(argv)

    if args.bench:
        _bench(args.bench, args.difficulty if "--difficulty" in (argv or sys.argv) else None)
        return 0

    level = generate_level(args.seed, args.difficulty, length_tiles=args.length)
    meta = level["generator"]
    print(f"{level['name']}: {meta['length_tiles']} тайлов, {len(level['objects'])} объектов, "
          f"лимит {level['time_limit']} с")
    print("чанки: " + " ".join(meta["chunks"]) + f" | финиш: {meta['finish']}")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(level, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"записан {path}")

    if args.ascii:
        from tools.level_ascii import render
        print(render(level))

    if args.check:
        from tools.level_check import check_level, format_report
        report = check_level(level, bot=args.bot)
        print(format_report(report))
        return 0 if report.ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

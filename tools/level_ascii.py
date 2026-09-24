"""Уровень символами: быстрый взгляд на карту без окна.

    .venv/bin/python -m tools.level_ascii maps/level_warmup.json
    .venv/bin/python -m tools.level_ascii maps/custom_level.json --detail --width 100

Рисует по тайлам, сверху вниз, кусками по 120 колонок (--width). Над каждым
куском — линейка с номерами колонок, слева — номер ряда ty.

Обозначения по умолчанию:
    #  кирпич / земля / блок      T  труба          ?  блок «?»
    o  монета                     g  goomba          k  koopa
    M  старт игрока               |  шток            *  шар на штоке
    F  флаг                       C  замок           .  пусто
С --detail твёрдые блоки различаются: = земля, B кирпич, X камень лестницы.
"""

from __future__ import annotations

import argparse
import sys

from tools.level_tiles import LevelSource, level_cells, load_level

# ===============================================================
# СИМВОЛЫ
# ===============================================================

CHARS = {
    "ground": "#",
    "brick": "#",
    "hard": "#",
    "stone": "#",
    "block": "#",
    "tube_top_l": "T",
    "tube_top_r": "T",
    "tube_l": "T",
    "tube_r": "T",
    "question": "?",
    "coin": "o",
    "flag": "F",
    "object": "*",
    "stick": "|",
    "ball": "*",
    "castle": "C",
    "deco": ",",
    "mario": "M",
    "goomba": "g",
    "koopa": "k",
    "entity": "e",
}

DETAIL_CHARS = {
    "ground": "=",
    "brick": "B",
    "hard": "X",
    "stone": "X",
}

# Если в клетке несколько объектов — рисуется самый важный
PRIORITY = {
    "mario": 0, "goomba": 1, "koopa": 1, "entity": 1,
    "flag": 2, "question": 3, "coin": 4, "object": 4,
}

LEGEND = (
    "# блок  T труба  ? блок «?»  o монета  g goomba  k koopa  "
    "M старт  | шток  * шар  F флаг  C замок"
)
DETAIL_LEGEND = "= земля  B кирпич  X камень  " + LEGEND.replace("# блок  ", "")


def tile_chars(data: dict, detail: bool = False) -> dict[tuple[int, int], str]:
    """Клетка → символ (с учётом приоритета при наложении)."""

    best: dict[tuple[int, int], tuple[int, str]] = {}
    for cell in level_cells(data):
        kind = cell.kind
        if kind.startswith("castle"):
            kind = "castle"
        char = (DETAIL_CHARS.get(kind) if detail else None) or CHARS.get(kind, ",")
        rank = PRIORITY.get(kind, 5)
        key = (cell.tx, cell.ty)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, char)
    return {key: char for key, (_, char) in best.items()}


def render(level: LevelSource, width: int = 120, detail: bool = False,
           legend: bool = True) -> str:
    """ASCII-картинка уровня кусками по width колонок."""

    data = load_level(level)
    chars = tile_chars(data, detail)
    if not chars:
        return "(пустой уровень)"

    min_tx = min(0, min(tx for tx, _ in chars))
    max_tx = max(tx for tx, _ in chars)
    max_ty = max(ty for _, ty in chars)
    min_ty = min(0, min(ty for _, ty in chars))

    lines = []
    name = data.get("name") or "без имени"
    time_limit = data.get("time_limit")
    header = f"{name}: {max_tx - min_tx + 1} тайлов"
    if time_limit is not None:
        header += f", лимит {time_limit} с"
    lines.append(header)

    for start in range(min_tx, max_tx + 1, width):
        stop = min(start + width, max_tx + 1)
        lines.append("")

        # Линейка: номер колонки над каждой десятой
        ruler = [" "] * (stop - start)
        for tx in range(start, stop):
            if tx % 10 == 0:
                label = str(tx)
                for i, ch in enumerate(label):
                    if tx - start + i < len(ruler):
                        ruler[tx - start + i] = ch
        lines.append("    " + "".join(ruler))

        for ty in range(max_ty, min_ty - 1, -1):
            row = "".join(chars.get((tx, ty), ".") for tx in range(start, stop))
            lines.append(f"{ty:>3} {row}")

    if legend:
        lines.append("")
        lines.append(DETAIL_LEGEND if detail else LEGEND)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Уровень символами по тайлам")
    parser.add_argument("level", help="путь к JSON уровня (например maps/level_warmup.json)")
    parser.add_argument("--width", type=int, default=120, help="колонок в куске (120)")
    parser.add_argument("--detail", action="store_true",
                        help="различать землю (=), кирпич (B) и камень (X)")
    parser.add_argument("--no-legend", action="store_true", help="без легенды внизу")
    args = parser.parse_args(argv)

    print(render(args.level, width=args.width, detail=args.detail, legend=not args.no_legend))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Секретная комната уровня 1-1 (maps/custom_level.json), как в Super Mario Bros.

    .venv/bin/python -m tools.level_1_1            # вписать ключ "warps" в custom_level.json
    .venv/bin/python -m tools.level_1_1 --dry-run  # показать комнату, файл не трогать

Уровень 1-1 — работа автора из редактора: его объекты не трогаются, скрипт
только добавляет (или заменяет) ключ "warps". Как в оригинале: четвёртая
труба (левый верхний тайл (1872, 304), высота 4) уводит в подземную
комнату с монетами, выход — боковая труба у правой стены, игрок вылезает
из трубы (5872, 272) ближе к концу уровня.

Комната как бонусная в SMB 1-1: кирпичные стены и потолок, кирпичный
блок посередине, на нём и над ним 19 монет, ряд из трёх блоков «?» у
выхода. Сверх оригинала — монеты на полу и четыре «?» над кирпичным
блоком: труба уводит почти к концу уровня (столбцы 58 → 183), и комната
должна стоить заметно больше пропущенного (18 ударов «?» и 6 врагов,
~2400 очков, плюс ~17 с пути — ~330 бонуса за время). Иначе труба —
это пробег на скорость, а не тщательная игра.
"""

from __future__ import annotations

import argparse
import json
import sys

from tools.level_tiles import MAPS_DIR, RoomBuilder, tile_of, warp_entry

LEVEL_FILE = MAPS_DIR / "custom_level.json"

# Левый верхний тайл трубы-входа и трубы-выхода (пиксели центра → тайлы)
ENTRY_PIPE = (tile_of(1872), tile_of(304))   # (58, 9) — четвёртая труба, высота 4
EXIT_PIPE = (tile_of(5872), tile_of(272))    # (183, 8) — труба перед лестницей, высота 3


def room_1_1() -> RoomBuilder:
    """Бонусная комната 1-1: блок из кирпича 7×3, 33 монеты, три «?» у выхода
    и четыре над кирпичным блоком."""

    room = RoomBuilder("1-1: комната с монетами").box(20, 12).start(2, 9)
    room.side_exit()

    # Кирпичный блок посередине: на нём два ряда монет, над ним ещё пять
    for ty in range(2, 5):
        room.row(4, 10, ty)
    room.coins(4, 10, 5)
    room.coins(4, 10, 6)
    room.coins(5, 9, 9)
    # Над кирпичным блоком — «?» через один: низ на 3 тайла выше его верха;
    # монеты рядов 9 и 10 берутся прыжком между блоками или стоя на них
    room.blocks(4, 8, "?.?.?.?")
    room.coins(4, 10, 10)

    # Монеты на полу: у точки появления и перед устьем
    room.coins(1, 3, 2)
    room.coins(11, 14, 2)

    # Три блока «?» над полом перед устьем: низ на 3 тайла выше пола
    room.blocks(12, 5, "???")
    return room


def build_warps() -> list[dict]:
    entry = warp_entry(ENTRY_PIPE, EXIT_PIPE, room_1_1(), "coins1")
    return [{k: v for k, v in entry.items() if not k.startswith("_")}]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Секретная комната уровня 1-1")
    parser.add_argument("--dry-run", action="store_true", help="не писать файл")
    args = parser.parse_args(argv)

    data = json.loads(LEVEL_FILE.read_text(encoding="utf-8"))
    objects_before = json.dumps(data.get("objects"), sort_keys=True)
    data["warps"] = build_warps()

    from tools.level_ascii import render
    for warp in data["warps"]:
        print(f"комната {warp['id']}: вход {ENTRY_PIPE}, выход {EXIT_PIPE}")
        print(render(dict(warp["room"], name=f"комната {warp['id']}"), legend=False))

    # Объекты автора — те же, что были
    assert json.dumps(data.get("objects"), sort_keys=True) == objects_before

    if not args.dry_run:
        LEVEL_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        print(f"warps → {LEVEL_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

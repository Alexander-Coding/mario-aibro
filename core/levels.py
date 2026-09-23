"""Список уровней игры: maps/levels.json.

    {"levels": [{"id": "1-1", "file": "custom_level.json", "name": "...",
                 "time_limit": 200, "split": "test"}]}

split — роль уровня для нейросети: "train" участвует в обучении,
"test" — отложенный уровень, на котором проверяют обобщение (модель его
не видела). Для живой игры split ничего не значит.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from map import BASE_DIR

MAPS_DIR = BASE_DIR / "maps"
MANIFEST = MAPS_DIR / "levels.json"


@dataclass(frozen=True)
class LevelInfo:
    id: str
    file: str
    name: str
    time_limit: float = 200.0
    split: str = "test"
    description: str = ""

    @property
    def path(self) -> Path:
        return MAPS_DIR / self.file


def load_levels(split: str | None = None) -> list[LevelInfo]:
    """Уровни из манифеста по порядку. Нет манифеста — один 1-1."""

    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        levels = [
            LevelInfo(
                id=str(item["id"]),
                file=str(item["file"]),
                name=str(item.get("name", item["id"])),
                time_limit=float(item.get("time_limit", 200.0)),
                split=str(item.get("split", "test")),
                description=str(item.get("description", "")),
            )
            for item in data.get("levels", [])
        ]
    else:
        levels = [LevelInfo(id="1-1", file="custom_level.json", name="1-1")]

    if split is not None:
        levels = [level for level in levels if level.split == split]

    return levels


def make_world(level: LevelInfo, **kwargs):
    """GameWorld для уровня из манифеста (имя и таймер — из манифеста)."""

    from core.world import GameWorld

    return GameWorld(
        level.path,
        name=level.name,
        time_limit=level.time_limit,
        **kwargs,
    )

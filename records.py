"""Рекорды уровней: кто набрал больше очков — человек или ИИ.

Файл records.json в корне проекта:

    {
        "1-1": {
            "human": {"score": 6380, "won": true, "date": "2026-09-23T21:04:11"},
            "ai":    {"score": 7120, "won": true, "date": "2026-09-23T21:10:52",
                      "details": {"coins": 12, "coins_total": 20, "secrets": 1, "secrets_total": 1}}
        }
    }

"details" — что собрано из возможного в забеге-рекорде (монеты, удары по
блокам «?», враги, тайники; ключи как в GameWorld.summary()). Поле
необязательное: старые записи без него читаются как раньше.

Запись обновляется в конце каждого уровня — и после победы, и после
смерти. Хранится лучший счёт: побеждает не тот, кто прошёл, а тот, кто
набрал больше очков. Файл пишется целиком через временный файл, чтобы
оборванная запись не испортила рекорды.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

RECORDS_PATH = Path(__file__).resolve().parent / "records.json"

# Кто играл: человек или нейросеть
HUMAN = "human"
AI = "ai"
PLAYERS = (HUMAN, AI)

PLAYER_NAMES = {
    HUMAN: "Человек",
    AI: "ИИ",
}


class Records:
    """Рекорды всех уровней. Путь можно подменить (тесты пишут во временный файл)."""

    def __init__(self, path: Union[str, Path, None] = None):
        self.path = Path(path) if path is not None else RECORDS_PATH
        self.data: dict[str, dict[str, dict]] = self._load()

    # ========================================================
    # ЧТЕНИЕ
    # ========================================================

    def _load(self) -> dict:
        """Читает файл. Нет файла или он битый — рекордов пока нет."""

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            print(f"[records] не удалось прочитать {self.path}: {error}")
            return {}

        if not isinstance(raw, dict):
            return {}

        data: dict[str, dict[str, dict]] = {}

        for level_id, entry in raw.items():
            if not isinstance(entry, dict):
                continue

            clean = {}
            for who in PLAYERS:
                record = entry.get(who)
                if isinstance(record, dict) and "score" in record:
                    try:
                        clean[who] = {
                            "score": int(record.get("score", 0)),
                            "won": bool(record.get("won", False)),
                            "date": str(record.get("date", "")),
                        }
                    except (TypeError, ValueError, OverflowError):
                        # OverflowError — Infinity в файле: int(float("inf"))
                        continue
                    details = _clean_details(record.get("details"))
                    if details:
                        clean[who]["details"] = details

            if clean:
                data[str(level_id)] = clean

        return data

    def reload(self) -> None:
        self.data = self._load()

    def get(self, level_id: str, who: str) -> Optional[dict]:
        """Рекорд уровня для человека или ИИ: {"score", "won", "date"} или None."""

        return self.data.get(str(level_id), {}).get(who)

    def level(self, level_id: str) -> dict[str, Optional[dict]]:
        """Оба рекорда уровня: {"human": ..., "ai": ...}."""

        return {who: self.get(level_id, who) for who in PLAYERS}

    def best_score(self, level_id: str, who: str) -> Optional[int]:
        record = self.get(level_id, who)
        return None if record is None else int(record["score"])

    def leader(self, level_id: str) -> Optional[str]:
        """Кто набрал на уровне больше: "human", "ai", "tie" или None (некому сравнивать)."""

        return compare_scores(
            self.best_score(level_id, HUMAN),
            self.best_score(level_id, AI),
        )

    # ========================================================
    # ЗАПИСЬ
    # ========================================================

    def update(
        self,
        level_id: str,
        who: str,
        score: int,
        won: bool,
        date: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> bool:
        """Сохраняет результат, если он лучше прежнего. True — рекорд обновлён.

        Лучше — больше очков; при равенстве очков победа лучше смерти.
        details — что собрано из возможного ({"coins": 12, "coins_total": 20, ...}),
        пишется вместе с рекордом.
        """

        if who not in PLAYERS:
            raise ValueError(f"who должен быть одним из {PLAYERS}, а не {who!r}")

        score = int(score)
        won = bool(won)
        current = self.get(level_id, who)

        better = (
            current is None
            or score > current["score"]
            or (score == current["score"] and won and not current["won"])
        )

        if not better:
            return False

        record = {
            "score": score,
            "won": won,
            "date": date or datetime.now().isoformat(timespec="seconds"),
        }
        details = _clean_details(details)
        if details:
            record["details"] = details

        self.data.setdefault(str(level_id), {})[who] = record
        self.save()
        return True

    def save(self) -> None:
        """Пишет файл целиком через временный — без полузаписанного JSON."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")

        try:
            tmp_path.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, self.path)
        except OSError as error:
            print(f"[records] не удалось сохранить {self.path}: {error}")
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _clean_details(details) -> dict:
    """Только целые числа по строковым ключам — остальное молча отбрасывается."""

    if not isinstance(details, dict):
        return {}

    clean = {}
    for key, value in details.items():
        if isinstance(value, bool):
            continue
        try:
            clean[str(key)] = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
    return clean


def compare_scores(human: Optional[int], ai: Optional[int]) -> Optional[str]:
    """Сравнение двух счетов: "human", "ai", "tie" или None, если сравнивать не с чем."""

    if human is None and ai is None:
        return None
    if ai is None:
        return HUMAN
    if human is None:
        return AI
    if human > ai:
        return HUMAN
    if ai > human:
        return AI
    return "tie"


def format_score(score: Optional[int]) -> str:
    """12345 → «12 345», None → «—»."""

    if score is None:
        return "—"
    return f"{int(score):,}".replace(",", " ")

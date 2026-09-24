"""Типизированные решения в стиле Jev.

Модель не пишет текст: на вопрос заранее известного типа она отвечает
значением и вероятностями. Три примитива, как у Jev:

- choice — выбор из вариантов:      «какое нажатие сейчас?»
- noul   — да/нет как вероятность:  «погибну ли в ближайшие 2 секунды?»
- score  — упорядоченные уровни:    «сколько ещё очков наберу до конца?»

    model = DecisionModel.load()
    answer = model.predict(world)
    answer["answers"]["move"]["choice"]          # "вправо+прыжок"
    answer["answers"]["danger"]["probability"]   # 0.07
    answer["answers"]["points"]["score"]         # "3–5 тыс."
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch

from ai.features import ACTION_LABELS, FeatureBuilder
from ai.model import POINT_BIN_CENTERS, POINT_BIN_LABELS, MarioJev
from core.world import GameWorld

MODEL_DIR = Path(__file__).resolve().parent / "models"

def default_model_path() -> Path:
    """Модель по умолчанию для игры и турнира — самая свежая итерация.

    Берётся самый новый откалиброванный `*_final.pt`, если он не старше
    самого нового `*_best.pt`; иначе — самый новый `*_best.pt`, затем
    `*_last.pt`. Так новая итерация обучения (ai.iterate --name iterN)
    сразу попадает в игру, а устаревший final не подменяет свежую модель.
    """

    def newest(pattern: str) -> Optional[Path]:
        found = sorted(MODEL_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        return found[-1] if found else None

    final, best, last = newest("*_final.pt"), newest("*_best.pt"), newest("*_last.pt")
    if final is not None and (best is None or final.stat().st_mtime >= best.stat().st_mtime):
        return final
    for path in (best, last):
        if path is not None:
            return path
    raise FileNotFoundError(
        f"Модель не найдена в {MODEL_DIR} (*_final.pt, *_best.pt, *_last.pt). "
        "Обучите: .venv/bin/python -m ai.train"
    )


def _confidence(probs: torch.Tensor) -> float:
    """1 − нормированная энтропия: 1 — уверен в одном варианте, 0 — всё равно."""
    p = probs.clamp_min(1e-9)
    entropy = float(-(p * p.log()).sum())
    return max(0.0, 1.0 - entropy / math.log(len(p)))


class DecisionModel:

    def __init__(self, model: MarioJev, name: str = "MarioJev", meta: Optional[dict] = None):
        self.model = model.eval()
        self.name = name
        self.meta = meta or {}
        self._builder: Optional[FeatureBuilder] = None
        self._world_id: Optional[int] = None

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DecisionModel":
        path = Path(path) if path else default_model_path()
        data = torch.load(path, map_location="cpu", weights_only=False)
        model = MarioJev(**data["config"])
        model.load_state_dict(data["state_dict"])
        # «MarioJev mariojev2 final» → «MarioJev 2 final» — коротко, чтобы влезло в панель ИИ
        label = path.stem.replace("mariojev", "").replace("_", " ").strip() or path.stem
        return cls(model, name=f"MarioJev {label}", meta=data.get("meta", {}))

    def _features(self, world: GameWorld) -> FeatureBuilder:
        # Статика уровня кэшируется, пока мир тот же
        if self._builder is None or self._world_id != id(world):
            self._builder = FeatureBuilder(world)
            self._world_id = id(world)
        return self._builder

    @torch.inference_mode()
    def raw(self, world: GameWorld, last_action: int = 0):
        obs = self._features(world).build(last_action)
        batch = {k: torch.as_tensor(v[None]) for k, v in obs.items()}
        logits, value, danger, points = self.model(batch)
        return logits[0], value[0], danger[0], points[0]

    def predict(self, world: GameWorld, last_action: int = 0, deterministic: bool = True,
                temperature: float = 1.0) -> dict:
        """temperature — как резко выбирать: 1.0 — как при обучении, меньше —
        увереннее (0 = argmax). Вероятности в ответе — всегда исходные."""
        started = time.perf_counter()
        logits, value, danger_logit, points_logits = self.raw(world, last_action)

        probs = torch.softmax(logits, -1)
        if deterministic or temperature <= 0:
            index = int(probs.argmax())
        else:
            index = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))

        danger = float(torch.sigmoid(danger_logit / self.model.danger_temperature))

        point_probs = torch.softmax(points_logits / self.model.points_temperature, -1)
        expected = float((point_probs * torch.tensor(POINT_BIN_CENTERS)).sum())
        point_index = int(point_probs.argmax())

        return {
            "answers": {
                "move": {
                    "type": "choice",
                    "choice": ACTION_LABELS[index],
                    "index": index,
                    "probabilities": {label: round(float(p), 4) for label, p in zip(ACTION_LABELS, probs)},
                    "confidence": round(_confidence(probs), 4),
                },
                "danger": {
                    "type": "noul",
                    "question": "Погибну ли в ближайшие 2 секунды?",
                    "probability": round(danger, 4),
                },
                "points": {
                    "type": "score",
                    "question": "Сколько ещё очков наберу до конца уровня?",
                    "score": POINT_BIN_LABELS[point_index],
                    "probabilities": {label: round(float(p), 4) for label, p in zip(POINT_BIN_LABELS, point_probs)},
                    "expected": round(expected),
                    "confidence": round(_confidence(point_probs), 4),
                },
            },
            "value": round(float(value), 4),
            "model": self.name,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

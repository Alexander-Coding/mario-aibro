"""Контроллер ИИ для живой игры и турнира.

Решение — раз в 4 кадра мира, как в среде обучения (ai/env.py): прыжок
нажимается на первом кадре повтора, остальные кадры держится то же
направление. Во время заставки финиша управление у игры.

    controller = load_controller()
    controller.reset(world)
    while not world.done:
        world.step(controller.act(world))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ai.decide import DecisionModel
from ai.env import FRAME_SKIP, action_inputs
from core.world import GameWorld, Inputs


class Controller:

    def __init__(self, decision_model: DecisionModel, deterministic: bool = True, name: Optional[str] = None,
                 temperature: float = 1.0):
        self.decisions = decision_model
        self.deterministic = deterministic
        self.temperature = temperature
        self.name = name or decision_model.name
        self._frame = 0
        self._action = 0
        self._last: Optional[dict] = None

    def reset(self, world: Optional[GameWorld] = None) -> None:
        self._frame = 0
        self._action = 0
        self._last = None

    def act(self, world: GameWorld) -> Inputs:
        if world.done or world.finishing:
            return Inputs()

        first = self._frame % FRAME_SKIP == 0
        if first:
            answer = self.decisions.predict(world, last_action=self._action, deterministic=self.deterministic,
                                            temperature=self.temperature)
            self._action = answer["answers"]["move"]["index"]
            self._last = answer

        self._frame += 1
        return action_inputs(self._action, first)

    def overlay(self) -> Optional[dict]:
        """Данные для панели ИИ в игре (формат — docs/CONTRACT.md)."""

        if self._last is None:
            return None

        answers = self._last["answers"]
        move = answers["move"]
        return {
            "model": self.name,
            "action": move["choice"],
            "probs": move["probabilities"],
            "confidence": move["confidence"],
            "danger": answers["danger"]["probability"],
            "expected": answers["points"]["expected"],
            "latency_ms": self._last["latency_ms"],
        }


def load_controller(model_path: Optional[Path] = None, deterministic: Optional[bool] = None,
                    temperature: Optional[float] = None) -> Controller:
    """FileNotFoundError, если модели нет (интерфейс это показывает).

    По умолчанию решения сэмплируются из распределения модели, как при
    обучении: argmax PPO-политики может выродиться в одно действие."""

    decisions = DecisionModel.load(model_path)
    if deterministic is None:
        # Режим игры, выбранный турниром, хранится в чекпойнте: у зрелой
        # политики argmax лучше сэмплирования (49.8 против 41.5 тыс. очков)
        deterministic = bool(decisions.meta.get("play_deterministic", False))
    if temperature is None:
        # Температура, подобранная турниром, хранится в чекпойнте
        temperature = float(decisions.meta.get("play_temperature", 1.0))
    return Controller(decisions, deterministic=deterministic, temperature=temperature)

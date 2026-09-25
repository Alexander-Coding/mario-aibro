"""Контроллер ИИ для живой игры и турнира.

Решение — раз в 4 кадра мира, как в среде обучения (ai/env.py): прыжок
нажимается на первом кадре повтора, остальные кадры держится то же
направление. Во время заставок (флагшток, труба) управление у игры.

    controller = load_controller()
    controller.reset(world)
    while not world.done:
        world.step(controller.act(world))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ai.decide import DecisionModel
from ai.env import FRAME_SKIP, action_inputs
from core.world import GameWorld, Inputs

# Петля: 45 решений (3 с при решении раз в 4 кадра) в одной области без
# новых очков и в пределах двух тайлов по X — ученик дёргается на месте
LOOP_WINDOW = 45
LOOP_SPAN = 64.0      # px, 2 тайла


class LoopMeter:
    """Метрики зацикливания для турнира (tools/tournament.py) и сбора
    примеров (ai/imitate.py) — одно определение на оба места.

    На каждом решении записывается X игрока, область (уровень или
    комната), счёт и горизонтальное направление решения (+1 вправо,
    −1 влево, 0 без него).

    - stall_frac — доля решений, которыми заканчивается окно из 45 решений
      в одной области с тем же счётом и размахом X меньше 2 тайлов;
    - stalls — число максимальных отрезков таких решений подряд;
    - flips_per_min — смены «вправо» ↔ «влево» (решения без направления
      пропускаются) на минуту игрового времени уровня.
    """

    def __init__(self):
        self.x: list[float] = []
        self.area: list[str] = []
        self.score: list[int] = []
        self.direction: list[int] = []

    def record(self, world: GameWorld, direction: int) -> None:
        self.x.append(float(world.player.center_x))
        self.area.append(world.area.key)
        self.score.append(int(world.score))
        self.direction.append(direction)

    def record_inputs(self, world: GameWorld, inputs: Inputs) -> None:
        self.record(world, int(inputs.right) - int(inputs.left))

    def summary(self, frames: int) -> dict:
        n = len(self.x)
        stalled = np.zeros(n, dtype=bool)
        if n >= LOOP_WINDOW:
            x = np.asarray(self.x)
            score = np.asarray(self.score)
            _, area = np.unique(np.asarray(self.area), return_inverse=True)
            windows = np.lib.stride_tricks.sliding_window_view
            xs, areas = windows(x, LOOP_WINDOW), windows(area, LOOP_WINDOW)
            end = slice(LOOP_WINDOW - 1, None)
            stalled[end] = ((xs.max(1) - xs.min(1) < LOOP_SPAN)
                            & (areas.min(1) == areas.max(1))
                            & (score[end] == score[: n - LOOP_WINDOW + 1]))
        starts = int(np.sum(stalled[1:] & ~stalled[:-1]) + (stalled[0] if n else 0))

        moves = [d for d in self.direction if d]
        flips = sum(a != b for a, b in zip(moves, moves[1:]))
        minutes = frames / 3600
        return {
            "decisions": n,
            "stall_frac": round(float(stalled.mean()), 4) if n else 0.0,
            "stalls": starts,
            "flips": flips,
            "flips_per_min": round(flips / minutes, 1) if minutes > 0 else 0.0,
        }


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
        if world.done or world.cutscene:
            # После заставки (труба) — сразу новое решение, как в среде
            # обучения: она проматывает заставку внутри одного шага
            self._frame = 0
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

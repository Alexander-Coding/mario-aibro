"""Открытая альтернатива Jev (Laya) как игрок без дообучения.

Laya — модель решений с API как у Jev (choice / score / noul), но
обучена на деловых задачах: маршрутизация писем, классификация. Про Марио
она не знает ничего. Здесь она получает текстовое описание сцены и
вопрос типа choice «какую кнопку нажать» — это честная проверка, насколько
универсальная модель решений годится для реального времени в игре.

Не запускается без пакета `laya` и весов (≈0.85 ГБ с Hugging Face,
convaiinnovations/laya). Ставить только с согласия владельца:

    uv pip install --python .venv/bin/python laya

Скорость в турнире не важна: симуляция ждёт решения. В живой игре
решение нужно за 66 мс (15 раз в секунду).
"""

from __future__ import annotations

import time
from typing import Optional

from ai.env import FRAME_SKIP, action_inputs
from core import rules
from core.world import GameWorld, Inputs
from Entities.Enemies.goomba import Goomba
from Objects.Coin import Coin
from Objects.ScoreBlock import ScoreBlock

# Действия в порядке ai.features.ACTIONS и их описания для модели
CHOICES = {
    "wait": "stand still and wait, for example until an enemy walks away",
    "right": "run right, the way ahead is clear and safe",
    "left": "step back left, away from danger",
    "jump": "jump straight up to hit a question block right above",
    "right_jump": "jump forward to the right over a pit, pipe, wall or onto an enemy's head",
    "left_jump": "jump back to the left",
}
CHOICE_INDEX = {name: i for i, name in enumerate(CHOICES)}

INSTRUCTIONS = (
    "You control Mario in a side-scrolling platformer. Pick the single best button press right now. "
    "Goal: collect as many points as possible (coins, question blocks, stomping enemies from above, "
    "grabbing the flagpole as high as possible) without dying, and reach the flagpole and the castle. "
    "Touching an enemy from the side hurts. Falling into a pit kills."
)


def describe(world: GameWorld) -> str:
    """Сцена словами: земля, ямы, препятствия, враги, бонусы, финиш."""

    player = world.player
    tile = rules.TILE
    tx = int(player.center_x // tile)
    feet = int(player.bottom // tile)

    parts = ["Mario is " + ("standing on the ground." if world.on_ground else "in the air.")]

    pit = next((d for d in range(1, 12) if not any(world.solid_at(tx + d, feet - k) for k in range(1, 14))), None)
    if pit is not None:
        width = 1
        while width < 10 and not any(world.solid_at(tx + pit + width, feet - k) for k in range(1, 14)):
            width += 1
        parts.append(f"A pit {width} tiles wide starts {pit} tiles ahead.")

    wall = next((d for d in range(1, 12) if world.solid_at(tx + d, feet)), None)
    if wall is not None:
        height = 1
        while height < 8 and world.solid_at(tx + wall, feet + height):
            height += 1
        parts.append(f"A wall or pipe {height} tiles tall is {wall} tiles ahead.")

    for enemy in sorted(world.enemies, key=lambda e: abs(e.center_x - player.center_x))[:3]:
        if not enemy.is_alive:
            continue
        dx = (enemy.center_x - player.center_x) / tile
        if not -3 <= dx <= 12:
            continue
        kind = "goomba" if isinstance(enemy, Goomba) else ("koopa shell" if getattr(enemy, "is_shell", False) else "koopa")
        side = f"{dx:.1f} tiles ahead" if dx >= 0 else f"{-dx:.1f} tiles behind"
        moving = "moving left" if enemy.change_x < 0 else ("moving right" if enemy.change_x > 0 else "not moving")
        parts.append(f"A {kind} is {side}, {moving}.")

    for obj in world.objects:
        dx = (obj.center_x - player.center_x) / tile
        dy = (obj.center_y - player.center_y) / tile
        if not 0 <= dx <= 8:
            continue
        if isinstance(obj, Coin):
            parts.append(f"A coin is {dx:.0f} tiles ahead and {dy:.0f} tiles up.")
        elif isinstance(obj, ScoreBlock) and obj.state > 0:
            parts.append(f"A question block is {dx:.0f} tiles ahead and {dy:.0f} tiles up.")

    pole = world.geometry.pole
    if pole is not None:
        parts.append(f"The flagpole is {(pole.x - player.center_x) / tile:.0f} tiles ahead.")

    parts.append(f"Time left: {world.time_left:.0f} seconds. Lives: {player.health}.")
    return " ".join(parts)


class LayaController:
    """Контроллер с тем же интерфейсом, что ai.agent.Controller."""

    stochastic = False

    def __init__(self, threads: int = 2):
        import torch
        from laya import Router  # noqa: WPS433 — пакет ставится отдельно

        # Не больше двух потоков: рядом может идти обучение MarioJev
        torch.set_num_threads(threads)

        # preload=True подняло бы все три чекпойнта (2.4 ГБ); нужен один —
        # английский, он грузится лениво при первом вопросе. На видеокарте
        # (GTX 1050 Ti, fp16) решение в десятки раз быстрее, чем на CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.router = Router(preload=False, device=self.device)
        self.name = f"Laya (open Jev, без дообучения, {self.device})"
        self._frame = 0
        self._action = 0
        self._last: Optional[dict] = None
        self.latencies: list[float] = []
        self.choices: dict[str, int] = {}

    def reset(self, world: Optional[GameWorld] = None) -> None:
        self._frame = 0
        self._action = 0

    def act(self, world: GameWorld) -> Inputs:
        if world.done or world.finishing:
            return Inputs()

        first = self._frame % FRAME_SKIP == 0
        if first:
            started = time.perf_counter()
            result = self.router.predict(
                {"body": describe(world)},
                {"move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CHOICES}},
            )
            self.latencies.append((time.perf_counter() - started) * 1000)
            answer = result["answers"]["move"]
            self._action = CHOICE_INDEX.get(answer.get("choice"), 1)
            self._last = answer
            self.choices[answer.get("choice")] = self.choices.get(answer.get("choice"), 0) + 1

        self._frame += 1
        return action_inputs(self._action, first)

    def overlay(self) -> Optional[dict]:
        if self._last is None:
            return None
        return {
            "model": self.name,
            "action": self._last.get("choice"),
            "probs": self._last.get("probabilities", {}),
            "confidence": self._last.get("confidence"),
        }

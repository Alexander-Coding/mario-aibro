"""Простой бот на правилах: бежит вправо, прыгает перед стеной, ямой и
врагом. Нужен как базовая линия турнира и как дымовой тест мира."""

from __future__ import annotations

from core import rules
from core.world import GameWorld, Inputs


class HeuristicBot:

    name = "Эвристика"

    def __init__(self):
        self._cooldown = 0

    def reset(self, world: GameWorld | None = None) -> None:
        self._cooldown = 0

    def overlay(self) -> None:
        return None

    def act(self, world: GameWorld) -> Inputs:
        player = world.player
        tile = rules.TILE

        tx = int(player.center_x // tile)
        foot = int((player.bottom - 1) // tile)

        wall_ahead = any(world.solid_at(tx + d, foot + 1) for d in (1, 2))
        pit_ahead = world.on_ground and not any(
            world.solid_at(tx + d, foot) for d in (1, 2)
        )
        enemy_ahead = any(
            enemy.is_alive
            and 0 < enemy.center_x - player.center_x < 110
            and abs(enemy.center_y - player.center_y) < 60
            for enemy in world.enemies
        )

        jump = False
        if self._cooldown > 0:
            self._cooldown -= 1
        elif world.on_ground and (wall_ahead or pit_ahead or enemy_ahead):
            jump = True
            self._cooldown = 8

        return Inputs(right=True, jump=jump)


def run(level: str = "maps/custom_level.json", max_frames: int = 20000) -> dict:
    world = GameWorld(level)
    bot = HeuristicBot()
    while not world.done and world.frame < max_frames:
        world.step(bot.act(world))
    return world.summary()


if __name__ == "__main__":
    import json, sys, time
    t = time.perf_counter()
    result = run(*(sys.argv[1:2] or []))
    result["fps"] = round(result["frames"] / (time.perf_counter() - t))
    print(json.dumps(result, ensure_ascii=False))

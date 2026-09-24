"""Типизированное состояние для нейросети.

Модель видит не картинку, а то же, что увидел бы внимательный игрок:

- grid     — карта тайлов вокруг Марио (8 каналов × 16 строк × 24 столбца):
             стены, блоки «?» с числом оставшихся ударов, монеты, враги,
             флагшток, замок. 4 тайла назад и 19 вперёд — почти экран.
- enemies  — до 8 ближайших врагов точно, с долями пикселя и скоростью
             (в сетке враг квантуется до тайла, а прыжок на голову требует
             точности);
- player   — скорость, опора, неуязвимость, жизни, время, где флагшток,
             предыдущее действие.

Одна и та же функция строит наблюдение и на обучении, и в живой игре —
иначе модель в окне увидела бы не то, на чём училась.
"""

from __future__ import annotations

import numpy as np

from core import rules
from core.world import FLAGPOLE, TO_CASTLE, GameWorld
from Entities.Enemies.goomba import Goomba
from Entities.Enemies.koopa import Koopa
from Objects.Coin import Coin
from Objects.ScoreBlock import ScoreBlock

TILE = rules.TILE

# ---------------------------------------------------------------
# Окно карты вокруг игрока, в тайлах
# ---------------------------------------------------------------

GRID_BACK = 4       # столбцов позади
GRID_AHEAD = 19     # столбцов впереди (включая столбец игрока — +1)
GRID_BELOW = 7      # строк ниже ног (земля, ямы)
GRID_ABOVE = 8      # строк выше ног (прыжок — 6.8 тайла)

GRID_W = GRID_BACK + GRID_AHEAD + 1
GRID_H = GRID_BELOW + GRID_ABOVE + 1

CHANNELS = (
    "solid",
    "block",      # блок «?»: оставшиеся удары / 4
    "coin",
    "goomba",
    "koopa",
    "shell",
    "pole",
    "castle",
)
N_CHANNELS = len(CHANNELS)

C_SOLID, C_BLOCK, C_COIN, C_GOOMBA, C_KOOPA, C_SHELL, C_POLE, C_CASTLE = range(N_CHANNELS)

# ---------------------------------------------------------------
# Враги
# ---------------------------------------------------------------

MAX_ENEMIES = 8
ENEMY_FEATURES = 9

# ---------------------------------------------------------------
# Действия
# ---------------------------------------------------------------

# (подпись, вправо, влево, прыжок). Прыжок — фронт нажатия на первом
# кадре повтора, как одно нажатие клавиши человеком.
ACTIONS: tuple[tuple[str, bool, bool, bool], ...] = (
    ("стоять", False, False, False),
    ("вправо", True, False, False),
    ("влево", False, True, False),
    ("прыжок", False, False, True),
    ("вправо+прыжок", True, False, True),
    ("влево+прыжок", False, True, True),
)
N_ACTIONS = len(ACTIONS)
ACTION_LABELS = tuple(label for label, *_ in ACTIONS)

PLAYER_FEATURES = 17 + N_ACTIONS


def observation_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "grid": (N_CHANNELS, GRID_H, GRID_W),
        "enemies": (MAX_ENEMIES, ENEMY_FEATURES),
        "enemy_mask": (MAX_ENEMIES,),
        "player": (PLAYER_FEATURES,),
    }


class FeatureBuilder:
    """Строит наблюдение для одного мира. Статика уровня кэшируется."""

    def __init__(self, world: GameWorld):
        self.world = world
        self._pad = GRID_BACK + GRID_AHEAD + 8

        # Сетка стен с полями: за краями уровня — пусто. Слева от уровня
        # для игрока нет стены — это пропасть (он уходит за x<0 и падает),
        # и модель должна видеть там пропасть, а не опору.
        # Индексация: padded[tx + pad, ty + pad].
        grid = world.tile_grid.astype(np.float32)
        pad = self._pad
        padded = np.zeros(
            (grid.shape[0] + 2 * pad, grid.shape[1] + 2 * pad), dtype=np.float32
        )
        padded[pad:pad + grid.shape[0], pad:pad + grid.shape[1]] = grid
        self._solid = padded

        # Статические слои финиша
        self._static = np.zeros((2,) + padded.shape, dtype=np.float32)
        geometry = world.geometry

        if geometry.pole is not None:
            pole = geometry.pole
            tx = int(pole.x // TILE) + pad
            ty0 = int(pole.bottom // TILE) + pad
            ty1 = int((pole.top - 1) // TILE) + pad
            if 0 <= tx < padded.shape[0]:
                self._static[0, tx, max(ty0, 0):min(ty1 + 1, padded.shape[1])] = 1.0

        if geometry.castle is not None:
            castle = geometry.castle
            tx0 = int(castle.left // TILE) + pad
            tx1 = int((castle.right - 1) // TILE) + pad
            ty0 = int(castle.bottom // TILE) + pad
            ty1 = int((castle.top - 1) // TILE) + pad
            self._static[1, max(tx0, 0):tx1 + 1, max(ty0, 0):ty1 + 1] = 1.0

    # -----------------------------------------------------------

    def build(self, last_action: int = 0) -> dict[str, np.ndarray]:
        world = self.world
        player = world.player
        pad = self._pad

        cx = int(player.center_x // TILE)
        feet = int(player.bottom // TILE)

        x0 = cx - GRID_BACK
        y0 = feet - GRID_BELOW

        grid = np.zeros((N_CHANNELS, GRID_H, GRID_W), dtype=np.float32)

        # --- стены и финиш: срез из кэша -----------------------------
        xs = slice(x0 + pad, x0 + pad + GRID_W)
        ys = slice(y0 + pad, y0 + pad + GRID_H)

        solid = self._window(self._solid, xs, ys)
        grid[C_SOLID] = solid.T
        grid[C_POLE] = self._window(self._static[0], xs, ys).T
        grid[C_CASTLE] = self._window(self._static[1], xs, ys).T

        # --- монеты и блоки ------------------------------------------
        for obj in world.objects:
            gx = int(obj.center_x // TILE) - x0
            gy = int(obj.center_y // TILE) - y0
            if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
                continue
            if isinstance(obj, Coin):
                grid[C_COIN, gy, gx] = 1.0
            elif isinstance(obj, ScoreBlock):
                grid[C_BLOCK, gy, gx] = max(obj.state, 0) / 4.0

        # --- враги ---------------------------------------------------
        enemies = np.zeros((MAX_ENEMIES, ENEMY_FEATURES), dtype=np.float32)
        mask = np.zeros(MAX_ENEMIES, dtype=np.float32)

        nearby = []
        for enemy in world.enemies:
            if not enemy.is_alive:
                continue
            gx = int(enemy.center_x // TILE) - x0
            gy = int(enemy.center_y // TILE) - y0
            if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
                continue

            shell = bool(getattr(enemy, "is_shell", False))
            if isinstance(enemy, Goomba):
                channel = C_GOOMBA
            elif shell:
                channel = C_SHELL
            else:
                channel = C_KOOPA
            grid[channel, gy, gx] = 1.0

            nearby.append((abs(enemy.center_x - player.center_x), enemy, shell))

        nearby.sort(key=lambda item: item[0])

        for slot, (_, enemy, shell) in enumerate(nearby[:MAX_ENEMIES]):
            dx = enemy.center_x - player.center_x
            dy = enemy.center_y - player.center_y
            enemies[slot] = (
                1.0 if isinstance(enemy, Goomba) else 0.0,
                1.0 if isinstance(enemy, Koopa) and not shell else 0.0,
                1.0 if shell else 0.0,
                dx / 320.0,
                dy / 320.0,
                enemy.change_x / 4.0,
                enemy.change_y / 10.0,
                getattr(enemy, "shell_timer", 0.0) / 5.0,
                1.0 if dx > 0 else 0.0,
            )
            mask[slot] = 1.0

        # --- игрок ---------------------------------------------------
        pole = world.geometry.pole
        if pole is not None:
            to_pole = (pole.x - player.center_x) / TILE
            climb = (pole.top - player.bottom) / pole.height
        else:
            to_pole = (world.geometry.door_x - player.center_x) / TILE
            climb = 0.0

        vector = np.zeros(PLAYER_FEATURES, dtype=np.float32)
        vector[:17] = (
            player.change_x / 4.0,
            player.change_y / 15.0,
            1.0 if world.on_ground else 0.0,
            1.0 if player.invincible_timer > 0 else 0.0,
            player.health / 3.0,
            world.time_left / max(world.time_limit, 1.0),
            min(world.time_left, 300.0) / 300.0,
            (player.center_x % TILE) / TILE,
            (player.bottom % TILE) / TILE,
            float(np.clip(to_pole / 64.0, -1.0, 1.0)),
            float(np.clip(to_pole / 8.0, -1.0, 1.0)),
            float(np.clip(climb, -1.0, 1.0)),
            1.0 if world.phase in (FLAGPOLE, TO_CASTLE) else 0.0,
            world.progress,
            float(np.clip(player.center_y / 640.0, -1.0, 2.0)),
            1.0 if player.jump_requested else 0.0,
            float(np.clip(player.center_x / 8000.0, 0.0, 2.0)),
        )
        vector[17 + int(last_action)] = 1.0

        return {
            "grid": grid,
            "enemies": enemies,
            "enemy_mask": mask,
            "player": vector,
        }

    @staticmethod
    def _window(array: np.ndarray, xs: slice, ys: slice) -> np.ndarray:
        """Срез с защитой от выхода за поля (очень длинный прыжок вверх)."""

        out = np.zeros((GRID_W, GRID_H), dtype=np.float32)
        sx0, sx1 = max(xs.start, 0), min(xs.stop, array.shape[0])
        sy0, sy1 = max(ys.start, 0), min(ys.stop, array.shape[1])
        if sx1 > sx0 and sy1 > sy0:
            out[sx0 - xs.start:sx1 - xs.start, sy0 - ys.start:sy1 - ys.start] = array[sx0:sx1, sy0:sy1]
        return out


def ascii_grid(obs: dict[str, np.ndarray]) -> str:
    """Отладка: как модель видит мир (верх — сверху)."""

    grid = obs["grid"]
    symbols = {C_SOLID: "#", C_BLOCK: "?", C_COIN: "o", C_GOOMBA: "g", C_KOOPA: "k", C_SHELL: "s", C_POLE: "|", C_CASTLE: "C"}
    rows = []
    for gy in range(GRID_H - 1, -1, -1):
        row = []
        for gx in range(GRID_W):
            ch = "."
            for channel in (C_CASTLE, C_POLE, C_SOLID, C_COIN, C_BLOCK, C_SHELL, C_KOOPA, C_GOOMBA):
                if grid[channel, gy, gx] > 0:
                    ch = symbols[channel]
            if gx == GRID_BACK and gy == GRID_BELOW:
                ch = "M"
            row.append(ch)
        rows.append("".join(row))
    return "\n".join(rows)

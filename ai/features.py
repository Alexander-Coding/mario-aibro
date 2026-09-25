"""Типизированное состояние для нейросети.

Модель видит не картинку, а то же, что увидел бы внимательный игрок:

- grid     — карта тайлов вокруг Марио (9 каналов × 16 строк × 28 столбцов):
             стены (блоки «?» тоже твёрдые), блоки «?» с числом оставшихся
             ударов, монеты, враги, флагшток, замок, трубы в секретные
             комнаты и выход из комнаты. 8 тайлов назад и 19 вперёд — почти
             экран вперёд и полэкрана назад: монета или блок «?», пролетевшие
             мимо, остаются видны, и за ними можно вернуться.
- enemies  — до 8 ближайших врагов точно, с долями пикселя и скоростью
             (в сетке враг квантуется до тайла, а прыжок на голову требует
             точности);
- player   — скорость, опора, неуязвимость, жизни, время, где флагшток,
             предыдущее действие; в комнате ли, стоит ли на трубе-входе,
             где ближайшая непосещённая труба-вход (или выход из комнаты).

Новые каналы и признаки дописаны В КОНЕЦ: наблюдение для прошлых моделей
(8 каналов, 6 действий) — это префикс нового (legacy_view). Окно карты
расширялось только назад: прошлые модели (24 столбца, 4 назад) видят
вырезку из середины нового окна, в точности свою прежнюю карту.

Одна и та же функция строит наблюдение и на обучении, и в живой игре —
иначе модель в окне увидела бы не то, на чём училась.
"""

from __future__ import annotations

import numpy as np

from core import rules
from core.world import FLAGPOLE, TO_CASTLE, Area, GameWorld
from Entities.Enemies.goomba import Goomba
from Entities.Enemies.koopa import Koopa
from Objects.Coin import Coin

TILE = rules.TILE

# ---------------------------------------------------------------
# Окно карты вокруг игрока, в тайлах
# ---------------------------------------------------------------

# Позади было 4 столбца: монета или блок «?» в 5 тайлах за спиной для
# ученика не существовали, и вернуться за ними он не мог. Планировщик
# (ai/planner.py) возвращается только за тем, что видно в GRID_BACK
GRID_BACK = 8       # столбцов позади
GRID_AHEAD = 19     # столбцов впереди (включая столбец игрока — +1)
GRID_BELOW = 7      # строк ниже ног (земля, ямы)
GRID_ABOVE = 8      # строк выше ног (прыжок — 6.8 тайла)

GRID_W = GRID_BACK + GRID_AHEAD + 1     # 28: делится на патчи 4×4 (4×7) и 4×2 (4×14)
GRID_H = GRID_BELOW + GRID_ABOVE + 1

# Окно всех моделей, обученных до расширения (у чекпойнта нет grid_w)
LEGACY_GRID_BACK = 4
LEGACY_GRID_W = LEGACY_GRID_BACK + GRID_AHEAD + 1

CHANNELS = (
    "solid",
    "block",      # блок «?»: оставшиеся удары / 4
    "coin",
    "goomba",
    "koopa",
    "shell",
    "pole",
    "castle",
    "warp",       # верх непосещённой трубы-входа; в комнате — устье выхода
)
N_CHANNELS = len(CHANNELS)

C_SOLID, C_BLOCK, C_COIN, C_GOOMBA, C_KOOPA, C_SHELL, C_POLE, C_CASTLE, C_WARP = range(N_CHANNELS)

# ---------------------------------------------------------------
# Враги
# ---------------------------------------------------------------

MAX_ENEMIES = 8
ENEMY_FEATURES = 9
# Точный список врагов — в прежнем окне (4 столбца позади), хотя сетка
# теперь видит 8: для прыжка на голову нужен враг рядом, а сзади в 5+
# тайлах он виден в канале сетки. Иначе дальние враги позади вытесняли
# бы из 8 слотов ближних впереди, и прошлые модели получали бы другой
# список, чем на обучении
ENEMY_BACK = LEGACY_GRID_BACK

# ---------------------------------------------------------------
# Действия
# ---------------------------------------------------------------

# (подпись, вправо, влево, прыжок, вниз). Прыжок — фронт нажатия на первом
# кадре повтора, как одно нажатие клавиши человеком. «Вниз» — последним:
# у прошлых моделей его не было, их действия — префикс списка.
ACTIONS: tuple[tuple[str, bool, bool, bool, bool], ...] = (
    ("стоять", False, False, False, False),
    ("вправо", True, False, False, False),
    ("влево", False, True, False, False),
    ("прыжок", False, False, True, False),
    ("вправо+прыжок", True, False, True, False),
    ("влево+прыжок", False, True, True, False),
    ("вниз", False, False, False, True),
)
N_ACTIONS = len(ACTIONS)
ACTION_LABELS = tuple(label for label, *_ in ACTIONS)

PLAYER_BASE = 17        # признаки игрока до one-hot прошлого действия
PLAYER_EXTRA = 4        # после него: в комнате, на трубе-входе, есть цель, до цели
PLAYER_FEATURES = PLAYER_BASE + N_ACTIONS + PLAYER_EXTRA


def legacy_view(obs: dict, channels: int, player_features: int,
                grid_w: int = GRID_W, grid_back: int = GRID_BACK) -> dict:
    """Наблюдение для модели прошлой версии: первые channels каналов,
    первые player_features признаков игрока (база + one-hot её действий) и
    её окно карты — grid_w столбцов, из них grid_back позади Марио.

    Окно расширялось только назад, поэтому старое — это столбцы
    GRID_BACK − grid_back … GRID_BACK − grid_back + grid_w нового, те же
    числа, что модель видела при обучении. Работает и на батче ([..., C, H, W])."""

    if (channels, player_features, grid_w, grid_back) == (N_CHANNELS, PLAYER_FEATURES, GRID_W, GRID_BACK):
        return obs
    first = GRID_BACK - grid_back
    if first < 0 or first + grid_w > GRID_W:
        raise ValueError(f"окно модели ({grid_w} столбцов, {grid_back} позади) не помещается "
                         f"в окно признаков ({GRID_W}, {GRID_BACK} позади)")
    view = dict(obs)
    view["grid"] = obs["grid"][..., :channels, :, first:first + grid_w]
    view["player"] = obs["player"][..., :player_features]
    return view


def model_view(obs: dict, model) -> dict:
    """legacy_view по размерам модели (MarioJev): так её и кормить везде —
    в игре, на валидации, при калибровке и в DAgger."""

    if model.grid_h != GRID_H:
        raise ValueError(f"модель учена на {model.grid_h} строках карты, признаки дают {GRID_H}")
    return legacy_view(obs, model.channels, model.player_features, model.grid_w, model.grid_back)


def observation_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "grid": (N_CHANNELS, GRID_H, GRID_W),
        "enemies": (MAX_ENEMIES, ENEMY_FEATURES),
        "enemy_mask": (MAX_ENEMIES,),
        "player": (PLAYER_FEATURES,),
    }


class FeatureBuilder:
    """Строит наблюдение для одного мира. Статика каждой части уровня
    (сам уровень, секретные комнаты) кэшируется при первом заходе."""

    def __init__(self, world: GameWorld):
        self.world = world
        self._pad = GRID_BACK + GRID_AHEAD + 8
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _area_static(self, area: Area) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(area.key)
        if cached is not None:
            return cached

        world = self.world
        pad = self._pad

        # Сетка стен с полями: за краями уровня — пусто. Слева от уровня
        # для игрока нет стены — это пропасть (он уходит за x<0 и падает),
        # и модель должна видеть там пропасть, а не опору.
        # Индексация: padded[tx + pad, ty + pad].
        grid = area.tile_grid.astype(np.float32)
        padded = np.zeros(
            (grid.shape[0] + 2 * pad, grid.shape[1] + 2 * pad), dtype=np.float32
        )
        padded[pad:pad + grid.shape[0], pad:pad + grid.shape[1]] = grid

        # Статические слои финиша — только у самого уровня
        static = np.zeros((2,) + padded.shape, dtype=np.float32)
        geometry = world.geometry

        if area is world.main and geometry.pole is not None:
            pole = geometry.pole
            tx = int(pole.x // TILE) + pad
            ty0 = int(pole.bottom // TILE) + pad
            ty1 = int((pole.top - 1) // TILE) + pad
            if 0 <= tx < padded.shape[0]:
                static[0, tx, max(ty0, 0):min(ty1 + 1, padded.shape[1])] = 1.0

        if area is world.main and geometry.castle is not None:
            castle = geometry.castle
            tx0 = int(castle.left // TILE) + pad
            tx1 = int((castle.right - 1) // TILE) + pad
            ty0 = int(castle.bottom // TILE) + pad
            ty1 = int((castle.top - 1) // TILE) + pad
            static[1, max(tx0, 0):tx1 + 1, max(ty0, 0):ty1 + 1] = 1.0

        self._cache[area.key] = (padded, static)
        return padded, static

    def _targets(self) -> list[tuple[float, float, float]]:
        """Куда стоит сходить: непосещённые трубы-входы или устье выхода.

        (x середины, y низа тайла-метки, ширина в тайлах) — метка в канале
        warp рисуется на тайлах верха трубы / устья."""

        world = self.world
        area = world.area
        if area is world.main:
            return [
                (warp.x, warp.top - TILE, 2)
                for warp in area.warps
                if warp.room not in world.rooms_visited
            ]
        if area.exit is not None:
            return [(area.exit.left + TILE / 2, area.exit.bottom, 1)]
        return []

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

        solid_layer, static = self._area_static(world.area)
        grid[C_SOLID] = self._window(solid_layer, xs, ys).T
        grid[C_POLE] = self._window(static[0], xs, ys).T
        grid[C_CASTLE] = self._window(static[1], xs, ys).T

        # --- монеты и блоки ------------------------------------------
        for obj in world.objects:
            if not isinstance(obj, Coin):
                continue
            gx = int(obj.center_x // TILE) - x0
            gy = int(obj.center_y // TILE) - y0
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                grid[C_COIN, gy, gx] = 1.0

        for block in world.blocks:
            gx = int(block.center_x // TILE) - x0
            gy = int(block.center_y // TILE) - y0
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                grid[C_BLOCK, gy, gx] = max(block.state, 0) / 4.0

        # --- трубы в комнаты и выход из комнаты -----------------------
        targets = self._targets()
        for tx_center, bottom, width in targets:
            gy = int((bottom + TILE / 2) // TILE) - y0
            first = int((tx_center - width * TILE / 2 + TILE / 2) // TILE) - x0
            for gx in range(first, first + width):
                if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                    grid[C_WARP, gy, gx] = 1.0

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

            if gx >= GRID_BACK - ENEMY_BACK:
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
        if world.in_room:
            # В комнате свои координаты: до флагштока отсюда не померить
            to_pole = 0.0
            climb = 0.0
        elif pole is not None:
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
        vector[PLAYER_BASE + int(last_action)] = 1.0

        # --- секретные комнаты ----------------------------------------
        nearest = min(targets, key=lambda t: abs(t[0] - player.center_x), default=None)
        extra = PLAYER_BASE + N_ACTIONS
        vector[extra] = 1.0 if world.in_room else 0.0
        # Стоит на трубе, за которой ещё не был: «вниз» ведёт к монетам
        warp = world.warp_under_player()
        vector[extra + 1] = 1.0 if warp is not None and warp.room not in world.rooms_visited else 0.0
        if nearest is not None:
            vector[extra + 2] = 1.0
            vector[extra + 3] = float(np.clip((nearest[0] - player.center_x) / (32.0 * TILE), -1.0, 1.0))

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
    """Отладка: как модель видит мир (верх — сверху). Понимает и полное
    окно, и вырезку legacy_view для моделей прошлой ширины."""

    grid = obs["grid"]
    width = grid.shape[-1]
    back = GRID_BACK - (GRID_W - width)
    symbols = {C_SOLID: "#", C_BLOCK: "?", C_COIN: "o", C_GOOMBA: "g", C_KOOPA: "k", C_SHELL: "s", C_POLE: "|",
               C_CASTLE: "C", C_WARP: "W"}
    rows = []
    for gy in range(GRID_H - 1, -1, -1):
        row = []
        for gx in range(width):
            ch = "."
            for channel in (C_CASTLE, C_POLE, C_SOLID, C_WARP, C_COIN, C_BLOCK, C_SHELL, C_KOOPA, C_GOOMBA):
                if grid[channel, gy, gx] > 0:
                    ch = symbols[channel]
            if gx == back and gy == GRID_BELOW:
                ch = "M"
            row.append(ch)
        rows.append("".join(row))
    return "\n".join(rows)

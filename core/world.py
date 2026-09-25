"""Мир игры без окна.

Вся игровая логика одного уровня: физика, враги, очки, таймер, флагшток и
вход в замок. Живая игра (GameView) только рисует этот мир и передаёт в
него нажатия, среда обучения передаёт в него решения нейросети. Правила
одни и те же, поэтому модель в окне ведёт себя так же, как на обучении.

    world = GameWorld("maps/custom_level.json")
    while not world.done:
        world.step(Inputs(right=True, jump=...))
    print(world.score, world.won)

Уровень может прятать секретные комнаты: труба-вход («вниз», стоя на ней)
ведёт в комнату, из комнаты выходят в боковую трубу и вылезают из
другой трубы дальше по уровню. Уровень и каждая комната — это Area со
своими стенами, объектами и врагами; world.walls, world.enemies и т. п.
всегда отдают ту часть, где сейчас игрок.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import arcade
import numpy as np

from core import rules
from core import snapshot as _snapshot
from Entities.Enemies.koopa import Koopa
from Entities.player import Player
from map import BASE_DIR, Map
from Objects.Coin import Coin
from Objects.Flag import Flag
from Objects.ScoreBlock import ScoreBlock

LevelSource = Union[str, Path, dict]

# Фазы уровня
PLAYING = "playing"
FLAGPOLE = "flagpole"
TO_CASTLE = "to_castle"
COMPLETE = "complete"
DEAD = "dead"

# Трубы: уход вниз в трубу-вход, уход вбок в трубу выхода из комнаты,
# подъём из трубы-выхода. Управление в это время у игры
PIPE_DOWN = "pipe_down"
PIPE_SIDE = "pipe_side"
PIPE_UP = "pipe_up"
PIPE_PHASES = (PIPE_DOWN, PIPE_SIDE, PIPE_UP)


class Inputs(NamedTuple):
    """Управление на один кадр.

    right/left — клавиша удерживается; jump — клавишу прыжка нажали в этом
    кадре (фронт нажатия, как on_key_press), а не держат; down — «вниз»
    удерживается (стоя на трубе-входе — уйти в неё).
    """

    right: bool = False
    left: bool = False
    jump: bool = False
    down: bool = False


@dataclass
class Event:
    """Что произошло за кадр. По событиям считается награда и рисуется UI."""

    kind: str
    points: int = 0
    info: dict = field(default_factory=dict)


# ===============================================================
# ГЕОМЕТРИЯ ФИНИША
# ===============================================================


@dataclass
class Flagpole:
    x: float
    bottom: float
    top: float

    @property
    def height(self) -> float:
        return max(self.top - self.bottom, 1.0)


@dataclass
class Castle:
    left: float
    right: float
    bottom: float
    top: float

    @property
    def door_x(self) -> float:
        return (self.left + self.right) / 2


@dataclass
class LevelGeometry:
    pole: Optional[Flagpole]
    castle: Optional[Castle]
    door_x: float
    left: float
    right: float

    @property
    def width(self) -> float:
        return self.right - self.left


def _resource(item: dict) -> str:
    return str(item.get("resource") or item.get("name") or "").strip().lower()


def _category(item: dict) -> str:
    return str(item.get("category") or "").strip().lower()


def build_geometry(data: dict) -> LevelGeometry:
    """Ищет в данных уровня флагшток и замок.

    Флагшток — столбец декораций «Flag stick», замок — декорации «Castle».
    Если флагштока нет, но есть объект Flag, шток считается справа от
    флага. Если нет ничего, финишем считается правый край уровня.
    """

    objects = data.get("objects", [])
    half = rules.TILE / 2

    xs = [float(o["x"]) for o in objects if "x" in o]
    left = (min(xs) - half) if xs else 0.0
    right = (max(xs) + half) if xs else 3000.0

    sticks = [o for o in objects if _resource(o) in ("flag stick", "flagstick", "flag_stick")]
    flags = [o for o in objects if _resource(o) == "flag"]
    castle_tiles = [o for o in objects if _resource(o) == "castle"]

    pole: Optional[Flagpole] = None

    if sticks:
        column = Counter(float(o["x"]) for o in sticks).most_common(1)[0][0]
        ys = [float(o["y"]) for o in sticks if float(o["x"]) == column]
        pole = Flagpole(x=column, bottom=min(ys) - half, top=max(ys) + half)

    elif flags:
        flag = flags[0]
        # Шток без декораций: от земли под флагом до флага
        pole = Flagpole(
            x=float(flag["x"]) + rules.TILE,
            bottom=float(flag["y"]) - 12 * rules.TILE,
            top=float(flag["y"]) + half,
        )

    castle: Optional[Castle] = None

    if castle_tiles:
        castle = Castle(
            left=min(float(o["x"]) for o in castle_tiles) - half,
            right=max(float(o["x"]) for o in castle_tiles) + half,
            bottom=min(float(o["y"]) for o in castle_tiles) - half,
            top=max(float(o["y"]) for o in castle_tiles) + half,
        )

    if castle is not None:
        door_x = castle.door_x
    elif pole is not None:
        door_x = pole.x + 6 * rules.TILE
    else:
        door_x = right - rules.TILE

    return LevelGeometry(pole=pole, castle=castle, door_x=door_x, left=left, right=right)


# ===============================================================
# ЗАГРУЗКА ДАННЫХ УРОВНЯ
# ===============================================================


def load_level_data(level: LevelSource) -> dict:
    """Путь к JSON или уже готовый словарь уровня → словарь."""

    if isinstance(level, dict):
        return level

    path = Path(level)
    if not path.is_absolute():
        path = BASE_DIR / path

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _tiles_overlapping(left: float, right: float, bottom: float, top: float):
    """Тайлы, которые пересекает прямоугольник (касание краем не считается)."""

    tile = rules.TILE
    tx0 = int(left // tile)
    tx1 = int((right - 1e-6) // tile)
    ty0 = int(bottom // tile)
    ty1 = int((top - 1e-6) // tile)

    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            yield tx, ty


class TileBody:
    """Физика врага по сетке тайлов.

    Повторяет PhysicsEnginePlatformer для статичных квадратных стен:
    гравитация, шаг по Y с прилипанием к краю блока, шаг по X с упором
    в стену. Arcade делает то же через проверку полигонов, выталкивая
    спрайт шагами по 0.25 px и пробуя «въехать на пандус» перед каждой
    стеной — на уровне 1-1 это 1.7 мс на врага за кадр против
    ~0.1 мс у игрока. Для игрока оставлен движок arcade: так управление
    ощущается ровно как в исходной игре.
    """

    def __init__(self, sprite, world: "GameWorld", gravity: float):
        self.sprite = sprite
        self.world = world
        self.gravity = gravity
        self._extents_cache: dict = {}

        # Проверка «стена впереди» в Enemy.update — тоже по сетке
        sprite.wall_probe = self.wall_ahead

    def _extents(self) -> tuple[float, float, float]:
        """Полуширина, низ и верх хитбокса относительно центра.

        Свойства left/right/bottom/top у arcade каждый раз пересчитывают
        точки хитбокса; габариты меняются только со сменой текстуры
        (у Koopa в панцире другой рост), поэтому кэш по текстуре.
        """
        sprite = self.sprite
        key = (id(sprite.texture), sprite.scale_y)
        extents = self._extents_cache.get(key)
        if extents is None:
            extents = (
                (sprite.right - sprite.left) / 2,
                sprite.center_y - sprite.bottom,
                sprite.top - sprite.center_y,
            )
            self._extents_cache[key] = extents
        return extents

    def _hits_at(self, x: float, y: float) -> list[tuple[int, int]]:
        half_w, down, up = self._extents()
        solid = self.world.solid_at
        return [
            (tx, ty)
            for tx, ty in _tiles_overlapping(x - half_w, x + half_w, y - down, y + up)
            if tx >= 0 and solid(tx, ty)
        ]

    def wall_ahead(self) -> bool:
        # Enemy.update и Koopa.update уже сдвинули спрайт на шаг вперёд
        # перед вызовом — проверяем там, где он сейчас
        sprite = self.sprite
        return bool(self._hits_at(sprite.center_x, sprite.center_y))

    def update(self) -> None:
        sprite = self.sprite
        tile = rules.TILE

        half_w, down, up = self._extents()

        x = sprite.center_x
        y = sprite.center_y + sprite.change_y - self.gravity
        change_y = sprite.change_y - self.gravity

        hits = self._hits_at(x, y)
        if hits:
            if change_y > 0:
                y = min(ty for _, ty in hits) * tile - up
            elif change_y < 0:
                y = (max(ty for _, ty in hits) + 1) * tile + down
            change_y = 0.0

        y = round(y, 2)

        if sprite.change_x:
            x += sprite.change_x
            hits = self._hits_at(x, y)
            if hits:
                if sprite.change_x > 0:
                    x = min(tx for tx, _ in hits) * tile - half_w
                else:
                    x = (max(tx for tx, _ in hits) + 1) * tile + half_w

        sprite.change_y = change_y
        sprite.position = (x, y)


class _QuietMap:
    """Map пишет в лог и в stdout на каждый спрайт — при обучении это
    тысячи строк на эпизод."""

    def __enter__(self):
        self._disabled = logging.root.manager.disable
        logging.disable(logging.INFO)
        self._stdout = contextlib.redirect_stdout(io.StringIO())
        self._stdout.__enter__()
        return self

    def __exit__(self, *exc):
        self._stdout.__exit__(*exc)
        logging.disable(self._disabled)
        return False


# ===============================================================
# ТРУБЫ И КОМНАТЫ
# ===============================================================


def _point(value) -> tuple[float, float]:
    """[x, y] или {"x": .., "y": ..} → (x, y)."""
    if isinstance(value, dict):
        return float(value["x"]), float(value["y"])
    x, y = value
    return float(x), float(y)


@dataclass
class Warp:
    """Труба-вход в секретную комнату и труба, из которой потом вылезти.

    В JSON уровня: "warps": [{"pipe": [x, y], "exit": [x, y], "room": {...}}],
    где pipe и exit — центр ЛЕВОГО ВЕРХНЕГО тайла трубы (как их кладёт
    редактор), room — комната в формате редактора плюс "exit": [x, y] —
    центр НИЖНЕГО тайла устья боковой трубы, в которую уходят из комнаты.
    """

    x: float            # середина трубы-входа
    top: float          # её верх — на нём стоит игрок
    room: str           # ключ комнаты в world.areas
    exit_x: float       # середина трубы-выхода в уровне
    exit_top: float     # её верх


@dataclass
class RoomExit:
    """Устье боковой трубы в комнате: упёрся в него, идя вправо, — выход."""

    left: float
    bottom: float
    top: float
    warp: Warp


def _build_tile_grid(walls) -> np.ndarray:
    """Статическая сетка твёрдых блоков: solid[tx, ty] = 1.

    Стены не разрушаются (пустой блок «?» — тоже стена), поэтому сетка
    строится один раз на часть уровня.
    """

    tiles = [
        (int(wall.center_x // rules.TILE), int(wall.center_y // rules.TILE))
        for wall in walls
    ]

    max_tx = max((t[0] for t in tiles), default=0)
    max_ty = max((t[1] for t in tiles), default=0)

    grid = np.zeros((max_tx + 64, max(max_ty, 24) + 16), dtype=np.uint8)

    for tx, ty in tiles:
        if tx >= 0 and ty >= 0:
            grid[tx, ty] = 1

    return grid


def _walls_on_grid(walls) -> bool:
    half = rules.TILE / 2
    for wall in walls:
        if abs(wall.width - rules.TILE) > 0.5 or abs(wall.height - rules.TILE) > 0.5:
            return False
        if (wall.center_x - half) % rules.TILE or (wall.center_y - half) % rules.TILE:
            return False
    return True


class Area:
    """Уровень или секретная комната: свои стены, объекты, враги.

    Блоки «?» лежат и в walls (они твёрдые — об них бьётся физика и
    игрока, и врагов), и в blocks — для ударов головой и для признаков
    нейросети. В objects остаются только монеты и флаг.
    """

    def __init__(self, world: "GameWorld", key: str, data: dict, *, kind: str, theme: str):
        self.key = key
        self.kind = kind
        self.theme = theme

        # Map может дописывать в словарь — отдаём копию
        with _QuietMap():
            self.map = Map(data=copy.deepcopy(data))

        self.walls = self.map.get_walls()
        self.objects = self.map.get_objects()

        self.blocks: list[ScoreBlock] = []
        for obj in list(self.objects):
            if isinstance(obj, ScoreBlock):
                obj.remove_from_sprite_lists()
                self.walls.append(obj)
                self.blocks.append(obj)

        # Без пространственного хэша физика проверяет каждый блок уровня
        # на каждом кадре: 2.9 тыс. кадров/с против 27 тыс. на уровне 1-1
        self.walls.enable_spatial_hashing()

        self.decorations = self.map.get_decorations()
        self.enemies = self.map.get_enemies()

        spawn_x, spawn_y = self.map.get_player_spawn()
        self.spawn = (float(spawn_x), float(spawn_y))

        self.tile_grid = _build_tile_grid(self.walls)

        # Враги — по тайлам, если все стены лежат ровной сеткой 32×32.
        # Иначе (нестандартные блоки из редактора) — движок arcade.
        self.tile_physics = _walls_on_grid(self.walls)

        self.enemy_engines: dict[Any, Any] = {
            enemy: (
                TileBody(enemy, world, enemy.gravity)
                if self.tile_physics
                else arcade.PhysicsEnginePlatformer(
                    player_sprite=enemy,
                    walls=self.walls,
                    gravity_constant=enemy.gravity,
                )
            )
            for enemy in self.enemies
        }

        self.active_enemies: set[int] = set()
        if not world.activate_enemies:
            self.active_enemies = {id(enemy) for enemy in self.enemies}

        # Движок игрока — свой на каждую часть: у неё свои стены
        self.physics_engine = arcade.PhysicsEnginePlatformer(
            player_sprite=world.player,
            walls=self.walls,
            gravity_constant=world.player.gravity,
        )

        self.warps: list[Warp] = []
        self.exit: Optional[RoomExit] = None

        # Границы части — для камеры
        sprites = list(self.walls) + list(self.objects) + list(self.decorations)
        if sprites:
            self.left = min(s.left for s in sprites)
            self.right = max(s.right for s in sprites)
            self.bottom = min(s.bottom for s in sprites)
            self.top = max(s.top for s in sprites)
        else:
            self.left, self.right, self.bottom, self.top = 0.0, 3000.0, 0.0, 600.0


# ===============================================================
# МИР
# ===============================================================


class GameWorld:
    """Один уровень игры. Шаг — ровно один кадр по 1/60 секунды."""

    def __init__(
        self,
        level: LevelSource,
        *,
        name: Optional[str] = None,
        time_limit: Optional[float] = None,
        activate_enemies: bool = True,
    ):
        self.level_data = load_level_data(level)
        self.level_path = None if isinstance(level, dict) else str(level)

        self.name = name or self.level_data.get("name") or (
            Path(self.level_path).stem if self.level_path else "Уровень"
        )

        self.time_limit = float(
            time_limit
            if time_limit is not None
            else self.level_data.get("time_limit", rules.DEFAULT_TIME_LIMIT)
        )

        self.activate_enemies = activate_enemies

        self.geometry = build_geometry(self.level_data)

        # Заполняется в reset()
        self.player: Optional[Player] = None
        self.main: Optional[Area] = None
        self.area: Optional[Area] = None
        self.areas: dict[str, Area] = {}
        self.warps: list[Warp] = []

        self.reset()

    # ===========================================================
    # RESET
    # ===========================================================

    def reset(self) -> None:
        """Пересобирает уровень с нуля."""

        self.player = Player()

        main_data = {k: v for k, v in self.level_data.items() if k != "warps"}
        self.main = Area(
            self, "main", main_data,
            kind="main", theme=str(self.level_data.get("theme", "overworld")),
        )
        self.areas = {"main": self.main}
        self.warps = []

        half = rules.TILE / 2
        for index, spec in enumerate(self.level_data.get("warps") or []):
            key = str(spec.get("id") or f"room{index + 1}")
            room_data = spec["room"]
            room = Area(
                self, key, room_data,
                kind="room", theme=str(room_data.get("theme", "underground")),
            )

            pipe_x, pipe_y = _point(spec["pipe"])
            exit_x, exit_y = _point(spec["exit"])
            warp = Warp(
                x=pipe_x + half,
                top=pipe_y + half,
                room=key,
                exit_x=exit_x + half,
                exit_top=exit_y + half,
            )
            self.main.warps.append(warp)
            self.warps.append(warp)

            if room_data.get("exit") is not None:
                mouth_x, mouth_y = _point(room_data["exit"])
                room.exit = RoomExit(
                    left=mouth_x - half,
                    bottom=mouth_y - half,
                    top=mouth_y + half + rules.TILE,
                    warp=warp,
                )

            self.areas[key] = room

        self.area = self.main

        self.spawn = self.main.spawn
        self.player.center_x, self.player.center_y = self.spawn

        self.flag_sprite: Optional[Flag] = next(
            (obj for obj in self.main.objects if isinstance(obj, Flag)), None
        )

        # Сколько всего можно взять — для итогов «сколько собрал из возможного»
        areas = list(self.areas.values())
        self.totals = {
            "coins": sum(isinstance(obj, Coin) for area in areas for obj in area.objects),
            "block_hits": sum(block.initial_hits for area in areas for block in area.blocks),
            "enemies": sum(len(area.enemies) for area in areas),
            "secrets": len(self.warps),
        }

        # Состояние
        self.phase = PLAYING
        self.frame = 0
        self.elapsed = 0.0
        # Таймер — в целых кадрах: float-счёт 1/60 копит ошибку, и на целых
        # секундах int(198.99999) отнимал 20 очков бонуса
        self._frames_left = int(round(self.time_limit / rules.FIXED_DT))
        self.score = 0
        self.stats: Counter = Counter()
        self.events: list[Event] = []
        self.death_cause: Optional[str] = None

        self.flag_bonus = 0
        self.flag_fraction: Optional[float] = None
        self.time_bonus = 0

        self.on_ground = False
        self.max_x = self.player.center_x

        self._knockback = 0
        self._sequence_frames = 0

        # Цепочка прыжков по врагам без приземления
        self._chain = 0

        # Трубы
        self._pipe_warp: Optional[Warp] = None
        self._pipe_frames = 0
        self.rooms_visited: set[str] = set()

    # ===========================================================
    # СНИМОК И ОТКАТ
    # ===========================================================

    def snapshot(self) -> "_snapshot.WorldSnapshot":
        """Точный снимок состояния — для планировщика, который проигрывает
        продолжения на симуляторе. Что именно снимается — core/snapshot.py.

        Новое изменяемое поле мира нужно внести в snapshot.WORLD_FIELDS
        (или в WORLD_STATIC, если оно не меняется после reset())."""
        return _snapshot.capture(self)

    def restore(self, state: "_snapshot.WorldSnapshot") -> None:
        """Возвращает мир к снимку. Дальше любые нажатия дают кадр в кадр
        то же, что дал бы мир из момента снимка. Снимок остаётся годным:
        его можно восстанавливать снова. После reset() — ValueError."""
        _snapshot.apply(self, state)

    # ===========================================================
    # ТЕКУЩАЯ ЧАСТЬ УРОВНЯ
    # ===========================================================

    @property
    def map(self) -> Map:
        return self.area.map

    @property
    def walls(self) -> arcade.SpriteList:
        return self.area.walls

    @property
    def objects(self) -> arcade.SpriteList:
        return self.area.objects

    @property
    def decorations(self) -> arcade.SpriteList:
        return self.area.decorations

    @property
    def enemies(self) -> arcade.SpriteList:
        return self.area.enemies

    @property
    def blocks(self) -> list[ScoreBlock]:
        return self.area.blocks

    @property
    def tile_grid(self) -> np.ndarray:
        return self.area.tile_grid

    @property
    def tile_physics(self) -> bool:
        return self.area.tile_physics

    @property
    def physics_engine(self) -> arcade.PhysicsEnginePlatformer:
        return self.area.physics_engine

    @property
    def enemy_physics_engines(self) -> dict:
        return self.area.enemy_engines

    @property
    def in_room(self) -> bool:
        return self.area is not self.main

    def solid_at(self, tx: int, ty: int) -> bool:
        # За краями уровня — пустота: у игрока слева от x=0 нет стены,
        # он там падает в пропасть
        grid = self.area.tile_grid
        if tx < 0 or ty < 0:
            return False
        if tx >= grid.shape[0] or ty >= grid.shape[1]:
            return False
        return bool(grid[tx, ty])

    def warp_under_player(self) -> Optional[Warp]:
        """Труба-вход, на которой стоит игрок (можно уйти «вниз»)."""

        player = self.player
        for warp in self.area.warps:
            if (
                abs(player.center_x - warp.x) <= rules.WARP_REACH
                and abs(player.bottom - warp.top) <= 2.0
            ):
                return warp
        return None


    # ===========================================================
    # ОЧКИ И СОБЫТИЯ
    # ===========================================================

    def add_score(self, points: int, kind: str, **info) -> None:
        """Единая точка начисления очков. Её зовут монеты и блоки."""

        self.score += int(points)
        self.stats[kind] += 1
        self.stats[f"{kind}_points"] += int(points)
        self.events.append(Event(kind, int(points), info))

    # ===========================================================
    # СВОЙСТВА
    # ===========================================================

    @property
    def time_left(self) -> float:
        return self._frames_left * rules.FIXED_DT

    @property
    def done(self) -> bool:
        return self.phase in (COMPLETE, DEAD)

    @property
    def won(self) -> bool:
        return self.phase == COMPLETE

    @property
    def finishing(self) -> bool:
        """Идёт заставка финиша — управление у игры, а не у игрока."""
        return self.phase in (FLAGPOLE, TO_CASTLE)

    @property
    def in_pipe(self) -> bool:
        """Марио въезжает в трубу или вылезает из неё."""
        return self.phase in PIPE_PHASES

    @property
    def cutscene(self) -> bool:
        """Любая заставка: финиш или труба. Нажатия в ней ничего не меняют."""
        return self.finishing or self.in_pipe

    @property
    def progress(self) -> float:
        """Доля пути от старта до двери замка по лучшей достигнутой X."""
        span = max(self.geometry.door_x - self.spawn[0], 1.0)
        return float(np.clip((self.max_x - self.spawn[0]) / span, 0.0, 1.0))

    # ===========================================================
    # ШАГ
    # ===========================================================

    def step(self, inputs: Inputs = Inputs()) -> list[Event]:
        """Один кадр. Возвращает события кадра."""

        self.events = []

        if self.done:
            return self.events

        self.frame += 1
        self.elapsed += rules.FIXED_DT

        if self.phase == PLAYING:
            self._step_playing(inputs)
        elif self.phase == FLAGPOLE:
            self._step_flagpole()
        elif self.phase == TO_CASTLE:
            self._step_to_castle()
        elif self.phase in PIPE_PHASES:
            self._step_pipe()

        if not self.done:
            self._step_enemies()

        if self.phase == PLAYING:
            self._check_collisions()

        self._remove_dead_enemies()

        if self.phase == PLAYING:
            self._check_time_and_death()

        if self.phase == PLAYING and not self.in_room:
            self._check_flagpole()

        # Прогресс — только по самому уровню: X в комнате — другие координаты
        if not self.in_room:
            self.max_x = max(self.max_x, self.player.center_x)

        return self.events

    # -----------------------------------------------------------

    def _step_playing(self, inputs: Inputs) -> None:
        player = self.player

        # «Вниз» на трубе-входе — в секретную комнату
        if inputs.down and self.on_ground and self._knockback == 0:
            warp = self.warp_under_player()
            if warp is not None:
                self._enter_pipe(warp)
                return

        if self._knockback > 0:
            self._knockback -= 1
        elif inputs.right and not inputs.left:
            player.move_right()
        elif inputs.left and not inputs.right:
            player.move_left()
        else:
            player.stop_moving()

        if inputs.jump:
            player.jump()

        # Порядок как в исходном GameView.on_update:
        # сначала физика, потом логика игрока
        rising = player.change_y - self.physics_engine.gravity_constant > 0
        before = player.position
        self.physics_engine.update()

        # Упёрся головой, а arcade выталкивает вниз по 1 px, пока не станет
        # свободно. В щели ровно в рост Марио (блок «?» над трубой выхода в
        # комнате 1-1) с дробной высотой прыжка свободного места не находится,
        # и его протаскивало сквозь пол до y≈−16 — «смерть в яму» в закрытой
        # комнате. Летел вверх, а оказался ниже старта — это удар головой
        if rising and player.center_y < before[1] - 2:
            player.position = before
            player.change_y = 0

        self.on_ground = self.physics_engine.can_jump()

        # Летел вверх, и физика обнулила скорость — упёрся головой
        if rising and player.change_y == 0:
            self._head_bump()

        if self.on_ground:
            self._chain = 0

        # Устье боковой трубы в комнате: упёрся в него, идя вправо
        room_exit = self.area.exit
        if (
            room_exit is not None
            and inputs.right
            and not inputs.left
            and self.on_ground
            and player.right >= room_exit.left - 2.0
            and room_exit.bottom - 2.0 <= player.bottom <= room_exit.bottom + 8.0
        ):
            self._enter_side_pipe(room_exit.warp)
            return

        player.update(self.on_ground, rules.FIXED_DT)

        # Анимацию крутим всегда: у кадров разные хитбоксы,
        # и без неё столкновения на обучении разошлись бы с игрой
        player.update_animations(rules.FIXED_DT)

    # -----------------------------------------------------------
    # БЛОКИ «?»
    # -----------------------------------------------------------

    def _head_bump(self) -> None:
        """Удар головой: выбить монету из блока «?» над головой.

        Бьётся тот тайл, под которым больше головы, — как в оригинале:
        если голова больше под кирпичом, блок «?» рядом не срабатывает.
        """

        player = self.player
        head = player.top
        block: Optional[ScoreBlock] = None
        block_overlap = 0.0

        for candidate in self.area.blocks:
            if abs(candidate.bottom - head) > 2.0:
                continue
            overlap = min(player.right, candidate.right) - max(player.left, candidate.left)
            if overlap > block_overlap:
                block, block_overlap = candidate, overlap

        if block is None:
            return

        # Кирпич рядом под большей частью головы — удар достаётся ему
        if self.tile_physics:
            tile = rules.TILE
            ty = int(block.center_y // tile)
            block_tx = int(block.center_x // tile)
            for tx in range(int(player.left // tile), int((player.right - 1e-6) // tile) + 1):
                if tx == block_tx or not self.solid_at(tx, ty):
                    continue
                overlap = min(player.right, (tx + 1) * tile) - max(player.left, tx * tile)
                if overlap > block_overlap and not any(
                    int(b.center_x // tile) == tx and int(b.center_y // tile) == ty
                    for b in self.area.blocks
                ):
                    return

        if block.hit():
            self.add_score(
                rules.BLOCK_POINTS, "block",
                x=block.center_x, y=block.center_y, left=block.state,
            )
        else:
            self.events.append(Event("bump", 0, {"x": block.center_x, "y": block.center_y}))

        # Враг, стоявший на блоке, слетает — как в Super Mario Bros
        for enemy in list(self.enemies):
            if not enemy.is_alive:
                continue
            if abs(enemy.bottom - block.top) > 3.0:
                continue
            if enemy.right <= block.left or enemy.left >= block.right:
                continue
            if isinstance(enemy, Koopa):
                enemy.kill()
            else:
                enemy.get_hit()
            self._score_enemy(enemy, rules.ENEMY_POINTS, via="bump")

    # -----------------------------------------------------------
    # ТРУБЫ
    # -----------------------------------------------------------

    def _enter_pipe(self, warp: Warp) -> None:
        player = self.player
        self.phase = PIPE_DOWN
        self._pipe_warp = warp
        self._pipe_frames = 0

        player.change_x = 0
        player.change_y = 0
        player.jump_requested = False
        player.center_x = warp.x
        player.change_state("idle")

        self.events.append(Event("pipe", 0, {"dir": "down", "room": warp.room}))

    def _enter_side_pipe(self, warp: Warp) -> None:
        player = self.player
        self.phase = PIPE_SIDE
        self._pipe_warp = warp
        self._pipe_frames = 0

        player.change_x = 0
        player.change_y = 0
        player.jump_requested = False
        player.set_direction(1)
        player.change_state("move")

        self.events.append(Event("pipe", 0, {"dir": "side", "room": warp.room}))

    def _step_pipe(self) -> None:
        player = self.player
        warp = self._pipe_warp
        self._pipe_frames += 1

        if self.phase == PIPE_DOWN:
            player.center_y -= rules.PIPE_SPEED
            if player.top <= warp.top:
                self._go_to_room(warp)

        elif self.phase == PIPE_SIDE:
            player.center_x += rules.PIPE_SPEED
            player.update_animations(rules.FIXED_DT)
            if self._pipe_frames >= rules.PIPE_SIDE_FRAMES:
                self._leave_room(warp)

        elif self.phase == PIPE_UP:
            player.center_y += rules.PIPE_SPEED
            if player.bottom >= warp.exit_top:
                player.bottom = warp.exit_top
                player.change_y = 0
                self.on_ground = True
                self.phase = PLAYING
                self.events.append(Event("pipe", 0, {"dir": "up", "room": warp.room}))

    def _go_to_room(self, warp: Warp) -> None:
        player = self.player
        room = self.areas[warp.room]

        self.area = room
        player.center_x, player.center_y = room.spawn
        player.change_x = 0
        player.change_y = 0
        self.on_ground = False
        self._knockback = 0
        self._chain = 0
        self.phase = PLAYING

        if warp.room not in self.rooms_visited:
            self.rooms_visited.add(warp.room)
            self.stats["secret"] += 1
            self.events.append(Event("secret", 0, {"room": warp.room}))

    def _leave_room(self, warp: Warp) -> None:
        player = self.player

        self.area = self.main
        player.center_x = warp.exit_x
        player.top = warp.exit_top
        player.change_x = 0
        player.change_y = 0
        player.set_direction(1)
        player.change_state("idle")
        self.on_ground = False
        self._pipe_frames = 0
        self.phase = PIPE_UP

    # -----------------------------------------------------------
    # ВРАГИ
    # -----------------------------------------------------------

    def _step_enemies(self) -> None:
        player_x = self.player.center_x
        area = self.area

        for enemy in area.enemies:
            if id(enemy) in area.active_enemies:
                continue
            if abs(enemy.center_x - player_x) < rules.ENEMY_ACTIVATION_DISTANCE:
                area.active_enemies.add(id(enemy))

        for enemy, engine in area.enemy_engines.items():
            if not enemy.is_alive or id(enemy) not in area.active_enemies:
                continue
            engine.update()

        for enemy in list(area.enemies):
            if id(enemy) not in area.active_enemies:
                continue

            enemy.update(rules.FIXED_DT, area.walls)
            enemy.update_animations(rules.FIXED_DT)

            # Враг сам упал в пропасть — убираем без очков
            if enemy.center_y < rules.DEATH_Y:
                enemy.remove_from_sprite_lists()
                area.enemy_engines.pop(enemy, None)

    def _score_enemy(self, enemy, points: int, *, via: str, kind: str = "enemy", **info) -> None:
        """Очки за врага — сразу в момент прыжка или удара, а не когда
        уберут спрайт: так их видно и игроку, и нейросети."""

        if kind == "enemy":
            enemy.scored = True
        self.add_score(points, kind, type=type(enemy).__name__.lower(), via=via,
                       x=enemy.center_x, y=enemy.center_y, **info)

    def _check_collisions(self) -> None:
        player = self.player
        health_before = player.health

        # Что с врагами было до прыжка — чтобы понять, кого растоптали
        before = {
            id(enemy): bool(getattr(enemy, "is_shell", False))
            for enemy in self.enemies
            if enemy.is_alive
        }

        player.check_enemy_collides(self.enemies)
        player.check_objects_collides(self, self.objects)

        for enemy in self.enemies:
            was_shell = before.get(id(enemy))
            if was_shell is None:
                continue
            if not enemy.is_alive:
                kind = "enemy"
            elif not was_shell and getattr(enemy, "is_shell", False):
                # Koopa ушла в панцирь. Очки — только в первый раз: через
                # 5 с она вылезает, и иначе её можно было бы топтать
                # бесконечно, по 100 очков каждые 5 секунд
                if getattr(enemy, "shell_scored", False):
                    continue
                enemy.shell_scored = True
                kind = "stomp"
            else:
                continue

            chain = min(self._chain, len(rules.STOMP_CHAIN) - 1)
            self._chain += 1
            self._score_enemy(enemy, rules.STOMP_CHAIN[chain], via="stomp", kind=kind, chain=self._chain)

        if player.health < health_before:
            self._knockback = rules.KNOCKBACK_FRAMES
            self.stats["hits"] += 1
            self.events.append(Event("hit", 0, {"health": player.health}))

    def _remove_dead_enemies(self) -> None:
        for enemy in list(self.enemies):
            if not enemy.is_alive and enemy.death_timer > 0.5:
                enemy.remove_from_sprite_lists()
                self.area.enemy_engines.pop(enemy, None)
                if not getattr(enemy, "scored", False):
                    self._score_enemy(enemy, rules.ENEMY_POINTS, via="other")

    def _check_time_and_death(self) -> None:
        player = self.player

        self._frames_left = max(0, self._frames_left - 1)

        if player.center_y < rules.DEATH_Y:
            player.dead()
            self._die("pit")
        elif not player.is_alive:
            self._die("enemy")
        elif self._frames_left <= 0:
            player.dead()
            self._die("time")

    def _credit_dead_enemies(self) -> None:
        """Враг убит, но очки ещё не начислены (такого пути сейчас нет,
        все убийства оплачиваются сразу) — начисляем в конце уровня:
        смерть оставляет набранные очки."""
        for enemy in list(self.enemies):
            if not enemy.is_alive:
                enemy.remove_from_sprite_lists()
                self.area.enemy_engines.pop(enemy, None)
                if not getattr(enemy, "scored", False):
                    self._score_enemy(enemy, rules.ENEMY_POINTS, via="other")

    def _die(self, cause: str) -> None:
        self._credit_dead_enemies()
        self.phase = DEAD
        self.death_cause = cause
        self.stats["deaths"] += 1
        self.events.append(Event("death", 0, {"cause": cause}))

    # ===========================================================
    # ФЛАГШТОК И ЗАМОК
    # ===========================================================

    def _check_flagpole(self) -> None:
        pole = self.geometry.pole
        player = self.player

        if pole is None:
            # Уровень без флагштока: финиш у двери или на краю
            if player.center_x >= self.geometry.door_x:
                self._complete()
            return

        # Зона захвата шире самого штока на ширину его основания:
        # игрок, упёршийся в блок-основание, тоже хватается за флаг.
        # Перелёт над верхушкой засчитывается как захват за верх.
        reach = rules.TILE / 2 + 1
        if player.right < pole.x - reach or player.left > pole.x + reach:
            return

        height = float(np.clip(player.bottom - pole.bottom, 0.0, pole.height))
        fraction = height / pole.height

        self.flag_fraction = fraction
        self.flag_bonus = rules.flag_points(fraction)
        self.add_score(self.flag_bonus, "flag", fraction=round(fraction, 3))

        self.phase = FLAGPOLE
        self._sequence_frames = 0

        player.change_x = 0
        player.change_y = 0
        player.set_direction(1)
        player.center_x = pole.x - player.width / 2 + 4
        # Перелёт над верхушкой: съезжать начинаем от шара, а не из воздуха
        player.center_y = min(
            max(player.center_y, pole.bottom + player.height / 2),
            pole.top - player.height / 2,
        )
        player.change_state("climb")

    def _step_flagpole(self) -> None:
        pole = self.geometry.pole
        player = self.player

        sliding = player.bottom > pole.bottom + 0.5

        if sliding:
            player.center_y -= min(rules.FLAG_SLIDE_SPEED, player.bottom - pole.bottom)
            player.update_animations(rules.FIXED_DT)
        else:
            self._sequence_frames += 1

        # Флаг едет вниз вместе с игроком
        flag = self.flag_sprite
        if flag is not None:
            flag_floor = pole.bottom + rules.TILE / 2
            if flag.center_y > flag_floor:
                flag.center_y = max(flag_floor, flag.center_y - rules.FLAG_SLIDE_SPEED)

        if not sliding and self._sequence_frames >= rules.FLAG_HOLD_FRAMES:
            # Спрыгиваем на правую сторону штока и идём к замку
            player.center_x = pole.x + rules.TILE / 2 + player.width / 2 + 1
            player.set_direction(1)
            player.change_state("idle")
            self.phase = TO_CASTLE
            self._sequence_frames = 0

    def _step_to_castle(self) -> None:
        player = self.player

        player.move_right()
        self.physics_engine.update()
        self.on_ground = self.physics_engine.can_jump()
        player.update(self.on_ground, rules.FIXED_DT)
        player.update_animations(rules.FIXED_DT)

        self._sequence_frames += 1

        reached = player.center_x >= self.geometry.door_x
        stuck = self._sequence_frames >= rules.CASTLE_WALK_MAX_FRAMES
        fell = player.center_y < rules.DEATH_Y

        if reached or stuck or fell:
            self._complete()

    def _complete(self) -> None:
        player = self.player
        self._credit_dead_enemies()

        player.stop_moving()
        player.change_y = 0
        player.visible = False

        seconds = int(self._frames_left * rules.FIXED_DT + 1e-9)
        self.time_bonus = seconds * rules.TIME_BONUS_PER_SECOND
        self.add_score(self.time_bonus, "time_bonus", seconds=seconds)

        self.phase = COMPLETE
        self.events.append(Event("complete", 0, {"score": self.score}))

    # ===========================================================
    # СВОДКА
    # ===========================================================

    def summary(self) -> dict:
        """Итог прохождения — для турнира и экрана результатов."""

        return {
            "level": self.name,
            "score": self.score,
            "won": self.won,
            "death": self.death_cause,
            "progress": round(self.progress, 4),
            "time_used": round(self.time_limit - self.time_left, 2),
            "time_left": round(self.time_left, 2),
            "flag_bonus": self.flag_bonus,
            "flag_fraction": (
                None if self.flag_fraction is None else round(self.flag_fraction, 3)
            ),
            "time_bonus": self.time_bonus,
            "coins": self.stats["coin"],
            "blocks": self.stats["block"],
            "enemies": self.stats["enemy"],
            "stomps": self.stats["stomp"],
            "secrets": self.stats["secret"],
            "hits": self.stats["hits"],
            "frames": self.frame,
            # Сколько было на уровне вместе с комнатами — «собрал из возможного»
            "coins_total": self.totals["coins"],
            "block_hits_total": self.totals["block_hits"],
            "enemies_total": self.totals["enemies"],
            "secrets_total": self.totals["secrets"],
        }


__all__ = [
    "GameWorld",
    "Inputs",
    "Event",
    "LevelGeometry",
    "Flagpole",
    "Castle",
    "build_geometry",
    "load_level_data",
    "PLAYING",
    "FLAGPOLE",
    "TO_CASTLE",
    "COMPLETE",
    "DEAD",
    "PIPE_DOWN",
    "PIPE_SIDE",
    "PIPE_UP",
    "PIPE_PHASES",
    "Area",
    "Warp",
    "RoomExit",
]

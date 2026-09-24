"""Мир игры без окна.

Вся игровая логика одного уровня: физика, враги, очки, таймер, флагшток и
вход в замок. Живая игра (GameView) только рисует этот мир и передаёт в
него нажатия, среда обучения передаёт в него решения нейросети. Правила
одни и те же, поэтому модель в окне ведёт себя так же, как на обучении.

    world = GameWorld("maps/custom_level.json")
    while not world.done:
        world.step(Inputs(right=True, jump=...))
    print(world.score, world.won)
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
from Entities.player import Player
from map import BASE_DIR, Map
from Objects.Flag import Flag

LevelSource = Union[str, Path, dict]

# Фазы уровня
PLAYING = "playing"
FLAGPOLE = "flagpole"
TO_CASTLE = "to_castle"
COMPLETE = "complete"
DEAD = "dead"


class Inputs(NamedTuple):
    """Управление на один кадр.

    right/left — клавиша удерживается; jump — клавишу прыжка нажали в этом
    кадре (фронт нажатия, как on_key_press), а не держат.
    """

    right: bool = False
    left: bool = False
    jump: bool = False


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
        self.map: Optional[Map] = None
        self.player: Optional[Player] = None
        self.walls = arcade.SpriteList()
        self.objects = arcade.SpriteList()
        self.decorations = arcade.SpriteList()
        self.enemies = arcade.SpriteList()
        self.physics_engine: Optional[arcade.PhysicsEnginePlatformer] = None
        self.enemy_physics_engines: dict[Any, arcade.PhysicsEnginePlatformer] = {}

        self.reset()

    # ===========================================================
    # RESET
    # ===========================================================

    def reset(self) -> None:
        """Пересобирает уровень с нуля."""

        # Map может дописывать в словарь — отдаём копию
        with _QuietMap():
            self.map = Map(data=copy.deepcopy(self.level_data))

        self.walls = self.map.get_walls()
        # Без пространственного хэша физика проверяет каждый блок уровня
        # на каждом кадре: 2.9 тыс. кадров/с против 27 тыс. на уровне 1-1
        self.walls.enable_spatial_hashing()

        self.objects = self.map.get_objects()
        self.decorations = self.map.get_decorations()
        self.enemies = self.map.get_enemies()

        spawn_x, spawn_y = self.map.get_player_spawn()
        self.spawn = (float(spawn_x), float(spawn_y))

        self.player = Player()
        self.player.center_x = spawn_x
        self.player.center_y = spawn_y

        self.physics_engine = arcade.PhysicsEnginePlatformer(
            player_sprite=self.player,
            walls=self.walls,
            gravity_constant=self.player.gravity,
        )

        self._build_tile_grid()

        # Враги — по тайлам, если все стены лежат ровной сеткой 32×32.
        # Иначе (нестандартные блоки из редактора) — движок arcade.
        self.tile_physics = self._walls_on_grid()

        self.enemy_physics_engines = {
            enemy: (
                TileBody(enemy, self, enemy.gravity)
                if self.tile_physics
                else arcade.PhysicsEnginePlatformer(
                    player_sprite=enemy,
                    walls=self.walls,
                    gravity_constant=enemy.gravity,
                )
            )
            for enemy in self.enemies
        }

        self._active_enemies: set[int] = set()
        if not self.activate_enemies:
            self._active_enemies = {id(enemy) for enemy in self.enemies}

        self.flag_sprite: Optional[Flag] = next(
            (obj for obj in self.objects if isinstance(obj, Flag)), None
        )

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

    # ===========================================================
    # СЕТКА ТАЙЛОВ ДЛЯ НАБЛЮДЕНИЯ
    # ===========================================================

    def _build_tile_grid(self) -> None:
        """Статическая сетка твёрдых блоков: solid[tx, ty] = 1.

        Стены не разрушаются, поэтому сетка строится один раз на уровень.
        """

        tiles = [
            (int(wall.center_x // rules.TILE), int(wall.center_y // rules.TILE))
            for wall in self.walls
        ]

        max_tx = max((t[0] for t in tiles), default=0)
        max_ty = max((t[1] for t in tiles), default=0)

        self.tile_grid = np.zeros((max_tx + 64, max(max_ty, 24) + 16), dtype=np.uint8)

        for tx, ty in tiles:
            if tx >= 0 and ty >= 0:
                self.tile_grid[tx, ty] = 1

    def _walls_on_grid(self) -> bool:
        half = rules.TILE / 2
        for wall in self.walls:
            if abs(wall.width - rules.TILE) > 0.5 or abs(wall.height - rules.TILE) > 0.5:
                return False
            if (wall.center_x - half) % rules.TILE or (wall.center_y - half) % rules.TILE:
                return False
        return True

    def solid_at(self, tx: int, ty: int) -> bool:
        # За краями уровня — пустота: у игрока слева от x=0 нет стены,
        # он там падает в пропасть
        if tx < 0 or ty < 0:
            return False
        if tx >= self.tile_grid.shape[0] or ty >= self.tile_grid.shape[1]:
            return False
        return bool(self.tile_grid[tx, ty])

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

        if not self.done:
            self._step_enemies()

        if self.phase == PLAYING:
            self._check_collisions()

        self._remove_dead_enemies()

        if self.phase == PLAYING:
            self._check_time_and_death()

        if self.phase == PLAYING:
            self._check_flagpole()

        self.max_x = max(self.max_x, self.player.center_x)

        return self.events

    # -----------------------------------------------------------

    def _step_playing(self, inputs: Inputs) -> None:
        player = self.player

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
        self.physics_engine.update()
        self.on_ground = self.physics_engine.can_jump()

        player.update(self.on_ground, rules.FIXED_DT)

        # Анимацию крутим всегда: у кадров разные хитбоксы,
        # и без неё столкновения на обучении разошлись бы с игрой
        player.update_animations(rules.FIXED_DT)

    def _step_enemies(self) -> None:
        player_x = self.player.center_x

        for enemy in self.enemies:
            if id(enemy) in self._active_enemies:
                continue
            if abs(enemy.center_x - player_x) < rules.ENEMY_ACTIVATION_DISTANCE:
                self._active_enemies.add(id(enemy))

        for enemy, engine in self.enemy_physics_engines.items():
            if not enemy.is_alive or id(enemy) not in self._active_enemies:
                continue
            engine.update()

        for enemy in list(self.enemies):
            if id(enemy) not in self._active_enemies:
                continue

            enemy.update(rules.FIXED_DT, self.walls)
            enemy.update_animations(rules.FIXED_DT)

            # Враг сам упал в пропасть — убираем без очков
            if enemy.center_y < rules.DEATH_Y:
                enemy.remove_from_sprite_lists()
                self.enemy_physics_engines.pop(enemy, None)

    def _check_collisions(self) -> None:
        player = self.player
        health_before = player.health

        player.check_enemy_collides(self.enemies)
        player.check_objects_collides(self, self.objects)

        if player.health < health_before:
            self._knockback = rules.KNOCKBACK_FRAMES
            self.stats["hits"] += 1
            self.events.append(Event("hit", 0, {"health": player.health}))

    def _remove_dead_enemies(self) -> None:
        for enemy in list(self.enemies):
            if not enemy.is_alive and enemy.death_timer > 0.5:
                enemy.remove_from_sprite_lists()
                self.enemy_physics_engines.pop(enemy, None)
                self.add_score(
                    rules.ENEMY_POINTS,
                    "enemy",
                    type=type(enemy).__name__.lower(),
                )

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
        """Враг растоптан, но спрайт ещё не убран (очки идут через 0.5 с) —
        если уровень кончился раньше, начисляем сразу: смерть оставляет
        набранные очки."""
        for enemy in list(self.enemies):
            if not enemy.is_alive:
                enemy.remove_from_sprite_lists()
                self.enemy_physics_engines.pop(enemy, None)
                self.add_score(rules.ENEMY_POINTS, "enemy", type=type(enemy).__name__.lower())

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
            "hits": self.stats["hits"],
            "frames": self.frame,
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
]

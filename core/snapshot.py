"""Точный снимок мира и откат к нему.

Планировщику нужно проиграть на симуляторе сотни продолжений из одной
точки: снять состояние, прогнать кадры, вернуться, прогнать другие.

    state = world.snapshot()
    for plan in plans:
        world.restore(state)
        for inputs in plan:
            world.step(inputs)

После restore любые нажатия дают кадр в кадр то же, что дал бы исходный
мир из момента снимка: позиции и скорости, текстуры и хитбоксы, враги,
монеты, блоки, очки, события, фаза, таймеры, трубы, флаг. Один снимок
можно восстанавливать сколько угодно раз, мир между откатами живёт
дальше. Снимок годится только для своего мира и только до его reset().

copy.deepcopy тут не подходит: спрайты arcade его не поддерживают, а
копия уровня стоила бы миллисекунды. Поэтому снимок перечисляет
изменяемое состояние явно:

* спрайты-сущности (игрок и все враги всех частей уровня): слоты arcade
  (_position, _velocity, _texture, _scale, _color, _visible) и весь
  __dict__ — там анимация, здоровье, таймеры панциря, флаги scored и
  shell_scored, которые мир дописывает на ходу;
* блоки «?»: оставшиеся удары и текстура;
* состав списков спрайтов (монеты в objects, враги в enemies и в
  map.entities) — спрайты из них только убывают, поэтому любой
  достижимый состав — подпоследовательность исходного, и вернуть
  порядок можно вставкой на сохранённые места;
* у каждой части: словарь движков врагов, множество активных врагов,
  счётчик прыжков движка игрока;
* поля самого мира из WORLD_FIELDS плюс stats, events, rooms_visited.

Спрайты возвращаются через сеттеры arcade (position, texture, scale,
color, visible) — они же двигают спрайт в пространственном хэше и в
буферах SpriteList. В arcade 3.3 смена текстуры хитбокс не меняет
(точки берутся с первой текстуры), меняются только _width/_height; их
пересчитывают те же сеттеры.
"""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, Any

import arcade

if TYPE_CHECKING:
    from core.world import GameWorld

# Скалярные поля мира, которые меняются по ходу игры. Контейнеры
# (stats, events, rooms_visited) копируются отдельно
WORLD_FIELDS = (
    "area",
    "phase",
    "frame",
    "elapsed",
    "_frames_left",
    "score",
    "death_cause",
    "flag_bonus",
    "flag_fraction",
    "time_bonus",
    "on_ground",
    "max_x",
    "_knockback",
    "_sequence_frames",
    "_chain",
    "_pipe_warp",
    "_pipe_frames",
)

WORLD_CONTAINERS = ("stats", "events", "rooms_visited")

# Поля мира, которые после reset() не меняются: уровень, части, геометрия.
# Новое поле мира должно попасть либо сюда, либо в WORLD_FIELDS —
# tools/check_snapshot.py падает на поле, которого нет ни там, ни там
WORLD_STATIC = (
    "level_data",
    "level_path",
    "name",
    "time_limit",
    "activate_enemies",
    "geometry",
    "player",
    "main",
    "areas",
    "warps",
    "spawn",
    "flag_sprite",
    "totals",
    "_snapshot_plan",
)

_get_fields = itemgetter(*WORLD_FIELDS)


class _Plan:
    """Что снимать: список объектов мира, собранный один раз.

    Строится при первом снимке после reset(). Спрайт, убранный из
    списков раньше первого снимка, не нужен: ни один снимок его не
    вернёт, а убранного спрайта игра не касается.
    """

    __slots__ = ("player", "areas", "entities", "blocks", "lists", "engines", "plain")

    def __init__(self, world: "GameWorld"):
        self.player = world.player
        self.areas = tuple(world.areas.values())

        entities: list = [world.player]
        seen = {id(world.player)}
        lists: list[arcade.SpriteList] = []
        listed: set[int] = set()
        blocks: list = []
        engines: list = []

        for area in self.areas:
            for sprite_list in (area.objects, area.enemies, getattr(area.map, "entities", None)):
                if sprite_list is None or id(sprite_list) in listed:
                    continue
                listed.add(id(sprite_list))
                lists.append(sprite_list)

            for enemy in list(area.enemies) + list(area.enemy_engines):
                if id(enemy) not in seen:
                    seen.add(id(enemy))
                    entities.append(enemy)

            blocks.extend(area.blocks)

            # Движки arcade: игрока в каждой части и врагов в частях
            # без сетки тайлов. У них из состояния только счётчик прыжков
            engines.append(area.physics_engine)
            engines.extend(
                engine for engine in area.enemy_engines.values()
                if isinstance(engine, arcade.PhysicsEnginePlatformer)
            )

        self.entities = tuple(entities)
        self.blocks = tuple(blocks)
        self.lists = tuple(lists)
        self.engines = tuple(engines)

        # Спрайты, у которых меняется только позиция: флаг едет по штоку
        self.plain = tuple(s for s in (world.flag_sprite,) if s is not None)


class WorldSnapshot:
    """Состояние мира на один кадр. Внутри — только ссылки на неизменяемые
    значения и собственные копии контейнеров, поэтому снимок не портится,
    пока мир живёт дальше."""

    __slots__ = (
        "plan", "fields", "stats", "events", "rooms",
        "entities", "blocks", "lists", "engines_by_area", "active", "jumps", "plain",
    )

    @property
    def frame(self) -> int:
        return self.fields[WORLD_FIELDS.index("frame")]


def _plan(world: "GameWorld") -> _Plan:
    plan = world.__dict__.get("_snapshot_plan")
    if plan is None or plan.player is not world.player:
        plan = _Plan(world)
        world.__dict__["_snapshot_plan"] = plan
    return plan


def capture(world: "GameWorld") -> WorldSnapshot:
    plan = _plan(world)
    snap = WorldSnapshot()
    snap.plan = plan

    state = world.__dict__
    snap.fields = _get_fields(state)
    # Counter.copy() идёт через Counter.update — вдвое дольше простого dict
    snap.stats = dict(state["stats"])
    snap.events = list(state["events"])
    snap.rooms = frozenset(state["rooms_visited"])

    snap.entities = [
        (s._position, s._velocity, s._texture, s._scale, s._color, s._visible, s.__dict__.copy())
        for s in plan.entities
    ]
    snap.blocks = [(b.state, b._texture) for b in plan.blocks]
    snap.lists = [sl.sprite_list[:] for sl in plan.lists]
    snap.engines_by_area = [area.enemy_engines.copy() for area in plan.areas]
    snap.active = [frozenset(area.active_enemies) for area in plan.areas]
    snap.jumps = [engine.jumps_since_ground for engine in plan.engines]
    snap.plain = [s._position for s in plan.plain]
    return snap


def _restore_list(sprite_list: arcade.SpriteList, saved: list) -> None:
    """Возвращает состав списка спрайтов с исходным порядком.

    Лишние (их не бывает: спрайты только убывают, но пусть будет общо)
    убираются, недостающие встают на свои места. Если порядок всё же
    разошёлся — список собирается заново.
    """

    current = sprite_list.sprite_list
    keep = set(map(id, saved))
    for sprite in [s for s in current if id(s) not in keep]:
        sprite_list.remove(sprite)

    slots = sprite_list.sprite_slot
    for index, sprite in enumerate(saved):
        if sprite not in slots:
            sprite_list.insert(index, sprite)

    if sprite_list.sprite_list != saved:
        sprite_list.clear()
        sprite_list.extend(saved)


def apply(world: "GameWorld", snap: WorldSnapshot) -> None:
    plan = snap.plan
    if plan.player is not world.player or world.__dict__.get("_snapshot_plan") is not plan:
        raise ValueError("Снимок снят с другого мира или до reset()")

    state = world.__dict__
    state.update(zip(WORLD_FIELDS, snap.fields))

    # Контейнеры — на месте: на них могут держать ссылки снаружи.
    # stats чистится и заливается через dict.update: Counter.update складывает
    stats = state["stats"]
    if stats != snap.stats:
        stats.clear()
        dict.update(stats, snap.stats)
    state["events"] = list(snap.events)
    rooms = state["rooms_visited"]
    if rooms != snap.rooms:
        rooms.clear()
        rooms.update(snap.rooms)

    # Состав списков — до спрайтов: сеттеры позиции и текстуры обновляют
    # буферы тех списков, в которых спрайт состоит
    for sprite_list, saved in zip(plan.lists, snap.lists):
        if sprite_list.sprite_list != saved:
            _restore_list(sprite_list, saved)

    for area, engines, active in zip(plan.areas, snap.engines_by_area, snap.active):
        current = area.enemy_engines
        if current != engines:
            current.clear()
            current.update(engines)
        if area.active_enemies != active:
            area.active_enemies.clear()
            area.active_enemies.update(active)

    for engine, jumps in zip(plan.engines, snap.jumps):
        engine.jumps_since_ground = jumps

    for sprite, (position, velocity, texture, scale, color, visible, attrs) in zip(plan.entities, snap.entities):
        # Масштаб раньше текстуры: оба сеттера пересчитывают _width/_height,
        # последний — по итоговой паре
        if sprite._scale != scale:
            sprite.scale = scale
        if sprite._texture is not texture:
            sprite.texture = texture
        if sprite._position != position:
            sprite.position = position
        sprite._velocity = velocity
        if sprite._color != color:
            sprite.color = color
        if sprite._visible != visible:
            sprite.visible = visible
        # clear + update, а не одно update: мир дописывает врагам scored и
        # shell_scored, и лишний флаг из другой ветки должен исчезнуть
        attrs_now = sprite.__dict__
        attrs_now.clear()
        attrs_now.update(attrs)

    for block, (hits, texture) in zip(plan.blocks, snap.blocks):
        if block.state != hits:
            block.state = hits
        if block._texture is not texture:
            block.texture = texture

    for sprite, position in zip(plan.plain, snap.plain):
        if sprite._position != position:
            sprite.position = position


__all__ = ["WorldSnapshot", "capture", "apply", "WORLD_FIELDS", "WORLD_CONTAINERS", "WORLD_STATIC"]

from __future__ import annotations

import json
import logging
from pathlib import Path

import arcade

from Entities.Enemies.goomba import Goomba
from Entities.Enemies.koopa import Koopa
from Objects.Coin import Coin
from Objects.Flag import Flag
from Objects.ScoreBlock import ScoreBlock

logger = logging.getLogger(__name__)


# =============================================================
# PATHS
# =============================================================

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
MIN_TEXTURE_SIZE = 32


class Map:
    """
    Загружает карту, созданную в Level Editor.
    """

    MIN_TEXTURE_SIZE = MIN_TEXTURE_SIZE

    # =========================================================
    # REGISTRIES
    # =========================================================

    OBJECT_REGISTRY: dict[str, type] = {
        "coins":      Coin,
        "flag":       Flag,
        "scoreblock": ScoreBlock,
    }

    ENEMY_REGISTRY: dict[str, type] = {
        "goomba": Goomba,
        "koopa":  Koopa,
    }

    # =========================================================
    # CATEGORY ALIASES
    # =========================================================

    _BLOCK_ALIASES = {"blocks", "block", "блоки"}
    _OBJECT_ALIASES = {"objects", "object", "объекты"}
    _DECORATION_ALIASES = {
        "decorations", "decoration", "декорации", "decorates"
    }
    _ENTITY_ALIASES = {"entities", "entity", "сущности"}

    # =========================================================
    # INIT
    # =========================================================

    # Индекс файлов ресурсов общий на процесс: обход resources/ на каждую
    # загрузку карты заметен, когда уровни пересобираются тысячи раз
    _RESOURCE_INDEX_CACHE: dict[str, list[Path]] | None = None

    def __init__(self, map_file: str | Path | None = None, data: dict | None = None):
        # Уровень можно передать готовым словарем (генератор уровней)
        self._data = data

        # Путь к карте
        path = Path(map_file) if map_file is not None else Path("maps/<generated>.json")
        if not path.is_absolute():
            path = BASE_DIR / path
        self.map_file = path

        # ---------- CACHES ----------
        self._texture_cache: dict[Path, arcade.Texture] = {}
        self._path_cache: dict[tuple[str, str, str], Path | None] = {}

        # Индекс файлов ресурсов: имя файла -> список путей
        self._resource_index: dict[str, list[Path]] = {}
        self._build_resource_index()

        # ---------- MAP INFO ----------
        self.name = "Level_1"
        self.width = 3000
        self.height = 600
        self.player_spawn = {"x": 100, "y": 100}

        # ---------- SPRITE LISTS ----------
        self.walls = arcade.SpriteList()
        self.objects = arcade.SpriteList()
        self.decorations = arcade.SpriteList()
        self.enemies = arcade.SpriteList()
        self.entities = arcade.SpriteList()

        # ---------- CATEGORY DISPATCH ----------
        self._category_handlers = {
            **{alias: self.create_block_from_editor
               for alias in self._BLOCK_ALIASES},
            **{alias: self.create_object_from_editor
               for alias in self._OBJECT_ALIASES},
            **{alias: self.create_decoration_from_editor
               for alias in self._DECORATION_ALIASES},
            **{alias: self.create_entity_from_editor
               for alias in self._ENTITY_ALIASES},
        }

        # ---------- LOAD ----------
        self.load_map()

    # =========================================================
    # RESOURCE INDEX
    # =========================================================

    def _build_resource_index(self) -> None:
        """Один раз обходит resources/ и строит индекс по имени файла."""
        if Map._RESOURCE_INDEX_CACHE is not None:
            self._resource_index = Map._RESOURCE_INDEX_CACHE
            return
        if not RESOURCES_DIR.exists():
            return
        for file_path in RESOURCES_DIR.rglob("*"):
            if file_path.is_file():
                self._resource_index.setdefault(
                    file_path.name, []
                ).append(file_path)
        Map._RESOURCE_INDEX_CACHE = self._resource_index

    # =========================================================
    # TEXTURE CACHE
    # =========================================================

    def _get_texture(self, texture_file: Path) -> arcade.Texture | None:
        cached = self._texture_cache.get(texture_file)
        if cached is not None:
            return cached

        try:
            texture = arcade.load_texture(str(texture_file))
        except Exception as error:
            logger.warning(
                "[Map] Ошибка загрузки текстуры %s: %s",
                texture_file, error,
            )
            return None

        self._texture_cache[texture_file] = texture
        return texture

    # =========================================================
    # LOAD MAP
    # =========================================================

    def load_map(self) -> None:
        if self._data is not None:
            data = self._data
        else:
            if not self.map_file.exists():
                raise FileNotFoundError(
                    f"Файл карты не найден: {self.map_file}"
                )

            with open(self.map_file, "r", encoding="utf-8") as file:
                data = json.load(file)

        # ---------- MAP SETTINGS ----------
        self.name = data.get("name", "Unnamed Level")
        self.width = data.get("width", 3000)
        self.height = data.get("height", 600)
        self.player_spawn = data.get(
            "player_spawn", {"x": 100, "y": 100}
        )

        objects = data.get("objects", [])

        logger.info("[Map] Загрузка карты: %s", self.name)
        logger.info("[Map] Размер: %s x %s", self.width, self.height)
        logger.info("[Map] Объектов: %s", len(objects))

        for object_data in objects:
            try:
                self.create_editor_object(object_data)
            except Exception as error:
                logger.exception(
                    "[Map] Ошибка создания объекта %s: %s",
                    object_data, error,
                )

        logger.info("[Map] Карта успешно загружена")

    # =========================================================
    # CREATE EDITOR OBJECT (dispatch)
    # =========================================================

    def create_editor_object(self, data: dict) -> None:
        category = self.get_category(data)
        if not category:
            logger.warning(
                "[Map] Не удалось определить категорию: %s", data
            )
            return

        handler = self._category_handlers.get(category.lower())
        if handler is None:
            logger.warning("[Map] Неизвестная категория: %s", category)
            return

        handler(data)

    # =========================================================
    # CATEGORY
    # =========================================================

    def get_category(self, data: dict) -> str | None:
        for key in ("category", "type", "resource_category"):
            value = data.get(key)
            if value:
                return str(value)

        resource = data.get("resource")
        if isinstance(resource, dict):
            value = resource.get("category")
            if value:
                return str(value)

        return None

    # =========================================================
    # POSITION
    # =========================================================

    def get_position(self, data: dict) -> tuple[float, float]:
        return (
            float(data.get("x", 0)),
            float(data.get("y", 0)),
        )

    # =========================================================
    # TEXTURE PATH
    # =========================================================

    def get_texture_path(self, data: dict) -> str | None:
        for key in ("texture", "texture_path", "path"):
            value = data.get(key)
            if value:
                return str(value)

        resource = data.get("resource")
        if isinstance(resource, dict):
            for key in ("texture", "path", "texture_path"):
                value = resource.get(key)
                if value:
                    return str(value)

        return None

    # =========================================================
    # RESOURCE NAME
    # =========================================================

    def get_resource_name(self, data: dict) -> str:
        resource = data.get("resource")

        if isinstance(resource, str) and resource.strip():
            return resource.strip()

        for key in ("resource_name", "group"):
            value = data.get(key)
            if value:
                return str(value)

        if isinstance(resource, dict):
            for key in ("name", "group"):
                value = resource.get(key)
                if value:
                    return str(value)

        texture_path = self.get_texture_path(data)
        if texture_path:
            path = Path(texture_path.replace("\\", "/"))
            parts = path.parts
            lower_parts = [p.lower() for p in parts]

            if "resources" in lower_parts:
                idx = lower_parts.index("resources")
                if idx + 2 < len(parts):
                    return parts[idx + 2]

            if len(parts) >= 3:
                return parts[-2]

            if len(parts) == 1:
                return path.stem

        return "unknown"

    # =========================================================
    # RESOLVE TEXTURE PATH
    # =========================================================

    def resolve_texture_path(self, data: dict) -> Path | None:
        texture_path = self.get_texture_path(data)
        if not texture_path:
            return None

        texture_path = texture_path.replace("\\", "/")
        category = (self.get_category(data) or "").lower()
        resource_name = self.get_resource_name(data)

        cache_key = (category, resource_name, texture_path)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        result = self._resolve_texture_path_uncached(
            texture_path, category, resource_name
        )
        self._path_cache[cache_key] = result
        return result

    def _resolve_texture_path_uncached(
        self,
        texture_path: str,
        category: str,
        resource_name: str,
    ) -> Path | None:
        path = Path(texture_path)

        # 1. Абсолютный путь
        if path.is_absolute():
            return path if path.exists() else None

        project_path = BASE_DIR / path
        if project_path.exists():
            return project_path

        resource_path = RESOURCES_DIR / path
        if resource_path.exists():
            return resource_path

        if category and resource_name:
            candidate = RESOURCES_DIR / category / resource_name / path
            if candidate.exists():
                return candidate

        matches = self._resource_index.get(path.name)
        if matches:
            return matches[0]

        return None

    # =========================================================
    # CREATE SPRITE
    # =========================================================

    def create_sprite_from_editor(self, data: dict) -> arcade.Sprite | None:
        texture_file = self.resolve_texture_path(data)
        if texture_file is None:
            logger.warning("[Map] Текстура не найдена: %s", data)
            return None

        texture = self._get_texture(texture_file)
        if texture is None:
            return None

        texture_width = texture.width
        texture_height = texture.height

        scale = 1.0
        if (
            texture_width < self.MIN_TEXTURE_SIZE
            or texture_height < self.MIN_TEXTURE_SIZE
        ):
            scale_x = (
                self.MIN_TEXTURE_SIZE / texture_width
                if texture_width > 0 else 1.0
            )
            scale_y = (
                self.MIN_TEXTURE_SIZE / texture_height
                if texture_height > 0 else 1.0
            )
            scale = max(scale_x, scale_y)

        try:
            sprite = arcade.Sprite(texture, scale=scale)
        except Exception as error:
            logger.warning(
                "[Map] Ошибка создания Sprite %s: %s",
                texture_file, error,
            )
            return None

        sprite.center_x, sprite.center_y = self.get_position(data)
        return sprite

    # =========================================================
    # BLOCKS
    # =========================================================

    def create_block_from_editor(self, data: dict) -> None:
        sprite = self.create_sprite_from_editor(data)
        if sprite is None:
            return
        self.walls.append(sprite)

    # =========================================================
    # OBJECTS
    # =========================================================

    def create_object_from_editor(self, data: dict) -> None:
        sprite_name = self.get_resource_name(data).lower()
        cls = self.OBJECT_REGISTRY.get(sprite_name)
        if cls is None:
            return


        center_x, center_y = self.get_position(data)
        sprite = cls(center_x, center_y)

        # Переопределяем текстуру, если указана
        if self.get_texture_path(data):
            texture_file = self.resolve_texture_path(data)
            if texture_file is not None:
                texture = self._get_texture(texture_file)
                if texture is not None:
                    sprite.texture = texture

        self.objects.append(sprite)

    # =========================================================
    # DECORATIONS
    # =========================================================

    def create_decoration_from_editor(self, data: dict) -> None:
        sprite = self.create_sprite_from_editor(data)
        if sprite is None:
            return
        self.decorations.append(sprite)

    # =========================================================
    # ENTITIES
    # =========================================================

    def create_entity_from_editor(self, data: dict) -> None:
        entity_name = self.get_resource_name(data).lower()
        x, y = self.get_position(data)

        logger.debug("[Map] Entity: %s at (%s, %s)", entity_name, x, y)

        if entity_name == "mario":
            self.player_spawn = {"x": x, "y": y}
            logger.debug("[Map] Player spawn: %s, %s", x, y)
            return

        enemy_cls = self.ENEMY_REGISTRY.get(entity_name)
        if enemy_cls is None:
            logger.warning("[Map] Неизвестная сущность: %s", entity_name)
            return

        enemy = enemy_cls(x=x, y=y)
        self.enemies.append(enemy)
        self.entities.append(enemy)

    # =========================================================
    # GETTERS
    # =========================================================

    def get_player_spawn(self) -> tuple[float, float]:
        return self.player_spawn["x"], self.player_spawn["y"]

    def get_walls(self) -> arcade.SpriteList:
        return self.walls

    def get_objects(self) -> arcade.SpriteList:
        return self.objects

    def get_decorations(self) -> arcade.SpriteList:
        return self.decorations

    def get_enemies(self) -> arcade.SpriteList:
        return self.enemies

    def get_entities(self) -> arcade.SpriteList:
        return self.entities
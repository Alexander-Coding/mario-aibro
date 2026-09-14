from __future__ import annotations

import json
from pathlib import Path

import arcade

from Entities.Enemies.goomba import Goomba
from Entities.Enemies.koopa import Koopa
from Objects.Coin import Coin
from Objects.Flag import Flag

# =============================================================
# PATHS
# =============================================================

# Корень проекта.
# map.py находится, например:
#
# GAME/
# ├── map.py
# ├── resources/
# └── maps/
#
BASE_DIR = Path(__file__).resolve().parent

# Папка ресурсов
RESOURCES_DIR = BASE_DIR / "resources"

# Минимальный размер спрайта
MIN_TEXTURE_SIZE = 32


class Map:
    """
    Загружает карту, созданную в Level Editor.

    Категории ресурсов:

    Blocks      -> стены с коллизией
    Objects     -> обычные игровые объекты
    Decorations -> декоративные объекты
    Entities    -> сущности игры

    Поддерживаемые пути текстур:

        resources/blocks/ground/ground.png

    а также старые варианты:

        ground.png
        blocks/ground/ground.png
        /absolute/path/to/ground.png
    """

    def __init__(self, map_file: str):
        self.map_file = Path(map_file)

        # Если путь к карте относительный,
        # считаем его относительно проекта.
        if not self.map_file.is_absolute():
            self.map_file = BASE_DIR / self.map_file

        # =====================================================
        # MAP INFO
        # =====================================================

        self.name = "Unnamed Level"

        self.width = 3000
        self.height = 600

        self.player_spawn = {
            "x": 100,
            "y": 100
        }

        # =====================================================
        # SPRITE LISTS
        # =====================================================

        # Блоки с физикой
        self.walls = arcade.SpriteList()

        # Обычные игровые объекты
        self.objects = arcade.SpriteList()

        # Декорации
        self.decorations = arcade.SpriteList()

        # Враги
        self.enemies = arcade.SpriteList()

        # Все игровые сущности
        self.entities = arcade.SpriteList()

        # =====================================================
        # LOAD
        # =====================================================

        self.load_map()

    # =========================================================
    # LOAD MAP
    # =========================================================

    def load_map(self) -> None:
        """
        Загружает карту из JSON.
        """

        if not self.map_file.exists():
            raise FileNotFoundError(
                f"Файл карты не найден: {self.map_file}"
            )

        with open(
            self.map_file,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        # =====================================================
        # MAP SETTINGS
        # =====================================================

        self.name = data.get(
            "name",
            "Unnamed Level"
        )

        self.width = data.get(
            "width",
            3000
        )

        self.height = data.get(
            "height",
            600
        )

        self.player_spawn = data.get(
            "player_spawn",
            {
                "x": 100,
                "y": 100
            }
        )

        # =====================================================
        # LOAD EDITOR OBJECTS
        # =====================================================

        objects = data.get(
            "objects",
            []
        )

        print(
            f"\n[Map] Загрузка карты: {self.name}"
        )

        print(
            f"[Map] Размер: "
            f"{self.width} x {self.height}"
        )

        print(
            f"[Map] Объектов: "
            f"{len(objects)}"
        )

        for object_data in objects:

            try:
                self.create_editor_object(
                    object_data
                )

            except Exception as error:

                print(
                    f"[Map] Ошибка создания объекта: "
                    f"{object_data}"
                )

                print(
                    f"[Map] Ошибка: {error}"
                )

        print(
            "[Map] Карта успешно загружена"
        )

    # =========================================================
    # CREATE EDITOR OBJECT
    # =========================================================

    def create_editor_object(
        self,
        data: dict
    ) -> None:
        """
        Создаёт объект, сохранённый редактором.
        """

        category = self.get_category(data)

        if not category:

            print(
                f"[Map] Не удалось определить категорию: "
                f"{data}"
            )

            return

        category_lower = category.lower()

        # =====================================================
        # BLOCKS
        # =====================================================

        if category_lower in (
            "blocks",
            "block",
            "блоки"
        ):

            self.create_block_from_editor(
                data
            )

        # =====================================================
        # OBJECTS
        # =====================================================

        elif category_lower in (
            "objects",
            "object",
            "объекты"
        ):

            self.create_object_from_editor(
                data
            )

        # =====================================================
        # DECORATIONS
        # =====================================================

        elif category_lower in (
            "decorations",
            "decoration",
            "декорации",
            "decorates"
        ):

            self.create_decoration_from_editor(
                data
            )

        # =====================================================
        # ENTITIES
        # =====================================================

        elif category_lower in (
            "entities",
            "entity",
            "сущности"
        ):

            self.create_entity_from_editor(
                data
            )

        else:

            print(
                f"[Map] Неизвестная категория: "
                f"{category}"
            )

    # =========================================================
    # CATEGORY
    # =========================================================

    def get_category(
        self,
        data: dict
    ) -> str | None:
        """
        Получает категорию объекта.

        Поддерживает несколько вариантов JSON.
        """

        category = data.get(
            "category"
        )

        if category:
            return str(category)

        category = data.get(
            "type"
        )

        if category:
            return str(category)

        # Иногда resource может быть объектом
        resource = data.get(
            "resource"
        )

        if isinstance(resource, dict):

            category = resource.get(
                "category"
            )

            if category:
                return str(category)

        # Иногда категория может быть внутри
        # resource_category
        category = data.get(
            "resource_category"
        )

        if category:
            return str(category)

        return None

    # =========================================================
    # POSITION
    # =========================================================

    def get_position(
        self,
        data: dict
    ) -> tuple[float, float]:
        """
        Возвращает позицию объекта.
        """

        x = float(
            data.get(
                "x",
                0
            )
        )

        y = float(
            data.get(
                "y",
                0
            )
        )

        return x, y

    # =========================================================
    # TEXTURE PATH
    # =========================================================

    def get_texture_path(
        self,
        data: dict
    ) -> str | None:
        """
        Получает путь к текстуре из объекта карты.

        Поддерживает:

            texture
            texture_path
            path

        а также вложенный resource.
        """

        # =====================================================
        # DIRECT TEXTURE
        # =====================================================

        texture_path = data.get(
            "texture"
        )

        if texture_path:
            return str(texture_path)

        # =====================================================
        # TEXTURE PATH
        # =====================================================

        texture_path = data.get(
            "texture_path"
        )

        if texture_path:
            return str(texture_path)

        # =====================================================
        # PATH
        # =====================================================

        texture_path = data.get(
            "path"
        )

        if texture_path:
            return str(texture_path)

        # =====================================================
        # RESOURCE DICTIONARY
        # =====================================================

        resource = data.get(
            "resource"
        )

        if isinstance(resource, dict):

            texture_path = resource.get(
                "texture"
            )

            if texture_path:
                return str(texture_path)

            texture_path = resource.get(
                "path"
            )

            if texture_path:
                return str(texture_path)

            texture_path = resource.get(
                "texture_path"
            )

            if texture_path:
                return str(texture_path)

        return None

    # =========================================================
    # RESOURCE NAME
    # =========================================================

    def get_resource_name(
        self,
        data: dict
    ) -> str:
        """
        Получает имя ресурса.

        Например:

            resources/blocks/ground/ground.png

        ресурс:

            ground

        Для Entities:

            resources/entities/goomba/idle/0.png

        ресурс:

            goomba
        """

        # =====================================================
        # RESOURCE STRING
        # =====================================================

        resource = data.get(
            "resource"
        )

        # В текущем JSON редактора:
        #
        # "resource": "ground"
        #
        if isinstance(resource, str):

            if resource.strip():
                return resource.strip()

        # =====================================================
        # RESOURCE NAME
        # =====================================================

        name = data.get(
            "resource_name"
        )

        if name:
            return str(name)

        # =====================================================
        # GROUP
        # =====================================================

        name = data.get(
            "group"
        )

        if name:
            return str(name)

        # =====================================================
        # RESOURCE DICTIONARY
        # =====================================================

        if isinstance(resource, dict):

            name = resource.get(
                "name"
            )

            if name:
                return str(name)

            name = resource.get(
                "group"
            )

            if name:
                return str(name)

        # =====================================================
        # TRY TO DETERMINE FROM TEXTURE PATH
        # =====================================================

        texture_path = self.get_texture_path(
            data
        )

        if texture_path:

            path = Path(
                texture_path.replace(
                    "\\",
                    "/"
                )
            )

            parts = path.parts

            # -------------------------------------------------
            # resources/<category>/<resource>/<texture>
            # -------------------------------------------------

            for index, part in enumerate(parts):

                if part.lower() == "resources":

                    if index + 2 < len(parts):

                        return parts[index + 2]

            # -------------------------------------------------
            # Если путь:
            #
            # blocks/ground/ground.png
            # -------------------------------------------------

            if len(parts) >= 3:

                return parts[-2]

            # -------------------------------------------------
            # Если просто:
            #
            # ground.png
            #
            # имя ресурса определить невозможно,
            # поэтому используем имя файла без расширения.
            # -------------------------------------------------

            if len(parts) == 1:

                return path.stem

        return "unknown"

    # =========================================================
    # RESOLVE TEXTURE PATH
    # =========================================================

    def resolve_texture_path(
        self,
        data: dict
    ) -> Path | None:
        """
        Преобразует путь из JSON в реальный путь файла.

        Новый формат:

            resources/blocks/ground/ground.png

        Старые форматы:

            ground.png

            blocks/ground/ground.png

            /absolute/path/to/ground.png
        """

        texture_path = self.get_texture_path(
            data
        )

        if not texture_path:
            return None

        # Нормализуем Windows-разделители
        texture_path = texture_path.replace(
            "\\",
            "/"
        )

        path = Path(
            texture_path
        )

        # =====================================================
        # 1. ABSOLUTE PATH
        # =====================================================

        if path.is_absolute():

            if path.exists():
                return path

            return None

        # =====================================================
        # 2. PROJECT-RELATIVE PATH
        #
        # resources/blocks/ground/ground.png
        # =====================================================

        project_path = BASE_DIR / path

        if project_path.exists():
            return project_path

        # =====================================================
        # 3. RESOURCE ROOT
        #
        # blocks/ground/ground.png
        # =====================================================

        resource_path = RESOURCES_DIR / path

        if resource_path.exists():
            return resource_path

        # =====================================================
        # 4. ONLY FILE NAME
        #
        # ground.png
        #
        # Пытаемся найти файл внутри:
        #
        # resources/<category>/<resource>/
        # =====================================================

        category = self.get_category(
            data
        )

        resource_name = self.get_resource_name(
            data
        )

        if category and resource_name:

            candidate = (
                RESOURCES_DIR
                / category
                / resource_name
                / path
            )

            if candidate.exists():
                return candidate

        # =====================================================
        # 5. SEARCH BY FILE NAME
        # =====================================================

        filename = path.name

        if filename:

            try:

                matches = list(
                    RESOURCES_DIR.rglob(
                        filename
                    )
                )

                if matches:

                    return matches[0]

            except Exception:
                pass

        return None

    # =========================================================
    # CREATE NORMAL SPRITE
    # =========================================================

    def create_sprite_from_editor(
        self,
        data: dict
    ) -> arcade.Sprite | None:
        """
        Создаёт Sprite по данным редактора.

        Если текстура меньше 32x32,
        Sprite автоматически увеличивается
        минимум до 32 пикселей по каждой стороне
        с сохранением пропорций.
        """

        # =====================================================
        # RESOLVE PATH
        # =====================================================

        texture_file = self.resolve_texture_path(
            data
        )

        if texture_file is None:

            print(
                "[Map] Текстура не найдена."
            )

            print(
                f"[Map] Данные объекта: {data}"
            )

            return None

        # =====================================================
        # LOAD TEXTURE
        # =====================================================

        try:

            texture = arcade.load_texture(
                str(texture_file)
            )

        except Exception as error:

            print(
                f"[Map] Ошибка загрузки текстуры "
                f"{texture_file}: {error}"
            )

            return None

        # =====================================================
        # GET TEXTURE SIZE
        # =====================================================

        texture_width = texture.width
        texture_height = texture.height

        # По умолчанию масштаб 1
        scale = 1.0

        # =====================================================
        # MINIMUM 32x32
        # =====================================================

        if (
            texture_width < MIN_TEXTURE_SIZE
            or texture_height < MIN_TEXTURE_SIZE
        ):

            scale_x = (
                MIN_TEXTURE_SIZE
                / texture_width
                if texture_width > 0
                else 1.0
            )

            scale_y = (
                MIN_TEXTURE_SIZE
                / texture_height
                if texture_height > 0
                else 1.0
            )

            # Берём максимальный масштаб,
            # чтобы обе стороны были не меньше 32.
            scale = max(
                scale_x,
                scale_y
            )

        # =====================================================
        # CREATE SPRITE
        # =====================================================

        try:

            # Для Arcade 3.3.3:
            #
            # arcade.Sprite(texture, scale=...)
            #
            # а не:
            #
            # arcade.Sprite(texture=texture, scale=...)
            #
            sprite = arcade.Sprite(
                texture,
                scale=scale
            )

        except Exception as error:

            print(
                f"[Map] Ошибка создания Sprite "
                f"{texture_file}: {error}"
            )

            return None

        # =====================================================
        # POSITION
        # =====================================================

        x, y = self.get_position(
            data
        )

        sprite.center_x = x
        sprite.center_y = y

        # =====================================================
        # DEBUG
        # =====================================================

        if scale != 1.0:

            print(
                f"[Map] Текстура увеличена: "
                f"{texture_file.name} "
                f"{texture_width}x{texture_height}"
                f" -> "
                f"{sprite.width:.0f}x{sprite.height:.0f}"
            )

        return sprite

    # =========================================================
    # BLOCKS
    # =========================================================

    def create_block_from_editor(
        self,
        data: dict
    ) -> None:
        """
        Создаёт блок.

        Все блоки добавляются в walls,
        поэтому Player может взаимодействовать
        с ними через PhysicsEngine.
        """

        sprite = self.create_sprite_from_editor(
            data
        )

        if sprite is None:
            return

        self.walls.append(
            sprite
        )

    # =========================================================
    # OBJECTS
    # =========================================================

    def create_object_from_editor(
        self,
        data: dict
    ) -> None:
        """
        Создаёт обычный объект.

        Пока объекты не имеют физики.
        """

        sprite_name = self.get_resource_name(data)

        if sprite_name == 'Coins':
            sprite = Coin()
        elif sprite_name == 'Flag':
            sprite = Flag()
        else:
            return

        x, y = self.get_position(data)
        sprite.center_x = x
        sprite.center_y = y
        texture_path = self.get_texture_path(data)
        sprite.texture = arcade.load_texture(texture_path)
        self.objects.append(sprite)

    # =========================================================
    # DECORATIONS
    # =========================================================

    def create_decoration_from_editor(
        self,
        data: dict
    ) -> None:
        """
        Создаёт декоративный объект.
        """

        sprite = self.create_sprite_from_editor(
            data
        )

        if sprite is None:
            return

        self.decorations.append(
            sprite
        )

    # =========================================================
    # ENTITIES
    # =========================================================

    def create_entity_from_editor(
        self,
        data: dict
    ) -> None:
        """
        Создаёт сущность.

        Тип сущности определяется по имени ресурса:

            mario
            goomba
            koopa
        """

        entity_name = (
            self.get_resource_name(
                data
            )
            .lower()
        )

        x, y = self.get_position(
            data
        )

        print(
            f"[Map] Entity: "
            f"{entity_name} "
            f"at ({x}, {y})"
        )

        # =====================================================
        # MARIO
        # =====================================================

        if entity_name == "mario":

            self.player_spawn = {
                "x": x,
                "y": y
            }

            print(
                f"[Map] Player spawn установлен: "
                f"{x}, {y}"
            )

            return

        # =====================================================
        # GOOMBA
        # =====================================================

        if entity_name == "goomba":

            enemy = Goomba(
                x=x,
                y=y
            )

            self.enemies.append(
                enemy
            )

            self.entities.append(
                enemy
            )

            return

        # =====================================================
        # KOOPA
        # =====================================================

        if entity_name == "koopa":

            enemy = Koopa(
                x=x,
                y=y
            )

            self.enemies.append(
                enemy
            )

            self.entities.append(
                enemy
            )

            return

        # =====================================================
        # UNKNOWN ENTITY
        # =====================================================

        print(
            f"[Map] Неизвестная сущность: "
            f"{entity_name}"
        )

    # =========================================================
    # GETTERS
    # =========================================================

    def get_player_spawn(
        self
    ) -> tuple[float, float]:

        return (
            self.player_spawn["x"],
            self.player_spawn["y"]
        )

    # ---------------------------------------------------------

    def get_walls(
        self
    ) -> arcade.SpriteList:

        return self.walls

    # ---------------------------------------------------------

    def get_objects(
        self
    ) -> arcade.SpriteList:

        return self.objects

    # ---------------------------------------------------------

    def get_decorations(
        self
    ) -> arcade.SpriteList:

        return self.decorations

    # ---------------------------------------------------------

    def get_enemies(
        self
    ) -> arcade.SpriteList:

        return self.enemies

    # ---------------------------------------------------------

    def get_entities(
        self
    ) -> arcade.SpriteList:

        return self.entities
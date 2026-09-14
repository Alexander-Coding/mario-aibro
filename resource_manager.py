from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import arcade


# ============================================================
# CONSTANTS
# ============================================================

RESOURCES_DIR = Path("resources")

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
}


# ============================================================
# RESOURCE TEXTURE
# ============================================================

@dataclass
class ResourceTexture:
    """
    Одна текстура внутри логического ресурса.

    Например:

        resources/decorations/castle/top.png
        resources/decorations/castle/middle.png
        resources/decorations/castle/bottom.png

    relative_path:
        Путь внутри конкретного ресурса:

        top.png
        idle/idle_1.png

    full_path:
        Полный путь относительно папки проекта:

        resources/decorations/castle/top.png
    """

    # Путь внутри ресурса
    relative_path: str

    # Полный путь к файлу
    path: Path

    # Путь относительно проекта
    full_path: str

    # Загруженная текстура Arcade
    texture: arcade.Texture


# ============================================================
# GAME RESOURCE
# ============================================================

@dataclass
class GameResource:
    """
    Логический игровой ресурс.

    Например:

        category = "blocks"
        name = "bricks"

    Физически:

        resources/blocks/bricks/
    """

    category: str
    name: str
    root_path: Path

    textures: dict[str, ResourceTexture] = field(
        default_factory=dict
    )

    preview_key: str | None = None

    # ========================================================
    # PREVIEW TEXTURE
    # ========================================================

    @property
    def preview_texture(self) -> arcade.Texture | None:

        if self.preview_key is not None:

            asset = self.textures.get(
                self.preview_key
            )

            if asset is not None:
                return asset.texture

        if self.textures:

            return next(
                iter(self.textures.values())
            ).texture

        return None

    # ========================================================
    # FILES
    # ========================================================

    @property
    def files(self) -> list[ResourceTexture]:

        return list(
            self.textures.values()
        )

    # ========================================================
    # GET TEXTURE BY RELATIVE PATH
    # ========================================================

    def get_texture(
        self,
        relative_path: str,
    ) -> arcade.Texture | None:

        asset = self.textures.get(
            relative_path
        )

        if asset is None:
            return None

        return asset.texture

    # ========================================================
    # GET RESOURCE TEXTURE
    # ========================================================

    def get_resource_texture(
        self,
        relative_path: str,
    ) -> ResourceTexture | None:

        return self.textures.get(
            relative_path
        )

    # ========================================================
    # GET TEXTURE BY FULL PATH
    # ========================================================

    def get_texture_by_full_path(
        self,
        full_path: str,
    ) -> arcade.Texture | None:

        normalized = (
            Path(full_path)
            .as_posix()
        )

        for asset in self.textures.values():

            if (
                Path(asset.full_path).as_posix()
                == normalized
            ):
                return asset.texture

        return None

    # ========================================================
    # GET ASSET BY FULL PATH
    # ========================================================

    def get_resource_texture_by_full_path(
        self,
        full_path: str,
    ) -> ResourceTexture | None:

        normalized = (
            Path(full_path)
            .as_posix()
        )

        for asset in self.textures.values():

            if (
                Path(asset.full_path).as_posix()
                == normalized
            ):
                return asset

        return None

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:

        return (
            f"GameResource("
            f"category={self.category!r}, "
            f"name={self.name!r}, "
            f"textures={len(self.textures)}"
            f")"
        )


# ============================================================
# RESOURCE MANAGER
# ============================================================

class ResourceManager:
    """
    Загружает ресурсы из структуры:

    resources/
    ├── entities/
    │   ├── mario/
    │   │   ├── idle/
    │   │   │   └── idle.png
    │   │   └── run/
    │   │       └── run.png
    │   │
    │   ├── goomba/
    │   └── koopa/
    │
    ├── decorations/
    │   ├── castle/
    │   ├── cloud/
    │   └── bush/
    │
    ├── objects/
    │   ├── coins/
    │   └── flag/
    │
    └── blocks/
        ├── bricks/
        └── tubes/

    Каждая папка непосредственно внутри категории —
    отдельный логический GameResource.
    """

    def __init__(
        self,
        resources_dir: Path | str = RESOURCES_DIR,
    ):

        self.resources_dir = Path(
            resources_dir
        )

        # Абсолютный путь к папке проекта
        self.project_root = Path.cwd()

        # Все ресурсы:
        #
        # {
        #     ("blocks", "bricks"): GameResource(...)
        # }
        self.resources: dict[
            tuple[str, str],
            GameResource
        ] = {}

        # Список категорий
        self.categories: list[str] = []

        self.load()

    # ========================================================
    # LOAD
    # ========================================================

    def load(self) -> None:

        self.resources.clear()
        self.categories.clear()

        if not self.resources_dir.exists():

            print(
                "[ResourceManager] "
                "Папка ресурсов не найдена: "
                f"{self.resources_dir}"
            )

            return

        category_dirs = sorted(
            [
                path
                for path in self.resources_dir.iterdir()
                if path.is_dir()
            ],
            key=lambda path: path.name.lower(),
        )

        # ----------------------------------------------------
        # PREFERRED CATEGORY ORDER
        # ----------------------------------------------------

        preferred_order = [
            "entities",
            "decorations",
            "objects",
            "blocks",
        ]

        ordered_categories: list[Path] = []

        # Сначала категории в нужном порядке
        for category_name in preferred_order:

            for path in category_dirs:

                if (
                    path.name.lower()
                    == category_name.lower()
                ):

                    ordered_categories.append(
                        path
                    )

        # Затем остальные категории
        for path in category_dirs:

            if path not in ordered_categories:

                ordered_categories.append(
                    path
                )

        self.categories = [
            category_dir.name
            for category_dir in ordered_categories
        ]

        # ----------------------------------------------------
        # LOAD EVERY CATEGORY
        # ----------------------------------------------------

        for category_dir in ordered_categories:

            self._load_category(
                category_dir
            )

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        print(
            "[ResourceManager] Загружены ресурсы:"
        )

        for category in self.categories:

            resources = (
                self.get_category_resources(
                    category
                )
            )

            names = ", ".join(
                resource.name
                for resource in resources
            )

            print(
                f"  {category}: {names}"
            )

    # ========================================================
    # LOAD CATEGORY
    # ========================================================

    def _load_category(
        self,
        category_dir: Path,
    ) -> None:

        category = category_dir.name

        resource_dirs = sorted(
            [
                path
                for path in category_dir.iterdir()
                if path.is_dir()
            ],
            key=lambda path: path.name.lower(),
        )

        for resource_dir in resource_dirs:

            resource = self._load_resource(
                category=category,
                resource_dir=resource_dir,
            )

            if resource is None:
                continue

            key = (
                resource.category,
                resource.name,
            )

            self.resources[key] = resource

    # ========================================================
    # LOAD RESOURCE
    # ========================================================

    def _load_resource(
        self,
        category: str,
        resource_dir: Path,
    ) -> GameResource | None:

        image_files = sorted(
            [
                path
                for path in resource_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )
            ],
            key=lambda path: (
                path.as_posix().lower()
            ),
        )

        if not image_files:

            print(
                "[ResourceManager] "
                f"Ресурс "
                f"{category}/{resource_dir.name} "
                f"не содержит изображений"
            )

            return None

        # ----------------------------------------------------
        # CREATE RESOURCE
        # ----------------------------------------------------

        resource = GameResource(
            category=category,
            name=resource_dir.name,
            root_path=resource_dir,
        )

        # ----------------------------------------------------
        # LOAD TEXTURES
        # ----------------------------------------------------

        for path in image_files:

            try:

                texture = arcade.load_texture(
                    str(path)
                )

            except Exception as exc:

                print(
                    "[ResourceManager] "
                    f"Не удалось загрузить "
                    f"{path}: {exc}"
                )

                continue

            # ------------------------------------------------
            # PATH INSIDE RESOURCE
            #
            # resources/entities/mario/idle/a.png
            #
            # -> idle/a.png
            # ------------------------------------------------

            relative_path = (
                path.relative_to(
                    resource_dir
                )
                .as_posix()
            )

            # ------------------------------------------------
            # FULL PATH RELATIVE TO PROJECT
            #
            # resources/entities/mario/idle/a.png
            # ------------------------------------------------

            try:

                full_path = (
                    path.relative_to(
                        self.project_root
                    )
                    .as_posix()
                )

            except ValueError:

                # Если resources находится вне cwd,
                # сохраняем обычный POSIX путь.
                full_path = path.as_posix()

            # ------------------------------------------------
            # CREATE RESOURCE TEXTURE
            # ------------------------------------------------

            resource.textures[
                relative_path
            ] = ResourceTexture(
                relative_path=relative_path,
                path=path,
                full_path=full_path,
                texture=texture,
            )

        # ----------------------------------------------------
        # EMPTY RESOURCE
        # ----------------------------------------------------

        if not resource.textures:

            return None

        # ----------------------------------------------------
        # FIND PREVIEW
        # ----------------------------------------------------

        resource.preview_key = (
            self._find_preview_key(
                resource
            )
        )

        return resource

    # ========================================================
    # FIND PREVIEW
    # ========================================================

    def _find_preview_key(
        self,
        resource: GameResource,
    ) -> str | None:

        keys = list(
            resource.textures.keys()
        )

        if not keys:
            return None

        # ----------------------------------------------------
        # PRIORITY FILE NAMES
        # ----------------------------------------------------

        preferred_names = [
            "preview.png",
            "preview.jpg",
            "preview.jpeg",
            "preview.webp",

            "icon.png",
            "icon.jpg",
            "icon.jpeg",
            "icon.webp",

            "thumbnail.png",
            "thumbnail.jpg",
            "thumbnail.jpeg",
            "thumbnail.webp",
        ]

        for preferred_name in preferred_names:

            for key in keys:

                filename = (
                    Path(key)
                    .name
                    .lower()
                )

                if filename == preferred_name:

                    return key

        # ----------------------------------------------------
        # ENTITIES -> IDLE FIRST
        # ----------------------------------------------------

        if (
            resource.category.lower()
            == "entities"
        ):

            idle_keys = [
                key
                for key in keys
                if (
                    Path(key).parts
                    and
                    Path(key).parts[0].lower()
                    == "idle"
                )
            ]

            if idle_keys:

                return sorted(
                    idle_keys,
                    key=lambda value:
                    value.lower(),
                )[0]

        # ----------------------------------------------------
        # IDLE FOR OTHER RESOURCES
        # ----------------------------------------------------

        idle_keys = [
            key
            for key in keys
            if (
                Path(key).parts
                and
                Path(key).parts[0].lower()
                == "idle"
            )
        ]

        if idle_keys:

            return sorted(
                idle_keys,
                key=lambda value:
                value.lower(),
            )[0]

        # ----------------------------------------------------
        # FILES IN RESOURCE ROOT
        # ----------------------------------------------------

        root_files = [
            key
            for key in keys
            if len(
                Path(key).parts
            ) == 1
        ]

        if root_files:

            return sorted(
                root_files,
                key=lambda value:
                value.lower(),
            )[0]

        # ----------------------------------------------------
        # FIRST FILE
        # ----------------------------------------------------

        return sorted(
            keys,
            key=lambda value:
            value.lower(),
        )[0]

    # ========================================================
    # GET RESOURCE
    # ========================================================

    def get(
        self,
        category: str,
        name: str,
    ) -> GameResource | None:

        return self.resources.get(
            (
                category,
                name,
            )
        )

    # ========================================================
    # REQUIRE RESOURCE
    # ========================================================

    def require(
        self,
        category: str,
        name: str,
    ) -> GameResource:

        resource = self.get(
            category,
            name,
        )

        if resource is None:

            raise KeyError(
                "Ресурс не найден: "
                f"{category}/{name}"
            )

        return resource

    # ========================================================
    # GET CATEGORY RESOURCES
    # ========================================================

    def get_category_resources(
        self,
        category: str,
    ) -> list[GameResource]:

        result = [
            resource
            for (
                resource_category,
                _,
            ), resource in self.resources.items()
            if resource_category == category
        ]

        result.sort(
            key=lambda resource:
            resource.name.lower()
        )

        return result

    # ========================================================
    # GET ALL RESOURCES
    # ========================================================

    def get_all_resources(
        self,
    ) -> list[GameResource]:

        return list(
            self.resources.values()
        )

    # ========================================================
    # HAS RESOURCE
    # ========================================================

    def has(
        self,
        category: str,
        name: str,
    ) -> bool:

        return (
            category,
            name,
        ) in self.resources

    # ========================================================
    # FIND TEXTURE BY FULL PATH
    # ========================================================

    def find_texture(
        self,
        texture_path: str,
    ) -> ResourceTexture | None:
        """
        Ищет текстуру по полному пути.

        Например:

        resources/blocks/bricks/brick.png
        """

        normalized_path = (
            Path(texture_path)
            .as_posix()
        )

        for resource in self.resources.values():

            for asset in resource.textures.values():

                if (
                    Path(asset.full_path).as_posix()
                    == normalized_path
                ):

                    return asset

        return None

    # ========================================================
    # GET TEXTURE BY FULL PATH
    # ========================================================

    def get_texture(
        self,
        texture_path: str,
    ) -> arcade.Texture | None:

        asset = self.find_texture(
            texture_path
        )

        if asset is None:
            return None

        return asset.texture

    # ========================================================
    # FIND RESOURCE BY TEXTURE PATH
    # ========================================================

    def find_resource_by_texture_path(
        self,
        texture_path: str,
    ) -> GameResource | None:

        normalized_path = (
            Path(texture_path)
            .as_posix()
        )

        for resource in self.resources.values():

            for asset in resource.textures.values():

                if (
                    Path(asset.full_path).as_posix()
                    == normalized_path
                ):

                    return resource

        return None
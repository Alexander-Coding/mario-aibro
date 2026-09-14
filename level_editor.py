from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import arcade

from resource_manager import (
    ResourceManager,
    GameResource,
    ResourceTexture,
)


# ============================================================
# CONSTANTS
# ============================================================

GRID_SIZE = 32

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

TOP_BAR_HEIGHT = 58
BOTTOM_BAR_HEIGHT = 34

SIDEBAR_WIDTH = 360
CATEGORY_WIDTH = 110

PALETTE_CARD_HEIGHT = 58
PALETTE_GAP = 8

INSPECTOR_ITEM_HEIGHT = 58
INSPECTOR_GAP = 8

PREVIEW_SIZE = 34

# Папка ресурсов.
RESOURCES_DIR = Path("resources")

# Минимальная ширина карты.
MAP_MIN_WIDTH = 1280

# Пустое пространство справа от последнего объекта.
MAP_PADDING_CELLS = 10

# Файл карты.
MAP_FILE = Path("maps/custom_level.json")


# ============================================================
# CAMERA
# ============================================================

CAMERA_SMOOTH_SPEED = 12.0
CAMERA_SCROLL_SPEED = 220.0


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PaletteItem:
    resource: GameResource
    texture: arcade.Texture


@dataclass
class EditorObject:
    category: str
    name: str
    x: float
    y: float
    texture_path: str | None = None
    sprite: arcade.Sprite | None = None


# ============================================================
# LEVEL EDITOR
# ============================================================

class LevelEditorView(arcade.View):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # RESOURCE MANAGER
        # ----------------------------------------------------

        self.resource_manager = ResourceManager()

        # ----------------------------------------------------
        # SELECTION
        # ----------------------------------------------------

        self.selected_category: str | None = None
        self.selected_resource: GameResource | None = None
        self.selected_texture_path: str | None = None

        # ----------------------------------------------------
        # OBJECTS
        # ----------------------------------------------------

        self.objects: list[EditorObject] = []

        self.object_sprites = arcade.SpriteList()

        # ----------------------------------------------------
        # CAMERAS
        # ----------------------------------------------------

        self.camera = arcade.Camera2D()
        self.ui_camera = arcade.Camera2D()

        self.camera_x = 0.0
        self.camera_y = 0.0

        self.target_camera_x = 0.0

        # ----------------------------------------------------
        # MOUSE
        # ----------------------------------------------------

        self.mouse_x = 0
        self.mouse_y = 0

        self.world_mouse_x = 0.0
        self.world_mouse_y = 0.0

        # ----------------------------------------------------
        # SHIFT + DRAG
        # ----------------------------------------------------

        self.is_shift_dragging = False

        self.drag_start_x: float | None = None
        self.drag_start_y: float | None = None

        self.drag_current_x: float | None = None
        self.drag_current_y: float | None = None

        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        self.shift_pressed = False

        # ----------------------------------------------------
        # SCROLL
        # ----------------------------------------------------

        self.palette_scroll = 0
        self.inspector_scroll = 0

        # ----------------------------------------------------
        # LAYOUT
        # ----------------------------------------------------

        self.window_width = DEFAULT_WIDTH
        self.window_height = DEFAULT_HEIGHT

        self.category_left = 0
        self.category_right = 0

        self.sidebar_left = 0
        self.sidebar_right = 0

        self.world_left = 0
        self.world_right = 0
        self.world_bottom = 0
        self.world_top = 0

        self.inspector_bottom = 0
        self.inspector_top = 0
        self.inspector_height = 0

        self.palette_bottom = 0
        self.palette_top = 0

        # ----------------------------------------------------
        # PALETTE
        # ----------------------------------------------------

        self.palette_items: list[PaletteItem] = []

        # ----------------------------------------------------
        # HOVER
        # ----------------------------------------------------

        self.hover_texture: arcade.Texture | None = None
        self.hover_scale = 1.0

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.title_text = arcade.Text(
            text="Редактор уровня",
            x=16,
            y=20,
            font_size=24,
            color=arcade.color.WHITE,
        )

        self.category_title_text = arcade.Text(
            text="Категории",
            x=16,
            y=0,
            font_size=16,
            color=arcade.color.WHITE,
        )

        self.resources_title_text = arcade.Text(
            text="Ресурсы",
            x=0,
            y=0,
            font_size=18,
            color=arcade.color.WHITE,
        )

        self.inspector_title_text = arcade.Text(
            text="Состав ресурса",
            x=0,
            y=0,
            font_size=18,
            color=arcade.color.WHITE,
        )

        self.status_text = arcade.Text(
            text="",
            x=470,
            y=15,
            font_size=13,
            color=arcade.color.WHITE,
        )

        self.save_button = arcade.Text(
            text="Сохранить",
            x=150,
            y=15,
            font_size=14,
            color=arcade.color.WHITE,
        )

        self.load_button = arcade.Text(
            text="Загрузить",
            x=250,
            y=15,
            font_size=14,
            color=arcade.color.WHITE,
        )

        self.clear_button = arcade.Text(
            text="Очистить",
            x=350,
            y=15,
            font_size=14,
            color=arcade.color.WHITE,
        )

        # ----------------------------------------------------
        # LAYOUT
        # ----------------------------------------------------

        self.recalculate_layout()

        # ----------------------------------------------------
        # DEFAULT CATEGORY
        # ----------------------------------------------------

        if self.resource_manager.categories:
            self.select_category(
                self.resource_manager.categories[0]
            )

    # ========================================================
    # MAP WIDTH
    # ========================================================

    def get_map_width(self) -> float:

        if not self.objects:
            return max(
                MAP_MIN_WIDTH,
                self.get_world_view_width(),
            )

        max_x = max(
            obj.x
            for obj in self.objects
        )

        max_cell = int(
            max_x // GRID_SIZE
        )

        width_cells = (
            max_cell
            + 1
            + MAP_PADDING_CELLS
        )

        return max(
            MAP_MIN_WIDTH,
            width_cells * GRID_SIZE,
            self.get_world_view_width(),
        )

    # ========================================================
    # VIEWPORT WIDTH
    # ========================================================

    def get_world_view_width(self) -> float:

        return max(
            1,
            self.sidebar_left,
        )

    # ========================================================
    # CAMERA LIMITS
    # ========================================================

    def get_camera_min_x(self) -> float:

        return (
            self.get_world_view_width()
            / 2
        )

    def get_camera_max_x(self) -> float:

        viewport_width = (
            self.get_world_view_width()
        )

        map_width = (
            self.get_map_width()
        )

        return max(
            viewport_width / 2,
            map_width
            - viewport_width / 2,
        )

    # ========================================================
    # CLAMP CAMERA
    # ========================================================

    def clamp_camera(self):

        min_x = self.get_camera_min_x()
        max_x = self.get_camera_max_x()

        self.target_camera_x = max(
            min_x,
            min(
                self.target_camera_x,
                max_x,
            ),
        )

        self.camera_x = max(
            min_x,
            min(
                self.camera_x,
                max_x,
            ),
        )

        self.camera.position = (
            self.camera_x,
            self.camera_y,
        )

    # ========================================================
    # LAYOUT
    # ========================================================

    def recalculate_layout(self):

        old_camera_x = self.camera_x

        self.window_width = self.window.width
        self.window_height = self.window.height

        w = self.window_width
        h = self.window_height

        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        self.sidebar_left = (
            w - SIDEBAR_WIDTH
        )

        self.sidebar_right = w

        self.category_left = (
            self.sidebar_left
        )

        self.category_right = (
            self.sidebar_left
            + CATEGORY_WIDTH
        )

        # ----------------------------------------------------
        # WORLD SCREEN AREA
        # ----------------------------------------------------

        self.world_left = 0

        self.world_right = (
            self.sidebar_left
        )

        self.world_bottom = (
            BOTTOM_BAR_HEIGHT
        )

        self.world_top = (
            h - TOP_BAR_HEIGHT
        )

        # ----------------------------------------------------
        # INSPECTOR
        # ----------------------------------------------------

        self.inspector_height = max(
            260,
            min(
                330,
                int(h * 0.46),
            ),
        )

        max_inspector_height = max(
            180,
            self.world_top
            - self.world_bottom
            - 100,
        )

        self.inspector_height = min(
            self.inspector_height,
            max_inspector_height,
        )

        self.inspector_bottom = (
            self.world_bottom
        )

        self.inspector_top = (
            self.inspector_bottom
            + self.inspector_height
        )

        # ----------------------------------------------------
        # PALETTE
        # ----------------------------------------------------

        self.palette_bottom = (
            self.inspector_top
        )

        self.palette_top = (
            self.world_top
        )

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        self.title_text.x = 16
        self.title_text.y = h - 40

        self.category_title_text.x = (
            self.category_left + 12
        )

        self.category_title_text.y = (
            self.world_top - 30
        )

        self.resources_title_text.x = (
            self.category_right + 12
        )

        self.resources_title_text.y = (
            self.palette_top - 30
        )

        self.inspector_title_text.x = (
            self.category_left + 12
        )

        self.inspector_title_text.y = (
            self.inspector_top - 30
        )

        # ----------------------------------------------------
        # UI CAMERA
        # ----------------------------------------------------

        self.ui_camera.position = (
            w / 2,
            h / 2,
        )

        # ----------------------------------------------------
        # WORLD CAMERA
        # ----------------------------------------------------

        self.camera_y = (
            self.world_bottom
            + self.world_top
        ) / 2

        # Первый запуск.
        if self.camera_x == 0:

            self.camera_x = (
                self.get_camera_min_x()
            )

        else:

            self.camera_x = old_camera_x

        if self.target_camera_x == 0:

            self.target_camera_x = (
                self.camera_x
            )

        self.clamp_camera()

    # ========================================================
    # UPDATE
    # ========================================================

    def on_update(
        self,
        delta_time: float,
    ):

        difference = (
            self.target_camera_x
            - self.camera_x
        )

        factor = 1.0 - pow(
            0.001,
            delta_time * CAMERA_SMOOTH_SPEED,
        )

        self.camera_x += (
            difference * factor
        )

        if abs(difference) < 0.01:
            self.camera_x = (
                self.target_camera_x
            )

        self.clamp_camera()

    # ========================================================
    # CATEGORY
    # ========================================================

    def select_category(
        self,
        category: str,
    ):

        self.selected_category = category

        resources = (
            self.resource_manager
            .get_category_resources(
                category
            )
        )

        if category.lower() == "entities":

            resources = [
                resource
                for resource in resources
                if any(
                    Path(asset.relative_path).parts
                    and
                    Path(asset.relative_path)
                    .parts[0]
                    .lower()
                    == "idle"
                    for asset in resource.files
                )
            ]

        self.palette_items.clear()

        for resource in resources:

            texture = (
                resource.preview_texture
            )

            if texture is None:
                continue

            self.palette_items.append(
                PaletteItem(
                    resource=resource,
                    texture=texture,
                )
            )

        self.palette_scroll = 0

        if self.palette_items:

            self.select_resource(
                self.palette_items[0].resource
            )

        else:

            self.selected_resource = None
            self.selected_texture_path = None
            self.inspector_scroll = 0

    # ========================================================
    # RESOURCE
    # ========================================================

    def select_resource(
        self,
        resource: GameResource,
    ):

        self.selected_resource = resource

        self.selected_texture_path = None

        files = self.get_resource_files(
            resource
        )

        if files:

            self.selected_texture_path = (
                files[0].relative_path
            )

        self.inspector_scroll = 0

        self.status_text.text = (
            f"Выбран: {resource.name}"
        )

    # ========================================================
    # RESOURCE FILES
    # ========================================================

    def get_resource_files(
        self,
        resource: GameResource,
    ) -> list[ResourceTexture]:

        files = resource.files

        if resource.category.lower() == "entities":

            files = [
                asset
                for asset in files
                if (
                    Path(asset.relative_path).parts
                    and
                    Path(asset.relative_path)
                    .parts[0]
                    .lower()
                    == "idle"
                )
            ]

        return sorted(
            files,
            key=lambda asset:
            asset.relative_path.lower(),
        )

    # ========================================================
    # SELECTED ASSET
    # ========================================================

    def get_selected_asset(
        self,
    ) -> ResourceTexture | None:

        if self.selected_resource is None:
            return None

        if self.selected_texture_path is None:
            return None

        for asset in self.selected_resource.files:

            if (
                asset.relative_path
                == self.selected_texture_path
            ):

                return asset

        return None

    # ========================================================
    # TEXTURE SCALE
    # ========================================================

    def get_texture_scale(
        self,
        texture: arcade.Texture,
    ) -> float:

        width = texture.width
        height = texture.height

        if (
            width < GRID_SIZE
            or height < GRID_SIZE
        ):

            scale_x = (
                GRID_SIZE / width
                if width < GRID_SIZE
                else 1.0
            )

            scale_y = (
                GRID_SIZE / height
                if height < GRID_SIZE
                else 1.0
            )

            return max(
                scale_x,
                scale_y,
            )

        return 1.0

    # ========================================================
    # GRID SNAP
    # ========================================================

    def snap_to_grid(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:

        cell_x = (
            int(x // GRID_SIZE)
            * GRID_SIZE
        )

        cell_y = (
            int(y // GRID_SIZE)
            * GRID_SIZE
        )

        return (
            cell_x + GRID_SIZE / 2,
            cell_y + GRID_SIZE / 2,
        )

    # ========================================================
    # CELL
    # ========================================================

    def get_cell(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int]:

        return (
            int(x // GRID_SIZE),
            int(y // GRID_SIZE),
        )

    # ========================================================
    # WORLD MOUSE
    # ========================================================

    def update_world_mouse(
        self,
        x: float,
        y: float,
    ):

        self.mouse_x = x
        self.mouse_y = y

        result = self.camera.unproject(
            (x, y)
        )

        self.world_mouse_x = result[0]
        self.world_mouse_y = result[1]

    # ========================================================
    # DRAW
    # ========================================================

    def on_draw(self):

        self.clear()

        # ----------------------------------------------------
        # WORLD
        # ----------------------------------------------------

        self.camera.use()

        self.draw_world()
        self.draw_grid()

        self.object_sprites.draw()

        self.draw_drag_preview()
        self.draw_hover_object()

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.ui_camera.use()

        self.draw_ui()

    # ========================================================
    # WORLD
    # ========================================================

    def draw_world(self):

        viewport_width = (
            self.window_width
        )

        viewport_height = (
            self.window_height
        )

        left = (
            self.camera_x
            - viewport_width / 2
            - GRID_SIZE * 4
        )

        right = (
            self.camera_x
            + viewport_width / 2
            + GRID_SIZE * 4
        )

        bottom = (
            self.camera_y
            - viewport_height / 2
            - GRID_SIZE * 4
        )

        top = (
            self.camera_y
            + viewport_height / 2
            + GRID_SIZE * 4
        )

        arcade.draw_lrbt_rectangle_filled(
            left,
            right,
            bottom,
            top,
            arcade.color.SKY_BLUE,
        )

    # ========================================================
    # GRID
    # ========================================================

    def draw_grid(self):

        viewport_width = (
            self.window_width
        )

        viewport_height = (
            self.window_height
        )

        left = (
            self.camera_x
            - viewport_width / 2
            - GRID_SIZE
        )

        right = (
            self.camera_x
            + viewport_width / 2
            + GRID_SIZE
        )

        bottom = (
            self.camera_y
            - viewport_height / 2
            - GRID_SIZE
        )

        top = (
            self.camera_y
            + viewport_height / 2
            + GRID_SIZE
        )

        bottom = max(
            0,
            bottom,
        )

        map_width = self.get_map_width()

        start_x = max(
            0,
            int(left // GRID_SIZE)
            * GRID_SIZE,
        )

        end_x = min(
            map_width,
            (
                int(right // GRID_SIZE)
                + 1
            ) * GRID_SIZE,
        )

        for x in range(
            int(start_x),
            int(end_x) + 1,
            GRID_SIZE,
        ):

            arcade.draw_line(
                x,
                bottom,
                x,
                top,
                arcade.color.LIGHT_GRAY,
                1,
            )

        start_y = (
            int(bottom // GRID_SIZE)
            * GRID_SIZE
        )

        end_y = (
            int(top // GRID_SIZE)
            + 1
        ) * GRID_SIZE

        for y in range(
            start_y,
            end_y + 1,
            GRID_SIZE,
        ):

            arcade.draw_line(
                0,
                y,
                map_width,
                y,
                arcade.color.LIGHT_GRAY,
                1,
            )

    # ========================================================
    # UI
    # ========================================================

    def draw_ui(self):

        w = self.window_width
        h = self.window_height

        # ----------------------------------------------------
        # TOP BAR
        # ----------------------------------------------------

        arcade.draw_lrbt_rectangle_filled(
            0,
            w,
            h - TOP_BAR_HEIGHT,
            h,
            arcade.color.DARK_BLUE,
        )

        self.title_text.draw()

        # ----------------------------------------------------
        # BOTTOM BAR
        # ----------------------------------------------------

        arcade.draw_lrbt_rectangle_filled(
            0,
            w,
            0,
            BOTTOM_BAR_HEIGHT,
            arcade.color.DARK_GRAY,
        )

        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        arcade.draw_lrbt_rectangle_filled(
            self.sidebar_left,
            self.sidebar_right,
            0,
            h,
            arcade.color.DIM_GRAY,
        )

        # ----------------------------------------------------
        # CATEGORIES
        # ----------------------------------------------------

        arcade.draw_lrbt_rectangle_filled(
            self.category_left,
            self.category_right,
            self.world_bottom,
            self.world_top,
            arcade.color.DARK_SLATE_GRAY,
        )

        self.category_title_text.draw()

        self.draw_categories()

        # ----------------------------------------------------
        # PALETTE
        # ----------------------------------------------------

        self.resources_title_text.draw()

        self.draw_palette()

        # ----------------------------------------------------
        # INSPECTOR
        # ----------------------------------------------------

        arcade.draw_lrbt_rectangle_filled(
            self.category_left,
            self.sidebar_right,
            self.inspector_bottom,
            self.inspector_top,
            arcade.color.DARK_GRAY,
        )

        self.inspector_title_text.draw()

        self.draw_inspector()

        # ----------------------------------------------------
        # BUTTONS
        # ----------------------------------------------------

        self.save_button.draw()
        self.load_button.draw()
        self.clear_button.draw()

        self.status_text.draw()

    # ========================================================
    # CATEGORIES
    # ========================================================

    def draw_categories(self):

        start_y = (
            self.world_top - 70
        )

        for index, category in enumerate(
            self.resource_manager.categories
        ):

            y = (
                start_y
                - index * 42
            )

            if (
                y + 32
                < self.inspector_top
            ):
                break

            selected = (
                category
                == self.selected_category
            )

            if selected:

                arcade.draw_lrbt_rectangle_filled(
                    self.category_left + 6,
                    self.category_right - 6,
                    y - 2,
                    y + 32,
                    arcade.color.BLUE,
                )

            text = arcade.Text(
                text=category,
                x=self.category_left + 12,
                y=y + 7,
                font_size=12,
                color=arcade.color.WHITE,
            )

            text.draw()

    # ========================================================
    # PALETTE
    # ========================================================

    def draw_palette(self):

        left = (
            self.category_right + 8
        )

        right = (
            self.sidebar_right - 8
        )

        content_bottom = (
            self.palette_bottom
        )

        content_top_limit = (
            self.palette_top - 42
        )

        arcade.draw_lrbt_rectangle_filled(
            left,
            right,
            content_bottom,
            content_top_limit,
            arcade.color.DIM_GRAY,
        )

        content_top = (
            content_top_limit
            + self.palette_scroll
        )

        for index, item in enumerate(
            self.palette_items
        ):

            top = (
                content_top
                - index
                * (
                    PALETTE_CARD_HEIGHT
                    + PALETTE_GAP
                )
            )

            bottom = (
                top
                - PALETTE_CARD_HEIGHT
            )

            if (
                bottom > content_top_limit
                or top < content_bottom
            ):
                continue

            selected = (
                self.selected_resource
                is item.resource
            )

            arcade.draw_lrbt_rectangle_filled(
                left,
                right,
                bottom,
                top,
                (
                    arcade.color.BLUE
                    if selected
                    else arcade.color.GRAY
                ),
            )

            texture = item.texture

            icon_scale = min(
                PREVIEW_SIZE
                / texture.width,
                PREVIEW_SIZE
                / texture.height,
            )

            icon_w = (
                texture.width
                * icon_scale
            )

            icon_h = (
                texture.height
                * icon_scale
            )

            icon_x = (
                left
                + 12
                + icon_w / 2
            )

            icon_y = (
                bottom
                + PALETTE_CARD_HEIGHT / 2
            )

            arcade.draw_texture_rect(
                texture,
                arcade.LRBT(
                    icon_x - icon_w / 2,
                    icon_x + icon_w / 2,
                    icon_y - icon_h / 2,
                    icon_y + icon_h / 2,
                ),
            )

            name_text = arcade.Text(
                text=item.resource.name,
                x=left + 58,
                y=icon_y - 7,
                font_size=13,
                color=arcade.color.WHITE,
            )

            name_text.draw()

    # ========================================================
    # INSPECTOR
    # ========================================================

    def draw_inspector(self):

        left = (
            self.category_left + 8
        )

        right = (
            self.sidebar_right - 8
        )

        if self.selected_resource is None:

            text = arcade.Text(
                text="Ресурс не выбран",
                x=left + 8,
                y=self.inspector_top - 65,
                font_size=13,
                color=arcade.color.WHITE,
            )

            text.draw()

            return

        files = self.get_resource_files(
            self.selected_resource
        )

        if not files:

            text = arcade.Text(
                text="Нет доступных элементов",
                x=left + 8,
                y=self.inspector_top - 65,
                font_size=13,
                color=arcade.color.WHITE,
            )

            text.draw()

            return

        content_top = (
            self.inspector_top
            - 52
            + self.inspector_scroll
        )

        for index, asset in enumerate(
            files
        ):

            top = (
                content_top
                - index
                * (
                    INSPECTOR_ITEM_HEIGHT
                    + INSPECTOR_GAP
                )
            )

            bottom = (
                top
                - INSPECTOR_ITEM_HEIGHT
            )

            if (
                bottom > self.inspector_top - 45
                or top < self.inspector_bottom
            ):
                continue

            selected = (
                asset.relative_path
                == self.selected_texture_path
            )

            arcade.draw_lrbt_rectangle_filled(
                left,
                right,
                bottom,
                top,
                (
                    arcade.color.BLUE
                    if selected
                    else arcade.color.GRAY
                ),
            )

            texture = asset.texture

            icon_scale = min(
                PREVIEW_SIZE
                / texture.width,
                PREVIEW_SIZE
                / texture.height,
            )

            icon_w = (
                texture.width
                * icon_scale
            )

            icon_h = (
                texture.height
                * icon_scale
            )

            icon_x = (
                left
                + 12
                + icon_w / 2
            )

            icon_y = (
                bottom
                + INSPECTOR_ITEM_HEIGHT / 2
            )

            arcade.draw_texture_rect(
                texture,
                arcade.LRBT(
                    icon_x - icon_w / 2,
                    icon_x + icon_w / 2,
                    icon_y - icon_h / 2,
                    icon_y + icon_h / 2,
                ),
            )

            file_text = arcade.Text(
                text=asset.relative_path,
                x=left + 58,
                y=icon_y - 7,
                font_size=11,
                color=arcade.color.WHITE,
            )

            file_text.draw()

    # ========================================================
    # HOVER
    # ========================================================

    def draw_hover_object(self):

        if self.hover_texture is None:
            return

        if self.is_shift_dragging:
            return

        if not (
            self.world_left
            <= self.mouse_x
            <= self.world_right
            and
            self.world_bottom
            <= self.mouse_y
            <= self.world_top
        ):
            return

        x, y = self.snap_to_grid(
            self.world_mouse_x,
            self.world_mouse_y,
        )

        texture = self.hover_texture

        scale = self.hover_scale

        width = (
            texture.width
            * scale
        )

        height = (
            texture.height
            * scale
        )

        arcade.draw_texture_rect(
            texture,
            arcade.LRBT(
                x - width / 2,
                x + width / 2,
                y - height / 2,
                y + height / 2,
            ),
        )

    # ========================================================
    # DRAG PREVIEW
    # ========================================================

    def draw_drag_preview(self):

        if not self.is_shift_dragging:
            return

        asset = self.get_selected_asset()

        if asset is None:
            return

        if (
            self.drag_start_x is None
            or self.drag_start_y is None
            or self.drag_current_x is None
            or self.drag_current_y is None
        ):
            return

        start_cell_x, start_cell_y = (
            self.get_cell(
                self.drag_start_x,
                self.drag_start_y,
            )
        )

        current_cell_x, current_cell_y = (
            self.get_cell(
                self.drag_current_x,
                self.drag_current_y,
            )
        )

        min_x = min(
            start_cell_x,
            current_cell_x,
        )

        max_x = max(
            start_cell_x,
            current_cell_x,
        )

        min_y = min(
            start_cell_y,
            current_cell_y,
        )

        max_y = max(
            start_cell_y,
            current_cell_y,
        )

        texture = asset.texture

        scale = self.get_texture_scale(
            texture
        )

        width = (
            texture.width
            * scale
        )

        height = (
            texture.height
            * scale
        )

        for cell_y in range(
            min_y,
            max_y + 1,
        ):

            for cell_x in range(
                min_x,
                max_x + 1,
            ):

                center_x = (
                    cell_x
                    * GRID_SIZE
                    + GRID_SIZE / 2
                )

                center_y = (
                    cell_y
                    * GRID_SIZE
                    + GRID_SIZE / 2
                )

                arcade.draw_texture_rect(
                    texture,
                    arcade.LRBT(
                        center_x - width / 2,
                        center_x + width / 2,
                        center_y - height / 2,
                        center_y + height / 2,
                    ),
                )

    # ========================================================
    # MOUSE MOTION
    # ========================================================

    def on_mouse_motion(
        self,
        x,
        y,
        dx,
        dy,
    ):

        self.mouse_x = x
        self.mouse_y = y

        if not (
            self.world_left
            <= x
            <= self.world_right
            and
            self.world_bottom
            <= y
            <= self.world_top
        ):

            self.hover_texture = None
            return

        self.update_world_mouse(
            x,
            y,
        )

        if self.is_shift_dragging:

            self.drag_current_x = (
                self.world_mouse_x
            )

            self.drag_current_y = (
                self.world_mouse_y
            )

            return

        asset = (
            self.get_selected_asset()
        )

        if asset is None:

            self.hover_texture = None

            return

        self.hover_texture = (
            asset.texture
        )

        self.hover_scale = (
            self.get_texture_scale(
                asset.texture
            )
        )

    # ========================================================
    # MOUSE PRESS
    # ========================================================

    def on_mouse_press(
        self,
        x,
        y,
        button,
        modifiers,
    ):

        self.shift_pressed = bool(
            modifiers
            & arcade.key.MOD_SHIFT
        )

        # ----------------------------------------------------
        # RIGHT CLICK = DELETE
        # ----------------------------------------------------

        if button == arcade.MOUSE_BUTTON_RIGHT:

            if (
                self.world_left
                <= x
                <= self.world_right
                and
                self.world_bottom
                <= y
                <= self.world_top
            ):

                self.update_world_mouse(
                    x,
                    y,
                )

                self.remove_object_at(
                    self.world_mouse_x,
                    self.world_mouse_y,
                )

            return

        if button != arcade.MOUSE_BUTTON_LEFT:
            return

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if (
            130 <= x <= 220
            and
            0 <= y <= BOTTOM_BAR_HEIGHT
        ):

            self.save_map()
            return

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        if (
            230 <= x <= 320
            and
            0 <= y <= BOTTOM_BAR_HEIGHT
        ):

            self.load_map()
            return

        # ----------------------------------------------------
        # CLEAR
        # ----------------------------------------------------

        if (
            330 <= x <= 420
            and
            0 <= y <= BOTTOM_BAR_HEIGHT
        ):

            self.clear_map()
            return

        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        if (
            self.sidebar_left
            <= x
            <= self.sidebar_right
        ):

            if (
                self.category_left
                <= x
                <= self.category_right
                and
                self.inspector_top
                <= y
                <= self.world_top
            ):

                self.handle_category_click(
                    x,
                    y,
                )

                return

            if (
                self.category_right
                <= x
                <= self.sidebar_right
                and
                self.palette_bottom
                <= y
                <= self.palette_top
            ):

                self.handle_palette_click(
                    x,
                    y,
                )

                return

            if (
                self.category_left
                <= x
                <= self.sidebar_right
                and
                self.inspector_bottom
                <= y
                <= self.inspector_top
            ):

                self.handle_inspector_click(
                    x,
                    y,
                )

                return

            return

        # ----------------------------------------------------
        # WORLD
        # ----------------------------------------------------

        if (
            self.world_left
            <= x
            <= self.world_right
            and
            self.world_bottom
            <= y
            <= self.world_top
        ):

            self.update_world_mouse(
                x,
                y,
            )

            if (
                modifiers
                & arcade.key.MOD_SHIFT
            ):

                self.is_shift_dragging = True

                self.drag_start_x = (
                    self.world_mouse_x
                )

                self.drag_start_y = (
                    self.world_mouse_y
                )

                self.drag_current_x = (
                    self.world_mouse_x
                )

                self.drag_current_y = (
                    self.world_mouse_y
                )

                return

            self.place_selected_object()

    # ========================================================
    # MOUSE RELEASE
    # ========================================================

    def on_mouse_release(
        self,
        x,
        y,
        button,
        modifiers,
    ):

        if (
            button
            != arcade.MOUSE_BUTTON_LEFT
        ):
            return

        if not self.is_shift_dragging:
            return

        # Если отпустили мышь над sidebar,
        # оставляем последнюю координату мира.
        if (
            self.world_left
            <= x
            <= self.world_right
            and
            self.world_bottom
            <= y
            <= self.world_top
        ):

            self.update_world_mouse(
                x,
                y,
            )

            self.drag_current_x = (
                self.world_mouse_x
            )

            self.drag_current_y = (
                self.world_mouse_y
            )

        self.place_drag_rectangle()

        self.is_shift_dragging = False

        self.drag_start_x = None
        self.drag_start_y = None

        self.drag_current_x = None
        self.drag_current_y = None

        self.hover_texture = None

    # ========================================================
    # PLACE DRAG RECTANGLE
    # ========================================================

    def place_drag_rectangle(self):

        asset = self.get_selected_asset()

        if asset is None:
            return

        if (
            self.drag_start_x is None
            or self.drag_start_y is None
            or self.drag_current_x is None
            or self.drag_current_y is None
        ):
            return

        start_cell_x, start_cell_y = (
            self.get_cell(
                self.drag_start_x,
                self.drag_start_y,
            )
        )

        end_cell_x, end_cell_y = (
            self.get_cell(
                self.drag_current_x,
                self.drag_current_y,
            )
        )

        min_x = min(
            start_cell_x,
            end_cell_x,
        )

        max_x = max(
            start_cell_x,
            end_cell_x,
        )

        min_y = min(
            start_cell_y,
            end_cell_y,
        )

        max_y = max(
            start_cell_y,
            end_cell_y,
        )

        for cell_y in range(
            min_y,
            max_y + 1,
        ):

            for cell_x in range(
                min_x,
                max_x + 1,
            ):

                if cell_y < 0:
                    continue

                center_x = (
                    cell_x
                    * GRID_SIZE
                    + GRID_SIZE / 2
                )

                center_y = (
                    cell_y
                    * GRID_SIZE
                    + GRID_SIZE / 2
                )

                if center_x < GRID_SIZE / 2:
                    continue

                if self.has_object_at(
                    center_x,
                    center_y,
                ):
                    continue

                self.create_object(
                    center_x,
                    center_y,
                    asset,
                )

    # ========================================================
    # CREATE OBJECT
    # ========================================================

    def create_object(
        self,
        x: float,
        y: float,
        asset: ResourceTexture,
    ):

        texture = asset.texture

        scale = (
            self.get_texture_scale(
                texture
            )
        )

        sprite = arcade.Sprite(
            texture,
            scale=scale,
        )

        sprite.center_x = x
        sprite.center_y = y

        self.object_sprites.append(
            sprite
        )

        if self.selected_resource is None:
            return

        self.objects.append(
            EditorObject(
                category=(
                    self.selected_resource.category
                ),
                name=(
                    self.selected_resource.name
                ),
                x=x,
                y=y,
                texture_path=(
                    asset.relative_path
                ),
                sprite=sprite,
            )
        )

        self.clamp_camera()

    # ========================================================
    # PLACE SINGLE
    # ========================================================

    def place_selected_object(self):

        asset = (
            self.get_selected_asset()
        )

        if asset is None:
            return

        x, y = self.snap_to_grid(
            self.world_mouse_x,
            self.world_mouse_y,
        )

        if y < GRID_SIZE / 2:
            return

        if x < GRID_SIZE / 2:
            return

        if self.has_object_at(
            x,
            y,
        ):
            return

        self.create_object(
            x,
            y,
            asset,
        )

    # ========================================================
    # HAS OBJECT
    # ========================================================

    def has_object_at(
        self,
        x,
        y,
    ) -> bool:

        for obj in self.objects:

            if (
                abs(obj.x - x)
                < 0.01
                and
                abs(obj.y - y)
                < 0.01
            ):

                return True

        return False

    # ========================================================
    # DELETE
    # ========================================================

    def remove_object_at(
        self,
        x,
        y,
    ):

        cell_x, cell_y = (
            self.get_cell(
                x,
                y,
            )
        )

        if cell_x < 0 or cell_y < 0:
            return

        center_x = (
            cell_x
            * GRID_SIZE
            + GRID_SIZE / 2
        )

        center_y = (
            cell_y
            * GRID_SIZE
            + GRID_SIZE / 2
        )

        for index in range(
            len(self.objects) - 1,
            -1,
            -1,
        ):

            obj = self.objects[index]

            if (
                abs(obj.x - center_x)
                < 0.01
                and
                abs(obj.y - center_y)
                < 0.01
            ):

                if obj.sprite:

                    self.object_sprites.remove(
                        obj.sprite
                    )

                self.objects.pop(index)

                self.status_text.text = (
                    "Объект удалён"
                )

                self.clamp_camera()

                return

    # ========================================================
    # CATEGORY CLICK
    # ========================================================

    def handle_category_click(
        self,
        x,
        y,
    ):

        start_y = (
            self.world_top - 70
        )

        for index, category in enumerate(
            self.resource_manager.categories
        ):

            item_y = (
                start_y
                - index * 42
            )

            if (
                item_y - 2
                <= y
                <= item_y + 32
            ):

                self.select_category(
                    category
                )

                return

    # ========================================================
    # PALETTE CLICK
    # ========================================================

    def handle_palette_click(
        self,
        x,
        y,
    ):

        content_top_limit = (
            self.palette_top - 42
        )

        content_top = (
            content_top_limit
            + self.palette_scroll
        )

        for index, item in enumerate(
            self.palette_items
        ):

            top = (
                content_top
                - index
                * (
                    PALETTE_CARD_HEIGHT
                    + PALETTE_GAP
                )
            )

            bottom = (
                top
                - PALETTE_CARD_HEIGHT
            )

            if (
                bottom
                <= y
                <= top
            ):

                self.select_resource(
                    item.resource
                )

                return

    # ========================================================
    # INSPECTOR CLICK
    # ========================================================

    def handle_inspector_click(
        self,
        x,
        y,
    ):

        if self.selected_resource is None:
            return

        files = (
            self.get_resource_files(
                self.selected_resource
            )
        )

        content_top = (
            self.inspector_top
            - 52
            + self.inspector_scroll
        )

        for index, asset in enumerate(
            files
        ):

            top = (
                content_top
                - index
                * (
                    INSPECTOR_ITEM_HEIGHT
                    + INSPECTOR_GAP
                )
            )

            bottom = (
                top
                - INSPECTOR_ITEM_HEIGHT
            )

            if (
                bottom
                <= y
                <= top
            ):

                self.selected_texture_path = (
                    asset.relative_path
                )

                self.hover_texture = (
                    asset.texture
                )

                self.hover_scale = (
                    self.get_texture_scale(
                        asset.texture
                    )
                )

                return

    # ========================================================
    # SCROLL
    # ========================================================

    def on_mouse_scroll(
        self,
        x: float,
        y: float,
        scroll_x: float,
        scroll_y: float,
    ):

        # =====================================================
        # WORLD
        # =====================================================

        if (
            self.world_left
            <= x
            <= self.world_right
            and
            self.world_bottom
            <= y
            <= self.world_top
        ):

            # Горизонтальный scroll touchpad.
            if abs(scroll_x) > 0:

                self.target_camera_x -= (
                    scroll_x
                    * CAMERA_SCROLL_SPEED
                )

                self.clamp_camera()

                return

            # Обычное колесо.
            #
            # В редакторе карта горизонтальная,
            # поэтому колесо двигает карту вправо/влево.
            if scroll_y != 0:

                self.target_camera_x -= (
                    scroll_y
                    * CAMERA_SCROLL_SPEED
                )

                self.clamp_camera()

                return

        # =====================================================
        # SIDEBAR
        # =====================================================

        if not (
            self.sidebar_left
            <= x
            <= self.sidebar_right
        ):
            return

        # =====================================================
        # INSPECTOR
        # =====================================================

        if (
            self.inspector_bottom
            <= y
            <= self.inspector_top
        ):

            if self.selected_resource is None:
                return

            files = (
                self.get_resource_files(
                    self.selected_resource
                )
            )

            content_height = (
                len(files)
                * (
                    INSPECTOR_ITEM_HEIGHT
                    + INSPECTOR_GAP
                )
            )

            visible_height = (
                self.inspector_height
                - 55
            )

            max_scroll = max(
                0,
                content_height
                - visible_height,
            )

            self.inspector_scroll -= (
                scroll_y * 35
            )

            self.inspector_scroll = max(
                0,
                min(
                    self.inspector_scroll,
                    max_scroll,
                ),
            )

            return

        # =====================================================
        # PALETTE
        # =====================================================

        if (
            self.palette_bottom
            <= y
            <= self.palette_top
        ):

            content_height = (
                len(self.palette_items)
                * (
                    PALETTE_CARD_HEIGHT
                    + PALETTE_GAP
                )
            )

            visible_height = (
                self.palette_top
                - self.palette_bottom
                - 42
            )

            max_scroll = max(
                0,
                content_height
                - visible_height,
            )

            self.palette_scroll -= (
                scroll_y * 35
            )

            self.palette_scroll = max(
                0,
                min(
                    self.palette_scroll,
                    max_scroll,
                ),
            )

    # ========================================================
    # KEY PRESS
    # ========================================================

    def on_key_press(
        self,
        key,
        modifiers,
    ):

        if key in (
            arcade.key.LSHIFT,
            arcade.key.RSHIFT,
        ):

            self.shift_pressed = True

        elif key == arcade.key.S:

            self.save_map()

        elif key == arcade.key.L:

            self.load_map()

        elif key == arcade.key.DELETE:

            self.clear_map()

        elif key == arcade.key.HOME:

            self.target_camera_x = (
                self.get_camera_min_x()
            )

            self.clamp_camera()

        elif key == arcade.key.END:

            self.target_camera_x = (
                self.get_camera_max_x()
            )

            self.clamp_camera()

    # ========================================================
    # KEY RELEASE
    # ========================================================

    def on_key_release(
        self,
        key,
        modifiers,
    ):

        if key in (
            arcade.key.LSHIFT,
            arcade.key.RSHIFT,
        ):

            self.shift_pressed = False

    # ========================================================
    # CLEAR
    # ========================================================

    def clear_map(self):

        self.objects.clear()

        self.object_sprites.clear()

        self.target_camera_x = (
            self.get_camera_min_x()
        )

        self.camera_x = (
            self.target_camera_x
        )

        self.clamp_camera()

        self.status_text.text = (
            "Карта очищена"
        )

    # ========================================================
    # RESOURCE PATH FOR JSON
    # ========================================================

    def get_json_texture_path(
        self,
        obj: EditorObject,
    ) -> str | None:

        """
        Возвращает путь относительно проекта:

        resources/blocks/ground/ground.png

        а не:

        /Users/.../GAME/resources/blocks/ground/ground.png

        """

        if not obj.texture_path:
            return None

        relative_texture = Path(
            obj.texture_path
        )

        path = (
            RESOURCES_DIR
            / obj.category
            / obj.name
            / relative_texture
        )

        return path.as_posix()

    # ========================================================
    # SAVE
    # ========================================================

    def save_map(self):

        data = {
            "objects": []
        }

        for obj in self.objects:

            texture_path = (
                self.get_json_texture_path(
                    obj
                )
            )

            data["objects"].append({
                "x": obj.x,
                "y": obj.y,
                "category": obj.category,
                "resource": obj.name,
                "texture_path": texture_path,
            })

        MAP_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        MAP_FILE.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=4,
            ),
            encoding="utf-8",
        )

        self.status_text.text = (
            f"Сохранено: {MAP_FILE}"
        )

        print(
            f"[Editor] Карта сохранена: {MAP_FILE}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load_map(self):

        if not MAP_FILE.exists():

            self.status_text.text = (
                "Файл карты не найден"
            )

            return

        try:

            data = json.loads(
                MAP_FILE.read_text(
                    encoding="utf-8"
                )
            )

            self.objects.clear()
            self.object_sprites.clear()

            for item in data.get(
                "objects",
                [],
            ):

                category = item.get(
                    "category"
                )

                # ------------------------------------------------
                # Новый формат:
                #
                # "resource": "ground"
                #
                # Старый формат:
                #
                # "name": "ground"
                # ------------------------------------------------

                name = item.get(
                    "resource",
                    item.get("name"),
                )

                if not category or not name:
                    continue

                x = float(
                    item.get(
                        "x",
                        0,
                    )
                )

                y = float(
                    item.get(
                        "y",
                        0,
                    )
                )

                texture_path = item.get(
                    "texture_path"
                )

                # ------------------------------------------------
                # RESOURCE
                # ------------------------------------------------

                resource = (
                    self.resource_manager.get(
                        category,
                        name,
                    )
                )

                if resource is None:

                    print(
                        "[Editor] "
                        f"Ресурс не найден: "
                        f"{category}/{name}"
                    )

                    continue

                # ------------------------------------------------
                # TEXTURE
                # ------------------------------------------------

                asset = None

                if texture_path:

                    normalized_path = (
                        Path(
                            texture_path
                        ).as_posix()
                    )

                    # ============================================
                    # НОВЫЙ ФОРМАТ
                    #
                    # resources/blocks/ground/ground.png
                    # ============================================

                    resources_prefix = (
                        RESOURCES_DIR.as_posix()
                        + "/"
                    )

                    if normalized_path.startswith(
                        resources_prefix
                    ):

                        resource_relative = (
                            normalized_path[
                                len(resources_prefix):
                            ]
                        )

                        expected_prefix = (
                            f"{category}/"
                            f"{name}/"
                        )

                        if resource_relative.startswith(
                            expected_prefix
                        ):

                            relative_texture = (
                                resource_relative[
                                    len(expected_prefix):
                                ]
                            )

                            for candidate in (
                                resource.files
                            ):

                                if (
                                    candidate.relative_path
                                    == relative_texture
                                ):

                                    asset = candidate
                                    break

                    # ============================================
                    # Обычный относительный путь:
                    #
                    # ground.png
                    # ============================================

                    if asset is None:

                        for candidate in (
                            resource.files
                        ):

                            if (
                                candidate.relative_path
                                == normalized_path
                            ):

                                asset = candidate
                                break

                    # ============================================
                    # СТАРЫЙ ФОРМАТ:
                    #
                    # /Users/.../resources/...
                    # ============================================

                    if asset is None:

                        absolute_path = str(
                            Path(
                                texture_path
                            ).resolve()
                        )

                        for candidate in (
                            resource.files
                        ):

                            if (
                                candidate.full_path
                                == absolute_path
                            ):

                                asset = candidate
                                break

                # ------------------------------------------------
                # FALLBACK
                # ------------------------------------------------

                if asset is None:

                    files = (
                        self.get_resource_files(
                            resource
                        )
                    )

                    if files:
                        asset = files[0]

                if asset is None:
                    continue

                # ------------------------------------------------
                # SPRITE
                # ------------------------------------------------

                scale = (
                    self.get_texture_scale(
                        asset.texture
                    )
                )

                sprite = arcade.Sprite(
                    asset.texture,
                    scale=scale,
                )

                sprite.center_x = x
                sprite.center_y = y

                self.object_sprites.append(
                    sprite
                )

                self.objects.append(
                    EditorObject(
                        category=category,
                        name=name,
                        x=x,
                        y=y,
                        texture_path=(
                            asset.relative_path
                        ),
                        sprite=sprite,
                    )
                )

            # ------------------------------------------------
            # ПОКАЗЫВАЕМ НАЧАЛО КАРТЫ
            # ------------------------------------------------

            self.target_camera_x = (
                self.get_camera_min_x()
            )

            self.camera_x = (
                self.target_camera_x
            )

            self.clamp_camera()

            self.status_text.text = (
                f"Загружено: {MAP_FILE}"
            )

            print(
                f"[Editor] Карта загружена: {MAP_FILE}"
            )

        except Exception as exc:

            self.status_text.text = (
                f"Ошибка загрузки: {exc}"
            )

            print(
                f"[Editor] Ошибка загрузки: {exc}"
            )

    # ========================================================
    # RESIZE
    # ========================================================

    def on_resize(
        self,
        width,
        height,
    ):

        super().on_resize(
            width,
            height,
        )

        self.recalculate_layout()
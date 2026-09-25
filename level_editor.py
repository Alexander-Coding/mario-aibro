from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
CATEGORY_WIDTH = 130

PALETTE_CARD_HEIGHT = 58
PALETTE_GAP = 8

INSPECTOR_ITEM_HEIGHT = 58
INSPECTOR_GAP = 8

PREVIEW_SIZE = 34

RESOURCES_DIR = Path("resources")

MAP_MIN_WIDTH = 1280
MAP_PADDING_CELLS = 10
MAP_FILE = Path("maps/custom_level.json")

# ------------------------------------------------------------
# Камера и скролл
# ------------------------------------------------------------
CAMERA_SMOOTH_SPEED = 20.0              # мягкое, но отзывчивое догоняние
CAMERA_WHEEL_STEP = 100.0              # импульс от одного щелчка колеса
CAMERA_WHEEL_MAX_VELOCITY = 2500.0     # максимум скорости колеса
CAMERA_WHEEL_DECAY = 0.0000            # затухание инерции (за ~0.3 сек)

SCROLL_SMOOTH_SPEED = 200.0
INSPECTOR_SCROLL_STEP = 5.0
PALETTE_SCROLL_STEP = 250.0

# ------------------------------------------------------------
# Буфер свободного места справа от viewport
# ------------------------------------------------------------
MIN_BUFFER_CELLS = 4                # минимум клеток пустоты справа
BUFFER_GROW_SPEED_CELLS = 12.0         # максимум клеток/сек при большом дефиците


# ============================================================
# THEME
# ============================================================

class Theme:
    TOP_BAR       = (28, 32, 48)
    BOTTOM_BAR    = (22, 26, 38)
    SIDEBAR_BG    = (34, 38, 56)
    CATEGORY_BG   = (30, 34, 50)
    PALETTE_BG    = (26, 30, 44)
    INSPECTOR_BG  = (30, 34, 50)

    CARD          = (48, 54, 76)
    CARD_HOVER    = (60, 68, 94)
    CARD_SELECTED = (86, 128, 255)
    CARD_SELECTED_HOVER = (110, 148, 255)

    ACCENT        = (255, 186, 66)
    ACCENT_HOVER  = (255, 208, 110)
    DANGER        = (232, 88, 88)
    DANGER_HOVER  = (248, 116, 116)
    SUCCESS       = (98, 200, 128)

    TEXT          = (238, 240, 250)
    TEXT_DIM      = (160, 168, 190)
    TEXT_MUTED    = (110, 118, 140)

    SKY           = (108, 176, 240)
    GRID          = (255, 255, 255, 28)

    SHADOW        = (0, 0, 0, 60)
    OUTLINE       = (78, 86, 112)


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


@dataclass
class Button:
    label: str
    x: float
    y: float
    width: float
    height: float
    on_click: Callable[[], None]
    color: tuple = Theme.CARD
    hover_color: tuple = Theme.CARD_HOVER

    def hit(self, mx: float, my: float) -> bool:
        return (
            self.x <= mx <= self.x + self.width
            and self.y <= my <= self.y + self.height
        )


# ============================================================
# LEVEL EDITOR
# ============================================================

class LevelEditorView(arcade.View):

    def __init__(self, return_to_menu: Callable[[], None] | None = None):
        super().__init__()

        self.return_to_menu = return_to_menu

        self.resource_manager = ResourceManager()

        self.selected_category: str | None = None
        self.selected_resource: GameResource | None = None
        self.selected_texture_path: str | None = None

        self.objects: list[EditorObject] = []
        self.object_sprites = arcade.SpriteList()

        self._cells: dict[tuple[int, int], EditorObject] = {}

        self._map_width_cache: float | None = None
        self.dynamic_map_width = 0.0

        # ----------------------------------------------------
        # CAMERAS
        # ----------------------------------------------------
        self.camera = arcade.Camera2D()
        self.ui_camera = arcade.Camera2D()

        self.camera_x = 0.0
        self.camera_y = 0.0
        self.target_camera_x = 0.0

        self._camera_wheel_velocity = 0.0

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

        self.shift_pressed = False

        # ----------------------------------------------------
        # SCROLL
        # ----------------------------------------------------
        self.palette_scroll = 0.0
        self.target_palette_scroll = 0.0
        self.inspector_scroll = 0.0
        self.target_inspector_scroll = 0.0

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
        # DIALOG
        # ----------------------------------------------------
        self.dialog_visible = False
        self.dialog_message = "Выйти в главное меню?"
        self.dialog_buttons: list[Button] = []

        self._build_buttons()

        self._category_texts: dict[str, arcade.Text] = {}
        self._palette_texts: dict[str, arcade.Text] = {}
        self._inspector_texts: dict[str, arcade.Text] = {}

        # ----------------------------------------------------
        # UI TEXTS
        # ----------------------------------------------------
        self.title_text = arcade.Text(
            "Редактор уровня", 16, 20, font_size=20,
            color=Theme.TEXT, bold=True,
        )
        self.category_title_text = arcade.Text(
            "Категории", 16, 0, font_size=14,
            color=Theme.TEXT_DIM, bold=True,
        )
        self.resources_title_text = arcade.Text(
            "Ресурсы", 0, 0, font_size=16,
            color=Theme.TEXT, bold=True,
        )
        self.inspector_title_text = arcade.Text(
            "Состав ресурса", 0, 0, font_size=16,
            color=Theme.TEXT, bold=True,
        )
        self.status_text = arcade.Text(
            "", 470, 15, font_size=13, color=Theme.TEXT_DIM,
        )
        self.dialog_title = arcade.Text(
            "", 0, 0, font_size=20, color=Theme.TEXT, bold=True,
            anchor_x="center",
        )

        self.recalculate_layout()

        if self.resource_manager.categories:
            self.select_category(self.resource_manager.categories[0])

    # ========================================================
    # SCISSOR HELPERS
    # ========================================================

    def _push_scissor(self, left, bottom, right, top):
        left, bottom = int(left), int(bottom)
        right, top = int(right), int(top)
        width = max(0, right - left)
        height = max(0, top - bottom)

        ctx = getattr(self.window, "ctx", None)
        if ctx is not None and hasattr(ctx, "scissor"):
            old = ctx.scissor
            ctx.scissor = (left, bottom, width, height)
            return ("ctx", old)

        if hasattr(arcade, "set_clip_rect"):
            old = arcade.get_clip_rect() if hasattr(arcade, "get_clip_rect") else None
            arcade.set_clip_rect(left, bottom, right, top)
            return ("arcade", old)

        return (None, None)

    def _pop_scissor(self, state):
        kind, old = state
        if kind == "ctx":
            self.window.ctx.scissor = old
        elif kind == "arcade":
            if old is None:
                arcade.reset_clip_rect()
            else:
                arcade.set_clip_rect(*old)

    # ========================================================
    # BUTTONS
    # ========================================================

    def _build_buttons(self):
        self.btn_save  = Button("Сохранить", 130, 6, 90, BOTTOM_BAR_HEIGHT - 12,
                                self.save_map, Theme.CARD, Theme.CARD_HOVER)
        self.btn_load  = Button("Загрузить", 230, 6, 90, BOTTOM_BAR_HEIGHT - 12,
                                self.load_map, Theme.CARD, Theme.CARD_HOVER)
        self.btn_clear = Button("Очистить", 330, 6, 90, BOTTOM_BAR_HEIGHT - 12,
                                self.clear_map, Theme.DANGER, Theme.DANGER_HOVER)
        self.btn_menu  = Button("← В меню", DEFAULT_WIDTH - 130, 6, 120,
                                BOTTOM_BAR_HEIGHT - 12,
                                self._request_exit, Theme.CARD, Theme.CARD_HOVER)

        self.bottom_buttons = [self.btn_save, self.btn_load, self.btn_clear]

    def _request_exit(self):
        self.dialog_visible = True

    def _confirm_exit(self, confirmed: bool):
        self.dialog_visible = False
        if confirmed and self.return_to_menu is not None:
            self.return_to_menu()

    def _build_dialog_buttons(self):
        w = self.window_width
        h = self.window_height
        bw, bh = 120, 40
        cx = w / 2
        cy = h / 2

        self.dialog_buttons = [
            Button(
                "Выйти",
                cx - bw - 10, cy - bh - 20, bw, bh,
                lambda: self._confirm_exit(True),
                Theme.DANGER, Theme.DANGER_HOVER,
            ),
            Button(
                "Отмена",
                cx + 10, cy - bh - 20, bw, bh,
                lambda: self._confirm_exit(False),
                Theme.CARD, Theme.CARD_HOVER,
            ),
        ]

    # ========================================================
    # MAP WIDTH
    # ========================================================

    def get_map_width(self) -> float:
        if self._map_width_cache is not None:
            return self._map_width_cache

        if not self.objects:
            base = max(MAP_MIN_WIDTH, self.get_world_view_width())
        else:
            max_x = max(obj.x for obj in self.objects)
            max_cell = int(max_x // GRID_SIZE)
            width_cells = max_cell + 1 + MAP_PADDING_CELLS
            base = max(
                MAP_MIN_WIDTH,
                width_cells * GRID_SIZE,
                self.get_world_view_width(),
            )

        width = max(base, self.dynamic_map_width)
        self._map_width_cache = width
        return width

    def _invalidate_map_width(self):
        self._map_width_cache = None

    def get_world_view_width(self) -> float:
        return max(1, self.sidebar_left)

    def get_camera_min_x(self) -> float:
        return self.get_world_view_width() / 2

    def get_camera_max_x(self) -> float:
        viewport_width = self.get_world_view_width()
        map_width = self.get_map_width()
        return max(viewport_width / 2, map_width - viewport_width / 2)

    # ========================================================
    # LAYOUT
    # ========================================================

    def recalculate_layout(self):
        old_camera_x = self.camera_x

        self.window_width = self.window.width
        self.window_height = self.window.height

        w, h = self.window_width, self.window_height

        self.sidebar_left = w - SIDEBAR_WIDTH
        self.sidebar_right = w
        self.category_left = self.sidebar_left
        self.category_right = self.sidebar_left + CATEGORY_WIDTH

        self.world_left = 0
        self.world_right = self.sidebar_left
        self.world_bottom = BOTTOM_BAR_HEIGHT
        self.world_top = h - TOP_BAR_HEIGHT

        self.inspector_height = max(
            260,
            min(330, int(h * 0.46)),
        )
        max_inspector_height = max(
            180,
            self.world_top - self.world_bottom - 100,
        )
        self.inspector_height = min(self.inspector_height, max_inspector_height)
        self.inspector_bottom = self.world_bottom
        self.inspector_top = self.inspector_bottom + self.inspector_height

        self.palette_bottom = self.inspector_top
        self.palette_top = self.world_top

        self.title_text.x = 16
        self.title_text.y = h - 40

        self.category_title_text.x = self.category_left + 12
        self.category_title_text.y = self.world_top - 30

        self.resources_title_text.x = self.category_right + 12
        self.resources_title_text.y = self.palette_top - 30

        self.inspector_title_text.x = self.category_left + 12
        self.inspector_title_text.y = self.inspector_top - 30

        self.btn_menu.x = w - 130
        self.btn_menu.y = 6

        self.ui_camera.position = (w / 2, h / 2)

        self.camera_y = (self.world_bottom + self.world_top) / 2

        if self.camera_x == 0:
            self.camera_x = self.get_camera_min_x()
        else:
            self.camera_x = old_camera_x

        if self.target_camera_x == 0:
            self.target_camera_x = self.camera_x

        self._build_dialog_buttons()

    # ========================================================
    # UPDATE
    # ========================================================

    def on_update(self, delta_time: float):
        # --- Поддержка буфера свободного места справа от viewport ---
        viewport_width = self.get_world_view_width()
        viewport_right_world = self.camera_x + viewport_width / 2
        map_width = self.get_map_width()
        buffer_right = map_width - viewport_right_world

        min_buffer = MIN_BUFFER_CELLS * GRID_SIZE

        if buffer_right < min_buffer:
            deficit = min_buffer - buffer_right
            deficit_cells = deficit / GRID_SIZE

            # Скорость тем выше, чем больше дефицит (но не выше лимита)
            speed_cells = min(BUFFER_GROW_SPEED_CELLS, deficit_cells * 4.0)
            growth = speed_cells * GRID_SIZE * delta_time

            if growth > 0:
                self.dynamic_map_width += growth
                self._invalidate_map_width()

        # --- Плавный скролл палитры/инспектора ---
        factor_scroll = 1.0 - pow(0.001, delta_time * SCROLL_SMOOTH_SPEED)

        self.palette_scroll += (
            self.target_palette_scroll - self.palette_scroll
        ) * factor_scroll
        self.inspector_scroll += (
            self.target_inspector_scroll - self.inspector_scroll
        ) * factor_scroll

        if abs(self.target_palette_scroll - self.palette_scroll) < 0.1:
            self.palette_scroll = self.target_palette_scroll
        if abs(self.target_inspector_scroll - self.inspector_scroll) < 0.1:
            self.inspector_scroll = self.target_inspector_scroll

        # --- Инерция колеса ---
        if self._camera_wheel_velocity != 0.0:
            self.target_camera_x += self._camera_wheel_velocity * delta_time
            self._camera_wheel_velocity *= pow(CAMERA_WHEEL_DECAY, delta_time)
            if abs(self._camera_wheel_velocity) < 1.0:
                self._camera_wheel_velocity = 0.0

        # --- Клампим target ДО интерполяции ---
        min_x = self.get_camera_min_x()
        max_x = self.get_camera_max_x()
        self.target_camera_x = max(min_x, min(self.target_camera_x, max_x))

        # --- Плавное движение камеры ---
        difference = self.target_camera_x - self.camera_x
        factor = 1.0 - pow(0.001, delta_time * CAMERA_SMOOTH_SPEED)
        self.camera_x += difference * factor

        # --- Финальный clamp ---
        self.camera_x = max(min_x, min(self.camera_x, max_x))
        self.camera.position = (self.camera_x, self.camera_y)

    def _is_mouse_in_world(self) -> bool:
        return (
            self.world_left <= self.mouse_x <= self.world_right
            and self.world_bottom <= self.mouse_y <= self.world_top
        )

    # ========================================================
    # CATEGORY / RESOURCE
    # ========================================================

    def select_category(self, category: str):
        self.selected_category = category

        resources = self.resource_manager.get_category_resources(category)

        if category.lower() == "entities":
            resources = [
                r for r in resources
                if any(
                    Path(a.relative_path).parts
                    and Path(a.relative_path).parts[0].lower() == "idle"
                    for a in r.files
                )
            ]

        self.palette_items.clear()
        self._palette_texts.clear()

        for resource in resources:
            texture = resource.preview_texture
            if texture is None:
                continue
            self.palette_items.append(PaletteItem(resource=resource, texture=texture))

        self.palette_scroll = 0.0
        self.target_palette_scroll = 0.0

        if self.palette_items:
            self.select_resource(self.palette_items[0].resource)
        else:
            self.selected_resource = None
            self.selected_texture_path = None
            self.inspector_scroll = 0.0
            self.target_inspector_scroll = 0.0

    def select_resource(self, resource: GameResource):
        self.selected_resource = resource
        self.selected_texture_path = None

        files = self.get_resource_files(resource)
        if files:
            self.selected_texture_path = files[0].relative_path

        self.inspector_scroll = 0.0
        self.target_inspector_scroll = 0.0
        self.status_text.text = f"Выбран: {resource.name}"

    def get_resource_files(self, resource: GameResource) -> list[ResourceTexture]:
        files = resource.files

        if resource.category.lower() == "entities":
            files = [
                a for a in files
                if Path(a.relative_path).parts
                and Path(a.relative_path).parts[0].lower() == "idle"
            ]

        return sorted(files, key=lambda a: a.relative_path.lower())

    def get_selected_asset(self) -> ResourceTexture | None:
        if self.selected_resource is None or self.selected_texture_path is None:
            return None

        for asset in self.get_resource_files(self.selected_resource):
            if asset.relative_path == self.selected_texture_path:
                return asset

        return None

    # ========================================================
    # UTIL
    # ========================================================

    def get_texture_scale(self, texture: arcade.Texture) -> float:
        width, height = texture.width, texture.height

        if width < GRID_SIZE or height < GRID_SIZE:
            scale_x = GRID_SIZE / width if width < GRID_SIZE else 1.0
            scale_y = GRID_SIZE / height if height < GRID_SIZE else 1.0
            return max(scale_x, scale_y)

        return 1.0

    def snap_to_grid(self, x: float, y: float) -> tuple[float, float]:
        cell_x = int(x // GRID_SIZE) * GRID_SIZE
        cell_y = int(y // GRID_SIZE) * GRID_SIZE
        return cell_x + GRID_SIZE / 2, cell_y + GRID_SIZE / 2

    def get_cell(self, x: float, y: float) -> tuple[int, int]:
        return int(x // GRID_SIZE), int(y // GRID_SIZE)

    def update_world_mouse(self, x: float, y: float):
        self.mouse_x = x
        self.mouse_y = y

        self.world_mouse_x = self.camera_x + (x - self.window_width / 2)
        self.world_mouse_y = self.camera_y + (y - self.window_height / 2)

    # ========================================================
    # DRAW
    # ========================================================

    def on_draw(self):
        self.clear()

        self.camera.use()
        self.draw_world()
        self.draw_grid()
        self.object_sprites.draw()
        self.draw_drag_preview()
        self.draw_hover_object()

        self.ui_camera.use()
        self.draw_ui()
        self.draw_dialog()

    # ========================================================
    # WORLD
    # ========================================================

    def draw_world(self):
        viewport_width = self.window_width
        viewport_height = self.window_height

        left = self.camera_x - viewport_width / 2 - GRID_SIZE * 4
        right = self.camera_x + viewport_width / 2 + GRID_SIZE * 4
        bottom = self.camera_y - viewport_height / 2 - GRID_SIZE * 4
        top = self.camera_y + viewport_height / 2 + GRID_SIZE * 4

        arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, Theme.SKY)

    def draw_grid(self):
        viewport_width = self.window_width
        viewport_height = self.window_height

        left = self.camera_x - viewport_width / 2 - GRID_SIZE
        right = self.camera_x + viewport_width / 2 + GRID_SIZE
        bottom = self.camera_y - viewport_height / 2 - GRID_SIZE
        top = self.camera_y + viewport_height / 2 + GRID_SIZE

        bottom = max(0, bottom)

        map_width = self.get_map_width()

        start_x = max(0, int(left // GRID_SIZE) * GRID_SIZE)
        end_x = min(map_width, (int(right // GRID_SIZE) + 1) * GRID_SIZE)

        for x in range(int(start_x), int(end_x) + 1, GRID_SIZE):
            arcade.draw_line(x, bottom, x, top, Theme.GRID, 1)

        start_y = int(bottom // GRID_SIZE) * GRID_SIZE
        end_y = (int(top // GRID_SIZE) + 1) * GRID_SIZE

        for y in range(start_y, end_y + 1, GRID_SIZE):
            arcade.draw_line(0, y, map_width, y, Theme.GRID, 1)

    # ========================================================
    # UI
    # ========================================================

    def draw_ui(self):
        w, h = self.window_width, self.window_height

        arcade.draw_lrbt_rectangle_filled(0, w, h - TOP_BAR_HEIGHT, h, Theme.TOP_BAR)
        self.title_text.draw()

        arcade.draw_lrbt_rectangle_filled(0, w, 0, BOTTOM_BAR_HEIGHT, Theme.BOTTOM_BAR)

        arcade.draw_lrbt_rectangle_filled(
            self.sidebar_left, self.sidebar_right, 0, h, Theme.SIDEBAR_BG,
        )

        arcade.draw_lrbt_rectangle_filled(
            self.category_left, self.category_right,
            self.world_bottom, self.world_top, Theme.CATEGORY_BG,
        )
        self.category_title_text.draw()
        self.draw_categories()

        self.resources_title_text.draw()
        self.draw_palette()

        arcade.draw_lrbt_rectangle_filled(
            self.category_left, self.sidebar_right,
            self.inspector_bottom, self.inspector_top, Theme.INSPECTOR_BG,
        )
        self.inspector_title_text.draw()
        self.draw_inspector()

        for btn in self.bottom_buttons + [self.btn_menu]:
            self._draw_button(btn)

        self.status_text.draw()

    def _draw_button(self, btn: Button):
        hover = btn.hit(self.mouse_x, self.mouse_y)
        color = btn.hover_color if hover else btn.color

        arcade.draw_lrbt_rectangle_filled(
            btn.x, btn.x + btn.width, btn.y, btn.y + btn.height, color,
        )
        arcade.draw_lrbt_rectangle_outline(
            btn.x, btn.x + btn.width, btn.y, btn.y + btn.height,
            Theme.OUTLINE, 1,
        )

        text = arcade.Text(
            btn.label,
            btn.x + btn.width / 2,
            btn.y + btn.height / 2,
            font_size=13,
            color=Theme.TEXT,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )
        text.draw()

    # ========================================================
    # CATEGORIES
    # ========================================================

    def draw_categories(self):
        start_y = self.world_top - 70
        mouse_y = self.mouse_y

        for index, category in enumerate(self.resource_manager.categories):
            y = start_y - index * 42

            if y + 32 < self.inspector_top:
                break

            selected = category == self.selected_category

            is_hover = (
                self.category_left <= self.mouse_x <= self.category_right
                and y - 2 <= mouse_y <= y + 32
            )

            if selected:
                bg = Theme.CARD_SELECTED_HOVER if is_hover else Theme.CARD_SELECTED
            elif is_hover:
                bg = Theme.CARD_HOVER
            else:
                bg = None

            if bg is not None:
                arcade.draw_lrbt_rectangle_filled(
                    self.category_left + 6, self.category_right - 6,
                    y - 2, y + 32, bg,
                )

            text = self._category_texts.get(category)
            if text is None:
                text = arcade.Text(
                    category,
                    self.category_left + 12, y + 7,
                    font_size=12,
                    color=Theme.TEXT,
                )
                self._category_texts[category] = text
            else:
                text.y = y + 7

            text.draw()

    # ========================================================
    # PALETTE
    # ========================================================

    def draw_palette(self):
        left = self.category_right + 8
        right = self.sidebar_right - 8
        content_bottom = self.palette_bottom
        content_top_limit = self.palette_top - 42

        arcade.draw_lrbt_rectangle_filled(
            left, right, content_bottom, content_top_limit, Theme.PALETTE_BG,
        )

        scissor_state = self._push_scissor(
            left, content_bottom, right, content_top_limit,
        )

        try:
            content_top = content_top_limit + self.palette_scroll
            step = PALETTE_CARD_HEIGHT + PALETTE_GAP
            mouse_x, mouse_y = self.mouse_x, self.mouse_y

            for index, item in enumerate(self.palette_items):
                top = content_top - index * step
                bottom = top - PALETTE_CARD_HEIGHT

                if bottom > content_top_limit or top < content_bottom:
                    continue

                selected = self.selected_resource is item.resource
                hover = left <= mouse_x <= right and bottom <= mouse_y <= top

                if selected:
                    bg = Theme.CARD_SELECTED_HOVER if hover else Theme.CARD_SELECTED
                elif hover:
                    bg = Theme.CARD_HOVER
                else:
                    bg = Theme.CARD

                arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, bg)

                texture = item.texture
                icon_scale = min(
                    PREVIEW_SIZE / texture.width,
                    PREVIEW_SIZE / texture.height,
                )
                icon_w = texture.width * icon_scale
                icon_h = texture.height * icon_scale
                icon_x = left + 12 + icon_w / 2
                icon_y = bottom + PALETTE_CARD_HEIGHT / 2

                arcade.draw_texture_rect(
                    texture,
                    arcade.LRBT(
                        icon_x - icon_w / 2, icon_x + icon_w / 2,
                        icon_y - icon_h / 2, icon_y + icon_h / 2,
                    ),
                )

                text = self._palette_texts.get(item.resource.name)
                if text is None:
                    text = arcade.Text(
                        item.resource.name,
                        left + 58, icon_y - 7,
                        font_size=13, color=Theme.TEXT,
                    )
                    self._palette_texts[item.resource.name] = text
                else:
                    text.y = icon_y - 7

                text.draw()
        finally:
            self._pop_scissor(scissor_state)

    # ========================================================
    # INSPECTOR
    # ========================================================

    def draw_inspector(self):
        left = self.category_left + 8
        right = self.sidebar_right - 8

        content_top_limit = self.inspector_top - 45
        content_bottom_limit = self.inspector_bottom + 6

        if self.selected_resource is None:
            self._draw_muted(left + 8, self.inspector_top - 65, "Ресурс не выбран")
            return

        files = self.get_resource_files(self.selected_resource)
        if not files:
            self._draw_muted(left + 8, self.inspector_top - 65, "Нет доступных элементов")
            return

        scissor_state = self._push_scissor(
            left, content_bottom_limit, right, content_top_limit,
        )

        try:
            content_top = content_top_limit + self.inspector_scroll
            step = INSPECTOR_ITEM_HEIGHT + INSPECTOR_GAP
            mouse_x, mouse_y = self.mouse_x, self.mouse_y

            for index, asset in enumerate(files):
                top = content_top - index * step
                bottom = top - INSPECTOR_ITEM_HEIGHT

                if bottom > content_top_limit or top < content_bottom_limit:
                    continue

                selected = asset.relative_path == self.selected_texture_path
                hover = left <= mouse_x <= right and bottom <= mouse_y <= top

                if selected:
                    bg = Theme.CARD_SELECTED_HOVER if hover else Theme.CARD_SELECTED
                elif hover:
                    bg = Theme.CARD_HOVER
                else:
                    bg = Theme.CARD

                arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, bg)

                texture = asset.texture
                icon_scale = min(
                    PREVIEW_SIZE / texture.width,
                    PREVIEW_SIZE / texture.height,
                )
                icon_w = texture.width * icon_scale
                icon_h = texture.height * icon_scale
                icon_x = left + 12 + icon_w / 2
                icon_y = bottom + INSPECTOR_ITEM_HEIGHT / 2

                arcade.draw_texture_rect(
                    texture,
                    arcade.LRBT(
                        icon_x - icon_w / 2, icon_x + icon_w / 2,
                        icon_y - icon_h / 2, icon_y + icon_h / 2,
                    ),
                )

                text = self._inspector_texts.get(asset.relative_path)
                if text is None:
                    text = arcade.Text(
                        asset.relative_path,
                        left + 58, icon_y - 7,
                        font_size=11, color=Theme.TEXT,
                    )
                    self._inspector_texts[asset.relative_path] = text
                else:
                    text.y = icon_y - 7

                text.draw()
        finally:
            self._pop_scissor(scissor_state)

    def _draw_muted(self, x: float, y: float, msg: str):
        arcade.Text(msg, x, y, font_size=13, color=Theme.TEXT_MUTED).draw()

    # ========================================================
    # HOVER / DRAG PREVIEW
    # ========================================================

    def draw_hover_object(self):
        if self.hover_texture is None or self.is_shift_dragging:
            return

        if not self._is_mouse_in_world():
            return

        x, y = self.snap_to_grid(self.world_mouse_x, self.world_mouse_y)
        texture = self.hover_texture
        scale = self.hover_scale
        width = texture.width * scale
        height = texture.height * scale

        arcade.draw_texture_rect(
            texture,
            arcade.LRBT(x - width / 2, x + width / 2, y - height / 2, y + height / 2),
            alpha=180,
        )

    def draw_drag_preview(self):
        if not self.is_shift_dragging:
            return

        asset = self.get_selected_asset()
        if asset is None:
            return

        if any(v is None for v in (
            self.drag_start_x, self.drag_start_y,
            self.drag_current_x, self.drag_current_y,
        )):
            return

        start_cell_x, start_cell_y = self.get_cell(self.drag_start_x, self.drag_start_y)
        current_cell_x, current_cell_y = self.get_cell(self.drag_current_x, self.drag_current_y)

        min_x, max_x = min(start_cell_x, current_cell_x), max(start_cell_x, current_cell_x)
        min_y, max_y = min(start_cell_y, current_cell_y), max(start_cell_y, current_cell_y)

        texture = asset.texture
        scale = self.get_texture_scale(texture)
        width = texture.width * scale
        height = texture.height * scale

        for cell_y in range(min_y, max_y + 1):
            for cell_x in range(min_x, max_x + 1):
                center_x = cell_x * GRID_SIZE + GRID_SIZE / 2
                center_y = cell_y * GRID_SIZE + GRID_SIZE / 2

                arcade.draw_texture_rect(
                    texture,
                    arcade.LRBT(
                        center_x - width / 2, center_x + width / 2,
                        center_y - height / 2, center_y + height / 2,
                    ),
                    alpha=140,
                )

    # ========================================================
    # MOUSE
    # ========================================================

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x = x
        self.mouse_y = y

        if not self._is_mouse_in_world():
            self.hover_texture = None
            return

        self.update_world_mouse(x, y)

        if self.is_shift_dragging:
            self.drag_current_x = self.world_mouse_x
            self.drag_current_y = self.world_mouse_y
            return

        asset = self.get_selected_asset()
        if asset is None:
            self.hover_texture = None
            return

        self.hover_texture = asset.texture
        self.hover_scale = self.get_texture_scale(asset.texture)

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse_x, self.mouse_y = x, y

        if self.dialog_visible:
            if button == arcade.MOUSE_BUTTON_LEFT:
                for btn in self.dialog_buttons:
                    if btn.hit(x, y):
                        btn.on_click()
                        return
            elif button == arcade.MOUSE_BUTTON_RIGHT:
                self._confirm_exit(False)
            return

        self.shift_pressed = bool(modifiers & arcade.key.MOD_SHIFT)

        if button == arcade.MOUSE_BUTTON_RIGHT:
            if self._is_mouse_in_world():
                self.update_world_mouse(x, y)
                self.remove_object_at(self.world_mouse_x, self.world_mouse_y)
            return

        if button != arcade.MOUSE_BUTTON_LEFT:
            return

        for btn in self.bottom_buttons + [self.btn_menu]:
            if btn.hit(x, y):
                btn.on_click()
                return

        if self.sidebar_left <= x <= self.sidebar_right:
            if (
                self.category_left <= x <= self.category_right
                and self.inspector_top <= y <= self.world_top
            ):
                self.handle_category_click(x, y)
                return

            if (
                self.category_right <= x <= self.sidebar_right
                and self.palette_bottom <= y <= self.palette_top
            ):
                self.handle_palette_click(x, y)
                return

            if (
                self.category_left <= x <= self.sidebar_right
                and self.inspector_bottom <= y <= self.inspector_top
            ):
                self.handle_inspector_click(x, y)
                return

            return

        if self._is_mouse_in_world():
            self.update_world_mouse(x, y)

            if modifiers & arcade.key.MOD_SHIFT:
                self.is_shift_dragging = True
                self.drag_start_x = self.world_mouse_x
                self.drag_start_y = self.world_mouse_y
                self.drag_current_x = self.world_mouse_x
                self.drag_current_y = self.world_mouse_y
                return

            self.place_selected_object()

    def on_mouse_release(self, x, y, button, modifiers):
        if button != arcade.MOUSE_BUTTON_LEFT or not self.is_shift_dragging:
            return

        if self._is_mouse_in_world():
            self.update_world_mouse(x, y)
            self.drag_current_x = self.world_mouse_x
            self.drag_current_y = self.world_mouse_y

        self.place_drag_rectangle()

        self.is_shift_dragging = False
        self.drag_start_x = self.drag_start_y = None
        self.drag_current_x = self.drag_current_y = None
        self.hover_texture = None

    # ========================================================
    # PLACE / REMOVE
    # ========================================================

    def place_drag_rectangle(self):
        asset = self.get_selected_asset()
        if asset is None:
            return

        if any(v is None for v in (
            self.drag_start_x, self.drag_start_y,
            self.drag_current_x, self.drag_current_y,
        )):
            return

        start_cell_x, start_cell_y = self.get_cell(self.drag_start_x, self.drag_start_y)
        end_cell_x, end_cell_y = self.get_cell(self.drag_current_x, self.drag_current_y)

        min_x, max_x = min(start_cell_x, end_cell_x), max(start_cell_x, end_cell_x)
        min_y, max_y = min(start_cell_y, end_cell_y), max(start_cell_y, end_cell_y)

        for cell_y in range(min_y, max_y + 1):
            if cell_y < 0:
                continue

            for cell_x in range(min_x, max_x + 1):
                if cell_x < 0:
                    continue

                center_x = cell_x * GRID_SIZE + GRID_SIZE / 2
                center_y = cell_y * GRID_SIZE + GRID_SIZE / 2

                if self.has_object_at_cell(cell_x, cell_y):
                    continue

                self.create_object(center_x, center_y, asset)

    def create_object(self, x: float, y: float, asset: ResourceTexture):
        texture = asset.texture
        scale = self.get_texture_scale(texture)

        sprite = arcade.Sprite(texture, scale=scale)
        sprite.center_x = x
        sprite.center_y = y
        self.object_sprites.append(sprite)

        if self.selected_resource is None:
            return

        obj = EditorObject(
            category=self.selected_resource.category,
            name=self.selected_resource.name,
            x=x, y=y,
            texture_path=asset.relative_path,
            sprite=sprite,
        )
        self.objects.append(obj)

        cell_x, cell_y = self.get_cell(x, y)
        self._cells[(cell_x, cell_y)] = obj

        self._invalidate_map_width()

    def place_selected_object(self):
        asset = self.get_selected_asset()
        if asset is None:
            self.status_text.text = "Не выбран ресурс"
            return

        x, y = self.snap_to_grid(self.world_mouse_x, self.world_mouse_y)

        if x < GRID_SIZE / 2 or y < GRID_SIZE / 2:
            return

        cell_x, cell_y = self.get_cell(x, y)

        if self.has_object_at_cell(cell_x, cell_y):
            self.remove_object_at(x, y)

        self.create_object(x, y, asset)

    def has_object_at_cell(self, cell_x: int, cell_y: int) -> bool:
        return (cell_x, cell_y) in self._cells

    def remove_object_at(self, x: float, y: float):
        cell_x, cell_y = self.get_cell(x, y)

        if cell_x < 0 or cell_y < 0:
            return

        obj = self._cells.pop((cell_x, cell_y), None)
        if obj is None:
            return

        if obj.sprite:
            self.object_sprites.remove(obj.sprite)

        try:
            self.objects.remove(obj)
        except ValueError:
            pass

        self.status_text.text = "Объект удалён"
        self._invalidate_map_width()

    # ========================================================
    # SIDEBAR CLICKS
    # ========================================================

    def handle_category_click(self, x, y):
        start_y = self.world_top - 70

        for index, category in enumerate(self.resource_manager.categories):
            item_y = start_y - index * 42

            if item_y - 2 <= y <= item_y + 32:
                self.select_category(category)
                return

    def handle_palette_click(self, x, y):
        content_top_limit = self.palette_top - 42
        content_top = content_top_limit + self.palette_scroll
        step = PALETTE_CARD_HEIGHT + PALETTE_GAP

        for index, item in enumerate(self.palette_items):
            top = content_top - index * step
            bottom = top - PALETTE_CARD_HEIGHT

            if bottom <= y <= top:
                self.select_resource(item.resource)
                return

    def handle_inspector_click(self, x, y):
        if self.selected_resource is None:
            return

        files = self.get_resource_files(self.selected_resource)
        content_top = self.inspector_top - 52 + self.inspector_scroll
        step = INSPECTOR_ITEM_HEIGHT + INSPECTOR_GAP

        for index, asset in enumerate(files):
            top = content_top - index * step
            bottom = top - INSPECTOR_ITEM_HEIGHT

            if bottom <= y <= top:
                self.selected_texture_path = asset.relative_path
                self.hover_texture = asset.texture
                self.hover_scale = self.get_texture_scale(asset.texture)
                return

    # ========================================================
    # SCROLL
    # ========================================================

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        # ---- WORLD ----
        if self._is_mouse_in_world():
            impulse = -(scroll_x + scroll_y) * CAMERA_WHEEL_STEP

            self._camera_wheel_velocity = max(
                -CAMERA_WHEEL_MAX_VELOCITY,
                min(self._camera_wheel_velocity + impulse, CAMERA_WHEEL_MAX_VELOCITY),
            )
            return

        if not (self.sidebar_left <= x <= self.sidebar_right):
            return

        # ---- INSPECTOR ----
        if self.inspector_bottom <= y <= self.inspector_top:
            if self.selected_resource is None:
                return
            files = self.get_resource_files(self.selected_resource)
            content_height = len(files) * (INSPECTOR_ITEM_HEIGHT + INSPECTOR_GAP)
            visible_height = self.inspector_height - 55
            max_scroll = max(0, content_height - visible_height)

            self.target_inspector_scroll = max(
                0,
                min(
                    self.target_inspector_scroll + scroll_y * INSPECTOR_SCROLL_STEP,
                    max_scroll,
                ),
            )
            return

        # ---- PALETTE ----
        if self.palette_bottom <= y <= self.palette_top:
            content_height = len(self.palette_items) * (PALETTE_CARD_HEIGHT + PALETTE_GAP)
            visible_height = self.palette_top - self.palette_bottom - 42
            max_scroll = max(0, content_height - visible_height)

            self.target_palette_scroll = max(
                0,
                min(
                    self.target_palette_scroll + scroll_y * PALETTE_SCROLL_STEP,
                    max_scroll,
                ),
            )

    # ========================================================
    # KEYBOARD
    # ========================================================

    def on_key_press(self, key, modifiers):
        if self.dialog_visible:
            if key == arcade.key.ESCAPE:
                self._confirm_exit(False)
            elif key == arcade.key.ENTER:
                self._confirm_exit(True)
            return

        if key in (arcade.key.LSHIFT, arcade.key.RSHIFT):
            self.shift_pressed = True
        elif key == arcade.key.S:
            self.save_map()
        elif key == arcade.key.L:
            self.load_map()
        elif key == arcade.key.DELETE:
            self.clear_map()
        elif key == arcade.key.HOME:
            self.target_camera_x = self.get_camera_min_x()
        elif key == arcade.key.END:
            self.target_camera_x = self.get_camera_max_x()
        elif key == arcade.key.ESCAPE:
            self._request_exit()

    def on_key_release(self, key, modifiers):
        if key in (arcade.key.LSHIFT, arcade.key.RSHIFT):
            self.shift_pressed = False

    # ========================================================
    # CLEAR
    # ========================================================

    def clear_map(self):
        self.objects.clear()
        self.object_sprites.clear()
        self._cells.clear()
        self.dynamic_map_width = 0.0
        self._invalidate_map_width()

        # Очистка — это новая карта: name, time_limit и warps с координатами
        # труб прежнего уровня в неё не переносятся (ни из загруженного, ни с диска)
        self._loaded_template = {}

        self.target_camera_x = self.get_camera_min_x()
        self._camera_wheel_velocity = 0.0

        self.status_text.text = "Карта очищена"

    # ========================================================
    # SAVE / LOAD
    # ========================================================

    def get_json_texture_path(self, obj: EditorObject) -> str | None:
        if not obj.texture_path:
            return None
        return (RESOURCES_DIR / obj.category / obj.name / Path(obj.texture_path)).as_posix()

    def _level_template(self) -> dict:
        """Ключи уровня в порядке файла; objects — только как место в порядке.

        Редактор не показывает name, time_limit, theme, warps и прочее, но и
        терять их при сохранении не должен: берём их из загруженного файла,
        а если карту ещё не загружали — из того, что сейчас лежит на диске.
        После «Очистить» шаблон пустой: новая карта сохраняется без чужих ключей.
        """

        template = getattr(self, "_loaded_template", None)
        if template is not None:
            return template

        try:
            data = json.loads(MAP_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {key: (None if key == "objects" else value) for key, value in data.items()}

    def save_map(self):
        objects = []

        for obj in self.objects:
            objects.append({
                "x": obj.x,
                "y": obj.y,
                "category": obj.category,
                "resource": obj.name,
                "texture_path": self.get_json_texture_path(obj),
            })

        # Лишние ключи файла — на своих местах, objects — на своём (или в конце)
        data = {
            key: (objects if key == "objects" else value)
            for key, value in self._level_template().items()
        }
        data["objects"] = objects

        MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
        MAP_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

        self.status_text.text = f"Сохранено: {MAP_FILE}"
        print(f"[Editor] Карта сохранена: {MAP_FILE}")

    def load_map(self):
        if not MAP_FILE.exists():
            self.status_text.text = "Файл карты не найден"
            return

        try:
            data = json.loads(MAP_FILE.read_text(encoding="utf-8"))

            # Лишние ключи (name, time_limit, warps…) вернутся в файл при сохранении
            self._loaded_template = {
                key: (None if key == "objects" else value) for key, value in data.items()
            }

            self.objects.clear()
            self.object_sprites.clear()
            self._cells.clear()
            self.dynamic_map_width = 0.0
            self._camera_wheel_velocity = 0.0

            for item in data.get("objects", []):
                category = item.get("category")
                name = item.get("resource", item.get("name"))

                if not category or not name:
                    continue

                x = float(item.get("x", 0))
                y = float(item.get("y", 0))
                texture_path = item.get("texture_path")

                resource = self.resource_manager.get(category, name)
                if resource is None:
                    print(f"[Editor] Ресурс не найден: {category}/{name}")
                    continue

                asset = self._resolve_loaded_asset(resource, category, name, texture_path)

                if asset is None:
                    files = self.get_resource_files(resource)
                    if files:
                        asset = files[0]

                if asset is None:
                    continue

                scale = self.get_texture_scale(asset.texture)
                sprite = arcade.Sprite(asset.texture, scale=scale)
                sprite.center_x = x
                sprite.center_y = y
                self.object_sprites.append(sprite)

                obj = EditorObject(
                    category=category,
                    name=name,
                    x=x, y=y,
                    texture_path=asset.relative_path,
                    sprite=sprite,
                )
                self.objects.append(obj)

                cell_x, cell_y = self.get_cell(x, y)
                self._cells[(cell_x, cell_y)] = obj

            self._invalidate_map_width()

            self.target_camera_x = self.get_camera_min_x()
            # camera_x НЕ трогаем — плавно доедет

            self.status_text.text = f"Загружено: {MAP_FILE}"
            print(f"[Editor] Карта загружена: {MAP_FILE}")

        except Exception as exc:
            self.status_text.text = f"Ошибка загрузки: {exc}"
            print(f"[Editor] Ошибка загрузки: {exc}")

    def _resolve_loaded_asset(
        self,
        resource: GameResource,
        category: str,
        name: str,
        texture_path: str | None,
    ) -> ResourceTexture | None:
        if not texture_path:
            return None

        normalized = Path(texture_path).as_posix()
        prefix = RESOURCES_DIR.as_posix() + "/"

        if normalized.startswith(prefix):
            resource_relative = normalized[len(prefix):]
            expected = f"{category}/{name}/"
            if resource_relative.startswith(expected):
                relative_texture = resource_relative[len(expected):]
                for candidate in resource.files:
                    if candidate.relative_path == relative_texture:
                        return candidate

        for candidate in resource.files:
            if candidate.relative_path == normalized:
                return candidate

        absolute = str(Path(texture_path).resolve())
        for candidate in resource.files:
            if candidate.full_path == absolute:
                return candidate

        return None

    # ========================================================
    # DIALOG
    # ========================================================

    def draw_dialog(self):
        if not self.dialog_visible:
            return

        w, h = self.window_width, self.window_height

        arcade.draw_lrbt_rectangle_filled(0, w, 0, h, (0, 0, 0, 140))

        bw, bh = 420, 200
        cx, cy = w / 2, h / 2
        left, right = cx - bw / 2, cx + bw / 2
        bottom, top = cy - bh / 2, cy + bh / 2

        arcade.draw_lrbt_rectangle_filled(
            left + 4, right + 4, bottom - 6, top - 6, Theme.SHADOW,
        )
        arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, Theme.SIDEBAR_BG)
        arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, Theme.OUTLINE, 2)

        self.dialog_title.text = self.dialog_message
        self.dialog_title.x = cx
        self.dialog_title.y = cy + 20
        self.dialog_title.draw()

        for btn in self.dialog_buttons:
            self._draw_button(btn)

    # ========================================================
    # RESIZE
    # ========================================================

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.recalculate_layout()
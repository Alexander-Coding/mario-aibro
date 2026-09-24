import arcade

from game import GameView
from level_editor import LevelEditorView


# ============================================================
# ЦВЕТА
# ============================================================

SKY_TOP = (98, 168, 232)
SKY_BOTTOM = (164, 208, 240)

GROUND = (196, 116, 60)
GROUND_DARK = (140, 78, 38)
GRASS = (100, 180, 70)
GRASS_DARK = (60, 130, 40)

TITLE_RED = (216, 46, 46)
TITLE_RED_DARK = (150, 20, 20)
TITLE_WHITE = (250, 250, 250)

BUTTON_TOP = (250, 200, 80)
BUTTON_BOTTOM = (232, 148, 40)
BUTTON_HOVER_TOP = (255, 224, 130)
BUTTON_HOVER_BOTTOM = (255, 176, 60)
BUTTON_BORDER = (120, 60, 10)
BUTTON_SHADOW = (0, 0, 0, 70)

TEXT_DARK = (60, 30, 10)
TEXT_LIGHT = (250, 250, 250)


class Button:
    """Кнопка с тенью и hover-эффектом."""

    def __init__(self, label, action, x, y, width, height):
        self.label = label
        self.action = action
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.hover = False

    def hit(self, mx, my) -> bool:
        left = self.x - self.width / 2
        right = self.x + self.width / 2
        bottom = self.y - self.height / 2
        top = self.y + self.height / 2
        return left <= mx <= right and bottom <= my <= top

    def draw(self):
        left = self.x - self.width / 2
        right = self.x + self.width / 2
        bottom = self.y - self.height / 2
        top = self.y + self.height / 2

        # Тень
        arcade.draw_lrbt_rectangle_filled(
            left + 4, right + 4,
            bottom - 6, top - 6,
            BUTTON_SHADOW,
        )

        # Основа (градиент из двух прямоугольников)
        top_color = BUTTON_HOVER_TOP if self.hover else BUTTON_TOP
        bottom_color = BUTTON_HOVER_BOTTOM if self.hover else BUTTON_BOTTOM

        mid_y = (bottom + top) / 2
        arcade.draw_lrbt_rectangle_filled(left, right, mid_y, top, top_color)
        arcade.draw_lrbt_rectangle_filled(left, right, bottom, mid_y, bottom_color)

        # Рамка
        arcade.draw_lrbt_rectangle_outline(
            left, right, bottom, top,
            BUTTON_BORDER, 3,
        )

        # Блик сверху
        arcade.draw_lrbt_rectangle_filled(
            left + 6, right - 6,
            top - 8, top - 4,
            (255, 255, 255, 90),
        )

        # Текст
        arcade.draw_text(
            self.label,
            self.x, self.y - 2,
            TEXT_DARK,
            font_size=22,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )


class MenuView(arcade.View):
    def __init__(self, open_editor=None, open_game=None, open_levels=None):
        super().__init__()
        self.open_editor = open_editor
        self.open_game = open_game
        self.open_levels = open_levels

        self.mouse_x = 0
        self.mouse_y = 0

        # Заголовок
        self.title_shadow = arcade.Text(
            "MARIO",
            0, 0,
            TITLE_RED_DARK,
            96,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

        self.title = arcade.Text(
            "MARIO",
            0, 0,
            TITLE_RED,
            96,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

        self.subtitle = arcade.Text(
            "Выберите режим игры",
            0, 0,
            (255, 255, 255),
            20,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

        # Кнопки (создаются в recalc_layout)
        self.buttons: list[Button] = []

    # ========================================================
    # LAYOUT
    # ========================================================

    def _recalc_layout(self):
        width = self.window.width
        height = self.window.height

        cx = width / 2

        self.title.x = cx
        self.title.y = height - 130

        self.title_shadow.x = cx + 4
        self.title_shadow.y = height - 134

        self.subtitle.x = cx
        self.subtitle.y = height - 210

        button_w = min(360, width * 0.4)
        button_h = 60

        # Пять кнопок: ряд плотнее, чем был для четырёх, и начинается
        # выше — нижняя кнопка не ложится на землю
        start_y = height / 2 + 70
        gap = 76

        labels = [
            ("ИГРАТЬ", "game"),
            ("ИИ", "ai"),
            ("УРОВНИ", "levels"),
            ("РЕДАКТОР УРОВНЕЙ", "editor"),
            ("ВЫХОД", "exit"),
        ]

        self.buttons = []
        for i, (label, action) in enumerate(labels):
            self.buttons.append(
                Button(
                    label, action,
                    cx,
                    start_y - i * gap,
                    button_w, button_h,
                )
            )

    # ========================================================
    # DRAW
    # ========================================================

    def on_draw(self):
        self.clear()
        self._recalc_layout()

        width = self.window.width
        height = self.window.height

        self._draw_background(width, height)
        self._draw_ground(width, height)
        self._draw_clouds(width, height)

        # Заголовок
        self.title_shadow.draw()
        self.title.draw()
        self.subtitle.draw()

        # Кнопки
        for btn in self.buttons:
            btn.hover = btn.hit(self.mouse_x, self.mouse_y)
            btn.draw()

    # ========================================================
    # BACKGROUND
    # ========================================================

    def _draw_background(self, width, height):
        """Градиент неба."""
        draw_sky(width, height)

    def _draw_ground(self, width, height):
        """Земля с травой внизу."""
        draw_ground(width, height)

    def _draw_clouds(self, width, height):
        """Простые облака."""
        draw_clouds(width, height)

    # ========================================================
    # INPUT
    # ========================================================

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x = x
        self.mouse_y = y

    def on_mouse_press(self, x, y, button, modifiers):
        if button != arcade.MOUSE_BUTTON_LEFT:
            return

        for btn in self.buttons:
            if btn.hit(x, y):
                self.handle_action(btn.action)
                return

    # ========================================================
    # ACTIONS
    # ========================================================

    def handle_action(self, action):
        if action == "game":
            if self.open_game is not None:
                self.open_game(ai_mode=False)
            else:
                self.window.show_view(GameView(ai_mode=False))

        elif action == "ai":
            if self.open_game is not None:
                self.open_game(ai_mode=True)
            else:
                self.window.show_view(GameView(ai_mode=True))

        elif action == "levels":
            if self.open_levels is not None:
                self.open_levels()
            else:
                # Импорт здесь: экран уровней сам берёт фон из этого модуля
                from ui.level_select import LevelSelectView

                def back_to_menu():
                    self.window.show_view(
                        MenuView(open_editor=self.open_editor, open_game=self.open_game)
                    )

                self.window.show_view(
                    LevelSelectView(open_game=self.open_game, return_to_menu=back_to_menu)
                )

        elif action == "editor":
            if self.open_editor is not None:
                self.open_editor()
            else:
                self.window.show_view(LevelEditorView())

        elif action == "exit":
            self.window.close()


# ============================================================
# ФОН МЕНЮ (общий для главного меню и экрана «УРОВНИ»)
# ============================================================


def draw_sky(width, height):
    """Градиент неба."""
    steps = 20
    for i in range(steps):
        t = i / steps
        r = int(SKY_TOP[0] * (1 - t) + SKY_BOTTOM[0] * t)
        g = int(SKY_TOP[1] * (1 - t) + SKY_BOTTOM[1] * t)
        b = int(SKY_TOP[2] * (1 - t) + SKY_BOTTOM[2] * t)

        y1 = height * (1 - t)
        y2 = height * (1 - (i + 1) / steps)

        arcade.draw_lrbt_rectangle_filled(
            0, width, y2, y1,
            (r, g, b),
        )


def draw_ground(width, height):
    """Земля с травой внизу."""
    ground_h = 90
    grass_h = 16

    # Земля
    arcade.draw_lrbt_rectangle_filled(
        0, width, 0, ground_h,
        GROUND,
    )
    arcade.draw_lrbt_rectangle_filled(
        0, width, 0, ground_h - 20,
        GROUND_DARK,
    )

    # Трава
    arcade.draw_lrbt_rectangle_filled(
        0, width, ground_h - grass_h, ground_h,
        GRASS,
    )
    arcade.draw_lrbt_rectangle_filled(
        0, width, ground_h - grass_h, ground_h - grass_h / 2,
        GRASS_DARK,
    )

    # Кирпичики на земле
    cell = 32
    for x in range(0, int(width), cell):
        arcade.draw_lrbt_rectangle_outline(
            x, x + cell,
            ground_h - 60, ground_h - 20,
            GROUND_DARK, 2,
        )


def draw_clouds(width, height):
    """Простые облака."""
    def cloud(cx, cy, s=1.0):
        arcade.draw_circle_filled(cx, cy, 28 * s, (255, 255, 255, 230))
        arcade.draw_circle_filled(cx - 26 * s, cy - 6 * s, 20 * s, (255, 255, 255, 230))
        arcade.draw_circle_filled(cx + 26 * s, cy - 6 * s, 22 * s, (255, 255, 255, 230))
        arcade.draw_circle_filled(cx - 12 * s, cy + 10 * s, 18 * s, (255, 255, 255, 230))
        arcade.draw_circle_filled(cx + 14 * s, cy + 12 * s, 16 * s, (255, 255, 255, 230))

    cloud(width * 0.15, height - 260, 1.1)
    cloud(width * 0.80, height - 320, 0.9)
    cloud(width * 0.55, height - 180, 0.7)

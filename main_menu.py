import arcade

from game import GameView


class MenuView(arcade.View):
    def __init__(self):
        super().__init__()

        self.background_color = arcade.color.SKY_BLUE

        self.title = arcade.Text(
            "MARIO",
            0,
            0,
            arcade.color.RED,
            64,
            anchor_x="center",
            anchor_y="center",
        )

        self.subtitle = arcade.Text(
            "Выберите режим игры",
            0,
            0,
            arcade.color.WHITE,
            22,
            anchor_x="center",
            anchor_y="center",
        )

        self.buttons = [
            ("ИГРАТЬ", "game"),
            ("ИИ", "ai"),
            ("РЕДАКТОР УРОВНЕЙ", "editor"),
            ("ВЫХОД", "exit"),
        ]

    def on_draw(self):
        self.clear()

        width = self.window.width
        height = self.window.height

        center_x = width / 2

        self.title.x = center_x
        self.title.y = height - 150
        self.title.draw()

        self.subtitle.x = center_x
        self.subtitle.y = height - 220
        self.subtitle.draw()

        button_width = min(360, width * 0.4)
        button_height = 60

        start_y = height / 2 + 40
        gap = 80

        for index, (text, _) in enumerate(self.buttons):
            y = start_y - index * gap
            self.draw_button(
                text,
                center_x,
                y,
                button_width,
                button_height,
            )

    def draw_button(
        self,
        text,
        x,
        y,
        width,
        height,
    ):
        left = x - width / 2
        bottom = y - height / 2

        arcade.draw_lbwh_rectangle_filled(
            left,
            bottom,
            width,
            height,
            arcade.color.DARK_BLUE,
        )

        arcade.draw_text(
            text,
            x,
            y,
            arcade.color.WHITE,
            font_size=20,
            anchor_x="center",
            anchor_y="center",
        )

    def on_mouse_press(self, x, y, button, modifiers):
        if button != arcade.MOUSE_BUTTON_LEFT:
            return

        width = self.window.width
        height = self.window.height

        center_x = width / 2

        button_width = min(360, width * 0.4)
        button_height = 60

        start_y = height / 2 + 40
        gap = 80

        for index, (_, action) in enumerate(self.buttons):
            button_y = start_y - index * gap

            left = center_x - button_width / 2
            right = center_x + button_width / 2

            bottom = button_y - button_height / 2
            top = button_y + button_height / 2

            if left <= x <= right and bottom <= y <= top:
                self.handle_action(action)
                return

    def handle_action(self, action):
        if action == "game":
            self.window.show_view(GameView(ai_mode=False))

        elif action == "ai":
            self.window.show_view(GameView(ai_mode=True))

        elif action == "editor":
            from level_editor import LevelEditorView

            self.window.show_view(LevelEditorView())

        elif action == "exit":
            self.window.close()
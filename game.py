import arcade
from typing import Optional, Callable

from Entities.player import Player
from map import Map


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Mario"

# Скорость плавного движения камеры
CAMERA_SPEED = 5.0

# Размер зоны, в которой камера не двигается
CAMERA_DEAD_ZONE_X = 400
CAMERA_DEAD_ZONE_Y = 300

# Цвета оверлея паузы
PAUSE_OVERLAY = (0, 0, 0, 160)
PAUSE_PANEL = (34, 38, 56)
PAUSE_PANEL_OUTLINE = (78, 86, 112)
PAUSE_BUTTON = (48, 54, 76)
PAUSE_BUTTON_HOVER = (86, 128, 255)
PAUSE_TEXT = (238, 240, 250)


class PauseButton:

    def __init__(self, label: str, x: float, y: float, w: float, h: float, on_click):
        self.label = label
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.on_click = on_click

    def hit(self, mx: float, my: float) -> bool:
        return (
            self.x <= mx <= self.x + self.width
            and self.y <= my <= self.y + self.height
        )


class GameView(arcade.View):
    def __init__(self, return_to_menu: Optional[Callable[[], None]] = None, ai_mode: bool = False):
        super().__init__()

        self.ai_mode = ai_mode
        self.return_to_menu = return_to_menu

        self.background_color = arcade.color.Color(147, 187, 236)

        # Игровые объекты
        self.player: Optional[Player] = None
        self.map: Optional[Map] = None

        self.enemies = arcade.SpriteList()
        self.walls = arcade.SpriteList()
        self.objects = arcade.SpriteList()
        self.decorations = arcade.SpriteList()

        # Физика
        self.physics_engine: Optional[arcade.PhysicsEnginePlatformer] = None
        self.enemy_physics_engines = {}

        # Камеры
        self.camera: Optional[arcade.Camera2D] = None
        self.gui_camera: Optional[arcade.Camera2D] = None

        # Состояние игры
        self.game_over = False
        self.score = 0

        # ----------------------------------------------------
        # ПАУЗА
        # ----------------------------------------------------
        self.paused = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.pause_buttons: list[PauseButton] = []

        self.setup()
        self._build_pause_buttons()

    # ========================================================
    # SETUP
    # ========================================================

    def setup(self) -> None:
        """Настройка игры"""

        self.player = Player()

        self.map = Map("maps/custom_level.json")
        self.walls = self.map.get_walls()
        self.objects = self.map.get_objects()
        self.decorations = self.map.get_decorations()

        self.enemies = self.map.get_enemies()

        spawn_x, spawn_y = self.map.get_player_spawn()

        self.player = Player()
        self.player.center_x = spawn_x
        self.player.center_y = spawn_y

        self.enemy_physics_engines = {}

        for enemy in self.enemies:
            self.enemy_physics_engines[enemy] = (
                arcade.PhysicsEnginePlatformer(
                    player_sprite=enemy,
                    walls=self.walls,
                    gravity_constant=enemy.gravity
                )
            )

        self.physics_engine = arcade.PhysicsEnginePlatformer(
            player_sprite=self.player,
            walls=self.walls,
            gravity_constant=self.player.gravity
        )

        self.camera = arcade.Camera2D()
        self.gui_camera = arcade.Camera2D()

        self.camera.position = (
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2
        )

        self.gui_camera.position = (
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2
        )

        self.game_over = False
        self.score = 0

        # Сбрасываем паузу
        self.paused = False

    # ========================================================
    # PAUSE BUTTONS
    # ========================================================

    def _build_pause_buttons(self):
        """Создаёт кнопки оверлея паузы."""

        panel_w, panel_h = 460, 260
        cx = WINDOW_WIDTH / 2
        cy = WINDOW_HEIGHT / 2

        bw, bh = 260, 56
        gap = 20

        # Кнопки расположены по центру панели ниже заголовка
        first_y = cy + 10
        second_y = first_y - bh - gap

        self.pause_buttons = [
            PauseButton(
                "Продолжить",
                cx - bw / 2,
                first_y - bh / 2,
                bw, bh,
                self.resume_game,
            ),
            PauseButton(
                "Выйти в меню",
                cx - bw / 2,
                second_y - bh / 2,
                bw, bh,
                self.exit_to_menu,
            ),
        ]

    def pause_game(self):
        self.paused = True

    def resume_game(self):
        self.paused = False

    def exit_to_menu(self):
        """Возврат в главное меню."""
        self.paused = False
        if self.return_to_menu is not None:
            self.return_to_menu()
        else:
            self.close()

    # ========================================================
    # DRAW
    # ========================================================

    def on_draw(self) -> None:
        """Отрисовка"""

        self.clear()

        # =====================
        # ИГРОВОЙ МИР
        # =====================

        if self.camera:
            self.camera.use()

        if self.map:
            self.walls.draw()
            self.decorations.draw()
            self.objects.draw()

        self.enemies.draw()

        if self.player:
            self.player.playerSprite.draw()

        # =====================
        # HUD
        # =====================

        if self.gui_camera:
            self.gui_camera.use()

        arcade.draw_text(
            f"Score: {self.score}",
            10,
            WINDOW_HEIGHT - 30,
            arcade.color.WHITE,
            16
        )

        arcade.draw_text(
            f"Lives: {self.player.health if self.player else 0}",
            10,
            WINDOW_HEIGHT - 55,
            arcade.color.WHITE,
            16
        )

        if self.game_over:
            arcade.draw_text(
                "GAME OVER",
                WINDOW_WIDTH / 2,
                WINDOW_HEIGHT / 2 + 30,
                arcade.color.RED,
                36,
                anchor_x="center",
                anchor_y="center"
            )

            arcade.draw_text(
                "Press R to restart",
                WINDOW_WIDTH / 2,
                WINDOW_HEIGHT / 2 - 20,
                arcade.color.WHITE,
                20,
                anchor_x="center",
                anchor_y="center"
            )

        # =====================
        # PAUSE OVERLAY
        # =====================
        if self.paused and not self.game_over:
            self._draw_pause_overlay()

    def _draw_pause_overlay(self):
        """Отрисовка оверлея паузы."""

        # Затемнение фона
        arcade.draw_lrbt_rectangle_filled(
            0, WINDOW_WIDTH, 0, WINDOW_HEIGHT,
            PAUSE_OVERLAY,
        )

        # Панель
        panel_w, panel_h = 460, 260
        cx = WINDOW_WIDTH / 2
        cy = WINDOW_HEIGHT / 2

        left = cx - panel_w / 2
        right = cx + panel_w / 2
        bottom = cy - panel_h / 2
        top = cy + panel_h / 2

        arcade.draw_lrbt_rectangle_filled(
            left + 4, right + 4, bottom - 6, top - 6,
            (0, 0, 0, 60),
        )
        arcade.draw_lrbt_rectangle_filled(
            left, right, bottom, top,
            PAUSE_PANEL,
        )
        arcade.draw_lrbt_rectangle_outline(
            left, right, bottom, top,
            PAUSE_PANEL_OUTLINE, 2,
        )

        # Заголовок
        arcade.draw_text(
            "ПАУЗА",
            cx,
            top - 50,
            PAUSE_TEXT,
            28,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

        # Кнопки
        for btn in self.pause_buttons:
            hover = btn.hit(self.mouse_x, self.mouse_y)
            color = PAUSE_BUTTON_HOVER if hover else PAUSE_BUTTON

            arcade.draw_lrbt_rectangle_filled(
                btn.x, btn.x + btn.width,
                btn.y, btn.y + btn.height,
                color,
            )
            arcade.draw_lrbt_rectangle_outline(
                btn.x, btn.x + btn.width,
                btn.y, btn.y + btn.height,
                PAUSE_PANEL_OUTLINE, 1,
            )

            arcade.draw_text(
                btn.label,
                btn.x + btn.width / 2,
                btn.y + btn.height / 2,
                PAUSE_TEXT,
                18,
                anchor_x="center",
                anchor_y="center",
                bold=True,
            )

    # ========================================================
    # UPDATE
    # ========================================================

    def on_update(self, delta_time: float) -> None:
        """Обновление игры"""

        # Пауза или game over — ничего не обновляем
        if self.paused or self.game_over or not self.player:
            return

        # =====================
        # ФИЗИКА
        # =====================
        if self.physics_engine:
            self.physics_engine.update()

        can_jump = False
        if self.physics_engine:
            can_jump = self.physics_engine.can_jump()

        # =====================
        # ИГРОК
        # =====================
        self.player.update(can_jump, delta_time)
        self.player.update_animations(delta_time)

        # =====================
        # ФИЗИКА ВРАГОВ
        # =====================
        for enemy, physics_engine in self.enemy_physics_engines.items():
            if not enemy.is_alive:
                continue
            physics_engine.update()

        # =====================
        # ВРАГИ
        # =====================
        for enemy in list(self.enemies):
            enemy.update(delta_time, self.walls)
            enemy.update_animations(delta_time)

        # =====================
        # СТОЛКНОВЕНИЯ
        # =====================

        self.player.check_enemy_collides(self.enemies)
        self.player.check_objects_collides(self, self.objects)

        # =====================
        # УДАЛЕНИЕ ВРАГОВ
        # =====================
        for enemy in list(self.enemies):
            if not enemy.is_alive:
                if enemy.death_timer > 0.5:
                    enemy.remove_from_sprite_lists()
                    self.score += 100

        # =====================
        # КАМЕРА
        # =====================
        self.update_camera(delta_time)

        # =====================
        # ПАДЕНИЕ ВНИЗ
        # =====================
        if self.player.center_y < -100:
            self.player.dead()
            self.game_over = True

    # ========================================================
    # CAMERA
    # ========================================================

    def update_camera(self, delta_time: float) -> None:
        """Плавная камера с dead zone"""

        if not self.player or not self.camera:
            return

        camera_x = self.camera.position.x
        camera_y = self.camera.position.y

        left = camera_x - WINDOW_WIDTH / 2
        right = camera_x + WINDOW_WIDTH / 2
        bottom = camera_y - WINDOW_HEIGHT / 2
        top = camera_y + WINDOW_HEIGHT / 2

        dead_left = left + CAMERA_DEAD_ZONE_X
        dead_right = right - CAMERA_DEAD_ZONE_X
        dead_bottom = bottom + CAMERA_DEAD_ZONE_Y
        dead_top = top - CAMERA_DEAD_ZONE_Y

        target_x = camera_x
        target_y = camera_y

        if self.player.center_x < dead_left:
            target_x += self.player.center_x - dead_left
        elif self.player.center_x > dead_right:
            target_x += self.player.center_x - dead_right

        if self.player.center_y < dead_bottom:
            target_y += self.player.center_y - dead_bottom
        elif self.player.center_y > dead_top:
            target_y += self.player.center_y - dead_top

        target_x = max(WINDOW_WIDTH / 2, target_x)
        target_y = max(WINDOW_HEIGHT / 2, target_y)

        speed = min(CAMERA_SPEED * delta_time, 1.0)

        new_x = camera_x + (target_x - camera_x) * speed
        new_y = camera_y + (target_y - camera_y) * speed

        self.camera.position = (new_x, new_y)

    # ========================================================
    # MOUSE
    # ========================================================

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x = x
        self.mouse_y = y

    def on_mouse_press(self, x, y, button, modifiers):
        if self.paused:
            if button == arcade.MOUSE_BUTTON_LEFT:
                for btn in self.pause_buttons:
                    if btn.hit(x, y):
                        btn.on_click()
                        return

    # ========================================================
    # KEYBOARD
    # ========================================================

    def on_key_press(self, symbol: int, modifiers: int) -> None:

        # =====================
        # ПАУЗА
        # =====================
        if self.paused:
            if symbol == arcade.key.ESCAPE:
                self.resume_game()
            elif symbol == arcade.key.ENTER:
                self.resume_game()
            return

        # =====================
        # GAME OVER
        # =====================
        if self.game_over:
            if symbol == arcade.key.R:
                self.setup()
            return

        if not self.player:
            return

        if symbol == arcade.key.RIGHT:
            self.player.move_right()
        elif symbol == arcade.key.LEFT:
            self.player.move_left()
        elif symbol in (arcade.key.SPACE, arcade.key.UP):
            self.player.jump()
        elif symbol == arcade.key.ESCAPE:
            self.pause_game()

    def on_key_release(self, symbol: int, modifiers: int) -> None:

        if not self.player:
            return

        if symbol in (arcade.key.RIGHT, arcade.key.LEFT):
            self.player.stop_moving()
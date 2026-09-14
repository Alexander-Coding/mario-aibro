import arcade
from typing import Optional

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


class GameView(arcade.View):
    def __init__(self, ai_mode: bool = False):
        super().__init__()

        self.ai_mode = ai_mode

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

        # Физика врагов
        self.enemy_physics_engines = {}

        # Камеры
        self.camera: Optional[arcade.Camera2D] = None
        self.gui_camera: Optional[arcade.Camera2D] = None

        # Состояние игры
        self.game_over = False
        self.score = 0

        self.setup()

    def setup(self) -> None:
        """Настройка игры"""

        # Игрок
        self.player = Player()

        # Карта
        self.map = Map("maps/custom_level.json")
        self.walls = self.map.get_walls()
        self.objects = self.map.get_objects()
        self.decorations = self.map.get_decorations()

        # Враги
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

        # Физика игрока
        self.physics_engine = arcade.PhysicsEnginePlatformer(
            player_sprite=self.player,
            walls=self.walls,
            gravity_constant=self.player.gravity
        )



        # Камеры
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

    def on_draw(self) -> None:
        """Отрисовка"""

        self.clear()

        # =====================
        # ИГРОВОЙ МИР
        # =====================

        if self.camera:
            self.camera.use()

        # Карта
        if self.map:
            self.walls.draw()
            self.decorations.draw()
            self.objects.draw()

        # Враги
        self.enemies.draw()

        # Игрок
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

    def on_update(self, delta_time: float) -> None:
        """Обновление игры"""

        if self.game_over or not self.player:
            return

        # =====================
        # ФИЗИКА
        # =====================

        if self.physics_engine:
            self.physics_engine.update()


        # Проверяем, стоит ли игрок на земле
        can_jump = False

        if self.physics_engine:
            can_jump = self.physics_engine.can_jump()

        # =====================
        # ИГРОК
        # =====================

        self.player.update(
            can_jump,
            delta_time
        )

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
        self.player.check_objects_collides(self.objects)

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

    def update_camera(self, delta_time: float) -> None:
        """Плавная камера с dead zone"""

        if not self.player or not self.camera:
            return

        # Camera2D.position — ЦЕНТР камеры
        camera_x = self.camera.position.x
        camera_y = self.camera.position.y

        # Левая/правая границы экрана в мире
        left = camera_x - WINDOW_WIDTH / 2
        right = camera_x + WINDOW_WIDTH / 2

        # Нижняя/верхняя границы экрана
        bottom = camera_y - WINDOW_HEIGHT / 2
        top = camera_y + WINDOW_HEIGHT / 2

        # Dead zone внутри экрана
        dead_left = left + CAMERA_DEAD_ZONE_X
        dead_right = right - CAMERA_DEAD_ZONE_X

        dead_bottom = bottom + CAMERA_DEAD_ZONE_Y
        dead_top = top - CAMERA_DEAD_ZONE_Y

        # Камера пока остаётся на месте
        target_x = camera_x
        target_y = camera_y

        # ==========================
        # ГОРИЗОНТАЛЬНОЕ ДВИЖЕНИЕ
        # ==========================

        if self.player.center_x < dead_left:

            target_x += (
                    self.player.center_x
                    - dead_left
            )

        elif self.player.center_x > dead_right:

            target_x += (
                    self.player.center_x
                    - dead_right
            )

        # ==========================
        # ВЕРТИКАЛЬНОЕ ДВИЖЕНИЕ
        # ==========================

        if self.player.center_y < dead_bottom:

            target_y += (
                    self.player.center_y
                    - dead_bottom
            )

        elif self.player.center_y > dead_top:

            target_y += (
                    self.player.center_y
                    - dead_top
            )

        # Не позволяем камере показывать
        # область левее начала карты
        target_x = max(
            WINDOW_WIDTH / 2,
            target_x
        )

        # Не позволяем камере идти ниже земли
        target_y = max(
            WINDOW_HEIGHT / 2,
            target_y
        )

        # ==========================
        # ПЛАВНОЕ ДВИЖЕНИЕ
        # ==========================

        speed = min(
            CAMERA_SPEED * delta_time,
            1.0
        )

        new_x = camera_x + (
                target_x - camera_x
        ) * speed

        new_y = camera_y + (
                target_y - camera_y
        ) * speed

        self.camera.position = (
            new_x,
            new_y
        )

    def on_key_press(
        self,
        symbol: int,
        modifiers: int
    ) -> None:

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

        elif symbol in (
            arcade.key.SPACE,
            arcade.key.UP
        ):
            self.player.jump()

        elif symbol == arcade.key.ESCAPE:
            self.close()

    def on_key_release(
        self,
        symbol: int,
        modifiers: int
    ) -> None:

        if not self.player:
            return

        if symbol in (
            arcade.key.RIGHT,
            arcade.key.LEFT
        ):
            self.player.stop_moving()



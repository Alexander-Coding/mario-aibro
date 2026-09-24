import arcade

from Entities.entity import Entity


class Player(Entity):
    def __init__(self):
        super().__init__(
            "resources/Entities/mario",
            scale=1.0
        )

        self.health = 3

        # Настройки
        self.jump_speed = 15
        self.move_speed = 4

        self.center_x = 32
        self.center_y = 100

        # Состояние
        self.is_jumping = False
        self.on_ground = False

        # Запрос прыжка
        self.jump_requested = False

        # Неуязвимость
        self.invincible_timer = 0.0
        self.invincible_duration = 1.0

        # Направление
        self.facing_direction = 1

        # Анимация
        self.animation_speed = 0.08

        # Спрайт
        self.playerSprite = arcade.SpriteList()
        self.playerSprite.append(self)

    def jump(self) -> None:
        """Запрос прыжка"""

        if not self.is_alive:
            return

        self.jump_requested = True

    def update(
        self,
        on_ground: bool,
        delta_time: float = 0
    ) -> None:

        if not self.is_alive:
            return

        # Получаем актуальную информацию
        # от PhysicsEngine
        self.on_ground = on_ground

        # =====================
        # ПРЫЖОК
        # =====================

        if self.jump_requested:

            if self.on_ground:

                self.change_y = self.jump_speed

                self.is_jumping = True

                self.on_ground = False

                self.change_state("jump")

            # Обрабатываем запрос
            self.jump_requested = False

        # =====================
        # НЕУЯЗВИМОСТЬ
        # =====================

        if self.invincible_timer > 0:

            self.invincible_timer -= delta_time

            if int(
                self.invincible_timer * 10
            ) % 2 == 0:

                self.alpha = 128

            else:
                self.alpha = 255

        else:

            self.invincible_timer = 0
            self.alpha = 255

        # =====================
        # СОСТОЯНИЯ
        # =====================

        if not self.on_ground:

            self.is_jumping = True

            if self.state != "jump":
                self.change_state("jump")

        else:

            self.is_jumping = False

            if self.change_x != 0:

                if self.state != "move":
                    self.change_state("move")

            else:

                if self.state != "idle":
                    self.change_state("idle")

    def check_enemy_collides(self, enemies: arcade.SpriteList) -> None:

        if not self.is_alive:
            return

        if self.invincible_timer > 0:
            return

        # «Сверху или сбоку» решается по скорости ДО отскока: иначе при
        # прыжке на двух гумб вплотную первая даёт отскок вверх, и вторая
        # в том же кадре засчитывается как удар сбоку
        falling = self.change_y
        stomped = False

        for enemy in enemies:
            if not enemy.is_alive:
                continue

            if not self.collides_with_sprite(enemy):
                continue

            health = self.health
            self.change_y = falling
            enemy.on_collides_with_player(self)

            if self.health < health or not self.is_alive:
                # Удар: отбрасывает от врага, а не всегда влево
                self.change_x = 5 if self.center_x >= enemy.center_x else -5
                return

            if self.change_y != falling:
                stomped = True

        if stomped:
            self.change_y = self.jump_speed * 0.6

    def check_objects_collides(self, game, objects: arcade.SpriteList) -> None:

        if not self.is_alive:
            return

        for obj in objects:
            if self.collides_with_sprite(obj):
                if hasattr(obj, "on_collides_with_player"):
                    obj.on_collides_with_player(game, self)

    def get_hit(self) -> None:

        if self.invincible_timer > 0:
            return

        super().get_hit()

        self.invincible_timer = (
            self.invincible_duration
        )

        self.change_y = 5

        self.change_x = (
            -5 * self.facing_direction
        )

    def dead(self) -> None:

        super().dead()

        self.change_x = 0
        self.change_y = 10
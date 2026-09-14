import arcade
from typing import Optional
from Entities.entity import Entity


class Enemy(Entity):
    def __init__(self, path_texture: str, scale: float = 1.0):
        super().__init__(path_texture, scale=scale)

        self.is_stomped: bool = False
        self.is_shell: bool = False
        self.stun_timer: float = 0
        self.death_timer: float = 0
        self.move_direction: int = -1  # -1 left, 1 right
        self.move_range: float = 200  # Дистанция движения
        self.start_x: float = 0

        # Настройка движения
        self.change_x = self.move_direction * self.move_speed

    def setup_movement(self, start_x: float) -> None:
        """Настройка начальной позиции для движения"""
        self.start_x = start_x

    def on_collides_with_player(self, player) -> None:
        """Обработка столкновения с игроком"""
        if self.is_stomped or not self.is_alive:
            return

        # Проверка, наступает ли игрок сверху
        if (player.change_y < 0 and player.center_y > self.center_y + self.height // 3):
            self.get_hit()
            # Игрок подпрыгивает после уничтожения врага
            player.change_y = player.jump_speed * 0.6
            return

        # Игрок получает урон
        player.get_hit()


    def update(self, delta_time=0, walls=None):
        if not self.is_alive:
            # Анимация смерти
            self.death_timer += delta_time
            if self.death_timer > 0.5:
                self.kill()
            return

        if not walls:
            return

        # Проверяем пространство перед врагом
        next_x = self.center_x + self.move_direction * self.move_speed

        old_x = self.center_x
        self.center_x = next_x

        hit_list = arcade.check_for_collision_with_list(self, walls)

        self.center_x = old_x

        if hit_list:
            # Стена впереди — разворачиваемся
            self.move_direction *= -1


        else:
            # Стены нет — продолжаем движение
            if self.move_direction > 0:
                self.move_right()
            if self.move_direction < 0:
                self.move_left()

    def get_hit(self) -> None:
        """Уничтожение врага"""
        self.is_stomped = True
        self.is_alive = False
        self.change_state("dead")
        self.change_x = 0
        self.death_timer = 0

    def dead(self) -> None:
        """Переопределение смерти"""
        self.is_stomped = True
        super().dead()
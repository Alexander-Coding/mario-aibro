import arcade

from Entities.enemy import Enemy


class Koopa(Enemy):

    def __init__(
        self,
        x: float = 300,
        y: float = 100
    ):
        super().__init__(
            "resources/Entities/koopa",
            scale=2
        )

        # Позиция
        self.center_x = x
        self.center_y = y

        # Движение
        self.move_speed = 1.5
        self.move_direction = -1

        self.change_x = (
            self.move_direction *
            self.move_speed
        )

        self.start_x = x

        # Здоровье
        self.health = 1

        # Анимация
        self.animation_speed = 0.15

        # Состояние Koopa
        self.is_stomped = False
        self.is_shell = False

        # =====================================================
        # Панцирь
        # =====================================================

        # Время нахождения в панцире
        self.shell_duration = 5.0

        # Оставшееся время
        self.shell_timer = 0.0

        # =====================================================
        # Таймеры
        # =====================================================

        self.stun_timer = 0.0
        self.death_timer = 0.0

        # Направление спрайта
        self.set_direction(
            self.move_direction
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        delta_time: float = 0,
        walls=None
    ) -> None:

        # Если Koopa мёртвая
        if not self.is_alive:
            self.death_timer += delta_time
            return

        # =====================================================
        # PANZER
        # =====================================================

        if self.is_shell:

            # Останавливаем движение
            self.change_x = 0

            # Отсчитываем время
            self.shell_timer -= delta_time

            # Если время закончилось
            if self.shell_timer <= 0:
                self.wake_up()

            return

        # =====================================================
        # ОБЫЧНОЕ ДВИЖЕНИЕ
        # =====================================================

        if not walls:
            return

        # Проверяем пространство перед Koopa
        next_x = (
            self.center_x
            + self.move_direction * self.move_speed
        )

        old_x = self.center_x

        self.center_x = next_x

        hit_list = self.walls_ahead(walls)

        self.center_x = old_x

        # =====================================================
        # СТЕНА
        # =====================================================

        if hit_list:

            self.reverse_direction()

        else:

            self.change_x = (
                self.move_direction
                * self.move_speed
            )

            self.set_direction(
                self.move_direction
            )

            self.change_state("move")

    # =========================================================
    # REVERSE
    # =========================================================

    def reverse_direction(self) -> None:

        self.move_direction *= -1

        self.change_x = (
            self.move_direction
            * self.move_speed
        )

        self.set_direction(
            self.move_direction
        )

    # =========================================================
    # COLLISION WITH PLAYER
    # =========================================================

    def on_collides_with_player(
        self,
        player
    ) -> None:

        if not self.is_alive:
            return

        # =====================================================
        # KOOPA В ПАНЦИРЕ
        # =====================================================

        if self.is_shell:

            # Игрок снова наступил сверху
            if (
                player.change_y < 0
                and
                player.center_y >
                self.center_y + self.height / 3
            ):

                # Полностью убиваем Koopa
                self.kill()

                # Отскок игрока
                player.change_y = (
                    player.jump_speed * 0.6
                )

            return

        # =====================================================
        # ОБЫЧНАЯ KOOPA
        # =====================================================

        # Игрок прыгнул сверху
        if (
            player.change_y < 0
            and
            player.center_y >
            self.center_y + self.height / 3
        ):

            self.get_hit()

            # Отскок игрока
            player.change_y = (
                player.jump_speed * 0.6
            )

            return

        # Игрок столкнулся сбоку
        player.get_hit()

    # =========================================================
    # GET HIT
    # =========================================================

    def get_hit(self) -> None:

        if not self.is_alive:
            return

        # Переводим Koopa в панцирь
        self.is_stomped = True
        self.is_shell = True

        # Запускаем таймер
        self.shell_timer = self.shell_duration

        # Останавливаем движение
        self.change_x = 0

        # Ставим анимацию панциря
        self.change_state("shield")

    # =========================================================
    # WAKE UP
    # =========================================================

    def wake_up(self) -> None:

        if not self.is_alive:
            return

        # Выходим из панциря
        self.is_shell = False
        self.is_stomped = False

        # Сбрасываем таймер
        self.shell_timer = 0.0

        # Возвращаем движение
        self.change_x = (
            self.move_direction
            * self.move_speed
        )

        # Возвращаем направление
        self.set_direction(
            self.move_direction
        )

        # Возвращаем анимацию
        self.change_state("move")

    # =========================================================
    # KILL
    # =========================================================

    def kill(self) -> None:

        if not self.is_alive:
            return

        self.is_alive = False

        self.is_stomped = True
        self.is_shell = False

        self.change_x = 0

        self.death_timer = 0.0

        # Анимация смерти
        if "dead" in self.texture_dict:
            self.change_state("dead")

    # =========================================================
    # DEAD
    # =========================================================

    def dead(self) -> None:

        self.kill()
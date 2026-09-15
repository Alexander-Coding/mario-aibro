import random
import time

import arcade

from Entities.player import Player


class ScoreBlock(arcade.Sprite):
    # Сколько секунд без касаний считать, что игрок отошёл
    TOUCH_TIMEOUT = 0.08

    def __init__(self, x, y,  scale=2):
        super().__init__(scale=scale)

        self.center_x = x
        self.center_y = y

        self.state = random.randint(1, 4)
        self.texture = arcade.load_texture(
            f"resources/objects/ScoreBlock/img_{self.state}.png"
        )

        # Время последнего касания игроком (монотонное)
        self._last_touch_time = 0.0
        # Обработано ли текущее касание
        self._hit_processed = False

    # ----------------------------------------------------
    # PRIVATE
    # ----------------------------------------------------

    def _touch_is_active(self) -> bool:
        """True, если игрок касался блока в этом «сеансе»."""
        return (time.monotonic() - self._last_touch_time) < self.TOUCH_TIMEOUT

    def _refresh_touch(self):
        """Отметить, что игрок касается блока прямо сейчас."""
        self._last_touch_time = time.monotonic()

    def _reset_touch(self):
        """Игрок больше не касается — разрешаем следующее срабатывание."""
        self._hit_processed = False

    # ----------------------------------------------------
    # PUBLIC
    # ----------------------------------------------------

    def on_collides_with_player(self, game, player: Player) -> None:
        """Вызывается, пока игрок пересекается с блоком (каждый кадр)."""
        # Если с прошлого касания прошло слишком много времени —
        # значит был разрыв, начинаем новый «сеанс»
        if not self._touch_is_active():
            self._hit_processed = False

        # Обновляем время касания
        self._refresh_touch()

        # Уже обработали это касание
        if self._hit_processed:
            return

        # Пустой блок
        if self.state == 0:
            self._hit_processed = True
            return

        # Бьём только снизу вверх
        if player.change_y <= 0:
            return

        # Засчитываем удар
        self._hit_processed = True

        game.score += 100
        self.state -= 1

        if self.state > 0:
            self.texture = arcade.load_texture(
                f"resources/objects/ScoreBlock/img_{self.state}.png"
            )
        else:
            try:
                self.texture = arcade.load_texture(
                    "resources/objects/ScoreBlock/img_0.png"
                )
            except Exception:
                pass
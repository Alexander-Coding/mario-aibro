import random
import time

import arcade

from Entities.player import Player


class ScoreBlock(arcade.Sprite):
    # Сколько секунд без касаний считать, что игрок отошёл
    TOUCH_TIMEOUT = 0.08

    def __init__(self, x, y,  scale=2, hits=None):
        super().__init__(scale=scale)

        self.center_x = x
        self.center_y = y

        # Сколько ударов в блоке. Случайно, но детерминированно по позиции:
        # один и тот же уровень даёт одинаковые блоки игроку и нейросети,
        # иначе соревнование «кто больше набрал» нечестное.
        if hits is None:
            hits = random.Random(f"{float(x):.1f}:{float(y):.1f}").randint(1, 4)

        self.state = int(hits)
        self.texture = arcade.load_texture(
            f"resources/objects/ScoreBlock/img_{self.state}.png"
        )

        # Время последнего касания игроком (монотонное)
        self._last_touch_time = -1.0
        # Обработано ли текущее касание
        self._hit_processed = False

    # ----------------------------------------------------
    # PRIVATE
    # ----------------------------------------------------

    @staticmethod
    def _now(game) -> float:
        """Время игры, а не часы машины.

        Мир (core/world.py) считает время кадрами. На ускоренном обучении
        кадр идёт за доли миллисекунды, и настенные часы засчитали бы
        один удар на сотни кадров вперёд.
        """
        elapsed = getattr(game, "elapsed", None)
        if elapsed is None:
            return time.monotonic()
        return float(elapsed)

    def _touch_is_active(self, game=None) -> bool:
        """True, если игрок касался блока в этом «сеансе»."""
        return (self._now(game) - self._last_touch_time) < self.TOUCH_TIMEOUT

    def _refresh_touch(self, game=None):
        """Отметить, что игрок касается блока прямо сейчас."""
        self._last_touch_time = self._now(game)

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
        if not self._touch_is_active(game):
            self._hit_processed = False

        # Обновляем время касания
        self._refresh_touch(game)

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

        if hasattr(game, "add_score"):
            game.add_score(100, "block")
        else:
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
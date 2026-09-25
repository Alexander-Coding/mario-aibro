import random

import arcade


class ScoreBlock(arcade.Sprite):
    """Блок «?» — твёрдый, как кирпич: на нём стоят, сквозь него не пройти.

    Удар головой снизу выбивает монету: +100 очков, пока в блоке остались
    удары (1–4). Пустой блок остаётся стеной. Удар засчитывает мир
    (core/world.py) по тому, во что упёрся игрок, двигаясь вверх: один
    прыжок — один удар, как в Super Mario Bros.
    """

    MAX_HITS = 4

    def __init__(self, x, y, scale=2, hits=None):
        super().__init__(scale=scale)

        self.center_x = x
        self.center_y = y

        # Сколько ударов в блоке. Случайно, но детерминированно по позиции:
        # один и тот же уровень даёт одинаковые блоки игроку и нейросети,
        # иначе соревнование «кто больше набрал» нечестное.
        if hits is None:
            hits = random.Random(f"{float(x):.1f}:{float(y):.1f}").randint(1, self.MAX_HITS)

        self.state = int(hits)
        self.initial_hits = self.state
        self._set_texture()

    def _set_texture(self) -> None:
        self.texture = arcade.load_texture(
            f"resources/objects/ScoreBlock/img_{max(self.state, 0)}.png"
        )

    @property
    def empty(self) -> bool:
        return self.state <= 0

    def hit(self) -> bool:
        """Удар снизу. True — выбита монета (блок не был пустым)."""

        if self.state <= 0:
            return False

        self.state -= 1
        self._set_texture()
        return True

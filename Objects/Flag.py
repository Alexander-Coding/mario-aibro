import arcade
from Entities.player import Player


class Flag(arcade.Sprite):
    def __init__(self, x, y,  scale = 2):
        super().__init__(scale=scale)

        self.center_x = x
        self.center_y = y

        self.max_height = 0

    def on_collides_with_player(self, game, player: Player) -> None:
        """Касание флага ничего не делает само.

        Захват флагштока, съезд вниз вместе с флагом, очки за высоту и
        проход в замок считает GameWorld (core/world.py) — по штоку, а не
        по спрайту флага, чтобы схватиться можно было на любой высоте.
        """
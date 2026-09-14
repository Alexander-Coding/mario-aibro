import arcade
from Entities.player import Player


class Coin(arcade.Sprite):
    def __init__(self, scale = 2):
        super().__init__(scale=scale)

    def on_collides_with_player(self, player: Player) -> None:
        """Обработка столкновения с игроком"""
        self.kill()
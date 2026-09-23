import arcade
from Entities.player import Player


class Coin(arcade.Sprite):
    def __init__(self, x, y, scale = 2):
        super().__init__(scale=scale)

        self.center_x = x
        self.center_y = y

    def on_collides_with_player(self, game, player: Player) -> None:
        """Обработка столкновения с игроком"""
        if hasattr(game, "add_score"):
            game.add_score(100, "coin")
        else:
            game.score += 100
        self.kill()
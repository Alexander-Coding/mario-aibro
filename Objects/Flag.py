import arcade
from Entities.player import Player


class Flag(arcade.Sprite):
    def __init__(self, x, y,  scale = 2):
        super().__init__(scale=scale)

        self.center_x = x
        self.center_y = y

        self.max_height = 0

    def on_collides_with_player(self, game, player: Player) -> None:
        """Обработка столкновения с игроком"""
        self.max_height = max(self.center_y, self.max_height)
        if player.center_y < self.center_y and not player.on_ground and self.max_height - self.center_y < 160:
            self.center_y = player.center_y
            player.center_x = self.center_x + 24
            player.stop_moving()
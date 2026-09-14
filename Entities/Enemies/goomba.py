from Entities.enemy import Enemy


class Goomba(Enemy):
    def __init__(self, x: float = 300, y: float = 100):
        super().__init__("resources/entities/goomba", scale=2)

        self.center_x = x
        self.center_y = y
        self.move_speed = 1.5
        self.move_direction = -1
        self.change_x = self.move_direction * self.move_speed
        self.start_x = x

        # Goomba специфичные параметры
        self.health = 1
        self.animation_speed = 0.15

        # Убедимся, что у него есть состояние "dead"
        if "dead" not in self.texture_dict:
            # Используем первый кадр как dead, если нет отдельной анимации
            first_texture = self.texture
            self.texture_dict["dead"] = [first_texture]

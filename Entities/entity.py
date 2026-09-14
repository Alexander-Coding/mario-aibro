import os
import arcade

from typing import Dict, List


class Entity(arcade.Sprite):

    def __init__(self, resource_path: str, scale: float = 1.0):
        super().__init__(scale=scale)
        # Анимация
        self.state = "idle"

        self.texture_dict: Dict[str, List[arcade.Texture]] = {}

        self.current_frame = 0
        self.frame_timer = 0.0

        self.animation_speed = 0.1

        # Направление
        self.direction = 1
        self.base_scale = scale

        # Основные параметры
        self.is_alive = True
        self.health = 1

        self.move_speed = 2

        # Гравитация
        self.gravity = 0.5

        # Загрузка текстур
        self.load_animations(resource_path)

        if "idle" in self.texture_dict:
            self.texture = self.texture_dict["idle"][0]

    def load_animations(self, path: str) -> None:

        if not os.path.exists(path):
            print(f"Папка не найдена: {path}")
            return

        for folder_name in os.listdir(path):

            folder_path = os.path.join(path, folder_name)

            if not os.path.isdir(folder_path):
                continue

            textures = []

            for file_name in sorted(os.listdir(folder_path)):

                file_path = os.path.join(folder_path, file_name)

                if not os.path.isfile(file_path) and not file_path.endswith((".png", ".jpg", ".jpeg", ".gif")):
                    continue

                try:
                    texture = arcade.load_texture(file_path)
                    textures.append(texture)

                except Exception as error:
                    print(f"Ошибка загрузки {file_path}: {error}")

            if textures:
                self.texture_dict[folder_name] = textures

    def update_animations(self, delta_time: float) -> None:
        if not self.is_alive:
            return

        frames = self.texture_dict.get(
            self.state
        )

        if not frames:
            return

        if len(frames) == 1:

            self.texture = frames[0]

            return

        self.frame_timer += delta_time

        if self.frame_timer >= self.animation_speed:

            self.frame_timer = 0

            self.current_frame = (
                self.current_frame + 1
            ) % len(frames)

            self.texture = frames[
                self.current_frame
            ]

    def change_state(self, new_state: str) -> None:
        if new_state == self.state:
            return

        self.state = new_state

        self.current_frame = 0
        self.frame_timer = 0

        frames = self.texture_dict.get(
            self.state
        )
        if frames:
            self.texture = frames[0]

    def set_direction(self, direction: int) -> None:
        if direction > 0:
            self.direction = 1
            self.scale_x = self.base_scale
        elif direction < 0:
            self.direction = -1
            self.scale_x = -self.base_scale

    def move_left(self) -> None:
        if not self.is_alive:
            return
        self.change_x = -self.move_speed
        self.set_direction(-1)
        self.change_state("move")

    def move_right(self) -> None:
        if not self.is_alive:
            return
        self.change_x = self.move_speed
        self.set_direction(1)
        self.change_state("move")

    def stop_moving(self) -> None:
        self.change_x = 0
        if self.is_alive:
            self.change_state("idle")

    def get_hit(self) -> None:
        self.health -= 1
        if self.health <= 0:
            self.dead()

    def dead(self) -> None:
        self.is_alive = False
        if "dead" in self.texture_dict:
            self.change_state("dead")
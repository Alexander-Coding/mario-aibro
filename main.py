import arcade

import main_menu
from level_editor import LevelEditorView
from game import GameView
from ui.level_select import LevelSelectView

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Mario"

arcade.SpriteList.DEFAULT_TEXTURE_FILTER = (
    arcade.gl.NEAREST,
    arcade.gl.NEAREST,
)


def main():
    window = arcade.Window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, resizable=True)
    print("[main] window created")

    def make_menu():
        return main_menu.MenuView(
            open_editor=open_editor,
            open_game=open_game,
            open_levels=open_levels,
        )

    def open_menu():
        print("[main] open_menu called")
        window.show_view(make_menu())

    def open_editor():
        print("[main] open_editor called")
        window.show_view(LevelEditorView(return_to_menu=open_menu))

    def open_game(ai_mode: bool = False, start_index: int = 0):
        # «ИГРАТЬ» — с первого уровня, «ИИ» — нейросеть проходит все уровни
        # подряд, из экрана «УРОВНИ» — с выбранного уровня до конца манифеста
        print("[main] open_game called, ai_mode =", ai_mode, "start_index =", start_index)
        window.show_view(
            GameView(return_to_menu=open_menu, ai_mode=ai_mode, start_index=start_index)
        )

    def open_levels():
        print("[main] open_levels called")
        window.show_view(LevelSelectView(open_game=open_game, return_to_menu=open_menu))

    print("[main] showing menu")
    window.show_view(make_menu())
    print("[main] running")
    arcade.run()


if __name__ == "__main__":
    main()

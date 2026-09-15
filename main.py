import arcade

import main_menu
from level_editor import LevelEditorView
from game import GameView

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

    def open_menu():
        print("[main] open_menu called")
        window.show_view(
            main_menu.MenuView(open_editor=open_editor, open_game=open_game)
        )

    def open_editor():
        print("[main] open_editor called")
        window.show_view(LevelEditorView(return_to_menu=open_menu))

    def open_game(ai_mode: bool = False):
        print("[main] open_game called, ai_mode =", ai_mode)
        window.show_view(GameView(return_to_menu=open_menu, ai_mode=ai_mode))

    print("[main] showing menu")
    window.show_view(
        main_menu.MenuView(open_editor=open_editor, open_game=open_game)
    )
    print("[main] running")
    arcade.run()


if __name__ == "__main__":
    main()
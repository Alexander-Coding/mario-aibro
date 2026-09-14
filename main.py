import arcade

import main_menu


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Mario"

arcade.SpriteList.DEFAULT_TEXTURE_FILTER = (
    arcade.gl.NEAREST,
    arcade.gl.NEAREST,
)

def main():
    window = arcade.Window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, resizable=True)
    menu_view = main_menu.MenuView()
    window.show_view(menu_view)
    arcade.run()


if __name__ == "__main__":
    main()
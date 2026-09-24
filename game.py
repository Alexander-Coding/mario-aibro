import arcade
from typing import Optional, Callable, Sequence

from core.levels import LevelInfo
from core.world import Inputs
from records import Records
from ui import hud
from ui.session import (
    ALL_DONE,
    DEAD,
    LEVEL_DONE,
    PLAYING,
    PlaySession,
)
from ui.text import TextPool


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Mario"

# Скорость плавного движения камеры
CAMERA_SPEED = 5.0

# Размер зоны, в которой камера не двигается
CAMERA_DEAD_ZONE_X = 400
CAMERA_DEAD_ZONE_Y = 300

# Цвета оверлея паузы
PAUSE_OVERLAY = (0, 0, 0, 160)
PAUSE_PANEL = (34, 38, 56)
PAUSE_PANEL_OUTLINE = (78, 86, 112)
PAUSE_BUTTON = (48, 54, 76)
PAUSE_BUTTON_HOVER = (86, 128, 255)
PAUSE_TEXT = (238, 240, 250)

# Клавиши: движение держим, прыжок — нажатие
KEYS_RIGHT = (arcade.key.RIGHT, arcade.key.D)
KEYS_LEFT = (arcade.key.LEFT, arcade.key.A)
KEYS_JUMP = (arcade.key.SPACE, arcade.key.UP, arcade.key.W)
KEYS_ENTER = (arcade.key.ENTER, arcade.key.RETURN, arcade.key.NUM_ENTER)


class PauseButton:

    def __init__(self, label: str, x: float, y: float, w: float, h: float, on_click):
        self.label = label
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.on_click = on_click

    def hit(self, mx: float, my: float) -> bool:
        return (
            self.x <= mx <= self.x + self.width
            and self.y <= my <= self.y + self.height
        )


class GameView(arcade.View):
    """Живая игра: рисует PlaySession и передаёт в него нажатия.

    Вся логика уровня — в core.world.GameWorld (та же, что на обучении
    нейросети), переходы между уровнями, итоги и рекорды — в
    ui.session.PlaySession. Здесь только клавиатура, камера, отрисовка и пауза.
    """

    def __init__(
        self,
        return_to_menu: Optional[Callable[[], None]] = None,
        ai_mode: bool = False,
        start_index: int = 0,
        levels: Optional[Sequence[LevelInfo]] = None,
        records: Optional[Records] = None,
        session: Optional[PlaySession] = None,
    ):
        super().__init__()

        self.ai_mode = ai_mode
        self.return_to_menu = return_to_menu

        self.background_color = arcade.color.Color(147, 187, 236)

        # Сеанс: уровни, мир, итоги. Модель ИИ грузим после первого кадра —
        # иначе окно на секунды замрёт на меню, пока импортируется torch
        self.session = session or PlaySession(
            ai_mode=ai_mode,
            start_index=start_index,
            levels=levels,
            records=records,
            defer_ai=True,
        )

        # Камеры
        self.camera: Optional[arcade.Camera2D] = None
        self.gui_camera: Optional[arcade.Camera2D] = None

        # Какой мир сейчас у камеры: сменился уровень — камера прыгает к игроку
        self._camera_serial = -1
        self._camera_floor = 0.0

        # Клавиатура: что держим и было ли нажатие прыжка с прошлого кадра
        self.keys_held: set[int] = set()
        self._jump_pressed = False

        # Надписи HUD и экранов
        self.texts = TextPool()
        self._frames_drawn = 0

        # ----------------------------------------------------
        # ПАУЗА
        # ----------------------------------------------------
        self.paused = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.pause_buttons: list[PauseButton] = []

        self.setup()
        self._build_pause_buttons()

    # ========================================================
    # СВОЙСТВА (для совместимости со старым кодом)
    # ========================================================

    @property
    def world(self):
        return self.session.world

    @property
    def player(self):
        # Мир пересобирается при перезапуске — игрока берём каждый раз заново
        return self.session.world.player

    @property
    def score(self) -> int:
        return self.session.world.score

    @property
    def game_over(self) -> bool:
        return self.session.state == DEAD

    # ========================================================
    # SETUP
    # ========================================================

    def setup(self) -> None:
        """Настройка камер и сброс паузы. Уровень держит PlaySession."""

        self.camera = arcade.Camera2D()
        self.gui_camera = arcade.Camera2D()

        width, height = self._size()
        self.gui_camera.position = (width / 2, height / 2)

        self._snap_camera()

        self.keys_held.clear()
        self._jump_pressed = False

        # Сбрасываем паузу
        self.paused = False

    def _size(self) -> tuple[float, float]:
        if self.window is not None:
            return float(self.window.width), float(self.window.height)
        return float(WINDOW_WIDTH), float(WINDOW_HEIGHT)

    # ========================================================
    # PAUSE BUTTONS
    # ========================================================

    def _build_pause_buttons(self):
        """Создаёт кнопки оверлея паузы."""

        width, height = self._size()

        cx = width / 2
        cy = height / 2

        bw, bh = 260, 56
        gap = 20

        # Кнопки расположены по центру панели ниже заголовка
        first_y = cy + 10
        second_y = first_y - bh - gap

        self.pause_buttons = [
            PauseButton(
                "Продолжить",
                cx - bw / 2,
                first_y - bh / 2,
                bw, bh,
                self.resume_game,
            ),
            PauseButton(
                "Выйти в меню",
                cx - bw / 2,
                second_y - bh / 2,
                bw, bh,
                self.exit_to_menu,
            ),
        ]

    def pause_game(self):
        self.paused = True

    def resume_game(self):
        self.paused = False

    def exit_to_menu(self):
        """Возврат в главное меню."""
        self.paused = False
        if self.return_to_menu is not None:
            self.return_to_menu()
        elif self.window is not None:
            self.window.close()

    # ========================================================
    # DRAW
    # ========================================================

    def on_draw(self) -> None:
        """Отрисовка"""

        self.clear()

        session = self.session
        world = session.world
        width, height = self._size()

        # =====================
        # ИГРОВОЙ МИР
        # =====================

        if self.camera:
            self.camera.use()

        world.walls.draw()
        world.decorations.draw()
        world.objects.draw()
        world.enemies.draw()

        # Вошёл в замок — игрока не видно
        player = world.player
        if player is not None and player.visible:
            player.playerSprite.draw()

        hud.draw_popups(self.texts, session.popups)

        # =====================
        # HUD
        # =====================

        if self.gui_camera:
            self.gui_camera.use()

        hud.draw_hud(self.texts, session, width, height)

        # Панель ИИ — пока идёт уровень; на экранах итогов она только мешает
        if session.ai_active:
            if session.state == PLAYING:
                hud.draw_ai_panel(self.texts, session, session.ai_overlay(), width, height)
        elif session.ai_error:
            hud.draw_ai_message(self.texts, session.ai_error, width, height)

        if session.state == PLAYING and session.flag_banner is not None:
            hud.draw_flag_banner(self.texts, session.flag_banner, width, height)

        # =====================
        # ЭКРАНЫ ИТОГОВ
        # =====================

        result = session.last_result

        if session.state == LEVEL_DONE and result is not None:
            hud.draw_level_result(self.texts, session, result, width, height)
        elif session.state == DEAD and result is not None:
            hud.draw_death(self.texts, session, result, width, height)
        elif session.state == ALL_DONE:
            hud.draw_all_done(self.texts, session, width, height)

        if session.ai_pending:
            hud.draw_loading(self.texts, "Загрузка ИИ…", width, height)

        # =====================
        # PAUSE OVERLAY
        # =====================
        if self.paused and session.state == PLAYING:
            self._draw_pause_overlay()

        self._frames_drawn += 1

    def _draw_pause_overlay(self):
        """Отрисовка оверлея паузы."""

        width, height = self._size()

        # Затемнение фона
        arcade.draw_lrbt_rectangle_filled(
            0, width, 0, height,
            PAUSE_OVERLAY,
        )

        # Панель
        panel_w, panel_h = 460, 260
        cx = width / 2
        cy = height / 2

        left = cx - panel_w / 2
        right = cx + panel_w / 2
        bottom = cy - panel_h / 2
        top = cy + panel_h / 2

        arcade.draw_lrbt_rectangle_filled(
            left + 4, right + 4, bottom - 6, top - 6,
            (0, 0, 0, 60),
        )
        arcade.draw_lrbt_rectangle_filled(
            left, right, bottom, top,
            PAUSE_PANEL,
        )
        arcade.draw_lrbt_rectangle_outline(
            left, right, bottom, top,
            PAUSE_PANEL_OUTLINE, 2,
        )

        # Заголовок
        self.texts.draw(
            "pause.title",
            "ПАУЗА",
            cx,
            top - 50,
            PAUSE_TEXT,
            28,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

        # Кнопки
        for index, btn in enumerate(self.pause_buttons):
            hover = btn.hit(self.mouse_x, self.mouse_y)
            color = PAUSE_BUTTON_HOVER if hover else PAUSE_BUTTON

            arcade.draw_lrbt_rectangle_filled(
                btn.x, btn.x + btn.width,
                btn.y, btn.y + btn.height,
                color,
            )
            arcade.draw_lrbt_rectangle_outline(
                btn.x, btn.x + btn.width,
                btn.y, btn.y + btn.height,
                PAUSE_PANEL_OUTLINE, 1,
            )

            self.texts.draw(
                f"pause.button{index}",
                btn.label,
                btn.x + btn.width / 2,
                btn.y + btn.height / 2,
                PAUSE_TEXT,
                18,
                anchor_x="center",
                anchor_y="center",
                bold=True,
            )

    # ========================================================
    # UPDATE
    # ========================================================

    def on_update(self, delta_time: float) -> None:
        """Обновление игры"""

        session = self.session

        # Модель ИИ грузим, когда на экране уже есть «Загрузка ИИ…»
        if session.ai_pending:
            if self._frames_drawn > 0:
                session.load_ai()
            return

        # Пауза — ничего не обновляем
        if self.paused:
            return

        # Мир крутится фиксированным шагом внутри сеанса: сколько шагов
        # сделать за этот кадр, решает аккумулятор, а не FPS окна
        session.update(delta_time, self._current_inputs())
        self._jump_pressed = False

        # =====================
        # КАМЕРА
        # =====================
        if session.world_serial != self._camera_serial:
            self._snap_camera()
        else:
            self.update_camera(delta_time)

        # Надписи исчезнувших «+100» больше не нужны
        if self._frames_drawn % 120 == 0:
            hud.forget_popups(self.texts, {popup.serial for popup in session.popups})

    def _current_inputs(self) -> Inputs:
        held = self.keys_held
        return Inputs(
            right=any(key in held for key in KEYS_RIGHT),
            left=any(key in held for key in KEYS_LEFT),
            jump=self._jump_pressed,
        )

    # ========================================================
    # CAMERA
    # ========================================================

    def _camera_bounds(self) -> tuple[float, float, float]:
        """Границы центра камеры: не левее начала уровня, не правее конца, не ниже земли."""

        width, height = self._size()
        geometry = self.session.world.geometry

        min_x = geometry.left + width / 2
        max_x = max(min_x, geometry.right - width / 2)
        min_y = self._camera_floor + height / 2

        return min_x, max_x, min_y

    def _snap_camera(self) -> None:
        """Новый уровень: камера сразу у игрока, без проезда через весь уровень."""

        if not self.camera:
            return

        world = self.session.world
        self._camera_serial = self.session.world_serial
        self._camera_floor = min((wall.bottom for wall in world.walls), default=0.0)

        min_x, max_x, min_y = self._camera_bounds()
        player = world.player

        x = min(max(player.center_x, min_x), max_x)
        y = max(player.center_y, min_y)

        self.camera.position = (x, y)

    def update_camera(self, delta_time: float) -> None:
        """Плавная камера с dead zone"""

        player = self.session.world.player

        if not player or not self.camera:
            return

        width, height = self._size()

        camera_x = self.camera.position.x
        camera_y = self.camera.position.y

        left = camera_x - width / 2
        right = camera_x + width / 2
        bottom = camera_y - height / 2
        top = camera_y + height / 2

        # На маленьком окне dead zone не должна схлопнуться в ноль
        dead_zone_x = min(CAMERA_DEAD_ZONE_X, width / 2 - 60)
        dead_zone_y = min(CAMERA_DEAD_ZONE_Y, height / 2 - 60)

        dead_left = left + dead_zone_x
        dead_right = right - dead_zone_x
        dead_bottom = bottom + dead_zone_y
        dead_top = top - dead_zone_y

        target_x = camera_x
        target_y = camera_y

        # Игрок в замке невидим — камера стоит
        if player.visible:
            if player.center_x < dead_left:
                target_x += player.center_x - dead_left
            elif player.center_x > dead_right:
                target_x += player.center_x - dead_right

            if player.center_y < dead_bottom:
                target_y += player.center_y - dead_bottom
            elif player.center_y > dead_top:
                target_y += player.center_y - dead_top

        min_x, max_x, min_y = self._camera_bounds()
        target_x = min(max(target_x, min_x), max_x)
        target_y = max(target_y, min_y)

        speed = min(CAMERA_SPEED * delta_time, 1.0)

        new_x = camera_x + (target_x - camera_x) * speed
        new_y = camera_y + (target_y - camera_y) * speed

        self.camera.position = (new_x, new_y)

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)

        if self.camera:
            self.camera.match_window()
        if self.gui_camera:
            self.gui_camera.match_window()
            self.gui_camera.position = (width / 2, height / 2)

        self._build_pause_buttons()

    # ========================================================
    # MOUSE
    # ========================================================

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x = x
        self.mouse_y = y

    def on_mouse_press(self, x, y, button, modifiers):
        if self.paused:
            if button == arcade.MOUSE_BUTTON_LEFT:
                for btn in self.pause_buttons:
                    if btn.hit(x, y):
                        btn.on_click()
                        return

    # ========================================================
    # KEYBOARD
    # ========================================================

    def on_key_press(self, symbol: int, modifiers: int) -> None:

        # Движение держим и между экранами: зажатая «вправо» после
        # перехода на следующий уровень продолжает работать
        self.keys_held.add(symbol)

        # =====================
        # ПАУЗА
        # =====================
        if self.paused:
            if symbol == arcade.key.ESCAPE:
                self.resume_game()
            elif symbol in KEYS_ENTER:
                self.resume_game()
            return

        session = self.session
        state = session.state

        # =====================
        # ИГРА
        # =====================
        if state == PLAYING:
            if symbol == arcade.key.ESCAPE:
                self.pause_game()
            elif symbol in KEYS_JUMP:
                self._jump_pressed = True
            elif symbol == arcade.key.R and not session.ai_pending:
                session.restart_level()
            return

        # =====================
        # ИТОГИ УРОВНЯ / СМЕРТЬ
        # =====================
        if state in (LEVEL_DONE, DEAD):
            if symbol in KEYS_ENTER:
                session.next_level()
            elif symbol == arcade.key.R:
                session.restart_level()
            elif symbol == arcade.key.ESCAPE:
                self.exit_to_menu()
            return

        # =====================
        # ВСЕ УРОВНИ ПОЗАДИ
        # =====================
        if state == ALL_DONE:
            if symbol == arcade.key.R:
                session.restart_run()
            elif symbol in KEYS_ENTER or symbol == arcade.key.ESCAPE:
                self.exit_to_menu()

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self.keys_held.discard(symbol)

    def on_hide_view(self) -> None:
        # Окно потеряло вид — отпущенные вне его клавиши не должны «залипнуть»
        self.keys_held.clear()
        self._jump_pressed = False

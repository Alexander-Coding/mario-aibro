"""Экран «УРОВНИ»: уровни из манифеста с рекордами человека и ИИ.

По каждому уровню видно лучший счёт человека, лучший счёт нейросети и
кто из них набрал больше — побеждает не тот, кто прошёл, а тот, у кого
больше очков. Уровень можно сыграть самому («ИГРАТЬ») или посмотреть,
как его проходит ИИ («ИИ»); дальше игра идёт по манифесту до конца.

Клавиши: ↑/↓ — выбор, Enter — играть, I — смотреть ИИ, Esc — в меню.
Фон и кнопки — как в главном меню автора (main_menu.py).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import arcade

from core import levels as core_levels
from core.levels import LevelInfo
from main_menu import (
    BUTTON_BORDER,
    BUTTON_BOTTOM,
    BUTTON_HOVER_BOTTOM,
    BUTTON_HOVER_TOP,
    BUTTON_TOP,
    Button,
    TEXT_DARK,
    TITLE_RED,
    TITLE_RED_DARK,
    draw_clouds,
    draw_ground,
    draw_sky,
)
from records import AI, HUMAN, Records, compare_scores, format_score
from ui.hud import GOLD, MUTED, PANEL, PANEL_OUTLINE, TEXT, draw_panel
from ui.text import TextPool

# ============================================================
# ЦВЕТА И РАЗМЕРЫ
# ============================================================

ROW_HEIGHT = 64
ROW_SELECTED = (86, 128, 255, 70)
ROW_HOVER = (255, 255, 255, 18)

AI_BUTTON_TOP = (140, 176, 255)
AI_BUTTON_BOTTOM = (86, 128, 255)
AI_BUTTON_HOVER_TOP = (170, 198, 255)
AI_BUTTON_HOVER_BOTTOM = (112, 150, 255)
AI_BUTTON_BORDER = (30, 50, 120)

HUMAN_COLOR = GOLD
AI_COLOR = (140, 176, 255)

DESCRIPTION_LIMIT = 58


def _small_button(texts: TextPool, key: str, label: str, left: float, bottom: float,
                  width: float, height: float, hover: bool, ai: bool = False) -> None:
    """Маленькая кнопка в стиле кнопок меню: тень, «градиент», рамка, блик."""

    right = left + width
    top = bottom + height
    mid = (bottom + top) / 2

    if ai:
        top_color = AI_BUTTON_HOVER_TOP if hover else AI_BUTTON_TOP
        bottom_color = AI_BUTTON_HOVER_BOTTOM if hover else AI_BUTTON_BOTTOM
        border, text_color = AI_BUTTON_BORDER, (255, 255, 255)
    else:
        top_color = BUTTON_HOVER_TOP if hover else BUTTON_TOP
        bottom_color = BUTTON_HOVER_BOTTOM if hover else BUTTON_BOTTOM
        border, text_color = BUTTON_BORDER, TEXT_DARK

    arcade.draw_lrbt_rectangle_filled(left + 3, right + 3, bottom - 4, top - 4, (0, 0, 0, 70))
    arcade.draw_lrbt_rectangle_filled(left, right, mid, top, top_color)
    arcade.draw_lrbt_rectangle_filled(left, right, bottom, mid, bottom_color)
    arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, border, 2)
    arcade.draw_lrbt_rectangle_filled(left + 4, right - 4, top - 6, top - 3, (255, 255, 255, 90))

    texts.draw(key, label, (left + right) / 2, mid - 1, text_color, 14, bold=True,
               anchor_x="center", anchor_y="center")


class LevelSelectView(arcade.View):

    def __init__(
        self,
        open_game: Optional[Callable[..., None]] = None,
        return_to_menu: Optional[Callable[[], None]] = None,
        records: Optional[Records] = None,
        levels: Optional[Sequence[LevelInfo]] = None,
    ):
        super().__init__()

        self.open_game = open_game
        self.return_to_menu = return_to_menu

        self.levels: list[LevelInfo] = list(levels if levels is not None else core_levels.load_levels())
        self.records = records if records is not None else Records()

        self.selected = 0
        self.scroll = 0

        self.mouse_x = 0
        self.mouse_y = 0

        self.texts = TextPool()

        # Где на экране кнопки строк: (left, bottom, right, top, действие, индекс)
        self._hit_areas: list[tuple[float, float, float, float, str, int]] = []
        self._row_areas: list[tuple[float, float, float, float, int]] = []
        self._visible_rows = 1

        self.back_button: Optional[Button] = None

    def on_show_view(self) -> None:
        # Рекорды могли измениться, пока шла игра
        self.records.reload()

    # ========================================================
    # DRAW
    # ========================================================

    def on_draw(self) -> None:
        self.clear()

        width = self.window.width
        height = self.window.height

        draw_sky(width, height)
        draw_ground(width, height)
        draw_clouds(width, height)

        cx = width / 2

        # Заголовок в стиле «MARIO» из меню
        self.texts.draw("title.shadow", "УРОВНИ", cx + 3, height - 72, TITLE_RED_DARK, 52,
                        bold=True, anchor_x="center", anchor_y="center")
        self.texts.draw("title", "УРОВНИ", cx, height - 69, TITLE_RED, 52,
                        bold=True, anchor_x="center", anchor_y="center")
        self.texts.draw_shadowed("subtitle", "Побеждает тот, кто набрал больше очков", cx, height - 122,
                                 (255, 255, 255), 16, bold=True, anchor_x="center", anchor_y="center")

        self._draw_table(width, height)

        # Кнопка «НАЗАД» — как кнопки меню
        self.back_button = Button("НАЗАД", "back", cx, 128, min(260, width * 0.3), 50)
        self.back_button.hover = self.back_button.hit(self.mouse_x, self.mouse_y)
        self.back_button.draw()

    def _draw_table(self, width: float, height: float) -> None:
        panel_w = min(1040, width - 40)
        left = width / 2 - panel_w / 2
        right = width / 2 + panel_w / 2
        top = height - 148
        bottom = 168

        draw_panel(left, right, bottom, top, fill=PANEL, outline=PANEL_OUTLINE)

        header_h = 40
        footer_h = 44
        rows_area = top - bottom - header_h - footer_h
        self._visible_rows = max(1, int(rows_area // ROW_HEIGHT))
        self._keep_selection_visible()

        pad = 20
        inner_l = left + pad
        inner_r = right - pad
        inner_w = inner_r - inner_l

        # Колонки
        col_human = inner_l + inner_w * 0.44
        col_ai = inner_l + inner_w * 0.58
        col_leader = inner_l + inner_w * 0.66
        buttons_right = inner_r

        y = top - 27
        self.texts.draw("head.level", "Уровень", inner_l, y, MUTED, 12, bold=True)
        self.texts.draw("head.human", "Человек", col_human, y, MUTED, 12, bold=True, anchor_x="right")
        self.texts.draw("head.ai", "ИИ", col_ai, y, MUTED, 12, bold=True, anchor_x="right")
        self.texts.draw("head.leader", "Больше набрал", col_leader, y, MUTED, 12, bold=True)
        arcade.draw_lrbt_rectangle_filled(inner_l, inner_r, top - header_h, top - header_h + 2, PANEL_OUTLINE)

        self._hit_areas = []
        self._row_areas = []

        if not self.levels:
            self.texts.draw("empty", "В манифесте maps/levels.json нет уровней", width / 2,
                            (top + bottom) / 2, MUTED, 16, anchor_x="center")
            return

        row_top = top - header_h - 4
        shown = self.levels[self.scroll:self.scroll + self._visible_rows]

        for slot, level in enumerate(shown):
            index = self.scroll + slot
            r_top = row_top - slot * ROW_HEIGHT
            r_bottom = r_top - ROW_HEIGHT + 4
            mid = (r_top + r_bottom) / 2

            hover = inner_l - 8 <= self.mouse_x <= inner_r + 8 and r_bottom <= self.mouse_y <= r_top
            if index == self.selected:
                arcade.draw_lrbt_rectangle_filled(inner_l - 8, inner_r + 8, r_bottom, r_top, ROW_SELECTED)
            elif hover:
                arcade.draw_lrbt_rectangle_filled(inner_l - 8, inner_r + 8, r_bottom, r_top, ROW_HOVER)
            self._row_areas.append((inner_l - 8, r_bottom, inner_r + 8, r_top, index))

            # Название и описание
            self.texts.draw(f"row{slot}.name", f"{index + 1}. {level.name}", inner_l, mid + 5,
                            TEXT, 17, bold=True)
            description = level.description or f"Время: {int(level.time_limit)} с"
            if len(description) > DESCRIPTION_LIMIT:
                description = description[:DESCRIPTION_LIMIT - 1].rstrip() + "…"
            self.texts.draw(f"row{slot}.desc", description, inner_l, mid - 16, MUTED, 11)

            # Рекорды
            human = self.records.get(level.id, HUMAN)
            ai = self.records.get(level.id, AI)
            self._draw_record(f"row{slot}.human", human, col_human, mid, HUMAN_COLOR)
            self._draw_record(f"row{slot}.ai", ai, col_ai, mid, AI_COLOR)

            # Кто набрал больше
            leader = compare_scores(
                human["score"] if human else None,
                ai["score"] if ai else None,
            )
            note = ""
            if leader is None:
                leader_text, leader_color = "—", MUTED
            elif leader == "tie":
                leader_text, leader_color = "ничья", TEXT
            elif human is None or ai is None:
                # Соперник ещё не играл — лидер «по умолчанию», пояснение строкой ниже
                leader_text = "Человек" if leader == HUMAN else "ИИ"
                leader_color = HUMAN_COLOR if leader == HUMAN else AI_COLOR
                note = "ИИ ещё не играл" if leader == HUMAN else "человек ещё не играл"
            else:
                gap = abs(human["score"] - ai["score"])
                leader_text = f"{'Человек' if leader == HUMAN else 'ИИ'}  +{format_score(gap)}"
                leader_color = HUMAN_COLOR if leader == HUMAN else AI_COLOR
            self.texts.draw(f"row{slot}.leader", leader_text, col_leader, mid + (0 if note else -6),
                            leader_color, 14, bold=leader not in (None, "tie"))
            self.texts.draw(f"row{slot}.leader.note", note, col_leader, mid - 17, MUTED, 10)

            # Кнопки строки
            bh = 36
            bw_play, bw_ai = 104, 56
            b_bottom = mid - bh / 2
            ai_left = buttons_right - bw_ai
            play_left = ai_left - 10 - bw_play

            for action, b_left, b_w in (("play", play_left, bw_play), ("ai", ai_left, bw_ai)):
                b_hover = b_left <= self.mouse_x <= b_left + b_w and b_bottom <= self.mouse_y <= b_bottom + bh
                _small_button(
                    self.texts, f"row{slot}.{action}",
                    "ИГРАТЬ" if action == "play" else "ИИ",
                    b_left, b_bottom, b_w, bh, b_hover, ai=action == "ai",
                )
                self._hit_areas.append((b_left, b_bottom, b_left + b_w, b_bottom + bh, action, index))

        # Прокрутка: сколько уровней за краем
        if len(self.levels) > self._visible_rows:
            more = f"{self.scroll + 1}–{self.scroll + len(shown)} из {len(self.levels)} · колесо мыши — прокрутка"
            self.texts.draw("scroll", more, right - pad, bottom + footer_h + 4, MUTED, 11, anchor_x="right")

        # Итог: сумма рекордов по всем уровням
        human_sum = sum(self.records.best_score(level.id, HUMAN) or 0 for level in self.levels)
        ai_sum = sum(self.records.best_score(level.id, AI) or 0 for level in self.levels)
        leader = compare_scores(human_sum or None, ai_sum or None)
        if leader == HUMAN:
            verdict, color = "впереди человек", HUMAN_COLOR
        elif leader == AI:
            verdict, color = "впереди ИИ", AI_COLOR
        elif leader == "tie":
            verdict, color = "ничья", TEXT
        else:
            verdict, color = "рекордов пока нет", MUTED

        total = (
            f"Сумма рекордов: человек {format_score(human_sum)} · ИИ {format_score(ai_sum)} — {verdict}"
        )
        # Итог и подсказка — в две строки: в одну они наезжают друг на друга
        self.texts.draw("total", total, inner_l, bottom + 24, color, 14, bold=True)
        self.texts.draw("keys", "↑↓ — выбор   Enter — играть   I — ИИ   Esc — меню", inner_r, bottom + 6,
                        MUTED, 11, anchor_x="right")

    def _draw_record(self, key: str, record: Optional[dict], x: float, mid: float, color) -> None:
        if record is None:
            self.texts.draw(key, "—", x, mid - 6, MUTED, 16, bold=True, anchor_x="right")
            self.texts.draw(key + ".state", "", x, mid - 22, MUTED, 10, anchor_x="right")
            return

        self.texts.draw(key, format_score(record["score"]), x, mid, color, 17, bold=True, anchor_x="right")
        self.texts.draw(key + ".state", "пройден" if record["won"] else "не дошёл", x, mid - 17,
                        MUTED, 10, anchor_x="right")

    def _keep_selection_visible(self) -> None:
        if not self.levels:
            self.selected = 0
            self.scroll = 0
            return

        self.selected = max(0, min(self.selected, len(self.levels) - 1))
        if self.selected < self.scroll:
            self.scroll = self.selected
        elif self.selected >= self.scroll + self._visible_rows:
            self.scroll = self.selected - self._visible_rows + 1
        self.scroll = max(0, min(self.scroll, max(0, len(self.levels) - self._visible_rows)))

    # ========================================================
    # ACTIONS
    # ========================================================

    def start(self, index: int, ai_mode: bool) -> None:
        """Играть с уровня index: дальше по манифесту до конца."""

        if not self.levels:
            return

        index = max(0, min(index, len(self.levels) - 1))

        if self.open_game is not None:
            self.open_game(ai_mode=ai_mode, start_index=index)
            return

        # Без колбэка (экран открыт сам по себе) — показываем игру напрямую
        from game import GameView

        self.window.show_view(
            GameView(return_to_menu=self.back, ai_mode=ai_mode, start_index=index, levels=self.levels)
        )

    def back(self) -> None:
        if self.return_to_menu is not None:
            self.return_to_menu()
        else:
            self.window.close()

    # ========================================================
    # INPUT
    # ========================================================

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x = x
        self.mouse_y = y

    def on_mouse_press(self, x, y, button, modifiers):
        if button != arcade.MOUSE_BUTTON_LEFT:
            return

        if self.back_button is not None and self.back_button.hit(x, y):
            self.back()
            return

        for left, bottom, right, top, action, index in self._hit_areas:
            if left <= x <= right and bottom <= y <= top:
                self.selected = index
                self.start(index, ai_mode=action == "ai")
                return

        for left, bottom, right, top, index in self._row_areas:
            if left <= x <= right and bottom <= y <= top:
                self.selected = index
                return

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if len(self.levels) <= self._visible_rows:
            return
        step = -1 if scroll_y > 0 else 1
        self.scroll = max(0, min(self.scroll + step, len(self.levels) - self._visible_rows))
        # Выбор не уезжает за край видимого
        self.selected = max(self.scroll, min(self.selected, self.scroll + self._visible_rows - 1))

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol in (arcade.key.UP, arcade.key.W):
            self.selected = max(0, self.selected - 1)
        elif symbol in (arcade.key.DOWN, arcade.key.S):
            self.selected = min(len(self.levels) - 1, self.selected + 1)
        elif symbol in (arcade.key.ENTER, arcade.key.RETURN, arcade.key.NUM_ENTER, arcade.key.SPACE):
            self.start(self.selected, ai_mode=False)
        elif symbol in (arcade.key.I, arcade.key.TAB):
            self.start(self.selected, ai_mode=True)
        elif symbol in (arcade.key.ESCAPE, arcade.key.BACKSPACE):
            self.back()
        else:
            return

        self._keep_selection_visible()

"""Отрисовка поверх мира: HUD, панель ИИ, всплывающие очки, экраны итогов.

Все функции только рисуют: состояние берут из PlaySession (ui/session.py)
и ничего в нём не меняют. Надписи идут через TextPool — у каждой свой ключ.
Экранные функции рисуют в координатах окна (gui-камера), draw_popups —
в мировых (камера мира).
"""

from __future__ import annotations

from typing import Optional

import arcade

from records import AI, HUMAN, PLAYER_NAMES, compare_scores, format_score
from ui.ai_control import AI_KEYBOARD_TEXT
from ui.session import (
    ALL_DONE,
    DEAD,
    DEATH_CAUSES,
    LEVEL_DONE,
    FlagBanner,
    LevelResult,
    PlaySession,
    Popup,
)
from ui.text import TextPool

# ============================================================
# ЦВЕТА (в тон оверлею паузы из game.py)
# ============================================================

PANEL = (34, 38, 56, 250)
PANEL_OUTLINE = (78, 86, 112)
PANEL_SHADOW = (0, 0, 0, 60)
ACCENT = (86, 128, 255)
TEXT = (238, 240, 250)
MUTED = (160, 168, 196)
GOLD = (255, 214, 64)
RED = (236, 76, 76)
GREEN = (110, 220, 120)
DIM = (0, 0, 0, 120)

HUD_BAR = (0, 0, 0, 105)
HUD_BAR_HEIGHT = 60

AI_BADGE = (86, 128, 255)
ERROR_PANEL = (120, 32, 36, 230)
ERROR_OUTLINE = (220, 110, 110)

BAR_BACK = (24, 26, 40)

# Длинное имя модели обрезается, чтобы не вылезти из панели ИИ
MODEL_NAME_LIMIT = 30

# Строка сообщения об ошибке ИИ длиннее — обрезается
MESSAGE_LINE_LIMIT = 96

# Надпись над флагом по порогам очков (rules.FLAG_TIERS)
FLAG_TITLES = {
    5000: "ИДЕАЛЬНО!",
    2000: "ОТЛИЧНО!",
    800: "ХОРОШО!",
    400: "НЕПЛОХО",
    100: "ЗАЦЕПИЛСЯ",
}

# Расшифровка причин смерти для таблицы (коротко)
DEATH_SHORT = {
    "pit": "пропасть",
    "enemy": "враг",
    "time": "время",
}


# ============================================================
# ОБЩЕЕ
# ============================================================


def draw_panel(left: float, right: float, bottom: float, top: float,
               fill=PANEL, outline=PANEL_OUTLINE) -> None:
    """Панель как у паузы: тень, заливка, рамка."""

    arcade.draw_lrbt_rectangle_filled(left + 4, right + 4, bottom - 6, top - 6, PANEL_SHADOW)
    arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, fill)
    arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, outline, 2)


def draw_bar(left: float, right: float, center_y: float, height: float,
             value: float, color) -> None:
    """Полоска 0..1: тёмная подложка и заливка."""

    value = max(0.0, min(1.0, float(value)))
    bottom = center_y - height / 2
    top = center_y + height / 2

    arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, BAR_BACK)
    if value > 0:
        arcade.draw_lrbt_rectangle_filled(left, left + (right - left) * value, bottom, top, color)


def _lerp_color(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def danger_color(value: float):
    """Зелёный → жёлтый → красный."""

    if value < 0.5:
        return _lerp_color(GREEN, GOLD, value * 2)
    return _lerp_color(GOLD, RED, (value - 0.5) * 2)


def _with_alpha(color, alpha: float):
    return (*color[:3], int(max(0, min(255, alpha))))


def _appear(state_time: float, delay: float, duration: float = 0.25) -> float:
    """0..1: плавное появление элемента экрана итогов с задержкой."""

    if state_time <= delay:
        return 0.0
    return min((state_time - delay) / duration, 1.0)


def _who_title(who: str) -> str:
    return PLAYER_NAMES.get(who, who)


def _who_lower(who: str) -> str:
    """«человек» / «ИИ» внутри фразы (аббревиатура остаётся заглавной)."""
    return "человек" if who == HUMAN else _who_title(who)


# ============================================================
# HUD
# ============================================================


def draw_hud(texts: TextPool, session: PlaySession, width: float, height: float) -> None:
    """Верхняя полоса: очки, жизни, уровень, рекорд, время, значок «ИИ»."""

    world = session.world
    top = height
    arcade.draw_lrbt_rectangle_filled(0, width, top - HUD_BAR_HEIGHT, top, HUD_BAR)

    label_y = top - 20
    value_y = top - 47

    x = 20

    # Значок «ИИ»: играет нейросеть
    if session.ai_active:
        arcade.draw_lrbt_rectangle_filled(x, x + 46, top - 46, top - 14, AI_BADGE)
        arcade.draw_lrbt_rectangle_outline(x, x + 46, top - 46, top - 14, (190, 206, 255), 2)
        texts.draw("hud.ai", "ИИ", x + 23, top - 30, TEXT, 15, bold=True,
                   anchor_x="center", anchor_y="center")
        x += 64

    # Очки
    texts.draw("hud.score.label", "ОЧКИ", x, label_y, MUTED, 11, bold=True)
    texts.draw_shadowed("hud.score", f"{world.score:06d}", x, value_y, TEXT, 21, bold=True)
    x += 150

    # Жизни: сердечки, потерянные — серые
    health = max(0, int(world.player.health)) if world.player else 0
    lost = max(0, 3 - health)
    texts.draw("hud.lives.label", "ЖИЗНИ", x, label_y, MUTED, 11, bold=True)
    hearts = texts.draw_shadowed("hud.lives", "♥ " * health if health else " ", x, value_y, RED, 20, bold=True)
    if lost:
        texts.draw("hud.lives.lost", "♥ " * lost, x + hearts.content_width, value_y, (90, 96, 120), 20, bold=True)

    # Уровень — по центру
    cx = width / 2
    texts.draw(
        "hud.level.label",
        f"УРОВЕНЬ {session.level_number} ИЗ {session.level_count}",
        cx, label_y, MUTED, 11, bold=True, anchor_x="center",
    )
    texts.draw_shadowed("hud.level", session.level.name, cx, value_y, TEXT, 19, bold=True, anchor_x="center")

    # Рекорд уровня: свой и соперника
    records = session.record_for()
    own = records.get(session.who)
    other_who = HUMAN if session.who == AI else AI
    other = records.get(other_who)

    rx = width - 360
    texts.draw("hud.record.label", "РЕКОРД", rx, label_y, MUTED, 11, bold=True)
    own_text = format_score(own["score"] if own else None)
    record = texts.draw_shadowed("hud.record", own_text, rx, value_y, GOLD if own else MUTED, 19, bold=True)
    texts.draw(
        "hud.record.other",
        f"{_who_title(other_who)}: {format_score(other['score'] if other else None)}",
        rx + record.content_width + 10, value_y + 2, MUTED, 12, bold=True,
    )

    # Время: целые секунды, последние 30 — красным
    seconds = int(world.time_left)
    time_color = RED if seconds < 30 and not world.done else TEXT
    tx = width - 110
    texts.draw("hud.time.label", "ВРЕМЯ", tx, label_y, MUTED, 11, bold=True)
    texts.draw_shadowed("hud.time", f"{seconds:03d}", tx, value_y, time_color, 21, bold=True)


# ============================================================
# ПАНЕЛЬ ИИ
# ============================================================


def draw_ai_panel(texts: TextPool, session: PlaySession, info: Optional[dict],
                  width: float, height: float) -> None:
    """Справа сверху: модель, текущее действие, топ-3 вероятностей, опасность, ожидаемые очки."""

    info = info or {}

    model = str(info.get("model") or session.ai_model_name() or "ИИ")
    action = info.get("action")
    probs = info.get("probs") or {}
    danger = info.get("danger")
    expected = info.get("expected")
    confidence = info.get("confidence")

    top_probs = []
    if isinstance(probs, dict):
        try:
            top_probs = sorted(
                ((str(name), float(p)) for name, p in probs.items()),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        except (TypeError, ValueError):
            top_probs = []

    # Ширина под самую длинную подпись действия («вправо+прыжок» жирным)
    panel_w = 370
    right = width - 16
    left = right - panel_w
    top = height - HUD_BAR_HEIGHT - 12

    # Высота панели по тому, что есть
    rows = 2 + len(top_probs)
    if confidence is not None:
        rows += 1
    if danger is not None:
        rows += 1
    if expected is not None:
        rows += 1
    if not info:
        rows += 1
    bottom = top - 20 - rows * 24

    draw_panel(left, right, bottom, top)

    pad = 14
    y = top - 24

    if len(model) > MODEL_NAME_LIMIT:
        model = model[:MODEL_NAME_LIMIT - 1].rstrip() + "…"
    texts.draw("ai.title", f"ИИ · {model}", left + pad, y, TEXT, 14, bold=True)

    y -= 26
    if not info:
        texts.draw("ai.wait", "ждёт первого решения…", left + pad, y, MUTED, 12)
        return

    texts.draw("ai.action.label", "Действие", left + pad, y, MUTED, 12, bold=True)
    texts.draw("ai.action", str(action or "—"), left + 104, y, GOLD, 14, bold=True)

    # Топ-3 вероятностей полосками
    bar_left = left + 170
    bar_right = right - 56
    for i, (name, p) in enumerate(top_probs):
        y -= 24
        color = ACCENT if i == 0 else (70, 88, 150)
        texts.draw(f"ai.prob.name{i}", name, left + pad, y, TEXT if i == 0 else MUTED, 12, bold=i == 0)
        draw_bar(bar_left, bar_right, y + 5, 10, p, color)
        texts.draw(f"ai.prob.value{i}", f"{int(round(p * 100))}%", right - pad, y, TEXT, 12,
                   bold=i == 0, anchor_x="right")

    if confidence is not None:
        y -= 24
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            value = 0.0
        texts.draw("ai.conf.label", "Уверенность", left + pad, y, MUTED, 12, bold=True)
        draw_bar(bar_left, bar_right, y + 5, 10, value, (150, 160, 200))
        texts.draw("ai.conf", f"{int(round(value * 100))}%", right - pad, y, TEXT, 12,
                   anchor_x="right")

    if danger is not None:
        y -= 24
        try:
            value = float(danger)
        except (TypeError, ValueError):
            value = 0.0
        texts.draw("ai.danger.label", "Опасность", left + pad, y, MUTED, 12, bold=True)
        draw_bar(bar_left, bar_right, y + 5, 10, value, danger_color(value))
        texts.draw("ai.danger", f"{int(round(value * 100))}%", right - pad, y,
                   danger_color(value), 12, bold=True, anchor_x="right")

    if expected is not None:
        y -= 24
        try:
            expected_text = format_score(int(round(float(expected))))
        except (TypeError, ValueError):
            expected_text = str(expected)
        texts.draw("ai.expected.label", "Ожидаемые очки", left + pad, y, MUTED, 12, bold=True)
        texts.draw("ai.expected", expected_text, right - pad, y, GREEN, 13, bold=True, anchor_x="right")


def draw_ai_message(texts: TextPool, message: str, width: float, height: float) -> None:
    """Модель не найдена или сломалась: подсказка под HUD по центру."""

    lines = [
        line if len(line) <= MESSAGE_LINE_LIMIT else line[:MESSAGE_LINE_LIMIT - 1] + "…"
        for line in message.split("\n")
        if line
    ]
    if AI_KEYBOARD_TEXT not in message:
        lines.append(AI_KEYBOARD_TEXT)

    panel_w = min(760, width - 40)
    left = width / 2 - panel_w / 2
    right = width / 2 + panel_w / 2
    top = height - HUD_BAR_HEIGHT - 12
    bottom = top - 18 - 24 * len(lines)

    draw_panel(left, right, bottom, top, fill=ERROR_PANEL, outline=ERROR_OUTLINE)

    y = top - 26
    for i, line in enumerate(lines):
        is_first = i == 0
        texts.draw(
            f"ai.msg{i}", line, width / 2, y,
            TEXT if is_first else (255, 214, 214), 15 if is_first else 12,
            bold=is_first, anchor_x="center",
        )
        y -= 24


def draw_loading(texts: TextPool, text: str, width: float, height: float) -> None:
    arcade.draw_lrbt_rectangle_filled(0, width, 0, height, DIM)
    draw_panel(width / 2 - 220, width / 2 + 220, height / 2 - 44, height / 2 + 44)
    texts.draw("loading", text, width / 2, height / 2, TEXT, 20, bold=True,
               anchor_x="center", anchor_y="center")


# ============================================================
# ВСПЛЫВАЮЩИЕ ОЧКИ (мировые координаты)
# ============================================================


def draw_popups(texts: TextPool, popups: list[Popup]) -> None:
    """«+100» поднимается и тает за время жизни."""

    for popup in popups:
        t = popup.progress
        rise = (48 if popup.big else 34) * t
        alpha = 255 if t < 0.6 else 255 * (1 - (t - 0.6) / 0.4)
        size = 22 if popup.big else 15

        texts.draw_shadowed(
            f"popup{popup.serial}",
            popup.text,
            popup.x,
            popup.y + rise,
            _with_alpha(popup.color, alpha),
            size,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )


def forget_popups(texts: TextPool, alive_serials: set[int]) -> None:
    """Надписи исчезнувших всплывашек больше не нужны — выкидываем из кэша."""

    texts.forget(
        lambda key: key.startswith("popup")
        and int(key[5:].split("#")[0]) not in alive_serials
    )


# ============================================================
# ФЛАГ
# ============================================================


def draw_flag_banner(texts: TextPool, banner: FlagBanner, width: float, height: float) -> None:
    """Крупно: «ОТЛИЧНО! Флаг на высоте 87% · +2000»."""

    t = banner.age
    appear = min(t / 0.2, 1.0)
    fade = 1.0 if t < 2.4 else max(0.0, 1 - (t - 2.4) / 0.6)
    alpha = 255 * appear * fade

    # Лёгкий «подскок» при появлении
    scale_shift = (1 - appear) * 20
    cy = height * 0.68 + scale_shift

    title = FLAG_TITLES.get(banner.points, "ФЛАГ!")
    texts.draw_shadowed("flag.title", title, width / 2, cy, _with_alpha(GOLD, alpha), 46,
                        bold=True, anchor_x="center", anchor_y="center", offset=3)
    texts.draw_shadowed(
        "flag.info",
        f"Флаг на высоте {banner.percent}%   +{format_score(banner.points)}",
        width / 2, cy - 50, _with_alpha(TEXT, alpha), 22,
        bold=True, anchor_x="center", anchor_y="center",
    )


# ============================================================
# ЭКРАН ИТОГОВ УРОВНЯ
# ============================================================


def _record_lines(result: LevelResult) -> list[tuple[str, tuple]]:
    """Строки сравнения с рекордом: свой прежний, соперник, лидер уровня."""

    who = result.who
    other = HUMAN if who == AI else AI

    before_own = result.record_before.get(who)
    after = result.record_after
    lines: list[tuple[str, tuple]] = []

    if result.new_record:
        if before_own is not None:
            lines.append((f"НОВЫЙ РЕКОРД!  Было {format_score(before_own['score'])}", GOLD))
        else:
            lines.append(("НОВЫЙ РЕКОРД!  Первый результат на уровне", GOLD))
    elif before_own is not None:
        gap = before_own["score"] - result.score
        tail = f" — не хватило {format_score(gap)}" if gap > 0 else ""
        lines.append((f"Рекорд ({_who_lower(who)}): {format_score(before_own['score'])}{tail}", MUTED))

    human_best = after.get(HUMAN)["score"] if after.get(HUMAN) else None
    ai_best = after.get(AI)["score"] if after.get(AI) else None
    leader = compare_scores(human_best, ai_best)

    other_best = after.get(other)
    if other_best is None:
        lines.append((f"{_who_title(other)} этот уровень ещё не проходил", MUTED))
    elif leader == "tie":
        lines.append((f"Ничья: человек и ИИ — по {format_score(human_best)}", TEXT))
    else:
        lines.append((
            f"Больше набрал {_who_title(leader)}: человек {format_score(human_best)} · ИИ {format_score(ai_best)}",
            GREEN if leader == who else TEXT,
        ))

    return lines


def _hint_lines(session: PlaySession) -> list[str]:
    advance_in = session.advance_in
    if advance_in is not None:
        what = "Итоги" if session.is_last_level else "Следующий уровень"
        return [f"{what} через {int(advance_in) + 1}…", "Esc — меню"]

    next_text = "итоги" if session.is_last_level else "следующий уровень"
    return [f"Enter — {next_text}    R — заново    Esc — меню"]


def draw_level_result(texts: TextPool, session: PlaySession, result: LevelResult,
                      width: float, height: float) -> None:
    """Уровень пройден: разбивка очков, итог, сравнение с рекордом."""

    t = session.state_time
    arcade.draw_lrbt_rectangle_filled(0, width, 0, height, _with_alpha(DIM, 120 * _appear(t, 0.0, 0.3)))

    panel_w, panel_h = 620, 500
    left = width / 2 - panel_w / 2
    right = width / 2 + panel_w / 2
    top = height / 2 + panel_h / 2
    bottom = height / 2 - panel_h / 2
    draw_panel(left, right, bottom, top)

    cx = width / 2
    who_text = "ИИ прошёл уровень!" if result.who == AI else "УРОВЕНЬ ПРОЙДЕН!"
    texts.draw_shadowed("res.title", who_text, cx, top - 46, GOLD, 30, bold=True, anchor_x="center")
    texts.draw("res.level", result.level_name, cx, top - 76, MUTED, 14, bold=True, anchor_x="center")

    b = result.breakdown
    fraction = b.get("flag_fraction")
    rows = [
        ("Блоки «?»", f"× {b['blocks']}", b["block_points"]),
        ("Монеты", f"× {b['coins']}", b["coin_points"]),
        ("Враги", f"× {b['enemies']}", b["enemy_points"]),
        ("Флаг", "—" if fraction is None else f"высота {int(round(fraction * 100))}%", b["flag_points"]),
        ("Бонус за время", f"{b['time_seconds']} с × 20", b["time_bonus"]),
    ]

    pad = 40
    y = top - 124
    for i, (name, detail, points) in enumerate(rows):
        a = _appear(t, 0.2 + i * 0.12)
        if a <= 0:
            y -= 34
            continue
        alpha = 255 * a
        texts.draw(f"res.row{i}.name", name, left + pad, y, _with_alpha(TEXT, alpha), 16, bold=True)
        texts.draw(f"res.row{i}.detail", detail, left + 290, y, _with_alpha(MUTED, alpha), 14)
        texts.draw(f"res.row{i}.points", format_score(points), right - pad, y,
                   _with_alpha(TEXT, alpha), 16, bold=True, anchor_x="right")
        y -= 34

    # Черта и итог (счётчик «набегает»)
    arcade.draw_lrbt_rectangle_filled(left + pad, right - pad, y + 18, y + 20, PANEL_OUTLINE)
    count = _appear(t, 0.8, 0.8)
    shown = int(result.score * count)
    texts.draw("res.total.label", "ИТОГО", left + pad, y - 16, TEXT, 22, bold=True)
    texts.draw_shadowed("res.total", format_score(shown), right - pad, y - 16, GOLD, 28,
                        bold=True, anchor_x="right")

    # Рекорды
    y -= 62
    if _appear(t, 1.4) > 0:
        for i, (line, color) in enumerate(_record_lines(result)):
            texts.draw(f"res.rec{i}", line, cx, y, color, 15 if i == 0 else 13,
                       bold=i == 0, anchor_x="center")
            y -= 26

    for i, line in enumerate(_hint_lines(session)):
        texts.draw(f"res.hint{i}", line, cx, bottom + 24 + 22 * (len(_hint_lines(session)) - 1 - i),
                   MUTED, 13, anchor_x="center")


# ============================================================
# ЭКРАН СМЕРТИ
# ============================================================


def draw_death(texts: TextPool, session: PlaySession, result: LevelResult,
               width: float, height: float) -> None:
    """Смерть: причина по-русски и набранные очки."""

    t = session.state_time
    arcade.draw_lrbt_rectangle_filled(0, width, 0, height, _with_alpha((40, 0, 0), 130 * _appear(t, 0.0, 0.4)))

    panel_w, panel_h = 580, 400
    left = width / 2 - panel_w / 2
    right = width / 2 + panel_w / 2
    top = height / 2 + panel_h / 2
    bottom = height / 2 - panel_h / 2
    draw_panel(left, right, bottom, top)

    cx = width / 2
    title = "ИИ ПРОИГРАЛ" if result.who == AI else "ВЫ ПРОИГРАЛИ"
    texts.draw_shadowed("dead.title", title, cx, top - 50, RED, 32, bold=True, anchor_x="center")
    texts.draw("dead.level", result.level_name, cx, top - 80, MUTED, 14, bold=True, anchor_x="center")

    cause = DEATH_CAUSES.get(result.death or "", "Игра окончена")
    texts.draw_shadowed("dead.cause", cause, cx, top - 132, TEXT, 26, bold=True, anchor_x="center")

    texts.draw("dead.score.label", "Набрано очков", cx, top - 180, MUTED, 14, bold=True, anchor_x="center")
    texts.draw_shadowed("dead.score", format_score(result.score), cx, top - 220, GOLD, 34,
                        bold=True, anchor_x="center")

    b = result.breakdown
    texts.draw(
        "dead.breakdown",
        f"блоки {format_score(b['block_points'])} · монеты {format_score(b['coin_points'])}"
        f" · враги {format_score(b['enemy_points'])}",
        cx, top - 252, MUTED, 13, anchor_x="center",
    )
    texts.draw("dead.note", "Флаг и бонус за время после смерти не начисляются",
               cx, top - 276, (130, 138, 166), 12, anchor_x="center")

    y = top - 312
    for i, (line, color) in enumerate(_record_lines(result)):
        texts.draw(f"dead.rec{i}", line, cx, y, color, 13, bold=i == 0, anchor_x="center")
        y -= 22

    hints = _hint_lines(session)
    for i, line in enumerate(hints):
        texts.draw(f"dead.hint{i}", line, cx, bottom + 22 + 22 * (len(hints) - 1 - i),
                   MUTED, 13, anchor_x="center")


# ============================================================
# ИТОГОВАЯ ТАБЛИЦА
# ============================================================


def draw_all_done(texts: TextPool, session: PlaySession, width: float, height: float) -> None:
    """Все уровни позади: таблица по уровням, сумма очков, сравнение с соперником."""

    results = session.run_results()
    who = session.who if not results else results[-1].who
    other = HUMAN if who == AI else AI

    arcade.draw_lrbt_rectangle_filled(0, width, 0, height, (0, 0, 0, 140))

    row_h = 34
    visible_rows = max(1, min(len(results), int((height - 320) // row_h)))
    panel_w = min(960, width - 40)
    panel_h = 250 + visible_rows * row_h
    left = width / 2 - panel_w / 2
    right = width / 2 + panel_w / 2
    top = height / 2 + panel_h / 2
    bottom = height / 2 - panel_h / 2
    draw_panel(left, right, bottom, top)

    cx = width / 2
    all_won = bool(results) and all(r.won for r in results)
    title = "ВСЕ УРОВНИ ПРОЙДЕНЫ!" if all_won else "ИТОГИ ЗАБЕГА"
    texts.draw_shadowed("all.title", title, cx, top - 46, GOLD, 30, bold=True, anchor_x="center")
    player = f"Играл: {_who_title(who)}"
    if who == AI and session.ai_model_name():
        player += f" ({session.ai_model_name()})"
    texts.draw("all.who", player, cx, top - 74, MUTED, 13, bold=True, anchor_x="center")

    # Колонки: доли ширины панели
    inner_l = left + 28
    inner_w = panel_w - 56
    cols = [
        ("Уровень", 0.00, "left"),
        ("Итог", 0.34, "left"),
        ("Флаг", 0.50, "right"),
        ("За время", 0.62, "right"),
        ("Очки", 0.76, "right"),
        ("Рекорд человека" if other == HUMAN else "Рекорд ИИ", 1.00, "right"),
    ]

    y = top - 116
    for i, (name, pos, align) in enumerate(cols):
        texts.draw(f"all.head{i}", name, inner_l + inner_w * pos, y, MUTED, 12, bold=True, anchor_x=align)
    arcade.draw_lrbt_rectangle_filled(inner_l, inner_l + inner_w, y - 10, y - 8, PANEL_OUTLINE)

    other_sum = 0
    other_known = 0
    for row, result in enumerate(results[:visible_rows]):
        y -= row_h
        fraction = result.breakdown.get("flag_fraction")
        other_record = result.record_after.get(other)
        if other_record is not None:
            other_sum += other_record["score"]
            other_known += 1

        status = "пройден" if result.won else DEATH_SHORT.get(result.death or "", "смерть")
        cells = [
            (result.level_name, TEXT),
            (status, GREEN if result.won else RED),
            ("—" if fraction is None else f"{int(round(fraction * 100))}%", TEXT),
            ("—" if not result.won else f"+{format_score(result.breakdown['time_bonus'])}", TEXT),
            (format_score(result.score), GOLD),
            (format_score(other_record["score"] if other_record else None), MUTED),
        ]
        for i, ((text, color), (_, pos, align)) in enumerate(zip(cells, cols)):
            texts.draw(f"all.r{row}c{i}", text, inner_l + inner_w * pos, y, color, 14,
                       bold=i in (0, 4), anchor_x=align)

    # Сумма и сравнение
    y -= 26
    arcade.draw_lrbt_rectangle_filled(inner_l, inner_l + inner_w, y + 14, y + 16, PANEL_OUTLINE)
    total = session.total_score
    texts.draw("all.total.label", "СУММА ОЧКОВ", inner_l, y - 16, TEXT, 20, bold=True)
    texts.draw_shadowed("all.total", format_score(total), inner_l + inner_w * 0.76, y - 16, GOLD, 26,
                        bold=True, anchor_x="right")
    if other_known:
        texts.draw("all.other", format_score(other_sum), inner_l + inner_w, y - 16, MUTED, 16,
                   bold=True, anchor_x="right")

    y -= 56
    if other_known:
        leader = compare_scores(
            total if who == HUMAN else other_sum,
            total if who == AI else other_sum,
        )
        if leader == "tie":
            verdict, color = "Ничья по сумме очков", TEXT
        elif leader == who:
            verdict, color = f"{_who_title(who)} набрал больше: {format_score(total)} против {format_score(other_sum)}", GREEN
        else:
            verdict, color = (
                f"Больше набрал {_who_lower(other)}: {format_score(other_sum)} против {format_score(total)}",
                RED,
            )
        if other_known < len(results):
            verdict += f" (рекорды есть на {other_known} из {len(results)} уровней)"
        texts.draw("all.verdict", verdict, cx, y, color, 15, bold=True, anchor_x="center")
    else:
        texts.draw("all.verdict", f"{_who_title(other)} эти уровни ещё не проходил — сравнить не с чем",
                   cx, y, MUTED, 14, anchor_x="center")

    texts.draw("all.hint", "R — пройти заново    Enter или Esc — в меню", cx, bottom + 24,
               MUTED, 13, anchor_x="center")


__all__ = [
    "draw_panel",
    "draw_bar",
    "draw_hud",
    "draw_ai_panel",
    "draw_ai_message",
    "draw_loading",
    "draw_popups",
    "forget_popups",
    "draw_flag_banner",
    "draw_level_result",
    "draw_death",
    "draw_all_done",
    "ALL_DONE",
    "DEAD",
    "LEVEL_DONE",
]

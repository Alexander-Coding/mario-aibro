"""Кэш надписей.

arcade.draw_text на каждом вызове собирает новую pyglet-надпись: раскладка
глифов, буферы. HUD, панель ИИ и экраны итогов рисуют по 20–40 строк за
кадр — это миллисекунды впустую. TextPool держит по одному arcade.Text на
ключ и трогает его свойства, только когда они поменялись.
"""

from __future__ import annotations

from typing import Optional

import arcade

SHADOW = (0, 0, 0, 150)


class TextPool:

    def __init__(self):
        self._texts: dict[str, arcade.Text] = {}
        self._params: dict[str, tuple] = {}

    def clear(self) -> None:
        self._texts.clear()
        self._params.clear()

    def forget(self, should_forget) -> None:
        """Выкидывает надписи, ключи которых подходят под условие."""

        for key in [key for key in self._texts if should_forget(key)]:
            self._texts.pop(key, None)
            self._params.pop(key, None)

    def __len__(self) -> int:
        return len(self._texts)

    def get(
        self,
        key: str,
        text: str,
        x: float,
        y: float,
        color=(255, 255, 255),
        size: float = 14,
        *,
        bold: bool = False,
        anchor_x: str = "left",
        anchor_y: str = "baseline",
        width: Optional[int] = None,
        align: str = "left",
        multiline: bool = False,
    ) -> arcade.Text:
        """Надпись по ключу: создаёт один раз, дальше только обновляет."""

        text = str(text)
        # arcade хранит цвет с альфой: (r, g, b) → (r, g, b, 255), иначе
        # сравнение ниже каждый кадр видело бы «новый» цвет и перекрашивало надпись
        color = tuple(int(c) for c in color)
        if len(color) == 3:
            color = (*color, 255)
        layout = (size, bold, anchor_x, anchor_y, width, align, multiline)

        label = self._texts.get(key)

        if label is None or self._params.get(key) != layout:
            label = arcade.Text(
                text,
                x,
                y,
                color,
                size,
                width=width,
                align=align,
                bold=bold,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                multiline=multiline,
            )
            self._texts[key] = label
            self._params[key] = layout
            return label

        if label.text != text:
            label.text = text
        if label.x != x or label.y != y:
            label.position = (x, y)
        if tuple(label.color) != color:
            label.color = color

        return label

    def draw(self, key: str, text: str, x: float, y: float, color=(255, 255, 255), size: float = 14, **kwargs) -> arcade.Text:
        label = self.get(key, text, x, y, color, size, **kwargs)
        label.draw()
        return label

    def draw_shadowed(
        self,
        key: str,
        text: str,
        x: float,
        y: float,
        color=(255, 255, 255),
        size: float = 14,
        *,
        shadow=SHADOW,
        offset: float = 2,
        **kwargs,
    ) -> arcade.Text:
        """Надпись с тенью — читается и на небе, и на кирпичах."""

        alpha = color[3] if len(color) == 4 else 255
        shadow_color = (*shadow[:3], int(shadow[3] * alpha / 255) if len(shadow) == 4 else alpha)

        self.draw(key + "#shadow", text, x + offset, y - offset, shadow_color, size, **kwargs)
        return self.draw(key, text, x, y, color, size, **kwargs)

    def width_of(self, key: str) -> float:
        """Ширина уже нарисованной надписи (для раскладки соседей)."""
        label = self._texts.get(key)
        return float(label.content_width) if label is not None else 0.0

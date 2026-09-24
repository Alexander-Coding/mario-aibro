"""Контроллер ИИ для живой игры: ленивая загрузка и безопасные вызовы.

Контракт (docs/CONTRACT.md, пишет владелец RL-части):

    from ai.agent import load_controller   # FileNotFoundError, если модели нет
    controller = load_controller()
    controller.reset(world)                # в начале уровня
    inputs = controller.act(world)         # каждый кадр мира
    info = controller.overlay()            # None или dict для HUD

ai/agent.py может ещё не существовать, модели может не быть, а импорт
torch может упасть — ни одна из этих бед не должна ронять игру. Тогда
на экране подсказка, как обучить модель, а управление — с клавиатуры.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

AI_MISSING_TEXT = "Модель ИИ не найдена. Обучите: .venv/bin/python -m ai.train"
AI_KEYBOARD_TEXT = "Управление — с клавиатуры"

# Загрузчик: () -> (контроллер или None, текст ошибки или None)
ControllerLoader = Callable[[], "tuple[Optional[Any], Optional[str]]"]


def load_ai_controller(path: Optional[str] = None) -> tuple[Optional[Any], Optional[str]]:
    """Импортирует ai.agent только сейчас и грузит модель.

    Возвращает (controller, None) или (None, текст для экрана).
    Ловит любую ошибку: нет модуля, нет файла модели, сломанный torch.
    """

    try:
        from ai.agent import load_controller
    except ModuleNotFoundError as error:
        if error.name in ("ai", "ai.agent"):
            # Нет самого ai/agent.py — RL-часть ещё не готова
            print(f"[ai] контроллер недоступен: {error}")
            return None, AI_MISSING_TEXT
        # ai/agent.py есть, но не хватает его зависимости (например, torch)
        traceback.print_exc()
        return None, f"{AI_MISSING_TEXT}\n(нет модуля {error.name})"
    except Exception as error:  # noqa: BLE001 — любая ошибка импорта не роняет игру
        traceback.print_exc()
        return None, f"{AI_MISSING_TEXT}\n(ошибка импорта: {type(error).__name__}: {error})"

    try:
        controller = load_controller(path) if path else load_controller()
    except FileNotFoundError as error:
        print(f"[ai] модели нет: {error}")
        return None, AI_MISSING_TEXT
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        return None, f"{AI_MISSING_TEXT}\n(ошибка загрузки: {type(error).__name__}: {error})"

    if controller is None or not hasattr(controller, "act"):
        return None, AI_MISSING_TEXT

    return controller, None


def controller_name(controller: Any) -> str:
    """Имя модели для HUD: из overlay()["model"], атрибута name или класса."""

    if controller is None:
        return ""
    name = getattr(controller, "name", None) or getattr(controller, "model_name", None)
    return str(name) if name else type(controller).__name__


# Контроллеры, у которых overlay() уже падал (чтобы не сыпать в лог)
_overlay_failed: set[int] = set()


def safe_overlay(controller: Any) -> Optional[dict]:
    """controller.overlay() без риска: ошибка или не словарь — значит панели нечего показать."""

    if controller is None or not hasattr(controller, "overlay"):
        return None

    try:
        info = controller.overlay()
    except Exception as error:  # noqa: BLE001
        # Панель зовётся каждый кадр — пишем в лог только первую ошибку
        if id(controller) not in _overlay_failed:
            _overlay_failed.add(id(controller))
            print(f"[ai] overlay() упал: {type(error).__name__}: {error}")
        return None

    return info if isinstance(info, dict) else None

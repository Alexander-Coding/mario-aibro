"""Ручные уровни: описание по тайлам → maps/level_*.json.

    .venv/bin/python -m tools.level_build            # пересобрать все пять + манифест
    .venv/bin/python -m tools.level_build warmup     # один уровень (манифест не трогает)
    .venv/bin/python -m tools.level_build --dry-run --check --ascii  # посмотреть, не писать

Каждый уровень ниже — это расстановка вручную, колонка за колонкой, через
LevelBuilder (tools/level_tiles.py). JSON в maps/ — результат сборки, его
не правят руками: поменять уровень — поправить функцию и пересобрать.

Главное правило игры — побеждает тот, кто набрал больше очков. Поэтому в
каждом уровне есть выбор «риск против очков»: монеты над ямами и выше
обычного прыжка, ряды блоков «?» (1–4 удара в каждом), враги, которых
выгоднее растоптать или сбить ударом снизу, чем обойти, секретная комната
за трубой и лестница перед флагом, с которой можно схватить шток за верх
(5000 против 100 у основания). Уровень должен окупать тщательное
прохождение, а не пробег на скорость.

Ряды: земля — ty 4–5, игрок стоит в ряду 6. Прыжок поднимает на 6.8
тайла и уносит на 7.5 тайла; обязательные прыжки — не выше 4 и через
ямы не шире 4 тайлов (tools/level_check.py проверяет это симуляцией).

Блок «?» твёрдый. Ряды блоков — только над ровной землёй, низом на 3–4
тайла выше пола (ряд 9 или 10), и не над ямой и не над разбегом прыжка:
блок над головой обрывает прыжок, и игрок падает в яму. Враг на ряду
блоков (enemy_pen — между бортиками) погибает от удара снизу.

Секретная комната: труба-вход («вниз», стоя на ней) → комната
(RoomBuilder: пол, стены, потолок, 20–40 монет, блоки «?» на много
ударов) → боковая труба у правой стены → игрок вылезает из трубы-выхода
дальше по уровню. Комната стоит заметно больше того, что пропускается.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable

from tools.level_tiles import MAPS_DIR, STAND_TY, LevelBuilder, RoomBuilder, time_limit_for

# ===============================================================
# СЕКРЕТНЫЕ КОМНАТЫ
# ===============================================================
# Координаты комнаты свои: пол — ряды 0–1, игрок стоит в ряду 2, потолок —
# верхний ряд коробки. Точка появления — (2, 9): игрок падает сверху слева.
# Устье выхода — в столбце width − 5 у правой стены.


def _room(width: int) -> RoomBuilder:
    room = RoomBuilder("комната").box(width, 12).start(2, 9)
    room.side_exit()
    return room


def room_warmup() -> RoomBuilder:
    """Учебная комната: дорожка монет по полу, ряд «?B?B?», монеты на нём и над ним."""
    r = _room(20)
    r.coins(3, 13, 2)
    r.blocks(5, 5, "?B?B?")
    r.coins(5, 9, 6)
    r.coins(5, 9, 8)
    return r


def room_goombas() -> RoomBuilder:
    """Гумба на полке из блоков «?»: сбить ударом снизу. Вторая — на полу."""
    r = _room(22)
    r.blocks(4, 5, "B?B?B?B")
    r.enemy_pen(4, 10, 5, at=(7,))
    r.coins(5, 9, 8)
    r.coins(3, 11, 2)
    r.coins(12, 16, 2)
    r.coins(12, 16, 4)
    r.enemy("goomba", 14)
    r.blocks(13, 6, "?.?")
    return r


def room_pipes() -> RoomBuilder:
    """Три кирпичные ступени с монетами и ряд «?» у выхода."""
    r = _room(20)
    r.row(3, 5, 4)
    r.row(7, 9, 6)
    r.coins(3, 5, 5)
    r.coins(3, 5, 6)
    r.coins(7, 9, 7)
    r.coins(7, 9, 8)
    r.coins(7, 9, 9)
    r.coins(10, 14, 2)
    r.blocks(11, 5, "?B?")
    r.coins(11, 13, 6)
    return r


def room_skybridge() -> RoomBuilder:
    """Монетный зал: три ряда монет над полом и полкой, блоки «?» по краям."""
    r = _room(22)
    r.row(6, 11, 5)
    r.coins(6, 11, 6)
    r.coins(6, 11, 7)
    r.coins(6, 11, 9)
    r.coins(3, 15, 2)
    r.question(3, 5)
    r.question(14, 5)
    r.question(15, 5)
    return r


def room_treasury() -> RoomBuilder:
    """Хранилище: 40 монет, шесть блоков «?» и полка из кирпича."""
    r = _room(24)
    r.coins(3, 18, 2)
    r.blocks(4, 5, "?B??B?")
    r.coins(4, 9, 6)
    r.coins(4, 9, 7)
    r.row(12, 17, 6)
    r.coins(12, 17, 7)
    r.coins(12, 17, 8)
    r.blocks(13, 10, "?..?")
    return r


# ===============================================================
# 1. РАЗМИНКА
# ===============================================================


def build_warmup() -> LevelBuilder:
    """Учит прыжкам: дорожки монет показывают траекторию.

    Всё мягкое: ступенька, трубы 2–4, ямы 2–3, редкие гумбы. Ряд «?» как
    в 1-1, по нему ходит гумба между бортиками — удар снизу сбивает её.
    Невысокая труба за ямой ведёт в секретную комнату, выход — из высокой
    трубы перед лестницей. В конце — лестница в 8 ступеней; дорожка монет
    с её верхушки ведёт на верх штока.
    """

    b = LevelBuilder(
        "Разминка",
        "Учит прыжкам: дорожки монет показывают траекторию. Труба за ямой "
        "ведёт в секретную комнату. С верхушки лестницы в конце можно "
        "схватить флаг за самый верх.",
    )
    b.ground(0, 150)
    b.start(3)

    # --- Монеты над головой и первый ряд блоков: прыжки на месте
    b.coins(7, 10, 8)
    b.blocks(12, 9, "?B?")

    # --- Ступенька в один блок: дорожка показывает прыжок с разбега
    b.column(19, STAND_TY, STAND_TY)
    b.jump_trail(16)

    # --- Труба высотой 2
    b.tube(28, 2)
    b.jump_trail(25)

    # --- Яма в 2 тайла
    b.pit(35, 36)
    b.jump_trail(33)

    # --- Первая гумба, над ней монеты — награда за прыжок сверху
    b.enemy("goomba", 46)
    b.coins(43, 47, 9)

    # --- Кирпичная полка: запрыгнуть (4 тайла), наверху монеты, над ней «?»
    b.row(51, 56, 9)
    b.coins(51, 56, 10)
    b.question(53, 13)
    b.question(54, 13)

    # --- Труба 3, за ней яма 3
    b.tube(61, 3)
    b.pit(67, 69)
    b.jump_trail(66)

    # --- Горка: три ступени вверх и три вниз, монеты над вершиной
    b.stairs_up(75, 3)
    b.stairs_down(78, 3)
    b.coins(76, 79, 10)

    # --- Две гумбы подряд: с горки на них — цепочка прыжков
    b.enemy("goomba", 86)
    b.enemy("goomba", 88)

    # --- Ряд «?» как в 1-1, по нему ходит гумба; один блок повыше
    b.blocks(95, 9, "B?B?B?B")
    b.enemy_pen(95, 101, 9, at=(98,))
    b.question(98, 13)

    # --- Koopa: первый прыжок — в панцирь, второй — добить
    b.enemy("koopa", 107)

    # --- Яма 2 и сразу труба 2 — вход в секретную комнату
    b.pit(113, 114)
    b.tube(117, 2)
    b.jump_trail(112)

    # --- Монетный коридор — бонус тому, кто бежит, а не стоит
    b.coins(122, 129, STAND_TY + 1)

    # --- Самая высокая труба уровня — из неё вылезают из комнаты
    b.tube(133, 4)
    b.warp(117, 133, room_warmup(), "warmup")

    # --- Лестница к флагу: 8 ступеней, площадка 2, до штока 4 тайла
    pole = b.final_stairs(139, height=8, top_width=2, run=4)
    b.jump_trail(pole - 5, until_col=pole - 2)

    return b


# ===============================================================
# 2. ТРУБЫ И ЯМЫ
# ===============================================================


def build_pipes() -> LevelBuilder:
    """Трубы 2–4 вперемешку с ямами 2–4 и врагами между трубами.

    Ямы стоят сразу за трубами: прыгать приходится с высоты и далеко.
    Дорожки монет над ямами висят по траектории прыжка с края; самая
    широкая яма (4) — с самой длинной дорожкой. Труба после «колодца»
    уводит в комнату и выводит за следующей ямой, мимо загона с гумбой.
    """

    b = LevelBuilder(
        "Трубы и ямы",
        "Трубы разной высоты, ямы от 2 до 4 тайлов, враги между трубами. "
        "Монеты над ямами — тем, кто прыгает с самого края. Одна труба — в "
        "секретную комнату.",
    )
    b.ground(0, 190)
    b.start(3)

    b.tube(10, 2)
    b.enemy("goomba", 15)
    b.tube(19, 3)

    # Яма 2, за ней труба на расстоянии двух тайлов
    b.pit(24, 25)
    b.jump_trail(23)
    b.tube(28, 2)

    # Две гумбы в «колодце» между трубами, над ними «?»
    b.enemy("goomba", 33)
    b.enemy("goomba", 35)
    b.question(34, 9)
    b.tube(38, 4)

    # Яма 3 с дорожкой
    b.pit(43, 45)
    b.jump_trail(42)
    b.tube(49, 3)

    # Яма сразу за трубой: прыжок с высоты
    b.pit(52, 53)
    b.jump_trail(50, land_ty=STAND_TY)
    b.tube(57, 4)

    b.enemy("koopa", 63)
    b.tube(67, 2)

    # Самая широкая яма — 4 тайла
    b.pit(71, 74)
    b.jump_trail(70)

    # Ряд блоков над гумбой, как в 1-1
    b.blocks(78, 9, "?B?B?")
    b.enemy("goomba", 80)

    # Две трубы и «колодец» с монетами между ними; вторая — вход в комнату
    b.tube(86, 3)
    b.coins(88, 89, STAND_TY)
    b.tube(90, 3)

    # Яма 3 и высокая труба за узкой площадкой
    b.pit(96, 98)
    b.jump_trail(95)
    b.tube(101, 4)

    # Загон на ряду «?»: гумба между бортиками, удар снизу её сбивает
    b.blocks(104, 9, "B?B?B")
    b.enemy_pen(104, 108, 9, at=(106,))
    b.tube(111, 2)
    b.warp(90, 111, room_pipes(), "pipes")

    b.pit(115, 117)
    b.jump_trail(114)
    b.tube(120, 3)
    b.pit(124, 125)
    b.jump_trail(123)
    b.tube(128, 4)

    # Коридор между трубами: гумба и купа
    b.enemy("goomba", 131)
    b.enemy("koopa", 134)
    b.tube(138, 2)

    # Две лестницы и яма между ними, как в 1-1
    b.stairs_up(144, 4)
    b.pit(148, 149)
    b.stairs_down(150, 4)
    b.jump_trail(147, STAND_TY + 4, land_ty=STAND_TY + 4)

    # Гумбы у подножия лестницы: прыжок с неё — цепочка
    b.enemy("goomba", 157)
    b.enemy("goomba", 159)
    b.pit(164, 166)
    b.jump_trail(163)

    pole = b.final_stairs(172, height=8, top_width=1, run=4)
    b.jump_trail(pole - 5, until_col=pole - 2)
    return b


# ===============================================================
# 3. КОРИДОР ГУМБ
# ===============================================================


def build_goombas() -> LevelBuilder:
    """Плотные группы врагов: на них набираются очки.

    Коридор под кирпичной крышей: внизу безопасный путь с гумбами и
    низким потолком (прыжок в 3 тайла — ровно растоптать гумбу), наверху
    по крыше — koopa, монеты и «?», дыры в крыше роняют обратно в коридор.
    В крыше — блоки «?»: удар изнутри коридора сбивает koopa, что ходит
    над ними. Загон между трубами держит врагов на месте. Первая труба —
    вход в секретную комнату, выход — из высокой трубы у коридора.
    """

    b = LevelBuilder(
        "Коридор гумб",
        "Группы гумб и koopa — очки за каждого растоптанного и сбитого снизу. "
        "Под крышей безопасный коридор, по крыше — рискованный путь с монетами. "
        "Первая труба — в секретную комнату.",
    )
    b.ground(0, 190)
    b.start(3)

    # --- Ряд «?» у старта и парад: три гумбы подряд, монеты над ними
    b.blocks(7, 9, "?B?B?")
    for tx in (14, 16, 18):
        b.enemy("goomba", tx)
    b.coins(13, 19, 9)
    b.tube(24, 2)

    for tx in (30, 32):
        b.enemy("goomba", tx)
    b.enemy("koopa", 35)
    b.coins(29, 36, 9)
    b.tube(40, 4)
    b.warp(24, 40, room_goombas(), "goombas")

    # --- Коридор: крыша из кирпича в ряду 10 (внутри 4 ряда высоты)
    roof = 10
    b.row(44, 88, roof)
    # Дыры в крыше — сорвался, и ты снова внизу
    b.clear(60, 61, roof, roof)
    b.clear(76, 77, roof, roof)
    # Блоки «?» в крыше: бить изнутри коридора, koopa над ними слетает
    for tx in (48, 53, 55, 69, 71, 84):
        b.question(tx, roof)

    # Внизу: гумбы группами и редкие монеты
    for tx in (50, 52, 64, 66, 68, 80, 82):
        b.enemy("goomba", tx, STAND_TY)
    b.coins(56, 58, 8)
    b.coins(72, 74, 8)

    # Наверху: koopa, монеты и блоки «?»
    b.enemy("koopa", 54, roof + 1)
    b.enemy("koopa", 70, roof + 1)
    b.coins(45, 52, roof + 2)
    b.coins(63, 68, roof + 2)
    b.coins(79, 87, roof + 2)
    for tx in (57, 73, 85):
        b.question(tx, roof + 4)

    # --- Ямы с гумбами на островах
    b.pit(94, 95)
    b.jump_trail(93)
    b.enemy("goomba", 99)
    b.pit(102, 104)
    b.jump_trail(101)
    b.enemy("goomba", 106)
    b.enemy("goomba", 108)
    b.pit(111, 112)
    b.jump_trail(110)

    # --- Загон: две koopa между трубами никуда не денутся
    b.tube(117, 3)
    b.enemy("koopa", 121)
    b.enemy("koopa", 123)
    b.tube(126, 3)
    b.coins(120, 124, 10)

    # --- Плато с гумбами: лестница вверх, стол высотой 4, лестница вниз
    b.stairs_up(133, 3)
    for tx in range(136, 149):
        b.column(tx, STAND_TY, STAND_TY + 3)
    b.stairs_down(149, 3)
    for tx in (139, 142, 145):
        b.enemy("goomba", tx)
    b.coins(137, 147, STAND_TY + 6)

    # --- Последняя волна перед лестницей
    for tx in (157, 159, 161):
        b.enemy("goomba", tx)
    b.enemy("koopa", 164)

    pole = b.final_stairs(168, height=8, top_width=2, run=4)
    b.jump_trail(pole - 5, until_col=pole - 2)
    return b


# ===============================================================
# 4. СОКРОВИЩНИЦА
# ===============================================================


def build_treasury() -> LevelBuilder:
    """Блоки «?» и россыпи монет: чем жаднее, тем опаснее.

    Блок «?» над столбиком посреди ямы бьётся, только если встать на
    столбик — промахнулся ногой, и ты внизу. Монеты висят выше обычного
    прыжка и ниже его. Хранилище с монетами под крышей, в крыше блоки
    «?» — бить изнутри. Высокая труба ведёт в секретное хранилище и
    выводит за пропастью со ступенями. Ряд «?» с гумбой наверху. В конце —
    башня, а за ней яма до флагштока: прыгай с разбега на верх штока
    (5000) или падай.
    """

    b = LevelBuilder(
        "Сокровищница",
        "Блоки «?» и монеты: чем жаднее, тем опаснее. Высокая труба — в "
        "секретное хранилище. Финал — башня над ямой: прыжок на верх флага за 5000.",
    )
    b.ground(0, 230)
    b.start(3)

    # --- Прихожая: монетное поле под рядом «?» и пирамида блоков
    b.coins(7, 14, STAND_TY + 1)
    b.coins(8, 13, STAND_TY + 2)
    b.blocks(8, 10, "?B?B?")
    b.question(17, 9)
    b.question(19, 9)
    b.question(18, 13)
    b.row(18, 18, 9)

    # --- Яма 5 со столбиком шириной 2 посередине, «?» над левым его краем:
    # прыгать дальше — с правого края, иначе голова упрётся в блок
    b.pit(25, 29)
    b.column(27, 4, 5, "ground")
    b.column(28, 4, 5, "ground")
    b.question(27, 9)
    b.jump_trail(24)

    # --- Яма 3, монеты выше обычной дорожки (на две клетки)
    b.pit(33, 35)
    b.jump_trail(32)
    b.coins(33, 35, 13)

    # --- Три кирпича-островка над широкой ямой, на каждом столбик монет
    b.pit(41, 55)
    for tx, ty in ((44, 6), (48, 7), (52, 6)):
        b.row(tx, tx, ty)
        b.coins(tx, tx, ty + 1)
        b.coins(tx, tx, ty + 2)
        b.coin(tx, ty + 3)
    b.coin(46, 11)
    b.coin(50, 11)

    # --- Россыпь над гумбами
    for tx in (60, 62, 64):
        b.enemy("goomba", tx)
    b.coins(59, 65, 9)
    b.coins(60, 64, 10)

    # --- Хранилище: стены высотой 3, крыша в ряду 9 с дырой, внутри монеты;
    # в крыше два «?» (бить изнутри), над ними ещё два (стоя на крыше)
    b.column(70, STAND_TY, 8, "brick")
    b.column(80, STAND_TY, 8, "brick")
    b.row(70, 80, 9)
    b.clear(74, 76, 9, 9)
    for tx in range(71, 80):
        b.coin(tx, STAND_TY)
        if tx not in (74, 75, 76):
            b.coin(tx, STAND_TY + 1)
    for tx in (72, 78):
        b.question(tx, 9)
        b.question(tx, 13)

    # --- Лестница с ямой посередине, монеты над ямой
    b.stairs_up(86, 4)
    b.pit(90, 92)
    b.stairs_down(93, 4)
    b.jump_trail(89, STAND_TY + 4, land_ty=STAND_TY + 4)

    # --- Трубы с монетами наверху, между ними koopa; высокая — в хранилище
    b.tube(101, 3)
    b.coins(101, 102, 9)
    b.enemy("koopa", 106)
    b.tube(109, 4)
    b.coins(109, 110, 10)

    # --- Ступени над пропастью: кирпичи шириной 2, монеты по дорожкам
    b.pit(115, 136)
    for tx, ty in ((118, 6), (123, 8), (128, 8), (133, 6)):
        b.row(tx, tx + 1, ty)
    b.jump_trail(114)
    b.jump_trail(119, 7)
    b.jump_trail(124, 9, land_ty=9)
    b.jump_trail(129, 9, land_ty=7)
    b.jump_trail(134, 7, land_ty=STAND_TY)
    b.coins(125, 126, 13)

    # --- Труба-выход из хранилища
    b.tube(139, 2)
    b.warp(109, 139, room_treasury(), "treasury")

    # --- Коридор блоков «?»: ряд над землёй, по нему ходит гумба, один повыше
    b.blocks(143, 9, "B?B?B?B")
    b.enemy_pen(143, 149, 9, at=(146,))
    b.question(146, 13)
    b.enemy("goomba", 151)
    b.enemy("goomba", 153)

    # --- Узкие ямы подряд с дорожками монет
    for start in (156, 162, 168):
        b.pit(start, start + 2)
        b.jump_trail(start - 1)

    b.enemy("koopa", 176)
    b.blocks(173, 9, "?B?")
    b.coins(173, 175, 10)

    # --- Башня и яма перед флагштоком: вниз пути нет, только прыжок
    pole = b.final_stairs(182, height=8, top_width=2, run=4)
    b.pit(pole - 4, pole - 2)
    b.jump_trail(pole - 5, until_col=pole - 2)
    return b


# ===============================================================
# 5. НЕБЕСНЫЙ МОСТ
# ===============================================================


def build_skybridge() -> LevelBuilder:
    """Три пропасти и цепочки парящих кирпичных платформ.

    Первая пропасть — широкие платформы и подъём до ряда 11, над средней
    платформой высоко висят монеты и «?». Между пропастями — ряд «?» с
    гумбой. Вторая — длинная цепочка узких платформ вверх-вниз. Третья —
    камни по одному тайлу; труба перед ней — вход в секретную комнату,
    выход — за пропастью, как короткий путь в Super Mario Bros.
    """

    b = LevelBuilder(
        "Небесный мост",
        "Длинные пропасти и цепочки парящих платформ. Монеты в воздухе — "
        "для тех, кто прыгает точно. Труба перед последней пропастью — в "
        "секретную комнату.",
    )
    b.ground(0, 215)
    b.start(3)

    b.coins(6, 10, 9)
    b.tube(14, 2)
    b.pit(19, 20)
    b.jump_trail(18)
    b.enemy("goomba", 25)

    # --- Пропасть 1: платформы по 3 тайла, вверх до ряда 11 и вниз
    b.pit(30, 61)
    chain = ((34, 6), (40, 8), (46, 10), (52, 8), (58, 6))
    for tx, ty in chain:
        b.row(tx, tx + 2, ty)
    b.jump_trail(29)
    for tx, ty in chain:
        b.jump_trail(tx + 2, ty + 1)
    b.coins(45, 49, 16)
    b.question(46, 14)  # над левым краем платформы: разбег с правого края свободен

    # --- Земля: ряд «?» с гумбой наверху, ещё одна гумба внизу
    b.blocks(65, 9, "B?B?B?B")
    b.enemy_pen(65, 71, 9, at=(68,))
    b.enemy("goomba", 74)

    # --- Пропасть 2: узкие платформы, подъём до ряда 12 и спуск
    b.pit(80, 124)
    chain = ((83, 7), (88, 9), (93, 11), (98, 12), (104, 10), (109, 8), (114, 6), (119, 7))
    for tx, ty in chain:
        width = 3 if tx == 98 else 2
        b.row(tx, tx + width - 1, ty)
    b.jump_trail(79)
    for tx, ty in chain:
        width = 3 if tx == 98 else 2
        b.jump_trail(tx + width - 1, ty + 1)
    b.coins(97, 101, 17)
    b.question(98, 16)

    # --- Передышка: ряд «?», koopa, труба-вход в комнату
    b.blocks(128, 9, "?B?")
    b.enemy("koopa", 132)
    b.tube(137, 3)

    # --- Пропасть 3: камни по одному тайлу, над каждым монеты
    b.pit(145, 175)
    for tx in range(148, 173, 4):
        b.row(tx, tx, 6)
        b.coin(tx, 8)
        b.coin(tx, 9)
    b.jump_trail(144)

    # --- Труба-выход из комнаты за пропастью
    b.tube(177, 2)
    b.warp(137, 177, room_skybridge(), "skybridge")

    pole = b.final_stairs(182, height=8, top_width=1, run=4)
    b.jump_trail(pole - 5, until_col=pole - 2)
    return b


# ===============================================================
# СБОРКА
# ===============================================================

# id уровня → (файл, сборщик)
LEVELS: dict[str, tuple[str, Callable[[], LevelBuilder]]] = {
    "warmup": ("level_warmup.json", build_warmup),
    "pipes": ("level_pipes.json", build_pipes),
    "goombas": ("level_goombas.json", build_goombas),
    "treasury": ("level_treasury.json", build_treasury),
    "skybridge": ("level_skybridge.json", build_skybridge),
}


# Место в манифесте maps/levels.json: после 1-1, по возрастанию сложности.
# split: "train" — нейросеть на нём учится, "test" — отложенный уровень, на
# котором проверяют, что она умеет играть, а не заучила трассу. Это порядок
# и роли для НОВОЙ записи; у записей, что уже есть в манифесте, id, split и
# место не меняются — их ведёт владелец обучения.
MANIFEST_ORDER: tuple[tuple[str, str, str], ...] = (
    ("warmup", "warmup", "test"),
    ("goombas", "goombas", "train"),
    ("pipes", "pipes", "train"),
    ("skybridge", "skybridge", "test"),
    ("treasury", "treasury", "test"),
)


def update_manifest(built: dict[str, dict]) -> str:
    """Обновить ручные уровни в maps/levels.json.

    Запись уровня, которая уже есть (ищется по файлу), остаётся на своём
    месте со своими id и split — меняются имя, лимит времени и описание.
    Уровня нет в манифесте — он добавляется после 1-1 и уже вписанных, в
    порядке MANIFEST_ORDER. Чужие записи не трогаются.
    """

    path = MAPS_DIR / "levels.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"levels": []}
    levels = list(manifest.get("levels", []))
    by_file = {item.get("file"): item for item in levels}

    added = []
    for level_id, key, split in MANIFEST_ORDER:
        data = built.get(key) or build(key)
        item = by_file.get(LEVELS[key][0])
        if item is None:
            item = {"id": level_id, "file": LEVELS[key][0], "split": split}
            added.append(item)
        item.update(name=data["name"], time_limit=data["time_limit"],
                    description=data["description"])

    if added:
        # После 1-1 и уже вписанных ручных уровней (или в начало)
        ours = {file_name for file_name, _ in LEVELS.values()}
        anchor = max((i + 1 for i, item in enumerate(levels)
                      if item.get("id") == "1-1" or item.get("file") in ours), default=0)
        levels = levels[:anchor] + added + levels[anchor:]

    # Порядок ключей записи как в манифесте
    order = ("id", "file", "name", "time_limit", "split", "description")
    manifest["levels"] = [
        {**{k: item[k] for k in order if k in item},
         **{k: v for k, v in item.items() if k not in order}}
        for item in levels
    ]
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    return str(path)


def measured_seconds(data: dict) -> float | None:
    """Самый долгий тщательный проход в GameWorld (tools.level_check,
    с комнатой и без): сценарный бот собирает всё, что даёт граф. None —
    проход сорвался, остаётся оценка по формуле."""
    from tools.level_check import check_level

    report = check_level(data, bot=False, sim=False, thorough=True)
    done = [tour["seconds"] for tour in report.tours if tour["ok"]]
    return max(done) if done and len(done) == len(report.tours) else None


def build(level_id: str, measure: bool = True) -> dict:
    """Собрать уровень. Лимит времени — по тщательному прохождению с
    запасом 30 % (tools/level_tiles.time_limit_for): по формуле (весь путь,
    все удары по блокам «?», монеты, комната) и по замеру в GameWorld,
    что больше; и не меньше лимита по пробегу. Формула на трубах и мосту
    занижала: монеты в воздухе берутся отдельными прыжками."""
    _, builder = LEVELS[level_id]
    b = builder()
    if b.time_limit is None:
        b.time_limit = time_limit_for(b.run_tiles(), b.thorough_seconds())
        measured = measured_seconds(b.to_level()) if measure else None
        if measured is not None:
            b.time_limit = max(b.time_limit, time_limit_for(0.0, measured))
    return b.to_level()


def write(level_id: str, data: dict) -> str:
    file_name, _ = LEVELS[level_id]
    path = MAPS_DIR / file_name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Собрать ручные уровни в maps/")
    parser.add_argument("levels", nargs="*", help=f"какие ({', '.join(LEVELS)}); по умолчанию все")
    parser.add_argument("--dry-run", action="store_true", help="не писать файлы")
    parser.add_argument("--ascii", action="store_true", help="напечатать уровень символами")
    parser.add_argument("--check", action="store_true", help="проверить tools.level_check (без бота)")
    parser.add_argument("--no-measure", action="store_true",
                        help="лимит только по формуле, без замера тщательного прохода в GameWorld")
    args = parser.parse_args(argv)

    ids = args.levels or list(LEVELS)
    unknown = [i for i in ids if i not in LEVELS]
    if unknown:
        parser.error(f"нет таких уровней: {', '.join(unknown)}")

    failed = False
    built: dict[str, dict] = {}
    for level_id in ids:
        data = build(level_id, measure=not args.no_measure)
        built[level_id] = data
        target = "(не записан)" if args.dry_run else write(level_id, data)
        print(f"{level_id}: {len(data['objects'])} объектов, лимит {data['time_limit']} с → {target}")

        if args.ascii:
            from tools.level_ascii import render
            print(render(data))
        if args.check:
            from tools.level_check import check_level, format_report
            report = check_level(data, bot=False)
            print(format_report(report))
            failed = failed or not report.ok
        print()

    if not args.dry_run and not args.levels:
        print(f"манифест → {update_manifest(built)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

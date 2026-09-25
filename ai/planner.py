"""Планировщик-учитель: поиск по симулятору вместо правил со скрытым состоянием.

Прежний учитель (ai/teacher.py) — модель it4a плюс правила сбора со
счётчиками, запретами и сценариями выхода из тупика. Ученик этих
счётчиков не видит: на одно наблюдение он выучивал два ответа и
чередовал их («вправо+прыжок / влево» по кругу). У планировщика ответ —
функция состояния мира и прошлого действия (оно есть в наблюдении
ученика), памяти между решениями нет. Прежняя версия держала «тёплый
старт» — план прошлого решения как лишнего кандидата; проверка показала,
что без него она стояла в 90 % решений, то есть ответ зависел от истории.

Каждое решение (раз в 4 кадра, как у модели) — лучевой поиск на копии
мира (world.snapshot() / world.restore(), мир возвращается ровно таким,
каким был):

- корень: все 6 действий на 4 кадра. Прыжок в воздухе не срабатывает, и
  такие ветки дают в точности то же состояние, что без прыжка, — они
  сливаются с действием без прыжка (у дубля вероятность 0, оценка та же);
- дальше шаги растут (до 1.4–1.6 с вперёд; у флагштока до 2.1 с): хватает,
  чтобы вернуться за монетой позади и снова пойти вперёд. Границы шагов —
  на кадрах мира (ALIGNED_GRID), и продолжение плана прошлого решения
  ложится на сетку следующего шаг в шаг. В воздухе ветвится только
  горизонталь;
- первые шаги у каждого первого действия остаётся своя лучшая ветка,
  дальше — общий луч. Ранжирует луч ценность + притяжение к видимым
  несобранным предметам и к высокому захвату флага (иначе ветка «назад за
  монетой» или «разбег к краю лестницы» отсекается раньше, чем до цели
  дойдёт); оценки действий (last_q) и выбор — по ценности листа с
  потенциалом (ниже): цель без пути по карте в неё не входит, и
  недостижимая монета не заставит висеть под ней;
- лист в воздухе досчитывается до приземления: над пропастью — смерть,
  в яме без выхода — ценность места падения; оценка действия — по лучшему
  листу, досчитанному до земли;
- «сделать сейчас» (_shifted): лучший план, начатый с действия «на месте»
  (стоять на земле, бежать в стену), — ещё и без этого начала, а у
  остальных действий — сдвинутый на одно решение. В неподвижном мире
  «сделать сейчас» не хуже, а стоять на ничьей значило стоять вечно: на
  следующем решении мир тот же.

Ценность листа — оценка итогового счёта (в очках игры):

    очки + кадры_таймера / 3 − K_PROGRESS · путь_до_цели − HIT · потерянные_жизни
        + POTENTIAL · max по целям (ценность · (1 − путь_до_неё / R))

Потенциал — лучшая видимая цель (монета, блок «?», непосещённая труба-вход
в окне ученика), до которой карта знает путь короче R тайлов: подойти к ней
уже ценно, взять — ценнее (POTENTIAL < 1). Без него ценность давал только
план, успевший взять цель за горизонт, а такой план находился не на каждом
решении — и учитель без памяти ходил по кругу. Встать на непосещённую
трубу-вход — конечный лист с бонусом трубы (дальше правило «вниз»).

Бонус за время — 20 очков за секунду, то есть 1/3 очка за кадр; путь до
цели — не X, а длина пути по карте проходимости (NavField: стоять можно
на пустой клетке над твёрдой, переходы — шаг, падение с края и прыжки по
физике игрока). В «хранилище» сокровищницы (загон с крышей, выход — дыра
в крыше левее правой стены) шаг влево к дыре — это продвижение, а упор в
правую стену — нет. Смерть — очки − DEATH (теряются флаг и бонус за
время), флагшток — очки с флагом + бонус за оставшиеся секунды. В
секретной комнате путь до цели — до устья выхода плюс путь от трубы, из
которой вылезешь, до флага. На арене без флага (ai/levelgen.build_drill)
цель — ближайший видимый несобранный предмет, а когда их нет, — самый
правый столбец, куда можно дойти. Поправки пути (доли очка, решают ничьи):
раньше взятое дороже (EARLY), стоять — чуть дороже (IDLE); разворот
вправо ↔ влево стоит FLIP_COST. Взятое вне окна ученика не в счёт,
видимое впереди — со скидкой AHEAD_LOSS (попадётся и так).

Правило одно: стоя на непосещённой трубе-входе — «вниз». Остальное — поиск.

Для сбора примеров (ai/imitate.py): last_source ("plan" / "rule"),
last_q — оценки 7 действий относительно лучшего (0 у лучшего, −inf у
невозможных), last_probs = softmax(Q / TAU) по действиям без дублей.

    planner = Planner()
    planner.reset(world)
    while not world.done:
        world.step(planner.act(world))
"""

from __future__ import annotations

import contextlib
import heapq
import inspect
import math
import time
from collections import Counter
from typing import Optional

import numpy as np

from ai import features
from ai.env import FRAME_SKIP, action_inputs
from core import rules
from core.world import (
    COMPLETE,
    DEAD,
    FLAGPOLE,
    PIPE_DOWN,
    PIPE_SIDE,
    PLAYING,
    TO_CASTLE,
    GameWorld,
    Inputs,
)
from Entities.player import Player
from Objects.Coin import Coin

TILE = rules.TILE

# Индексы ai.features.ACTIONS
STAND, RIGHT, LEFT, JUMP, RIGHT_JUMP, LEFT_JUMP, DOWN = range(7)
N_ACTIONS = 7
MOVES = (STAND, RIGHT, LEFT, JUMP, RIGHT_JUMP, LEFT_JUMP)
HORIZONTAL = {STAND: 0, RIGHT: 1, LEFT: -1, JUMP: 0, RIGHT_JUMP: 1, LEFT_JUMP: -1}
# (горизонталь, прыжок на первом кадре) → действие
BY_INPUTS = {(HORIZONTAL[a], a in (JUMP, RIGHT_JUMP, LEFT_JUMP)): a for a in MOVES}

# ---------------------------------------------------------------
# Поиск
# ---------------------------------------------------------------

# Длины шагов при ALIGNED_GRID = False (сетка от решения; см. ниже сетку по
# часам мира). Первый — ровно повтор решения (FRAME_SKIP). Первые
# шесть — по 4 кадра: план прошлого решения без первого шага ложится на
# сетку следующего поиска на первые 24 кадра (памяти нет — план
# находится заново, и чем мельче голова, тем чаще тот же). Замер без
# тёплого старта (6 уровней + 20 процедурных + 10 арен): (4,4,4,8,8,8,
# 12,12,16,20) — 5/6 и 19/20 побед, петли на уровнях 17.9 %; эти 12 шагов —
# 6/6 и 19/20, петли на уровнях 0, арены собраны 10/10; решение 148 мс против
# 121; все шаги по 4 кадра — 4/6 и 15/20, 300 смен направления в минуту
STEPS = (FRAME_SKIP, 4, 4, 4, 4, 4, 8, 8, 8, 12, 16, 20)
HORIZON = sum(STEPS)
# У флагштока горизонт длиннее: разбег по верху лестницы и прыжок к верху
# штока — ~2 с, и с горизонтом 1.6 с «схватиться низко сейчас» (400)
# выглядело лучше, чем «добежать до края и прыгнуть» (800–5000 за границей горизонта)
POLE_STEPS = STEPS + (16, 16)
POLE_NEAR = 16          # тайлов до штока
# Сетка по часам мира (ALIGNED_GRID): границы шагов — не «через столько
# кадров от решения», а на кадрах мира: первые 24 кадра — каждые 4, до 48 —
# на кратных 8, дальше до 96 (у штока до 128) — на кратных 16. На следующем
# решении (через 4 кадра) каждая граница прошлой сетки есть и в новой, и
# продолжение прошлого лучшего плана представимо шаг в шаг — раньше после 20
# кадров оно на сетку не ложилось, и план, найденный на одном решении, на
# следующем находился не всегда (петли «найден — не найден»). Горизонт
# кончается на кадре мира, кратном 16: 84–96 кадров от решения
ALIGNED_GRID = True
GRID_ZONES = ((24, 4), (48, 8), (96, 16))
POLE_GRID_END = 128
# До этой глубины у каждого первого действия остаётся своя лучшая ветка
DIVERSE_DEPTH = 4
# Потом — общий луч: столько веток, не больше PER_GROUP от одного первого
# действия. Луч 4 против 3 на 6 уровнях: очки те же, решение на 15 % дольше
BEAM = 3
PER_GROUP = 2
# Ветвление: на земле (стоять — только в начале: ждать врага имеет смысл
# сейчас, а не через секунду) и в воздухе
GROUND_EARLY = (RIGHT, RIGHT_JUMP, LEFT, LEFT_JUMP, JUMP, STAND)
GROUND_LATE = (RIGHT, RIGHT_JUMP, LEFT, LEFT_JUMP, JUMP)
AIR = (RIGHT, LEFT, STAND)
STAND_UNTIL = 3
# Воздух в карте проходимости: ряд вниз и тайл вбок (в тайлах бега)
AIR_DOWN = 0.5
AIR_SIDE = 1.0
# Траектории карты проходимости — не дольше (кадров)
ARC_FRAMES = 150
# Прыжок с места поднимает ноги на 6.8 тайла, игрок ростом в тайл: монета
# до 7 рядов над клеткой, где стоишь, достаётся (блок — до 7 рядов над
# головой). Было 5, и монеты на 6–7 рядов выше пола не получали поля —
# учитель за ними не ходил (проверка: 0 из 24 в сценарии «позади»)
JUMP_REACH = 7
# Лист в воздухе: досчитать до приземления (не над пропастью ли он); у
# каждого первого действия — пока лучший лист не досчитан, не больше LAND_TRIES
LAND_FRAMES = 45
LAND_TRIES = 8
# Кандидаты «сделать сейчас» (Planner._shifted): сдвигать на одно решение и
# лучшие листы действий не «на месте»; кругов снятия «на месте» не больше
SHIFT_ALL = True
NOW_ROUNDS = 3

# ---------------------------------------------------------------
# Ценность (в очках игры)
# ---------------------------------------------------------------

# Бонус за время: 20 очков за секунду — 1/3 очка за кадр
TIME_COST = rules.TIME_BONUS_PER_SECOND * rules.FIXED_DT
# Тайл пути стоит в 1.5 раза больше, чем время на него бегом (8 кадров):
# без предметов вперёд всегда выгоднее, монета в 100 очков окупает крюк
# в несколько секунд
K_PROGRESS = 1.5 * TIME_COST * TILE / 4.0
DEATH = 6000.0
HIT = 1500.0
WARP_BONUS = 1500.0
# Путь до цели неизвестен (карта не нашла прохода): как очень далеко
LOST_EXTRA = 40.0
# Спешка: если времени меньше URGENT × нужного на путь, прогресс дороже
URGENT = 1.5
URGENT_MAX = 4.0

# Притяжение для ранжирования луча: предмет в R тайлах пути тянет с
# силой ценность × (1 − путь/R). В оценки действий не входит
ATTRACT_RADIUS = 10.0
BLOCK_ATTRACT = 100.0
COIN_ATTRACT = 100.0

# Потенциал цели — в самой ценности листа (и в оценках действий): лучшая из
# видимых целей, до которых карта проходимости знает путь, добавляет
# вес × ценность × (1 − путь/радиус). Лист, подошедший к цели, но не
# успевший взять её за горизонт, теперь хуже взявшего не на всю ценность
# цели, а на её долю: план, найденный на одном решении и не найденный на
# следующем, больше не перекидывает выбор на сотни очков. Без этого учитель
# без памяти ходил по кругу: «вправо, влево, стоять» у трубы позади
# (gen67002, gen67003), под блоками (gen68038), в комнате (gen7009), под
# платформой с монетами (gen61016) — до конца времени.
# - Максимум, а не сумма: у максимума конусов нет вершин, кроме самих
#   целей, — от любой точки путь к лучшей цели его растит. Сумма давала
#   седло: две цели по разные стороны гасили друг друга, и учитель стоял
#   между ними (gen7009, 125 с).
# - Монеты и блоки: вес 0.4, ценность не больше POTENTIAL_CAP (блок — по
#   оставшимся ударам), радиус 7 тайлов. Наклон 0.4·100/7 = 5.7 очка на
#   тайл больше цены пути K_PROGRESS (4): к видимой монете позади учитель
#   идёт. При наклоне ровно K (вес 0.4, радиус 10) «назад» и «вперёд» были
#   равны, и учитель качался на высокой платформе pipes (60 кадров по
#   кругу); при весе 0.25 наклон меньше K, и под платформой gen61016 снова
#   петля. Потолок 40 очков меньше, чем даёт бег вперёд за горизонт (≈48):
#   цель, путь к которой карта знает, а луч взять не может (монета над
#   пустым блоком за щелью в 1 тайл, 1-1), не держит учителя на месте.
# - Трубы: вес 0.6, радиус ATTRACT_RADIUS. Непосещённая труба-вход в окне
#   ученика берётся наверняка — встал на неё, дальше правило «вниз»
#   (и это конечный лист с бонусом трубы).
# Взять цель всегда выгоднее, чем стоять рядом (вес < 1). Цели без пути по
# карте потенциала не дают — недостижимое не держит на месте
POTENTIAL = 0.4
POTENTIAL_CAP = 100.0
POTENTIAL_RADIUS = 7.0
WARP_POTENTIAL = 0.6
WARP_POTENTIAL_RADIUS = ATTRACT_RADIUS
# Неконечные листы — в один момент, на горизонте: бонус за время у них
# одинаковый, как если бы лист, обрезанный раньше горизонта, дальше стоял.
# Раньше короткий лист (ветка, отсечённая лучом, или кандидат «сделать
# сейчас») получал бонус за несыгранные кадры: «постоять 4 кадра» стоило на
# 30 очков больше, чем полный план без прогресса, и вместе с потенциалом
# рядом с целью, которую луч не брал, учитель стоял вечно (skybridge x≈66,
# 110 с). Конечные листы (флаг, трубы, смерть) — по своему времени
EQUAL_TIME = True
# Ближе стольких тайлов к флагштоку ранг получает очки флага за высоту подлёта
FLAG_ATTRACT = 12
# На земле подлёт ещё не случился — его очки в ранге с весом: ветка, уже
# летящая к штоку высоко, не должна уступать тем, кто только собирается
FLAG_GROUND = 0.8

# Раньше взятое — чуть дороже: очки на кадре t дают ещё EARLY·(1 − t/горизонт).
# Иначе в неподвижном мире «постоять, потом взять» ровно равно «взять,
# потом постоять», и на ничьей учитель вечно откладывал (стоял у выхода
# из комнаты под монетами)
EARLY = 0.1
# По той же причине «стоять» в плане стоит чуть-чуть, и раньше — дороже:
# на ничьей «сделать сейчас, постоять потом» лучше, чем наоборот. Очки за
# кадр стояния — доли очка: решает только ничьи
IDLE = 0.05

# Разворот (вправо ↔ влево, в том числе через «стоять») стоит FLIP_COST
# очков — и внутри плана, и первым действием против прошлого (оно есть в
# наблюдении ученика). Без этого почти равные оценки дёргали учителя
# туда-обратно в воздухе: 72 смены в минуту при нуле петель
FLIP_COST = 10.0

# Мягкие метки: softmax(Q / TAU)
TAU = 30.0

# Предметы вне окна ученика, но в пределах досягаемости поиска: их очки
# вычитаются — учитель не бегает за тем, чего ученик не видит
HIDDEN_RADIUS = 20
# Видимые предметы впереди (правее, чем игрок уже доходил) идут в оценку со
# скидкой: они и так попадутся по дороге, а те, что позади, уйдут из окна
# ученика и пропадут. Иначе поиск с горизонтом 1.6 с брал монеты впереди
# (ближе) и навсегда оставлял блок «?» и монеты за спиной. Скидка только
# на взятое, штрафа за оставленное нет — недостижимый предмет позади не
# держит учителя на месте. Замер на 6 уровнях: 0.5 — 126.5 тыс. очков и
# 74 % ударов по «?», 0.8 — 129.2 тыс. и 82 %
AHEAD_LOSS = 0.8


def _aligned_steps(frame: int, end: int) -> tuple:
    """Длины шагов сетки по часам мира (ALIGNED_GRID) от решения на кадре frame.

    Решения идут через FRAME_SKIP кадров, поэтому кадр решения по модулю 4
    постоянен (между заставками), и выравнивание — по frame − frame % 4."""
    base = frame - frame % FRAME_SKIP
    steps = []
    elapsed = 0
    zones = [(limit, size) for limit, size in GRID_ZONES[:-1]] + [(end, GRID_ZONES[-1][1])]
    for limit, size in zones:
        while True:
            nxt = elapsed + FRAME_SKIP
            # следующая граница, кратная size по часам мира
            nxt = ((base + nxt + size - 1) // size) * size - base
            if nxt > limit:
                break
            steps.append(nxt - elapsed)
            elapsed = nxt
    return tuple(steps)


# ===============================================================
# КАРТА ПРОХОДИМОСТИ
# ===============================================================


def _arc_patterns(frames: int):
    """Нажатия для траекторий карты проходимости.

    Прыжок со сносом вбок: сначала wait кадров без него, потом hold кадров
    в сторону (None — до приземления). Сход с края: сразу в сторону, hold
    кадров. Возвращает список (прыжок?, направление, h[t]) — h[t] ∈ {−1, 0, 1}."""

    patterns = []

    def horizontal(direction: int, wait: int, hold: Optional[int]) -> np.ndarray:
        h = np.zeros(frames + 1)
        end = frames + 1 if hold is None else min(frames + 1, wait + 1 + hold)
        h[wait + 1:end] = direction
        return h

    for direction in (1, -1):
        for wait in (0, 6, 12, 18, 24, 30):
            for hold in (None, 6, 12, 20):
                patterns.append((True, direction, horizontal(direction, wait, hold)))
        for hold in (None, 8, 16):
            patterns.append((False, direction, horizontal(direction, 0, hold)))
    patterns.append((True, 0, np.zeros(frames + 1)))
    return patterns


class NavField:
    """Путь до цели по одной части уровня (уровень или комната).

    Клетка (tx, ty) «стоячая», если она пустая, а под ней твёрдый тайл;
    игрок 26×32 помещается в клетку. Переходы: шаг в соседнюю стоячую
    клетку (1 тайл), падение с края и прыжки. Прыжки и падения — это
    траектории физики игрока (скорость прыжка, гравитация, бег) по сетке
    стен, как у движка: голова в потолок — вертикальная скорость 0, бок в
    стену — стоп по X, падая на тайл — приземление (_trace). Нажатия — набор
    шаблонов «прыжок, подождать, снос вбок столько-то кадров». Цена
    перехода — кадры / 8, то есть «тайлы бега»: путь до цели = кадры до неё
    при беге.

    Стены (и блоки «?», и пустые блоки) не меняются, поэтому граф строится
    один раз на часть уровня (~0.2 с). Для пустых клеток в воздухе значение —
    лучшее из падения вниз и вбок на тайл за ряд (_air).
    """

    def __init__(self, world: GameWorld, area):
        player = world.player
        self.gravity = float(player.gravity)
        self.half_w = (player.right - player.left) / 2
        self.height = player.top - player.bottom
        speed = float(player.move_speed)
        self.frames_per_tile = TILE / speed

        solid = area.tile_grid.astype(bool)
        self.solid = solid
        self.nx, self.ny = solid.shape
        empty = ~solid
        standable = np.zeros_like(solid)
        standable[:, 1:] = empty[:, 1:] & solid[:, :-1]
        self.empty = empty
        self.standable = standable

        xs, ys = np.nonzero(standable)
        self.cell_x, self.cell_y = xs, ys
        self.n = len(xs)
        self.index = np.full(solid.shape, -1, dtype=np.int64)
        self.index[xs, ys] = np.arange(self.n)
        # Вершина-цель для переходов «пересёк линию флагштока»
        self.goal_node = self.n

        self.goal_x: Optional[float] = None
        self.goal_cells: list[int] = []
        # Арена без флага (ai/levelgen.build_drill): door_x там — правая стена
        # арены, единственная клетка «за ней» — на крыше, снаружи, и путь до
        # цели был бесконечен везде: ни одно действие не приближало ни к чему,
        # оценки плоские, учитель стоял или дрожал «вправо / влево» (замер
        # проверки: до 44 % решений в топтании). Цель такой арены — ближайший
        # видимый несобранный предмет (Planner._context), а когда их нет —
        # самый правый столбец, куда можно дойти от старта (ниже)
        self.goalless = False
        if area is world.main:
            pole = world.geometry.pole
            if pole is not None:
                # Захват: правый край игрока дотянулся до штока (world._check_flagpole)
                self.goal_x = pole.x - (TILE / 2 + 1) - self.half_w
                centers = xs * TILE + TILE / 2
                self.goal_cells = [int(i) for i in np.nonzero(centers >= self.goal_x)[0]]
            else:
                self.goalless = True
        elif area.exit is not None:
            ty = int(round(area.exit.bottom / TILE))
            tx = int(area.exit.left // TILE) - 1
            # Клетка перед устьем; если там не стоять — ближайшая слева в том ряду
            while tx >= 0 and not (0 <= ty < self.ny and self.standable[tx, ty]):
                tx -= 1
            if tx >= 0:
                self.goal_cells = [int(self.index[tx, ty])]

        self._build_graph(float(player.jump_speed), speed)
        if self.goalless:
            self.goal_cells = self._rightmost_reachable(area.spawn)
        self.goal = self.field(self.goal_cells + ([self.goal_node] if self.goal_x is not None else []),
                               goal_air=True)
        self.lost = self.lost_of(self.goal)
        self._item_fields: dict = {}

    @staticmethod
    def lost_of(grid: np.ndarray) -> float:
        """Путь «не знаю куда»: дальше самого дальнего известного."""
        finite = grid[np.isfinite(grid)]
        return (float(finite.max()) if finite.size else 0.0) + LOST_EXTRA

    def _cell_of(self, x: float, bottom: float) -> int:
        """Стоячая клетка под точкой (x, низ игрока): вниз по столбцу; −1 — нет."""
        tx = int(x // TILE)
        if not 0 <= tx < self.nx:
            return -1
        for ty in range(min(int((bottom + 1.0) // TILE), self.ny - 1), -1, -1):
            if self.standable[tx, ty]:
                return int(self.index[tx, ty])
            if not self.empty[tx, ty]:
                break
        return -1

    def _rightmost_reachable(self, spawn: tuple[float, float]) -> list[int]:
        """Клетки самого правого столбца, куда можно дойти от старта."""
        start = self._cell_of(spawn[0], spawn[1] - self.height / 2)
        if start < 0:
            return []
        seen = np.zeros(self.n + 1, dtype=bool)
        seen[start] = True
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in self.forward[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        reached = np.nonzero(seen[:self.n])[0]
        right = self.cell_x[reached].max()
        return [int(i) for i in reached[self.cell_x[reached] == right]]

    # -----------------------------------------------------------

    def _solid(self, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
        inside = (tx >= 0) & (tx < self.nx) & (ty >= 0) & (ty < self.ny)
        out = np.zeros(tx.shape, dtype=bool)
        out[inside] = self.solid[tx[inside], ty[inside]]
        return out

    def _build_graph(self, jump_speed: float, speed: float) -> None:
        n = self.n
        srcs: list[np.ndarray] = []
        dsts: list[np.ndarray] = []
        costs: list[np.ndarray] = []

        # Шаг в соседнюю стоячую клетку
        for side in (1, -1):
            nb_x = self.cell_x + side
            ok = (nb_x >= 0) & (nb_x < self.nx)
            ok[ok] = self.standable[nb_x[ok], self.cell_y[ok]]
            srcs.append(np.nonzero(ok)[0])
            dsts.append(self.index[nb_x[ok], self.cell_y[ok]])
            costs.append(np.ones(int(ok.sum())))

        if n:
            for jump, direction, h in _arc_patterns(ARC_FRAMES):
                src, dst, cost = self._trace(jump, direction, h, jump_speed, speed)
                srcs.append(src)
                dsts.append(dst)
                costs.append(cost)

        src = np.concatenate(srcs) if srcs else np.zeros(0, dtype=np.int64)
        dst = np.concatenate(dsts) if dsts else np.zeros(0, dtype=np.int64)
        cost = np.concatenate(costs) if costs else np.zeros(0)

        # Один переход на пару клеток — самый дешёвый
        order = np.lexsort((cost, dst, src))
        src, dst, cost = src[order], dst[order], cost[order]
        first = np.ones(len(src), dtype=bool)
        first[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
        src, dst, cost = src[first], dst[first], cost[first]
        self.edges = len(src)

        # Обратный граф: Дейкстра идёт от цели
        reverse: list[list[tuple[float, int]]] = [[] for _ in range(n + 1)]
        forward: list[list[int]] = [[] for _ in range(n + 1)]
        for s, d, c in zip(src.tolist(), dst.tolist(), cost.tolist()):
            reverse[d].append((c, s))
            forward[s].append(d)
        self.reverse = reverse
        # Прямой — только для «куда дойти от старта» (арена без флага)
        self.forward = forward

    def _trace(self, jump: bool, direction: int, h: np.ndarray, jump_speed: float, speed: float):
        """Траектория из каждой стоячей клетки разом (векторно по клеткам) —
        по той же схеме, что PhysicsEnginePlatformer для сетки тайлов:
        гравитация, шаг по Y с упором (голова в потолок — скорость 0,
        падая на тайл — приземление), шаг по X с упором в стену.

        Возвращает переходы (откуда, куда, цена): приземление в стоячую
        клетку ниже/выше/сбоку или пересечение линии флагштока."""

        hw, hgt, g, tile = self.half_w, self.height, self.gravity, TILE
        x = (self.cell_x * tile + tile / 2).astype(np.float64)
        b = (self.cell_y * tile).astype(np.float64)
        cells = np.arange(self.n)
        if jump:
            vy = np.full(self.n, float(jump_speed))
        else:
            # Сход с края: стартуем уже за краем клетки, если там пусто
            vy = np.zeros(self.n)
            x = x + direction * (tile / 2 + hw + 1)
            side = self.cell_x + direction
            inside = (side >= 0) & (side < self.nx)
            ok = np.zeros(self.n, dtype=bool)
            ok[inside] = self.empty[side[inside], self.cell_y[inside]] & ~self.standable[side[inside], self.cell_y[inside]]
            x, b, vy, cells = x[ok], b[ok], vy[ok], cells[ok]
        start_offset = 0.0 if jump else (tile / 2 + hw + 1) / speed

        out_src, out_dst, out_cost = [], [], []
        solid = self._solid
        floor = np.floor
        for t in range(1, len(h)):
            if cells.size == 0:
                break
            # --- по Y
            vy = vy - g
            nb = b + vy
            txl = floor((x - hw) / tile).astype(np.int64)
            txr = floor((x + hw - 1e-6) / tile).astype(np.int64)
            up = vy > 0
            row = np.where(up, floor((nb + hgt - 1e-6) / tile), floor(nb / tile)).astype(np.int64)
            hit = solid(txl, row) | solid(txr, row)
            landed = hit & ~up
            bumped = hit & up
            nb = np.where(bumped, row * tile - hgt, nb)
            nb = np.where(landed, (row + 1) * tile, nb)
            vy = np.where(hit, 0.0, vy)
            b = nb

            # --- по X
            if h[t]:
                nx_ = x + h[t] * speed
                edge = floor((nx_ + hw - 1e-6) / tile if h[t] > 0 else (nx_ - hw) / tile).astype(np.int64)
                ryb = floor(b / tile).astype(np.int64)
                ryt = floor((b + hgt - 1e-6) / tile).astype(np.int64)
                wall = solid(edge, ryb) | solid(edge, ryt)
                stop = edge * tile - hw if h[t] > 0 else (edge + 1) * tile + hw
                x = np.where(wall, stop, nx_)

            done = landed.copy()
            cost = t / self.frames_per_tile + (0.25 if jump else start_offset / self.frames_per_tile)

            # Линия флагштока — цель
            if self.goal_x is not None:
                goal = (x >= self.goal_x) & ~done
                if goal.any():
                    out_src.append(cells[goal])
                    out_dst.append(np.full(int(goal.sum()), self.goal_node))
                    out_cost.append(np.full(int(goal.sum()), cost))
                    done |= goal

            if landed.any():
                ly = floor(b[landed] / tile).astype(np.int64)
                lx = [floor(x[landed] / tile).astype(np.int64), txl[landed], txr[landed]]
                dst = np.full(ly.shape, -1, dtype=np.int64)
                for col in lx:
                    ok = (col >= 0) & (col < self.nx) & (ly >= 0) & (ly < self.ny) & (dst < 0)
                    dst[ok] = np.where(self.standable[col[ok], ly[ok]], self.index[col[ok], ly[ok]], -1)
                src = cells[landed]
                good = (dst >= 0) & (dst != src)
                if not jump:
                    # Сход с края — только падение ниже, шаг вбок — обычный шаг
                    good &= ly < self.cell_y[src]
                out_src.append(src[good])
                out_dst.append(dst[good])
                out_cost.append(np.full(int(good.sum()), cost))

            done |= b < rules.DEATH_Y
            keep = ~done
            x, b, vy, cells = x[keep], b[keep], vy[keep], cells[keep]

        if not out_src:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, np.zeros(0)
        return np.concatenate(out_src), np.concatenate(out_dst), np.concatenate(out_cost)

    def field(self, targets, limit: float = math.inf, goal_air: bool = False, touch=()) -> np.ndarray:
        """Путь (в тайлах бега) из каждой клетки до ближайшей из targets.

        limit — дальше не искать (поле притяжения предмета нужно рядом);
        touch — пустые клетки в воздухе, где предмет уже берётся (путь 0)."""

        n = self.n
        dist = [math.inf] * (n + 1)
        heap = []
        for target in targets:
            dist[target] = 0.0
            heap.append((0.0, target))
        heapq.heapify(heap)
        reverse = self.reverse
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist[node] or d > limit:
                continue
            for cost, src in reverse[node]:
                nd = d + cost
                if nd < dist[src]:
                    dist[src] = nd
                    heapq.heappush(heap, (nd, src))

        full = np.full((self.nx, self.ny), np.inf)
        if n:
            full[self.cell_x, self.cell_y] = np.asarray(dist[:n])
        for tx, ty in touch:
            if 0 <= tx < self.nx and 0 <= ty < self.ny and self.empty[tx, ty]:
                full[tx, ty] = 0.0
        return self._air(full, goal_air)

    def _air(self, full: np.ndarray, goal_air: bool = False) -> np.ndarray:
        """Клетки в воздухе: упасть на ряд вниз прямо или на тайл вбок.

        Цена — время, как у переходов: падение с места на первый ряд ~11
        кадров, дальше быстрее (девять рядов — 34 кадра), поэтому ряд вниз
        стоит AIR_DOWN; тайл вбок — не меньше, чем бегом (AIR_SIDE), и за
        него успеваешь опуститься на ряд. Высота сама по себе не «далеко»:
        иначе прыжок к верху флагштока выглядел бы уходом от цели.

        goal_air — поле цели главного уровня: флагшток хватают на любой
        высоте, поэтому пустые клетки за линией захвата — цель (0)."""

        empty, standable = self.empty, self.standable
        if goal_air and self.goal_x is not None:
            columns = np.arange(self.nx) * TILE + TILE / 2 >= self.goal_x
            full[columns[:, None] & empty] = 0.0
        for y in range(1, self.ny):
            below = full[:, y - 1]
            through = np.where(empty[:, y - 1], below + AIR_DOWN, np.inf)
            best = through.copy()
            # Вбок: соседний столбец пуст в этом ряду и в ряду ниже
            side_ok = empty[:, y] & empty[:, y - 1]
            left = np.full(self.nx, np.inf)
            left[1:] = np.where(side_ok[:-1], below[:-1] + AIR_SIDE, np.inf)
            right = np.full(self.nx, np.inf)
            right[:-1] = np.where(side_ok[1:], below[1:] + AIR_SIDE, np.inf)
            best = np.minimum(best, np.minimum(left, right))
            # Вбок на уступ в этом же ряду: ноги уже не ниже его верха. Без
            # этого клетка над ямой рядом с краем была «без пути», хотя сносом
            # в сторону на край садишься (gen 7001: сход с платформы в яме)
            row = full[:, y]
            ledge = np.where(standable[:, y], row, np.inf)
            best[1:] = np.minimum(best[1:], ledge[:-1] + AIR_SIDE)
            best[:-1] = np.minimum(best[:-1], ledge[1:] + AIR_SIDE)
            air = empty[:, y] & ~standable[:, y]
            full[air, y] = np.minimum(full[air, y], best[air])
        return full

    # -----------------------------------------------------------

    def probe(self, x: float, bottom: float, vy: float, on_ground: bool) -> tuple[int, float, int]:
        """Где игрок на карте: (столбец слева, доля до правого, ряд ног).

        Летит вверх — ряд берётся на высоте, до которой он ещё поднимется
        (пока над головой пусто): иначе середина прыжка на уступ выглядела
        бы как «упадёт обратно»."""

        ty = int((bottom + 1.0) // TILE)
        if not on_ground and vy > 0:
            top = bottom + vy * vy / (2 * self.gravity)
            ty_top = int((top + 1.0) // TILE)
            col = int(x // TILE)
            while ty < ty_top and ty + 1 < self.ny and 0 <= col < self.nx and self.empty[col, ty + 1]:
                ty += 1
        ty = min(max(ty, 0), self.ny - 1)
        u = x / TILE - 0.5
        i = math.floor(u)
        return i, u - i, ty

    def read(self, grid: np.ndarray, probe: tuple[int, float, int]) -> float:
        """Значение поля в точке игрока: между двумя соседними столбцами."""

        i, f, ty = probe
        a = grid[i, ty] if 0 <= i < self.nx else math.inf
        b = grid[i + 1, ty] if 0 <= i + 1 < self.nx else math.inf
        if a < math.inf and b < math.inf:
            return float(a + (b - a) * f)
        if a < math.inf:
            return float(a)
        return float(b)

    def item_field(self, key, cells: list[tuple[int, int]], touch=()) -> Optional[np.ndarray]:
        """Поле пути до предмета (клетки, откуда его берут); кэш по ключу.

        touch — клетки ног в воздухе, где предмет уже взят (монета — её клетка
        и под ней, блок — под ним): прыжок за предметом по полю не «удаляется»
        от него. Поле ног в прыжке читается на высоте подъёма (probe), и без
        этого прыжок к монете на 7 рядов выше пола выглядел как уход от неё на
        3.5 тайла — ветку с прыжком луч отсекал, потенциал держал учителя
        рядом (skybridge x≈66, 110 с). Без клеток, откуда брать, поля нет."""

        cached = self._item_fields.get(key, False)
        if cached is not False:
            return cached
        targets = [int(self.index[tx, ty]) for tx, ty in cells
                   if 0 <= tx < self.nx and 0 <= ty < self.ny and self.index[tx, ty] >= 0]
        grid = self.field(targets, limit=ATTRACT_RADIUS + 2, touch=touch) if targets else None
        self._item_fields[key] = grid
        return grid

    def goal_to(self, cells: list[tuple[int, int]]) -> np.ndarray:
        """Поле пути до ближайшей из клеток (цель арены без флага); кэш по набору."""
        key = ("goal",) + tuple(sorted(set(cells)))
        cached = self._item_fields.get(key)
        if cached is None:
            targets = [int(self.index[tx, ty]) for tx, ty in set(cells)
                       if 0 <= tx < self.nx and 0 <= ty < self.ny and self.index[tx, ty] >= 0]
            cached = self._item_fields[key] = self.field(targets)
        return cached

    def jump_column(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Пустые клетки над клетками прыжка за предметом — на высоту прыжка.

        Поле ног в прыжке читается на высоте подъёма (probe), и прыжок с
        клетки, откуда предмет берут, иначе выглядел бы уходом от него на
        несколько тайлов: ветку с прыжком луч отсекал, а стоять у цели было
        не хуже (gen90021, gen90042, gen90057: монеты на ряду блоков над
        платформой, 100 с на месте)."""
        out = []
        for cx, cy in cells:
            for y in range(cy + 1, min(cy + 1 + JUMP_REACH, self.ny)):
                if not self.empty[cx, y]:
                    break
                out.append((cx, y))
        return out

    def cells_below(self, tx: int, ty: int, reach: int = JUMP_REACH) -> list[tuple[int, int]]:
        """Стоячие клетки в столбце tx не ниже reach рядов под (tx, ty),
        с пустым столбцом между ними: отсюда прыжком достаётся (tx, ty)."""

        out = []
        if not 0 <= tx < self.nx:
            return out
        for y in range(min(ty, self.ny - 1), max(ty - reach, 0) - 1, -1):
            if y < 0 or not self.empty[tx, y]:
                break
            if self.standable[tx, y]:
                out.append((tx, y))
                break
        return out


# ===============================================================
# БЫСТРАЯ ПРОВЕРКА МОНЕТ (только на время поиска)
# ===============================================================

# Player.check_objects_collides на каждом кадре проверяет игрока со всеми
# монетами части уровня через arcade.check_for_collision: в сокровищнице
# (180 монет) это половина времени симуляции. Arcade первым делом сравнивает
# квадрат расстояния по X с квадратом суммы «радиусов» и при промахе
# возвращает False. Та же проверка здесь, до вызова, теми же операциями
# над теми же числами, — результат, порядок обхода живого списка (с его
# пропуском соседа после kill()) и побочные эффекты те же, только без
# дорогого вызова для далёких монет. Подменяется на время decide() и только
# если исходный метод — ровно тот, что ниже (иначе поиск идёт по исходному).

_EXPECTED_OBJECTS_COLLIDES = """
if not self.is_alive:
    return
for obj in objects:
    if self.collides_with_sprite(obj):
        if hasattr(obj, "on_collides_with_player"):
            obj.on_collides_with_player(game, self)
"""


def _body(function) -> str:
    try:
        lines = inspect.getsource(function).splitlines()[1:]
    except (OSError, TypeError):
        return ""
    return "\n".join(line.strip() for line in lines if line.strip())


_ORIGINAL_OBJECTS_COLLIDES = Player.check_objects_collides
_FAST_OK = _body(_ORIGINAL_OBJECTS_COLLIDES) == "\n".join(
    line.strip() for line in _EXPECTED_OBJECTS_COLLIDES.splitlines() if line.strip())
FAST_COLLISIONS = True


def _objects_collides_fast(self, game, objects) -> None:
    if not self.is_alive:
        return
    px = self._position[0]
    pw, ph = self._width, self._height
    pr = pw if pw > ph else ph
    for obj in objects:
        ow, oh = obj._width, obj._height
        radius = (pr + (ow if ow > oh else oh)) * 0.71
        dx = px - obj._position[0]
        if dx * dx > radius * radius:
            continue
        if self.collides_with_sprite(obj):
            if hasattr(obj, "on_collides_with_player"):
                obj.on_collides_with_player(game, self)


@contextlib.contextmanager
def _fast_object_collisions():
    if not (FAST_COLLISIONS and _FAST_OK and Player.check_objects_collides is _ORIGINAL_OBJECTS_COLLIDES):
        yield
        return
    Player.check_objects_collides = _objects_collides_fast
    try:
        yield
    finally:
        Player.check_objects_collides = _ORIGINAL_OBJECTS_COLLIDES


# ===============================================================
# ПЛАНИРОВЩИК
# ===============================================================


class _Node:
    __slots__ = ("snap", "group", "actions", "value", "rank", "ground", "terminal", "horizontal", "order",
                 "landed", "inputs", "steps", "points", "early", "dir")

    def __init__(self, snap, group, actions, value, rank, ground, terminal, horizontal, order):
        self.snap = snap
        # Первое действие ветки: по нему считаются оценки действий
        self.group = group
        self.actions = actions
        self.value = value
        self.rank = rank
        self.ground = ground
        self.terminal = terminal
        self.horizontal = horizontal
        self.order = order
        # Лист в воздухе уже досчитан до приземления (_land)
        self.landed = False
        # Покадровые нажатия (у кандидата «прошлый план»; у веток — из actions)
        self.inputs: Optional[list] = None
        # Длины шагов действий у кандидата «сделать сейчас» (у веток — STEPS)
        self.steps: tuple = ()
        # Очки (без взятого вне окна ученика) и надбавка за раннее взятие
        self.points = 0.0
        self.early = 0.0
        # Последнее ненулевое направление по пути (для цены разворота)
        self.dir = 0

    def frames(self, steps: tuple) -> list:
        if self.inputs is not None:
            return self.inputs
        out = []
        for depth, action in enumerate(self.actions):
            out.append(action_inputs(action, True))
            out.extend([action_inputs(action, False)] * (steps[depth] - 1))
        return out


class _Context:
    """Всё, что поиск считает один раз на решение."""

    __slots__ = ("nav", "goal", "lost", "offset", "k", "health", "items", "hidden", "warps", "area", "pole", "steps",
                 "horizon", "frames0")


class Planner:

    name = "Планировщик"

    def __init__(self):
        self._navs: dict[str, NavField] = {}
        # Мир, для которого построены карты. Ссылка, а не id(): новый мир
        # эпизода может получить id старого, и карты были бы от чужого уровня
        self._world: Optional[GameWorld] = None
        self.reset()

    def reset(self, world: Optional[GameWorld] = None) -> None:
        self._frame = 0
        self._action = STAND
        self.last_source = "plan"
        self.last_q = np.full(N_ACTIONS, -np.inf)
        self.last_probs = np.zeros(N_ACTIONS)
        self.last_probs[STAND] = 1.0
        self.last_plan: tuple = ()
        self.last_frames = 0
        self.last_seconds = 0.0
        if world is not None:
            self._bind(world)

    def overlay(self) -> None:
        return None

    # -----------------------------------------------------------

    def act(self, world: GameWorld) -> Inputs:
        if world.done or world.cutscene:
            self._frame = 0
            return Inputs()

        first = self._frame % FRAME_SKIP == 0
        if first:
            self._action = self.decide(world, self._action)
        self._frame += 1
        return action_inputs(self._action, first)

    # -----------------------------------------------------------

    def _bind(self, world: GameWorld) -> None:
        """Карты проходимости — свои у каждого мира (уровня)."""
        if world is not self._world:
            self._world = world
            self._navs = {}

    def nav(self, world: GameWorld, area=None) -> NavField:
        area = area or world.area
        nav = self._navs.get(area.key)
        if nav is None:
            nav = self._navs[area.key] = NavField(world, area)
        return nav

    def _exit_offset(self, world: GameWorld, area) -> float:
        """В комнате: путь от трубы, из которой вылезешь, до флага."""
        if area is world.main or area.exit is None:
            return 0.0
        warp = area.exit.warp
        main = self.nav(world, world.main)
        probe = main.probe(warp.exit_x, warp.exit_top, 0.0, True)
        value = main.read(main.goal, probe)
        return value if value < math.inf else main.lost

    # -----------------------------------------------------------

    def decide(self, world: GameWorld, last_action: int) -> int:
        started = time.perf_counter()
        self._bind(world)
        self.last_frames = 0

        if world.done or world.cutscene:
            self._label({STAND: 0.0}, {}, STAND)
            self.last_source = "rule"
            self.last_seconds = time.perf_counter() - started
            return STAND

        # Правило: стоит на трубе, за которой ещё не был, — вниз
        warp = world.warp_under_player()
        standing = world.on_ground or world.physics_engine.can_jump()
        if (warp is not None and warp.room not in world.rooms_visited and standing
                and world._knockback == 0):
            self._label({DOWN: 0.0}, {}, DOWN)
            self.last_source = "rule"
            self.last_plan = (DOWN,)
            self.last_seconds = time.perf_counter() - started
            return DOWN

        root = world.snapshot()
        try:
            with _fast_object_collisions():
                action = self._search(world, root, last_action)
        finally:
            world.restore(root)
        self.last_source = "plan"
        self.last_seconds = time.perf_counter() - started
        return action

    # -----------------------------------------------------------

    def _context(self, world: GameWorld) -> _Context:
        ctx = _Context()
        area = world.area
        ctx.area = area
        ctx.pole = world.geometry.pole if area is world.main else None
        near = ctx.pole is not None and abs(ctx.pole.x - world.player.center_x) < POLE_NEAR * TILE
        if ALIGNED_GRID:
            ctx.steps = _aligned_steps(world.frame, POLE_GRID_END if near else GRID_ZONES[-1][0])
        else:
            ctx.steps = POLE_STEPS if near else STEPS
        ctx.horizon = sum(ctx.steps)
        ctx.nav = nav = self.nav(world)
        ctx.offset = self._exit_offset(world, area)
        player = world.player
        ctx.health = player.health
        ctx.frames0 = world._frames_left

        # Окно ученика (ai.features): что он видит — за тем и можно идти
        cx = int(player.center_x // TILE)
        feet = int(player.bottom // TILE)
        x0, x1 = cx - features.GRID_BACK, cx + features.GRID_AHEAD
        y0, y1 = feet - features.GRID_BELOW, feet + features.GRID_ABOVE

        def visible(tx: int, ty: int) -> bool:
            return x0 <= tx <= x1 and y0 <= ty <= y1

        # «Впереди» — правее самого дальнего места, где игрок уже был
        # (world.max_x, в наблюдении ученика это world.progress), а не
        # правее игрока: от игрока отсчёт сдвигается с каждым шагом, и
        # монета на границе столбца то дешевела, то дорожала — учитель
        # качался влево-вправо (замер: 700 смен в минуту в комнате goombas
        # при другом наборе шагов). В комнатах скидки нет: всё видно сразу
        reach = int(world.max_x // TILE) if area is world.main else None

        def ahead(tx: int) -> bool:
            return reach is not None and tx > reach

        items = []
        hidden = []
        # Арена без флага: клетки, откуда берутся видимые несобранные предметы
        targets: list[tuple[int, int]] = []
        for obj in area.objects:
            if not isinstance(obj, Coin):
                continue
            tx, ty = int(obj.center_x // TILE), int(obj.center_y // TILE)
            if abs(tx - cx) > HIDDEN_RADIUS:
                continue
            if visible(tx, ty):
                if nav.standable[tx, ty]:
                    # Монета лежит на тайле: берут, встав на её клетку. Прыжок
                    # из соседнего столбца её не заденет (надо подняться выше и
                    # сесть сверху — это дуги карты), а поле «отсюда прыжком»
                    # держало учителя под ней (gen68038, монета на пустом блоке)
                    cells = [(tx, ty)]
                    touch = [(tx, ty)]
                else:
                    cells = []
                    for side in (0, -1, 1):
                        cells += nav.cells_below(tx + side, ty)
                    touch = nav.jump_column(cells) + [(tx, ty), (tx, ty - 1)]
                grid = nav.item_field(("coin", tx, ty), cells, touch=touch)
                targets += cells
                # Впереди — со скидкой и в оценке, и в притяжении
                keep = 1.0 - AHEAD_LOSS if ahead(tx) else 1.0
                if grid is not None:
                    items.append(("coin", obj, grid, keep * COIN_ATTRACT, keep * rules.COIN_POINTS))
                if ahead(tx):
                    hidden.append(("coin", obj, 1, AHEAD_LOSS * rules.COIN_POINTS))
            else:
                hidden.append(("coin", obj, 1, rules.COIN_POINTS))
        for block in area.blocks:
            if block.state <= 0:
                continue
            tx, ty = int(block.center_x // TILE), int(block.center_y // TILE)
            if abs(tx - cx) > HIDDEN_RADIUS:
                continue
            if visible(tx, ty):
                cells = nav.cells_below(tx, ty - 1, JUMP_REACH - 1)
                grid = nav.item_field(("block", tx, ty), cells, touch=((tx, ty - 1),))
                targets += cells
                keep = 1.0 - AHEAD_LOSS if ahead(tx) else 1.0
                if grid is not None:
                    items.append(("block", block, grid, keep * BLOCK_ATTRACT, keep * rules.BLOCK_POINTS))
                if ahead(tx):
                    hidden.append(("block", block, block.state, AHEAD_LOSS * rules.BLOCK_POINTS))
            else:
                hidden.append(("block", block, block.state, rules.BLOCK_POINTS))
        # Враги вне окна: очки за них тоже не в счёт (по первому засчитанному
        # удару; цепочку прыжков не пересчитываем)
        for enemy in area.enemies:
            if not enemy.is_alive or getattr(enemy, "scored", False):
                continue
            tx, ty = int(enemy.center_x // TILE), int(enemy.center_y // TILE)
            if abs(tx - cx) <= HIDDEN_RADIUS and not visible(tx, ty):
                hidden.append(("enemy", enemy, int(getattr(enemy, "shell_scored", False)), rules.ENEMY_POINTS))
        ctx.items = items
        ctx.hidden = hidden

        # Цель: флаг или выход из комнаты; на арене без флага — ближайший
        # видимый несобранный предмет. Набор предметов — из состояния мира,
        # и поле до него — функция состояния, как и всё остальное
        ctx.goal, ctx.lost = nav.goal, nav.lost
        if nav.goalless and targets:
            goal = nav.goal_to(targets)
            probe = nav.probe(player.center_x, player.bottom, player.change_y, world.on_ground)
            if nav.read(goal, probe) < math.inf:
                ctx.goal, ctx.lost = goal, nav.lost_of(goal)

        # Спешка: времени впритык на путь до флага — прогресс дороже
        probe = nav.probe(player.center_x, player.bottom, player.change_y, world.on_ground)
        dist = nav.read(ctx.goal, probe)
        dist = (dist if dist < math.inf else ctx.lost) + ctx.offset
        need = dist * nav.frames_per_tile * rules.FIXED_DT
        urgency = URGENT * need / max(world.time_left, 1e-3)
        ctx.k = K_PROGRESS * float(np.clip(urgency, 1.0, URGENT_MAX))

        warps = []
        for warp in area.warps:
            if warp.room in world.rooms_visited:
                continue
            ty = int(round(warp.top / TILE))
            cells = [(int((warp.x + side) // TILE), ty) for side in (-TILE / 2, TILE / 2)]
            if not any(visible(tx, ty) for tx, ty in cells):
                continue
            grid = nav.item_field(("warp", warp.room), cells)
            warps.append((warp, grid))
        ctx.warps = warps
        return ctx

    def _evaluate(self, world: GameWorld, ctx: _Context) -> tuple[float, float, bool, float]:
        """(ценность, ранг для луча, лист ли это, очки без взятого вне окна)."""

        phase = world.phase
        score = float(world.score)
        frames_left = world._frames_left

        # Предметы вне окна ученика не в счёт, впереди — со скидкой AHEAD_LOSS
        for kind, obj, amount, loss in ctx.hidden:
            if kind == "coin":
                if not obj.sprite_lists:
                    score -= loss
            elif kind == "block":
                if obj.state < amount:
                    score -= loss * (amount - obj.state)
            elif getattr(obj, "scored", False) or int(getattr(obj, "shell_scored", False)) > amount:
                score -= loss

        points = score
        if phase == DEAD:
            value = score - DEATH
            return value, value, True, points
        if phase in (FLAGPOLE, TO_CASTLE, COMPLETE):
            if phase != COMPLETE:
                seconds = int(frames_left * rules.FIXED_DT + 1e-9)
                score += seconds * rules.TIME_BONUS_PER_SECOND
            return score, score, True, points

        player = world.player
        lost = (ctx.health - player.health) * HIT
        base = score + frames_left * TIME_COST - lost

        if phase == PIPE_SIDE:
            # Ушёл в трубу выхода из комнаты: дальше путь от трубы в уровне
            value = base - ctx.k * self._exit_offset(world, world.area)
            return value, value, True, points
        if phase == PIPE_DOWN or world.area is not ctx.area:
            value = base - ctx.k * self._exit_offset(world, world.area)
            return value, value, True, points

        nav = ctx.nav
        on_ground = world.on_ground
        probe = nav.probe(player.center_x, player.bottom, player.change_y, on_ground)
        dist = nav.read(ctx.goal, probe)
        if dist == math.inf:
            dist = ctx.lost
        value = base - ctx.k * (dist + ctx.offset)

        # Встал на непосещённую трубу-вход — дальше правило «вниз»: лист
        # конечный, бонус трубы засчитан сразу. Раньше бонус получал только
        # лист, стоящий на трубе ровно в конце горизонта, и такой план
        # находился не на каждом решении (петля «вправо, влево, стоять» у
        # трубы позади: gen67002, gen67003)
        if on_ground and ctx.warps:
            under = world.warp_under_player()
            if under is not None and under.room not in world.rooms_visited and \
                    any(warp is under for warp, _ in ctx.warps):
                value += WARP_BONUS
                return value, value, True, points

        # Неконечные листы сравниваются в один момент — на горизонте (EQUAL_TIME)
        if EQUAL_TIME:
            value += (ctx.frames0 - ctx.horizon - frames_left) * TIME_COST

        # Потенциал (POTENTIAL): видимые достижимые цели — часть их ценности
        # уже в листе, по близости пути до них; притяжение (attract) — только
        # в ранг луча, оптимистично
        attract = 0.0
        potential = 0.0
        for warp, grid in ctx.warps:
            if warp.room in world.rooms_visited or grid is None:
                continue
            d = nav.read(grid, probe)
            if d < ATTRACT_RADIUS:
                attract = max(attract, WARP_BONUS * (1.0 - d / ATTRACT_RADIUS))
            if d < WARP_POTENTIAL_RADIUS:
                potential = max(potential, WARP_POTENTIAL * WARP_BONUS * (1.0 - d / WARP_POTENTIAL_RADIUS))
        for kind, obj, grid, worth, points_each in ctx.items:
            if kind == "coin":
                if not obj.sprite_lists:
                    continue
                left = 1
            else:
                left = obj.state
                if left <= 0:
                    continue
            d = nav.read(grid, probe)
            if d < ATTRACT_RADIUS:
                attract = max(attract, worth * (1.0 - d / ATTRACT_RADIUS))
            if d < POTENTIAL_RADIUS:
                worth_now = POTENTIAL * min(points_each * left, POTENTIAL_CAP)
                potential = max(potential, worth_now * (1.0 - d / POTENTIAL_RADIUS))
        value += potential
        attract = max(0.0, attract - potential)

        # У флагштока ветка, которая долетит до штока высоко, не должна
        # отсекаться раньше захвата: ранг получает очки флага за высоту, на
        # которой игрок пересечёт линию захвата, если бежать вправо (в воздухе —
        # по баллистике, на земле — с лучшим прыжком). Высота сама по себе
        # не в счёт: прыжок сразу с начала лестницы высок, но до штока не долетит
        pole = ctx.pole
        if pole is not None and abs(pole.x - player.center_x) < FLAG_ATTRACT * TILE:
            attract += self._flag_potential(player, on_ground, pole, nav)
        return value, value + attract, False, points

    @staticmethod
    def _flag_potential(player, on_ground: bool, pole, nav: NavField) -> float:
        """Очки флага, если бежать вправо: в воздухе — по баллистике, на
        земле — с лучшего места прыжка до края опоры (дальше она кончится)."""
        goal_x = nav.goal_x
        speed, g, jump = player.move_speed, player.gravity, player.jump_speed
        x = player.center_x
        if goal_x - x <= 0:
            return 0.0
        if on_ground:
            # Край опоры: стоячие клетки этого ряда подряд вправо
            tx, ty = int(x // TILE), int((player.bottom + 1.0) // TILE)
            last = tx
            while last + 1 < nav.nx and 0 <= ty < nav.ny and nav.standable[last + 1, ty] and last - tx < 16:
                last += 1
            # Край с запасом в полтайла: решения идут шагами по 16 px, и
            # оттолкнуться с самого края удаётся не всегда
            edge = (last + 1) * TILE + nav.half_w - TILE / 2
            # Лучше всего оттолкнуться за вершину прыжка до штока
            takeoff = min(max(goal_x - speed * jump / g, x), edge)
            t = (goal_x - takeoff) / speed
            rise = jump * (t - 1) - g * t * (t - 1) / 2
            weight = FLAG_GROUND
        else:
            t = (goal_x - x) / speed
            rise = player.change_y * t - g * t * (t + 1) / 2
            weight = 1.0
        fraction = min(max((player.bottom + rise - pole.bottom) / pole.height, 0.0), 1.0)
        return weight * float(rules.flag_points(fraction))

    def _run(self, world: GameWorld, action: int, frames: int) -> None:
        first = action_inputs(action, True)
        rest = action_inputs(action, False)
        step = world.step
        n = 0
        while n < frames and world.phase == PLAYING:
            step(first if n == 0 else rest)
            n += 1
        self.last_frames += n

    @staticmethod
    def _fingerprint(world: GameWorld) -> tuple:
        p = world.player
        return (world.area.key, world.phase, p.center_x, p.center_y, p.change_x, p.change_y, p.scale_x,
                world.score, p.health, world._knockback, world.on_ground, len(world.objects), len(world.enemies))

    def _node(self, world: GameWorld, ctx: _Context, group: int, actions: tuple, parent_points: float,
              parent_early: float, parent_dir: int, t_end: int, order: int) -> _Node:
        """Узел из текущего состояния мира: снимок, оценка и поправки пути —
        надбавка за раннее взятие, цена стояния и разворотов."""
        value, rank, terminal, points = self._evaluate(world, ctx)
        weight = max(0.0, 1.0 - t_end / ctx.horizon)
        early = parent_early + EARLY * max(0.0, points - parent_points) * weight
        action = actions[-1]
        if action == STAND:
            early -= IDLE * weight * ctx.steps[min(len(actions), len(ctx.steps)) - 1]
        h = HORIZONTAL.get(action, 0)
        if h and parent_dir and h != parent_dir:
            early -= FLIP_COST
        node = _Node(world.snapshot(), group, actions, value + early, rank + early, world.on_ground, terminal,
                     h, order)
        node.points = points
        node.early = early
        node.dir = h or parent_dir
        return node

    def _search(self, world: GameWorld, root, last_action: int) -> int:
        ctx = self._context(world)
        steps = ctx.steps
        order = 0

        # --- корень: все 6 действий, дубли прыжка — к действию без прыжка
        duplicate: dict[int, int] = {}
        seen: dict[tuple, _Node] = {}
        frontier: list[_Node] = []
        leaves: list[_Node] = []
        root_points = float(world.score)
        last_dir = HORIZONTAL.get(int(last_action), 0)
        # Действия «на месте»: игрок после них там же, где был (стоять на
        # земле, бежать в стену) — см. _shifted
        still = self._still(world)
        idle: set[int] = set()
        for action in MOVES:
            world.restore(root)
            self._run(world, action, steps[0])
            if self._still(world) == still:
                idle.add(action)
            fp = self._fingerprint(world)
            if fp in seen:
                duplicate[action] = seen[fp].group
                continue
            node = self._node(world, ctx, action, (action,), root_points, 0.0, last_dir, steps[0], order)
            order += 1
            seen[fp] = node
            (leaves if node.terminal else frontier).append(node)

        # --- дальше: луч. Одинаковые состояния сливаются только внутри
        # одного первого действия: «постоять, потом взять» и «взять, потом
        # постоять» — разные ответы, и оценку каждый получает свою
        t_end = steps[0]
        for depth in range(1, len(steps)):
            step = steps[depth]
            t_end += step
            kids: list[_Node] = []
            seen = {}
            for node in frontier:
                if node.ground:
                    actions = GROUND_EARLY if depth < STAND_UNTIL else GROUND_LATE
                else:
                    actions = AIR
                for action in actions:
                    world.restore(node.snap)
                    self._run(world, action, step)
                    fp = (node.group,) + self._fingerprint(world)
                    if fp in seen:
                        continue
                    kid = self._node(world, ctx, node.group, node.actions + (action,), node.points, node.early,
                                     node.dir, t_end, order)
                    order += 1
                    seen[fp] = kid
                    (leaves if kid.terminal else kids).append(kid)
            frontier = self._select(kids, depth)
            chosen = {id(k) for k in frontier}
            leaves.extend(k for k in kids if id(k) not in chosen)
            if not frontier:
                break
        leaves.extend(frontier)
        beam_frames = self.last_frames

        # --- оценки первых действий: лучший лист, не падающий в пропасть
        best_leaf = {group: self._best_landed(world, ctx, [leaf for leaf in leaves if leaf.group == group])
                     for group in sorted({leaf.group for leaf in leaves})}
        land_frames = self.last_frames - beam_frames

        # --- «сделать сейчас»: лучший план, начатый с действия «на месте»
        # (постоять, бежать в стену), — без этого начала. В неподвижном мире
        # он не хуже, но луч сравнивает ветки в разных фазах прыжка, и
        # сдвинутый на 4 кадра план из луча выпадал: «постоять» или «в
        # стену» выигрывало на долях очка, а на следующем решении мир тот же —
        # и учитель стоял вечно (замер проверки без тёплого старта прошлой
        # версии: 90 % решений в топтании). Кандидаты считаются из того же
        # состояния — ответ остаётся функцией состояния и прошлого действия
        # Первый круг: у действий «на месте» — без них в начале, у остальных
        # (SHIFT_ALL) — сдвиг на одно решение. Следующие круги: если лучшим у
        # действия «на месте» стал такой кандидат, а он снова начинается с
        # «на месте», — и его без этого начала (иначе сдвиг «влево, постоять,
        # прыжок» давал «постоять, прыжок», и учитель стоял: gen 7005)
        pending = sorted(best_leaf)
        for round_ in range(NOW_ROUNDS):
            changed = []
            for group in pending:
                leaf = best_leaf[group]
                wait = 0
                while wait < len(leaf.actions) and leaf.actions[wait] in idle:
                    wait += 1
                if wait == 0:
                    if not SHIFT_ALL or round_ > 0:
                        continue
                    wait = 1
                node = self._shifted(world, root, ctx, leaf, wait, duplicate, order, root_points, last_dir)
                if node is None:
                    continue
                order += 1
                held = best_leaf.get(node.group)
                if held is None or (node.value, -node.order) > (held.value, -held.order):
                    best_leaf[node.group] = node
                    changed.append(node.group)
            pending = sorted(set(changed) & idle)
            if not pending:
                break
        now_frames = self.last_frames - beam_frames - land_frames
        q = {group: leaf.value for group, leaf in best_leaf.items()}

        # Кадры симуляции по частям поиска: луч, «сделать сейчас», досчёт до земли (без кандидатов)
        self.last_breakdown = (beam_frames, now_frames, land_frames, order)
        # Для отладки и панели: лучший лист каждого первого действия
        self.last_leaves = {g: (round(n.value, 1), n.actions) for g, n in best_leaf.items()}
        chosen = self._label(q, duplicate, last_action)
        leaf = best_leaf.get(chosen)
        self.last_plan = leaf.actions if leaf is not None else (chosen,)
        return chosen

    def _best_landed(self, world: GameWorld, ctx: _Context, mine: list[_Node]) -> _Node:
        """Лучший лист, досчитанный до земли. Досчитываем, пока лучший в
        воздухе (не больше LAND_TRIES): раньше досчитывались три первых, и
        четвёртый, недосчитанный, летящий в яму, мог стать лучшим."""
        mine.sort(key=lambda n: (-n.value, n.order))
        for _ in range(LAND_TRIES):
            top = mine[0]
            if top.terminal or top.ground or top.landed:
                break
            self._land(world, top, ctx)
            mine.sort(key=lambda n: (-n.value, n.order))
        return mine[0]

    @staticmethod
    def _still(world: GameWorld) -> tuple:
        """Что должно не измениться у действия «на месте»."""
        p = world.player
        return (world.area.key, world.phase, p.center_x, p.bottom, p.change_y, world.score, p.health,
                world.on_ground)

    def _shifted(self, world: GameWorld, root, ctx: _Context, leaf: _Node, wait: int, duplicate: dict[int, int],
                 order: int, root_points: float, last_dir: int) -> Optional[_Node]:
        """Кандидат «сделать сейчас»: нажатия листа без первых wait шагов,
        сразу из корня.

        Если выброшенные шаги — «на месте», лист на том же месте траектории,
        только на skip кадров раньше: в неподвижном мире итог тот же — не
        хуже (при EQUAL_TIME бонус за время у листов один, ничью решают
        поправки пути: взятое раньше дороже, стоять раньше — дороже). Хвост до
        полного горизонта не добавляем: грубое продолжение («держать
        последнее действие» или «стоять») уводило с края платформы в яму, и
        «постоять» снова выигрывало на обрезке горизонта (skybridge, x≈42).
        Сдвиг прочих шагов — другая траектория, её оценка — по симуляции."""

        if leaf.inputs is not None:
            actions, steps, frames = leaf.actions, leaf.steps, leaf.inputs
        else:
            actions, steps, frames = leaf.actions, tuple(ctx.steps[:len(leaf.actions)]), leaf.frames(ctx.steps)
        if wait >= len(actions):
            return None
        frames = frames[sum(steps[:wait]):]

        # Группа — действие, которое дают первые 4 кадра
        first = frames[0]
        group = BY_INPUTS[(int(first.right) - int(first.left), bool(first.jump))]
        group = duplicate.get(group, group)

        world.restore(root)
        early = self._replay(world, ctx, frames, (0.0, world.score, last_dir, 0))[0]
        # Поправки пути уже посчитаны покадрово: t_end = горизонт, чтобы _node их не добавлял
        node = self._node(world, ctx, group, actions[wait:], root_points, early, 0, ctx.horizon, order)
        node.horizontal = HORIZONTAL.get(actions[-1], 0)
        node.inputs = frames
        node.steps = steps[wait:]
        # Досчитать до земли сразу: он сравнивается с досчитанными листами
        if not node.terminal and not node.ground:
            self._land(world, node, ctx)
        return node

    def _replay(self, world: GameWorld, ctx: _Context, frames: list, track: tuple) -> tuple:
        """Покадровые нажатия с поправками пути, как у веток: раннее взятие,
        стояние, развороты. track — (надбавка, очки, направление, кадров от корня)."""
        early, score, direction, n = track
        idle = Inputs()
        done = 0
        for inputs in frames:
            if world.phase != PLAYING:
                break
            world.step(inputs)
            n += 1
            done += 1
            weight = max(0.0, 1.0 - n / ctx.horizon)
            if world.score > score:
                early += EARLY * (world.score - score) * weight
                score = world.score
            if inputs == idle:
                early -= IDLE * weight
            h = int(inputs.right) - int(inputs.left)
            if h and direction and h != direction:
                early -= FLIP_COST
            direction = h or direction
        self.last_frames += done
        return early, score, direction, n

    def _select(self, kids: list[_Node], depth: int) -> list[_Node]:
        if not kids:
            return []
        by_rank = sorted(kids, key=lambda n: (-n.rank, n.order))
        keep: list[_Node] = []
        if depth < DIVERSE_DEPTH:
            taken = set()
            for node in by_rank:
                if node.group not in taken:
                    taken.add(node.group)
                    keep.append(node)
        else:
            count: Counter = Counter()
            for node in by_rank:
                if len(keep) >= BEAM:
                    break
                if count[node.group] >= PER_GROUP:
                    continue
                count[node.group] += 1
                keep.append(node)
        best = max(kids, key=lambda n: (n.value, -n.order))
        if all(best is not node for node in keep):
            keep.append(best)
        return keep

    def _land(self, world: GameWorld, leaf: _Node, ctx: _Context) -> None:
        """Лист в воздухе: держим его горизонталь без прыжка до земли.
        Разбился — ценность смерти; получил удар — минус HIT; упал туда,
        откуда до цели не дойти (яма без дна в пределах досчёта, тупик), —
        ценность места падения. Последнее — поправка: лист, сошедший с края
        ровно на последнем кадре, читался по клетке края (ноги ещё на её
        высоте), и «постоять, потом добежать до края» было на пару очков
        лучше, чем «добежать и прыгнуть» (проверка без тёплого старта: gen
        7001 стоял 95 с перед ямой)."""

        world.restore(leaf.snap)
        health = world.player.health
        h = leaf.horizontal
        inputs = Inputs(right=h > 0, left=h < 0)
        n = 0
        while n < LAND_FRAMES and world.phase == PLAYING and not world.on_ground:
            world.step(inputs)
            n += 1
        self.last_frames += n
        if world.phase == DEAD:
            leaf.value = self._evaluate(world, ctx)[0] + leaf.early
            leaf.landed = True
            return
        if world.player.health < health:
            leaf.value -= HIT * (health - world.player.health)
        if world.phase == PLAYING and world.area is ctx.area:
            player = world.player
            probe = ctx.nav.probe(player.center_x, player.bottom, player.change_y, world.on_ground)
            if ctx.nav.read(ctx.goal, probe) == math.inf:
                # Кадры досчёта — не время плана: их бонус за время возвращаем
                # (при EQUAL_TIME время неконечного листа и так одно — горизонт).
                # Ценность места приземления всегда (не только «без пути»)
                # проверена: хуже — держать горизонталь до земли грубее, чем
                # оценка клетки в воздухе (gen 7001, 1-1, treasury)
                fallen = self._evaluate(world, ctx)[0] + leaf.early + (0.0 if EQUAL_TIME else n * TIME_COST)
                leaf.value = min(leaf.value, fallen)
        leaf.landed = True

    # -----------------------------------------------------------

    def _label(self, q: dict[int, float], duplicate: dict[int, int], last_action: int) -> int:
        """last_q (относительно лучшего), last_probs и выбранное действие."""

        valid = sorted(q)
        values = np.array([q[a] for a in valid], dtype=np.float64)
        top = values.max()

        # Цена разворота против прошлого действия уже в оценках (FLIP_COST)
        scores = values
        chosen = valid[int(np.argmax(scores))]

        z = (scores - scores.max()) / TAU
        p = np.exp(z)
        p /= p.sum()

        self.last_q = np.full(N_ACTIONS, -np.inf)
        self.last_probs = np.zeros(N_ACTIONS)
        for a, value, prob in zip(valid, values, p):
            self.last_q[a] = value - top
            self.last_probs[a] = prob
        for dup, canon in duplicate.items():
            if canon in q:
                self.last_q[dup] = q[canon] - top
        return chosen

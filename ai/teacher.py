"""Учитель для подражания: нынешняя модель ведёт по уровню, правила — собирают.

Обучение с подкреплением под награду collect само не нашло «встать под
блок «?» и прыгать на месте» и «нажать вниз на трубе»: дообучение модели
it4a за 150–200 тыс. шагов держало удары по «?» на 6–7 % и почти не
заходило в тайники, а тренировочные арены без флага за 100 обновлений
этого не изменили. Учитель показывает эти навыки явно, а новая модель
сначала учится ему подражать (ai/imitate.py), потом дообучается PPO.

Решение — раз в 4 кадра, как у модели. Правила включаются, только когда
безопасно: на земле, рядом нет врага, до цели ровный пол без ям. Во всём
остальном (ямы, враги, лестницы, флаг) решает модель — она это умеет.

    teacher = Teacher(DecisionModel.load("ai/models/it4a_best.pt"))
    teacher.reset(world)
    while not world.done:
        world.step(teacher.act(world))
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from ai.decide import DecisionModel
from ai.env import FRAME_SKIP, action_inputs
from core import rules
from core.world import GameWorld, Inputs
from Objects.Coin import Coin

TILE = rules.TILE

# Индексы ai.features.ACTIONS
STAND, RIGHT, LEFT, JUMP, RIGHT_JUMP, LEFT_JUMP, DOWN = range(7)

ALIGN = 8            # px: голова целиком под блоком — удар достаётся ему
BLOCK_AHEAD = 7      # тайлов вперёд, за которыми учитель пойдёт к блоку
BLOCK_BEHIND = 3     # и назад
WARP_AHEAD = 10      # тайлов вперёд до трубы-входа
ENEMY_CLEAR = 3.5    # тайлов: ближе враг — правила молчат, решает модель
GIVE_UP = 40         # решений на одну цель, потом цель бросается
STUCK = 50           # решений почти на месте без очков — тупик (дольше GIVE_UP:
                     # сначала бросается недостижимая цель, потом — выход из тупика)
# Выход из тупика: назад, прыжок вперёд, вперёд. В «хранилище» сокровищницы
# модель бесконечно прыгала вправо в стену под крышей из блоков
ESCAPE = (LEFT,) * 8 + (RIGHT_JUMP,) + (RIGHT,) * 6
ESCAPE_ROOM = (RIGHT_JUMP,) + (RIGHT,) * 8


class Teacher:

    name = "Учитель (модель + правила сбора)"

    def __init__(self, decisions: DecisionModel):
        self.decisions = decisions
        self.reset()

    def reset(self, world: Optional[GameWorld] = None) -> None:
        self._frame = 0
        self._action = STAND
        self._target = None
        self._target_steps = 0
        # Бюджет решений на цель — свой у каждой и не обнуляется при смене:
        # два правила, перехватывавшие друг у друга цель каждое решение,
        # иначе никогда не доходили до GIVE_UP
        self._key_steps: Counter = Counter()
        self._stomp_hold = False
        self._banned: set = set()
        self._history: list[tuple[float, int, str]] = []
        self._escape: list[int] = []
        self.last_source = "model"

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

    def decide(self, world: GameWorld, last_action: int) -> int:
        if self._stuck(world):
            # В комнате — сразу прыжок вправо: через бортик ряда блоков
            self._escape = list(ESCAPE_ROOM if world.in_room else ESCAPE)
            self._history.clear()
        if self._escape:
            action = self._escape[0]
            # Выход из тупика вслепую уводил с платформы в яму: шаг в
            # сторону без пола под следующим столбцом — отмена
            if self._pit_ahead(world, action):
                self._escape = []
            else:
                self.last_source = "escape"
                return self._escape.pop(0)

        action = self._rules(world)
        if action is not None:
            self.last_source = "rules"
            return action
        self.last_source = "model"
        answer = self.decisions.predict(world, last_action=last_action, deterministic=True)
        return int(answer["answers"]["move"]["index"])

    def _pit_ahead(self, world: GameWorld, action: int) -> bool:
        side = {LEFT: -1, LEFT_JUMP: -1, RIGHT: 1}.get(action, 0)
        if side == 0 or not (world.on_ground or world.physics_engine.can_jump()):
            return False
        player = world.player
        feet = int((player.bottom + 1) // TILE)
        tx = int((player.center_x + side * TILE) // TILE)
        return not any(world.solid_at(tx, ty) for ty in range(feet - 1, max(feet - 8, -1), -1))

    def _stuck(self, world: GameWorld) -> bool:
        player = world.player
        self._history.append((player.center_x, world.score, world.area.key))
        self._history = self._history[-STUCK:]
        if len(self._history) < STUCK:
            return False
        xs = [x for x, _, _ in self._history]
        same = len({(score, area) for _, score, area in self._history}) == 1
        return same and max(xs) - min(xs) < 24

    # -----------------------------------------------------------
    # ПРАВИЛА
    # -----------------------------------------------------------

    def _rules(self, world: GameWorld) -> Optional[int]:
        # Опора — и за кадр до приземления: решение раз в 4 кадра, и цикл
        # «приземлился — сразу прыгнул» может ни разу не попасть на землю
        grounded = world.on_ground or world.physics_engine.can_jump()

        # На трубе-входе, за которой ещё не был, — вниз
        warp = world.warp_under_player()
        if warp is not None and grounded and warp.room not in world.rooms_visited:
            return DOWN

        # Враг идёт навстречу по тому же полу — прыжок на месте: пока Марио
        # в воздухе, враг подходит под него, и приземление — на голову
        if self._stomp_hold:
            # После прыжка на врага — стоять в воздухе, пока не приземлился;
            # раньше 12 решений «стоять» вслепую держали и на земле под ударом
            if not grounded:
                return STAND
            self._stomp_hold = False
        if grounded and self._stomp(world):
            self._stomp_hold = True
            return JUMP

        if self._enemy_near(world):
            self._target = None
            return None

        action = self._blocks(world, grounded)
        if action is not None:
            return action

        if not world.in_room:
            return self._warp(world)

        return self._room(world)

    def _stomp(self, world: GameWorld) -> bool:
        player = world.player
        for enemy in world.enemies:
            if not enemy.is_alive or getattr(enemy, "is_shell", False):
                continue
            if abs(enemy.bottom - player.bottom) > 8:
                continue
            dx = enemy.center_x - player.center_x
            toward = (dx > 0 and enemy.change_x < 0) or (dx < 0 and enemy.change_x > 0)
            if toward and 40 <= abs(dx) <= 90 and self._flat_to(world, enemy.center_x) and self._headroom(world):
                return True
        return False

    @staticmethod
    def _headroom(world: GameWorld) -> bool:
        """Над головой свободно на весь прыжок (~7 тайлов): под низким
        потолком прыжок на месте кончался ударом о врага сбоку."""
        player = world.player
        tx = int(player.center_x // TILE)
        feet = int((player.bottom + 1) // TILE)
        return not any(world.solid_at(tx + k, ty) for k in (-1, 0, 1) for ty in range(feet + 1, feet + 7))

    def _enemy_near(self, world: GameWorld) -> bool:
        player = world.player
        return any(
            enemy.is_alive
            and abs(enemy.center_x - player.center_x) < ENEMY_CLEAR * TILE
            and abs(enemy.center_y - player.center_y) < 3 * TILE
            for enemy in world.enemies
        )

    def _pursue(self, key) -> bool:
        """Идти к цели можно: она не брошена и время на неё не вышло."""
        if key in self._banned:
            return False
        if key != self._target:
            self._target = key
        self._key_steps[key] += 1
        self._target_steps = self._key_steps[key]
        if self._target_steps > GIVE_UP:
            self._banned.add(key)
            self._target = None
            return False
        return True

    @staticmethod
    def _feet_row(world: GameWorld) -> Optional[int]:
        """Ряд, где будут ноги после приземления: над первым твёрдым тайлом
        под игроком. None — под ним пропасть."""
        player = world.player
        tx = int(player.center_x // TILE)
        for ty in range(int((player.bottom + 1) // TILE), -1, -1):
            if world.solid_at(tx, ty - 1):
                return ty
        return None

    def _flat_to(self, world: GameWorld, x: float, feet: Optional[int] = None) -> bool:
        """До x — ровный пол на уровне ног, без ям и стен."""
        player = world.player
        if feet is None:
            feet = int((player.bottom + 1) // TILE)
        a, b = sorted((int(player.center_x // TILE), int(x // TILE)))
        # С запасом в тайл по краям: у края ряда блоков правило то включалось,
        # то нет, и Марио качался «к блоку — бег модели» без конца (ученик не
        # видит счётчика GIVE_UP, которым из этого выбирался учитель)
        for tx in range(a - 1, b + 2):
            if not world.solid_at(tx, feet - 1):
                return False
            if world.solid_at(tx, feet) or world.solid_at(tx, feet + 1):
                return False
        return True

    def _blocks(self, world: GameWorld, grounded: bool) -> Optional[int]:
        """Блок «?» с ударами над головой — прыжок; рядом — подойти под него.

        Работает и в прыжке: модель бежит, прыгая без остановки, и в моменты
        решений Марио почти всегда в воздухе. Тогда учитель ведёт по
        горизонтали без прыжка — Марио приземляется и дальше идёт пешком."""

        player = world.player

        feet = int((player.bottom + 1) // TILE) if grounded else self._feet_row(world)
        if feet is None:
            return None
        best = None
        for block in world.blocks:
            if block.state <= 0:
                continue
            bx, by = int(block.center_x // TILE), int(block.center_y // TILE)
            # В прыжке — только блоки выше головы: блок под ногами (Марио на
            # ряду блоков) снизу отсюда не выбить, а цель на нём качала
            # учителя между «к блоку» и «к трубе»
            if not grounded and block.bottom < player.top:
                continue
            # Низ блока на 1–6 тайлов выше головы и между ними пусто
            if not 1 <= by - feet <= 6:
                continue
            if any(world.solid_at(bx, ty) for ty in range(feet, by)):
                continue
            dx = block.center_x - player.center_x
            if not -BLOCK_BEHIND * TILE <= dx <= BLOCK_AHEAD * TILE:
                continue
            if not self._flat_to(world, block.center_x, feet):
                continue
            if best is None or abs(dx) < abs(best[0]):
                best = (dx, block)

        if best is None:
            return None
        dx, block = best
        if not self._pursue(("block", id(block))):
            return None
        if abs(dx) <= ALIGN:
            return JUMP if grounded else STAND
        return RIGHT if dx > 0 else LEFT

    def _warp(self, world: GameWorld) -> Optional[int]:
        """Непосещённая труба-вход впереди: подойти, запрыгнуть, встать."""

        player = world.player
        ahead = [
            w for w in world.warps
            if w.room not in world.rooms_visited
            and -TILE <= w.x - player.center_x <= WARP_AHEAD * TILE
        ]
        if not ahead:
            return None
        warp = min(ahead, key=lambda w: abs(w.x - player.center_x))
        if not self._pursue(("warp", warp.room)):
            return None

        dx = warp.x - player.center_x
        above = player.bottom >= warp.top - 1

        if above and abs(dx) <= 12:
            return STAND            # над трубой — перестать двигаться и сесть на неё
        if above:
            return RIGHT if dx > 0 else LEFT

        pipe_left = warp.x - TILE
        if not (world.on_ground or world.physics_engine.can_jump()):
            return RIGHT if dx > 0 else LEFT  # в прыжке жать к трубе, пока не над ней
        if dx < 0:
            return None
        if pipe_left - player.right <= 4:
            return RIGHT_JUMP       # вплотную — прыжок вверх вдоль стенки трубы
        if self._flat_to(world, pipe_left - TILE / 2):
            return RIGHT
        return None

    def _room(self, world: GameWorld) -> Optional[int]:
        """В комнате: монеты с пола прыжком строго вверх, потом — к устью.

        Прыжков наискосок нет: ими учитель заскакивал на ряд блоков между
        бортиками и упирался в стену до конца таймера."""

        player = world.player
        room_exit = world.area.exit
        feet = int((player.bottom + 1) // TILE)
        floor = int(room_exit.bottom // TILE) if room_exit is not None else feet
        grounded = world.on_ground or world.physics_engine.can_jump()

        # Стоит на трубе выхода — шаг назад, спрыгнуть к устью
        if room_exit is not None and feet > floor and player.center_x > room_exit.left - TILE:
            return LEFT

        if grounded and feet == floor:
            best = None
            for coin in world.objects:
                if not isinstance(coin, Coin):
                    continue
                cx, cy = int(coin.center_x // TILE), int(coin.center_y // TILE)
                if not 0 <= cy - feet <= 6:
                    continue
                if any(world.solid_at(cx, ty) for ty in range(feet, cy)):
                    continue
                if not self._flat_to(world, coin.center_x):
                    continue
                dx = coin.center_x - player.center_x
                if best is None or abs(dx) < abs(best[0]):
                    best = (dx, coin, cy - feet)
            if best is not None:
                dx, coin, rise = best
                if self._pursue(("coin", id(coin))):
                    if abs(dx) <= 10:
                        return JUMP if rise >= 1 else STAND
                    return RIGHT if dx > 0 else LEFT
        return RIGHT

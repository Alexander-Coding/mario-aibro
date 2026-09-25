"""Gymnasium-среда поверх GameWorld.

Один шаг среды = 4 кадра игры (решение 15 раз в секунду, как у быстрого
человека). Прыжок нажимается на первом кадре повтора.

Награда — это очки игры, а не «дошёл ли»: побеждает тот, кто набрал
больше. К ним добавлены:

- подсказка «иди вперёд» в виде потенциала F = γ·Φ(s') − Φ(s),
  Φ = 30 · прогресс, Φ(конец) = 0. Такая подсказка не меняет, какая
  стратегия лучшая (Ng, Harada, Russell, 1999), а только ускоряет
  обучение. В ноль телескопируется ДИСКОНТИРОВАННАЯ сумма; простая сумма
  наград за эпизод равна −(1−γ)·ΣΦ(s_t) — поэтому «награда эпизода» в
  журналах не равна очкам/100, сравнивать надо очки;
- штраф за потерю жизни и за смерть: смерть и так отнимает флаг и бонус
  за время, штраф лишь делает это заметным раньше.

Режим награды за продвижение (reward_mode):
- "progress" (по умолчанию с итерации 3): + 30·Δ(лучший X)/длина уровня —
  только за новый максимум, откат назад не штрафуется и не фармится. Не
  потенциальная, но за уровень суммируется ровно в 30 и не даёт дрейфа;
- "potential": γ·Φ(s') − Φ(s). Инвариантна, но при γ < 0.99 даёт дрейф
  −(1−γ)·Φ за шаг: у финиша при γ=0.97 это −0.9 за шаг, больше, чем
  награда за бег. На первых двух итерациях (γ=0.995) политика училась
  медленно: градиент политики равнялся градиенту энтропии;
- "collect" (с итерации 6): «проходить, а не спидранить». Модель it4a
  на 1-1 брала 2 удара по «?» из 33, 1 врага из 12 и ни одной монеты —
  3 360 из 5 660 очков давал бонус за время. Причина — дисконт: с γ=0.97
  (горизонт ~2 с) задержка ради четырёх ударов по блоку обесценивает всё,
  что впереди, и бег выгоднее. Поэтому:
  · подсказка за продвижение слабее (progress_scale, по умолчанию 10);
  · штраф за оставленное: когда предмет ушёл на 5 тайлов за спину (лучший
    X), а его не взяли, — miss_penalty × его очки/100: монета, каждый
    невыбитый удар блока «?», живой враг. Штраф приходит через ~1 с после
    решения пробежать мимо, а не в конце уровня — дисконт его не съедает.
    Выход из комнаты с несобранным — тот же штраф за оставленное в ней.
    Пропущенное прыжком через трубу-выход не штрафуется: это честный
    срез пути, как в Super Mario Bros;
  · secret_bonus за первый вход в секретную комнату — чтобы «вниз» на
    трубе нашлось, пока монеты комнаты ещё не выучены.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core import rules
from core.levels import load_levels
from core.world import GameWorld, Inputs
from Objects.Coin import Coin
from ai.features import (
    ACTIONS,
    GRID_H,
    GRID_W,
    MAX_ENEMIES,
    ENEMY_FEATURES,
    N_ACTIONS,
    N_CHANNELS,
    PLAYER_FEATURES,
    FeatureBuilder,
)

FRAME_SKIP = 4
GAMMA = 0.995                # по умолчанию для потенциальной подсказки

POINTS_SCALE = 100.0        # 100 очков = 1.0 награды
SHAPING_SCALE = 30.0        # Φ = 30 · прогресс
HIT_PENALTY = 3.0
DEATH_PENALTY = 10.0

# Эпизод обрезается, если столько шагов подряд нет нового лучшего X
STUCK_STEPS = 600

# Режим "collect"
COLLECT_PROGRESS_SCALE = 10.0
MISS_PENALTY = 0.5
SECRET_BONUS = 3.0
MISS_MARGIN = 5 * rules.TILE        # предмет «оставлен», когда он на 5 тайлов позади


def action_inputs(action: int, first_frame: bool) -> Inputs:
    _, right, left, jump, down = ACTIONS[int(action)]
    return Inputs(right=right, left=left, jump=jump and first_frame, down=down)


class MissTracker:
    """Что игрок оставил позади несобранным — для штрафа режима "collect".

    Предметы части уровня сортируются по X; указатель идёт за лучшим X
    игрока. Монета оставлена, если её спрайт ещё в уровне; блок — сколько
    в нём осталось ударов; враг — если жив (враги ходят, поэтому их
    проверяем каждый шаг, а не по указателю)."""

    def __init__(self, area):
        self.area = area
        items = [(obj.center_x, "coin", obj) for obj in area.objects if isinstance(obj, Coin)]
        items += [(block.center_x, "block", block) for block in area.blocks]
        items.sort(key=lambda item: item[0])
        self.items = items
        self.cursor = 0
        self.enemies = [enemy for enemy in area.enemies]

    @staticmethod
    def _left_points(kind: str, obj) -> int:
        if kind == "coin":
            return 100 if obj.sprite_lists else 0
        return 100 * max(obj.state, 0)

    def advance(self, x_behind: float, penalize: bool = True) -> float:
        """Очки/100, оставленные позади x_behind (без штрафа — просто пропустить)."""

        left = 0
        while self.cursor < len(self.items) and self.items[self.cursor][0] < x_behind:
            _, kind, obj = self.items[self.cursor]
            left += self._left_points(kind, obj)
            self.cursor += 1

        still = []
        for enemy in self.enemies:
            if not enemy.is_alive or not enemy.sprite_lists:
                continue
            if enemy.center_x < x_behind:
                left += 100
            else:
                still.append(enemy)
        self.enemies = still

        return left / POINTS_SCALE if penalize else 0.0

    def remaining(self) -> float:
        """Всё, что ещё не взято (выход из комнаты)."""
        return self.advance(float("inf"))


# ===============================================================
# ИСТОЧНИК УРОВНЕЙ
# ===============================================================


class LevelSampler:
    """Откуда брать уровень на каждый эпизод.

    mix — доля процедурных уровней (новый уровень каждый эпизод — модель не
    может заучить, ей приходится уметь играть); остальное — уровни из
    манифеста со split="train". Сложность процедурных растёт с обучением.
    """

    def __init__(
        self,
        mix: float = 0.8,
        difficulty: float = 0.3,
        levels: Optional[list] = None,
        seed: Optional[int] = None,
        drill: float = 0.0,
    ):
        self.mix = mix
        self.drill = drill
        self.difficulty = difficulty
        self.rng = random.Random(seed)
        self.train_levels = levels if levels is not None else load_levels(split="train")
        self._generator: Optional[Callable] = None
        self._drill: Optional[Callable] = None
        self._generator_error: Optional[str] = None

        try:
            from ai.levelgen import build_drill, generate_level
            self._generator = generate_level
            self._drill = build_drill
        except Exception as error:  # генератор ещё не готов или сломан
            self._generator_error = repr(error)

    def set_difficulty(self, difficulty: float) -> None:
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def sample(self) -> GameWorld:
        # Тренировка навыка (арена с блоками «?», без флага) — ai/levelgen.build_drill
        if self._drill is not None and self.drill > 0 and self.rng.random() < self.drill:
            seed = self.rng.randrange(1_000_000_000)
            return GameWorld(self._drill(seed, self.rng.uniform(0.0, self.difficulty)))

        use_generated = self._generator is not None and (
            not self.train_levels or self.rng.random() < self.mix
        )

        if use_generated:
            # Сложность — равномерно до текущего потолка: лёгкие уровни не
            # забываются, пока растёт планка
            difficulty = self.rng.uniform(0.0, self.difficulty)
            seed = self.rng.randrange(1_000_000_000)
            data = self._generator(seed, difficulty)
            return GameWorld(data)

        if self.train_levels:
            level = self.rng.choice(self.train_levels)
            return GameWorld(level.path, name=level.name, time_limit=level.time_limit)

        # Ни генератора, ни train-уровней — единственный уровень автора
        return GameWorld("maps/custom_level.json")


# ===============================================================
# СРЕДА
# ===============================================================


class MarioEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        sampler: Optional[LevelSampler] = None,
        world_factory: Optional[Callable[[], GameWorld]] = None,
        frame_skip: int = FRAME_SKIP,
        max_steps: int = 6000,
        reward_mode: str = "progress",
        gamma: float = GAMMA,
        progress_scale: Optional[float] = None,
        miss_penalty: float = MISS_PENALTY,
        secret_bonus: float = SECRET_BONUS,
    ):
        super().__init__()

        self.reward_mode = reward_mode
        self.gamma = gamma
        collect = reward_mode == "collect"
        self.progress_scale = float(
            progress_scale if progress_scale is not None
            else (COLLECT_PROGRESS_SCALE if collect else SHAPING_SCALE)
        )
        self.miss_penalty = float(miss_penalty) if collect else 0.0
        self.secret_bonus = float(secret_bonus) if collect else 0.0
        self._misses: dict[str, MissTracker] = {}

        self.sampler = sampler
        self.world_factory = world_factory
        self.frame_skip = frame_skip
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (N_CHANNELS, GRID_H, GRID_W), np.float32),
            "enemies": spaces.Box(-10.0, 10.0, (MAX_ENEMIES, ENEMY_FEATURES), np.float32),
            "enemy_mask": spaces.Box(0.0, 1.0, (MAX_ENEMIES,), np.float32),
            "player": spaces.Box(-10.0, 10.0, (PLAYER_FEATURES,), np.float32),
        })

        self.world: Optional[GameWorld] = None
        self.features: Optional[FeatureBuilder] = None
        self.last_action = 0
        self.steps = 0
        self.best_x = 0.0
        self.progress_x = 0.0
        self.since_best = 0
        self.potential = 0.0

    # -----------------------------------------------------------

    def set_difficulty(self, difficulty: float) -> None:
        if self.sampler is not None:
            self.sampler.set_difficulty(difficulty)

    def _new_world(self) -> GameWorld:
        if self.world_factory is not None:
            return self.world_factory()
        if self.sampler is not None:
            return self.sampler.sample()
        return GameWorld("maps/custom_level.json")

    def _span(self) -> float:
        world = self.world
        return max(world.geometry.door_x - world.spawn[0], 1.0)

    def _phi(self) -> float:
        world = self.world
        progress = float(np.clip((world.player.center_x - world.spawn[0]) / self._span(), 0.0, 1.0))
        return SHAPING_SCALE * progress

    # -----------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None and self.sampler is not None:
            self.sampler.rng.seed(seed)

        self.world = self._new_world()
        self.features = FeatureBuilder(self.world)
        # У тренировки без флага своя длина эпизода — дальше обрезка, а не смерть
        self.episode_steps = int(self.world.level_data.get("train_steps") or self.max_steps)

        self.last_action = 0
        self.steps = 0
        self.best_x = self.world.player.center_x
        self.progress_x = self.world.player.center_x
        self.since_best = 0
        self.potential = self._phi()
        self._misses = {}
        self.missed = 0.0

        return self.features.build(self.last_action), {"level": self.world.name}

    def _tracker(self, area) -> MissTracker:
        tracker = self._misses.get(area.key)
        if tracker is None:
            tracker = self._misses[area.key] = MissTracker(area)
        return tracker

    def step(self, action: int):
        world = self.world
        score_before = world.score
        hits_before = world.stats["hits"]

        area_before = world.area
        secrets_before = world.stats["secret"]
        events = []

        for frame in range(self.frame_skip):
            if world.done or world.cutscene:
                break
            events += world.step(action_inputs(action, frame == 0))

        # Заставки (флагшток, труба) от решений не зависят, таймер в них
        # стоит — проматываем за один шаг, чтобы не тратить шаги обучения
        while world.cutscene and not world.done:
            events += world.step(Inputs())

        self.steps += 1
        self.last_action = int(action)

        terminated = world.done
        points = world.score - score_before
        hits = world.stats["hits"] - hits_before

        # --- награда -------------------------------------------------
        reward = points / POINTS_SCALE
        reward -= HIT_PENALTY * hits

        if world.phase == "dead":
            reward -= DEATH_PENALTY

        if self.reward_mode == "potential":
            new_potential = 0.0 if terminated else self._phi()
            reward += self.gamma * new_potential - self.potential
            self.potential = new_potential
        elif not world.in_room:
            x = min(world.player.center_x, world.geometry.door_x)
            gained = max(0.0, x - self.progress_x)
            self.progress_x = max(self.progress_x, x)
            reward += self.progress_scale * gained / self._span()

        if self.reward_mode == "collect":
            reward += self.secret_bonus * (world.stats["secret"] - secrets_before)
            reward -= self.miss_penalty * self._collect_misses(world, area_before, events)

        # --- застрял -------------------------------------------------
        # Набрал очки — не застрял: в комнате свои координаты, а выбивать
        # блок можно и стоя на месте
        if not world.in_room and world.player.center_x > self.best_x + 1.0:
            self.best_x = world.player.center_x
            self.since_best = 0
        elif points > 0:
            self.since_best = 0
        else:
            self.since_best += 1

        truncated = not terminated and (
            self.steps >= self.episode_steps or self.since_best >= STUCK_STEPS
        )

        info = {"points": points}
        if terminated or truncated:
            world_summary = world.summary()
            info["episode_summary"] = world_summary
            info["episode_summary"]["truncated"] = bool(truncated)
            info["episode_summary"]["missed"] = round(self.missed, 2)
            info["episode_summary"]["drill"] = "train_steps" in world.level_data
            info["episode_summary"]["steps"] = self.steps

        return self.features.build(self.last_action), float(reward), terminated, truncated, info

    def _collect_misses(self, world: GameWorld, area_before, events) -> float:
        """Сколько очков/100 оставлено позади за этот шаг."""

        missed = 0.0

        # Вышел из комнаты — оставленное в ней уже не взять
        if area_before is not world.main and world.area is world.main:
            missed += self._tracker(area_before).remaining()

        if not world.in_room:
            tracker = self._tracker(world.main)
            # Вылез из трубы-выхода: всё, что перепрыгнули, — не штраф
            warped = any(e.kind == "pipe" and e.info.get("dir") == "up" for e in events)
            behind = world.max_x - MISS_MARGIN
            if warped:
                tracker.advance(behind, penalize=False)
            elif not world.done or world.won:
                missed += tracker.advance(behind)

        self.missed += missed
        return missed

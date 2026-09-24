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
  медленно: градиент политики равнялся градиенту энтропии.
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


def action_inputs(action: int, first_frame: bool) -> Inputs:
    _, right, left, jump = ACTIONS[int(action)]
    return Inputs(right=right, left=left, jump=jump and first_frame)


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
    ):
        self.mix = mix
        self.difficulty = difficulty
        self.rng = random.Random(seed)
        self.train_levels = levels if levels is not None else load_levels(split="train")
        self._generator: Optional[Callable] = None
        self._generator_error: Optional[str] = None

        try:
            from ai.levelgen import generate_level
            self._generator = generate_level
        except Exception as error:  # генератор ещё не готов или сломан
            self._generator_error = repr(error)

    def set_difficulty(self, difficulty: float) -> None:
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def sample(self) -> GameWorld:
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
    ):
        super().__init__()

        self.reward_mode = reward_mode
        self.gamma = gamma

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

        self.last_action = 0
        self.steps = 0
        self.best_x = self.world.player.center_x
        self.progress_x = self.world.player.center_x
        self.since_best = 0
        self.potential = self._phi()

        return self.features.build(self.last_action), {"level": self.world.name}

    def step(self, action: int):
        world = self.world
        score_before = world.score
        hits_before = world.stats["hits"]

        for frame in range(self.frame_skip):
            if world.done or world.finishing:
                break
            world.step(action_inputs(action, frame == 0))

        # Заставка финиша от решений не зависит (таймер в ней стоит) —
        # проматываем её за один шаг, чтобы не тратить шаги обучения
        while world.finishing and not world.done:
            world.step(Inputs())

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
        else:
            x = min(world.player.center_x, world.geometry.door_x)
            gained = max(0.0, x - self.progress_x)
            self.progress_x = max(self.progress_x, x)
            reward += SHAPING_SCALE * gained / self._span()

        # --- застрял -------------------------------------------------
        if world.player.center_x > self.best_x + 1.0:
            self.best_x = world.player.center_x
            self.since_best = 0
        else:
            self.since_best += 1

        truncated = not terminated and (
            self.steps >= self.max_steps or self.since_best >= STUCK_STEPS
        )

        info = {"points": points}
        if terminated or truncated:
            info["episode_summary"] = world.summary()
            info["episode_summary"]["truncated"] = bool(truncated)

        return self.features.build(self.last_action), float(reward), terminated, truncated, info

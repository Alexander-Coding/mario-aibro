"""Проверка снимков мира: GameWorld.snapshot() / restore() — кадр в кадр.

    .venv/bin/python -m tools.check_snapshot                 # всё: 6 уровней + 6 процедурных
    .venv/bin/python -m tools.check_snapshot --quick         # по одному прогону на уровень
    .venv/bin/python -m tools.check_snapshot --bench-only    # только замер скорости на 1-1
    .venv/bin/python -m tools.check_snapshot --workers 4 --json ai/runs/snap/check.json

Как проверяется. Мир подменяется наследником CheckedWorld: любой водитель
(эвристический бот, модель, случайные нажатия, сценарный бот труб из
tools/level_check) зовёт его step() как обычно, а тот:

* рядом держит «эталон» — такой же мир, который получает те же нажатия,
  но никогда не откатывается; после каждого кадра полные отпечатки
  (игрок, мир, все враги всех частей, монеты, блоки, флаг) обязаны
  совпасть — значит, после всех откатов основная траектория та же;
* на случайных кадрах снимает snapshot() и копит отпечатки эталона на
  следующие N кадров — это «исходное продолжение» S1 с реальными
  нажатиями водителя. Через N кадров мир откатывается к снимку и
  проигрывает S1 — отпечатки должны совпасть кадр в кадр; затем тот же
  снимок восстанавливается ещё раз со случайной S2, ещё раз с S3 и
  снова с S2 (повтор обязан совпасть с первым прогоном S2). Каждый
  --fresh-every-й снимок сверяет S3 ещё и с новым миром, прошедшим с
  нуля весь журнал нажатий. Потом мир возвращается к кадру, где был,
  — тоже через restore().

Сценарный бот level_check двигает игрока напрямую (доводка шага
сдвигом) — такие сдвиги пишутся в журнал и повторяются при
проигрывании и в эталоне.

Покрытие: у каждого проверенного снимка — метки того, что случилось в
его продолжениях (монета, удар по блоку, опустошённый блок, растоптанная
гумба, koopa в панцирь, добитый панцирь, удар и отброс, смерть в яму,
от врага, по времени, труба вниз → комната → боковая труба → подъём,
флаг, путь в замок, финиш) и в каком состоянии он снят. В конце —
таблица «сценарий → сколько снимков проверено» и скорость. Код выхода
не 0 при любом расхождении или непокрытом обязательном сценарии.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import contextlib
import io
import json
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.world as world_module
from core import snapshot as snapshot_module
from core.levels import load_levels
from core.world import TO_CASTLE, GameWorld, Inputs

# Что обязано попасть в проверенные снимки хотя бы раз
REQUIRED = (
    "coin", "block", "block_emptied", "stomp_goomba", "koopa_shell", "shell_kill",
    "damage", "death_pit", "death_enemy", "death_time",
    "pipe_down", "secret", "pipe_side", "pipe_up", "pipe_roundtrip",
    "flag", "to_castle", "complete",
)

MODEL_PATH = ROOT / "ai" / "models" / "mariojev_final.pt"


# ===============================================================
# ОТПЕЧАТОК КАДРА
# ===============================================================


def _tex(texture) -> str:
    # Блок «?» грузит текстуру заново на каждый удар — сравниваем по файлу
    path = texture.file_path
    return str(path) if path is not None else texture.cache_name


def sprite_index(world: GameWorld) -> dict[int, int]:
    """Номера спрайтов свежего мира: у двух миров одного уровня совпадают,
    по ним сравниваются составы списков."""
    index: dict[int, int] = {}
    for area in world.areas.values():
        for sprite in list(area.enemies) + list(area.objects) + list(area.blocks):
            index.setdefault(id(sprite), len(index))
    return index


def _entity(sprite, index) -> tuple:
    return (
        index.get(id(sprite), -1), type(sprite).__name__,
        sprite._position, sprite._velocity, _tex(sprite._texture), sprite._scale,
        sprite._width, sprite._height, sprite._color, sprite._visible,
        tuple(map(tuple, sprite.hit_box.get_adjusted_points())),
        sprite.state, sprite.current_frame, sprite.frame_timer, sprite.direction,
        sprite.is_alive, sprite.health,
    )


def fingerprint(world: GameWorld, index: dict[int, int], events: list) -> tuple:
    """Всё наблюдаемое состояние кадра — парами (что, значение)."""

    p = world.player
    out: list[tuple[str, Any]] = [
        ("player", _entity(p, {})),
        ("player.more", (p.invincible_timer, p.jump_requested, p.on_ground, p.is_jumping,
                         p.facing_direction, p.change_x, p.change_y)),
        ("world", (world.score, world.frame, world.elapsed, world.phase, world.area.key,
                   world._frames_left, world.on_ground, world.max_x, world._knockback,
                   world._chain, world._pipe_frames, world._sequence_frames,
                   world.death_cause, world.flag_bonus, world.flag_fraction, world.time_bonus,
                   None if world._pipe_warp is None else world._pipe_warp.room)),
        ("stats", tuple(sorted(world.stats.items()))),
        ("rooms", tuple(sorted(world.rooms_visited))),
        ("events", tuple((e.kind, e.points, tuple(sorted(e.info.items()))) for e in events)),
        ("world.events", len(world.events)),
    ]

    for key, area in world.areas.items():
        ids = lambda sprites: tuple(index.get(id(s), -1) for s in sprites)  # noqa: E731
        out.append((f"{key}.enemies", tuple(
            _entity(e, index) + (e.is_shell, getattr(e, "shell_timer", None), e.death_timer,
                                 e.move_direction, e.is_stomped, getattr(e, "scored", None),
                                 getattr(e, "shell_scored", None))
            for e in area.enemies
        )))
        out.append((f"{key}.entities", ids(area.map.entities)))
        out.append((f"{key}.engines", ids(area.enemy_engines)))
        out.append((f"{key}.active", tuple(sorted(index.get(i, -1) for i in area.active_enemies))))
        out.append((f"{key}.objects", ids(area.objects)))
        out.append((f"{key}.blocks", tuple((b.state, _tex(b._texture)) for b in area.blocks)))
        out.append((f"{key}.jumps", area.physics_engine.jumps_since_ground))

    flag = world.flag_sprite
    if flag is not None:
        out.append(("flag", flag._position))

    return tuple(out)


def _first_difference(x, y, path: str) -> str:
    # Спуск до первого различающегося элемента: у врагов отпечаток длинный
    if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
        for k, (u, v) in enumerate(zip(x, y)):
            if u != v:
                return _first_difference(u, v, f"{path}[{k}]")
    return f"{path}: {str(x)[:200]} != {str(y)[:200]}"


def diff(a: tuple, b: tuple) -> str:
    for (name, x), (_, y) in zip(a, b):
        if x != y:
            return _first_difference(x, y, name)
    return "длина отпечатка"


# ===============================================================
# МЕТКИ СЦЕНАРИЕВ
# ===============================================================


def event_tags(events) -> set[str]:
    tags: set[str] = set()
    kinds = []
    for e in events:
        kind, info = e.kind, e.info
        kinds.append((kind, info.get("dir")))
        if kind == "coin":
            tags.add("coin")
        elif kind == "block":
            tags.add("block")
            if info.get("left") == 0:
                tags.add("block_emptied")
        elif kind == "bump":
            tags.add("bump_empty")
        elif kind == "stomp":
            tags.add("koopa_shell")
        elif kind == "enemy":
            if info.get("via") == "stomp" and info.get("type") == "goomba":
                tags.add("stomp_goomba")
            elif info.get("via") == "stomp" and info.get("type") == "koopa":
                tags.add("shell_kill")
            elif info.get("via") == "bump":
                tags.add("bump_kill")
            else:
                tags.add("enemy_other")
        elif kind == "hit":
            tags.add("damage")
        elif kind == "death":
            tags.add(f"death_{info.get('cause')}")
        elif kind == "pipe":
            tags.add(f"pipe_{info.get('dir')}")
        elif kind in ("secret", "flag", "complete", "time_bonus"):
            tags.add(kind)
    # Вся петля трубы в одном продолжении: вниз → комната → вбок → вверх
    seq = [k for k in kinds if k[0] in ("pipe", "secret")]
    names = ["down" if k == ("pipe", "down") else "secret" if k[0] == "secret" else
             "side" if k == ("pipe", "side") else "up" for k in seq]
    it = iter(names)
    if all(any(n == want for n in it) for want in ("down", "secret", "side", "up")):
        tags.add("pipe_roundtrip")
    return tags


def state_tags(world: GameWorld) -> set[str]:
    tags = {f"snap:{world.phase}"}
    if world.in_room:
        tags.add("snap:room")
    if world.player.invincible_timer > 0:
        tags.add("snap:invincible")
    if world._knockback > 0:
        tags.add("snap:knockback")
    if not world.on_ground:
        tags.add("snap:airborne")
    for enemy in world.enemies:
        if enemy.is_alive and getattr(enemy, "is_shell", False):
            tags.add("snap:shell")
        if not enemy.is_alive:
            tags.add("snap:dying_enemy")
    return tags


# ===============================================================
# МИР ПОД ПРОВЕРКОЙ
# ===============================================================


def random_sequence(rng: random.Random, length: int, style: int) -> list[tuple[Inputs, None]]:
    """Нажатия удержаниями по 1–16 кадров. style 0 — больше бега вправо и
    прыжков, 1 — больше «вниз», влево и стояния (трубы, яма слева от
    старта), 2 — почти всё время стоять (смерть от врага, по времени)."""

    weights = ([4, 1, 3, 1, 1, 1, 1], [1, 3, 1, 1, 2, 3, 3], [1, 0, 0, 0, 1, 0, 10])[style]
    moves = [
        (True, False, False, False),   # вправо
        (False, True, False, False),   # влево
        (True, False, True, False),    # вправо + прыжок
        (False, True, True, False),    # влево + прыжок
        (False, False, True, False),   # прыжок
        (False, False, False, True),   # вниз
        (False, False, False, False),  # стоять
    ]
    out: list[tuple[Inputs, None]] = []
    while len(out) < length:
        right, left, jump, down = rng.choices(moves, weights)[0]
        hold = rng.randint(1, 16)
        for k in range(hold):
            # Прыжок — фронт нажатия, как в среде: только на первом кадре
            out.append((Inputs(right, left, jump and k == 0, down), None))
    return out[:length]


class Pending:
    __slots__ = ("snap", "frame", "n", "ref", "events", "log_start", "tags", "fresh", "phases")

    def __init__(self, snap, frame, n, log_start, tags, fresh):
        self.snap = snap
        self.frame = frame
        self.n = n
        self.ref: list[tuple] = []
        self.events: list = []
        self.phases: set[str] = set()
        self.log_start = log_start
        self.tags = tags
        self.fresh = fresh


class Report:
    """Итог одного прогона (сериализуется между процессами)."""

    def __init__(self, job: str):
        self.job = job
        self.checks = 0
        self.frames = 0
        self.coverage: Counter = Counter()
        self.mismatches: list[str] = []
        self.snapshot_us: list[float] = []
        self.restore_us: list[float] = []
        self.fresh = 0
        self.unknown_fields: set[str] = set()

    def merge(self, other: "Report") -> None:
        self.checks += other.checks
        self.frames += other.frames
        self.coverage.update(other.coverage)
        self.mismatches.extend(other.mismatches)
        self.snapshot_us.extend(other.snapshot_us)
        self.restore_us.extend(other.restore_us)
        self.fresh += other.fresh
        self.unknown_fields |= other.unknown_fields


# Параметры для CheckedWorld, созданных внутри чужого кода (level_check)
CONFIG: dict[str, Any] = {}


class CheckedWorld(GameWorld):
    """GameWorld, который сверяет снимки на ходу (см. docstring модуля)."""

    def __init__(self, level, **kwargs):
        super().__init__(level, **kwargs)
        cfg = CONFIG
        self._ck_report: Report = cfg["report"]
        self._ck_rng = random.Random(cfg["seed"] * 1000 + cfg.setdefault("worlds", 0))
        cfg["worlds"] += 1
        self._ck_level = level
        self._ck_kwargs = kwargs
        self._ck_ref = GameWorld(level, **kwargs)
        self._ck_index = sprite_index(self)
        self._ck_ref_index = sprite_index(self._ck_ref)
        self._ck_log: list[tuple[Inputs, Optional[tuple]]] = []
        self._ck_pending: list[Pending] = []
        self._ck_last_pos = self.player._position
        self._ck_count = 0
        self._ck_rate = cfg["rate"]
        self._ck_max_pending = cfg["max_pending"]
        self._ck_fresh_every = cfg["fresh_every"]
        self._ck_n = cfg["n_range"]
        start = fingerprint(self, self._ck_index, [])
        if start != fingerprint(self._ck_ref, self._ck_ref_index, []):
            self._ck_fail("старт: два новых мира различаются")

    # -----------------------------------------------------------

    def _ck_fail(self, message: str) -> None:
        report = self._ck_report
        if len(report.mismatches) < 20:
            report.mismatches.append(f"[{report.job} кадр {self.frame}] {message}")
        else:
            report.mismatches.append("...")

    def _ck_run(self, sequence) -> tuple[list[tuple], list, set[str]]:
        prints, events, phases = [], [], set()
        base_step = GameWorld.step
        for inputs, nudge in sequence:
            if nudge is not None:
                self.player.position = nudge
            step_events = base_step(self, inputs)
            prints.append(fingerprint(self, self._ck_index, step_events))
            events.extend(step_events)
            phases.add(self.phase)
        return prints, events, phases

    def _ck_restore(self, snap) -> None:
        t0 = time.perf_counter()
        self.restore(snap)
        self._ck_report.restore_us.append((time.perf_counter() - t0) * 1e6)

    def _ck_compare(self, what: str, got: list, want: list, frame0: int) -> bool:
        if len(got) != len(want):
            self._ck_fail(f"{what}: длина {len(got)} != {len(want)}")
            return False
        for k, (a, b) in enumerate(zip(got, want)):
            if a != b:
                self._ck_fail(f"{what}: снимок кадра {frame0}, +{k + 1} кадр: {diff(a, b)}")
                return False
        return True

    def _ck_verify(self, pending: Pending) -> None:
        report = self._ck_report
        rng = self._ck_rng
        snap = pending.snap
        m = len(pending.ref)
        s1 = self._ck_log[pending.log_start:pending.log_start + m]
        ok = True
        tags = set(pending.tags)
        tags |= event_tags(pending.events)

        # S1: реальное продолжение водителя, эталон — мир без откатов
        self._ck_restore(snap)
        got, _, _ = self._ck_run(s1)
        ok &= self._ck_compare("S1 после restore", got, pending.ref, pending.frame)
        if TO_CASTLE in pending.phases:
            tags.add("to_castle")

        # S2, S3 — другие нажатия из того же снимка, потом S2 ещё раз
        n2 = max(m, 30)
        s2 = random_sequence(rng, n2, 0)
        s3 = random_sequence(rng, rng.randint(30, max(31, self._ck_n[1])), 1)
        self._ck_restore(snap)
        f2, e2, p2 = self._ck_run(s2)
        self._ck_restore(snap)
        f3, e3, p3 = self._ck_run(s3)
        self._ck_restore(snap)
        f2b, _, _ = self._ck_run(s2)
        ok &= self._ck_compare("S2 повтор после S3", f2b, f2, pending.frame)
        for events, phases in ((e2, p2), (e3, p3)):
            tags |= event_tags(events)
            if TO_CASTLE in phases:
                tags.add("to_castle")

        # Сверка с новым миром, прошедшим журнал с нуля
        if pending.fresh:
            report.fresh += 1
            with contextlib.redirect_stdout(io.StringIO()):
                fresh = GameWorld(self._ck_level, **self._ck_kwargs)
            fresh_index = sprite_index(fresh)
            for inputs, nudge in self._ck_log[:pending.log_start]:
                if nudge is not None:
                    fresh.player.position = nudge
                GameWorld.step(fresh, inputs)
            # Сдвиг кадра снимка случился до snapshot() — он уже в снимке
            nudge = self._ck_log[pending.log_start][1] if pending.log_start < len(self._ck_log) else None
            if nudge is not None:
                fresh.player.position = nudge
            want = []
            for inputs, _ in s3:
                want.append(fingerprint(fresh, fresh_index, GameWorld.step(fresh, inputs)))
            ok &= self._ck_compare("S3 против нового мира", f3, want, pending.frame)

        report.checks += 1
        if ok:
            report.coverage.update(tags)
            report.coverage["verified"] += 1

    # -----------------------------------------------------------

    def step(self, inputs: Inputs = Inputs()) -> list:
        report = self._ck_report
        rng = self._ck_rng

        # Сдвиг игрока снаружи (доводка шага у сценарного бота) — в эталон
        # и в журнал
        nudge = None
        if self.player._position != self._ck_last_pos:
            nudge = self.player._position
            self._ck_ref.player.position = nudge

        if (not self.done and len(self._ck_pending) < self._ck_max_pending
                and rng.random() < self._ck_rate):
            t0 = time.perf_counter()
            snap = self.snapshot()
            report.snapshot_us.append((time.perf_counter() - t0) * 1e6)
            self._ck_count += 1
            fresh = self._ck_fresh_every > 0 and self._ck_count % self._ck_fresh_every == 0
            self._ck_pending.append(Pending(
                snap, self.frame, rng.randint(*self._ck_n), len(self._ck_log),
                state_tags(self), fresh,
            ))

        events = GameWorld.step(self, inputs)
        ref_events = self._ck_ref.step(inputs)
        self._ck_log.append((inputs, nudge))
        report.frames += 1

        mine = fingerprint(self, self._ck_index, events)
        ref = fingerprint(self._ck_ref, self._ck_ref_index, ref_events)
        if mine != ref:
            self._ck_fail(f"основная траектория разошлась с эталоном: {diff(mine, ref)}")
            # Дальше сравнивать нечего — всё поедет каскадом
            self._ck_pending.clear()
            self._ck_ref = None
            raise RuntimeError("lockstep")

        for pending in self._ck_pending:
            pending.ref.append(ref)
            pending.events.extend(ref_events)
            pending.phases.add(self._ck_ref.phase)

        due = [p for p in self._ck_pending if len(p.ref) >= p.n or self.done]
        if due:
            self._ck_flush(due, ref)

        self._ck_last_pos = self.player._position
        return events

    def _ck_flush(self, due: list[Pending], ref: Optional[tuple]) -> None:
        back = self.snapshot()
        for pending in due:
            self._ck_verify(pending)
            self._ck_pending.remove(pending)
        # Назад в «сейчас» — тоже через restore; сверка с эталоном
        self._ck_restore(back)
        if ref is not None:
            now = fingerprint(self, self._ck_index, self.events)
            if now != ref:
                self._ck_fail(f"возврат в текущий кадр: {diff(now, ref)}")

    def ck_finish(self) -> None:
        """Конец прогона: проверить недокопившие снимки и поля мира."""
        if self._ck_pending and self._ck_ref is not None:
            ref = fingerprint(self._ck_ref, self._ck_ref_index, self._ck_ref.events)
            self._ck_flush(list(self._ck_pending), ref)
        known = set(snapshot_module.WORLD_FIELDS) | set(snapshot_module.WORLD_CONTAINERS)
        known |= set(snapshot_module.WORLD_STATIC)
        extra = {k for k in self.__dict__ if not k.startswith("_ck_")} - known
        self._ck_report.unknown_fields |= extra


# ===============================================================
# ВОДИТЕЛИ
# ===============================================================


def drive(world: CheckedWorld, driver, max_frames: int) -> None:
    if hasattr(driver, "reset"):
        driver.reset(world)
    while not world.done and world.frame < max_frames:
        world.step(driver.act(world))


class RandomDriver:
    def __init__(self, seed: int, style: int):
        self.rng = random.Random(seed)
        self.style = style
        self.queue: list = []

    def act(self, world) -> Inputs:
        if not self.queue:
            self.queue = [i for i, _ in random_sequence(self.rng, 64, self.style)]
        return self.queue.pop(0)


class PipeSeeker:
    """Эвристика плюс «вниз» на трубе-входе: чтобы в продолжения попадали
    трубы и на уровнях с врагами (сценарный бот level_check ходит по
    уровню без врагов)."""

    def __init__(self):
        from tools.heuristic_bot import HeuristicBot
        self.bot = HeuristicBot()

    def reset(self, world=None) -> None:
        self.bot.reset(world)

    def act(self, world) -> Inputs:
        if world.cutscene:
            return Inputs()
        player = world.player
        if not world.in_room:
            todo = [w for w in world.main.warps if w.room not in world.rooms_visited]
            if todo:
                warp = min(todo, key=lambda w: abs(w.x - player.center_x))
                dx = warp.x - player.center_x
                if world.warp_under_player() is warp:
                    return Inputs(down=True)
                if abs(dx) < 260:
                    if abs(dx) <= 6:
                        return Inputs()
                    go_right = dx > 0
                    jump = world.on_ground and world.frame % 20 == 0
                    return Inputs(right=go_right, left=not go_right, jump=jump)
        return self.bot.act(world)


def run_job(job: dict) -> Report:
    import torch  # noqa: F401  (модель водителя; потоки — по одному)
    torch.set_num_threads(1)

    report = Report(job["name"])
    CONFIG.clear()
    CONFIG.update(
        report=report, seed=job["seed"], rate=job["rate"], max_pending=3,
        fresh_every=job["fresh_every"], n_range=tuple(job["n_range"]),
    )
    level = job["level"]
    kind = job["driver"]
    kwargs = {}
    if job.get("time_limit") is not None:
        kwargs["time_limit"] = job["time_limit"]

    try:
        if kind == "walker":
            # Сценарный бот level_check: трубы и тщательный сбор блоков
            # на уровне без врагов. Его миры создаются внутри — подменяем класс
            from tools import level_check
            original = world_module.GameWorld
            world_module.GameWorld = CheckedWorld
            created: list[CheckedWorld] = []
            orig_init = CheckedWorld.__init__

            def tracking_init(self, *a, **kw):
                orig_init(self, *a, **kw)
                created.append(self)

            CheckedWorld.__init__ = tracking_init
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    level_check.check_level(level, bot=False, sim=True, thorough=job.get("thorough", False))
            finally:
                world_module.GameWorld = original
                CheckedWorld.__init__ = orig_init
            for world in created:
                world.ck_finish()
            return report

        with contextlib.redirect_stdout(io.StringIO()):
            world = CheckedWorld(level, **kwargs)
        if kind == "bot":
            from tools.heuristic_bot import HeuristicBot
            driver = HeuristicBot()
        elif kind == "pipes":
            driver = PipeSeeker()
        elif kind == "model":
            from ai.agent import load_controller
            torch.manual_seed(job["seed"])
            driver = load_controller(MODEL_PATH, deterministic=False)
        elif kind == "random":
            driver = RandomDriver(job["seed"], job.get("style", 0))
        else:
            raise ValueError(kind)
        drive(world, driver, job["max_frames"])
        world.ck_finish()
    except RuntimeError as error:
        if str(error) != "lockstep":
            raise
    except Exception as error:  # noqa: BLE001 — падение тоже расхождение
        import traceback
        report.mismatches.append(f"[{report.job}] исключение: {error!r}\n{traceback.format_exc()[-1500:]}")
    return report


def init_worker() -> None:
    import torch
    torch.set_num_threads(1)


# ===============================================================
# ЗАМЕР СКОРОСТИ
# ===============================================================


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    return {
        "n": len(v),
        "median_us": round(v[len(v) // 2], 1),
        "p95_us": round(v[min(len(v) - 1, int(len(v) * 0.95))], 1),
        "max_us": round(v[-1], 1),
    }


def bench(iterations: int = 2000, rollout: int = 60, seed: int = 0) -> dict:
    """1-1, середина уровня: снимок, прогон rollout случайных кадров,
    откат — как у планировщика. Время прогона не входит."""

    from tools.heuristic_bot import HeuristicBot

    level = next(level for level in load_levels() if level.id == "1-1")
    with contextlib.redirect_stdout(io.StringIO()):
        world = GameWorld(level.path, time_limit=level.time_limit)
    bot = HeuristicBot()
    middle = (world.spawn[0] + world.geometry.door_x) / 2
    while not world.done and world.player.center_x < middle:
        world.step(bot.act(world))

    rng = random.Random(seed)
    snaps, restores, combined, idle = [], [], [], []
    moves = random_sequence(rng, iterations * rollout, 0)
    for i in range(iterations):
        t0 = time.perf_counter()
        state = world.snapshot()
        t1 = time.perf_counter()
        for inputs, _ in moves[i * rollout:(i + 1) * rollout]:
            world.step(inputs)
        t2 = time.perf_counter()
        world.restore(state)
        t3 = time.perf_counter()
        world.restore(state)
        t4 = time.perf_counter()
        snaps.append((t1 - t0) * 1e6)
        restores.append((t3 - t2) * 1e6)
        combined.append((t1 - t0 + t3 - t2) * 1e6)
        idle.append((t4 - t3) * 1e6)

    return {
        "level": "1-1",
        "x": round(world.player.center_x, 1),
        "frame": world.frame,
        "rollout_frames": rollout,
        "snapshot": _stats(snaps),
        "restore_after_rollout": _stats(restores),
        "snapshot_plus_restore": _stats(combined),
        "restore_unchanged": _stats(idle),
    }


# ===============================================================
# ЗАПУСК
# ===============================================================


def make_jobs(quick: bool, seed: int) -> list[dict]:
    from ai.levelgen import generate_level

    levels: list[tuple[str, Any, float]] = [
        (level.id, str(level.path), level.time_limit) for level in load_levels()
    ]
    for i, (gen_seed, difficulty) in enumerate(
        [(11, 0.2), (12, 0.4), (13, 0.5), (14, 0.6), (15, 0.8), (16, 1.0)]
    ):
        data = generate_level(gen_seed, difficulty)
        levels.append((f"gen{gen_seed}d{difficulty}", data, float(data.get("time_limit", 200))))

    base = dict(rate=1 / 25, fresh_every=4, n_range=(30, 300), max_frames=4000)
    jobs = []
    for k, (name, level, limit) in enumerate(levels):
        s = seed * 100 + k
        spec = [
            ("bot", {}),
            ("model", {}),
            ("pipes", {}),
            ("random", {"style": 0, "max_frames": 2500}),
            ("random", {"style": 1, "max_frames": 2500}),
            # Стоять под гумбами: три удара — смерть от врага
            ("random", {"style": 2, "time_limit": 40.0, "max_frames": 2500, "tag": "idle"}),
            # Таймаут: короткий таймер и в основном стояние
            ("random", {"style": 2, "time_limit": 6.0, "max_frames": 800, "tag": "timeout"}),
            # Длинные продолжения: вся петля трубы (вниз → комната →
            # вбок → вверх) должна поместиться в одно
            ("walker", {"thorough": k < 6, "n_range": (30, 900)}),
        ]
        if quick:
            spec = [spec[0], spec[2], spec[5], spec[6], ("walker", {"thorough": False, "n_range": (30, 900)})]
        for driver, extra in spec:
            job = dict(base, level=level, driver=driver, seed=s,
                       name=f"{name}/{extra.get('tag', driver)}{extra.get('style', '')}")
            job.update({key: value for key, value in extra.items() if key != "tag"})
            jobs.append(job)
    return jobs


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--only", default=None, help="только прогоны, в имени которых есть эта строка")
    parser.add_argument("--json", default=None, help="куда записать итог")
    args = parser.parse_args(argv)

    result: dict[str, Any] = {}
    failed = False

    if not args.bench_only:
        jobs = make_jobs(args.quick, args.seed)
        if args.only:
            jobs = [job for job in jobs if args.only in job["name"]]
        t0 = time.perf_counter()
        total = Report("всего")
        per_job = []
        with Pool(min(args.workers, len(jobs)), initializer=init_worker, maxtasksperchild=4) as pool:
            for report in pool.imap_unordered(run_job, jobs):
                total.merge(report)
                per_job.append((report.job, report.checks, report.coverage["verified"],
                                report.frames, len(report.mismatches)))
                mark = "OK " if not report.mismatches else "ERR"
                print(f"{mark} {report.job:<28} кадров {report.frames:>6}  снимков {report.checks:>4}"
                      f"  проверено {report.coverage['verified']:>4}", flush=True)
        elapsed = time.perf_counter() - t0

        print(f"\nПрогонов {len(jobs)}, кадров {total.frames}, снимков {total.checks}, "
              f"сверок с новым миром {total.fresh}, {elapsed:.0f} с")

        print("\nСценарий                 проверено снимков")
        for name in REQUIRED:
            print(f"  {name:<22} {total.coverage[name]:>6}{'' if total.coverage[name] else '   <-- НЕ ПОКРЫТ'}")
        others = sorted(k for k in total.coverage if k not in REQUIRED and k != "verified")
        for name in others:
            print(f"  {name:<22} {total.coverage[name]:>6}")

        print("\nВремя в прогонах (мкс): snapshot", _stats(total.snapshot_us),
              "\n                        restore ", _stats(total.restore_us))

        missing = [name for name in REQUIRED if not total.coverage[name]]
        if total.unknown_fields:
            print("\nПоля мира не учтены в core/snapshot.py:", sorted(total.unknown_fields))
        if total.mismatches:
            print(f"\nРАСХОЖДЕНИЯ ({len(total.mismatches)}):")
            for line in total.mismatches[:30]:
                print("  " + line)
        failed = bool(total.mismatches or missing or total.unknown_fields
                      or total.coverage["verified"] != total.checks)
        result.update(
            jobs=len(jobs), frames=total.frames, checks=total.checks,
            verified=total.coverage["verified"], fresh=total.fresh, seconds=round(elapsed, 1),
            coverage=dict(total.coverage), missing=missing, mismatches=total.mismatches,
            unknown_fields=sorted(total.unknown_fields),
            timing_in_runs={"snapshot": _stats(total.snapshot_us), "restore": _stats(total.restore_us)},
            per_job=per_job,
        )

    if not args.no_bench:
        b = bench()
        result["bench"] = b
        print("\nЗамер на 1-1 (середина уровня, x={x}, кадр {frame}, откат после {rollout_frames} кадров):"
              .format(**b))
        for key in ("snapshot", "restore_after_rollout", "snapshot_plus_restore", "restore_unchanged"):
            print(f"  {key:<24} {b[key]}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\nИТОГ:", "ОШИБКИ" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

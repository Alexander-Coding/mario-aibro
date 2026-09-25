"""Обучение подражанием учителю: сбор примеров и обучение.

Учителей два. Старый (ai/teacher.py) — модель it4a плюс правила сбора со
скрытым состоянием (бюджеты на цель, запреты, выход из тупика). Ученик
без памяти этого состояния не видит: на одно и то же наблюдение он
выучивал оба ответа и чередовал их — «вправо+прыжок / влево» по кругу.
Замер на mariojev_final: на skybridge 67 % решений в петле и 1408 смен
направления за 3 прогона. Новый — планировщик ai/planner.py: ищет план
на копии мира, скрытого состояния нет, и каждое решение подписывает
оценками всех действий (last_q) и распределением по ним (last_probs).
Ученик учится на мягких метках: где два действия почти равны, он не
выбирает одно наугад, а знает, что они равны.

    # примеры: планировщик играет в среде обучения, DART-шум на исполнении
    python -m ai.imitate collect --teacher planner --noise 0.1 --episodes 1500 --out ai/runs/plan1
    # обучение: ученик v2 с нуля, мягкие метки с весом 0.7
    python -m ai.imitate train --data ai/runs/plan1 --arch v2 --name mjv2
    # DAgger: играет ученик, планировщик подписывает, что надо было сделать
    python -m ai.imitate collect --teacher planner --student ai/models/mjv2_best.pt --beta 0.3 --out ai/runs/plan2
    python -m ai.imitate train --data ai/runs/plan1,ai/runs/plan2 --init ai/models/mjv2_best.pt --name mjv2b

    # старый учитель: путь к модели вместо planner, ученик v1 от it4a
    python -m ai.imitate collect --teacher ai/models/it4a_best.pt --out ai/runs/demos
    python -m ai.imitate train --data ai/runs/demos --arch v1 --name bc1

Одного подражания мало: ученик bc1 (95 % совпадений с учителем) на
отложенных уровнях застревал там, куда учитель не заходил, — кружил в
воздухе над пустыми блоками, упирался в стену с трубы выхода комнаты.
DAgger (Ross, Gordon, Bagnell, 2011) добирает примеры из состояний, в
которые попадает сам ученик; DART (Laskey и др., 2017) — шум на
исполнении при подписи учителя: учитель показывает, как вернуться с
небольшого отклонения, ещё до того, как обучен ученик.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent

# Откуда решение учителя. Порядок не менять: в старых примерах записан
# индекс. model/rules/escape — старый учитель, plan/rule — планировщик
SOURCES = ("model", "rules", "escape", "plan", "rule")
RULE_SOURCES = (SOURCES.index("rules"), SOURCES.index("rule"))

# Оценка недопустимого действия у планировщика — -inf; в файл (float16)
# она пишется как -1e4, чтобы в массиве не было бесконечностей
Q_INVALID = -1e4

# DART-шум исполняет случайное из действий 0..5: «вниз» вне трубы — то же
# «стоять», а на трубе уводит в комнату — шуму там делать нечего
NOISE_ACTIONS = 6

# Окно карты шарда (строк, столбцов, столбцов позади Марио). В шардах до
# расширения окна назад ключа нет — там 24 столбца, 4 позади
WINDOW_KEY = "grid_window"

# Сторож петель при сборе. Планировщик зацикливается редко (≈3 из 100
# процедурных уровней, чаще — в состояниях, куда его завёл ученик), но
# такой эпизод крутит петлю до конца таймера, и уверенные метки петли
# занимают до 16 % выборки. Застрял — это STUCK_WINDOW решений (4 с) в
# одной области без очков и с размахом X меньше STUCK_SPAN.
STUCK_WINDOW = 60
STUCK_SPAN = 4 * 32
# Эпизоды пишутся на диск частями: сбор идёт часами, а WSL на ПК уже
# перезапускался посреди прогона и уносил всё несохранённое
FLUSH_EPISODES = 10


class StuckGuard:
    """Застрял ли игрок: последние STUCK_WINDOW решений — одна область,
    один счёт, размах X меньше STUCK_SPAN."""

    def __init__(self):
        self.trace: list[tuple[float, str, int]] = []

    def reset(self) -> None:
        self.trace.clear()

    def record(self, world) -> None:
        self.trace.append((float(world.player.center_x), world.area.key, int(world.score)))
        if len(self.trace) > STUCK_WINDOW:
            del self.trace[0]

    def stuck(self) -> bool:
        if len(self.trace) < STUCK_WINDOW:
            return False
        xs = [x for x, _, _ in self.trace]
        return len({(area, score) for _, area, score in self.trace}) == 1 and max(xs) - min(xs) < STUCK_SPAN

    def moved_since(self, start: tuple[float, str, int]) -> bool:
        """С момента start были очки, смена области или сдвиг на STUCK_SPAN."""
        x, area, score = self.trace[-1]
        return area != start[1] or score != start[2] or abs(x - start[0]) >= STUCK_SPAN


def drill_cleared(world) -> bool:
    """Арена-тренировка (без флага) выбрана дочиста: монет нет, в блоках
    не осталось ударов. Дальше планировщик жмёт в стену почти наугад
    (мягкая метка ≈ равномерная) — такие примеры ничему не учат."""
    from Objects.Coin import Coin

    if "train_steps" not in world.level_data:
        return False
    return (not any(isinstance(obj, Coin) for obj in world.objects)
            and all(block.state <= 0 for block in world.blocks))


def grid_window() -> tuple[int, int, int]:
    """Окно карты, которое сейчас строит ai/features.py."""
    from ai.features import GRID_BACK, GRID_H, GRID_W
    return GRID_H, GRID_W, GRID_BACK


def _npz_shape(npz, key: str) -> tuple[int, ...]:
    """Форма массива в .npz по заголовку, не распаковывая сам массив
    (карта шарда — сотни мегабайт)."""
    with npz.zip.open(f"{key}.npy") as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        return tuple(reader(f)[0])


def shard_window(npz) -> tuple[int, int, int]:
    """Окно карты шарда: записанное, а у старых шардов — по ширине карты."""
    from ai.features import GRID_BACK, LEGACY_GRID_BACK, LEGACY_GRID_W
    if WINDOW_KEY in npz.files:
        return tuple(int(v) for v in npz[WINDOW_KEY])
    height, width = _npz_shape(npz, "grid")[-2:]
    return height, width, LEGACY_GRID_BACK if width == LEGACY_GRID_W else GRID_BACK


def make_teacher(spec: str):
    """planner — планировщик ai/planner.py, иначе путь к модели старого учителя."""
    if spec == "planner":
        from ai.planner import Planner
        return Planner()
    from ai.decide import DecisionModel
    from ai.teacher import Teacher
    return Teacher(DecisionModel.load(spec))


# ===============================================================
# СБОР ПРИМЕРОВ
# ===============================================================


def _collect_worker(job: tuple) -> dict:
    worker, episodes, seed, teacher_spec, out_dir, drill, mix, student_path, beta, sample, noise = job
    torch.set_num_threads(1)

    from ai.agent import LoopMeter
    from ai.decide import DecisionModel
    from ai.env import MarioEnv, LevelSampler, action_inputs

    teacher = make_teacher(teacher_spec)
    student = DecisionModel.load(student_path) if student_path else None
    sampler = LevelSampler(mix=mix, difficulty=1.0, seed=seed, drill=drill)
    env = MarioEnv(sampler=sampler, reward_mode="collect")
    # Мягкие метки пишутся, если учитель их даёт (планировщик)
    soft = hasattr(teacher, "last_probs")

    rng = random.Random(seed)
    keys = ["grid", "enemies", "enemy_mask", "player", "action", "source", "episode"] + (["probs", "q"] if soft else [])
    buffers: dict[str, list] = {key: [] for key in keys}
    summaries = []
    counts = np.zeros(len(SOURCES), dtype=np.int64)
    parts = 0
    samples = 0
    # Время воркера целиком и время на решения учителя: из них — скорость
    # сбора (решений в секунду на воркер) и доля учителя в ней
    started = time.perf_counter()
    teacher_seconds = 0.0

    def flush() -> None:
        nonlocal parts, samples
        if not buffers["action"]:
            return
        arrays = {k: np.stack(v) if k in ("grid", "enemies", "enemy_mask", "player", "probs", "q") else np.asarray(v)
                  for k, v in buffers.items()}
        # Окно карты — в самом шарде: примеры разной ширины учить вместе нельзя
        arrays[WINDOW_KEY] = np.asarray(grid_window(), dtype=np.int16)
        np.savez_compressed(Path(out_dir) / f"shard_{worker:02d}_{parts:03d}.npz", **arrays)
        counts[:] += np.bincount(arrays["source"], minlength=len(SOURCES))
        samples += len(arrays["action"])
        parts += 1
        for value in buffers.values():
            value.clear()

    for episode in range(episodes):
        # Сложность — равномерно по всему диапазону
        sampler.set_difficulty(rng.uniform(0.05, 1.0))
        obs, _ = env.reset()
        teacher.reset(env.world)
        loops = LoopMeter()
        guard = StuckGuard()
        first = len(buffers["action"])
        # Ученик застрял — управление у учителя, пока тот не сдвинется с
        # места: ярлыки в петле ученика ценны, только если учитель из неё
        # выходит. Не вышел за STUCK_WINDOW решений — петля и у учителя
        takeover: Optional[tuple[int, tuple[float, str, int]]] = None
        cut = None
        last = 0
        while True:
            # Учитель решает всегда: это подпись примера (планировщик
            # возвращает мир в точности в то состояние, в котором взял)
            tick = time.perf_counter()
            label = teacher.decide(env.world, last)
            teacher_seconds += time.perf_counter() - tick
            action = label
            if takeover is None:
                if student is not None and rng.random() >= beta:
                    # Ученик ходит так, как будет играть: argmax или сэмпл
                    action = int(student.predict(env.world, last_action=last, deterministic=not sample)
                                 ["answers"]["move"]["index"])
                if noise > 0 and rng.random() < noise:
                    # DART: исполняется случайное действие, подпись — учителя
                    action = rng.randrange(NOISE_ACTIONS)
            buffers["grid"].append(np.round(obs["grid"] * 4).astype(np.uint8))
            buffers["enemies"].append(obs["enemies"].astype(np.float16))
            buffers["enemy_mask"].append(obs["enemy_mask"].astype(np.uint8))
            buffers["player"].append(obs["player"].astype(np.float16))
            buffers["action"].append(label)
            buffers["source"].append(SOURCES.index(teacher.last_source))
            buffers["episode"].append(worker * 100_000 + episode)
            if soft:
                probs, q = _soft_label(teacher, label)
                buffers["probs"].append(probs)
                buffers["q"].append(q)
            loops.record_inputs(env.world, action_inputs(action, True))
            obs, _, terminated, truncated, info = env.step(action)
            last = action
            if terminated or truncated:
                summary = info["episode_summary"]
                break

            guard.record(env.world)
            if drill_cleared(env.world):
                cut = "drill_cleared"
            elif takeover is not None and guard.moved_since(takeover[1]):
                takeover = None
            elif guard.stuck():
                if student is not None and takeover is None:
                    # Окно петли ученика оставляем; дальше решает учитель
                    takeover = (len(buffers["action"]) - STUCK_WINDOW, guard.trace[-1])
                    guard.reset()
                else:
                    # Застрял сам учитель: его петлю (и петлю ученика перед
                    # ней) — вон из выборки, эпизод окончен
                    drop = takeover[0] if takeover is not None else len(buffers["action"]) - STUCK_WINDOW
                    for value in buffers.values():
                        del value[max(drop, first):]
                    cut = "stuck"
            if cut is not None:
                summary = env.world.summary()
                summary["drill"] = "train_steps" in env.world.level_data
                break

        summary["loops"] = loops.summary(summary["frames"])
        summary["cut"] = cut
        summaries.append(summary)
        if (episode + 1) % FLUSH_EPISODES == 0:
            flush()

    flush()
    return {"worker": worker, "samples": samples, "summaries": summaries, "sources": counts.tolist(),
            "parts": parts, "seconds": time.perf_counter() - started, "teacher_seconds": teacher_seconds}


def _soft_label(teacher, label: int) -> tuple[np.ndarray, np.ndarray]:
    """Распределение и оценки учителя по 7 действиям для записи в float16.

    Если учитель на этом решении их не дал — подпись вырождается в жёсткую:
    вся вероятность на выбранном действии."""
    from ai.features import N_ACTIONS

    probs = getattr(teacher, "last_probs", None)
    q = getattr(teacher, "last_q", None)
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float32)
        # NaN или не распределение (все Q = -inf) — NaN в потере разнёс бы
        # его по всем весам, и лучший чекпойнт не сохранился бы ни разу
        if not np.all(np.isfinite(probs)) or probs.min() < 0 or abs(probs.sum() - 1.0) > 1e-2:
            probs = None
    if probs is None:
        probs = np.zeros(N_ACTIONS, dtype=np.float32)
        probs[label] = 1.0
    if q is None:
        q = np.full(N_ACTIONS, Q_INVALID, dtype=np.float32)
        q[label] = 0.0
    q = np.nan_to_num(np.asarray(q, dtype=np.float32), nan=Q_INVALID, neginf=Q_INVALID, posinf=-Q_INVALID)
    return np.asarray(probs, dtype=np.float16), np.clip(q, Q_INVALID, -Q_INVALID).astype(np.float16)


def collect(args) -> None:
    from multiprocessing import get_context

    # Воркеры однопоточные: иначе каждый поднимает потоки на все ядра, и
    # 10 воркеров на 12-поточном ПК (он общий) стоят в очереди друг к другу
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per = math.ceil(args.episodes / args.jobs)
    jobs = [(w, per, args.seed * 1000 + w, args.teacher, str(out), args.drill, args.mix, args.student, args.beta,
             args.sample, args.noise) for w in range(args.jobs)]

    started = time.time()
    with get_context("spawn").Pool(args.jobs) as pool:
        results = pool.map(_collect_worker, jobs)

    summaries = [s for r in results for s in r["summaries"]]
    levels = [s for s in summaries if not s.get("drill")]

    def share(key, total):
        rows = [s for s in levels if s.get(total)]
        return round(float(np.mean([s[key] / s[total] for s in rows])), 3) if rows else None

    def loops_mean(key):
        return round(float(np.mean([s["loops"][key] for s in summaries])), 4) if summaries else None

    sources = np.sum([r["sources"] for r in results], axis=0).tolist()
    worker_seconds = sum(r["seconds"] for r in results)
    samples = sum(r["samples"] for r in results)
    report = {
        "teacher": args.teacher,
        "student": args.student,
        "beta": args.beta if args.student else None,
        "noise": args.noise,
        "samples": samples,
        "episodes": len(summaries),
        "minutes": round((time.time() - started) / 60, 1),
        # Скорость одного воркера (по сумме их времени, без простоя пула в
        # конце) и доля времени, которую съедает учитель
        "decisions_per_s_worker": round(samples / worker_seconds, 2) if worker_seconds else None,
        "teacher_share": round(sum(r["teacher_seconds"] for r in results) / worker_seconds, 3)
        if worker_seconds else None,
        "workers": [{"worker": r["worker"], "samples": r["samples"], "seconds": round(r["seconds"], 1)}
                    for r in results],
        "sources": {name: count for name, count in zip(SOURCES, sources) if count},
        "win_rate": round(float(np.mean([s["won"] for s in levels])), 3) if levels else None,
        "score": round(float(np.mean([s["score"] for s in levels]))) if levels else None,
        "collected": {"coins": share("coins", "coins_total"), "blocks": share("blocks", "block_hits_total"),
                      "enemies": share("enemies", "enemies_total"), "secrets": share("secrets", "secrets_total")},
        # Петли (ai/agent.py LoopMeter) — по исполненным действиям, т. е. с шумом и ходами ученика
        "loops": {"stall_frac": loops_mean("stall_frac"), "flips_per_min": loops_mean("flips_per_min"),
                  "stalls": loops_mean("stalls")},
        # Эпизоды, оборванные сторожем: учитель застрял (его петля выброшена)
        # или арена выбрана дочиста
        "cuts": {"stuck": sum(s.get("cut") == "stuck" for s in summaries),
                 "drill_cleared": sum(s.get("cut") == "drill_cleared" for s in summaries)},
        "episodes_loops": [{"level": s["level"], "drill": s.get("drill", False), "won": s["won"],
                            "cut": s.get("cut"), **s["loops"]}
                           for s in summaries],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "episodes_loops"}, ensure_ascii=False, indent=1))


# ===============================================================
# ОБУЧЕНИЕ
# ===============================================================


def load_demos(data_dirs: str) -> dict[str, np.ndarray]:
    """Папки с примерами через запятую; номера эпизодов разводятся по папкам.

    Массивы собираются в заранее выделенную память: 1.5 млн примеров — это
    5.2 ГБ одной только карты, а склейка списков держала бы их дважды.
    Шарды старого учителя без мягких меток и шарды планировщика с ними
    смешиваются: has_probs отмечает примеры, у которых метка мягкая. Шарды
    с разным окном карты — нет (ValueError); окно — в data["grid_window"]."""

    shards = [(n, path) for n, data_dir in enumerate(d for d in data_dirs.split(",") if d)
              for path in sorted(Path(data_dir).glob("shard_*.npz"))]
    if not shards:
        raise FileNotFoundError(f"нет shard_*.npz в {data_dirs}")
    sizes, specs, windows = [], {}, {}
    for _, path in shards:
        with np.load(path) as shard:
            sizes.append(len(shard["action"]))
            windows.setdefault(shard_window(shard), []).append(path)
            for key in shard.files:
                if key not in specs and key != WINDOW_KEY:
                    value = shard[key]
                    specs[key] = (value.shape[1:], value.dtype)
    if len(windows) > 1:
        # Столбец карты в шардах разной ширины — разные места мира: смесь
        # учила бы модель видеть монету то в 4, то в 8 тайлах позади
        listing = "; ".join(f"{h}×{w}, {b} позади: {len(paths)} шардов, например {paths[0]}"
                            for (h, w, b), paths in windows.items())
        raise ValueError(f"шарды с разным окном карты не смешиваются — {listing}. "
                         f"Учите на папках одной ширины (старые — 24 столбца, новые — {grid_window()[1]})")

    total = sum(sizes)
    data = {key: np.zeros((total, *shape), dtype=dtype) for key, (shape, dtype) in specs.items()}
    data["has_probs"] = np.zeros(total, dtype=bool)
    position = 0
    for (n, path), size in zip(shards, sizes):
        with np.load(path) as shard:
            for key in shard.files:
                if key == WINDOW_KEY:
                    continue
                value = shard[key]
                if key == "episode":
                    value = value + n * 10_000_000
                data[key][position:position + size] = value
            data["has_probs"][position:position + size] = "probs" in shard.files
        position += size
    data[WINDOW_KEY] = next(iter(windows))
    return data


def resolve_arch(args) -> dict:
    """Архитектура ученика: --arch (ai/model.py ARCHS) и явные размеры поверх."""
    from ai.model import ARCHS

    config = dict(ARCHS[args.arch])
    for key, value in (("d_model", args.d_model), ("layers", args.layers), ("heads", args.heads)):
        if value is not None:
            config[key] = value
    if args.patch:
        config["patch_h"], config["patch_w"] = (int(v) for v in args.patch.lower().split("x"))
    return config


def bf16_available() -> bool:
    try:
        return bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
    except Exception:
        return False


def train(args) -> None:
    from ai.model import PATCH, MarioJev, count_parameters, expand_state_dict
    from ai.train import MODEL_DIR, load_checkpoint, save_checkpoint

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    data = load_demos(args.data)
    total = len(data["action"])

    # Примеры «вслепую» (выход из тупика) из наблюдения не выводятся — не учим
    keep = data["source"] != SOURCES.index("escape")
    episodes = np.unique(data["episode"])
    rng = np.random.default_rng(args.seed)
    held = set(rng.choice(episodes, size=max(1, len(episodes) // 20), replace=False).tolist())
    is_val = np.isin(data["episode"], list(held))
    train_idx = np.flatnonzero(keep & ~is_val)
    val_idx = np.flatnonzero(keep & is_val)

    # Решения правил — редкие и новые, ради них всё и затевается: вес выше
    is_rule = np.isin(data["source"], RULE_SOURCES)
    weights = np.where(is_rule, args.rules_weight, 1.0).astype(np.float32)

    # Мягкая метка — распределение учителя; в float16 сумма чуть гуляет
    if "probs" in data:
        probs = data["probs"].astype(np.float32)
        probs /= np.maximum(probs.sum(1, keepdims=True), 1e-6)
        data["probs"] = probs
    has_probs = data["has_probs"]
    counts = np.bincount(data["source"], minlength=len(SOURCES))
    print(f"[imitate] примеров {total:,}: обучение {len(train_idx):,}, проверка {len(val_idx):,}; "
          + ", ".join(f"{name} {int(c):,}" for name, c in zip(SOURCES, counts) if c)
          + f"; с мягкой меткой {int(has_probs.sum()):,}", flush=True)

    # --- модель ----------------------------------------------------
    init = args.init
    if init is None:
        # v1 по-старому продолжает it4a, новые архитектуры учатся с нуля
        init = str(ROOT / "ai" / "models" / "it4a_best.pt") if args.arch == "v1" else "none"
    # Окно карты модели — окно примеров: модель учится видеть ровно то, что в них
    height, width, back = data[WINDOW_KEY]
    window = {"grid_h": height, "grid_w": width, "grid_back": back}
    if init == "none":
        config = {**resolve_arch(args), **window}
        model = MarioJev(**config).to(device)
    else:
        # Архитектура — от исходной модели: веса другой нарезки не переносятся.
        # Окно шире, чем у неё (модели до расширения назад — 24 столбца), —
        # позиции токенов переносятся со сдвигом (ai/model.py widen_grid_pos)
        base, _ = load_checkpoint(Path(init), device)
        config = {**{k: base.config[k] for k in ("d_model", "heads", "layers", "patch_h", "patch_w")}, **window}
        model = MarioJev(**config).to(device)
        if base.config != model.config:
            try:
                model.load_state_dict(expand_state_dict(base.state_dict(), model, base.config))
            except ValueError as error:
                raise SystemExit(f"[imitate] {init} (окно {base.grid_w} столбцов, {base.grid_back} позади) не "
                                 f"продолжить на примерах с окном {width} ({back} позади): {error}") from None
        else:
            model.load_state_dict(base.state_dict())
    legacy = (model.patch_h, model.patch_w) == (PATCH, PATCH)
    amp = device.type == "cpu" and (args.amp == "bf16" or (args.amp == "auto" and bf16_available()))
    print(f"[imitate] модель {model.config}, параметров {count_parameters(model):,}, от {init}; "
          f"устройство {device}, потоков {torch.get_num_threads()}, bf16 {amp}", flush=True)

    def batch(idx):
        return {
            "grid": torch.as_tensor(data["grid"][idx], device=device).float() / 4.0,
            "enemies": torch.as_tensor(data["enemies"][idx], device=device).float(),
            "enemy_mask": torch.as_tensor(data["enemy_mask"][idx], device=device).float(),
            "player": torch.as_tensor(data["player"][idx], device=device).float(),
        }

    def logits_of(idx):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            return model(batch(idx))[0].float()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    steps_per_epoch = math.ceil(len(train_idx) / args.batch)
    steps_total = args.epochs * steps_per_epoch
    if args.max_steps:
        steps_total = min(steps_total, args.max_steps)
    step = 0
    best = -math.inf

    @torch.no_grad()
    def validate() -> dict:
        """Совпадение argmax с учителем (всё и на правилах) и KL(учитель ‖ ученик)
        на примерах с мягкой меткой — насколько ученик повторяет всё
        распределение, а не только лучший ответ."""
        model.eval()
        correct = rules_correct = rules_total = 0
        kl_sum, kl_total = 0.0, 0
        for start in range(0, len(val_idx), 4096):
            idx = val_idx[start:start + 4096]
            logits = logits_of(idx)
            pred = logits.argmax(-1).cpu().numpy()
            target = data["action"][idx]
            correct += int((pred == target).sum())
            rule = is_rule[idx]
            rules_correct += int((pred[rule] == target[rule]).sum())
            rules_total += int(rule.sum())
            soft = has_probs[idx]
            if soft.any():
                p = torch.as_tensor(data["probs"][idx[soft]], device=device)
                log_q = torch.log_softmax(logits[torch.as_tensor(soft, device=device)], -1)
                kl = (torch.special.xlogy(p, p) - p * log_q).sum(-1)
                kl_sum += float(kl.sum())
                kl_total += int(soft.sum())
        model.train()
        return {"accuracy": correct / max(len(val_idx), 1),
                "rules_accuracy": rules_correct / max(rules_total, 1) if rules_total else None,
                "kl": kl_sum / kl_total if kl_total else None}

    print(f"[imitate] шагов {steps_total:,} по {args.batch} (эпоха — {steps_per_epoch:,}), "
          f"мягкие метки с весом {args.soft_weight}", flush=True)
    model.train()
    started = time.time()
    epoch = 0
    while step < steps_total:
        order = rng.permutation(train_idx)
        losses = []
        epoch_started = time.time()
        for start in range(0, len(order), args.batch):
            if step >= steps_total:
                break
            idx = np.sort(order[start:start + args.batch])
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / steps_total))
            for group in optimizer.param_groups:
                group["lr"] = lr
            logits = logits_of(idx)
            log_p = torch.log_softmax(logits, -1)
            target = torch.as_tensor(data["action"][idx], device=device).long()
            per = -log_p.gather(1, target[:, None]).squeeze(1)
            soft = has_probs[idx]
            if soft.any() and args.soft_weight > 0:
                # loss = (1−a)·CE(жёсткая) + a·CE(мягкая) — только там, где мягкая есть
                probs = torch.as_tensor(data["probs"][idx], device=device)
                soft_ce = -(probs * log_p).sum(-1)
                mask = torch.as_tensor(soft, device=device)
                per = torch.where(mask, (1 - args.soft_weight) * per + args.soft_weight * soft_ce, per)
            w = torch.as_tensor(weights[idx], device=device)
            loss = (per * w).sum() / w.sum()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if args.log_every and step % args.log_every == 0:
                rate = len(losses) * args.batch / (time.time() - epoch_started)
                print(f"[imitate]   шаг {step:,}/{steps_total:,}: loss {np.mean(losses[-args.log_every:]):.4f}, "
                      f"{rate:,.0f} прим/с", flush=True)
        epoch += 1

        rate = len(losses) * args.batch / (time.time() - epoch_started)
        val = validate()
        kl = f", KL {val['kl']:.4f}" if val["kl"] is not None else ""
        rules = f", на правилах {val['rules_accuracy']:.3f}" if val["rules_accuracy"] is not None else ""
        print(f"[imitate] эпоха {epoch}/{args.epochs}: loss {np.mean(losses):.4f}, "
              f"точность {val['accuracy']:.3f}{rules}{kl}; {rate:,.0f} прим/с", flush=True)
        # Лучшая — по совпадению с учителем (и на правилах), а при мягких
        # метках ещё и по KL: оно штрафует за уверенность там, где учителю всё равно
        score = val["accuracy"] + (val["rules_accuracy"] or 0.0) - (val["kl"] or 0.0)
        if score > best:
            best = score
            cpu_model = MarioJev(**model.config)
            cpu_model.load_state_dict({k: v.float().cpu() for k, v in model.state_dict().items()})
            save_checkpoint(MODEL_DIR / f"{args.name}_best.pt", cpu_model, {
                "imitation": {"data": str(args.data), "init": init, "epoch": epoch, "step": step, **val,
                              "samples": len(train_idx), "soft_samples": int(has_probs[train_idx].sum()),
                              "rules_weight": args.rules_weight, "soft_weight": args.soft_weight,
                              "arch": model.config, "grid_window": [height, width, back], "bf16": amp,
                              "minutes": round((time.time() - started) / 60, 1)},
                "reward_mode": "collect",
                # Ученик старого учителя (патч 4×4) играет сэмплом: argmax без
                # памяти зацикливался там, где учитель выбирался счётчиками
                # (турнир: сэмпл 62 тыс., argmax 42). Планировщик без скрытого
                # состояния последователен — ученик v2 играет argmax;
                # калибровка/турнир могут поменять
                "play_deterministic": not legacy,
                "play_temperature": 1.0,
            })

    print(f"[imitate] сохранено: {MODEL_DIR / f'{args.name}_best.pt'} "
          f"({(time.time() - started) / 60:.1f} мин)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--episodes", type=int, default=1500)
    c.add_argument("--jobs", type=int, default=10, help="воркеров (однопоточных); ПК общий — не больше 10")
    c.add_argument("--teacher", default=str(ROOT / "ai" / "models" / "it4a_best.pt"),
                   help="planner — планировщик ai/planner.py; иначе путь к модели старого учителя")
    c.add_argument("--out", default=str(ROOT / "ai" / "runs" / "demos"))
    c.add_argument("--drill", type=float, default=0.1)
    c.add_argument("--mix", type=float, default=0.7)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--student", default=None, help="DAgger: играет эта модель, учитель только подписывает")
    c.add_argument("--beta", type=float, default=0.3, help="DAgger: доля решений, которые всё же делает учитель")
    c.add_argument("--sample", action="store_true",
                   help="DAgger: ученик выбирает сэмплом, а не argmax — так он играет в игре (argmax зацикливался)")
    c.add_argument("--noise", type=float, default=0.0,
                   help="DART: с этой вероятностью исполняется случайное из действий 0..5, подпись — учителя")

    t = sub.add_parser("train")
    t.add_argument("--data", default=str(ROOT / "ai" / "runs" / "demos"))
    t.add_argument("--arch", default="v2", help="v1, v2, v2l (ai/model.py ARCHS)")
    t.add_argument("--d-model", type=int, default=None)
    t.add_argument("--layers", type=int, default=None)
    t.add_argument("--heads", type=int, default=None)
    t.add_argument("--patch", default=None, help="патч «рядов x столбцов», например 4x2")
    t.add_argument("--init", default=None,
                   help="none — с нуля; путь — продолжить эту модель (архитектура — её). "
                        "По умолчанию: v1 — от it4a_best, остальные — с нуля")
    t.add_argument("--name", default="bc1")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--max-steps", type=int, default=0, help="оборвать после стольких шагов (проверка конвейера)")
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--rules-weight", type=float, default=3.0)
    t.add_argument("--soft-weight", type=float, default=0.7,
                   help="доля мягкой метки в потере на примерах, где она есть")
    t.add_argument("--amp", choices=("auto", "bf16", "off"), default="auto",
                   help="bf16 на CPU: на Ryzen 7500F в 1.4–1.7 раза быстрее fp32")
    t.add_argument("--threads", type=int, default=10)
    t.add_argument("--log-every", type=int, default=200)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    collect(args) if args.command == "collect" else train(args)


if __name__ == "__main__":
    main()

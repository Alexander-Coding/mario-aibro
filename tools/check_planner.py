"""Проверка планировщика-учителя (ai/planner.py): прохождения и сценарии.

Прохождения: 6 уровней из maps/levels.json и процедурные уровни
ai.levelgen (сложности равномерно по 0.05…1, как у сбора примеров в
ai/imitate.py). По каждому: очки, победа, причина смерти, взято из
возможного (монеты, удары по «?», враги, тайники), время, время решения
(медиана и p95), кадров симуляции на решение и петли (ai/agent.LoopMeter:
доля решений в 3-секундном топтании, число таких отрезков, самое долгое в
секундах, смены «вправо» ↔ «влево» в минуту).

    .venv/bin/python -m tools.check_planner                    # 6 уровней + 20 процедурных
    .venv/bin/python -m tools.check_planner --levels 1-1 --gen 0
    .venv/bin/python -m tools.check_planner --levels none --gen 0 --drills 10   # арены без флага
    .venv/bin/python -m tools.check_planner --scenarios        # загон, возврат, места петель
    .venv/bin/python -m tools.check_planner --loops --workers 3  # петли без памяти: труба позади, игра
    .venv/bin/python -m tools.check_planner --selftest 60      # мир не меняется, ответ детерминирован
    .venv/bin/python -m tools.check_planner --gen 0 --set BEAM=4   # подбор констант ai/planner.py

Сценарии (--scenarios), каждый печатает PASS/FAIL и трассу:
(a) «хранилище» сокровищницы — старт у правой стены, слева и посередине
    загона: выйти за 6 с (и сколько монет загона взято до выхода);
(b) возврат — на 6 уровнях телепорт на 3…GRID_BACK тайлов дальше
    несобранных монет и блоков «?» (всё левее окна ученика считается уже
    собранным): за 5 с взять не меньше половины очков позади и уйти вперёд;
(c) места петель модели mariojev_final: верх последней лестницы 1-1,
    комнаты skybridge и treasury, главный уровень treasury справа от
    загона — без петель дойти до флага / выхода из комнаты / на 15 тайлов.

Петли без памяти (--loops, отдельно от 16 сценариев):
(d) непосещённая труба-вход позади — телепорт на 3/5/7/9 тайлов правее
    (gen67002, gen67003) и под блоками у трубы (gen65001): 10 с без топтания;
(e) естественная игра процедурных уровней, где прежний учитель ходил по
    кругу до конца времени (gen68038, gen7009, gen61016, gen90046, gen90021,
    gen90042, gen90057): победа и ни одного отрезка без очков дольше 10 с на
    участке уже 4 тайлов.

Арены (--drills, ai/levelgen.build_drill, как в сборе примеров с --drill):
флага нет, эпизод обрывается на train_steps решениях (20 с). Победы там
не бывает — смотрим, сколько взято из возможного и нет ли топтания.

Процессы — не больше 6, в каждом один поток (ПК общий).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

OUT_DIR = ROOT / "ai" / "runs"
MAX_WORKERS = 6
GEN_SEED = 7000
DRILL_SEED = 8000

# Приёмка (задача планировщика): отсюда и вердикт в конце
ACCEPT_STALL_FRAC = 0.02
ACCEPT_LONGEST_STALL = 5.0


def _threads() -> None:
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


# ===============================================================
# ПРОХОЖДЕНИЕ
# ===============================================================


def _make_world(spec: dict):
    from core.world import GameWorld

    if spec["kind"] == "level":
        from core.levels import load_levels, make_world
        level = next(level for level in load_levels() if level.id == spec["id"])
        return make_world(level)
    if spec["kind"] == "drill":
        from ai.levelgen import build_drill
        return GameWorld(build_drill(spec["seed"], spec["difficulty"]))
    from ai.levelgen import generate_level
    return GameWorld(generate_level(spec["seed"], spec["difficulty"]))


def _longest_stall(meter) -> float:
    """Самое долгое топтание, секунды: окно LOOP_WINDOW решений плюс
    сколько ещё решений подряд оно не кончалось."""
    from ai.agent import LOOP_SPAN, LOOP_WINDOW
    from ai.env import FRAME_SKIP

    n = len(meter.x)
    if n < LOOP_WINDOW:
        return 0.0
    x = np.asarray(meter.x)
    score = np.asarray(meter.score)
    _, area = np.unique(np.asarray(meter.area), return_inverse=True)
    windows = np.lib.stride_tricks.sliding_window_view
    xs, areas = windows(x, LOOP_WINDOW), windows(area, LOOP_WINDOW)
    stalled = ((xs.max(1) - xs.min(1) < LOOP_SPAN) & (areas.min(1) == areas.max(1))
               & (score[LOOP_WINDOW - 1:] == score[: n - LOOP_WINDOW + 1]))
    best = run = 0
    for flag in stalled:
        run = run + 1 if flag else 0
        best = max(best, run)
    if best == 0:
        return 0.0
    return (best + LOOP_WINDOW - 1) * FRAME_SKIP / 60.0


def _apply_overrides(overrides: dict) -> None:
    """--set ИМЯ=ЗНАЧЕНИЕ: константы ai/planner.py для подбора (в каждом процессе)."""
    if not overrides:
        return
    import ai.planner as planner
    for name, value in overrides.items():
        setattr(planner, name, value)
    planner.HORIZON = sum(planner.STEPS)
    if "STEPS" in overrides and "POLE_STEPS" not in overrides:
        planner.POLE_STEPS = planner.STEPS + (16, 16)


def play(spec: dict) -> dict:
    """Одно прохождение планировщиком; решения — ровно как в игре (act)."""
    _threads()
    _apply_overrides(spec.get("overrides"))
    from ai.agent import LoopMeter
    from ai.env import FRAME_SKIP
    from ai.planner import Planner

    world = _make_world(spec)
    planner = Planner()
    planner.reset(world)
    meter = LoopMeter()
    seconds, frames, sources = [], [], {}
    started = time.perf_counter()
    # Арена обрывается, как в ai/env.MarioEnv: train_steps решений по FRAME_SKIP кадров
    steps = world.level_data.get("train_steps")
    limit = steps * FRAME_SKIP if steps else 60 * 60 * 10
    # Арена: решение, к которому взяты все монеты и удары по «?» (дальше
    # делать нечего, и стояние у стены — не петля)
    cleared = None
    while not world.done and world.frame < limit:
        deciding = not world.cutscene and planner._frame % FRAME_SKIP == 0
        inputs = planner.act(world)
        if deciding:
            if steps and cleared is None and _arena_left(world) == 0:
                cleared = len(meter.x)
            meter.record_inputs(world, inputs)
            seconds.append(planner.last_seconds)
            frames.append(planner.last_frames)
            sources[planner.last_source] = sources.get(planner.last_source, 0) + 1
        world.step(inputs)

    summary = world.summary()
    loops = meter.summary(world.frame)
    loops["longest_s"] = round(_longest_stall(meter), 2)
    if steps:
        # Петли, пока было что брать
        n = len(meter.x) if cleared is None else cleared
        head = LoopMeter()
        head.x, head.area, head.score, head.direction = meter.x[:n], meter.area[:n], meter.score[:n], meter.direction[:n]
        loops["open_stall_frac"] = head.summary(n * FRAME_SKIP)["stall_frac"]
        loops["open_decisions"] = n
        loops["open_longest_s"] = round(_longest_stall(head), 2)
        loops["cleared_s"] = None if cleared is None else round(cleared * FRAME_SKIP / 60, 1)
    return {
        **spec,
        **summary,
        "loops": loops,
        "decide_ms_p50": round(1e3 * statistics.median(seconds), 1) if seconds else 0.0,
        "decide_ms_p95": round(1e3 * float(np.percentile(seconds, 95)), 1) if seconds else 0.0,
        "frames_p50": int(statistics.median(frames)) if frames else 0,
        "sources": sources,
        "wall_s": round(time.perf_counter() - started, 1),
    }


def _arena_left(world) -> int:
    """Сколько на арене ещё брать: монеты и удары по «?»."""
    from Objects.Coin import Coin
    return (sum(isinstance(obj, Coin) for obj in world.main.objects)
            + sum(max(block.state, 0) for block in world.main.blocks))


def _pct(got: float, total: float) -> str:
    return f"{got:.0f}/{total:.0f}" + (f" ({100 * got / total:.0f}%)" if total else "")


def print_row(r: dict) -> None:
    loops = r["loops"]
    print(f"  {r['name']:30s} {'ПОБЕДА' if r['won'] else 'смерть: ' + str(r['death']):16s} очки {r['score']:6d}  "
          f"монеты {_pct(r['coins'], r['coins_total']):13s} «?» {_pct(r['blocks'], r['block_hits_total']):12s} "
          f"враги {r['enemies']}/{r['enemies_total']}  тайники {r['secrets']}/{r['secrets_total']}  "
          f"время {r['time_used']:.0f} с  флаг {r['flag_bonus']}  решение {r['decide_ms_p50']:.0f}/{r['decide_ms_p95']:.0f} мс "
          f"({r['frames_p50']} кадров)  петли {100 * loops['stall_frac']:.1f}% ×{loops['stalls']} "
          f"(дольше всего {loops['longest_s']:.1f} с)  L/R {loops['flips_per_min']:.1f}/мин"
          + (f"  пока было что брать: петли {100 * loops['open_stall_frac']:.1f}% (дольше всего "
             f"{loops['open_longest_s']:.1f} с), всё взято {loops['cleared_s'] if loops['cleared_s'] is not None else '—'} с"
             if "open_stall_frac" in loops else ""), flush=True)


def totals(rows: list[dict]) -> dict:
    decisions = sum(r["loops"]["decisions"] for r in rows)
    minutes = sum(r["frames"] for r in rows) / 3600
    sums = {key: sum(r[key] for r in rows) for key in
            ("score", "coins", "coins_total", "blocks", "block_hits_total", "enemies", "enemies_total",
             "secrets", "secrets_total")}
    return {
        **sums,
        "levels": len(rows),
        "wins": sum(bool(r["won"]) for r in rows),
        "stall_frac": sum(r["loops"]["stall_frac"] * r["loops"]["decisions"] for r in rows) / max(decisions, 1),
        "stalls": sum(r["loops"]["stalls"] for r in rows),
        "longest_stall_s": max((r["loops"]["longest_s"] for r in rows), default=0.0),
        "flips_per_min": sum(r["loops"]["flips"] for r in rows) / minutes if minutes else 0.0,
        "decide_ms_p50": statistics.median([r["decide_ms_p50"] for r in rows]) if rows else 0.0,
        "decide_ms_p95_max": max((r["decide_ms_p95"] for r in rows), default=0.0),
        **_open_totals(rows),
    }


def _open_totals(rows: list[dict]) -> dict:
    """Арены: петли, пока было что брать, и сколько арен собрано целиком."""
    arenas = [r for r in rows if "open_stall_frac" in r["loops"]]
    if not arenas:
        return {}
    decisions = sum(r["loops"]["open_decisions"] for r in arenas)
    return {
        "open_stall_frac": sum(r["loops"]["open_stall_frac"] * r["loops"]["open_decisions"] for r in arenas)
        / max(decisions, 1),
        "open_longest_s": max(r["loops"]["open_longest_s"] for r in arenas),
        "cleared": sum(r["loops"]["cleared_s"] is not None for r in arenas),
    }


def print_totals(label: str, t: dict) -> None:
    print(f"  ИТОГО {label}: побед {t['wins']}/{t['levels']}, очки {t['score']}, "
          f"монеты {_pct(t['coins'], t['coins_total'])}, «?» {_pct(t['blocks'], t['block_hits_total'])}, "
          f"враги {t['enemies']}/{t['enemies_total']}, тайники {t['secrets']}/{t['secrets_total']}, "
          f"петли {100 * t['stall_frac']:.2f}% ×{t['stalls']} (дольше всего {t['longest_stall_s']:.1f} с), "
          f"L/R {t['flips_per_min']:.1f}/мин, решение медиана {t['decide_ms_p50']:.0f} мс"
          + (f"; пока было что брать: петли {100 * t['open_stall_frac']:.2f}% (дольше всего "
             f"{t['open_longest_s']:.1f} с), собраны целиком {t['cleared']}/{t['levels']}" if "cleared" in t else ""),
          flush=True)


def run_levels(args) -> dict:
    from multiprocessing import get_context

    from core.levels import load_levels

    specs = []
    wanted = set(args.levels.split(",")) if args.levels else None
    for level in load_levels():
        if wanted is None or level.id in wanted:
            specs.append({"kind": "level", "id": level.id, "name": level.id, "split": level.split})
    for i in range(args.gen):
        difficulty = round(0.05 + 0.95 * i / max(args.gen - 1, 1), 3)
        seed = args.seed + i
        specs.append({"kind": "gen", "seed": seed, "difficulty": difficulty, "name": f"gen {seed} d{difficulty:.2f}"})
    for i in range(args.drills):
        difficulty = round(0.05 + 0.95 * i / max(args.drills - 1, 1), 3)
        seed = args.drill_seed + i
        specs.append({"kind": "drill", "seed": seed, "difficulty": difficulty,
                      "name": f"арена {seed} d{difficulty:.2f}"})

    for spec in specs:
        spec["overrides"] = args.overrides
    workers = max(1, min(args.workers, MAX_WORKERS, len(specs)))
    print(f"Прохождений: {len(specs)}, процессов: {workers}", flush=True)
    rows: list[dict] = []
    with get_context("spawn").Pool(workers) as pool:
        for row in pool.imap_unordered(play, specs):
            print_row(row)
            rows.append(row)

    order = {spec["name"]: i for i, spec in enumerate(specs)}
    rows.sort(key=lambda r: order[r["name"]])
    manual = [r for r in rows if r["kind"] == "level"]
    generated = [r for r in rows if r["kind"] == "gen"]
    drills = [r for r in rows if r["kind"] == "drill"]
    print("\nПо порядку:")
    for r in rows:
        print_row(r)
    out = {"rows": rows}
    print()
    if manual:
        out["levels_total"] = totals(manual)
        print_totals("6 уровней" if len(manual) == 6 else "уровни", out["levels_total"])
    if generated:
        out["gen_total"] = totals(generated)
        print_totals("процедурные", out["gen_total"])
    if drills:
        out["drill_total"] = totals(drills)
        print_totals("арены", out["drill_total"])
    # Петли — главный критерий: ни одного топтания дольше порога
    for label, key in (("уровни", "levels_total"), ("процедурные", "gen_total"), ("арены", "drill_total")):
        t = out.get(key)
        if t:
            # На арене считается только время, пока было что брать
            frac, longest = ((t["open_stall_frac"], t["open_longest_s"]) if "cleared" in t
                             else (t["stall_frac"], t["longest_stall_s"]))
            ok = frac < ACCEPT_STALL_FRAC and longest <= ACCEPT_LONGEST_STALL
            print(f"  Петли ({label}): доля {100 * frac:.2f}% (порог {100 * ACCEPT_STALL_FRAC:.0f}%), "
                  f"дольше всего {longest:.1f} с (порог {ACCEPT_LONGEST_STALL:.0f} с) — "
                  f"{'OK' if ok else 'FAIL'}")
    return out


# ===============================================================
# СЦЕНАРИИ
# ===============================================================


TILE = 32
DECISIONS_PER_S = 15

# (a) «Хранилище» сокровищницы: загон на главном уровне. Стены — тайлы 70 и
# 80 (ряды 6–8), пол — ряд 5, крыша — ряд 9 с дырой 74–76: выход только
# через дыру, в 4–6 тайлах левее правой стены
PEN = {"left_wall": 70, "right_wall": 80, "floor": 5, "roof": 9, "gap": (74, 76)}
PEN_STARTS = {"справа у стены": 79, "слева": 71, "посередине": 75}
PEN_LIMIT_S = 6.0

# (b) Возврат: телепорт на 3…GRID_BACK тайлов дальше предметов; за
# BACK_LIMIT_S взять не меньше половины очков позади и уйти вперёд
BACK_LIMIT_S = 5.0
FORWARD_LIMIT_S = 10.0
BACK_LEVELS = ("1-1", "warmup", "goombas", "pipes", "skybridge", "treasury")

# (c) Места, где зацикливалась модель mariojev_final (ai/runs/diag_loops.json):
# 1-1 — верх последней лестницы перед флагом (y≈17), комнаты skybridge и
# treasury, главный уровень treasury у правой стены загона. tx="stairs" —
# левый край самого высокого ряда, где можно стоять, в столбцах 185–225
SPOTS = (
    {"level": "1-1", "room": False, "tx": "stairs", "row": "top", "label": "1-1: верх последней лестницы"},
    {"level": "skybridge", "room": True, "tx": 16, "row": "floor", "label": "skybridge, комната: x≈16 на полу"},
    {"level": "skybridge", "room": True, "tx": 15, "row": "top", "label": "skybridge, комната: x≈15 на блоках"},
    {"level": "treasury", "room": True, "tx": 8, "row": "top", "label": "treasury, комната: x≈8 на блоках"},
    {"level": "treasury", "room": True, "tx": 9, "row": "floor", "label": "treasury, комната: x≈9 на полу"},
    {"level": "treasury", "room": True, "tx": 7, "row": "floor", "label": "treasury, комната: x≈7 на полу"},
    {"level": "treasury", "room": False, "tx": 82, "row": "floor", "label": "treasury: x≈82, справа от загона"},
)
SPOT_LIMIT_S = 40.0
# На главном уровне цель места — уйти вперёд на столько тайлов (или флаг)
SPOT_ADVANCE = 15

# (d), (e) — отдельный режим --loops: места, где учитель без памяти ходил по
# кругу (проверки ai/runs/notes_verify_planner.md, ai/runs/notes_loopfix.md).
# (d) Непосещённая труба-вход позади: телепорт на dx тайлов правее трубы
# (seed, сложность, x трубы в тайлах, dx, обеднить блоки рядом); за
# WARP_LIMIT_S — без топтания (войти не обязательно: на 9 тайлах труба уже
# дальше радиуса потенциала)
LOOP_WARPS = tuple((seed, d, pipe, dx, False) for seed, d, pipe in ((67002, 0.28, 53), (67003, 0.37, 71))
                   for dx in (3, 5, 7, 9)) + ((65001, 0.26, None, 60, True), (65001, 0.26, None, 60, False))
WARP_LIMIT_S = 10.0
# (e) Естественная игра: уровень целиком; прежний учитель стоял или качался
# до конца времени (68038, 7009, 61016, 90046), промежуточная версия с
# потенциалом стояла у монет на ряду блоков над платформой (90021, 90042,
# 90057). PASS — победа и ни одного отрезка без очков дольше
# NATURAL_NOSCORE_S на участке уже 4 тайлов
LOOP_NATURAL = ((68038, 0.976), (7009, 0.5), (61016, 0.85), (90046, 0.791), (90021, 0.388), (90042, 0.726),
                (90057, 0.968))
NATURAL_NOSCORE_S = 10.0
NATURAL_SPAN = 4.0


def scenario_specs(only: str | None = None) -> list[dict]:
    specs = []
    for where in PEN_STARTS:
        specs.append({"kind": "pen", "start": where, "name": f"(a) загон treasury: {where}"})
    for level in BACK_LEVELS:
        specs.append({"kind": "back", "level": level, "name": f"(b) возврат: {level}"})
    for spot in SPOTS:
        specs.append({"kind": "spot", **spot, "name": f"(c) {spot['label']}"})
    if only:
        specs = [s for s in specs if only.lower() in s["name"].lower()]
    return specs


def _level_world(level_id: str):
    from core.levels import load_levels, make_world
    return make_world(next(level for level in load_levels() if level.id == level_id))


def _teleport(world, tx: int, ty: int) -> None:
    """Игрок стоит на клетке (tx, ty): ноги на ty·32, середина тайла."""
    player = world.player
    player.center_x = tx * TILE + TILE / 2
    player.bottom = ty * TILE
    player.change_x = 0
    player.change_y = 0
    world.on_ground = True
    if not world.in_room:
        world.max_x = max(world.max_x, player.center_x)


def _standable(world, tx: int, ty: int) -> bool:
    return not world.solid_at(tx, ty) and world.solid_at(tx, ty - 1)


def _keys(inputs) -> str:
    return "".join(k for k, on in zip("LRJD", (inputs.left, inputs.right, inputs.jump, inputs.down)) if on) or "-"


class _Runner:
    """Играет планировщиком по кадрам и пишет короткую трассу."""

    def __init__(self, world):
        from ai.agent import LoopMeter
        from ai.planner import Planner

        self.world = world
        self.planner = Planner()
        self.planner.reset(world)
        self.meter = LoopMeter()
        self.trace: list[str] = []
        self.decisions = 0
        self.seconds: list[float] = []

    def step(self) -> None:
        from ai.env import FRAME_SKIP
        world, planner = self.world, self.planner
        deciding = not world.cutscene and planner._frame % FRAME_SKIP == 0
        inputs = planner.act(world)
        if deciding:
            self.meter.record_inputs(world, inputs)
            self.seconds.append(planner.last_seconds)
            if self.decisions % 8 == 0:
                p = world.player
                self.trace.append(
                    f"t={world.frame / 60:5.2f}с {world.area.key:10s} x={p.center_x / TILE:6.2f} "
                    f"y={p.bottom / TILE:5.2f} {_keys(inputs):3s} очки {world.score}")
            self.decisions += 1
        world.step(inputs)

    def loops(self) -> dict:
        out = self.meter.summary(max(self.world.frame, 1))
        out["longest_s"] = round(_longest_stall(self.meter), 2)
        return out


def _scenario_pen(spec: dict) -> dict:
    world = _level_world("treasury")
    g = PEN
    # Геометрия загона: если уровень поменяли — сценарий честно падает
    rows = range(g["floor"] + 1, g["roof"])
    ok = (all(world.solid_at(g["left_wall"], y) and world.solid_at(g["right_wall"], y) for y in rows)
          and all(world.solid_at(x, g["floor"]) for x in range(g["left_wall"], g["right_wall"] + 1))
          and all(not world.solid_at(x, g["roof"]) for x in range(g["gap"][0], g["gap"][1] + 1))
          and all(world.solid_at(x, g["roof"]) for x in range(g["left_wall"], g["gap"][0]))
          and all(world.solid_at(x, g["roof"]) for x in range(g["gap"][1] + 1, g["right_wall"] + 1)))
    if not ok:
        return {"pass": False, "verdict": "геометрия загона не совпала с ожидаемой", "trace": []}

    from Objects.Coin import Coin
    inside = [c for c in world.main.objects if isinstance(c, Coin)
              and g["left_wall"] * TILE < c.center_x < g["right_wall"] * TILE
              and g["floor"] * TILE < c.center_y < g["roof"] * TILE]
    tx = PEN_STARTS[spec["start"]]
    _teleport(world, tx, g["floor"] + 1)
    run = _Runner(world)
    start = world.frame
    left_at = None
    limit = int(20 * 60)
    while not world.done and world.frame - start < limit:
        run.step()
        p = world.player
        x_in = g["left_wall"] * TILE < p.center_x < (g["right_wall"] + 1) * TILE
        out = (not x_in) or (world.on_ground and p.bottom >= (g["roof"] + 1) * TILE)
        if out and left_at is None:
            left_at = (world.frame - start) / 60
            coins_at_exit = sum(1 for c in inside if not c.sprite_lists)
        if left_at is not None and world.frame - start > (left_at + 4) * 60:
            break
    loops = run.loops()
    passed = left_at is not None and left_at <= PEN_LIMIT_S
    coins = coins_at_exit if left_at is not None else sum(1 for c in inside if not c.sprite_lists)
    verdict = (f"вышел за {left_at:.2f} с (порог {PEN_LIMIT_S:.0f} с)" if left_at is not None
               else f"не вышел за {limit / 60:.0f} с") + f", монет в загоне взял {coins}/{len(inside)} до выхода; " \
              f"петли {loops['stalls']} (дольше всего {loops['longest_s']:.1f} с)"
    return {"pass": bool(passed), "verdict": verdict, "left_s": left_at, "coins": coins, "coins_total": len(inside),
            "loops": loops, "trace": run.trace[:40]}


def _backtrack_spot(world) -> dict | None:
    """Место на ровной земле, позади которого (3…GRID_BACK тайлов, в окне
    ученика по высоте) есть несобранные монеты и блоки «?» с ударами,
    достижимые по карте проходимости планировщика; врагов ближе 8 тайлов нет.
    Из таких — где позади больше всего очков, лучше с блоком и монетами сразу."""
    from ai import features
    from ai.planner import NavField
    from Objects.Coin import Coin

    back = features.GRID_BACK
    main = world.main
    nav = NavField(world, main)
    enemies = [e.center_x / TILE for e in main.enemies]

    def reachable_coin(c) -> bool:
        tx, ty = int(c.center_x // TILE), int(c.center_y // TILE)
        if 0 <= tx < nav.nx and 0 <= ty < nav.ny and nav.standable[tx, ty]:
            return True
        return any(nav.cells_below(tx + side, ty) for side in (0, -1, 1))

    coins_all = [c for c in main.objects if isinstance(c, Coin) and reachable_coin(c)]
    blocks_all = [b for b in main.blocks if b.state > 0
                  and nav.cells_below(int(b.center_x // TILE), int(b.center_y // TILE) - 1)]
    best = None
    for fy in range(1, 20):
        for tx in range(12, int(world.geometry.door_x // TILE) - 12):
            # Ровная площадка под телепортом (4 тайла) и нет врагов рядом
            if not all(_standable(world, x, fy) for x in range(tx - 3, tx + 1)):
                continue
            if any(abs(e - tx) < 8 for e in enemies):
                continue
            # Труба в тайник рядом увела бы в комнату — не про возврат
            if any(abs(w.x / TILE - tx) < 20 for w in main.warps):
                continue
            lo, hi = tx - back, tx - 3
            below, above = fy - features.GRID_BELOW, fy + features.GRID_ABOVE
            blocks = [b for b in blocks_all if lo <= int(b.center_x // TILE) <= hi
                      and below <= int(b.center_y // TILE) <= above]
            coins = [c for c in coins_all if lo <= int(c.center_x // TILE) <= hi
                     and below <= int(c.center_y // TILE) <= above]
            if not coins and not blocks:
                continue
            worth = 100 * (len(coins) + sum(b.state for b in blocks))
            key = (bool(blocks) and bool(coins), worth)
            if best is None or key > best[0]:
                best = (key, {"tx": tx, "ty": fy, "blocks": blocks, "coins": coins, "worth": worth})
    return None if best is None else best[1]


def _scenario_back(spec: dict) -> dict:
    world = _level_world(spec["level"])
    spot = _backtrack_spot(world)
    if spot is None:
        return {"pass": False, "verdict": "не нашлось места: ровная земля, позади монеты или блок «?»", "trace": []}
    tx, ty, blocks, coins, worth = spot["tx"], spot["ty"], spot["blocks"], spot["coins"], spot["worth"]
    hits0 = {id(b): b.state for b in blocks}

    # Прошлое как при обычной игре: всё левее окна ученика уже собрано и
    # побеждено, иначе учитель (честно) шёл бы собирать весь уровень назад
    from ai import features
    from Objects.Coin import Coin
    edge = (tx - features.GRID_BACK) * TILE
    for c in [c for c in world.main.objects if isinstance(c, Coin) and c.center_x < edge]:
        c.remove_from_sprite_lists()
    for b in world.main.blocks:
        while b.center_x < edge and b.state > 0:
            b.hit()
    for e in [e for e in world.main.enemies if e.center_x < edge]:
        e.remove_from_sprite_lists()
        world.main.enemy_engines.pop(e, None)

    def collected() -> int:
        return 100 * (sum(hits0[id(b)] - b.state for b in blocks) + sum(1 for c in coins if not c.sprite_lists))

    _teleport(world, tx, ty)
    run = _Runner(world)
    start = world.frame
    got_at = None
    forward_at = None
    while not world.done and world.frame - start < (BACK_LIMIT_S + FORWARD_LIMIT_S) * 60:
        run.step()
        t = (world.frame - start) / 60
        if got_at is None and 2 * collected() >= worth:
            got_at = t
        if got_at is not None and world.player.center_x > (tx + 3) * TILE and not world.in_room:
            forward_at = t
            break
    loops = run.loops()
    passed = got_at is not None and got_at <= BACK_LIMIT_S and forward_at is not None
    where = (f"телепорт в ({tx}, {ty}); позади: блоки «?» {[int(b.center_x // TILE) for b in blocks]}, "
             f"монеты {sorted({int(c.center_x // TILE) for c in coins})} — всего {worth} очков")
    verdict = (f"{where}; взял {collected()} "
               + (f"(половину — за {got_at:.2f} с)" if got_at is not None else "(меньше половины)")
               + (f", вперёд ушёл к {forward_at:.2f} с" if forward_at is not None else ", вперёд не ушёл")
               + f"; петли {loops['stalls']}")
    return {"pass": bool(passed), "verdict": verdict, "got_s": got_at, "forward_s": forward_at,
            "collected": collected(), "worth": worth, "loops": loops, "trace": run.trace[:40]}


def _scenario_spot(spec: dict) -> dict:
    world = _level_world(spec["level"])
    if spec["room"]:
        warp = world.warps[0]
        world._go_to_room(warp)
    tx = spec["tx"]
    if tx == "stairs":
        top = max(y for x in range(185, 226) for y in range(1, 40) if _standable(world, x, y))
        tx = min(x for x in range(185, 226) if _standable(world, x, top))
    rows = [y for y in range(1, 40) if _standable(world, tx, y)]
    if not rows:
        return {"pass": False, "verdict": f"в столбце {tx} негде стоять", "trace": []}
    # «top» — самая высокая стоячая клетка столбца под потолком части, «floor» — самая низкая
    inside = [y for y in rows if (y + 2) * TILE <= world.area.top] or rows
    ty = max(inside) if spec["row"] == "top" else min(inside)
    _teleport(world, tx, ty)
    run = _Runner(world)
    start = world.frame
    reached = None
    while not world.done and world.frame - start < SPOT_LIMIT_S * 60:
        in_room = world.in_room
        run.step()
        if spec["room"] and in_room and not world.in_room:
            reached = (world.frame - start) / 60
            break
        if not spec["room"] and spec["tx"] != "stairs" and world.player.center_x >= (tx + SPOT_ADVANCE) * TILE:
            reached = (world.frame - start) / 60
            break
    if not spec["room"] and world.won:
        reached = (world.frame - start) / 60
    loops = run.loops()
    passed = reached is not None and loops["stalls"] == 0
    goal = ("выход из комнаты" if spec["room"] else "флаг" if spec["tx"] == "stairs"
            else f"{SPOT_ADVANCE} тайлов вперёд")
    verdict = (f"старт ({tx}, {ty}); " + (f"{goal} за {reached:.2f} с" if reached is not None else f"{goal} не достигнут")
               + f"; петли {loops['stalls']} (дольше всего {loops['longest_s']:.1f} с), очки {world.score}, "
               f"флаг {world.flag_bonus}")
    return {"pass": bool(passed), "verdict": verdict, "reached_s": reached, "loops": loops,
            "trace": run.trace[:40]}


def loop_specs(only: str | None = None) -> list[dict]:
    specs = []
    for seed, difficulty, pipe, dx, deplete in LOOP_WARPS:
        where = f"труба x={pipe}, +{dx}" if pipe is not None else f"x={dx}" + (", блоки пусты" if deplete else "")
        specs.append({"kind": "warp", "seed": seed, "difficulty": difficulty, "pipe": pipe, "dx": dx,
                      "deplete": deplete, "name": f"(d) труба позади gen{seed}: {where}"})
    for seed, difficulty in LOOP_NATURAL:
        specs.append({"kind": "natural", "seed": seed, "difficulty": difficulty,
                      "name": f"(e) игра gen{seed} d{difficulty:.2f}"})
    if only:
        specs = [s for s in specs if only.lower() in s["name"].lower()]
    return specs


def _noscore(rows: list[tuple], span: float) -> float:
    """Самый долгий отрезок (с) без очков в одной части уровня с размахом X < span тайлов.
    rows — (кадр, часть, x в тайлах, очки) на каждом решении."""
    longest = 0.0
    j = 0
    for i in range(len(rows)):
        while j < i and (rows[j][1] != rows[i][1] or rows[j][3] != rows[i][3]
                         or max(r[2] for r in rows[j:i + 1]) - min(r[2] for r in rows[j:i + 1]) >= span):
            j += 1
        longest = max(longest, (rows[i][0] - rows[j][0]) / 60)
    return longest


def _scenario_warp(spec: dict) -> dict:
    """(d): старт правее непосещённой трубы-входа; топтание — FAIL."""
    world = _make_world(spec)
    if spec["pipe"] is not None:
        warp = min(world.main.warps, key=lambda w: abs(w.x / TILE - spec["pipe"]))
        tx = int((warp.x + TILE) // TILE) + spec["dx"]
    else:
        tx = spec["dx"]
    if spec["deplete"]:
        for block in world.main.blocks:
            if abs(block.center_x / TILE - tx) < 12:
                while block.state > 0:
                    block.hit()
    rows = [y for y in range(1, 30) if _standable(world, tx, y)]
    _teleport(world, tx, min(rows))
    run = _Runner(world)
    start = world.frame
    while not world.done and world.frame - start < WARP_LIMIT_S * 60 and not (world.in_room or world.cutscene):
        run.step()
    entered = world.in_room or world.cutscene
    loops = run.loops()
    passed = loops["stalls"] == 0
    verdict = (f"старт ({tx}, {min(rows)}); " + (f"вошёл за {(world.frame - start) / 60:.2f} с" if entered
                                                 else "не входил") +
               f"; топтаний {loops['stalls']}, смен направления {loops['flips']}")
    return {"pass": bool(passed), "verdict": verdict, "entered": bool(entered), "loops": loops,
            "trace": run.trace[:20]}


def _scenario_natural(spec: dict) -> dict:
    """(e): уровень целиком; победа и нет долгого отрезка без очков на месте."""
    from ai.env import FRAME_SKIP
    world = _make_world(spec)
    run = _Runner(world)
    rows = []
    while not world.done:
        if not world.cutscene and run.planner._frame % FRAME_SKIP == 0:
            rows.append((world.frame, world.area.key, world.player.center_x / TILE, world.score))
        run.step()
    longest = _noscore(rows, NATURAL_SPAN)
    loops = run.loops()
    passed = world.won and longest <= NATURAL_NOSCORE_S
    verdict = (f"{'победа' if world.won else 'нет победы: ' + str(world.death_cause)}"
               f", очки {world.score}, время {world.frame / 60:.0f} с; без очков на {NATURAL_SPAN:.0f} тайлах "
               f"дольше всего {longest:.1f} с; топтаний {loops['stalls']}, смен направления {loops['flips_per_min']}/мин")
    return {"pass": bool(passed), "verdict": verdict, "noscore_s": round(longest, 2), "loops": loops,
            "trace": run.trace[:0]}


def run_scenario(spec: dict) -> dict:
    _threads()
    _apply_overrides(spec.get("overrides"))
    kind = spec["kind"]
    if kind == "pen":
        result = _scenario_pen(spec)
    elif kind == "back":
        result = _scenario_back(spec)
    elif kind == "warp":
        result = _scenario_warp(spec)
    elif kind == "natural":
        result = _scenario_natural(spec)
    else:
        result = _scenario_spot(spec)
    result["name"] = spec["name"]
    return result


def run_scenarios(args) -> dict:
    from multiprocessing import get_context

    specs = loop_specs(args.only) if args.loops else scenario_specs(args.only)
    for spec in specs:
        spec["overrides"] = args.overrides
    workers = max(1, min(args.workers, MAX_WORKERS, len(specs)))
    print(f"Сценариев: {len(specs)}, процессов: {workers}", flush=True)
    results = []
    with get_context("spawn").Pool(workers) as pool:
        for result in pool.imap_unordered(run_scenario, specs):
            results.append(result)
            print(f"{'PASS' if result['pass'] else 'FAIL'}  {result['name']}: {result['verdict']}", flush=True)
            for line in result["trace"]:
                print("      " + line)
    order = {spec["name"]: i for i, spec in enumerate(specs)}
    results.sort(key=lambda r: order[r["name"]])
    passed = sum(r["pass"] for r in results)
    print(f"\nСценарии: {passed}/{len(results)} PASS")
    for r in results:
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['name']}: {r['verdict']}")
    return {"scenarios": results, "passed": passed, "total": len(results)}


# ===============================================================
# САМОПРОВЕРКА ИНТЕРФЕЙСА
# ===============================================================


def _deep_state(world) -> tuple:
    """Всё наблюдаемое состояние мира, которое мог бы испортить поиск."""
    from Objects.Coin import Coin
    p = world.player
    areas = []
    for area in world.areas.values():
        areas.append((
            area.key,
            tuple((round(e.center_x, 4), round(e.center_y, 4), e.change_x, e.change_y, e.is_alive,
                   getattr(e, "scored", False)) for e in area.enemies),
            sum(isinstance(o, Coin) for o in area.objects),
            tuple(b.state for b in area.blocks),
            len(area.active_enemies),
        ))
    return (world.frame, world.phase, world.score, world.area.key, world._frames_left, world.on_ground,
            p.center_x, p.center_y, p.change_x, p.change_y, p.health, p.state, id(p.texture),
            tuple(sorted(world.stats.items())), tuple(sorted(world.rooms_visited)), tuple(areas))


def run_selftest(args) -> dict:
    """decide() оставляет мир как был; одно состояние — один ответ; метки в порядке."""
    import copy

    from ai.env import FRAME_SKIP, action_inputs
    from ai.planner import DOWN, N_ACTIONS, Planner

    problems = []
    checked = 0
    for level_id in ("1-1", "treasury"):
        world = _level_world(level_id)
        planner = Planner()
        planner.reset(world)
        decisions = 0
        while not world.done and decisions < args.selftest:
            if world.cutscene or planner._frame % FRAME_SKIP:
                world.step(planner.act(world))
                continue
            before = _deep_state(world)
            last = planner._action
            action = planner.decide(world, last)
            q, probs = planner.last_q.copy(), planner.last_probs.copy()
            if _deep_state(world) != before:
                problems.append(f"{level_id}: решение {decisions} изменило мир")
            # Тот же ответ у свежего планировщика — дважды, и у того, что
            # играет с начала уровня: памяти между решениями нет, ответ —
            # функция состояния и прошлого действия
            fresh = [Planner(), Planner()]
            answers = []
            for other in fresh:
                other.reset(world)
                answers.append((other.decide(world, last), copy.copy(other.last_q.tolist())))
            if answers[0] != answers[1]:
                problems.append(f"{level_id}: решение {decisions} не детерминировано")
            if answers[0] != (action, q.tolist()):
                problems.append(f"{level_id}: решение {decisions} зависит от истории (свежий планировщик ответил иначе)")
            if abs(probs.sum() - 1.0) > 1e-6 or (probs < 0).any():
                problems.append(f"{level_id}: решение {decisions}: вероятности не распределение")
            if planner.last_source == "plan" and np.isfinite(q[DOWN]):
                problems.append(f"{level_id}: решение {decisions}: «вниз» без правила")
            if q.shape != (N_ACTIONS,) or not np.isfinite(q[action]) or q[action] < q.max() - 1e-6:
                problems.append(f"{level_id}: решение {decisions}: выбран не лучший по last_q")
            checked += 1
            # Дальше — как в игре: act() повторит выбор на 4 кадра
            planner._action = action
            planner._frame = 1
            world.step(action_inputs(action, True))
            decisions += 1
    for line in problems[:20]:
        print("  " + line)
    print(f"Самопроверка: решений {checked}, замечаний {len(problems)} — {'OK' if not problems else 'FAIL'}")
    return {"selftest_decisions": checked, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default=None, help="id уровней через запятую (по умолчанию все 6)")
    parser.add_argument("--gen", type=int, default=20, help="сколько процедурных уровней")
    parser.add_argument("--seed", type=int, default=GEN_SEED, help="seed первого процедурного уровня")
    parser.add_argument("--drills", type=int, default=0, help="сколько арен без флага (ai/levelgen.build_drill)")
    parser.add_argument("--drill-seed", type=int, default=DRILL_SEED, help="seed первой арены")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--scenarios", action="store_true", help="сценарии вместо прохождений")
    parser.add_argument("--loops", action="store_true",
                        help="места петель без памяти: труба позади, естественная игра (вместо прохождений)")
    parser.add_argument("--selftest", type=int, default=0, metavar="N",
                        help="проверка интерфейса на N решениях двух уровней вместо прохождений")
    parser.add_argument("--only", default=None, help="сценарии, в имени которых есть эта строка")
    parser.add_argument("--json", default=None, help="куда сохранить (по умолчанию ai/runs/check_planner*.json)")
    parser.add_argument("--set", action="append", default=[], metavar="ИМЯ=ЗНАЧЕНИЕ",
                        help="подменить константу ai/planner.py (подбор), например --set BEAM=3")
    args = parser.parse_args()
    import ast
    args.overrides = {}
    for item in args.set:
        name, _, value = item.partition("=")
        args.overrides[name.strip()] = ast.literal_eval(value.strip())

    started = time.perf_counter()
    if args.selftest:
        out = run_selftest(args)
        default = OUT_DIR / "check_planner_selftest.json"
    elif args.scenarios or args.loops:
        out = run_scenarios(args)
        default = OUT_DIR / ("check_planner_loops.json" if args.loops else "check_planner_scenarios.json")
    else:
        out = run_levels(args)
        default = OUT_DIR / "check_planner.json"
    out["wall_s"] = round(time.perf_counter() - started, 1)

    from ai import planner
    _apply_overrides(args.overrides)
    out["overrides"] = args.overrides
    out["constants"] = {name: getattr(planner, name) for name in dir(planner)
                        if name.isupper() and isinstance(getattr(planner, name), (int, float, tuple))}

    path = Path(args.json) if args.json else default
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(f"\nСохранено: {path} ({out['wall_s']:.0f} с)")


if __name__ == "__main__":
    main()

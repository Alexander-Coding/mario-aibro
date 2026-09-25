"""Турнир: кто наберёт больше очков.

Каждый участник проходит каждый уровень манифеста. Побеждает сумма очков,
а не число пройденных уровней. Случайные участники (и модель в режиме
сэмплирования) играют несколько раз — берётся среднее.

    .venv/bin/python -m tools.tournament
    .venv/bin/python -m tools.tournament --model ai/models/mariojev_final.pt --runs 5
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

from core.levels import load_levels
from core.world import GameWorld, Inputs
from tools.heuristic_bot import HeuristicBot

OUT_DIR = Path(__file__).resolve().parent.parent / "ai" / "runs"
MAX_FRAMES = 60 * 400


class RandomAgent:
    """Жмёт случайное из действий ai.features.ACTIONS раз в 4 кадра."""

    name = "Случайный"
    stochastic = True

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._frame = 0
        self._action = 0

    def reset(self, world=None):
        self._frame = 0

    def act(self, world):
        from ai.env import FRAME_SKIP, action_inputs
        from ai.features import N_ACTIONS
        first = self._frame % FRAME_SKIP == 0
        if first:
            self._action = self.rng.randrange(N_ACTIONS)
        self._frame += 1
        return action_inputs(self._action, first)


class JumperAgent:
    """Всегда «вправо+прыжок». В него вырождается argmax недообученной
    PPO-политики, и на части уровней он удивительно силён — поэтому это
    обязательная планка: модель должна обыгрывать и его."""

    name = "Прыгун (всегда вправо+прыжок)"
    stochastic = False

    def __init__(self):
        self._frame = 0

    def reset(self, world=None):
        self._frame = 0

    def act(self, world):
        from ai.env import FRAME_SKIP, action_inputs
        first = self._frame % FRAME_SKIP == 0
        self._frame += 1
        return action_inputs(4, first)


class HeuristicAgent(HeuristicBot):
    name = "Эвристика"
    stochastic = False


# Что считаем «взятым» и где в summary() лежит «сколько было всего»
COLLECTED = ("coins", "blocks", "enemies", "secrets")
TOTAL_KEYS = {"coins": "coins_total", "blocks": "block_hits_total", "enemies": "enemies_total", "secrets": "secrets_total"}


def play(agent, level) -> dict:
    from ai.agent import LoopMeter
    from ai.env import FRAME_SKIP

    world = GameWorld(level.path, name=level.name, time_limit=level.time_limit)
    agent.reset(world)
    # Петли: на каждом кадре решения — X, область, счёт и направление нажатия
    loops = LoopMeter()
    started = time.perf_counter()
    while not world.done and world.frame < MAX_FRAMES:
        decision = world.frame % FRAME_SKIP == 0 and not world.cutscene
        inputs = agent.act(world)
        if decision:
            loops.record_inputs(world, inputs)
        world.step(inputs)
    summary = world.summary()
    summary["wall_s"] = round(time.perf_counter() - started, 2)
    summary["loops"] = loops.summary(world.frame)
    return summary


def loop_totals(games: list[dict]) -> dict:
    """Петли по нескольким играм: доля — по всем решениям, смены — на минуту всех игр."""
    decisions = sum(g["loops"]["decisions"] for g in games)
    minutes = sum(g["frames"] for g in games) / 3600
    return {
        "stall_frac": sum(g["loops"]["stall_frac"] * g["loops"]["decisions"] for g in games) / max(decisions, 1),
        "stalls": sum(g["loops"]["stalls"] for g in games) / max(len(games), 1),
        "flips_per_min": sum(g["loops"]["flips"] for g in games) / minutes if minutes else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=None, help="чекпойнт MarioJev (можно несколько)")
    parser.add_argument("--runs", type=int, default=3, help="повторов для случайных участников")
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--temperatures", default="1.0", help="температуры сэмплирования моделей через запятую")
    parser.add_argument("--no-argmax", action="store_true")
    parser.add_argument("--laya", action="store_true", help="добавить Laya (открытая альтернатива Jev) без дообучения")
    parser.add_argument("--teacher", default=None,
                        help="добавить учителя ai/teacher.py (модель по этому пути + правила сбора) — потолок подражания")
    parser.add_argument("--planner", action="store_true",
                        help="добавить планировщик ai/planner.py — учителя ученика v2")
    parser.add_argument("--only", default=None, help="играть только участников, в имени которых есть эта строка")
    parser.add_argument("--levels", default=None, help="id уровней через запятую")
    parser.add_argument("--out", default=str(OUT_DIR / "tournament.json"))
    args = parser.parse_args()

    levels = load_levels()
    if args.levels:
        wanted = set(args.levels.split(","))
        levels = [level for level in levels if level.id in wanted]

    entrants = []
    if not args.no_random:
        entrants.append(("Случайный", lambda run: RandomAgent(seed=run), True))
    entrants.append(("Эвристика", lambda run: HeuristicAgent(), False))
    entrants.append(("Прыгун (всегда вправо+прыжок)", lambda run: JumperAgent(), False))

    from ai.agent import load_controller
    from ai.decide import default_model_path

    models = args.model or []
    if not models:
        try:
            models = [str(default_model_path())]
        except FileNotFoundError:
            models = []

    temperatures = [float(t) for t in args.temperatures.split(",") if t.strip()]
    for path in models:
        stem = Path(path).stem
        if not args.no_argmax:
            entrants.append((f"MarioJev {stem} (argmax)", lambda run, p=path: load_controller(p, deterministic=True), False))
        for temp in temperatures:
            label = "сэмпл" if temp == 1.0 else f"T={temp:g}"
            entrants.append((f"MarioJev {stem} ({label})",
                             lambda run, p=path, t=temp: load_controller(p, deterministic=False, temperature=t), True))

    if args.teacher:
        from ai.decide import DecisionModel
        from ai.teacher import Teacher
        entrants.append(("Учитель (модель + правила сбора)",
                         lambda run, p=args.teacher: Teacher(DecisionModel.load(p)), False))

    if args.planner:
        from ai.planner import Planner
        entrants.append(("Планировщик", lambda run: Planner(), False))

    if args.laya:
        from ai.laya_agent import LayaController
        laya = LayaController()
        entrants.append(("Laya (open Jev, без дообучения)", lambda run, c=laya: c, False))

    if args.only:
        entrants = [e for e in entrants if args.only.lower() in e[0].lower()]

    results = {}
    for name, factory, stochastic in entrants:
        runs = args.runs if stochastic else 1
        per_level = {}
        for level in levels:
            games = [play(factory(run), level) for run in range(runs)]
            per_level[level.id] = {
                "split": level.split,
                "score": float(np.mean([g["score"] for g in games])),
                "best": max(g["score"] for g in games),
                "win_rate": float(np.mean([g["won"] for g in games])),
                "flag_bonus": float(np.mean([g["flag_bonus"] for g in games])),
                "progress": float(np.mean([g["progress"] for g in games])),
                "deaths": [g["death"] for g in games if g["death"]],
                "collected": {key: float(np.mean([g[key] for g in games])) for key in COLLECTED},
                "time_used": float(np.mean([g["time_used"] for g in games])),
                "loops": loop_totals(games),
                "games": games,
            }
            got = per_level[level.id]["collected"]
            print(f"  {name:32s} {level.id:10s} очки {per_level[level.id]['score']:8.0f}  "
                  f"побед {per_level[level.id]['win_rate']:4.0%}  путь {per_level[level.id]['progress']:4.0%}  "
                  f"монет {got['coins']:.0f}/{games[0]['coins_total']}  «?» {got['blocks']:.0f}/{games[0]['block_hits_total']}  "
                  f"врагов {got['enemies']:.0f}/{games[0]['enemies_total']}  тайников {got['secrets']:.1f}/{games[0]['secrets_total']}  "
                  f"время {per_level[level.id]['time_used']:.0f} с", flush=True)
        agent_obj = factory(0)
        latencies = getattr(agent_obj, "latencies", None)
        results[name] = {
            "latency_ms_p50": float(np.median(latencies)) if latencies else None,
            "choices": getattr(agent_obj, "choices", None),
            "total": sum(v["score"] for v in per_level.values()),
            "test_total": sum(v["score"] for v in per_level.values() if v["split"] == "test"),
            "wins": sum(v["win_rate"] for v in per_level.values()),
            "collected": {key: sum(v["collected"][key] for v in per_level.values()) for key in COLLECTED},
            # Петли по всем уровням и повторам; stalls — на уровень (сумма средних по уровням)
            "loops": {**loop_totals([g for v in per_level.values() for g in v["games"]]),
                      "stalls": sum(v["loops"]["stalls"] for v in per_level.values())},
            "levels": per_level,
        }

    ranking = sorted(results.items(), key=lambda kv: -kv[1]["total"])

    header = "| Место | Участник | Сумма очков | На отложенных (test) | Пройдено уровней | " + " | ".join(l.id for l in levels) + " |"
    lines = [header, "|" + "---|" * (5 + len(levels))]
    for place, (name, r) in enumerate(ranking, 1):
        cells = " | ".join(f"{r['levels'][l.id]['score']:.0f}" for l in levels)
        lines.append(f"| {place} | {name} | **{r['total']:.0f}** | {r['test_total']:.0f} | {r['wins']:.1f} из {len(levels)} | {cells} |")
    # Сколько взято из возможного: главное отличие «проходит» от «спидранит»
    first_games = {l.id: results[ranking[0][0]]["levels"][l.id]["games"][0] for l in levels} if ranking else {}
    possible = {key: sum(g[f"{TOTAL_KEYS[key]}"] for g in first_games.values()) for key in COLLECTED}
    # Петли (ai/agent.py LoopMeter): доля решений в 3-секундном топтании на
    # месте без очков, число таких отрезков на прохождение всех уровней и
    # смены «вправо» ↔ «влево» в минуту — признак дёрганья
    lines += ["", "| Участник | Монеты | Удары по «?» | Враги | Секретные комнаты | Время на уровни, с "
                  "| Петли, % решений | Петель | Смен L/R в мин |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in ranking:
        got = r["collected"]
        seconds = sum(v["time_used"] for v in r["levels"].values())
        loops = r["loops"]
        lines.append(f"| {name} | " + " | ".join(f"{got[k]:.0f} из {possible[k]}" if k != "secrets" else f"{got[k]:.1f} из {possible[k]}"
                                               for k in COLLECTED) + f" | {seconds:.0f} "
                     f"| {100 * loops['stall_frac']:.1f} | {loops['stalls']:.1f} | {loops['flips_per_min']:.1f} |")
    table = "\n".join(lines)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"levels": [l.__dict__ for l in levels], "results": results, "table": table},
                                         ensure_ascii=False, indent=1, default=str))
    print("\n" + table)
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()

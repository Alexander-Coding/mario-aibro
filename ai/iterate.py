"""Одна итерация: обучение → калибровка → турнир → запись в журнал.

    .venv/bin/python -m ai.iterate --name iter2 --init ai/models/mariojev_best.pt --steps 1500000 --mix 0.6

Журнал итераций — ai/runs/iterations.jsonl: какая модель, на чём училась,
сколько очков набрала на каждом уровне. По нему видно, стала ли новая
итерация лучше прошлой по главному правилу — сумме очков.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "ai" / "models"
RUNS = ROOT / "ai" / "runs"


def run(cmd: list[str]) -> None:
    print("[iterate] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--init", default=None, help="продолжить с чекпойнта")
    parser.add_argument("--steps", type=int, default=1_500_000)
    parser.add_argument("--mix", type=float, default=0.6, help="доля процедурных уровней, остальное — train-уровни")
    parser.add_argument("--difficulty-start", type=float, default=0.25)
    parser.add_argument("--envs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent", type=float, default=0.01)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    py = sys.executable
    started = time.time()

    if not args.skip_train:
        cmd = [py, "-m", "ai.train", "--name", args.name, "--steps", str(args.steps), "--envs", str(args.envs),
               "--mix", str(args.mix), "--difficulty-start", str(args.difficulty_start),
               "--lr", str(args.lr), "--ent", str(args.ent)]
        if args.init:
            cmd += ["--init", args.init]
        run(cmd)

    best = MODELS / f"{args.name}_best.pt"
    final = MODELS / f"{args.name}_final.pt"
    run([py, "-m", "ai.calibrate", "--model", str(best), "--out", str(final)])

    out = RUNS / f"tournament_{args.name}.json"
    run([py, "-m", "tools.tournament", "--model", str(final), "--runs", str(args.runs), "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    ours = {name: r for name, r in result["results"].items() if name.startswith("MarioJev")}
    entry = {
        "name": args.name,
        "init": args.init,
        "steps": args.steps,
        "mix": args.mix,
        "minutes": round((time.time() - started) / 60, 1),
        "totals": {name: {"total": r["total"], "test_total": r["test_total"], "wins": r["wins"],
                          "levels": {lid: v["score"] for lid, v in r["levels"].items()}}
                   for name, r in ours.items()},
    }
    with open(RUNS / "iterations.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(json.dumps(entry, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Калибровка ответов модели — то, что у Jev называется calibrated decisions.

После обучения политики:
1. играем полные эпизоды итоговой политикой — так, как она будет играть
   в игре (argmax или сэмпл с температурой, --play) — и записываем для
   каждого решения признаки кодировщика, «погиб ли в ближайшие 2 с» и
   «сколько очков набрано от этого момента до конца уровня»;
2. учим головы points (ответ типа score) и danger (noul) заново на
   замороженном кодировщике: у модели из подражания (ai/imitate.py)
   кодировщик переучен, а головы остались от прошлой версии;
3. подбираем температуру для danger и points на отложенной части
   эпизодов, чтобы вероятности совпадали с частотами;
4. печатаем ECE (ожидаемую ошибку калибровки) до и после.

    .venv/bin/python -m ai.calibrate --model ai/models/mariojev_best.pt --episodes 120
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch import nn

from ai.env import MarioEnv
from ai.features import model_view
from ai.model import POINT_BIN_LABELS, points_to_bin
from ai.train import DANGER_HORIZON, MODEL_DIR, RUNS_DIR, load_checkpoint, save_checkpoint, to_torch


def level_factories(n_generated: int, seed: int):
    from core.levels import load_levels
    from core.world import GameWorld

    rng = random.Random(seed)
    factories = []

    try:
        from ai.levelgen import generate_level
        for _ in range(n_generated):
            s, d = rng.randrange(10**9), rng.uniform(0.0, 1.0)
            factories.append(lambda s=s, d=d: GameWorld(generate_level(s, d)))
    except Exception:
        pass

    # Только train-уровни: тестовые остаются нетронутыми до турнира
    for level in load_levels(split="train"):
        factories.append(lambda lv=level: GameWorld(lv.path, name=lv.name, time_limit=lv.time_limit))

    return factories


@torch.no_grad()
def collect(model, factories, deterministic: bool, temperature: float = 1.0):
    """Список эпизодов: признаки [T, d], danger-цели [T], очки-до-конца [T]."""

    episodes = []
    for factory in factories:
        env = MarioEnv(world_factory=factory)
        obs, _ = env.reset()
        feats, points, dead_at = [], [], None
        step = 0
        while True:
            view = model_view(obs, model)
            batch = to_torch({k: v[None] for k, v in view.items()}, "cpu")
            h = model.encode(batch)
            logits = model.policy_head(h)
            if deterministic:
                action = int(logits.argmax(-1))
            else:
                action = int(torch.distributions.Categorical(logits=logits / temperature).sample())
            feats.append(h[0].numpy())
            obs, _, terminated, truncated, info = env.step(action)
            points.append(info["points"])
            step += 1
            if terminated or truncated:
                if info["episode_summary"].get("death"):
                    dead_at = step - 1
                break

        steps = len(feats)
        points = np.asarray(points, dtype=np.float32)
        to_go = np.cumsum(points[::-1])[::-1].copy()          # очки с этого шага до конца
        danger = np.zeros(steps, dtype=np.float32)
        if dead_at is not None:
            danger[max(0, dead_at - DANGER_HORIZON + 1):dead_at + 1] = 1.0

        episodes.append({"h": np.stack(feats), "danger": danger, "to_go": to_go})
    return episodes


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Ожидаемая ошибка калибровки для бинарного прогноза."""

    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if idx.any():
            total += idx.mean() * abs(probs[idx].mean() - labels[idx].mean())
    return float(total)


def multiclass_ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float32)
    return ece(confidence, correct, bins)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, binary: bool) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        t = log_t.exp()
        if binary:
            loss = nn.functional.binary_cross_entropy_with_logits(logits / t, targets)
        else:
            loss = nn.functional.cross_entropy(logits / t, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_DIR / "mariojev_best.pt"))
    parser.add_argument("--out", default=str(MODEL_DIR / "mariojev_final.pt"))
    parser.add_argument("--episodes", type=int, default=120, help="процедурных эпизодов (+ train-уровни)")
    parser.add_argument("--play", choices=("argmax", "sample"), default=None,
                        help="как модель играет в игре; по умолчанию — из чекпойнта (play_deterministic)")
    parser.add_argument("--temperature", type=float, default=None, help="температура сэмпла; по умолчанию из чекпойнта")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    model, meta = load_checkpoint(Path(args.model))
    deterministic = (args.play == "argmax") if args.play else bool(meta.get("play_deterministic", False))
    temperature = args.temperature if args.temperature is not None else float(meta.get("play_temperature", 1.0))
    factories = level_factories(args.episodes, args.seed)
    print(f"[calibrate] эпизодов: {len(factories)}, игра: "
          f"{'argmax' if deterministic else f'сэмпл T={temperature:g}'}", flush=True)

    episodes = collect(model, factories, deterministic=deterministic, temperature=temperature)
    rng = random.Random(args.seed)
    rng.shuffle(episodes)
    split = int(len(episodes) * 0.8)
    train_eps, hold_eps = episodes[:split], episodes[split:]
    if len(hold_eps) < 2:
        # Отложенные эпизоды делятся пополам: на подбор температуры и на честный ECE
        parser.error(f"эпизодов {len(episodes)}: отложено {len(hold_eps)}, нужно хотя бы 2 — увеличьте --episodes")

    def stack(eps, key):
        return torch.as_tensor(np.concatenate([e[key] for e in eps]))

    h_train, h_hold = stack(train_eps, "h"), stack(hold_eps, "h")
    d_train, d_hold = stack(train_eps, "danger"), stack(hold_eps, "danger")
    p_train = points_to_bin(stack(train_eps, "to_go"))
    p_hold = points_to_bin(stack(hold_eps, "to_go"))
    print(f"[calibrate] решений: обучение {len(h_train):,}, проверка {len(h_hold):,}; "
          f"доля «погиб в 2 с» {float(d_hold.mean()):.3f}", flush=True)

    # --- головы points и danger на замороженном кодировщике ----------
    head = model.points_head
    for layer in head:
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = torch.bincount(p_train, minlength=len(POINT_BIN_LABELS)).float()
    for epoch in range(60):
        perm = torch.randperm(len(h_train))
        for start in range(0, len(perm), 1024):
            idx = perm[start:start + 1024]
            loss = nn.functional.cross_entropy(head(h_train[idx]), p_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # danger — только если в выборке есть смерти, иначе учить нечему
    danger_retrained = int(d_train.sum()) >= 20
    if danger_retrained:
        danger = model.danger_head
        for layer in danger:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        optimizer = torch.optim.Adam(danger.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(40):
            perm = torch.randperm(len(h_train))
            for start in range(0, len(perm), 1024):
                idx = perm[start:start + 1024]
                loss = nn.functional.binary_cross_entropy_with_logits(danger(h_train[idx]).squeeze(-1), d_train[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Температура подбирается на одной половине отложенных эпизодов, а ECE
    # честно меряется на другой: иначе параметр оценивается там же, где подобран
    fit_eps, eval_eps = hold_eps[: len(hold_eps) // 2], hold_eps[len(hold_eps) // 2:]
    h_fit, h_eval = stack(fit_eps, "h"), stack(eval_eps, "h")
    d_fit, d_eval = stack(fit_eps, "danger"), stack(eval_eps, "danger")
    p_fit, p_eval = points_to_bin(stack(fit_eps, "to_go")), points_to_bin(stack(eval_eps, "to_go"))

    with torch.no_grad():
        d_fit_logits = model.danger_head(h_fit).squeeze(-1)
        d_eval_logits = model.danger_head(h_eval).squeeze(-1)
        p_fit_logits = head(h_fit)
        p_eval_logits = head(h_eval)

    before = {
        "danger_ece": ece(torch.sigmoid(d_eval_logits).numpy(), d_eval.numpy()),
        "points_ece": multiclass_ece(torch.softmax(p_eval_logits, -1).numpy(), p_eval.numpy()),
        "points_accuracy": float((p_eval_logits.argmax(-1) == p_eval).float().mean()),
    }

    # Без смертей в выборке температуру «опасности» подобрать нельзя: подбор
    # сожмёт вероятность к нулю (на 8 эпизодах без смертей вышло T=0.105)
    notes = []
    positives = int(d_fit.sum())
    if 5 <= positives <= len(d_fit) - 5:
        t_danger = fit_temperature(d_fit_logits, d_fit, binary=True)
    else:
        t_danger = 1.0
        notes.append(f"danger: в выборке подбора {positives} шагов перед смертью — температура не подбиралась")
    t_points = fit_temperature(p_fit_logits, p_fit, binary=False)

    # Температура принимается, только если на честной половине не хуже
    def eval_ece_danger(t):
        return ece(torch.sigmoid(d_eval_logits / t).numpy(), d_eval.numpy())

    def eval_ece_points(t):
        return multiclass_ece(torch.softmax(p_eval_logits / t, -1).numpy(), p_eval.numpy())

    if eval_ece_danger(t_danger) > eval_ece_danger(1.0):
        notes.append(f"danger: T={t_danger:.3f} ухудшила ECE на проверке — оставлена 1.0")
        t_danger = 1.0
    if eval_ece_points(t_points) > eval_ece_points(1.0):
        notes.append(f"points: T={t_points:.3f} ухудшила ECE на проверке — оставлена 1.0")
        t_points = 1.0

    model.danger_temperature.fill_(t_danger)
    model.points_temperature.fill_(t_points)

    after = {
        "danger_ece": ece(torch.sigmoid(d_eval_logits / t_danger).numpy(), d_eval.numpy()),
        "points_ece": multiclass_ece(torch.softmax(p_eval_logits / t_points, -1).numpy(), p_eval.numpy()),
        "danger_brier": float(((torch.sigmoid(d_eval_logits / t_danger) - d_eval) ** 2).mean()),
        "danger_base_rate": float(d_eval.mean()),
    }

    report = {
        "source_model": args.model,
        "episodes": len(episodes),
        "decisions_train": len(h_train),
        "decisions_holdout": len(h_hold),
        "episodes_fit_eval": [len(fit_eps), len(eval_eps)],
        "notes": notes,
        "play": "argmax" if deterministic else f"sample T={temperature:g}",
        "danger_head_retrained": danger_retrained,
        "temperature": {"danger": t_danger, "points": t_points},
        "before": before,
        "after": after,
        "points_class_counts": counts.tolist(),
    }

    meta = dict(meta, calibration=report, play_deterministic=deterministic, play_temperature=temperature)
    save_checkpoint(Path(args.out), model, meta)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[calibrate] сохранено: {args.out}")


if __name__ == "__main__":
    main()

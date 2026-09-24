"""Обучение MarioJev: PPO по очкам на процедурных и train-уровнях.

    .venv/bin/python -m ai.train --steps 5000000 --envs 8

Как устроено:
- 8 процессов-сред, в каждой свой поток уровней (ai/env.py: LevelSampler);
- PPO с GAE; награда нормируется на разброс дисконтированного дохода;
- вспомогательная задача «погибну ли в ближайшие 2 с» (голова danger)
  учится вместе с политикой и помогает кодировщику замечать опасность;
- учебный план: потолок сложности процедурных уровней растёт 0.25 → 1.0
  за первые 60 % обучения;
- валидация — фиксированный набор процедурных уровней и train-уровней;
  лучшая модель выбирается по ней. Тестовые (отложенные) уровни модель
  не видит ни при обучении, ни при выборе чекпойнта.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch import nn

from ai.env import GAMMA, LevelSampler, MarioEnv
from ai.features import observation_shapes
from ai.model import MarioJev, count_parameters

AI_DIR = Path(__file__).resolve().parent
MODEL_DIR = AI_DIR / "models"
RUNS_DIR = AI_DIR / "runs"

DANGER_HORIZON = 30   # шагов среды = 2 секунды игры


# ===============================================================
# ВСПОМОГАТЕЛЬНОЕ
# ===============================================================


class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, batch_count = float(np.mean(x)), float(np.var(x)), x.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return math.sqrt(self.var) + 1e-8


def to_torch(obs: dict, device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}


def make_env(rank: int, mix: float, difficulty: float, base_seed: int, train_levels: list,
             reward_mode: str = "progress", gamma: float = GAMMA):
    def _init():
        # Каждый процесс — на одном потоке torch, чтобы не драться за ядра
        torch.set_num_threads(1)
        sampler = LevelSampler(mix=mix, difficulty=difficulty, seed=base_seed + 7919 * rank, levels=train_levels)
        return MarioEnv(sampler=sampler, reward_mode=reward_mode, gamma=gamma)
    return _init


def danger_targets(dead: np.ndarray, done: np.ndarray, horizon: int):
    """Цель головы danger: погибнет ли агент в ближайшие horizon шагов.

    dead/done — [T, N]. Шаги, у которых окно в horizon шагов выходит за
    конец буфера, маскируются ВСЕ, независимо от исхода: иначе на хвосте
    оставались бы только эпизоды, которые успели закончиться (почти всегда
    смертью), и голова переоценивала бы риск.
    """

    steps, envs = dead.shape
    target = np.zeros((steps, envs), dtype=np.float32)
    mask = np.zeros((steps, envs), dtype=np.float32)

    for env in range(envs):
        distance = math.inf
        known = False
        for t in range(steps - 1, -1, -1):
            if done[t, env]:
                known = True
                distance = 0 if dead[t, env] else math.inf
            else:
                distance = distance + 1

            if steps - t < horizon:
                continue
            target[t, env] = 1.0 if (known and distance < horizon) else 0.0
            mask[t, env] = 1.0

    return target, mask


# ===============================================================
# ОЦЕНКА
# ===============================================================


def validation_worlds(train_levels: list, n_generated: int = 8):
    """Фиксированный набор для выбора лучшей модели: процедурные уровни
    на отдельных сидах + train-уровни, зафиксированные при старте прогона
    (манифест между валидациями могут поменять — список берётся из main)."""

    from core.world import GameWorld

    factories = []

    try:
        from ai.levelgen import generate_level
        for i in range(n_generated):
            difficulty = (i + 0.5) / n_generated
            seed = 900_000 + i
            factories.append((f"gen-{seed}-{difficulty:.2f}", lambda s=seed, d=difficulty: GameWorld(generate_level(s, d))))
    except ImportError:
        pass

    for level in train_levels:
        factories.append((level.id, lambda lv=level: GameWorld(lv.path, name=lv.name, time_limit=lv.time_limit)))

    if not factories:
        factories.append(("1-1", lambda: GameWorld("maps/custom_level.json")))

    return factories


@torch.no_grad()
def run_episode(model: MarioJev, world_factory, deterministic: bool = True, device="cpu", max_steps: int = 6000,
                seed: int | None = None):
    env = MarioEnv(world_factory=world_factory, max_steps=max_steps)
    obs, _ = env.reset()
    total = 0.0
    # Своё зерно — чтобы сэмплированная валидация повторялась и не сбивала
    # генератор случайных чисел основного цикла
    with torch.random.fork_rng():
        if seed is not None:
            torch.manual_seed(seed)
        return _play(model, env, obs, total, deterministic, device)


def _play(model, env, obs, total, deterministic, device):
    while True:
        action, _, _ = model.act(to_torch({k: v[None] for k, v in obs.items()}, device), deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        total += reward
        if terminated or truncated:
            summary = info["episode_summary"]
            summary["return"] = round(total, 2)
            return summary


def evaluate(model: MarioJev, factories, deterministic: bool = False, device="cpu", runs: int = 2) -> dict:
    """По умолчанию — сэмплированием, как политика и играет при обучении.

    Argmax у PPO-политики бывает вырожденным: 23.09 на первом прогоне он на
    всех уровнях был «вправо+прыжок» и давал одну и ту же траекторию на
    чекпойнтах с разными весами, а лучшая модель по нему не менялась с
    40-го обновления."""
    was_training = model.training
    model.eval()
    results = [
        dict(run_episode(model, factory, deterministic, device, seed=1000 * run + i), id=name, run=run)
        for run in range(1 if deterministic else runs)
        for i, (name, factory) in enumerate(factories)
    ]
    model.train(was_training)
    return {
        "score": float(np.mean([r["score"] for r in results])),
        "win_rate": float(np.mean([r["won"] for r in results])),
        "progress": float(np.mean([r["progress"] for r in results])),
        "per_level": results,
    }


# ===============================================================
# СОХРАНЕНИЕ
# ===============================================================


def _sha1(path: Path) -> str:
    import hashlib
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def save_checkpoint(path: Path, model: MarioJev, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config, "meta": meta}, path)


def load_checkpoint(path: Path, device="cpu") -> tuple[MarioJev, dict]:
    data = torch.load(path, map_location=device, weights_only=False)
    model = MarioJev(**data["config"]).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data.get("meta", {})


# ===============================================================
# ОБУЧЕНИЕ
# ===============================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--envs", type=int, default=12)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent", type=float, default=0.03,
                        help="0.01 на первом прогоне схлопнул 4 из 6 действий к нулю за 100 обновлений")
    parser.add_argument("--ent-final", type=float, default=0.005)
    parser.add_argument("--val-runs", type=int, default=3, help="повторов сэмплированной валидации на уровень")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.97,
                        help="0.995 на итерациях 1–2 давал шумные преимущества на эпизодах в ~340 решений")
    parser.add_argument("--reward", choices=("progress", "potential"), default="progress")
    parser.add_argument("--danger-coef", type=float, default=0.1)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="при 0.5 градиент критика (норма 1–4) урезал и шаг политики: KL ~0.0005 на обновление")
    parser.add_argument("--mix", type=float, default=0.8, help="доля процедурных уровней")
    parser.add_argument("--difficulty-start", type=float, default=0.25)
    parser.add_argument("--difficulty-ramp", type=float, default=0.6, help="доля обучения до полной сложности")
    parser.add_argument("--eval-every", type=int, default=25, help="обновлений между валидациями")
    parser.add_argument("--name", default="mariojev")
    parser.add_argument("--init", default=None, help="продолжить с чекпойнта")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
                        help="где обновлять веса; сбор опыта — всегда на копии модели на CPU")
    parser.add_argument("--threads", type=int, default=1, help="потоки torch для сбора опыта; больше — хуже, OpenMP ждёт занятые ядра")
    parser.add_argument("--update-threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="потоки torch на обновление весов на CPU: среды в это время стоят")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from stable_baselines3.common.vec_env import SubprocVecEnv

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    from core.levels import load_levels

    # Набор уровней фиксируется один раз на весь прогон: и для сред, и для
    # валидации. Отложенные (test) уровни не должны попасть ни туда, ни туда.
    train_levels = load_levels(split="train")
    test_levels = load_levels(split="test")
    train_files = {level.file for level in train_levels}
    leaked = [level.id for level in test_levels if level.file in train_files]
    if leaked:
        raise SystemExit(f"[train] test-уровни попали в train: {leaked}")

    run_dir = RUNS_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(
        dict(vars(args), train_levels=[{"id": l.id, "file": l.file, "sha1": _sha1(l.path)} for l in train_levels],
             test_levels=[l.id for l in test_levels]),
        indent=2, ensure_ascii=False))
    print(f"[train] train-уровни: {[l.id for l in train_levels]}  отложенные: {[l.id for l in test_levels]}", flush=True)

    envs = SubprocVecEnv(
        [make_env(rank, args.mix, args.difficulty_start, args.seed * 1000, train_levels, args.reward, args.gamma)
         for rank in range(args.envs)],
        start_method="spawn",
    )

    init_meta: dict = {}
    if args.init:
        model, init_meta = load_checkpoint(Path(args.init), device)
        model.train()
        if Path(args.init).resolve().parent == MODEL_DIR.resolve() and Path(args.init).stem.startswith(f"{args.name}_"):
            raise SystemExit("[train] --init из чекпойнта с тем же --name перезапишет его: дайте новое имя итерации")
    else:
        model = MarioJev().to(device)

    # Прошлый прогон уже знает, у кого какой старый чекпойнт калибровки —
    # новый прогон не должен молча уступать ему место в игре
    stale = MODEL_DIR / f"{args.name}_final.pt"
    if stale.exists():
        stale.rename(stale.with_suffix(".pt.old"))

    # Копия для сбора опыта и валидации: на батче из 12 наблюдений CPU
    # отвечает за ~3 мс, а GPU (MPS) — за ~17 мс на одних накладных
    # расходах. Обновление весов, наоборот, на MPS в 7 раз быстрее.
    actor = MarioJev(**model.config)
    actor.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    actor.eval()
    cpu = torch.device("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    print(f"[train] параметров: {count_parameters(model):,}  сред: {args.envs}  устройство: {device}", flush=True)

    shapes = observation_shapes()
    T, N = args.n_steps, args.envs

    buf_obs = {k: torch.zeros((T, N) + s, dtype=torch.float32) for k, s in shapes.items()}
    buf_actions = torch.zeros((T, N), dtype=torch.long)
    buf_logp = torch.zeros((T, N))
    buf_values = torch.zeros((T, N))
    buf_rewards = np.zeros((T, N), dtype=np.float32)
    buf_done = np.zeros((T, N), dtype=np.float32)
    buf_dead = np.zeros((T, N), dtype=np.float32)

    return_rms = RunningMeanStd()
    # Критик прошлого чекпойнта обучен в масштабе его нормировки награды
    if init_meta.get("return_rms"):
        return_rms.mean, return_rms.var, return_rms.count = (init_meta["return_rms"][k] for k in ("mean", "var", "count"))
    running_return = np.zeros(N, dtype=np.float64)

    obs = envs.reset()
    total_updates = max(1, args.steps // (T * N))
    global_step = 0
    best_val = -math.inf
    started = time.time()

    episodes_file = open(run_dir / "episodes.jsonl", "a", encoding="utf-8")
    log_file = open(run_dir / "log.csv", "a", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if log_file.tell() == 0:
        log.writerow(["update", "steps", "sps", "difficulty", "ep_score", "ep_win", "ep_progress", "ep_len",
                      "pg_loss", "v_loss", "entropy", "kl", "danger_loss", "val_score", "val_win", "clip_frac", "grad_norm",
                      "p_actions"])

    recent: list[dict] = []

    for update in range(1, total_updates + 1):
        frac = (update - 1) / total_updates

        # --- расписания ---------------------------------------------
        lr = args.lr * (1.0 - frac)
        for group in optimizer.param_groups:
            group["lr"] = lr
        ent_coef = args.ent + (args.ent_final - args.ent) * frac
        difficulty = min(1.0, args.difficulty_start + (1.0 - args.difficulty_start) * frac / args.difficulty_ramp)
        envs.env_method("set_difficulty", difficulty)

        # --- сбор опыта ---------------------------------------------
        for t in range(T):
            obs_t = to_torch(obs, cpu)
            with torch.no_grad():
                action, logp, value = actor.act(obs_t)

            for k in buf_obs:
                buf_obs[k][t] = obs_t[k].cpu()
            buf_actions[t] = action.cpu()
            buf_logp[t] = logp.cpu()
            buf_values[t] = value.cpu()

            obs, rewards, dones, infos = envs.step(action.cpu().numpy())
            rewards = rewards.astype(np.float32)

            for i, info in enumerate(infos):
                if not dones[i]:
                    continue
                summary = info.get("episode_summary", {})
                truncated = bool(info.get("TimeLimit.truncated") or summary.get("truncated"))
                buf_dead[t, i] = 1.0 if summary.get("death") else 0.0

                # Обрезанный эпизод — не конец игры: досчитываем хвост
                # по оценке критика на последнем наблюдении
                if truncated and "terminal_observation" in info:
                    term = {k: torch.as_tensor(v[None], dtype=torch.float32)
                            for k, v in info["terminal_observation"].items()}
                    with torch.no_grad():
                        _, v_term, _, _ = actor(term)
                    rewards[i] += args.gamma * float(v_term.item()) * return_rms.std

                summary["update"] = update
                summary["difficulty"] = round(difficulty, 3)
                episodes_file.write(json.dumps(summary, ensure_ascii=False) + "\n")
                recent.append(summary)

            # нормировка награды по разбросу дохода
            running_return = running_return * args.gamma + rewards
            return_rms.update(running_return)
            running_return[dones] = 0.0

            buf_rewards[t] = rewards / return_rms.std
            buf_done[t] = dones.astype(np.float32)

        global_step += T * N
        episodes_file.flush()

        # --- GAE -----------------------------------------------------
        with torch.no_grad():
            _, next_value, _, _ = actor(to_torch(obs, cpu))
        next_value = next_value.numpy()

        values = buf_values.numpy()
        advantages = np.zeros((T, N), dtype=np.float32)
        last = np.zeros(N, dtype=np.float32)
        for t in range(T - 1, -1, -1):
            nonterminal = 1.0 - buf_done[t]
            nv = next_value if t == T - 1 else values[t + 1]
            delta = buf_rewards[t] + args.gamma * nv * nonterminal - values[t]
            last = delta + args.gamma * args.gae_lambda * nonterminal * last
            advantages[t] = last
        returns = advantages + values

        d_target, d_mask = danger_targets(buf_dead, buf_done, DANGER_HORIZON)

        # --- обновление ---------------------------------------------
        torch.set_num_threads(args.update_threads)
        model.train()
        flat = lambda x: x.reshape((T * N,) + x.shape[2:])
        b_obs = {k: flat(v).to(device) for k, v in buf_obs.items()}
        b_actions = flat(buf_actions).to(device)
        b_logp = flat(buf_logp).to(device)
        b_adv = torch.as_tensor(flat(advantages), device=device)
        b_ret = torch.as_tensor(flat(returns), device=device)
        b_dt = torch.as_tensor(flat(d_target), device=device)
        b_dm = torch.as_tensor(flat(d_mask), device=device)

        # Нормировка преимуществ — один раз на весь батч: в минибатче из
        # одного-двух элементов std даёт NaN или ±0.7 вне зависимости от величины
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        stats = {"pg": [], "v": [], "ent": [], "kl": [], "danger": [], "clip": [], "grad": []}
        indices = np.arange(T * N)
        for _ in range(args.epochs):
            np.random.shuffle(indices)
            for start in range(0, T * N, args.minibatch):
                mb = torch.as_tensor(indices[start:start + args.minibatch], device=device)
                mb_obs = {k: v[mb] for k, v in b_obs.items()}

                new_logp, entropy, value, danger_logit, _ = model.evaluate(mb_obs, b_actions[mb])
                ratio = torch.exp(new_logp - b_logp[mb])
                adv = b_adv[mb]

                pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
                v_loss = 0.5 * ((value - b_ret[mb]) ** 2).mean()
                ent = entropy.mean()

                dm = b_dm[mb]
                danger_loss = (
                    nn.functional.binary_cross_entropy_with_logits(danger_logit, b_dt[mb], reduction="none") * dm
                ).sum() / dm.sum().clamp(min=1.0)

                loss = pg_loss + args.vf_coef * v_loss - ent_coef * ent + args.danger_coef * danger_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean()
                stats["pg"].append(pg_loss.item())
                stats["v"].append(v_loss.item())
                stats["ent"].append(ent.item())
                stats["kl"].append(kl.item())
                stats["danger"].append(danger_loss.item())
                stats["clip"].append(((ratio - 1).abs() > args.clip).float().mean().item())
                stats["grad"].append(float(grad_norm))

        actor.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
        torch.set_num_threads(args.threads)

        # --- журнал -------------------------------------------------
        sps = global_step / (time.time() - started)
        window = recent[-100:]
        ep_score = float(np.mean([e["score"] for e in window])) if window else float("nan")
        ep_win = float(np.mean([e["won"] for e in window])) if window else float("nan")
        ep_prog = float(np.mean([e["progress"] for e in window])) if window else float("nan")
        ep_len = float(np.mean([e["frames"] for e in window])) / 4 if window else float("nan")

        val_score = val_win = ""
        meta = {"update": update, "steps": global_step, "difficulty": difficulty, "args": vars(args),
                "return_rms": {"mean": return_rms.mean, "var": return_rms.var, "count": return_rms.count},
                "train_levels": [l.id for l in train_levels]}

        if update % args.eval_every == 0 or update == total_updates:
            val = evaluate(actor, validation_worlds(train_levels), deterministic=False, device=cpu, runs=args.val_runs)
            val_score, val_win = round(val["score"], 1), round(val["win_rate"], 3)
            meta["validation"] = {"score": val["score"], "win_rate": val["win_rate"], "progress": val["progress"]}
            (run_dir / f"val_{update:05d}.json").write_text(json.dumps(val, ensure_ascii=False, indent=1))
            if val["score"] > best_val:
                best_val = val["score"]
                save_checkpoint(MODEL_DIR / f"{args.name}_best.pt", actor, meta)
            save_checkpoint(MODEL_DIR / f"{args.name}_last.pt", actor, meta)

        log.writerow([update, global_step, round(sps), round(difficulty, 3), round(ep_score, 1), round(ep_win, 3),
                      round(ep_prog, 3), round(ep_len, 1), round(np.mean(stats["pg"]), 4), round(np.mean(stats["v"]), 4),
                      round(np.mean(stats["ent"]), 4), round(np.mean(stats["kl"]), 5), round(np.mean(stats["danger"]), 4),
                      val_score, val_win, round(np.mean(stats["clip"]), 4), round(np.mean(stats["grad"]), 3),
                      " ".join(f"{x:.2f}" for x in np.bincount(buf_actions.numpy().ravel(), minlength=6) / buf_actions.numel())])
        log_file.flush()

        if update % 5 == 0 or update == 1 or val_score != "":
            print(f"[{update:5d}/{total_updates}] шагов {global_step:>9,}  {sps:5.0f}/с  слож {difficulty:.2f}  "
                  f"очки {ep_score:7.0f}  побед {ep_win:5.1%}  путь {ep_prog:5.1%}  энтр {np.mean(stats['ent']):.3f}  "
                  f"kl {np.mean(stats['kl']):.4f}  клип {np.mean(stats['clip']):.3f}  |g| {np.mean(stats['grad']):.2f}" + (f"  | валид: очки {val_score} побед {val_win}" if val_score != "" else ""),
                  flush=True)

    save_checkpoint(MODEL_DIR / f"{args.name}_last.pt", actor, meta)
    envs.close()
    episodes_file.close()
    log_file.close()
    print(f"[train] готово за {(time.time() - started) / 60:.1f} мин, лучшая валидация {best_val:.0f}", flush=True)


if __name__ == "__main__":
    main()

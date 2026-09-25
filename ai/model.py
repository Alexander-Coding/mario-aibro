"""MarioJev — модель решений в духе Jev: не пишет текст, а отвечает на
типизированные вопросы вероятностями.

Вход — типизированное состояние (ai/features.py), выход — четыре головы:

- choice (move)   — какое нажатие сделать: распределение по 7 действиям;
- value           — критик для обучения (ожидаемая награда);
- noul (danger)   — вероятность погибнуть в ближайшие 2 секунды;
- score (points)  — сколько очков ещё будет набрано до конца уровня,
                    распределение по диапазонам.

Устройство: карта 9×16×28 режется на патчи → токены (как в ViT),
+ токен на каждого врага + токен игрока + CLS; трансформер смешивает их
(«враг в двух тайлах, а над ним блок “?”»); головы читают CLS.

Две архитектуры (ARCHS): v1 — патчи 4×4 (28 токенов карты), 3 слоя по 96,
на ней обучены все модели до планировщика; v2 — патчи 4 ряда × 2 столбца
(56 токенов: по горизонтали вдвое точнее — край блока «?» и монеты в
тайле от Марио перестают сливаться в один патч) и больше ёмкости под
мягкие метки планировщика. Размер патча и окно карты (grid_h, grid_w,
grid_back) хранятся в конфиге чекпойнта; у старых чекпойнтов их нет —
окно прежнее (24 столбца, 4 позади), патч выводится из весов
(config_from_state_dict). Все модели до расширения окна обучены на 24
столбцах и играют на вырезке из нового (ai/features.py model_view).

Почему патчи, а не свёртки: в сборке torch для macOS нет MKLDNN, и
свёртка на CPU идёт медленным путём (346 мкс на крошечный слой), а
умножение матриц — через Accelerate (340 GFLOPS на ядро). Патч + линейный
слой — это одно умножение матриц.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from ai.features import (
    ENEMY_FEATURES,
    GRID_BACK,
    GRID_H,
    GRID_W,
    LEGACY_GRID_BACK,
    LEGACY_GRID_W,
    MAX_ENEMIES,
    N_ACTIONS,
    N_CHANNELS,
    PLAYER_FEATURES,
)

# Диапазоны для головы «сколько ещё очков» (правые границы, очки)
POINT_BINS = (500, 1500, 3000, 5000, 8000, float("inf"))
POINT_BIN_LABELS = ("до 500", "500–1.5 тыс.", "1.5–3 тыс.", "3–5 тыс.", "5–8 тыс.", "8 тыс.+")
POINT_BIN_CENTERS = (250.0, 1000.0, 2250.0, 4000.0, 6500.0, 10000.0)


def points_to_bin(points: torch.Tensor) -> torch.Tensor:
    bins = torch.zeros_like(points, dtype=torch.long)
    for edge in POINT_BINS[:-1]:
        bins += (points >= edge).long()
    return bins


PATCH = 4   # патч v1 (и всех чекпойнтов без patch_h/patch_w в конфиге)

# Архитектуры для новых моделей. v2 выбрана замером на ПК (Ryzen 5 7500F,
# ai/runs/notes_student.md): одно наблюдение ≤ 3 мс на одном потоке, 10 эпох
# по 1.5 млн примеров ≤ 3 ч. Обучение упирается в умножения матриц, цена
# ~ токены × d² × слои: v2 (58 токенов, 128×3) — 0.6 скорости v1 при bf16,
# v2l (128×4) — 0.3–0.4, 10 эпох не влезают в 3 ч при чужой нагрузке на ПК.
# Окно 28 (8 столбцов позади) добавило v2 8 токенов карты (66 вместо 58):
# на ПК без чужой нагрузки обучение 1120–1200 прим/с против 1520–1560 на
# окне 24 (10 эпох × 1.5 млн — 3.5–3.7 ч), одно наблюдение 1.07 мс против
# 1.01 (ai/runs/notes_rearview.md). В 3 ч влезают: d_model 112 — 2.8 ч,
# патч 4×4 — 1.8 ч (но край блока и монета снова в одном патче)
ARCHS = {
    "v1": {"d_model": 96, "heads": 4, "layers": 3, "patch_h": 4, "patch_w": 4},
    "v2": {"d_model": 128, "heads": 4, "layers": 3, "patch_h": 4, "patch_w": 2},
    "v2l": {"d_model": 128, "heads": 4, "layers": 4, "patch_h": 4, "patch_w": 2},
}


class Block(nn.Module):
    """Слой трансформера: pre-LN, внимание через scaled_dot_product_attention.

    Свой, а не nn.TransformerEncoderLayer: у того с norm_first нет быстрого
    пути, и на батче из 8 наблюдений он тратит ~3 мс на слой на накладные
    расходы операций."""

    def __init__(self, d_model: int, heads: int, ff: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Linear(ff, d_model))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch, tokens, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(batch, tokens, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(a.transpose(1, 2).reshape(batch, tokens, d))
        return x + self.ff(self.ln2(x))


class MarioJev(nn.Module):

    def __init__(self, d_model: int = 96, heads: int = 4, layers: int = 3, dropout: float = 0.0,
                 channels: int = N_CHANNELS, player_features: int = PLAYER_FEATURES, actions: int = N_ACTIONS,
                 patch_h: int = PATCH, patch_w: int = PATCH,
                 grid_h: int = GRID_H, grid_w: int = GRID_W, grid_back: int = GRID_BACK):
        super().__init__()
        if grid_h % patch_h or grid_w % patch_w:
            raise ValueError(f"патч {patch_h}×{patch_w} не делит карту {grid_h}×{grid_w}")

        # Размеры входа и выхода — в конфиге: модели прошлых версий
        # (8 каналов, 6 действий) грузятся и играют на префиксе наблюдения
        self.channels = channels
        self.player_features = player_features
        self.actions = actions
        self.patch_h = patch_h
        self.patch_w = patch_w
        # Окно карты, на котором модель учится: у прошлых моделей — 24
        # столбца (4 позади), наблюдение им режется из нового (model_view)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.grid_back = grid_back
        self.config = {"d_model": d_model, "heads": heads, "layers": layers,
                       "channels": channels, "player_features": player_features, "actions": actions,
                       "patch_h": patch_h, "patch_w": patch_w,
                       "grid_h": grid_h, "grid_w": grid_w, "grid_back": grid_back}

        # --- карта: патчи 4×4 → 4×7 = 28 токенов (v1), 4×2 → 4×14 = 56 (v2);
        # на прежнем окне 24 — 4×6 = 24 и 4×12 = 48
        patch_size = channels * patch_h * patch_w
        self.grid_stem = nn.Sequential(
            nn.Linear(patch_size, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )
        self.grid_tokens = (grid_h // patch_h) * (grid_w // patch_w)
        self.grid_pos = nn.Parameter(torch.zeros(1, self.grid_tokens, d_model))

        # --- враги и игрок -------------------------------------------
        self.enemy_embed = nn.Sequential(
            nn.Linear(ENEMY_FEATURES, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.player_embed = nn.Sequential(
            nn.Linear(player_features, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        # Тип токена: 0 CLS, 1 карта, 2 враг, 3 игрок
        self.type_embed = nn.Embedding(4, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        self.blocks = nn.ModuleList([Block(d_model, heads, 2 * d_model) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

        # --- головы --------------------------------------------------
        self.policy_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, actions))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.danger_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.points_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, len(POINT_BINS)))

        # Калибровка (температуры) — подбирается после обучения, ai/calibrate.py
        self.register_buffer("danger_temperature", torch.ones(1))
        self.register_buffer("points_temperature", torch.ones(1))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.grid_pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        # Маленькие стартовые логиты политики — равномерный выбор в начале
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    # -----------------------------------------------------------

    def encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = obs["grid"]
        batch = grid.shape[0]

        # B, C, H, W → B, (H/ph)·(W/pw), C·ph·pw
        ph, pw = self.patch_h, self.patch_w
        patches = grid.reshape(batch, self.channels, self.grid_h // ph, ph, self.grid_w // pw, pw)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, self.grid_tokens, -1)
        g = self.grid_stem(patches) + self.grid_pos + self.type_embed.weight[1]

        e = self.enemy_embed(obs["enemies"]) + self.type_embed.weight[2]   # B, 8, d
        p = self.player_embed(obs["player"]).unsqueeze(1) + self.type_embed.weight[3]
        c = self.cls.expand(batch, -1, -1) + self.type_embed.weight[0]

        tokens = torch.cat([c, p, g, e], dim=1)

        # Пустые слоты врагов не участвуют во внимании: маска B×1×1×T,
        # True — токен можно читать
        keep = torch.ones(batch, 2 + self.grid_tokens, dtype=torch.bool, device=grid.device)
        attn_mask = torch.cat([keep, obs["enemy_mask"] > 0.5], dim=1)[:, None, None, :]

        h = tokens
        for block in self.blocks:
            h = block(h, attn_mask)
        return self.norm(h[:, 0])

    def forward(self, obs: dict[str, torch.Tensor]):
        h = self.encode(obs)
        return (
            self.policy_head(h),
            self.value_head(h).squeeze(-1),
            self.danger_head(h).squeeze(-1),
            self.points_head(h),
        )

    # -----------------------------------------------------------

    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        logits, value, _, _ = self(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs: dict[str, torch.Tensor], actions: torch.Tensor):
        logits, value, danger, points = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value, danger, points


def config_from_state_dict(state_dict: dict, config: dict) -> dict:
    """Конфиг чекпойнта, дополненный размерами из самих весов (у старых
    чекпойнтов в конфиге их нет).

    Окно карты: чекпойнты без grid_w обучены до расширения окна назад —
    на 24 столбцах, 4 из них позади (LEGACY_GRID_W, LEGACY_GRID_BACK).

    Патч: число токенов карты (grid_pos) задаёт его площадь, но не форму —
    4×4, 2×8 и 8×2 дают одинаковые 24 токена. Все чекпойнты без
    patch_h/patch_w в конфиге обучены на 4×4, поэтому из подходящих форм
    берётся 4×4, иначе — самая квадратная."""

    config = dict(config)
    config.setdefault("grid_w", LEGACY_GRID_W)
    config.setdefault("grid_back", LEGACY_GRID_BACK)
    config.setdefault("grid_h", GRID_H)
    height, width = config["grid_h"], config["grid_w"]
    grid_tokens = state_dict["grid_pos"].shape[1]
    stem_in = state_dict["grid_stem.0.weight"].shape[1]
    if "patch_h" not in config or "patch_w" not in config:
        area = height * width // grid_tokens
        shapes = [(h, area // h) for h in range(1, height + 1)
                  if area % h == 0 and height % h == 0 and width % (area // h) == 0]
        if not shapes or height * width % grid_tokens:
            raise ValueError(f"не подобрать патч под {grid_tokens} токенов карты {height}×{width}")
        patch_h, patch_w = (PATCH, PATCH) if (PATCH, PATCH) in shapes else min(shapes, key=lambda s: abs(s[0] - s[1]))
        config.setdefault("patch_h", patch_h)
        config.setdefault("patch_w", patch_w)
    if (height // config["patch_h"]) * (width // config["patch_w"]) != grid_tokens:
        raise ValueError(f"в весах {grid_tokens} токенов карты, а окно {height}×{width} "
                         f"патчами {config['patch_h']}×{config['patch_w']} даёт другое число")
    config.setdefault("channels", stem_in // (config["patch_h"] * config["patch_w"]))
    config.setdefault("player_features", state_dict["player_embed.0.weight"].shape[1])
    config.setdefault("actions", state_dict["policy_head.2.weight"].shape[0])
    return config


def widen_grid_pos(source: torch.Tensor, old_config: dict, model: "MarioJev") -> torch.Tensor:
    """Позиции токенов карты на прежнем окне → на окне модели.

    Окно расширялось по краям (назад — с 4 до 8 столбцов), патч тот же:
    прежний токен (ряд r, столбец c) — это новый (r, c + сдвиг), где сдвиг —
    добавленные сзади столбцы в патчах. Новые столбцы начинаются копией
    крайнего прежнего: для модели они сначала «там же, где край», и
    различать их она учится на новых примерах."""

    ph, pw = model.patch_h, model.patch_w
    if (old_config["patch_h"], old_config["patch_w"], old_config["grid_h"]) != (ph, pw, model.grid_h):
        raise ValueError(f"патч модели {ph}×{pw} (карта {model.grid_h} строк) не совпадает с исходным "
                         f"{old_config['patch_h']}×{old_config['patch_w']} ({old_config['grid_h']} строк); "
                         f"учите с нуля (--init none)")
    back = model.grid_back - old_config["grid_back"]
    ahead = (model.grid_w - model.grid_back) - (old_config["grid_w"] - old_config["grid_back"])
    if back < 0 or ahead < 0 or back % pw or ahead % pw:
        raise ValueError(f"окно {old_config['grid_w']} ({old_config['grid_back']} позади) не переносится в "
                         f"{model.grid_w} ({model.grid_back} позади) патчами шириной {pw}")
    rows, d = model.grid_h // ph, source.shape[-1]
    old_cols, new_cols, shift = old_config["grid_w"] // pw, model.grid_w // pw, back // pw
    grid = source.reshape(1, rows, old_cols, d)
    value = torch.cat([grid[:, :, :1].expand(-1, -1, shift, -1), grid,
                       grid[:, :, -1:].expand(-1, -1, new_cols - old_cols - shift, -1)], dim=2)
    return value.reshape(1, rows * new_cols, d).clone()


def expand_state_dict(old: dict, model: "MarioJev", old_config: dict | None = None) -> dict:
    """Веса модели прошлой версии → веса модели с новыми каналами,
    признаками, действиями и окном карты (для продолжения обучения).

    Новые входы дописаны в конец (ai/features.py), поэтому старые веса
    ложатся в начало, а новые столбцы — нулями: пока модель им не
    научится, она видит мир как раньше. Новое действие начинается копией
    «стоять» (смещение на 0.5 ниже): вне трубы-входа оно и есть «стоять».
    Окно карты шире прежнего — позиции токенов переносятся со сдвигом
    (widen_grid_pos). old_config — конфиг исходного чекпойнта; без него
    окно и патч выводятся из весов, как у старых чекпойнтов."""

    old_config = config_from_state_dict(old, old_config or {})
    new = model.state_dict()
    out = {}
    patch = model.patch_h * model.patch_w
    same_grid = all(old_config[k] == model.config[k] for k in ("patch_h", "patch_w", "grid_h", "grid_w", "grid_back"))

    for key, target in new.items():
        source = old.get(key)
        if source is None:
            out[key] = target
            continue
        if key == "grid_pos" and not same_grid:
            # Другое окно или нарезка карты — токены другие, даже если их столько же
            out[key] = widen_grid_pos(source, old_config, model)
            continue
        if source.shape == target.shape:
            out[key] = source
            continue

        value = torch.zeros_like(target)
        if key == "grid_stem.0.weight":
            # Столбцы — патч (канал, ph, pw) подряд по каналам
            old_channels = source.shape[1] // patch
            value[:, :old_channels * patch] = source
        elif key == "player_embed.0.weight":
            # База признаков, one-hot старых действий и признаки после него
            # (комната, труба) — каждый на своё место в новой раскладке
            from ai.features import N_ACTIONS, PLAYER_BASE
            old_actions = old["policy_head.2.weight"].shape[0]
            value[:, :PLAYER_BASE + old_actions] = source[:, :PLAYER_BASE + old_actions]
            old_extra = source.shape[1] - PLAYER_BASE - old_actions
            if old_extra > 0:
                start = PLAYER_BASE + N_ACTIONS
                value[:, start:start + old_extra] = source[:, PLAYER_BASE + old_actions:]
        elif key in ("policy_head.2.weight", "policy_head.2.bias"):
            # Новое действие «вниз» вне трубы-входа — то же, что «стоять»:
            # начинаем с копии «стоять» чуть ниже, чтобы выбор не сломался,
            # а на трубе было что пробовать
            value[:source.shape[0]] = source
            value[source.shape[0]:] = source[0]
            if key.endswith("bias"):
                value[source.shape[0]:] -= 0.5
        else:
            raise ValueError(f"не знаю, как расширить {key}: {tuple(source.shape)} → {tuple(target.shape)}")
        out[key] = value

    return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

"""MarioJev — модель решений в духе Jev: не пишет текст, а отвечает на
типизированные вопросы вероятностями.

Вход — типизированное состояние (ai/features.py), выход — четыре головы:

- choice (move)   — какое нажатие сделать: распределение по 6 действиям;
- value           — критик для обучения (ожидаемая награда);
- noul (danger)   — вероятность погибнуть в ближайшие 2 секунды;
- score (points)  — сколько очков ещё будет набрано до конца уровня,
                    распределение по диапазонам.

Устройство: карта 8×16×24 режется на патчи 4×4 → 24 токена (как в ViT),
+ токен на каждого врага + токен игрока + CLS; трансформер из 3 слоёв
смешивает их («враг в двух тайлах, а над ним блок “?”»); головы читают CLS.

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
    GRID_H,
    GRID_W,
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


PATCH = 4


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

    def __init__(self, d_model: int = 96, heads: int = 4, layers: int = 3, dropout: float = 0.0):
        super().__init__()

        self.config = {"d_model": d_model, "heads": heads, "layers": layers}

        # --- карта: патчи 4×4 → 4×6 = 24 токена ------------------------
        patch_size = N_CHANNELS * PATCH * PATCH
        self.grid_stem = nn.Sequential(
            nn.Linear(patch_size, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )
        self.grid_tokens = (GRID_H // PATCH) * (GRID_W // PATCH)
        self.grid_pos = nn.Parameter(torch.zeros(1, self.grid_tokens, d_model))

        # --- враги и игрок -------------------------------------------
        self.enemy_embed = nn.Sequential(
            nn.Linear(ENEMY_FEATURES, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.player_embed = nn.Sequential(
            nn.Linear(PLAYER_FEATURES, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        # Тип токена: 0 CLS, 1 карта, 2 враг, 3 игрок
        self.type_embed = nn.Embedding(4, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        self.blocks = nn.ModuleList([Block(d_model, heads, 2 * d_model) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

        # --- головы --------------------------------------------------
        self.policy_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, N_ACTIONS))
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

        # B, C, H, W → B, (H/4)·(W/4), C·4·4
        patches = grid.view(batch, N_CHANNELS, GRID_H // PATCH, PATCH, GRID_W // PATCH, PATCH)
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

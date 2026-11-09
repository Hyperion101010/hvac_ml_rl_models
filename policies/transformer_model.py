"""Causal Transformer encoder with outcome, Q, and BC heads.

Act/train use Q + BC only; outcome head is kept for checkpoint compatibility.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from policies.base import ACTIONS
from policies.features import CAT_FIELDS, FeatureSchema, MAX_LEN, NUM_FIELDS
from policies.outcome_values import OUTCOME_CLASSES


class StepEmbedding(nn.Module):
    def __init__(self, schema: FeatureSchema, d_model: int = 128) -> None:
        super().__init__()
        self.schema = schema
        self.d_model = d_model
        self.cat_embs = nn.ModuleList(
            [
                nn.Embedding(len(schema.vocabs[f]), d_model)
                for f in CAT_FIELDS
            ]
        )
        self.weather_emb = nn.Embedding(len(schema.vocabs["weather_extreme"]), d_model)
        self.action_emb = nn.Embedding(len(schema.vocabs["action"]), d_model)
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)
        self.num_proj = nn.Linear(len(NUM_FIELDS), d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        cats: torch.Tensor,
        nums: torch.Tensor,
        actions: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # cats: [B, L, n_cats] last dim weather separate as index len(CAT_FIELDS)
        x = self.action_emb(actions) + self.pos_emb(positions)
        for i, emb in enumerate(self.cat_embs):
            x = x + emb(cats[..., i])
        x = x + self.weather_emb(cats[..., len(CAT_FIELDS)])
        x = x + self.num_proj(nums)
        return self.norm(self.out(x))


class CausalTransformerPolicyNet(nn.Module):
    def __init__(
        self,
        schema: FeatureSchema,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        action_conditioned_q: bool = False,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.d_model = d_model
        self.n_actions = len(ACTIONS)
        self.n_outcomes = len(OUTCOME_CLASSES)
        self.action_conditioned_q = bool(action_conditioned_q)
        self.embed = StepEmbedding(schema, d_model=d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.n_actions * self.n_outcomes),
        )
        if self.action_conditioned_q:
            self.q_action_emb = nn.Embedding(self.n_actions, d_model)
            self.q_head = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.q_action_emb = None
            self.q_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.n_actions),
            )
        self.bc_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.n_actions),
        )

    def _q_values(self, h_t: torch.Tensor) -> torch.Tensor:
        if not self.action_conditioned_q:
            return self.q_head(h_t)
        b = h_t.size(0)
        a_ids = torch.arange(self.n_actions, device=h_t.device)
        a_emb = self.q_action_emb(a_ids).unsqueeze(0).expand(b, -1, -1)
        h_exp = h_t.unsqueeze(1).expand(-1, self.n_actions, -1)
        return self.q_head(torch.cat([h_exp, a_emb], dim=-1)).squeeze(-1)

    def encode(
        self,
        cats: torch.Tensor,
        nums: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, length, _ = cats.shape
        positions = torch.arange(length, device=cats.device).unsqueeze(0).expand(b, -1)
        # map absolute positions among valid tokens: use arange; pad left so last is current
        h = self.embed(cats, nums, actions, positions)
        # Use bool masks consistently (True = blocked / pad).
        causal = torch.triu(
            torch.ones(length, length, device=cats.device, dtype=torch.bool), diagonal=1
        )
        key_pad = ~mask  # True = pad
        encoded = self.encoder(h, mask=causal, src_key_padding_mask=key_pad)
        # last valid index in the right-padded prefix
        lengths = mask.long().sum(dim=1).clamp(min=1)
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d_model)
        h_t = encoded.gather(1, idx).squeeze(1)
        h_t = torch.nan_to_num(h_t, nan=0.0)
        return h_t

    def forward(
        self,
        cats: torch.Tensor,
        nums: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h_t = self.encode(cats, nums, actions, mask)
        outcome_logits = self.outcome_head(h_t).view(-1, self.n_actions, self.n_outcomes)
        q_values = self._q_values(h_t)
        bc_logits = self.bc_head(h_t)
        return {
            "h": h_t,
            "outcome_logits": outcome_logits,
            "q_values": q_values,
            "bc_logits": bc_logits,
        }


class CausalTransformerAWBCNet(nn.Module):
    """Encoder + policy logits + state value for advantage-weighted BC."""

    def __init__(
        self,
        schema: FeatureSchema,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.d_model = d_model
        self.n_actions = len(ACTIONS)
        self.embed = StepEmbedding(schema, d_model=d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pi_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.n_actions),
        )
        self.v_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def encode(
        self,
        cats: torch.Tensor,
        nums: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, length, _ = cats.shape
        positions = torch.arange(length, device=cats.device).unsqueeze(0).expand(b, -1)
        h = self.embed(cats, nums, actions, positions)
        causal = torch.triu(
            torch.ones(length, length, device=cats.device, dtype=torch.bool), diagonal=1
        )
        key_pad = ~mask
        encoded = self.encoder(h, mask=causal, src_key_padding_mask=key_pad)
        lengths = mask.long().sum(dim=1).clamp(min=1)
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d_model)
        h_t = encoded.gather(1, idx).squeeze(1)
        return torch.nan_to_num(h_t, nan=0.0)

    def forward(
        self,
        cats: torch.Tensor,
        nums: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h_t = self.encode(cats, nums, actions, mask)
        return {
            "h": h_t,
            "pi_logits": self.pi_head(h_t),
            "v": self.v_head(h_t).squeeze(-1),
        }


def bc_support_mask(bc_logits: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    probs = F.softmax(bc_logits, dim=-1)
    support = probs >= threshold
    # ensure at least argmax
    argmax = probs.argmax(dim=-1)
    support.scatter_(1, argmax.unsqueeze(-1), True)
    return support


def masked_argmax(scores: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    very_neg = torch.finfo(scores.dtype).min / 4
    masked = scores.masked_fill(~support, very_neg)
    return masked.argmax(dim=-1)


# Re-export for older imports
from policies.safety import RISK_ESCALATE, should_escalate  # noqa: E402

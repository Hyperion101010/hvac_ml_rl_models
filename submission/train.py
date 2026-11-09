"""Train causal transformer with BC CE + conservative FQI (Q-only act).

Self-contained copy of train/train_transformer_fqi.py for submission/.
Vendored: ACTIONS, features, outcome_values, transformer model, safety helpers.
Checkpoint selection still uses eval.ope from the parent repo when available.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ACTIONS = (
    "WAIT",
    "CHECK_IN",
    "ESTIMATE_NUDGE",
    "ASK_OBJECTION",
    "OFFER_SCHEDULING",
    "MEMBERSHIP_TOUCH",
    "PARK_THREAD",
    "ESCALATE_TO_HUMAN",
)

# ---- vendored policies/features.py ----
"""Step tokenization and history sequence building for the causal transformer.

Token time order: [hist_0, ..., hist_{t-1}, current_obs].
All robustness lives here / in act — do not change the act signature.
"""


MAX_LEN = 4
PAD_ACTION = "<PAD>"
UNK = "<UNK>"

CAT_FIELDS = (
    "topic",
    "equipment_type",
    "season",
    "track_hint",
    "heat",
    "estimate_value_bucket",
    "membership_status",
    "customer_signal",
)

NUM_FIELDS = (
    "turn",
    "day",
    "ignored_count",
    "touch_count",
    "user_patience",
    "urgency_score",
    "price_sensitivity_score",
    "service_risk_score",
    "relationship_score",
)

DEFAULT_VOCABS: Dict[str, List[str]] = {
    "topic": [
        "aging_equipment",
        "competitor_quote",
        "install_warranty",
        "maintenance_checkin",
        "membership_renewal",
        "open_estimate",
        "plan_visit",
        "repair_vs_replace",
        "service_issue",
        "weather_tip",
    ],
    "equipment_type": [
        "air_quality",
        "boiler",
        "central_ac",
        "furnace",
        "heat_pump",
        "mini_split",
        "thermostat",
    ],
    "season": ["fall", "spring", "summer", "winter"],
    "track_hint": ["deal", "nurture", "relationship"],
    "heat": ["cold", "hot", "warm"],
    "estimate_value_bucket": ["high", "low", "medium", "none"],
    "membership_status": ["active", "expiring", "lapsed", "none"],
    "customer_signal": [
        "ask_spouse",
        "competitor",
        "none",
        "positive",
        "price_objection",
        "timing_delay",
        "unhappy",
    ],
    "weather_extreme": ["False", "True"],
    "action": [PAD_ACTION, *ACTIONS],
}


def _stoid(vocab: List[str]) -> Dict[str, int]:
    return {v: i for i, v in enumerate(vocab)}


class FeatureSchema:
    def __init__(self, vocabs: Optional[Dict[str, List[str]]] = None) -> None:
        self.vocabs = {k: list(v) for k, v in (vocabs or DEFAULT_VOCABS).items()}
        self.cat_to_id = {k: _stoid(self.vocabs[k]) for k in CAT_FIELDS}
        self.cat_to_id["weather_extreme"] = _stoid(self.vocabs["weather_extreme"])
        self.action_to_id = _stoid(self.vocabs["action"])
        self.unk_action = self.action_to_id.get(UNK, 0)
        self.pad_action_id = self.action_to_id[PAD_ACTION]
        self.n_actions = len(ACTIONS)
        self.action_name_to_idx = {a: i for i, a in enumerate(ACTIONS)}

    def cat_id(self, field: str, value: Any) -> int:
        key = str(value)
        table = self.cat_to_id[field]
        if key not in table:
            # reserve 0-style unk by appending if needed; map unknown to 0 of vocab
            return 0
        return table[key]

    def action_id(self, action: Optional[str]) -> int:
        if action is None or action == PAD_ACTION:
            return self.pad_action_id
        return self.action_to_id.get(str(action), self.pad_action_id)

    def encode_step(
        self, observation: Dict[str, Any], action: Optional[str]
    ) -> Dict[str, Any]:
        cats = [self.cat_id(f, observation.get(f, UNK)) for f in CAT_FIELDS]
        cats.append(self.cat_id("weather_extreme", bool(observation.get("weather_extreme", False))))
        nums = []
        for f in NUM_FIELDS:
            try:
                nums.append(float(observation.get(f, 0.0) or 0.0))
            except (TypeError, ValueError):
                nums.append(0.0)
        # normalize day-ish counts lightly
        day = nums[NUM_FIELDS.index("day")]
        nums[NUM_FIELDS.index("day")] = min(day, 30.0) / 30.0
        turn = nums[NUM_FIELDS.index("turn")]
        nums[NUM_FIELDS.index("turn")] = turn / 3.0
        for name in ("ignored_count", "touch_count", "user_patience"):
            idx = NUM_FIELDS.index(name)
            nums[idx] = nums[idx] / 3.0
        return {
            "cats": cats,
            "nums": nums,
            "action_id": self.action_id(action),
            "turn": int(observation.get("turn", 0) or 0),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"vocabs": self.vocabs}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSchema":
        return cls(vocabs=data.get("vocabs"))


def _item_case_id(item: Dict[str, Any]) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    if "case_id" in item:
        return str(item["case_id"])
    obs = item.get("observation")
    if isinstance(obs, dict) and "case_id" in obs:
        return str(obs["case_id"])
    return None


def _item_obs(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    obs = item.get("observation")
    if isinstance(obs, dict):
        return obs
    # allow raw observation-shaped dicts
    if "topic" in item and "turn" in item:
        return item
    return None


def _item_action(item: Dict[str, Any]) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    a = item.get("action")
    return str(a) if a is not None else None


def _item_turn(item: Dict[str, Any]) -> int:
    obs = _item_obs(item) or {}
    try:
        return int(obs.get("turn", 0) or 0)
    except (TypeError, ValueError):
        return 0


def filter_history(
    observation: Dict[str, Any], history: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    case_id = str(observation.get("case_id", ""))
    filtered: List[Dict[str, Any]] = []
    for item in history or []:
        cid = _item_case_id(item)
        if case_id and cid is not None and cid != case_id:
            continue
        if _item_obs(item) is None:
            continue
        filtered.append(item)
    filtered.sort(key=_item_turn)
    return filtered


def history_actions_match(
    filtered: Sequence[Dict[str, Any]], observation: Dict[str, Any]
) -> bool:
    prev = list(observation.get("previous_actions") or [])
    got = [_item_action(h) for h in filtered]
    got = [a for a in got if a]
    return got == prev


def build_sequence(
    observation: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    schema: Optional[FeatureSchema] = None,
) -> Tuple[List[Dict[str, Any]], List[bool]]:
    """Return (step_feature_dicts length<=4, pad_mask True=valid)."""

    schema = schema or FeatureSchema()
    filtered = filter_history(observation, history)
    steps: List[Tuple[Dict[str, Any], Optional[str]]] = []

    if filtered and history_actions_match(filtered, observation):
        for item in filtered:
            obs = _item_obs(item)
            assert obs is not None
            steps.append((obs, _item_action(item)))
        steps.append((observation, None))
    else:
        # Fallback: current observation only; action history from previous_actions
        # as synthetic prior tokens with minimal state (copy current static fields).
        prev = list(observation.get("previous_actions") or [])
        for i, action in enumerate(prev[- (MAX_LEN - 1) :]):
            # Use a shallow copy of current obs with turn approximated
            syn = dict(observation)
            syn["turn"] = i
            syn["previous_actions"] = prev[: i + 1]
            syn["last_action"] = action
            steps.append((syn, action))
        steps.append((observation, None))

    # Keep last MAX_LEN tokens (current must remain last)
    if len(steps) > MAX_LEN:
        steps = steps[-(MAX_LEN):]

    encoded = [schema.encode_step(obs, action) for obs, action in steps]
    valid = [True] * len(encoded)
    # Right-pad so valid tokens are a contiguous prefix (avoids NaNs from
    # attending when every key is padding under a causal+pad mask).
    while len(encoded) < MAX_LEN:
        pad = {
            "cats": [0] * (len(CAT_FIELDS) + 1),
            "nums": [0.0] * len(NUM_FIELDS),
            "action_id": schema.pad_action_id,
            "turn": 0,
        }
        encoded.append(pad)
        valid.append(False)
    return encoded, valid


def collate_sequences(
    batch: Sequence[Tuple[List[Dict[str, Any]], List[bool]]],
) -> Dict[str, Any]:
    import torch

    b = len(batch)
    n_cats = len(CAT_FIELDS) + 1
    n_nums = len(NUM_FIELDS)
    cats = torch.zeros(b, MAX_LEN, n_cats, dtype=torch.long)
    nums = torch.zeros(b, MAX_LEN, n_nums, dtype=torch.float32)
    actions = torch.zeros(b, MAX_LEN, dtype=torch.long)
    turns = torch.zeros(b, MAX_LEN, dtype=torch.long)
    mask = torch.zeros(b, MAX_LEN, dtype=torch.bool)
    for i, (steps, valid) in enumerate(batch):
        for t in range(MAX_LEN):
            cats[i, t] = torch.tensor(steps[t]["cats"], dtype=torch.long)
            nums[i, t] = torch.tensor(steps[t]["nums"], dtype=torch.float32)
            actions[i, t] = int(steps[t]["action_id"])
            turns[i, t] = int(steps[t]["turn"])
            mask[i, t] = bool(valid[t])
    return {
        "cats": cats,
        "nums": nums,
        "actions": actions,
        "turns": turns,
        "mask": mask,
    }

# ---- vendored policies/outcome_values.py ----
from typing import Iterable, Mapping
"""Terminal outcome value table. continue / max_steps use logged r, not this table."""

OUTCOME_CLASSES = (
    "continue",
    "max_steps",
    "sold_estimate",
    "booked_appointment",
    "membership_converted",
    "positive_review",
    "opted_out",
    "escalated",
    "parked",
)

# Used in V_imm at decision time (not continue / max_steps)
TABLE_OUTCOMES = (
    "sold_estimate",
    "booked_appointment",
    "membership_converted",
    "positive_review",
    "opted_out",
    "escalated",
    "parked",
)

CONTINUE_IDX = OUTCOME_CLASSES.index("continue")
MAX_STEPS_IDX = OUTCOME_CLASSES.index("max_steps")


class OutcomeValues:
    def __init__(self, values: Mapping[str, float], escalate_unhappy: float, escalate_other: float):
        self.values = dict(values)
        self.escalate_unhappy = float(escalate_unhappy)
        self.escalate_other = float(escalate_other)

    def value_for(self, outcome: str, unhappy: bool = False) -> float:
        if outcome == "escalated":
            return self.escalate_unhappy if unhappy else self.escalate_other
        if outcome in ("continue", "max_steps"):
            raise KeyError(f"{outcome} has no fixed table value; use logged reward")
        return float(self.values[outcome])

    def tensor_table(self, unhappy_flags) -> "Any":
        """Return [B, C] table values; continue/max_steps slots are 0 (unused in V_imm)."""
        import torch

        if not torch.is_tensor(unhappy_flags):
            unhappy_flags = torch.tensor(unhappy_flags, dtype=torch.bool)
        b = unhappy_flags.shape[0]
        table = torch.zeros(b, len(OUTCOME_CLASSES), dtype=torch.float32)
        for i, name in enumerate(OUTCOME_CLASSES):
            if name in ("continue", "max_steps"):
                continue
            if name == "escalated":
                table[:, i] = torch.where(
                    unhappy_flags,
                    torch.full((), self.escalate_unhappy),
                    torch.full((), self.escalate_other),
                )
            else:
                table[:, i] = self.values.get(name, 0.0)
        return table

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "escalate_unhappy": self.escalate_unhappy,
            "escalate_other": self.escalate_other,
            "outcome_classes": list(OUTCOME_CLASSES),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutcomeValues":
        return cls(
            values=data["values"],
            escalate_unhappy=data["escalate_unhappy"],
            escalate_other=data["escalate_other"],
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OutcomeValues":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def compute_outcome_values(rows: Iterable[Dict[str, Any]]) -> OutcomeValues:
    buckets: Dict[str, List[float]] = defaultdict(list)
    esc_u: List[float] = []
    esc_o: List[float] = []
    for row in rows:
        outcome = row["info"]["outcome"]
        rew = float(row["reward"])
        if outcome == "escalated":
            if row["observation"].get("customer_signal") == "unhappy":
                esc_u.append(rew)
            else:
                esc_o.append(rew)
        else:
            buckets[outcome].append(rew)

    values = {}
    for name in TABLE_OUTCOMES:
        if name == "escalated":
            continue
        xs = buckets.get(name) or [0.0]
        values[name] = sum(xs) / len(xs)
    return OutcomeValues(
        values=values,
        escalate_unhappy=(sum(esc_u) / len(esc_u)) if esc_u else 20.0,
        escalate_other=(sum(esc_o) / len(esc_o)) if esc_o else -18.0,
    )


def outcome_to_index(name: str) -> int:
    return OUTCOME_CLASSES.index(name)

# ---- vendored policies/safety.py ----
"""Shared hard safety rules (torch-free)."""

RISK_ESCALATE = 0.75
PARK_IGNORED_COUNT = 2


def should_escalate(observation: Dict[str, Any]) -> bool:
    if observation.get("customer_signal") == "unhappy":
        return True
    try:
        return float(observation.get("service_risk_score", 0.0) or 0.0) > RISK_ESCALATE
    except (TypeError, ValueError):
        return False


def should_park(observation: Dict[str, Any]) -> bool:
    """Hard park after repeated ignores (best-return action in logs)."""
    try:
        return int(observation.get("ignored_count", 0) or 0) >= PARK_IGNORED_COUNT
    except (TypeError, ValueError):
        return False


def hard_safety_action(observation: Dict[str, Any]) -> Optional[str]:
    """Escalate first, then park; None if no hard rule fires."""
    if should_escalate(observation):
        return "ESCALATE_TO_HUMAN"
    if should_park(observation):
        return "PARK_THREAD"
    return None

# ---- vendored policies/transformer_model.py (FQI net only) ----
"""Causal Transformer encoder with outcome, Q, and BC heads.

Act/train use Q + BC only; outcome head is kept for checkpoint compatibility.
"""



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


def bc_support_mask(
    bc_logits: torch.Tensor,
    threshold: float = 0.01,
    force_action_idx: Optional[int] = None,
) -> torch.Tensor:
    probs = F.softmax(bc_logits, dim=-1)
    support = probs >= threshold
    # ensure at least argmax
    argmax = probs.argmax(dim=-1)
    support.scatter_(1, argmax.unsqueeze(-1), True)
    if force_action_idx is not None:
        support[:, int(force_action_idx)] = True
    return support


def masked_argmax(scores: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    very_neg = torch.finfo(scores.dtype).min / 4
    masked = scores.masked_fill(~support, very_neg)
    return masked.argmax(dim=-1)


# Re-export for older imports

# ---- train loop ----


SUBMISSION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SUBMISSION_ROOT.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROOT = SUBMISSION_ROOT


def load_episodes(path: Path, limit_rows: int = 0) -> List[List[Dict[str, Any]]]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            eid = row["episode_id"]
            if eid not in by:
                order.append(eid)
            by[eid].append(row)
            if limit_rows > 0 and i + 1 >= limit_rows:
                break
    episodes = []
    for eid in order:
        episodes.append(sorted(by[eid], key=lambda r: r["step"]))
    return episodes


def history_from_prior_steps(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "observation": dict(s["observation"]),
            "action": s["action"],
            "reward": 0.0,
            "done": False,
        }
        for s in steps
    ]


class TransitionDataset(Dataset):
    def __init__(self, episodes: Sequence[Sequence[Dict[str, Any]]], schema: FeatureSchema):
        self.schema = schema
        self.samples: List[Dict[str, Any]] = []
        for ep in episodes:
            for t, row in enumerate(ep):
                hist = history_from_prior_steps(ep[:t])
                cur_steps, cur_mask = build_sequence(row["observation"], hist, schema)
                next_steps = next_mask = None
                if (not row["done"]) and row.get("next_observation"):
                    next_hist = history_from_prior_steps(ep[:t]) + [
                        {
                            "observation": dict(row["observation"]),
                            "action": row["action"],
                            "reward": 0.0,
                            "done": False,
                        }
                    ]
                    next_steps, next_mask = build_sequence(
                        row["next_observation"], next_hist, schema
                    )
                self.samples.append(
                    {
                        "cur": (cur_steps, cur_mask),
                        "next": (next_steps, next_mask) if next_steps is not None else None,
                        "action_idx": ACTIONS.index(row["action"]),
                        "outcome_idx": outcome_to_index(row["info"]["outcome"]),
                        "reward": float(row["reward"]),
                        "done": bool(row["done"]),
                        "unhappy": row["observation"].get("customer_signal") == "unhappy",
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_transitions(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cur = collate_sequences([b["cur"] for b in batch])
    has_next = [b["next"] is not None for b in batch]
    next_batch = []
    for b in batch:
        if b["next"] is not None:
            next_batch.append(b["next"])
        else:
            next_batch.append(b["cur"])
    nxt = collate_sequences(next_batch)
    return {
        "cur": cur,
        "next": nxt,
        "has_next": torch.tensor(has_next, dtype=torch.bool),
        "action_idx": torch.tensor([b["action_idx"] for b in batch], dtype=torch.long),
        "outcome_idx": torch.tensor([b["outcome_idx"] for b in batch], dtype=torch.long),
        "reward": torch.tensor([b["reward"] for b in batch], dtype=torch.float32),
        "done": torch.tensor([b["done"] for b in batch], dtype=torch.bool),
        "unhappy": torch.tensor([b["unhappy"] for b in batch], dtype=torch.bool),
    }


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.05) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def compute_batch_losses(
    model: CausalTransformerPolicyNet,
    target: CausalTransformerPolicyNet,
    batch: Dict[str, Any],
    device: torch.device,
    gamma: float,
    reward_scale: float,
    bc_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cur = {k: v.to(device) for k, v in batch["cur"].items()}
    nxt = {k: v.to(device) for k, v in batch["next"].items()}
    action_idx = batch["action_idx"].to(device)
    reward = batch["reward"].to(device)
    done = batch["done"].to(device)
    has_next = batch["has_next"].to(device)

    out = model(cur["cats"], cur["nums"], cur["actions"], cur["mask"])
    loss_bc = F.cross_entropy(out["bc_logits"], action_idx)

    with torch.no_grad():
        tgt = target(nxt["cats"], nxt["nums"], nxt["actions"], nxt["mask"])
        support = bc_support_mask(tgt["bc_logits"], threshold=bc_thresh)
        q_next = tgt["q_values"].masked_fill(
            ~support, torch.finfo(tgt["q_values"].dtype).min / 4
        )
        max_next = q_next.max(dim=-1).values
        bootstrap = (~done) & has_next
        y = reward / reward_scale + gamma * bootstrap.float() * max_next

    q_pred = out["q_values"][torch.arange(len(action_idx), device=device), action_idx]
    loss_q = F.smooth_l1_loss(q_pred, y)
    return loss_bc, loss_q


def evaluate_loader(
    model: CausalTransformerPolicyNet,
    target: CausalTransformerPolicyNet,
    loader: DataLoader,
    device: torch.device,
    gamma: float,
    reward_scale: float,
    bc_thresh: float,
    lam: float,
) -> Dict[str, float]:
    model.eval()
    total = total_bc = total_q = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            loss_bc, loss_q = compute_batch_losses(
                model, target, batch, device, gamma, reward_scale, bc_thresh
            )
            loss = loss_bc + lam * loss_q
            total += float(loss.item())
            total_bc += float(loss_bc.item())
            total_q += float(loss_q.item())
            n_batches += 1
    denom = max(n_batches, 1)
    return {
        "loss": total / denom,
        "bc": total_bc / denom,
        "q": total_q / denom,
    }


def save_checkpoint(
    path: Path,
    model: CausalTransformerPolicyNet,
    schema: FeatureSchema,
    outcome_values: OutcomeValues,
    args: argparse.Namespace,
    gamma: float,
    lam: float,
    bc_thresh: float,
    reward_scale: float,
    *,
    epoch: int,
    selection_metric: str,
    selection_value: float,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "schema": schema.to_dict(),
        "outcome_values": outcome_values.to_dict(),
        "config": {
            "kind": "fqi",
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "gamma": gamma,
            "bc_threshold": bc_thresh,
            "lambda_q": lam,
            "reward_scale": reward_scale,
            "tau": float(args.tau),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "epochs_planned": int(args.epochs),
            "epoch": int(epoch),
            "seed": int(args.seed),
            "action_conditioned_q": bool(getattr(args, "action_conditioned_q", False)),
            "selection_metric": selection_metric,
            "selection_value": float(selection_value),
            "actions": list(ACTIONS),
            "outcome_classes": list(OUTCOME_CLASSES),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def _ope_wis_for_model(
    model: CausalTransformerPolicyNet,
    schema: FeatureSchema,
    episodes: Sequence[Sequence[Dict[str, Any]]],
    bc_thresh: float,
    gamma: float,
    clip: float,
) -> float:
    """Trajectory WIS of current weights on logged-history valid episodes."""
    from eval.ope import evaluate_ope

    cpu_model = copy.deepcopy(model).cpu().eval()
    park_idx = ACTIONS.index("PARK_THREAD")

    def act(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        hard = hard_safety_action(observation)
        if hard is not None:
            return hard
        steps, valid = build_sequence(observation, history, schema)
        batch = collate_sequences([(steps, valid)])
        with torch.no_grad():
            out = cpu_model(
                batch["cats"], batch["nums"], batch["actions"], batch["mask"]
            )
        force = park_idx if should_park(observation) else None
        support = bc_support_mask(
            out["bc_logits"], threshold=bc_thresh, force_action_idx=force
        )
        idx = int(masked_argmax(out["q_values"], support)[0].item())
        return ACTIONS[idx]

    metrics = evaluate_ope(
        "train_select", act, episodes, gamma=gamma, clip=clip, q_fn=None
    )
    return float(metrics["wis"])


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_episodes = load_episodes(Path(args.train_logs), limit_rows=args.limit_rows)
    schema = FeatureSchema()
    flat_rows = [r for ep in train_episodes for r in ep]
    outcome_values = compute_outcome_values(flat_rows)

    dataset = TransitionDataset(train_episodes, schema)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_transitions,
        num_workers=0,
    )

    valid_path = Path(args.valid_logs)
    valid_loader: Optional[DataLoader] = None
    valid_episodes: List[List[Dict[str, Any]]] = []
    ope_episodes: List[List[Dict[str, Any]]] = []
    if valid_path.exists():
        valid_episodes = load_episodes(valid_path, limit_rows=args.limit_rows)
        valid_dataset = TransitionDataset(valid_episodes, schema)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_transitions,
            num_workers=0,
        )
        n_ope = int(args.ope_episodes)
        ope_episodes = valid_episodes[:n_ope] if n_ope > 0 else valid_episodes
    else:
        print(f"warning: valid logs not found at {valid_path}; skipping validation")

    acq = bool(args.action_conditioned_q)
    model = CausalTransformerPolicyNet(
        schema,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        action_conditioned_q=acq,
    ).to(device)
    target = copy.deepcopy(model).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "transformer_fqi_acq.pt" if acq else "transformer_fqi.pt"
    ckpt_path = out_dir / ckpt_name
    if acq:
        print(f"action-conditioned Q enabled; checkpoint -> {ckpt_path}")

    gamma = args.gamma
    lam = args.lambda_q
    bc_thresh = args.bc_threshold
    reward_scale = float(args.reward_scale)
    if reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")

    best_ope = float("-inf")
    best_epoch = -1
    select_by_ope = bool(args.select_by_ope) and bool(ope_episodes)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = total_bc = total_q = 0.0
        n_batches = 0
        for batch in loader:
            loss_bc, loss_q = compute_batch_losses(
                model, target, batch, device, gamma, reward_scale, bc_thresh
            )
            loss = loss_bc + lam * loss_q
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            soft_update(target, model, tau=args.tau)

            total += float(loss.item())
            total_bc += float(loss_bc.item())
            total_q += float(loss_q.item())
            n_batches += 1

        denom = max(n_batches, 1)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={total/denom:.4f} "
            f"bc={total_bc/denom:.4f} "
            f"q={total_q/denom:.4f}"
        )

        if valid_loader is not None:
            metrics = evaluate_loader(
                model, target, valid_loader, device, gamma, reward_scale, bc_thresh, lam
            )
            print(
                f"epoch {epoch}/{args.epochs} "
                f"valid_loss={metrics['loss']:.4f} "
                f"bc={metrics['bc']:.4f} "
                f"q={metrics['q']:.4f}"
            )
            if select_by_ope:
                wis = _ope_wis_for_model(
                    model,
                    schema,
                    ope_episodes,
                    bc_thresh,
                    gamma,
                    clip=float(args.ope_clip),
                )
                print(f"epoch {epoch}/{args.epochs} valid_ope_wis={wis:.4f}")
                if wis > best_ope:
                    best_ope = wis
                    best_epoch = epoch
                    save_checkpoint(
                        ckpt_path,
                        model,
                        schema,
                        outcome_values,
                        args,
                        gamma,
                        lam,
                        bc_thresh,
                        reward_scale,
                        epoch=epoch,
                        selection_metric="ope_wis",
                        selection_value=wis,
                    )
                    print(f"  saved best checkpoint (ope_wis={best_ope:.4f})")
            else:
                score = -float(metrics["loss"])
                if score > best_ope:
                    best_ope = score
                    best_epoch = epoch
                    save_checkpoint(
                        ckpt_path,
                        model,
                        schema,
                        outcome_values,
                        args,
                        gamma,
                        lam,
                        bc_thresh,
                        reward_scale,
                        epoch=epoch,
                        selection_metric="neg_valid_loss",
                        selection_value=float(metrics["loss"]),
                    )
                    print(
                        f"  saved best checkpoint (valid_loss={metrics['loss']:.4f})"
                    )

    if valid_loader is None:
        save_checkpoint(
            ckpt_path,
            model,
            schema,
            outcome_values,
            args,
            gamma,
            lam,
            bc_thresh,
            reward_scale,
            epoch=args.epochs,
            selection_metric="final",
            selection_value=0.0,
        )
        print(f"saved {ckpt_path}")
    else:
        metric_name = "ope_wis" if select_by_ope else "neg_valid_loss"
        print(
            f"best epoch={best_epoch} {metric_name}={best_ope:.4f} "
            f"(left {ckpt_path} unchanged after last epoch)"
        )

    outcome_values.save(out_dir / "outcome_values.json")
    (out_dir / "schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2), encoding="utf-8"
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-logs", default=str(ROOT / ".." / "data" / "train_logs.jsonl"))
    p.add_argument("--valid-logs", default=str(ROOT / ".." / "data" / "valid_logs.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "models"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lambda-q", type=float, default=1.0)
    p.add_argument(
        "--reward-scale",
        type=float,
        default=100.0,
        help="Divide rewards for Q regression targets (act ranks by raw Q)",
    )
    p.add_argument(
        "--bc-threshold",
        type=float,
        default=0.01,
        help="BC support probability threshold at train/act time",
    )
    p.add_argument(
        "--action-conditioned-q",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score Q via MLP([h; a_emb]) per action; saves models/transformer_fqi_acq.pt (default: on)",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="Soft target-network update rate",
    )
    p.add_argument(
        "--select-by-ope",
        action="store_true",
        default=True,
        help="Select checkpoint by validation OPE WIS (default on)",
    )
    p.add_argument(
        "--no-select-by-ope",
        action="store_false",
        dest="select_by_ope",
        help="Select by validation loss instead of OPE WIS",
    )
    p.add_argument(
        "--ope-episodes",
        type=int,
        default=500,
        help="Valid episodes used for OPE checkpoint selection (0=all)",
    )
    p.add_argument("--ope-clip", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit-rows", type=int, default=0, help="0 = all rows")
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()

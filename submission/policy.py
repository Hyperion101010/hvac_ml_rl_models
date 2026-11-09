"""Self-contained HVAC follow-up act() policy (transformer FQI, action-conditioned Q)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
ACTIONS_SET = frozenset(ACTIONS)

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

RISK_ESCALATE = 0.75
PARK_IGNORED_COUNT = 2
RISK_SERVICE_ISSUE = 0.55
REL_CARE = 0.60
REL_TRACK_CARE = 0.50
URGENCY_OFFER = 0.65
URGENCY_EARLY_OFFER = 0.80

OBJECTION_SIGNALS = frozenset(
    {"price_objection", "competitor", "ask_spouse", "timing_delay"}
)
CARE_TOPICS = frozenset({"maintenance_checkin", "install_warranty", "weather_tip"})
MEM_TOPICS = frozenset({"membership_renewal", "plan_visit"})
DEAL_TOPICS = frozenset(
    {"open_estimate", "competitor_quote", "repair_vs_replace", "aging_equipment"}
)

_MODEL_PATH = Path(__file__).resolve().parent / "models" / "transformer_fqi_acq.pt"
_bundle: Optional[Dict[str, Any]] = None
_load_attempted = False
_fallback_warned = False
MODEL_LOADED = False

logger = logging.getLogger(__name__)


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
        cats.append(
            self.cat_id("weather_extreme", bool(observation.get("weather_extreme", False)))
        )
        nums = []
        for f in NUM_FIELDS:
            try:
                nums.append(float(observation.get(f, 0.0) or 0.0))
            except (TypeError, ValueError):
                nums.append(0.0)
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
        prev = list(observation.get("previous_actions") or [])
        for i, action in enumerate(prev[-(MAX_LEN - 1) :]):
            syn = dict(observation)
            syn["turn"] = i
            syn["previous_actions"] = prev[: i + 1]
            syn["last_action"] = action
            steps.append((syn, action))
        steps.append((observation, None))

    if len(steps) > MAX_LEN:
        steps = steps[-(MAX_LEN):]

    encoded = [schema.encode_step(obs, action) for obs, action in steps]
    valid = [True] * len(encoded)
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


def should_escalate(observation: Dict[str, Any]) -> bool:
    if observation.get("customer_signal") == "unhappy":
        return True
    try:
        return float(observation.get("service_risk_score", 0.0) or 0.0) > RISK_ESCALATE
    except (TypeError, ValueError):
        return False


def should_park(observation: Dict[str, Any]) -> bool:
    try:
        return int(observation.get("ignored_count", 0) or 0) >= PARK_IGNORED_COUNT
    except (TypeError, ValueError):
        return False


def hard_safety_action(observation: Dict[str, Any]) -> Optional[str]:
    if should_escalate(observation):
        return "ESCALATE_TO_HUMAN"
    if should_park(observation):
        return "PARK_THREAD"
    return None


def _is_ood_observation(observation: Dict[str, Any], schema: FeatureSchema) -> bool:
    for f in CAT_FIELDS:
        raw = observation.get(f)
        if raw is None:
            continue
        key = str(raw)
        table = schema.cat_to_id.get(f, {})
        if key not in table:
            return True
    return False


def product_rule(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    """README-aligned hand-coded policy. Extensible sibling to ML act() policies."""

    signal = observation["customer_signal"]
    risk = observation["service_risk_score"]
    ignored = observation["ignored_count"]
    patience = observation["user_patience"]
    touches = observation["touch_count"]
    estimate = observation["estimate_value_bucket"]
    topic = observation["topic"]
    membership = observation["membership_status"]
    rel = observation["relationship_score"]
    heat = observation["heat"]
    urgency = observation["urgency_score"]
    track = observation["track_hint"]
    prev = list(observation.get("previous_actions") or [])
    turn = observation["turn"]

    if signal == "unhappy" or risk > RISK_ESCALATE:
        return "ESCALATE_TO_HUMAN"
    if topic == "service_issue" and risk > RISK_SERVICE_ISSUE:
        return "ESCALATE_TO_HUMAN"

    if ignored >= 2:
        return "PARK_THREAD"
    if "OFFER_SCHEDULING" in prev and ignored >= 1:
        return "PARK_THREAD"
    if patience <= 0 and touches >= 1:
        return "PARK_THREAD" if ignored >= 1 else "WAIT"

    has_estimate = estimate != "none"
    care_mode = (
        topic in CARE_TOPICS
        or (rel > REL_CARE and not has_estimate)
        or (
            track == "relationship"
            and estimate in {"none", "low"}
            and rel > REL_TRACK_CARE
        )
    )

    if signal in OBJECTION_SIGNALS and (has_estimate or topic in DEAL_TOPICS):
        if prev.count("ASK_OBJECTION") >= 2:
            return "WAIT"
        return "ASK_OBJECTION"

    hot_open_estimate = has_estimate and heat == "hot" and topic in DEAL_TOPICS
    if topic in MEM_TOPICS or (
        membership in {"expiring", "lapsed"} and not hot_open_estimate
    ):
        return "MEMBERSHIP_TOUCH"

    if care_mode:
        if patience > 0 and ignored == 0 and touches < 2:
            return "CHECK_IN"
        return "WAIT"

    if has_estimate or topic in DEAL_TOPICS:
        offered = "OFFER_SCHEDULING" in prev
        if (
            signal == "positive"
            and heat in {"warm", "hot"}
            and urgency > URGENCY_OFFER
            and not offered
        ):
            return "OFFER_SCHEDULING"
        if turn == 0 and not (
            heat == "hot" and (signal == "positive" or urgency > URGENCY_EARLY_OFFER)
        ):
            return "ESTIMATE_NUDGE"
        if (
            heat == "hot"
            and touches >= 1
            and not offered
            and signal in {"positive", "none"}
        ):
            return "OFFER_SCHEDULING"
        return "ESTIMATE_NUDGE"

    return "WAIT"


StepEmbedding = None  # type: ignore
CausalTransformerPolicyNet = None  # type: ignore


def _ensure_module_bases() -> None:
    """Build nn.Module subclasses on first torch import (lazy)."""
    import torch
    from torch import nn

    global StepEmbedding, CausalTransformerPolicyNet
    if CausalTransformerPolicyNet is not None:
        return

    class _StepEmbedding(nn.Module):
        def __init__(self, schema: FeatureSchema, d_model: int = 128) -> None:
            super().__init__()
            self.schema = schema
            self.d_model = d_model
            self.cat_embs = nn.ModuleList(
                [nn.Embedding(len(schema.vocabs[f]), d_model) for f in CAT_FIELDS]
            )
            self.weather_emb = nn.Embedding(
                len(schema.vocabs["weather_extreme"]), d_model
            )
            self.action_emb = nn.Embedding(len(schema.vocabs["action"]), d_model)
            self.pos_emb = nn.Embedding(MAX_LEN, d_model)
            self.num_proj = nn.Linear(len(NUM_FIELDS), d_model)
            self.out = nn.Linear(d_model, d_model)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, cats, nums, actions, positions):
            x = self.action_emb(actions) + self.pos_emb(positions)
            for i, emb in enumerate(self.cat_embs):
                x = x + emb(cats[..., i])
            x = x + self.weather_emb(cats[..., len(CAT_FIELDS)])
            x = x + self.num_proj(nums)
            return self.norm(self.out(x))

    class _CausalTransformerPolicyNet(nn.Module):
        """Causal Transformer with outcome, Q, and BC heads (outcome kept for ckpt)."""

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
            self.embed = _StepEmbedding(schema, d_model=d_model)
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

        def _q_values(self, h_t):
            if not self.action_conditioned_q:
                return self.q_head(h_t)
            b = h_t.size(0)
            a_ids = torch.arange(self.n_actions, device=h_t.device)
            a_emb = self.q_action_emb(a_ids).unsqueeze(0).expand(b, -1, -1)
            h_exp = h_t.unsqueeze(1).expand(-1, self.n_actions, -1)
            return self.q_head(torch.cat([h_exp, a_emb], dim=-1)).squeeze(-1)

        def encode(self, cats, nums, actions, mask):
            b, length, _ = cats.shape
            positions = torch.arange(length, device=cats.device).unsqueeze(0).expand(
                b, -1
            )
            h = self.embed(cats, nums, actions, positions)
            causal = torch.triu(
                torch.ones(length, length, device=cats.device, dtype=torch.bool),
                diagonal=1,
            )
            key_pad = ~mask
            encoded = self.encoder(h, mask=causal, src_key_padding_mask=key_pad)
            lengths = mask.long().sum(dim=1).clamp(min=1)
            idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d_model)
            h_t = encoded.gather(1, idx).squeeze(1)
            h_t = torch.nan_to_num(h_t, nan=0.0)
            return h_t

        def forward(self, cats, nums, actions, mask) -> Dict[str, Any]:
            h_t = self.encode(cats, nums, actions, mask)
            outcome_logits = self.outcome_head(h_t).view(
                -1, self.n_actions, self.n_outcomes
            )
            q_values = self._q_values(h_t)
            bc_logits = self.bc_head(h_t)
            return {
                "h": h_t,
                "outcome_logits": outcome_logits,
                "q_values": q_values,
                "bc_logits": bc_logits,
            }

    StepEmbedding = _StepEmbedding
    CausalTransformerPolicyNet = _CausalTransformerPolicyNet


def bc_support_mask(bc_logits, threshold: float = 0.01, force_action_idx=None):
    import torch.nn.functional as F

    probs = F.softmax(bc_logits, dim=-1)
    support = probs >= threshold
    argmax = probs.argmax(dim=-1)
    support.scatter_(1, argmax.unsqueeze(-1), True)
    if force_action_idx is not None:
        support[:, int(force_action_idx)] = True
    return support


def masked_argmax(scores, support):
    import torch

    very_neg = torch.finfo(scores.dtype).min / 4
    masked = scores.masked_fill(~support, very_neg)
    return masked.argmax(dim=-1)


def _warn_fallback_once(reason: str) -> None:
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    msg = f"submission policy falling back to product_rule ({reason})"
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    logger.warning(msg)


def _load_bundle() -> Optional[Dict[str, Any]]:
    global _bundle, _load_attempted, MODEL_LOADED
    if _bundle is not None:
        return _bundle
    if _load_attempted and not MODEL_LOADED:
        return None
    _load_attempted = True
    try:
        import torch

        _ensure_module_bases()
        path = _MODEL_PATH
        if not path.exists():
            _warn_fallback_once(f"missing checkpoint {path.name}")
            MODEL_LOADED = False
            return None
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        schema = FeatureSchema.from_dict(ckpt["schema"])
        cfg = ckpt["config"]
        model = CausalTransformerPolicyNet(
            schema,
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
            action_conditioned_q=bool(cfg.get("action_conditioned_q", False)),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _bundle = {
            "model": model,
            "schema": schema,
            "bc_threshold": float(cfg.get("bc_threshold", 0.01)),
            "action_conditioned_q": bool(cfg.get("action_conditioned_q", False)),
        }
        MODEL_LOADED = True
        return _bundle
    except Exception as exc:
        _warn_fallback_once(f"load failed: {type(exc).__name__}: {exc}")
        MODEL_LOADED = False
        _bundle = None
        return None


def _model_act(
    observation: Dict[str, Any], history: List[Dict[str, Any]], bundle: Dict[str, Any]
) -> str:
    import torch

    if _is_ood_observation(observation, bundle["schema"]):
        return _safe_product_rule(observation, history)
    steps, valid = build_sequence(observation, history, bundle["schema"])
    batch = collate_sequences([(steps, valid)])
    model = bundle["model"]
    with torch.no_grad():
        out = model(batch["cats"], batch["nums"], batch["actions"], batch["mask"])
    force = ACTIONS.index("PARK_THREAD") if should_park(observation) else None
    support = bc_support_mask(
        out["bc_logits"],
        threshold=bundle["bc_threshold"],
        force_action_idx=force,
    )
    idx = int(masked_argmax(out["q_values"], support)[0].item())
    return ACTIONS[idx]


def _safe_product_rule(
    observation: Dict[str, Any], history: List[Dict[str, Any]]
) -> str:
    try:
        action = product_rule(observation, history)
        if action in ACTIONS_SET:
            return action
    except Exception:
        pass
    return "WAIT"


def act(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    try:
        if not isinstance(observation, dict):
            return "WAIT"
        obs = dict(observation)
        hist: List[Dict[str, Any]] = list(history) if history else []

        hard = hard_safety_action(obs)
        if hard is not None:
            return hard

        bundle = _load_bundle()
        if bundle is None:
            return _safe_product_rule(obs, hist)

        try:
            action = _model_act(obs, hist, bundle)
            if action in ACTIONS_SET:
                return action
            _warn_fallback_once("invalid model action")
            return _safe_product_rule(obs, hist)
        except Exception as exc:
            _warn_fallback_once(f"inference failed: {type(exc).__name__}: {exc}")
            return _safe_product_rule(obs, hist)
    except Exception:
        return "WAIT"

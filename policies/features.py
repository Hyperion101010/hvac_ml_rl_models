"""Step tokenization and history sequence building for the causal transformer.

Token time order: [hist_0, ..., hist_{t-1}, current_obs].
All robustness lives here / in act — do not change the act signature.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from policies.base import ACTIONS

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

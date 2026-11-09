"""One-step greedy baseline: argmax of gamma=0 FQE / one-step Q."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from policies.base import ACTIONS
from policies.registry import register
from policies.safety import hard_safety_action
from policies.transformer_model import bc_support_mask, masked_argmax
from policies.transformer_policy import (
    _forward,
    _product_rule_fallback,
    _try_load,
)

_ROOT = Path(__file__).resolve().parents[1]
_ONE_STEP = _ROOT / "models" / "transformer_one_step.pt"
_FQE = _ROOT / "models" / "transformer_fqe.pt"


def one_step_greedy_act(
    observation: Dict[str, Any], history: List[Dict[str, Any]]
) -> str:
    hard = hard_safety_action(observation)
    if hard is not None:
        return hard
    bundle = _try_load(_ONE_STEP) or _try_load(_FQE)
    if bundle is None:
        return _product_rule_fallback(observation, history)
    out = _forward(observation, history, bundle)
    thresh = float(bundle.get("bc_threshold", 0.01))
    support = bc_support_mask(out["bc_logits"], threshold=thresh)
    idx = int(masked_argmax(out["q_values"], support)[0].item())
    return ACTIONS[idx]


@register("one_step_greedy")
def one_step_greedy(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return one_step_greedy_act(observation, history)

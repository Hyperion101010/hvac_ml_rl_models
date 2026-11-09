"""Pluggable act() for transformer FQI / BC / AWBC — signature fixed."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from policies.base import ACTIONS
from policies.features import FeatureSchema, build_sequence, collate_sequences
from policies.outcome_values import OutcomeValues
from policies.registry import register
from policies.safety import RISK_ESCALATE, hard_safety_action, should_park
from policies.transformer_model import (
    CausalTransformerAWBCNet,
    CausalTransformerPolicyNet,
    bc_support_mask,
    masked_argmax,
)

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CKPT = _ROOT / "models" / "transformer_fqi.pt"
_ACQ_CKPT = _ROOT / "models" / "transformer_fqi_acq.pt"
_AWBC_CKPT = _ROOT / "models" / "transformer_awbc.pt"
_PARK_IDX = ACTIONS.index("PARK_THREAD")

_bundles: Dict[str, Dict[str, Any]] = {}


def _torch():
    import torch

    return torch


def _product_rule_fallback(
    observation: Dict[str, Any], history: List[Dict[str, Any]]
) -> str:
    from policies.rules import product_rule

    return product_rule(observation, history)


def _is_ood_observation(observation: Dict[str, Any], schema: FeatureSchema) -> bool:
    """True if a categorical field is outside the training vocab (maps to id 0 unk)."""
    from policies.features import CAT_FIELDS

    for f in CAT_FIELDS:
        raw = observation.get(f)
        if raw is None:
            continue
        key = str(raw)
        table = schema.cat_to_id.get(f, {})
        if key not in table:
            return True
    return False


def load_bundle(ckpt_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(ckpt_path) if ckpt_path else _DEFAULT_CKPT
    key = str(path)
    if key in _bundles:
        return _bundles[key]
    torch = _torch()
    if not path.exists():
        hint = "python -m train.train_transformer_fqi"
        if path.name.endswith("_acq.pt"):
            hint += " --action-conditioned-q"
        elif path.name.endswith("_awbc.pt"):
            hint = "python -m train.train_transformer_awbc"
        raise FileNotFoundError(f"Missing checkpoint {path}. Train with: {hint}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    schema = FeatureSchema.from_dict(ckpt["schema"])
    cfg = ckpt["config"]
    kind = cfg.get("kind", "fqi")
    if kind == "awbc":
        model: Any = CausalTransformerAWBCNet(
            schema,
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        bundle = {
            "path": key,
            "kind": "awbc",
            "model": model,
            "schema": schema,
            "reward_scale": float(cfg.get("reward_scale", 1.0)),
            "device": torch.device("cpu"),
        }
    else:
        # fqi / fqe / one_step share CausalTransformerPolicyNet
        model = CausalTransformerPolicyNet(
            schema,
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
            action_conditioned_q=bool(cfg.get("action_conditioned_q", False)),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        ov_raw = ckpt.get("outcome_values")
        outcome_values = (
            OutcomeValues.from_dict(ov_raw) if ov_raw is not None else None
        )
        bundle = {
            "path": key,
            "kind": kind if kind in ("fqi", "fqe") else "fqi",
            "model": model,
            "schema": schema,
            "outcome_values": outcome_values,
            "bc_threshold": float(cfg.get("bc_threshold", 0.01)),
            "reward_scale": float(cfg.get("reward_scale", 1.0)),
            "action_conditioned_q": bool(cfg.get("action_conditioned_q", False)),
            "device": torch.device("cpu"),
        }
    _bundles[key] = bundle
    return bundle


def _try_load(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    try:
        return load_bundle(ckpt_path)
    except Exception:
        return None


def _forward(observation: Dict[str, Any], history: List[Dict[str, Any]], bundle: Dict[str, Any]):
    steps, valid = build_sequence(observation, history, bundle["schema"])
    batch = collate_sequences([(steps, valid)])
    model = bundle["model"]
    with _torch().no_grad():
        out = model(
            batch["cats"],
            batch["nums"],
            batch["actions"],
            batch["mask"],
        )
    return out


def _fqi_act_from_bundle(
    observation: Dict[str, Any], history: List[Dict[str, Any]], bundle: Dict[str, Any]
) -> str:
    hard = hard_safety_action(observation)
    if hard is not None:
        return hard
    if _is_ood_observation(observation, bundle["schema"]):
        return _product_rule_fallback(observation, history)
    out = _forward(observation, history, bundle)
    force = _PARK_IDX if should_park(observation) else None
    support = bc_support_mask(
        out["bc_logits"],
        threshold=bundle["bc_threshold"],
        force_action_idx=force,
    )
    idx = int(masked_argmax(out["q_values"], support)[0].item())
    return ACTIONS[idx]


def transformer_fqi_act(
    observation: Dict[str, Any], history: List[Dict[str, Any]]
) -> str:
    bundle = _try_load(_DEFAULT_CKPT)
    if bundle is None:
        return _product_rule_fallback(observation, history)
    return _fqi_act_from_bundle(observation, history, bundle)


def transformer_fqi_acq_act(
    observation: Dict[str, Any], history: List[Dict[str, Any]]
) -> str:
    bundle = _try_load(_ACQ_CKPT)
    if bundle is None:
        return _product_rule_fallback(observation, history)
    return _fqi_act_from_bundle(observation, history, bundle)


def transformer_bc_act(
    observation: Dict[str, Any], history: List[Dict[str, Any]]
) -> str:
    hard = hard_safety_action(observation)
    if hard is not None:
        return hard
    bundle = _try_load(_DEFAULT_CKPT)
    if bundle is None:
        return _product_rule_fallback(observation, history)
    if _is_ood_observation(observation, bundle["schema"]):
        return _product_rule_fallback(observation, history)
    out = _forward(observation, history, bundle)
    idx = int(out["bc_logits"][0].argmax().item())
    return ACTIONS[idx]


def transformer_awbc_act(
    observation: Dict[str, Any], history: List[Dict[str, Any]]
) -> str:
    hard = hard_safety_action(observation)
    if hard is not None:
        return hard
    bundle = _try_load(_AWBC_CKPT)
    if bundle is None:
        return _product_rule_fallback(observation, history)
    if _is_ood_observation(observation, bundle["schema"]):
        return _product_rule_fallback(observation, history)
    out = _forward(observation, history, bundle)
    idx = int(out["pi_logits"][0].argmax().item())
    return ACTIONS[idx]


@register("transformer_fqi")
def transformer_fqi(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return transformer_fqi_act(observation, history)


@register("transformer_fqi_acq")
def transformer_fqi_acq(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return transformer_fqi_acq_act(observation, history)


@register("transformer_bc")
def transformer_bc(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return transformer_bc_act(observation, history)


@register("transformer_awbc")
def transformer_awbc(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return transformer_awbc_act(observation, history)


_ = RISK_ESCALATE

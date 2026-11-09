"""Pluggable act() for transformer FQI / BC — signature fixed."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from policies.base import ACTIONS
from policies.features import FeatureSchema, build_sequence, collate_sequences
from policies.outcome_values import OutcomeValues
from policies.registry import register
from policies.transformer_model import (
    RISK_ESCALATE,
    CausalTransformerPolicyNet,
    bc_support_mask,
    masked_argmax,
    should_escalate,
)

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CKPT = _ROOT / "models" / "transformer_fqi.pt"

_bundle: Optional[Dict[str, Any]] = None


def _torch():
    import torch

    return torch


def load_bundle(ckpt_path: Optional[Path] = None) -> Dict[str, Any]:
    global _bundle
    path = Path(ckpt_path) if ckpt_path else _DEFAULT_CKPT
    if _bundle is not None and _bundle.get("path") == str(path):
        return _bundle
    torch = _torch()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {path}. Train with: python -m train.train_transformer_fqi"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    schema = FeatureSchema.from_dict(ckpt["schema"])
    cfg = ckpt["config"]
    model = CausalTransformerPolicyNet(
        schema,
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 2),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    outcome_values = OutcomeValues.from_dict(ckpt["outcome_values"])
    _bundle = {
        "path": str(path),
        "model": model,
        "schema": schema,
        "outcome_values": outcome_values,
        "bc_threshold": float(cfg.get("bc_threshold", 0.05)),
        "reward_scale": float(cfg.get("reward_scale", 1.0)),
        "device": torch.device("cpu"),
    }
    return _bundle


def _forward(observation: Dict[str, Any], history: List[Dict[str, Any]], bundle: Dict[str, Any]):
    torch = _torch()
    steps, valid = build_sequence(observation, history, bundle["schema"])
    batch = collate_sequences([(steps, valid)])
    model: CausalTransformerPolicyNet = bundle["model"]
    with torch.no_grad():
        out = model(
            batch["cats"],
            batch["nums"],
            batch["actions"],
            batch["mask"],
        )
    return out


def transformer_fqi_act(
    observation: Dict[str, Any], history: List[Dict[str, Any]]
) -> str:
    if should_escalate(observation):
        return "ESCALATE_TO_HUMAN"
    bundle = load_bundle()
    out = _forward(observation, history, bundle)
    support = bc_support_mask(out["bc_logits"], threshold=bundle["bc_threshold"])
    idx = int(masked_argmax(out["q_values"], support)[0].item())
    return ACTIONS[idx]


def transformer_bc_act(
    observation: Dict[str, Any], history: List[Dict[str, Any]]
) -> str:
    if should_escalate(observation):
        return "ESCALATE_TO_HUMAN"
    out = _forward(observation, history, load_bundle())
    idx = int(out["bc_logits"][0].argmax().item())
    return ACTIONS[idx]


@register("transformer_fqi")
def transformer_fqi(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return transformer_fqi_act(observation, history)


@register("transformer_bc")
def transformer_bc(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return transformer_bc_act(observation, history)


# Silence unused import warning for RISK_ESCALATE documentation linkage
_ = RISK_ESCALATE

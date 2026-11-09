"""Shared hard safety rules (torch-free)."""

from __future__ import annotations

from typing import Any, Dict

RISK_ESCALATE = 0.75


def should_escalate(observation: Dict[str, Any]) -> bool:
    if observation.get("customer_signal") == "unhappy":
        return True
    try:
        return float(observation.get("service_risk_score", 0.0) or 0.0) > RISK_ESCALATE
    except (TypeError, ValueError):
        return False

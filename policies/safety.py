"""Shared hard safety rules (torch-free)."""

from __future__ import annotations

from typing import Any, Dict, Optional

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

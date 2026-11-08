"""Baseline policies for the HVAC follow-up take-home."""

from __future__ import annotations

from typing import Any, Dict, List


ACTIONS = [
    "WAIT",
    "CHECK_IN",
    "ESTIMATE_NUDGE",
    "ASK_OBJECTION",
    "OFFER_SCHEDULING",
    "MEMBERSHIP_TOUCH",
    "PARK_THREAD",
    "ESCALATE_TO_HUMAN",
]


def always_estimate_nudge(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return "ESTIMATE_NUDGE"


def always_check_in(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return "CHECK_IN"


def always_escalate(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return "ESCALATE_TO_HUMAN"


def simple_rule(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    """Intentionally decent but beatable hand-coded policy."""

    risk = observation["service_risk_score"]
    signal = observation["customer_signal"]
    track = observation["track_hint"]
    membership = observation["membership_status"]
    estimate = observation["estimate_value_bucket"]
    ignored = observation["ignored_count"]
    touches = observation["touch_count"]
    patience = observation["user_patience"]
    urgency = observation["urgency_score"]
    relationship = observation["relationship_score"]

    if signal == "unhappy" or risk > 0.78:
        return "ESCALATE_TO_HUMAN"
    if ignored >= 2:
        return "PARK_THREAD"
    if touches >= 4:
        return "WAIT"
    if track == "relationship":
        if membership in {"expiring", "lapsed"}:
            return "MEMBERSHIP_TOUCH"
        if relationship > 0.55:
            return "CHECK_IN"
    if estimate != "none":
        if signal in {"price_objection", "competitor", "ask_spouse", "timing_delay"} and patience > 0:
            return "ASK_OBJECTION"
        if signal == "positive" or urgency > 0.70:
            return "OFFER_SCHEDULING"
        return "ESTIMATE_NUDGE"
    if membership in {"active", "expiring", "lapsed"}:
        return "MEMBERSHIP_TOUCH"
    return "WAIT"


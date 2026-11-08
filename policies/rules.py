"""Hard-coded product rule and starter baseline re-exports."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from policies.base import ACTIONS
from policies.registry import register

_STARTER = Path(__file__).resolve().parents[1] / "starter"
if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))

from baselines import (  # noqa: E402
    always_check_in,
    always_escalate,
    always_estimate_nudge,
    simple_rule,
)

RISK_ESCALATE = 0.75
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


@register("product_rule")
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


register("simple_rule", simple_rule)
register("always_check_in", always_check_in)
register("always_estimate_nudge", always_estimate_nudge)
register("always_escalate", always_escalate)

__all__ = [
    "ACTIONS",
    "product_rule",
    "simple_rule",
    "always_check_in",
    "always_estimate_nudge",
    "always_escalate",
]

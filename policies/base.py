"""Shared policy types and action space."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

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

PolicyFn = Callable[[Dict[str, Any], List[Dict[str, Any]]], str]

"""Starter policy.

Candidates should replace this file. The act() function must return one of:

WAIT, CHECK_IN, ESTIMATE_NUDGE, ASK_OBJECTION, OFFER_SCHEDULING,
MEMBERSHIP_TOUCH, PARK_THREAD, ESCALATE_TO_HUMAN.
"""

from __future__ import annotations

from typing import Any, Dict, List

from baselines import simple_rule


def act(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return simple_rule(observation, history)


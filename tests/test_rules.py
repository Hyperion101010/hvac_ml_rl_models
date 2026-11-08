"""Unit tests for product_rule decision branches."""

from __future__ import annotations

import unittest
from typing import Any, Dict

from policies.rules import product_rule


def _obs(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "case_id": "test",
        "turn": 0,
        "day": 1,
        "topic": "open_estimate",
        "equipment_type": "furnace",
        "season": "summer",
        "weather_extreme": False,
        "track_hint": "nurture",
        "heat": "warm",
        "estimate_value_bucket": "medium",
        "membership_status": "none",
        "customer_signal": "none",
        "ignored_count": 0,
        "touch_count": 0,
        "user_patience": 2,
        "urgency_score": 0.4,
        "price_sensitivity_score": 0.5,
        "service_risk_score": 0.1,
        "relationship_score": 0.4,
        "previous_actions": [],
        "last_action": None,
    }
    base.update(overrides)
    return base


class ProductRuleTests(unittest.TestCase):
    def test_unhappy_escalates(self):
        action = product_rule(_obs(customer_signal="unhappy"), [])
        self.assertEqual(action, "ESCALATE_TO_HUMAN")

    def test_high_risk_escalates(self):
        action = product_rule(_obs(service_risk_score=0.9), [])
        self.assertEqual(action, "ESCALATE_TO_HUMAN")

    def test_ignored_parks(self):
        action = product_rule(_obs(ignored_count=2, touch_count=2), [])
        self.assertEqual(action, "PARK_THREAD")

    def test_no_reoffer_after_ignore(self):
        action = product_rule(
            _obs(
                previous_actions=["ESTIMATE_NUDGE", "OFFER_SCHEDULING"],
                ignored_count=1,
                touch_count=2,
                customer_signal="none",
                heat="hot",
                urgency_score=0.9,
            ),
            [],
        )
        self.assertEqual(action, "PARK_THREAD")

    def test_objection_asks(self):
        action = product_rule(
            _obs(customer_signal="price_objection", estimate_value_bucket="low"),
            [],
        )
        self.assertEqual(action, "ASK_OBJECTION")

    def test_warranty_check_in(self):
        action = product_rule(
            _obs(
                topic="install_warranty",
                estimate_value_bucket="none",
                track_hint="deal",
                relationship_score=0.75,
                customer_signal="none",
            ),
            [],
        )
        self.assertEqual(action, "CHECK_IN")

    def test_relationship_maintenance_does_not_nudge(self):
        action = product_rule(
            _obs(
                topic="maintenance_checkin",
                track_hint="relationship",
                estimate_value_bucket="low",
                relationship_score=0.86,
                customer_signal="positive",
                membership_status="active",
            ),
            [],
        )
        self.assertEqual(action, "CHECK_IN")

    def test_patience_zero_waits_instead_of_selling(self):
        action = product_rule(
            _obs(
                user_patience=0,
                touch_count=1,
                ignored_count=0,
                customer_signal="timing_delay",
                estimate_value_bucket="medium",
                previous_actions=["OFFER_SCHEDULING"],
            ),
            [],
        )
        self.assertEqual(action, "WAIT")

    def test_membership_touch(self):
        action = product_rule(
            _obs(
                topic="membership_renewal",
                estimate_value_bucket="none",
                membership_status="expiring",
                relationship_score=0.3,
            ),
            [],
        )
        self.assertEqual(action, "MEMBERSHIP_TOUCH")

    def test_deal_nudge_default(self):
        action = product_rule(
            _obs(
                topic="open_estimate",
                estimate_value_bucket="medium",
                heat="warm",
                turn=0,
                urgency_score=0.3,
                customer_signal="none",
            ),
            [],
        )
        self.assertEqual(action, "ESTIMATE_NUDGE")


if __name__ == "__main__":
    unittest.main()

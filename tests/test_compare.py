"""Tests for offline compare metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval.compare_policies import (
    evaluate_policy,
    group_episodes,
    load_rows,
    pairwise_disagreement,
)
from policies.base import ACTIONS
from policies.rules import always_check_in, product_rule


FIXTURE = """\
{"episode_id": "ep0", "step": 0, "action": "CHECK_IN", "action_prob": 0.5, "reward": 3.0, "done": false, "observation": {"turn": 0, "day": 0, "topic": "install_warranty", "equipment_type": "central_ac", "season": "fall", "weather_extreme": false, "track_hint": "nurture", "heat": "cold", "estimate_value_bucket": "none", "membership_status": "none", "customer_signal": "none", "ignored_count": 0, "touch_count": 0, "user_patience": 2, "urgency_score": 0.3, "price_sensitivity_score": 0.3, "service_risk_score": 0.1, "relationship_score": 0.8, "previous_actions": [], "last_action": null, "case_id": "ep0", "days_since_last_inbound": 99, "days_since_last_outbound": 99}, "next_observation": {}, "info": {"outcome": "continue", "cost_cents": 0.08, "invalid_action": false, "latency_ms": 1.0}}
{"episode_id": "ep0", "step": 1, "action": "ESTIMATE_NUDGE", "action_prob": 0.4, "reward": -9.0, "done": false, "observation": {"turn": 1, "day": 1, "topic": "install_warranty", "equipment_type": "central_ac", "season": "fall", "weather_extreme": false, "track_hint": "nurture", "heat": "hot", "estimate_value_bucket": "none", "membership_status": "none", "customer_signal": "positive", "ignored_count": 0, "touch_count": 1, "user_patience": 1, "urgency_score": 0.4, "price_sensitivity_score": 0.3, "service_risk_score": 0.1, "relationship_score": 0.7, "previous_actions": ["CHECK_IN"], "last_action": "CHECK_IN", "case_id": "ep0", "days_since_last_inbound": 0, "days_since_last_outbound": 0}, "next_observation": {}, "info": {"outcome": "continue", "cost_cents": 0.08, "invalid_action": false, "latency_ms": 1.0}}
{"episode_id": "ep0", "step": 2, "action": "ESTIMATE_NUDGE", "action_prob": 0.3, "reward": -66.0, "done": true, "observation": {"turn": 2, "day": 2, "topic": "install_warranty", "equipment_type": "central_ac", "season": "fall", "weather_extreme": false, "track_hint": "nurture", "heat": "warm", "estimate_value_bucket": "none", "membership_status": "none", "customer_signal": "none", "ignored_count": 2, "touch_count": 2, "user_patience": 1, "urgency_score": 0.3, "price_sensitivity_score": 0.3, "service_risk_score": 0.1, "relationship_score": 0.6, "previous_actions": ["CHECK_IN", "ESTIMATE_NUDGE"], "last_action": "ESTIMATE_NUDGE", "case_id": "ep0", "days_since_last_inbound": 1, "days_since_last_outbound": 0}, "next_observation": null, "info": {"outcome": "opted_out", "cost_cents": 0.08, "invalid_action": false, "latency_ms": 1.0}}
{"episode_id": "ep1", "step": 0, "action": "CHECK_IN", "action_prob": 0.5, "reward": 2.0, "done": true, "observation": {"turn": 0, "day": 0, "topic": "maintenance_checkin", "equipment_type": "furnace", "season": "summer", "weather_extreme": false, "track_hint": "relationship", "heat": "warm", "estimate_value_bucket": "none", "membership_status": "none", "customer_signal": "none", "ignored_count": 0, "touch_count": 0, "user_patience": 2, "urgency_score": 0.2, "price_sensitivity_score": 0.4, "service_risk_score": 0.1, "relationship_score": 0.85, "previous_actions": [], "last_action": null, "case_id": "ep1", "days_since_last_inbound": 99, "days_since_last_outbound": 99}, "next_observation": null, "info": {"outcome": "max_steps", "cost_cents": 0.08, "invalid_action": false, "latency_ms": 1.0}}
"""


class CompareMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        self.tmp.write(FIXTURE)
        self.tmp.close()
        self.path = Path(self.tmp.name)
        self.rows = load_rows(self.path, limit=0)
        self.episodes = group_episodes(self.rows)

    def tearDown(self) -> None:
        self.path.unlink(missing_ok=True)

    def test_load_and_group(self):
        self.assertEqual(len(self.rows), 4)
        self.assertEqual(len(self.episodes), 2)
        self.assertEqual([s["step"] for s in self.episodes[0]], [0, 1, 2])

    def test_evaluate_keys_and_agreement(self):
        result = evaluate_policy("product_rule", product_rule, self.episodes)
        for key in (
            "name",
            "n",
            "action_counts",
            "agreement_rate",
            "matched_n",
            "matched_log_mean_reward",
            "actions_chosen",
            "slice_agree",
            "slice_total",
        ):
            self.assertIn(key, result)
        self.assertEqual(result["n"], 4)
        self.assertEqual(sum(result["action_counts"].values()), 4)
        for action in ACTIONS:
            self.assertIn(action, result["action_counts"])
        # step0 CHECK_IN agree, step1 care WAIT/CHECK_IN vs nudge disagree,
        # step2 PARK vs nudge disagree, step3 CHECK_IN agree -> 2/4
        self.assertEqual(result["matched_n"], 2)
        self.assertAlmostEqual(result["agreement_rate"], 0.5)
        self.assertAlmostEqual(
            result["matched_log_mean_reward"], (3.0 + 2.0) / 2
        )
        if result["matched_n"]:
            # matched rewards only from agreeing steps
            self.assertIsNotNone(result["matched_log_mean_reward"])

    def test_pairwise_disagreement(self):
        a = evaluate_policy("product_rule", product_rule, self.episodes)
        b = evaluate_policy("always_check_in", always_check_in, self.episodes)
        pairs = pairwise_disagreement([a, b])
        self.assertEqual(len(pairs), 1)
        left, right, rate = pairs[0]
        self.assertEqual(left, "product_rule")
        self.assertEqual(right, "always_check_in")
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)


if __name__ == "__main__":
    unittest.main()

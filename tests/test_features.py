"""History sequence robustness tests."""

from __future__ import annotations

import unittest

from policies.features import (
    MAX_LEN,
    FeatureSchema,
    build_sequence,
    filter_history,
)


def _obs(case_id: str, turn: int = 0, **kw):
    base = {
        "case_id": case_id,
        "turn": turn,
        "day": turn,
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
        "touch_count": turn,
        "user_patience": 2,
        "urgency_score": 0.4,
        "price_sensitivity_score": 0.5,
        "service_risk_score": 0.1,
        "relationship_score": 0.4,
        "previous_actions": [],
        "last_action": None,
    }
    base.update(kw)
    return base


class FeatureSequenceTests(unittest.TestCase):
    def test_empty_history_length_one_valid(self):
        steps, valid = build_sequence(_obs("c1"), [])
        self.assertEqual(len(steps), MAX_LEN)
        self.assertEqual(sum(valid), 1)
        self.assertTrue(valid[0])
        self.assertFalse(valid[-1])

    def test_filters_other_case_ids(self):
        hist = [
            {"observation": _obs("other", turn=0), "action": "CHECK_IN"},
            {"observation": _obs("c1", turn=0), "action": "WAIT"},
        ]
        filtered = filter_history(_obs("c1", turn=1, previous_actions=["WAIT"]), hist)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["action"], "WAIT")

    def test_matched_history_builds_prefix(self):
        cur = _obs("c1", turn=1, previous_actions=["CHECK_IN"])
        hist = [{"observation": _obs("c1", turn=0), "action": "CHECK_IN"}]
        steps, valid = build_sequence(cur, hist)
        self.assertEqual(sum(valid), 2)
        self.assertTrue(valid[0])
        self.assertTrue(valid[1])
        self.assertFalse(valid[2])

    def test_mismatch_falls_back_to_previous_actions(self):
        cur = _obs("c1", turn=2, previous_actions=["WAIT", "CHECK_IN"])
        # polluted / disagreeing actions
        hist = [
            {"observation": _obs("c1", turn=0), "action": "ESTIMATE_NUDGE"},
            {"observation": _obs("other", turn=1), "action": "ASK_OBJECTION"},
        ]
        steps, valid = build_sequence(cur, hist)
        # 2 prev actions + current = 3 valid
        self.assertEqual(sum(valid), 3)

    def test_schema_encode_pad_action(self):
        schema = FeatureSchema()
        enc = schema.encode_step(_obs("c1"), None)
        self.assertEqual(enc["action_id"], schema.pad_action_id)


if __name__ == "__main__":
    unittest.main()

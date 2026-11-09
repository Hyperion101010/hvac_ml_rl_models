"""Model shape and policy mask tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from policies.base import ACTIONS
from policies.features import FeatureSchema, build_sequence, collate_sequences
from policies.transformer_model import (
    CausalTransformerPolicyNet,
    bc_support_mask,
    masked_argmax,
    should_escalate,
)
from policies.transformer_policy import transformer_fqi_act, transformer_bc_act, load_bundle


def _obs(**kw):
    base = {
        "case_id": "c1",
        "turn": 0,
        "day": 0,
        "topic": "open_estimate",
        "equipment_type": "furnace",
        "season": "summer",
        "weather_extreme": False,
        "track_hint": "deal",
        "heat": "hot",
        "estimate_value_bucket": "high",
        "membership_status": "none",
        "customer_signal": "none",
        "ignored_count": 0,
        "touch_count": 0,
        "user_patience": 2,
        "urgency_score": 0.5,
        "price_sensitivity_score": 0.5,
        "service_risk_score": 0.1,
        "relationship_score": 0.3,
        "previous_actions": [],
        "last_action": None,
    }
    base.update(kw)
    return base


class ModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        schema = FeatureSchema()
        model = CausalTransformerPolicyNet(schema, d_model=64, n_heads=4, n_layers=2)
        steps, valid = build_sequence(_obs(), [])
        batch = collate_sequences([(steps, valid)])
        out = model(batch["cats"], batch["nums"], batch["actions"], batch["mask"])
        self.assertEqual(tuple(out["outcome_logits"].shape), (1, 8, 9))
        self.assertEqual(tuple(out["q_values"].shape), (1, 8))
        self.assertEqual(tuple(out["bc_logits"].shape), (1, 8))

    def test_bc_support_and_argmax(self):
        logits = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        support = bc_support_mask(logits, threshold=0.05)
        self.assertTrue(bool(support[0, 1].item()))
        scores = torch.randn(1, 8)
        idx = masked_argmax(scores, support)
        self.assertEqual(int(idx.item()), int(scores[0].masked_fill(~support[0], -1e9).argmax()))

    def test_escalate_only_safety(self):
        self.assertTrue(should_escalate(_obs(customer_signal="unhappy")))
        self.assertTrue(should_escalate(_obs(service_risk_score=0.9)))
        self.assertFalse(should_escalate(_obs(ignored_count=2, service_risk_score=0.1)))


@unittest.skipUnless(
    (Path(__file__).resolve().parents[1] / "models" / "transformer_fqi.pt").exists(),
    "checkpoint not trained yet",
)
class PolicyCheckpointTests(unittest.TestCase):
    def test_act_returns_valid_action(self):
        load_bundle()
        a = transformer_fqi_act(_obs(), [])
        self.assertIn(a, ACTIONS)
        b = transformer_bc_act(_obs(), [])
        self.assertIn(b, ACTIONS)

    def test_unhappy_forces_escalate(self):
        a = transformer_fqi_act(_obs(customer_signal="unhappy"), [])
        self.assertEqual(a, "ESCALATE_TO_HUMAN")


if __name__ == "__main__":
    unittest.main()

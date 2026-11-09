"""Model shape and policy mask tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from policies.base import ACTIONS
from policies.features import FeatureSchema, build_sequence, collate_sequences
from policies.outcome_values import OutcomeValues, compute_outcome_values
from policies.transformer_model import (
    CausalTransformerPolicyNet,
    bc_support_mask,
    factored_scores,
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

    def test_factored_scores_shape(self):
        ov = OutcomeValues(
            values={
                "sold_estimate": 100.0,
                "booked_appointment": 30.0,
                "membership_converted": 18.0,
                "positive_review": 10.0,
                "opted_out": -60.0,
                "parked": -5.0,
            },
            escalate_unhappy=20.0,
            escalate_other=-18.0,
        )
        logits = torch.zeros(2, 8, 9)
        q = torch.zeros(2, 8)
        unhappy = torch.tensor([False, True])
        scores = factored_scores(logits, q, ov, unhappy)
        self.assertEqual(tuple(scores.shape), (2, 8))

    def test_factored_scores_reward_scale_on_q_only(self):
        ov = OutcomeValues(
            values={
                "sold_estimate": 0.0,
                "booked_appointment": 0.0,
                "membership_converted": 0.0,
                "positive_review": 0.0,
                "opted_out": 0.0,
                "parked": 0.0,
            },
            escalate_unhappy=0.0,
            escalate_other=0.0,
        )
        # Uniform over outcomes except put all mass on continue for action 0 via huge logit
        from policies.outcome_values import CONTINUE_IDX

        logits = torch.full((1, 8, 9), -20.0)
        logits[0, :, CONTINUE_IDX] = 20.0
        q = torch.ones(1, 8)
        unhappy = torch.tensor([False])
        s1 = factored_scores(logits, q, ov, unhappy, reward_scale=1.0)
        s100 = factored_scores(logits, q, ov, unhappy, reward_scale=100.0)
        # V_imm ~ 0; P(continue)~1 => score ~= Q * scale
        self.assertTrue(torch.allclose(s1, torch.ones(1, 8), atol=1e-3))
        self.assertTrue(torch.allclose(s100, torch.full((1, 8), 100.0), atol=1e-2))
        q2 = torch.full((1, 8), 2.0)
        s200 = factored_scores(logits, q2, ov, unhappy, reward_scale=100.0)
        self.assertTrue(torch.allclose(s200, 2.0 * s100, atol=1e-2))


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

"""AWBC return-to-go helpers and model smoke tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from policies.base import ACTIONS
from policies.features import FeatureSchema, build_sequence, collate_sequences
from policies.transformer_model import CausalTransformerAWBCNet
from train.train_transformer_awbc import returns_to_go


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


class ReturnsToGoTests(unittest.TestCase):
    def test_single_step(self):
        self.assertEqual(returns_to_go([5.0], 0.95), [5.0])

    def test_two_steps(self):
        g = returns_to_go([1.0, 2.0], 0.5)
        self.assertAlmostEqual(g[1], 2.0)
        self.assertAlmostEqual(g[0], 1.0 + 0.5 * 2.0)


class AWBCModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        schema = FeatureSchema()
        model = CausalTransformerAWBCNet(schema, d_model=64, n_heads=4, n_layers=2)
        steps, valid = build_sequence(_obs(), [])
        batch = collate_sequences([(steps, valid)])
        out = model(batch["cats"], batch["nums"], batch["actions"], batch["mask"])
        self.assertEqual(tuple(out["pi_logits"].shape), (1, 8))
        self.assertEqual(tuple(out["v"].shape), (1,))

    def test_checkpoint_roundtrip_act(self):
        from policies.transformer_policy import load_bundle, transformer_awbc_act
        import policies.transformer_policy as tp

        schema = FeatureSchema()
        model = CausalTransformerAWBCNet(schema, d_model=64, n_heads=4, n_layers=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transformer_awbc.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "schema": schema.to_dict(),
                    "config": {
                        "kind": "awbc",
                        "d_model": 64,
                        "n_heads": 4,
                        "n_layers": 1,
                        "reward_scale": 100.0,
                    },
                },
                path,
            )
            tp._bundles.clear()
            bundle = load_bundle(path)
            self.assertEqual(bundle["kind"], "awbc")
            # Point default AWBC path via temporary monkeypatch
            old = tp._AWBC_CKPT
            tp._AWBC_CKPT = path
            try:
                a = transformer_awbc_act(_obs(), [])
                self.assertIn(a, ACTIONS)
                self.assertEqual(
                    transformer_awbc_act(_obs(customer_signal="unhappy"), []),
                    "ESCALATE_TO_HUMAN",
                )
            finally:
                tp._AWBC_CKPT = old
                tp._bundles.clear()


if __name__ == "__main__":
    unittest.main()

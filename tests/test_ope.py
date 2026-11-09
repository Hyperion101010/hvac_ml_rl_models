"""Unit tests for WIS / DR off-policy helpers."""

from __future__ import annotations

import unittest

from eval.ope import (
    clip_ratio,
    discounted_return,
    episode_is_weight,
    evaluate_ope,
    step_dr_value,
)


class OPEHelperTests(unittest.TestCase):
    def test_discounted_return(self):
        self.assertAlmostEqual(discounted_return([1.0, 2.0], 0.5), 1.0 + 0.5 * 2.0)

    def test_clip_ratio(self):
        self.assertAlmostEqual(clip_ratio(1.0, 0.1, 20.0), 10.0)
        self.assertAlmostEqual(clip_ratio(1.0, 0.01, 20.0), 20.0)
        self.assertEqual(clip_ratio(1.0, 0.0, 20.0), 0.0)

    def test_episode_weight_match_and_mismatch(self):
        steps = [
            {"action": "WAIT", "action_prob": 0.5},
            {"action": "CHECK_IN", "action_prob": 0.25},
        ]
        w, ok = episode_is_weight(steps, ["WAIT", "CHECK_IN"], clip=20.0)
        self.assertTrue(ok)
        self.assertAlmostEqual(w, (1 / 0.5) * (1 / 0.25))
        w0, ok0 = episode_is_weight(steps, ["WAIT", "PARK_THREAD"], clip=20.0)
        self.assertFalse(ok0)
        self.assertEqual(w0, 0.0)

    def test_evaluate_ope_always_match(self):
        episodes = [
            [
                {
                    "episode_id": "e0",
                    "action": "WAIT",
                    "action_prob": 0.5,
                    "reward": 10.0,
                    "done": True,
                    "observation": {
                        "track_hint": "deal",
                        "customer_signal": "none",
                        "topic": "open_estimate",
                        "service_risk_score": 0.1,
                    },
                }
            ]
        ]

        def act(obs, hist):
            return "WAIT"

        def q_fn(obs, hist, action):
            return 0.0

        r = evaluate_ope(
            "wait",
            act,
            episodes,
            gamma=1.0,
            clip=20.0,
            q_fn=q_fn,
            reward_scale=1.0,
        )
        self.assertAlmostEqual(r["wis"], 10.0)
        self.assertAlmostEqual(r["coverage"], 1.0)
        self.assertGreater(r["ess"], 0.0)

    def test_dr_zero_q_reduces_to_is_style(self):
        steps = [
            {
                "action": "WAIT",
                "action_prob": 1.0,
                "reward": 4.0,
                "done": True,
                "observation": {"service_risk_score": 0.0, "customer_signal": "none"},
            }
        ]

        def q_fn(obs, hist, action):
            return 0.0

        val = step_dr_value(steps, ["WAIT"], q_fn, gamma=1.0, clip=20.0, reward_scale=1.0)
        # V=0, rho=1, contrib = 0 + 1*(4-0) = 4
        self.assertAlmostEqual(val, 4.0)


if __name__ == "__main__":
    unittest.main()

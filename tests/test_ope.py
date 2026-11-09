"""Unit tests for WIS / DR off-policy helpers."""

from __future__ import annotations

import unittest

from eval.ope import (
    clip_ratio,
    discounted_return,
    episode_is_weight,
    evaluate_ope,
    smoothed_pi_prob,
    step_dr_value,
    step_hint_flags,
)


class OPEHelperTests(unittest.TestCase):
    def test_discounted_return(self):
        self.assertAlmostEqual(discounted_return([1.0, 2.0], 0.5), 1.0 + 0.5 * 2.0)

    def test_clip_ratio(self):
        self.assertAlmostEqual(clip_ratio(1.0, 0.1, 20.0), 10.0)
        self.assertAlmostEqual(clip_ratio(1.0, 0.01, 20.0), 20.0)

    def test_smoothed_prob(self):
        self.assertAlmostEqual(smoothed_pi_prob("WAIT", "WAIT", 0.9), 0.9)
        other = smoothed_pi_prob("WAIT", "CHECK_IN", 0.9)
        self.assertAlmostEqual(other, 0.1 / 7)

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

    def test_hint_flags(self):
        flags = step_hint_flags(
            {
                "relationship_score": 0.9,
                "turn": 2,
                "ignored_count": 2,
                "customer_signal": "price_objection",
                "service_risk_score": 0.1,
            }
        )
        self.assertTrue(flags["high_relationship"])
        self.assertTrue(flags["late_ignored"])
        self.assertTrue(flags["objection"])

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
                        "relationship_score": 0.2,
                        "turn": 0,
                        "ignored_count": 0,
                        "case_id": "e0",
                    },
                }
            ]
        ]

        def act(obs, hist):
            return "WAIT"

        r = evaluate_ope(
            "wait",
            act,
            episodes,
            gamma=1.0,
            clip=20.0,
            q_fn=None,
            reward_scale=1.0,
            smooth=1.0,
            n_boot=20,
        )
        self.assertAlmostEqual(r["wis"], 10.0)
        self.assertIn("wis_ci", r)

    def test_dr_zero_q(self):
        steps = [
            {
                "action": "WAIT",
                "action_prob": 1.0,
                "reward": 4.0,
                "done": True,
                "observation": {
                    "service_risk_score": 0.0,
                    "customer_signal": "none",
                    "case_id": "e",
                    "turn": 0,
                },
            }
        ]

        def q_fn(obs, hist, action):
            return 0.0

        val = step_dr_value(
            steps,
            ["WAIT"],
            q_fn,
            gamma=1.0,
            clip=20.0,
            reward_scale=1.0,
            episode_id="e",
        )
        self.assertAlmostEqual(val, 4.0)


if __name__ == "__main__":
    unittest.main()

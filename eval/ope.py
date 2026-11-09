"""Off-policy estimators: trajectory WIS/SNIPS and step-wise doubly robust.

Behavior propensities: logged action_prob (optionally mixed with uniform for
smoothed evaluation policies). History for act/Q uses the logged prefix.

DR should use an independent FQE Q checkpoint (models/transformer_fqe.pt),
not the candidate policy's own Q.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from policies.base import ACTIONS, ACTIONS_SET, PolicyFn

QFn = Callable[[Dict[str, Any], List[Dict[str, Any]], str], float]
PiProbFn = Callable[[Dict[str, Any], List[Dict[str, Any]], str], float]

HINT_SLICES = (
    "high_relationship",
    "late_ignored",
    "objection",
    "unhappy_or_risk",
)
OBJECTION_SIGNALS = {
    "price_objection",
    "competitor",
    "ask_spouse",
    "timing_delay",
}
ESS_SLICE_MIN = 30.0


@dataclass
class EpisodeOPE:
    episode_id: str
    return_g: float
    weight: float
    dr_value: float
    n_steps: int
    fully_covered: bool
    step_slices: List[Dict[str, bool]] = field(default_factory=list)


def discounted_return(rewards: Sequence[float], gamma: float) -> float:
    g = 0.0
    for r in reversed(rewards):
        g = float(r) + gamma * g
    return g


def clip_ratio(pi_a: float, mu: float, clip: float) -> float:
    if mu <= 0:
        return 0.0
    return min(float(pi_a) / float(mu), float(clip))


def logged_history_prefix(steps: Sequence[Dict[str, Any]], t: int) -> List[Dict[str, Any]]:
    hist: List[Dict[str, Any]] = []
    for s in steps[:t]:
        hist.append(
            {
                "observation": dict(s["observation"]),
                "action": s["action"],
                "reward": 0.0,
                "done": bool(s["done"]),
            }
        )
    return hist


def step_hint_flags(observation: Dict[str, Any]) -> Dict[str, bool]:
    try:
        rel = float(observation.get("relationship_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        rel = 0.0
    try:
        turn = int(observation.get("turn", 0) or 0)
    except (TypeError, ValueError):
        turn = 0
    try:
        ignored = int(observation.get("ignored_count", 0) or 0)
    except (TypeError, ValueError):
        ignored = 0
    try:
        risk = float(observation.get("service_risk_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        risk = 0.0
    signal = str(observation.get("customer_signal") or "none")
    return {
        "high_relationship": rel >= 0.7,
        "late_ignored": turn >= 2 and ignored >= 2,
        "objection": signal in OBJECTION_SIGNALS,
        "unhappy_or_risk": signal == "unhappy" or risk > 0.75,
    }


def deterministic_pi_prob(chosen: str, action: str) -> float:
    return 1.0 if chosen == action else 0.0


def smoothed_pi_prob(chosen: str, action: str, mass: float = 0.9) -> float:
    """mass on argmax action, (1-mass)/(|A|-1) on others — or uniform remainder."""
    n = len(ACTIONS)
    if chosen == action:
        return float(mass)
    return (1.0 - float(mass)) / float(n - 1)


def collect_pi_actions(
    act: PolicyFn,
    episodes: Sequence[Sequence[Dict[str, Any]]],
) -> List[List[str]]:
    out: List[List[str]] = []
    for steps in episodes:
        chosen: List[str] = []
        for t, row in enumerate(steps):
            hist = logged_history_prefix(steps, t)
            a = act(dict(row["observation"]), hist)
            if a not in ACTIONS_SET:
                raise RuntimeError(f"Invalid action from policy: {a!r}")
            chosen.append(a)
        out.append(chosen)
    return out


def episode_is_weight(
    steps: Sequence[Dict[str, Any]],
    pi_actions: Sequence[str],
    clip: float,
    *,
    smooth: float = 1.0,
) -> Tuple[float, bool]:
    """Product of clipped pi/mu. smooth<1 mixes argmax with uniform."""
    w = 1.0
    fully = True
    for row, pi_a in zip(steps, pi_actions):
        logged = row["action"]
        mu = float(row.get("action_prob") or 0.0)
        if smooth >= 1.0:
            if pi_a != logged:
                return 0.0, False
            pia = 1.0
        else:
            pia = smoothed_pi_prob(pi_a, logged, mass=smooth)
            if pia <= 0:
                fully = False
        w *= clip_ratio(pia, mu, clip)
        if w == 0.0:
            return 0.0, False
    return w, fully


def step_dr_value(
    steps: Sequence[Dict[str, Any]],
    pi_actions: Sequence[str],
    q_fn: QFn,
    gamma: float,
    clip: float,
    reward_scale: float,
    *,
    episode_id: str,
    smooth: float = 1.0,
) -> float:
    total = 0.0
    rho_prev = 1.0
    disc = 1.0
    for t, row in enumerate(steps):
        obs = dict(row["observation"])
        hist = logged_history_prefix(steps, t)
        pi_a = pi_actions[t]
        a_log = row["action"]
        mu = float(row.get("action_prob") or 0.0)
        r = float(row["reward"])

        q_pi = q_fn(obs, hist, pi_a) * reward_scale
        q_log = q_fn(obs, hist, a_log) * reward_scale
        if smooth >= 1.0:
            pia = 1.0 if pi_a == a_log else 0.0
            v = q_pi
        else:
            # V = sum_a pi(a) Q(s,a)
            v = 0.0
            for a in ACTIONS:
                v += smoothed_pi_prob(pi_a, a, mass=smooth) * (
                    q_fn(obs, hist, a) * reward_scale
                )
            pia = smoothed_pi_prob(pi_a, a_log, mass=smooth)

        total += disc * rho_prev * v
        rho_t = clip_ratio(pia, mu, clip)
        total += disc * rho_prev * rho_t * (r - q_log)
        rho_prev *= rho_t
        disc *= gamma
        if rho_prev == 0.0:
            break
    return total


def _aggregate_is(
    weights: Sequence[float], returns: Sequence[float]
) -> Tuple[float, float, float]:
    if not weights:
        return 0.0, 0.0, 0.0
    w_sum = sum(weights)
    if w_sum <= 0:
        return 0.0, 0.0, 0.0
    wis = sum(w * g for w, g in zip(weights, returns)) / w_sum
    ess = (w_sum * w_sum) / sum(w * w for w in weights if w > 0)
    return wis, wis, ess


def bootstrap_ci(
    weights: Sequence[float],
    returns: Sequence[float],
    *,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(weights)
    if n == 0:
        return 0.0, 0.0
    stats: List[float] = []
    idx = list(range(n))
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in range(n)]
        w = [weights[i] for i in sample]
        g = [returns[i] for i in sample]
        wis, _, _ = _aggregate_is(w, g)
        stats.append(wis)
    stats.sort()
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def evaluate_ope(
    name: str,
    act: PolicyFn,
    episodes: Sequence[Sequence[Dict[str, Any]]],
    *,
    gamma: float = 0.95,
    clip: float = 20.0,
    q_fn: Optional[QFn] = None,
    reward_scale: float = 100.0,
    smooth: float = 1.0,
    n_boot: int = 200,
    baseline_records: Optional[Sequence[EpisodeOPE]] = None,
) -> Dict[str, Any]:
    pi_actions = collect_pi_actions(act, episodes)
    ep_records: List[EpisodeOPE] = []
    agree = 0
    n_steps = 0

    for steps, pi_acts in zip(episodes, pi_actions):
        rewards = [float(r["reward"]) for r in steps]
        g = discounted_return(rewards, gamma)
        w, covered = episode_is_weight(steps, pi_acts, clip, smooth=smooth)
        eid = str(steps[0].get("episode_id", ""))
        dr = 0.0
        if q_fn is not None:
            dr = step_dr_value(
                steps,
                pi_acts,
                q_fn,
                gamma,
                clip,
                reward_scale,
                episode_id=eid,
                smooth=smooth,
            )
        flags_list: List[Dict[str, bool]] = []
        for row, pi_a in zip(steps, pi_acts):
            n_steps += 1
            if pi_a == row["action"]:
                agree += 1
            flags_list.append(step_hint_flags(row["observation"]))
        ep_records.append(
            EpisodeOPE(
                episode_id=eid,
                return_g=g,
                weight=w,
                dr_value=dr,
                n_steps=len(steps),
                fully_covered=covered,
                step_slices=flags_list,
            )
        )

    weights = [e.weight for e in ep_records]
    returns = [e.return_g for e in ep_records]
    wis, snips, ess = _aggregate_is(weights, returns)
    lo, hi = bootstrap_ci(weights, returns, n_boot=n_boot, seed=hash(name) % 10_000)
    dr_mean = (
        sum(e.dr_value for e in ep_records) / len(ep_records) if ep_records else 0.0
    )
    coverage = (
        sum(1 for e in ep_records if e.fully_covered) / len(ep_records)
        if ep_records
        else 0.0
    )

    # Hint-driven step slices: episode included if any step matches; ESS on those eps
    by_hint: Dict[str, Dict[str, Any]] = {}
    for hint in HINT_SLICES:
        sub = [
            e
            for e in ep_records
            if any(flags.get(hint) for flags in e.step_slices)
        ]
        w_s = [e.weight for e in sub]
        g_s = [e.return_g for e in sub]
        wis_s, _, ess_s = _aggregate_is(w_s, g_s)
        entry = {
            "n_episodes": len(sub),
            "wis": wis_s,
            "ess": ess_s,
            "coverage": (
                sum(1 for e in sub if e.fully_covered) / len(sub) if sub else 0.0
            ),
            "hidden": ess_s < ESS_SLICE_MIN,
        }
        by_hint[hint] = entry

    paired = None
    if baseline_records is not None and len(baseline_records) == len(ep_records):
        # Paired WIS difference via shared bootstrap on delta of weighted returns
        # Use per-episode contribution approx: w_i * g_i normalized difference
        paired = {
            "wis_diff": wis
            - (
                sum(e.weight * e.return_g for e in baseline_records)
                / max(sum(e.weight for e in baseline_records), 1e-12)
                if sum(e.weight for e in baseline_records) > 0
                else 0.0
            ),
            "dr_diff": dr_mean
            - (sum(e.dr_value for e in baseline_records) / len(baseline_records)),
        }

    return {
        "name": name,
        "n_episodes": len(ep_records),
        "n_steps": n_steps,
        "agreement_rate": agree / n_steps if n_steps else 0.0,
        "wis": wis,
        "snips": snips,
        "ess": ess,
        "coverage": coverage,
        "dr": dr_mean,
        "wis_ci": (lo, hi),
        "by_hint": by_hint,
        "paired_vs_product_rule": paired,
        "_records": ep_records,
    }


def make_fqi_q_fn(ckpt_path: Optional[Path] = None) -> Tuple[QFn, float]:
    """Q-fn from an FQI/FQE checkpoint; cache key = (episode_id, turn)."""
    from policies.transformer_policy import _forward, load_bundle

    bundle = load_bundle(ckpt_path)
    scale = float(bundle.get("reward_scale", 100.0))
    cache: Dict[Tuple[str, int], List[float]] = {}

    def q_fn(
        observation: Dict[str, Any], history: List[Dict[str, Any]], action: str
    ) -> float:
        eid = str(observation.get("case_id") or observation.get("episode_id") or "")
        try:
            turn = int(observation.get("turn", len(history)) or len(history))
        except (TypeError, ValueError):
            turn = len(history)
        key = (eid, turn)
        vec = cache.get(key)
        if vec is None:
            out = _forward(observation, history, bundle)
            vec = out["q_values"][0].detach().cpu().tolist()
            cache[key] = vec
        return float(vec[ACTIONS.index(action)])

    return q_fn, scale

"""Off-policy estimators: trajectory WIS/SNIPS and step-wise doubly robust.

Behavior propensities come from logged action_prob. Evaluation policies are
treated as deterministic (argmax / rules): pi(a|s)=1[a=pi(s)].

History for act() and Q(s) uses the *logged* prefix so states match the MDP
in the logs (not the candidate's counterfactual history).

DR uses a fitted Q (default: transformer_fqi checkpoint). That makes DR for
transformer_fqi partially circular — prefer WIS/SNIPS for ranking when ESS
is healthy; treat DR as a secondary check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from policies.base import ACTIONS, ACTIONS_SET, PolicyFn

QFn = Callable[[Dict[str, Any], List[Dict[str, Any]], str], float]


@dataclass
class EpisodeOPE:
    episode_id: str
    return_g: float
    weight: float
    dr_value: float
    n_steps: int
    fully_covered: bool
    slice_keys: Dict[str, str]


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


def collect_pi_actions(
    act: PolicyFn,
    episodes: Sequence[Sequence[Dict[str, Any]]],
) -> List[List[str]]:
    """For each episode, list of pi(s_t) using logged history prefixes."""
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
) -> Tuple[float, bool]:
    """Product of clipped pi/mu along the episode; 0 if any mismatch."""
    w = 1.0
    for row, pi_a in zip(steps, pi_actions):
        logged = row["action"]
        mu = float(row.get("action_prob") or 0.0)
        if pi_a != logged:
            return 0.0, False
        w *= clip_ratio(1.0, mu, clip)
        if w == 0.0:
            return 0.0, False
    return w, True


def step_dr_value(
    steps: Sequence[Dict[str, Any]],
    pi_actions: Sequence[str],
    q_fn: QFn,
    gamma: float,
    clip: float,
    reward_scale: float,
) -> float:
    """sum_t γ^t [ ρ_{0:t-1} V(s_t) + ρ_{0:t} (r_t - Q(s_t, a_t)) ]."""
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

        # Q in scaled units from FQI; convert to raw reward units
        q_pi = q_fn(obs, hist, pi_a) * reward_scale
        q_log = q_fn(obs, hist, a_log) * reward_scale
        v = q_pi

        total += disc * rho_prev * v
        rho_t = clip_ratio(1.0 if pi_a == a_log else 0.0, mu, clip)
        total += disc * rho_prev * rho_t * (r - q_log)

        rho_prev = rho_prev * rho_t
        disc *= gamma
        if rho_prev == 0.0 and t + 1 < len(steps):
            # Remaining terms with rho_prev=0 only contribute 0 for IS residual;
            # V terms still need rho_prev — once zero, later V also zero.
            break
    return total


def evaluate_ope(
    name: str,
    act: PolicyFn,
    episodes: Sequence[Sequence[Dict[str, Any]]],
    *,
    gamma: float = 0.95,
    clip: float = 20.0,
    q_fn: Optional[QFn] = None,
    reward_scale: float = 100.0,
    slice_field: str = "track_hint",
) -> Dict[str, Any]:
    pi_actions = collect_pi_actions(act, episodes)
    ep_records: List[EpisodeOPE] = []
    agree = 0
    n_steps = 0

    for steps, pi_acts in zip(episodes, pi_actions):
        rewards = [float(r["reward"]) for r in steps]
        g = discounted_return(rewards, gamma)
        w, covered = episode_is_weight(steps, pi_acts, clip)
        dr = 0.0
        if q_fn is not None:
            dr = step_dr_value(steps, pi_acts, q_fn, gamma, clip, reward_scale)

        for row, pi_a in zip(steps, pi_acts):
            n_steps += 1
            if pi_a == row["action"]:
                agree += 1

        first = steps[0]["observation"]
        ep_records.append(
            EpisodeOPE(
                episode_id=str(steps[0].get("episode_id", "")),
                return_g=g,
                weight=w,
                dr_value=dr,
                n_steps=len(steps),
                fully_covered=covered,
                slice_keys={
                    "track_hint": str(first.get("track_hint")),
                    "customer_signal": str(first.get("customer_signal")),
                    "topic": str(first.get("topic")),
                },
            )
        )

    wis, snips, ess, cov = _aggregate_is(ep_records)
    dr_mean = (
        sum(e.dr_value for e in ep_records) / len(ep_records) if ep_records else 0.0
    )

    by_slice: Dict[str, Dict[str, Any]] = {}
    for field in ("track_hint", "customer_signal", "topic"):
        by_slice[field] = {}
        buckets = sorted({e.slice_keys[field] for e in ep_records})
        for b in buckets:
            sub = [e for e in ep_records if e.slice_keys[field] == b]
            w_s, s_s, ess_s, cov_s = _aggregate_is(sub)
            dr_s = sum(e.dr_value for e in sub) / len(sub) if sub else 0.0
            by_slice[field][b] = {
                "n_episodes": len(sub),
                "wis": w_s,
                "snips": s_s,
                "ess": ess_s,
                "coverage": cov_s,
                "dr": dr_s,
            }

    return {
        "name": name,
        "n_episodes": len(ep_records),
        "n_steps": n_steps,
        "agreement_rate": agree / n_steps if n_steps else 0.0,
        "wis": wis,
        "snips": snips,
        "ess": ess,
        "coverage": cov,
        "dr": dr_mean,
        "by_slice": by_slice,
    }


def _aggregate_is(
    records: Sequence[EpisodeOPE],
) -> Tuple[float, float, float, float]:
    if not records:
        return 0.0, 0.0, 0.0, 0.0
    weights = [e.weight for e in records]
    returns = [e.return_g for e in records]
    w_sum = sum(weights)
    covered = sum(1 for e in records if e.fully_covered) / len(records)
    if w_sum <= 0:
        return 0.0, 0.0, 0.0, covered
    wis = sum(w * g for w, g in zip(weights, returns)) / w_sum
    # SNIPS: same as WIS for trajectory IS with self-normalization
    snips = wis
    ess = (w_sum * w_sum) / sum(w * w for w in weights if w > 0)
    return wis, snips, ess, covered


def make_fqi_q_fn(ckpt_path: Optional[Path] = None) -> Tuple[QFn, float]:
    """Return (q_fn, reward_scale) using transformer FQI Q-head.

    Caches the full Q-vector per (id(observation), len(history)) call site
    so DR only runs one forward per step, not one per action.
    """
    from policies.transformer_policy import _forward, load_bundle

    bundle = load_bundle(ckpt_path)
    if bundle.get("kind") != "fqi":
        raise ValueError("DR Q checkpoint must be an FQI model")
    scale = float(bundle.get("reward_scale", 100.0))
    cache: Dict[Tuple[int, int], Any] = {}

    def q_fn(
        observation: Dict[str, Any], history: List[Dict[str, Any]], action: str
    ) -> float:
        key = (id(observation), len(history))
        vec = cache.get(key)
        if vec is None:
            out = _forward(observation, history, bundle)
            vec = out["q_values"][0].detach().cpu().tolist()
            cache[key] = vec
        return float(vec[ACTIONS.index(action)])

    return q_fn, scale

"""Side-by-side offline comparison of registered policies.

Matched-log mean reward is NOT a counterfactual value estimate; it only
averages logged rewards on steps where the candidate action equals the
logged action.

Usage (from package root):

    python -m eval.compare_policies \\
      --logs data/valid_logs.jsonl \\
      --policies product_rule,simple_rule,always_estimate_nudge
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from policies import get, list_policies
from policies.base import ACTIONS, ACTIONS_SET, PolicyFn


def load_rows(log_path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def group_episodes(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for row in rows:
        eid = row["episode_id"]
        if eid not in by_id:
            order.append(eid)
        by_id[eid].append(row)
    episodes: List[List[Dict[str, Any]]] = []
    for eid in order:
        episodes.append(sorted(by_id[eid], key=lambda r: r["step"]))
    return episodes


def evaluate_policy(
    name: str,
    act: PolicyFn,
    episodes: Sequence[Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    action_counts = Counter({a: 0 for a in ACTIONS})
    n = 0
    agree = 0
    matched_rewards: List[float] = []
    actions_chosen: List[str] = []
    logged_actions: List[str] = []
    slice_agree: Dict[str, Counter] = {
        "track_hint": Counter(),
        "customer_signal": Counter(),
        "topic": Counter(),
    }
    slice_total: Dict[str, Counter] = {
        "track_hint": Counter(),
        "customer_signal": Counter(),
        "topic": Counter(),
    }

    for steps in episodes:
        history: List[Dict[str, Any]] = []
        for row in steps:
            obs = dict(row["observation"])
            action = act(obs, list(history))
            if action not in ACTIONS_SET:
                raise RuntimeError(
                    f"Invalid action from {name!r}: {action!r}. "
                    f"Expected one of {', '.join(ACTIONS)}"
                )
            logged = row["action"]
            action_counts[action] += 1
            actions_chosen.append(action)
            logged_actions.append(logged)
            n += 1
            matched = action == logged
            if matched:
                agree += 1
                matched_rewards.append(float(row["reward"]))
            for key in slice_total:
                bucket = str(obs.get(key))
                slice_total[key][bucket] += 1
                if matched:
                    slice_agree[key][bucket] += 1
            history.append(
                {
                    "observation": obs,
                    "action": action,
                    "reward": 0.0,
                    "done": bool(row["done"]),
                }
            )

    matched_mean = (
        sum(matched_rewards) / len(matched_rewards) if matched_rewards else None
    )
    return {
        "name": name,
        "n": n,
        "action_counts": dict(action_counts),
        "agreement_rate": agree / n if n else 0.0,
        "matched_n": len(matched_rewards),
        "matched_log_mean_reward": matched_mean,
        "actions_chosen": actions_chosen,
        "logged_actions": logged_actions,
        "slice_agree": {k: dict(v) for k, v in slice_agree.items()},
        "slice_total": {k: dict(v) for k, v in slice_total.items()},
    }


def pairwise_disagreement(
    results: Sequence[Dict[str, Any]],
) -> List[Tuple[str, str, float]]:
    pairs: List[Tuple[str, str, float]] = []
    for i, left in enumerate(results):
        for right in results[i + 1 :]:
            a = left["actions_chosen"]
            b = right["actions_chosen"]
            if len(a) != len(b) or not a:
                rate = 0.0
            else:
                rate = sum(x != y for x, y in zip(a, b)) / len(a)
            pairs.append((left["name"], right["name"], rate))
    return pairs


def _fmt_rate(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.4f}"


def print_report(results: Sequence[Dict[str, Any]]) -> None:
    print(
        "NOTE: matched-log mean reward != counterfactual value; "
        "only averages logged rewards where pi(a)==logged_a."
    )
    print()
    header = (
        f"{'policy':22s} {'n':>7s} {'agree':>8s} {'matched_n':>9s} "
        f"{'matched_rew':>12s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:22s} {r['n']:7d} {_fmt_rate(r['agreement_rate']):>8s} "
            f"{r['matched_n']:9d} {_fmt_rate(r['matched_log_mean_reward']):>12s}"
        )

    print("\nAction distributions (counts):")
    for r in results:
        print(f"  [{r['name']}]")
        for action in ACTIONS:
            print(f"    {action:20s} {r['action_counts'].get(action, 0):6d}")

    print("\nPairwise disagreement rates:")
    for left, right, rate in pairwise_disagreement(results):
        print(f"  {left} vs {right}: {rate:.4f}")

    print("\nAgreement by slice (product_rule if present, else first policy):")
    focus = next((r for r in results if r["name"] == "product_rule"), results[0])
    for key in ("track_hint", "customer_signal", "topic"):
        totals = focus["slice_total"][key]
        agrees = focus["slice_agree"][key]
        top = sorted(totals.items(), key=lambda kv: -kv[1])[:8]
        print(f"  [{focus['name']} / {key}]")
        for bucket, total in top:
            rate = agrees.get(bucket, 0) / total if total else 0.0
            print(f"    {bucket:24s} n={total:5d} agree={rate:.3f}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        default=str(root / "data" / "valid_logs.jsonl"),
        help="Path to jsonl logs",
    )
    parser.add_argument(
        "--policies",
        default="product_rule,simple_rule,always_estimate_nudge",
        help="Comma-separated registered policy names",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max log rows (0 = all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered policies and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(list_policies()))
        return

    names = [n.strip() for n in args.policies.split(",") if n.strip()]
    if not names:
        raise SystemExit("No policies specified")

    rows = load_rows(Path(args.logs), args.limit)
    if not rows:
        raise SystemExit(f"No rows in {args.logs}")
    episodes = group_episodes(rows)

    results = []
    for name in names:
        results.append(evaluate_policy(name, get(name), episodes))
    print_report(results)


if __name__ == "__main__":
    main()

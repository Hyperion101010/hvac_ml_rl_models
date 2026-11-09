"""Run WIS / SNIPS / DR off-policy evaluation on registered policies.

Usage (from package root):

    python -m eval.run_ope \\
      --logs data/valid_logs.jsonl \\
      --policies product_rule,simple_rule,transformer_bc,transformer_awbc,transformer_fqi,transformer_fqi_acq
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from eval.compare_policies import group_episodes, load_rows
from eval.ope import evaluate_ope, make_fqi_q_fn
from policies import get, list_policies


def print_ope_report(results, slice_top: int = 6) -> None:
    print(
        "OPE notes: WIS/SNIPS use logged action_prob; deterministic pi. "
        "DR uses transformer_fqi Q (partially circular for that policy). "
        "ESS = effective sample size of episode weights; low ESS => unstable."
    )
    print()
    header = (
        f"{'policy':22s} {'n_ep':>6s} {'agree':>7s} {'cover':>7s} "
        f"{'ESS':>8s} {'WIS':>10s} {'SNIPS':>10s} {'DR':>10s}"
    )
    print(header)
    print("-" * len(header))
    ranked = sorted(results, key=lambda r: r["wis"], reverse=True)
    for r in ranked:
        print(
            f"{r['name']:22s} {r['n_episodes']:6d} {r['agreement_rate']:7.3f} "
            f"{r['coverage']:7.3f} {r['ess']:8.1f} "
            f"{r['wis']:10.3f} {r['snips']:10.3f} {r['dr']:10.3f}"
        )

    print("\nWIS by track_hint (top buckets by n):")
    for r in ranked:
        buckets = r["by_slice"]["track_hint"]
        top = sorted(buckets.items(), key=lambda kv: -kv[1]["n_episodes"])[:slice_top]
        print(f"  [{r['name']}]")
        for name, m in top:
            print(
                f"    {name:16s} n={m['n_episodes']:4d} "
                f"WIS={m['wis']:8.3f} ESS={m['ess']:6.1f} cover={m['coverage']:.2f}"
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", default=str(root / "data" / "valid_logs.jsonl"))
    p.add_argument(
        "--policies",
        default=(
            "product_rule,simple_rule,transformer_bc,"
            "transformer_awbc,transformer_fqi,transformer_fqi_acq"
        ),
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=20.0)
    p.add_argument(
        "--q-ckpt",
        default=str(root / "models" / "transformer_fqi.pt"),
        help="FQI checkpoint used as Q-hat for DR",
    )
    p.add_argument("--no-dr", action="store_true", help="Skip DR (WIS/SNIPS only)")
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        print("\n".join(list_policies()))
        return

    names = [n.strip() for n in args.policies.split(",") if n.strip()]
    rows = load_rows(Path(args.logs), args.limit)
    if not rows:
        raise SystemExit(f"No rows in {args.logs}")
    episodes = group_episodes(rows)

    q_fn = None
    reward_scale = 100.0
    if not args.no_dr:
        q_path = Path(args.q_ckpt)
        if not q_path.exists():
            print(f"warning: Q ckpt missing at {q_path}; running without DR")
        else:
            q_fn, reward_scale = make_fqi_q_fn(q_path)
            print(f"DR Q from {q_path} (reward_scale={reward_scale})")

    results = []
    for name in names:
        print(f"evaluating {name} ...", flush=True)
        results.append(
            evaluate_ope(
                name,
                get(name),
                episodes,
                gamma=args.gamma,
                clip=args.clip,
                q_fn=q_fn,
                reward_scale=reward_scale,
            )
        )
    print()
    print_ope_report(results)


if __name__ == "__main__":
    main()

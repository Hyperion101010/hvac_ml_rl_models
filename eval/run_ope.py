"""Run WIS / SNIPS / DR off-policy evaluation on registered policies.

Reports WIS at clip=5 and clip=20, bootstrap CIs, hint slices (ESS>=30),
paired diffs vs product_rule, and optional smoothed pi (90/10).

    python -m eval.run_ope
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval.compare_policies import group_episodes, load_rows
from eval.ope import ESS_SLICE_MIN, evaluate_ope, make_fqi_q_fn
from policies import get, list_policies


DEFAULT_POLICIES = (
    "product_rule,simple_rule,always_check_in,always_estimate_nudge,always_escalate,"
    "transformer_bc,transformer_awbc,transformer_fqi,transformer_fqi_acq,one_step_greedy"
)


def print_ope_report(
    results_by_clip: Dict[float, List[Dict[str, Any]]],
    *,
    smooth: float,
) -> None:
    print(
        "OPE: trajectory WIS/SNIPS + DR (independent FQE Q when available). "
        f"smooth={smooth} (1=deterministic). Hide hint slices with ESS<{ESS_SLICE_MIN}."
    )
    for clip, results in sorted(results_by_clip.items()):
        print()
        print(f"=== clip={clip} ===")
        header = (
            f"{'policy':22s} {'agree':>6s} {'cover':>6s} {'ESS':>7s} "
            f"{'WIS':>9s} {'CI_lo':>8s} {'CI_hi':>8s} {'DR':>9s} {'dWIS':>8s}"
        )
        print(header)
        print("-" * len(header))
        ranked = sorted(results, key=lambda r: r["wis"], reverse=True)
        for r in ranked:
            lo, hi = r.get("wis_ci", (0.0, 0.0))
            paired = r.get("paired_vs_product_rule") or {}
            dw = paired.get("wis_diff")
            dw_s = f"{dw:8.3f}" if dw is not None else f"{'n/a':>8s}"
            print(
                f"{r['name']:22s} {r['agreement_rate']:6.3f} {r['coverage']:6.3f} "
                f"{r['ess']:7.1f} {r['wis']:9.3f} {lo:8.3f} {hi:8.3f} "
                f"{r['dr']:9.3f} {dw_s}"
            )

        print("\nHint slices (shown only if ESS >= {:.0f}):".format(ESS_SLICE_MIN))
        for r in ranked:
            print(f"  [{r['name']}]")
            for hint, m in r.get("by_hint", {}).items():
                if m.get("hidden"):
                    print(f"    {hint:18s} hidden (ESS={m['ess']:.1f})")
                else:
                    print(
                        f"    {hint:18s} n={m['n_episodes']:4d} "
                        f"WIS={m['wis']:8.3f} ESS={m['ess']:6.1f}"
                    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", default=str(root / "data" / "valid_logs.jsonl"))
    p.add_argument("--policies", default=DEFAULT_POLICIES)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument(
        "--clips",
        default="5,20",
        help="Comma-separated IS weight clips",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=0.9,
        help="Argmax mass for smoothed pi (1.0 = deterministic)",
    )
    p.add_argument(
        "--q-ckpt",
        default=str(root / "models" / "transformer_fqe.pt"),
        help="Independent FQE checkpoint for DR",
    )
    p.add_argument("--no-dr", action="store_true")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        print("\n".join(list_policies()))
        return

    names = [n.strip() for n in args.policies.split(",") if n.strip()]
    clips = [float(x) for x in args.clips.split(",") if x.strip()]
    rows = load_rows(Path(args.logs), args.limit)
    if not rows:
        raise SystemExit(f"No rows in {args.logs}")
    episodes = group_episodes(rows)

    q_fn = None
    reward_scale = 100.0
    if not args.no_dr:
        q_path = Path(args.q_ckpt)
        if not q_path.exists():
            # fall back to FQI with a warning
            alt = root / "models" / "transformer_fqi.pt"
            if alt.exists():
                print(f"warning: {q_path} missing; falling back to {alt} for DR")
                q_path = alt
            else:
                print(f"warning: no Q ckpt; running without DR")
                q_path = None
        if q_path is not None:
            q_fn, reward_scale = make_fqi_q_fn(q_path)
            print(f"DR Q from {q_path} (reward_scale={reward_scale})")

    results_by_clip: Dict[float, List[Dict[str, Any]]] = {}
    for clip in clips:
        print(f"\n--- evaluating clip={clip} ---", flush=True)
        results: List[Dict[str, Any]] = []
        baseline_records = None
        # product_rule first for paired diffs when present
        ordered = list(names)
        if "product_rule" in ordered:
            ordered = ["product_rule"] + [n for n in ordered if n != "product_rule"]
        for name in ordered:
            print(f"  {name} ...", flush=True)
            r = evaluate_ope(
                name,
                get(name),
                episodes,
                gamma=args.gamma,
                clip=clip,
                q_fn=q_fn,
                reward_scale=reward_scale,
                smooth=args.smooth,
                n_boot=args.n_boot,
                baseline_records=baseline_records,
            )
            if name == "product_rule":
                baseline_records = r.get("_records")
            # drop heavy records from printed payload
            r = {k: v for k, v in r.items() if k != "_records"}
            results.append(r)
        results_by_clip[clip] = results

    print()
    print_ope_report(results_by_clip, smooth=args.smooth)


if __name__ == "__main__":
    main()

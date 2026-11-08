"""Smoke-test a submitted policy without exposing the evaluation simulator."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from baselines import ACTIONS


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import policy from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_policy"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "act"):
        raise RuntimeError("Policy module must define act(observation, history)")
    return module.act


def load_observations(log_path: Path, limit: int) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            observations.append(row["observation"])
            if len(observations) >= limit:
                break
    return observations


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(root / "starter" / "policy.py"))
    parser.add_argument("--logs", default=str(root / "data" / "train_logs.jsonl"))
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    act = load_policy(Path(args.policy))
    observations = load_observations(Path(args.logs), args.limit)
    if not observations:
        raise RuntimeError(f"No observations found in {args.logs}")

    counts = {action: 0 for action in ACTIONS}
    history: List[Dict[str, Any]] = []
    for idx, obs in enumerate(observations):
        action = act(dict(obs), list(history[-3:]))
        if action not in ACTIONS:
            raise RuntimeError(
                f"Invalid action at sample {idx}: {action!r}. "
                f"Expected one of {', '.join(ACTIONS)}"
            )
        counts[action] += 1
        history.append({"observation": obs, "action": action, "reward": 0.0, "done": False})

    print("Policy smoke test passed.")
    print(f"Samples checked: {len(observations)}")
    print("Action counts:")
    for action in ACTIONS:
        print(f"  {action}: {counts[action]}")


if __name__ == "__main__":
    main()


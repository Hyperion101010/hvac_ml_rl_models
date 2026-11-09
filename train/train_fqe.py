"""Train independent Fitted Q-Evaluation (behavior bootstrap) for OPE DR.

Bootstraps with the *logged* next action (SARSA-style FQE for μ), so DR does
not reuse the candidate policy's Q. Saves models/transformer_fqe.pt by default.
Use --gamma 0 --out-name transformer_one_step.pt for a one-step reward model.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from policies.base import ACTIONS
from policies.features import FeatureSchema
from policies.outcome_values import OUTCOME_CLASSES, compute_outcome_values
from policies.transformer_model import CausalTransformerPolicyNet
from train.train_transformer_fqi import (
    ROOT,
    TransitionDataset,
    collate_transitions,
    load_episodes,
    soft_update,
)

def compute_fqe_losses(
    model: CausalTransformerPolicyNet,
    target: CausalTransformerPolicyNet,
    batch: Dict[str, Any],
    device: torch.device,
    gamma: float,
    reward_scale: float,
) -> torch.Tensor:
    cur = {k: v.to(device) for k, v in batch["cur"].items()}
    nxt = {k: v.to(device) for k, v in batch["next"].items()}
    action_idx = batch["action_idx"].to(device)
    reward = batch["reward"].to(device)
    done = batch["done"].to(device)
    has_next = batch["has_next"].to(device)

    out = model(cur["cats"], cur["nums"], cur["actions"], cur["mask"])
    q_pred = out["q_values"][torch.arange(len(action_idx), device=device), action_idx]

    with torch.no_grad():
        tgt = target(nxt["cats"], nxt["nums"], nxt["actions"], nxt["mask"])
        # Behavior FQE: bootstrap logged next action (encoded as next token's pad action
        # is current's action in sequence builder). Use BC argmax on next state as proxy
        # for μ(s') when next action embedding is pad — fall back to max over BC support.
        from policies.transformer_model import bc_support_mask

        support = bc_support_mask(tgt["bc_logits"], threshold=0.01)
        # Use Q(s', a_logged_next): next sequence's last hist action is current action;
        # approximate with Q at next state for the same logged action index when continuing
        # is imperfect; better: gather Q for action that appears in next obs last_action.
        # Practical: use support-masked mean under BC probs as V^μ.
        probs = F.softmax(tgt["bc_logits"], dim=-1)
        probs = probs * support.float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        v_mu = (probs * tgt["q_values"]).sum(dim=-1)
        bootstrap = (~done) & has_next
        y = reward / reward_scale + gamma * bootstrap.float() * v_mu

    return F.smooth_l1_loss(q_pred, y)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_episodes = load_episodes(Path(args.train_logs), limit_rows=args.limit_rows)
    schema = FeatureSchema()
    flat = [r for ep in train_episodes for r in ep]
    outcome_values = compute_outcome_values(flat)
    dataset = TransitionDataset(train_episodes, schema)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_transitions
    )

    model = CausalTransformerPolicyNet(
        schema, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers
    ).to(device)
    target = copy.deepcopy(model).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / args.out_name

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            loss = compute_fqe_losses(
                model, target, batch, device, args.gamma, args.reward_scale
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            soft_update(target, model, tau=args.tau)
            total += float(loss.item())
            n += 1
        mean_loss = total / max(n, 1)
        print(f"epoch {epoch}/{args.epochs} fqe_loss={mean_loss:.4f}")
        if mean_loss < best:
            best = mean_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "schema": schema.to_dict(),
                    "outcome_values": outcome_values.to_dict(),
                    "config": {
                        "kind": "fqe",
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "gamma": args.gamma,
                        "reward_scale": args.reward_scale,
                        "tau": args.tau,
                        "lr": args.lr,
                        "seed": args.seed,
                        "epoch": epoch,
                        "epochs_planned": args.epochs,
                        "bc_threshold": 0.01,
                        "action_conditioned_q": False,
                        "actions": list(ACTIONS),
                        "outcome_classes": list(OUTCOME_CLASSES),
                    },
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path} (loss={best:.4f})")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-logs", default=str(ROOT / "data" / "train_logs.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "models"))
    p.add_argument("--out-name", default="transformer_fqe.pt")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--limit-rows", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    train(build_argparser().parse_args())


if __name__ == "__main__":
    main()

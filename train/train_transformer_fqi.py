"""Train causal transformer with outcome CE + BC CE + conservative FQI."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from policies.base import ACTIONS
from policies.features import FeatureSchema, build_sequence, collate_sequences
from policies.outcome_values import (
    OUTCOME_CLASSES,
    OutcomeValues,
    compute_outcome_values,
    outcome_to_index,
)
from policies.transformer_model import (
    CausalTransformerPolicyNet,
    bc_support_mask,
)


ROOT = Path(__file__).resolve().parents[1]


def load_episodes(path: Path, limit_rows: int = 0) -> List[List[Dict[str, Any]]]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            eid = row["episode_id"]
            if eid not in by:
                order.append(eid)
            by[eid].append(row)
            if limit_rows > 0 and i + 1 >= limit_rows:
                break
    episodes = []
    for eid in order:
        episodes.append(sorted(by[eid], key=lambda r: r["step"]))
    return episodes


def history_from_prior_steps(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "observation": dict(s["observation"]),
            "action": s["action"],
            "reward": 0.0,
            "done": False,
        }
        for s in steps
    ]


class TransitionDataset(Dataset):
    def __init__(self, episodes: Sequence[Sequence[Dict[str, Any]]], schema: FeatureSchema):
        self.schema = schema
        self.samples: List[Dict[str, Any]] = []
        for ep in episodes:
            for t, row in enumerate(ep):
                hist = history_from_prior_steps(ep[:t])
                cur_steps, cur_mask = build_sequence(row["observation"], hist, schema)
                next_steps = next_mask = None
                if (not row["done"]) and row.get("next_observation"):
                    next_hist = history_from_prior_steps(ep[:t]) + [
                        {
                            "observation": dict(row["observation"]),
                            "action": row["action"],
                            "reward": 0.0,
                            "done": False,
                        }
                    ]
                    next_steps, next_mask = build_sequence(
                        row["next_observation"], next_hist, schema
                    )
                self.samples.append(
                    {
                        "cur": (cur_steps, cur_mask),
                        "next": (next_steps, next_mask) if next_steps is not None else None,
                        "action_idx": ACTIONS.index(row["action"]),
                        "outcome_idx": outcome_to_index(row["info"]["outcome"]),
                        "reward": float(row["reward"]),
                        "done": bool(row["done"]),
                        "unhappy": row["observation"].get("customer_signal") == "unhappy",
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_transitions(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cur = collate_sequences([b["cur"] for b in batch])
    has_next = [b["next"] is not None for b in batch]
    next_batch = []
    for b in batch:
        if b["next"] is not None:
            next_batch.append(b["next"])
        else:
            next_batch.append(b["cur"])
    nxt = collate_sequences(next_batch)
    return {
        "cur": cur,
        "next": nxt,
        "has_next": torch.tensor(has_next, dtype=torch.bool),
        "action_idx": torch.tensor([b["action_idx"] for b in batch], dtype=torch.long),
        "outcome_idx": torch.tensor([b["outcome_idx"] for b in batch], dtype=torch.long),
        "reward": torch.tensor([b["reward"] for b in batch], dtype=torch.float32),
        "done": torch.tensor([b["done"] for b in batch], dtype=torch.bool),
        "unhappy": torch.tensor([b["unhappy"] for b in batch], dtype=torch.bool),
    }


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.05) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def compute_batch_losses(
    model: CausalTransformerPolicyNet,
    target: CausalTransformerPolicyNet,
    batch: Dict[str, Any],
    device: torch.device,
    gamma: float,
    reward_scale: float,
    bc_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur = {k: v.to(device) for k, v in batch["cur"].items()}
    nxt = {k: v.to(device) for k, v in batch["next"].items()}
    action_idx = batch["action_idx"].to(device)
    outcome_idx = batch["outcome_idx"].to(device)
    reward = batch["reward"].to(device)
    done = batch["done"].to(device)
    has_next = batch["has_next"].to(device)

    out = model(cur["cats"], cur["nums"], cur["actions"], cur["mask"])
    logits_a = out["outcome_logits"][torch.arange(len(action_idx), device=device), action_idx]
    loss_out = F.cross_entropy(logits_a, outcome_idx)
    loss_bc = F.cross_entropy(out["bc_logits"], action_idx)

    with torch.no_grad():
        tgt = target(nxt["cats"], nxt["nums"], nxt["actions"], nxt["mask"])
        support = bc_support_mask(tgt["bc_logits"], threshold=bc_thresh)
        q_next = tgt["q_values"].masked_fill(
            ~support, torch.finfo(tgt["q_values"].dtype).min / 4
        )
        max_next = q_next.max(dim=-1).values
        bootstrap = (~done) & has_next
        y = reward / reward_scale + gamma * bootstrap.float() * max_next

    q_pred = out["q_values"][torch.arange(len(action_idx), device=device), action_idx]
    loss_q = F.smooth_l1_loss(q_pred, y)
    return loss_out, loss_bc, loss_q


def evaluate_loader(
    model: CausalTransformerPolicyNet,
    target: CausalTransformerPolicyNet,
    loader: DataLoader,
    device: torch.device,
    gamma: float,
    reward_scale: float,
    bc_thresh: float,
    lam: float,
) -> Dict[str, float]:
    model.eval()
    total = total_out = total_bc = total_q = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            loss_out, loss_bc, loss_q = compute_batch_losses(
                model, target, batch, device, gamma, reward_scale, bc_thresh
            )
            loss = loss_out + loss_bc + lam * loss_q
            total += float(loss.item())
            total_out += float(loss_out.item())
            total_bc += float(loss_bc.item())
            total_q += float(loss_q.item())
            n_batches += 1
    denom = max(n_batches, 1)
    return {
        "loss": total / denom,
        "out": total_out / denom,
        "bc": total_bc / denom,
        "q": total_q / denom,
    }


def save_checkpoint(
    path: Path,
    model: CausalTransformerPolicyNet,
    schema: FeatureSchema,
    outcome_values: OutcomeValues,
    args: argparse.Namespace,
    gamma: float,
    lam: float,
    bc_thresh: float,
    reward_scale: float,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "schema": schema.to_dict(),
        "outcome_values": outcome_values.to_dict(),
        "config": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "gamma": gamma,
            "bc_threshold": bc_thresh,
            "lambda_q": lam,
            "reward_scale": reward_scale,
            "actions": list(ACTIONS),
            "outcome_classes": list(OUTCOME_CLASSES),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_episodes = load_episodes(Path(args.train_logs), limit_rows=args.limit_rows)
    schema = FeatureSchema()
    flat_rows = [r for ep in train_episodes for r in ep]
    outcome_values = compute_outcome_values(flat_rows)

    dataset = TransitionDataset(train_episodes, schema)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_transitions,
        num_workers=0,
    )

    valid_path = Path(args.valid_logs)
    valid_loader: Optional[DataLoader] = None
    if valid_path.exists():
        valid_episodes = load_episodes(valid_path, limit_rows=args.limit_rows)
        valid_dataset = TransitionDataset(valid_episodes, schema)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_transitions,
            num_workers=0,
        )
    else:
        print(f"warning: valid logs not found at {valid_path}; skipping validation")

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
    ckpt_path = out_dir / "transformer_fqi.pt"

    gamma = args.gamma
    lam = args.lambda_q
    bc_thresh = args.bc_threshold
    reward_scale = float(args.reward_scale)
    if reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")

    best_valid = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = total_out = total_bc = total_q = 0.0
        n_batches = 0
        for batch in loader:
            loss_out, loss_bc, loss_q = compute_batch_losses(
                model, target, batch, device, gamma, reward_scale, bc_thresh
            )
            loss = loss_out + loss_bc + lam * loss_q
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            soft_update(target, model, tau=args.tau)

            total += float(loss.item())
            total_out += float(loss_out.item())
            total_bc += float(loss_bc.item())
            total_q += float(loss_q.item())
            n_batches += 1

        denom = max(n_batches, 1)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={total/denom:.4f} "
            f"out={total_out/denom:.4f} "
            f"bc={total_bc/denom:.4f} "
            f"q={total_q/denom:.4f}"
        )

        if valid_loader is not None:
            metrics = evaluate_loader(
                model, target, valid_loader, device, gamma, reward_scale, bc_thresh, lam
            )
            print(
                f"epoch {epoch}/{args.epochs} "
                f"valid_loss={metrics['loss']:.4f} "
                f"out={metrics['out']:.4f} "
                f"bc={metrics['bc']:.4f} "
                f"q={metrics['q']:.4f}"
            )
            if metrics["loss"] < best_valid:
                best_valid = metrics["loss"]
                best_epoch = epoch
                save_checkpoint(
                    ckpt_path,
                    model,
                    schema,
                    outcome_values,
                    args,
                    gamma,
                    lam,
                    bc_thresh,
                    reward_scale,
                )
                print(f"  saved best checkpoint (valid_loss={best_valid:.4f})")

    if valid_loader is None:
        save_checkpoint(
            ckpt_path,
            model,
            schema,
            outcome_values,
            args,
            gamma,
            lam,
            bc_thresh,
            reward_scale,
        )
        print(f"saved {ckpt_path}")
    else:
        print(
            f"best epoch={best_epoch} valid_loss={best_valid:.4f} "
            f"(left {ckpt_path} unchanged after last epoch)"
        )

    outcome_values.save(out_dir / "outcome_values.json")
    (out_dir / "schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2), encoding="utf-8"
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-logs", default=str(ROOT / "data" / "train_logs.jsonl"))
    p.add_argument("--valid-logs", default=str(ROOT / "data" / "valid_logs.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "models"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lambda-q", type=float, default=1.0)
    p.add_argument(
        "--reward-scale",
        type=float,
        default=100.0,
        help="Divide rewards for Q targets; act uses Q * reward_scale",
    )
    p.add_argument("--bc-threshold", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit-rows", type=int, default=0, help="0 = all rows")
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()

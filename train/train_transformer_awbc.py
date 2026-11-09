"""Train advantage-weighted behavior cloning (AWBC) transformer policy."""

from __future__ import annotations

import argparse
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
from policies.transformer_model import CausalTransformerAWBCNet

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
    return [sorted(by[eid], key=lambda r: r["step"]) for eid in order]


def returns_to_go(rewards: Sequence[float], gamma: float) -> List[float]:
    """Discounted return-to-go: G_t = r_t + gamma * G_{t+1}."""
    g = 0.0
    out: List[float] = [0.0] * len(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        g = float(rewards[t]) + float(gamma) * g
        out[t] = g
    return out


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


class AWBCDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[Sequence[Dict[str, Any]]],
        schema: FeatureSchema,
        gamma: float,
    ) -> None:
        self.schema = schema
        self.samples: List[Dict[str, Any]] = []
        for ep in episodes:
            rewards = [float(r["reward"]) for r in ep]
            gt = returns_to_go(rewards, gamma)
            for t, row in enumerate(ep):
                hist = history_from_prior_steps(ep[:t])
                cur_steps, cur_mask = build_sequence(row["observation"], hist, schema)
                self.samples.append(
                    {
                        "cur": (cur_steps, cur_mask),
                        "action_idx": ACTIONS.index(row["action"]),
                        "return_to_go": gt[t],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_awbc(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cur = collate_sequences([b["cur"] for b in batch])
    return {
        "cur": cur,
        "action_idx": torch.tensor([b["action_idx"] for b in batch], dtype=torch.long),
        "return_to_go": torch.tensor(
            [b["return_to_go"] for b in batch], dtype=torch.float32
        ),
    }


def compute_batch_losses(
    model: CausalTransformerAWBCNet,
    batch: Dict[str, Any],
    device: torch.device,
    reward_scale: float,
    beta: float,
    w_max: float,
    lambda_v: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur = {k: v.to(device) for k, v in batch["cur"].items()}
    action_idx = batch["action_idx"].to(device)
    g = batch["return_to_go"].to(device) / reward_scale

    out = model(cur["cats"], cur["nums"], cur["actions"], cur["mask"])
    v = out["v"]
    loss_v = F.smooth_l1_loss(v, g)

    adv = g - v.detach()
    w = torch.exp(adv / beta).clamp(max=w_max)
    ce = F.cross_entropy(out["pi_logits"], action_idx, reduction="none")
    loss_pi = (w * ce).mean()
    loss = loss_pi + lambda_v * loss_v
    return loss, loss_pi, loss_v


def evaluate_loader(
    model: CausalTransformerAWBCNet,
    loader: DataLoader,
    device: torch.device,
    reward_scale: float,
    beta: float,
    w_max: float,
    lambda_v: float,
) -> Dict[str, float]:
    model.eval()
    total = total_pi = total_v = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            loss, loss_pi, loss_v = compute_batch_losses(
                model, batch, device, reward_scale, beta, w_max, lambda_v
            )
            total += float(loss.item())
            total_pi += float(loss_pi.item())
            total_v += float(loss_v.item())
            n_batches += 1
    denom = max(n_batches, 1)
    return {
        "loss": total / denom,
        "pi": total_pi / denom,
        "v": total_v / denom,
    }


def save_checkpoint(
    path: Path,
    model: CausalTransformerAWBCNet,
    schema: FeatureSchema,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "schema": schema.to_dict(),
        "config": {
            "kind": "awbc",
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "gamma": args.gamma,
            "beta": args.beta,
            "w_max": args.w_max,
            "lambda_v": args.lambda_v,
            "reward_scale": args.reward_scale,
            "actions": list(ACTIONS),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")
    if args.beta <= 0:
        raise ValueError("--beta must be positive")

    schema = FeatureSchema()
    train_episodes = load_episodes(Path(args.train_logs), limit_rows=args.limit_rows)
    train_ds = AWBCDataset(train_episodes, schema, gamma=args.gamma)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_awbc,
        num_workers=0,
    )

    valid_loader: Optional[DataLoader] = None
    valid_path = Path(args.valid_logs)
    if valid_path.exists():
        valid_ds = AWBCDataset(
            load_episodes(valid_path, limit_rows=args.limit_rows),
            schema,
            gamma=args.gamma,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_awbc,
            num_workers=0,
        )
    else:
        print(f"warning: valid logs not found at {valid_path}; skipping validation")

    model = CausalTransformerAWBCNet(
        schema, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "transformer_awbc.pt"

    best_valid = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = total_pi = total_v = 0.0
        n_batches = 0
        for batch in loader:
            loss, loss_pi, loss_v = compute_batch_losses(
                model,
                batch,
                device,
                args.reward_scale,
                args.beta,
                args.w_max,
                args.lambda_v,
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            total_pi += float(loss_pi.item())
            total_v += float(loss_v.item())
            n_batches += 1

        denom = max(n_batches, 1)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={total/denom:.4f} "
            f"pi={total_pi/denom:.4f} "
            f"v={total_v/denom:.4f}"
        )

        if valid_loader is not None:
            metrics = evaluate_loader(
                model,
                valid_loader,
                device,
                args.reward_scale,
                args.beta,
                args.w_max,
                args.lambda_v,
            )
            print(
                f"epoch {epoch}/{args.epochs} "
                f"valid_loss={metrics['loss']:.4f} "
                f"pi={metrics['pi']:.4f} "
                f"v={metrics['v']:.4f}"
            )
            if metrics["loss"] < best_valid:
                best_valid = metrics["loss"]
                best_epoch = epoch
                save_checkpoint(ckpt_path, model, schema, args)
                print(f"  saved best checkpoint (valid_loss={best_valid:.4f})")

    if valid_loader is None:
        save_checkpoint(ckpt_path, model, schema, args)
        print(f"saved {ckpt_path}")
    else:
        print(
            f"best epoch={best_epoch} valid_loss={best_valid:.4f} "
            f"(left {ckpt_path} unchanged after last epoch)"
        )

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
    p.add_argument("--beta", type=float, default=1.0, help="Advantage temperature")
    p.add_argument("--w-max", type=float, default=20.0, help="Max advantage weight")
    p.add_argument("--lambda-v", type=float, default=1.0)
    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit-rows", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    train(build_argparser().parse_args())


if __name__ == "__main__":
    main()

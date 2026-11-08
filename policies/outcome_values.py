"""Terminal outcome value table. continue / max_steps use logged r, not this table."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

OUTCOME_CLASSES = (
    "continue",
    "max_steps",
    "sold_estimate",
    "booked_appointment",
    "membership_converted",
    "positive_review",
    "opted_out",
    "escalated",
    "parked",
)

# Used in V_imm at decision time (not continue / max_steps)
TABLE_OUTCOMES = (
    "sold_estimate",
    "booked_appointment",
    "membership_converted",
    "positive_review",
    "opted_out",
    "escalated",
    "parked",
)

CONTINUE_IDX = OUTCOME_CLASSES.index("continue")
MAX_STEPS_IDX = OUTCOME_CLASSES.index("max_steps")


class OutcomeValues:
    def __init__(self, values: Mapping[str, float], escalate_unhappy: float, escalate_other: float):
        self.values = dict(values)
        self.escalate_unhappy = float(escalate_unhappy)
        self.escalate_other = float(escalate_other)

    def value_for(self, outcome: str, unhappy: bool = False) -> float:
        if outcome == "escalated":
            return self.escalate_unhappy if unhappy else self.escalate_other
        if outcome in ("continue", "max_steps"):
            raise KeyError(f"{outcome} has no fixed table value; use logged reward")
        return float(self.values[outcome])

    def tensor_table(self, unhappy_flags) -> "Any":
        """Return [B, C] table values; continue/max_steps slots are 0 (unused in V_imm)."""
        import torch

        if not torch.is_tensor(unhappy_flags):
            unhappy_flags = torch.tensor(unhappy_flags, dtype=torch.bool)
        b = unhappy_flags.shape[0]
        table = torch.zeros(b, len(OUTCOME_CLASSES), dtype=torch.float32)
        for i, name in enumerate(OUTCOME_CLASSES):
            if name in ("continue", "max_steps"):
                continue
            if name == "escalated":
                table[:, i] = torch.where(
                    unhappy_flags,
                    torch.full((), self.escalate_unhappy),
                    torch.full((), self.escalate_other),
                )
            else:
                table[:, i] = self.values.get(name, 0.0)
        return table

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "escalate_unhappy": self.escalate_unhappy,
            "escalate_other": self.escalate_other,
            "outcome_classes": list(OUTCOME_CLASSES),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutcomeValues":
        return cls(
            values=data["values"],
            escalate_unhappy=data["escalate_unhappy"],
            escalate_other=data["escalate_other"],
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OutcomeValues":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def compute_outcome_values(rows: Iterable[Dict[str, Any]]) -> OutcomeValues:
    buckets: Dict[str, List[float]] = defaultdict(list)
    esc_u: List[float] = []
    esc_o: List[float] = []
    for row in rows:
        outcome = row["info"]["outcome"]
        rew = float(row["reward"])
        if outcome == "escalated":
            if row["observation"].get("customer_signal") == "unhappy":
                esc_u.append(rew)
            else:
                esc_o.append(rew)
        else:
            buckets[outcome].append(rew)

    values = {}
    for name in TABLE_OUTCOMES:
        if name == "escalated":
            continue
        xs = buckets.get(name) or [0.0]
        values[name] = sum(xs) / len(xs)
    return OutcomeValues(
        values=values,
        escalate_unhappy=(sum(esc_u) / len(esc_u)) if esc_u else 20.0,
        escalate_other=(sum(esc_o) / len(esc_o)) if esc_o else -18.0,
    )


def outcome_to_index(name: str) -> int:
    return OUTCOME_CLASSES.index(name)

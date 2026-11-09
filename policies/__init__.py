"""Extensible follow-up policies.

Run side-by-side offline comparison from the package root:

    python -m eval.compare_policies \\
      --logs data/valid_logs.jsonl \\
      --policies product_rule,simple_rule,always_estimate_nudge

Smoke-test the default product rule:

    python starter/check_policy.py --policy policies/default_policy.py

Train the causal transformer FQI policy:

    python -m train.train_transformer_fqi

Smoke-test ML policy (after training):

    python starter/check_policy.py --policy policies/default_ml_policy.py

Compare:

    python -m eval.compare_policies \\
      --policies product_rule,simple_rule,transformer_bc,transformer_fqi
"""

from policies.base import ACTIONS, ACTIONS_SET, PolicyFn
from policies.registry import get, list_policies, register
from policies import rules as _rules  # noqa: F401  # register baselines + product_rule
from policies.rules import product_rule

try:
    from policies import transformer_policy as _transformer_policy  # noqa: F401
except ImportError:
    # Torch not installed; rule policies still work.
    _transformer_policy = None  # type: ignore

__all__ = [
    "ACTIONS",
    "ACTIONS_SET",
    "PolicyFn",
    "get",
    "list_policies",
    "register",
    "product_rule",
]

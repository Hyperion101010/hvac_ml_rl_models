"""Local default policy for starter/check_policy.py --policy."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from policies.rules import product_rule


def act(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return product_rule(observation, history)

"""Drop-in policy module for starter/check_policy.py --policy.

Exposes act(observation, history) only — pluggable signature.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from policies.transformer_policy import transformer_fqi_act


def act(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    return transformer_fqi_act(observation, history)

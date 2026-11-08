"""Name -> act() registry for side-by-side and future ML policies."""

from __future__ import annotations

from typing import Dict, List

from policies.base import PolicyFn

_REGISTRY: Dict[str, PolicyFn] = {}


def register(name: str, fn: PolicyFn | None = None):
    """Register a policy. Use as @register("name") or register("name", fn)."""

    def decorator(policy_fn: PolicyFn) -> PolicyFn:
        if name in _REGISTRY:
            raise ValueError(f"Policy already registered: {name}")
        _REGISTRY[name] = policy_fn
        return policy_fn

    if fn is not None:
        return decorator(fn)
    return decorator


def get(name: str) -> PolicyFn:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown policy {name!r}. Known: {known}")
    return _REGISTRY[name]


def list_policies() -> List[str]:
    return sorted(_REGISTRY)

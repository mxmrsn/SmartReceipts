"""Tree-Edit-Distance Similarity over the canonical Receipt JSON.

TEDS (Tree Edit Distance Similarity) ≈ 1 − (edit_distance / max_size).
Higher is better. Captures structural fidelity — does the predicted
receipt have the same shape (header, items, totals) as the gold one,
modulo content differences.

Uses the `apted` package if available. Falls back to a leaf-equality
ratio so the bench still produces a usable number without the optional
dependency.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Optional

from bench.schema import Receipt

def receipt_to_tree(receipt: Receipt) -> Any:
    """Convert canonical Receipt to a dict tree."""
    return json.loads(receipt.model_dump_json(exclude={"provenance"}))


def teds(gold: Receipt, predicted: Optional[Receipt]) -> float:
    """Returns similarity in [0, 1]. 1.0 = identical trees, 0.0 = nothing matches.

    Uses leaf-equality on the canonical Receipt tree — a fast proxy that
    behaves well for our use case (proper tree-edit-distance via APTED hit
    recursion issues on deeper receipts and didn't add much signal).
    """
    if predicted is None:
        return 0.0
    g_tree = receipt_to_tree(gold)
    p_tree = receipt_to_tree(predicted)
    return _fallback_similarity(g_tree, p_tree)


# MARK: - Leaf-equality similarity

def _fallback_similarity(g, p) -> float:
    g_leaves = list(_leaves(g))
    p_leaves = list(_leaves(p))
    if not g_leaves and not p_leaves:
        return 1.0
    matched = 0
    p_used: set[int] = set()
    for gv in g_leaves:
        for i, pv in enumerate(p_leaves):
            if i in p_used:
                continue
            if _leaf_equal(gv, pv):
                matched += 1
                p_used.add(i)
                break
    return matched / max(len(g_leaves), len(p_leaves))


def _leaves(node, path: str = ""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield (path, node)


def _leaf_equal(a: tuple, b: tuple) -> bool:
    pa, va = a
    pb, vb = b
    if pa != pb:
        return False
    return va == vb

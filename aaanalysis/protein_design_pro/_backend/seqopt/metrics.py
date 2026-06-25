"""
This is a script for the backend of the SeqOpt Pareto-quality metrics: hypervolume (the
volume of objective space a front dominates relative to a reference point) and spread (the
objective-space diversity of the front). Operates on maximization-normalized objectives.
"""
from typing import Optional
import numpy as np


# I Helper Functions
def _pareto_mask(W) -> np.ndarray:
    """Boolean mask of the non-dominated rows of a maximization-normalized objective array."""
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(W[j] >= W[i]) and np.any(W[j] > W[i]):
                keep[i] = False
                break
    return keep


def _hypervolume_2d(P, ref) -> float:
    """Exact 2-D hypervolume of a maximization front above the reference point.

    For a non-dominated maximization front (x increasing => y decreasing), the dominated
    region is the union of rectangles ``[ref0, x_i] x [ref1, y_i]``, whose area collapses
    to ``sum_i (x_i - x_{i-1}) * (y_i - ref1)`` with ``x_0 = ref0``.
    """
    pts = P[(P[:, 0] >= ref[0]) & (P[:, 1] >= ref[1])]
    if len(pts) == 0:
        return 0.0
    order = np.argsort(pts[:, 0], kind="mergesort")   # x ascending
    pts = pts[order]
    hv = 0.0
    prev_x = ref[0]
    for x, y in pts:
        hv += (x - prev_x) * (y - ref[1])
        prev_x = x
    return float(max(hv, 0.0))


# II Main Functions
def hypervolume(W, ref: Optional[np.ndarray] = None) -> float:
    """Hypervolume dominated by the front (maximization-normalized; ND via inclusion grid).

    Parameters
    ----------
    W : array-like, shape (n, m)
        Maximization-normalized objective values of the front (or full population).
    ref : np.ndarray, shape (m,), optional
        Reference (nadir) point. Defaults to the per-objective minimum minus a small margin.

    Returns
    -------
    hv : float
        Dominated hypervolume above ``ref`` (0 when the front is empty/degenerate).
    """
    W = np.asarray(W, dtype=float)
    if W.size == 0:
        return 0.0
    P = W[_pareto_mask(W)]
    m = P.shape[1]
    if ref is None:
        ref = W.min(axis=0) - 1e-9
    ref = np.asarray(ref, dtype=float)
    if m == 1:
        return float(max(P[:, 0].max() - ref[0], 0.0))
    if m == 2:
        return _hypervolume_2d(P, ref)
    # m >= 3: Monte-Carlo-free grid inclusion estimate over the bounding box.
    hi = P.max(axis=0)
    box = np.prod(np.maximum(hi - ref, 0.0))
    if box <= 0:
        return 0.0
    grid = np.linspace(0, 1, 12)
    mesh = np.array(np.meshgrid(*[grid] * m)).reshape(m, -1).T
    samples = ref + mesh * (hi - ref)
    dominated = np.zeros(len(samples), dtype=bool)
    for p in P:
        dominated |= np.all(samples <= p, axis=1)
    return float(box * dominated.mean())


def spread(W) -> float:
    """Objective-space diversity of a front: mean pairwise Euclidean distance (0 if <2 points)."""
    W = np.asarray(W, dtype=float)
    P = W[_pareto_mask(W)]
    n = len(P)
    if n < 2:
        return 0.0
    rng = P.max(axis=0) - P.min(axis=0)
    rng[rng == 0] = 1.0
    Pn = P / rng
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(Pn[i] - Pn[j]))
            count += 1
    return total / count

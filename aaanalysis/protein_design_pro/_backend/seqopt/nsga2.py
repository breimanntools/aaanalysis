"""
This is a script for the backend of the SeqOpt NSGA-II selection core: fast non-dominated
sorting, crowding-distance assignment, dominated-crowding tournament (mating selection) and
(mu+lambda) survival. Pure-Python / numpy; it operates only on objective-value arrays and is
independent of the sequence/genome encoding, so it is the unit that mirrors DEAP's selNSGA2
[Deb02]_ and is checked for equivalence against it.
"""
from typing import List, Tuple
import numpy as np

import aaanalysis.utils as ut


# I Helper Functions
def normalize_objectives_(F, goals):
    """Flip every objective to a maximization sense so one dominance rule covers all.

    Parameters
    ----------
    F : array-like, shape (n, m)
        Raw objective values (n variants, m objectives).
    goals : list of str
        Per-objective goal, ``"max"`` or ``"min"``.

    Returns
    -------
    W : np.ndarray, shape (n, m)
        Objective values with every ``"min"`` column negated, so larger is always better
        (DEAP's weighted ``wvalues`` convention with weights +1 / -1).
    """
    F = np.asarray(F, dtype=float)
    signs = np.array([1.0 if g == ut.LIST_OBJECTIVE_GOALS[0] else -1.0 for g in goals])
    return F * signs


def _dominates(w_a, w_b):
    """Return True if maximization row ``w_a`` Pareto-dominates ``w_b``."""
    return bool(np.all(w_a >= w_b) and np.any(w_a > w_b))


# II Main Functions
def fast_non_dominated_sort(W) -> Tuple[List[List[int]], np.ndarray]:
    """Fast non-dominated sort of maximization-normalized objectives (Deb et al. 2002).

    Parameters
    ----------
    W : array-like, shape (n, m)
        Maximization-normalized objective values (see :func:`normalize_objectives_`).

    Returns
    -------
    fronts : list of list of int
        Index groups by front; ``fronts[0]`` is the non-dominated (best) front. Indices
        keep ascending order within a front (deterministic tie-break).
    rank : np.ndarray, shape (n,)
        Per-variant front index (0 = best).
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    dominated_by = [[] for _ in range(n)]   # solutions that i dominates
    n_dominating = np.zeros(n, dtype=int)   # how many dominate i
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(W[i], W[j]):
                dominated_by[i].append(j)
                n_dominating[j] += 1
            elif _dominates(W[j], W[i]):
                dominated_by[j].append(i)
                n_dominating[i] += 1
    fronts: List[List[int]] = []
    current = [i for i in range(n) if n_dominating[i] == 0]
    rank = np.zeros(n, dtype=int)
    f = 0
    while current:
        fronts.append(sorted(current))
        nxt = []
        for i in current:
            for j in dominated_by[i]:
                n_dominating[j] -= 1
                if n_dominating[j] == 0:
                    rank[j] = f + 1
                    nxt.append(j)
        current = nxt
        f += 1
    return fronts, rank


def crowding_distance(W, front) -> np.ndarray:
    """Assign NSGA-II crowding distances to the members of one front.

    Parameters
    ----------
    W : array-like, shape (n, m)
        Maximization-normalized objective values.
    front : list of int
        Indices (into ``W``) of the front's members.

    Returns
    -------
    dist : np.ndarray, shape (len(front),)
        Crowding distance per member, aligned to ``front``; boundary points get ``inf``.
    """
    W = np.asarray(W, dtype=float)
    front = list(front)
    n = len(front)
    dist = np.zeros(n, dtype=float)
    if n == 0:
        return dist
    if n <= 2:
        dist[:] = np.inf
        return dist
    m = W.shape[1]
    sub = W[front]
    for k in range(m):
        order = np.argsort(sub[:, k], kind="mergesort")   # stable, deterministic
        lo, hi = sub[order[0], k], sub[order[-1], k]
        # DEAP normalizes each objective by ``nobj * (max - min)`` (assignCrowdingDist), so the
        # crowding values are byte-comparable with the reference oracle, not just rank-equal.
        norm = m * (hi - lo)
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        if norm == 0:
            continue
        for r in range(1, n - 1):
            prev_v = sub[order[r - 1], k]
            next_v = sub[order[r + 1], k]
            dist[order[r]] += (next_v - prev_v) / norm
    return dist


def crowded_better(rank_a, crowd_a, rank_b, crowd_b) -> bool:
    """Crowded-comparison operator: True if A is preferred over B (lower rank, else more spread)."""
    if rank_a != rank_b:
        return rank_a < rank_b
    return crowd_a > crowd_b


def dcd_tournament(rank, crowding, n_select, rng) -> List[int]:
    """Binary dominated-crowding tournament for mating selection (selTournamentDCD analogue).

    Parameters
    ----------
    rank : array-like, shape (n,)
        Non-dominated front index per individual.
    crowding : array-like, shape (n,)
        Crowding distance per individual.
    n_select : int
        Number of parent indices to draw (with replacement across tournaments).
    rng : random.Random
        Seeded Python RNG (the same stream the DEAP oracle consumes).

    Returns
    -------
    chosen : list of int
        Selected parent indices, length ``n_select``.
    """
    rank = np.asarray(rank)
    crowding = np.asarray(crowding)
    n = len(rank)
    chosen = []
    for _ in range(n_select):
        a = rng.randrange(n)
        b = rng.randrange(n)
        if crowded_better(rank[a], crowding[a], rank[b], crowding[b]):
            chosen.append(a)
        elif crowded_better(rank[b], crowding[b], rank[a], crowding[a]):
            chosen.append(b)
        else:
            chosen.append(a if rng.random() < 0.5 else b)
    return chosen


def select_nsga2(W, mu) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Select the ``mu`` survivors by NSGA-II (front order, crowding tie-break).

    Parameters
    ----------
    W : array-like, shape (n, m)
        Maximization-normalized objectives of the combined parent+offspring pool.
    mu : int
        Number of survivors to keep.

    Returns
    -------
    survivors : list of int
        Indices (into ``W``) of the kept individuals, ordered by (rank asc, crowding desc).
    rank : np.ndarray, shape (n,)
        Front index of every pool member.
    crowding : np.ndarray, shape (n,)
        Crowding distance of every pool member.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    fronts, rank = fast_non_dominated_sort(W)
    crowding = np.zeros(n, dtype=float)
    for front in fronts:
        d = crowding_distance(W, front)
        for idx, member in enumerate(front):
            crowding[member] = d[idx]
    survivors: List[int] = []
    for front in fronts:
        if len(survivors) + len(front) <= mu:
            survivors.extend(front)
        else:
            remaining = mu - len(survivors)
            # Fill the partial front by descending crowding (stable on ties by index).
            order = sorted(front, key=lambda i: (-crowding[i], i))
            survivors.extend(order[:remaining])
            break
    return survivors, rank, crowding


# III Fast (vectorized) engine — numerically identical fronts, numpy-vectorized for speed
def fast_non_dominated_sort_vec(W) -> Tuple[List[List[int]], np.ndarray]:
    """Vectorized fast non-dominated sort (``engine='fast'``).

    Produces the **same** fronts and ranks as :func:`fast_non_dominated_sort` (same dominance
    relation, same ascending-index order within a front) by computing the full pairwise
    dominance matrix with numpy broadcasting instead of a Python double loop.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    if n == 0:
        return [], np.zeros(0, dtype=int)
    ge = np.all(W[:, None, :] >= W[None, :, :], axis=2)
    gt = np.any(W[:, None, :] > W[None, :, :], axis=2)
    dom = ge & gt                       # dom[i, j] == True  <=>  i dominates j
    remaining = dom.sum(axis=0)         # how many points dominate j
    rank = np.zeros(n, dtype=int)
    placed = np.zeros(n, dtype=bool)
    fronts: List[List[int]] = []
    current = np.where(remaining == 0)[0]
    f = 0
    while current.size:
        rank[current] = f
        placed[current] = True
        fronts.append(sorted(current.tolist()))
        remaining = remaining - dom[current].sum(axis=0)
        current = np.where((remaining == 0) & (~placed))[0]
        f += 1
    return fronts, rank


def rank_and_crowding(W, engine="exact") -> Tuple[np.ndarray, np.ndarray]:
    """Assign rank + crowding to every row, via the exact or the fast (vectorized) sort.

    Both engines return identical ``rank`` and ``crowding`` (the fast path only vectorizes the
    O(n^2) dominance scan); ``engine='fast'`` is a speed option, not a different result.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    if engine == ut.LIST_SEQOPT_ENGINE[1]:      # "fast"
        fronts, rank = fast_non_dominated_sort_vec(W)
    else:                                       # "exact"
        fronts, rank = fast_non_dominated_sort(W)
    crowding = np.zeros(n, dtype=float)
    for front in fronts:
        d = crowding_distance(W, front)
        for idx, member in enumerate(front):
            crowding[member] = d[idx]
    return rank, crowding


def select_nsga2_engine(W, mu, engine="exact") -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Engine-aware NSGA-II survival selection (exact or vectorized sort, identical result).

    The fast path fills survivors **front-by-front in the same order** as :func:`select_nsga2`
    (ascending index within a front, partial front by descending crowding) so the survivor
    list -- not just its set -- is identical to the exact engine; population order drives the
    downstream RNG, so any order difference would otherwise cascade into different offspring.
    """
    if engine != ut.LIST_SEQOPT_ENGINE[1]:              # "exact"
        return select_nsga2(W, mu)
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    fronts, rank = fast_non_dominated_sort_vec(W)
    crowding = np.zeros(n, dtype=float)
    for front in fronts:
        d = crowding_distance(W, front)
        for idx, member in enumerate(front):
            crowding[member] = d[idx]
    survivors: List[int] = []
    for front in fronts:
        if len(survivors) + len(front) <= mu:
            survivors.extend(front)
        else:
            remaining = mu - len(survivors)
            order = sorted(front, key=lambda i: (-crowding[i], i))
            survivors.extend(order[:remaining])
            break
    return survivors, rank, crowding

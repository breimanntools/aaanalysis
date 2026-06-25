"""
This is a script for the backend of SeqOpt constraint handling: it penalizes the objective
rows of infeasible variants so the search avoids them, mirroring DEAP's ``DeltaPenalty``
(fixed worst objective for any infeasible individual) and ``ClosestValidPenalty`` (penalty
scaled by how many constraints are violated). Pure-Python / numpy.
"""
from typing import Callable, Dict, List
import numpy as np

import aaanalysis.utils as ut


# I Helper Functions
def constraint_violation(genome: Dict[int, str], constraints: List[Callable]) -> int:
    """Count violated feasibility constraints (0 = feasible). Each callable maps genome->bool."""
    return sum(1 for c in constraints if not bool(c(genome)))


# II Main Functions
def apply_penalty(F, genomes, constraints, goals, penalty="delta"):
    """Degrade the objective rows of infeasible variants so they are dominated.

    Parameters
    ----------
    F : array-like, shape (n, m)
        Raw objective matrix (n variants, m objectives), in each objective's own goal sense.
    genomes : list of dict
        The variants aligned to ``F`` rows.
    constraints : list of callable
        Feasibility predicates ``genome -> bool`` (True = feasible).
    goals : list of str
        Per-objective ``"max"`` / ``"min"``.
    penalty : str, default="delta"
        ``"delta"`` pushes every infeasible variant to a single worst value per objective;
        ``"closest_valid"`` scales the push by the number of violated constraints (the variant
        that is "less infeasible" is penalized less).

    Returns
    -------
    F_pen : np.ndarray, shape (n, m)
        Copy of ``F`` with infeasible rows penalized (feasible rows untouched). When every
        variant is infeasible the relative ``closest_valid`` ordering is still meaningful.
    """
    if not constraints:
        return np.asarray(F, dtype=float)
    F = np.array(F, dtype=float, copy=True)
    n, m = F.shape
    violations = np.array([constraint_violation(g, constraints) for g in genomes], dtype=int)
    feasible = violations == 0
    ref = F[feasible] if feasible.any() else F
    for c, goal in enumerate(goals):
        col = F[:, c]
        span = float(np.ptp(col)) or 1.0
        if goal == ut.LIST_OBJECTIVE_GOALS[0]:          # "max" — worse is smaller
            base = float(ref[:, c].min())
            step = span
            for r in np.where(~feasible)[0]:
                scale = violations[r] if penalty == ut.LIST_SEQOPT_PENALTY[1] else 1
                F[r, c] = base - step * scale
        else:                                           # "min" — worse is larger
            base = float(ref[:, c].max())
            step = span
            for r in np.where(~feasible)[0]:
                scale = violations[r] if penalty == ut.LIST_SEQOPT_PENALTY[1] else 1
                F[r, c] = base + step * scale
    return F

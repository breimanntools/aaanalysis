"""
This is a script for the backend of the SeqOpt evolution loops: the NSGA-II generational
loop (mating selection -> variation -> (mu+lambda) survival) and the importance-ordered
greedy walk. Both take injected ``fitness_fn`` (genomes -> objective matrix) and ``guide_fn``
(population -> per-position importance weights) callbacks, so this module never imports SeqMut
or ShapModel and stays a pure, dedicated SeqOpt backend.
"""
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

import aaanalysis.utils as ut
from .nsga2 import (normalize_objectives_, fast_non_dominated_sort,
                    dcd_tournament, select_nsga2, rank_and_crowding)
from .genome import (crossover_uniform, crossover_npoint, mutate, init_population, canonical)
from .metrics import hypervolume, spread


# I Helper Functions
def _crossover(g1, g2, crossover, n_mut_max, rng, cx_indpb=0.5):
    """Dispatch the configured crossover operator over two genomes."""
    if crossover == ut.LIST_SEQOPT_CROSSOVER[0]:        # "uniform"
        return crossover_uniform(g1, g2, n_mut_max, rng, cx_prob=cx_indpb)
    n_points = 1 if crossover == ut.LIST_SEQOPT_CROSSOVER[1] else 2   # one_point / two_point
    return crossover_npoint(g1, g2, n_mut_max, rng, n_points=n_points)


def _var_and(parents, wt_seq, positions, alphabet, n_mut_max, rng, cx_prob, mut_prob,
             crossover, mutation, weights) -> List[Dict[int, str]]:
    """DEAP ``varAnd`` analogue: crossover consecutive pairs (prob cx_prob), then mutate each."""
    off = [dict(p) for p in parents]
    for i in range(1, len(off), 2):
        if rng.random() < cx_prob:
            off[i - 1], off[i] = _crossover(off[i - 1], off[i], crossover, n_mut_max, rng)
    for i in range(len(off)):
        if rng.random() < mut_prob:
            off[i] = mutate(off[i], wt_seq, positions, alphabet, n_mut_max, rng,
                            mutation=mutation, weights=weights)
    return off


def _var_or(parents, wt_seq, positions, alphabet, n_mut_max, rng, cx_prob, mut_prob,
            crossover, mutation, weights, lambda_) -> List[Dict[int, str]]:
    """DEAP ``varOr`` analogue: each offspring is ONE of crossover / mutation / reproduction."""
    off: List[Dict[int, str]] = []
    for _ in range(lambda_):
        op = rng.random()
        if op < cx_prob:                                # crossover two random parents
            i, j = rng.randrange(len(parents)), rng.randrange(len(parents))
            c1, _c2 = _crossover(dict(parents[i]), dict(parents[j]), crossover, n_mut_max, rng)
            off.append(c1)
        elif op < cx_prob + mut_prob:                   # mutate one random parent
            i = rng.randrange(len(parents))
            off.append(mutate(dict(parents[i]), wt_seq, positions, alphabet, n_mut_max, rng,
                              mutation=mutation, weights=weights))
        else:                                           # reproduction (copy)
            off.append(dict(parents[rng.randrange(len(parents))]))
    return off


def _variation(parents, variation, wt_seq, positions, alphabet, n_mut_max, rng, cx_prob,
               mut_prob, crossover, mutation, weights, lambda_):
    """Dispatch varAnd / varOr."""
    if variation == ut.LIST_SEQOPT_VARIATION[1]:        # "or"
        return _var_or(parents, wt_seq, positions, alphabet, n_mut_max, rng, cx_prob, mut_prob,
                       crossover, mutation, weights, lambda_)
    return _var_and(parents, wt_seq, positions, alphabet, n_mut_max, rng, cx_prob, mut_prob,
                    crossover, mutation, weights)


def _update_hof(hof, genomes, F, goals, k):
    """Update a single-objective Hall of Fame (best-k by the primary objective, deduped)."""
    W = normalize_objectives_(F, goals)
    for g, primary in zip(genomes, W[:, 0]):
        hof[canonical(g)] = (float(primary), g)
    best = sorted(hof.values(), key=lambda t: -t[0])[:k]
    return dict((canonical(g), (p, g)) for p, g in best)


def _per_obj_best(F, goals, rank) -> List[float]:
    """Best raw value per objective on the current rank-0 front (in each objective's goal sense)."""
    F = np.asarray(F, dtype=float)
    front = F[np.asarray(rank) == 0]
    if len(front) == 0:
        front = F
    out = []
    for j, goal in enumerate(goals):
        col = front[:, j]
        out.append(float(col.max() if goal == ut.LIST_OBJECTIVE_GOALS[0] else col.min()))
    return out


def _per_obj_worst(F, goals) -> List[float]:
    """Worst raw value per objective over the population (the opposite extreme of the goal)."""
    F = np.asarray(F, dtype=float)
    return [float(F[:, j].min() if g == ut.LIST_OBJECTIVE_GOALS[0] else F[:, j].max())
            for j, g in enumerate(goals)]


def _update_archive(archive, genomes, F, goals):
    """Keep the running cumulative non-dominated set (DEAP ParetoFront analogue), unbounded.

    Adds the newly-evaluated genomes, then prunes the archive back to its non-dominated members
    so no best-ever solution is lost to per-generation crowding truncation.
    """
    for g, row in zip(genomes, np.asarray(F, dtype=float)):
        archive[canonical(g)] = (g, row)
    keys = list(archive)
    Wa = normalize_objectives_(np.array([archive[k][1] for k in keys], dtype=float), goals)
    _, rank = fast_non_dominated_sort(Wa)
    return {keys[i]: archive[keys[i]] for i in range(len(keys)) if rank[i] == 0}


def _front_rank_crowding(W) -> Tuple[np.ndarray, np.ndarray]:
    """Assign every row its non-dominated rank and crowding distance."""
    return rank_and_crowding(W)


# II Main Functions
def evolve_nsga2(wt_seq: str,
                 positions: List[int],
                 alphabet: List[str],
                 goals: List[str],
                 fitness_fn: Callable[[List[Dict[int, str]]], np.ndarray],
                 guide_fn: Callable[[Optional[List[Dict[int, str]]]], Optional[dict]],
                 rng,
                 pop_size: int = 50,
                 n_gen: int = 20,
                 n_mut_max: int = 5,
                 crossover: str = "uniform",
                 mutation: str = "substitution",
                 cx_prob: float = 0.5,
                 mut_prob: float = 0.2,
                 survival: str = "mu_plus_lambda",
                 variation: str = "and",
                 hof_size: int = 10,
                 suggest_seeds: Optional[List[Dict[int, str]]] = None,
                 ) -> dict:
    """Run the NSGA-II generational loop and return the final population + objectives + trace.

    Returns a dict with ``genomes`` (final population), ``F`` (raw objective matrix), ``rank``,
    ``crowding``, ``trajectory`` (per-generation hypervolume) and ``hall_of_fame`` (best-k
    single-objective genomes). ``variation`` selects varAnd/varOr; ``survival`` selects
    (mu+lambda) / (mu,lambda) / eaSimple.
    """
    weights = guide_fn(None)
    pop = init_population(pop_size, wt_seq, positions, alphabet, n_mut_max, rng,
                          weights=weights, suggest_seeds=suggest_seeds)
    F = np.asarray(fitness_fn(pop), dtype=float)
    W = normalize_objectives_(F, goals)
    rank, crowding = _front_rank_crowding(W)
    hof = _update_hof({}, pop, F, goals, hof_size)
    # Fixed reference (initial nadir) so the per-generation hypervolume is comparable across
    # generations -- under (mu+lambda) elitism the first front never worsens, so the trace is
    # non-decreasing (the convergence KPI).
    hv_ref = W.min(axis=0) - 1e-9
    archive = _update_archive({}, pop, F, goals)
    trajectory = [hypervolume(W, ref=hv_ref)]
    spread_traj = [spread(W)]
    best_traj = [_per_obj_best(F, goals, rank)]
    mean_traj = [list(F.mean(axis=0))]
    worst_traj = [_per_obj_worst(F, goals)]
    for _gen in range(n_gen):
        weights = guide_fn(pop)
        parent_idx = dcd_tournament(rank, crowding, pop_size, rng)
        parents = [pop[i] for i in parent_idx]
        offspring = _variation(parents, variation, wt_seq, positions, alphabet, n_mut_max, rng,
                               cx_prob, mut_prob, crossover, mutation, weights, pop_size)
        F_off = np.asarray(fitness_fn(offspring), dtype=float)
        hof = _update_hof(hof, offspring, F_off, goals, hof_size)
        archive = _update_archive(archive, offspring, F_off, goals)
        if survival == ut.LIST_SEQOPT_SURVIVAL[2]:      # "ea_simple" — generational replacement
            pop, F = offspring, F_off
        else:
            if survival == ut.LIST_SEQOPT_SURVIVAL[1]:  # "mu_comma_lambda"
                pool, F_pool = offspring, F_off
            else:                                       # "mu_plus_lambda" (default)
                pool, F_pool = pop + offspring, np.vstack([F, F_off])
            W_pool = normalize_objectives_(F_pool, goals)
            survivors, _, _ = select_nsga2(W_pool, pop_size)
            pop = [pool[i] for i in survivors]
            F = F_pool[survivors]
        W = normalize_objectives_(F, goals)
        rank, crowding = _front_rank_crowding(W)
        trajectory.append(hypervolume(W, ref=hv_ref))
        spread_traj.append(spread(W))
        best_traj.append(_per_obj_best(F, goals, rank))
        mean_traj.append(list(F.mean(axis=0)))
        worst_traj.append(_per_obj_worst(F, goals))
    hall_of_fame = [g for _p, g in sorted(hof.values(), key=lambda t: -t[0])]
    arch_genomes = [g for g, _row in archive.values()]
    arch_F = np.array([row for _g, row in archive.values()], dtype=float) if archive else \
        np.empty((0, len(goals)))
    return {"genomes": pop, "F": F, "rank": rank, "crowding": crowding,
            "spread_trajectory": spread_traj, "best_trajectory": best_traj,
            "mean_trajectory": mean_traj, "worst_trajectory": worst_traj,
            "pareto_archive": (arch_genomes, arch_F),
            "trajectory": trajectory, "hall_of_fame": hall_of_fame}


def evolve_greedy(wt_seq: str,
                  positions: List[int],
                  alphabet: List[str],
                  goals: List[str],
                  fitness_fn: Callable[[List[Dict[int, str]]], np.ndarray],
                  guide_fn: Callable[[Optional[List[Dict[int, str]]]], Optional[dict]],
                  n_mut_max: int = 5,
                  ) -> dict:
    """Importance-ordered greedy walk: step through positions highest-weight-first.

    At each position the best-scoring substitution (by the first/primary objective) is taken
    when it improves the running primary objective. Returns the accepted variants as the
    candidate set (its non-dominated subset is the front), with the per-generation trace.
    """
    weights = guide_fn(None) or {}
    ordered = sorted(positions, key=lambda p: (-float(weights.get(p, 0.0)), p))
    genome: Dict[int, str] = {}
    accepted: List[Dict[int, str]] = []
    primary_best = -np.inf
    trajectory: List[float] = []
    for pos in ordered:
        if len(genome) >= n_mut_max:
            break
        candidates = [{**genome, pos: aa} for aa in alphabet if aa != wt_seq[pos - 1]]
        if len(candidates) == 0:
            continue
        F_cand = normalize_objectives_(fitness_fn(candidates), goals)
        primary = F_cand[:, 0]
        best_i = int(np.argmax(primary))
        if primary[best_i] > primary_best:
            primary_best = float(primary[best_i])
            genome = candidates[best_i]
            accepted.append(dict(genome))
        trajectory.append(primary_best)
    if len(accepted) == 0:
        accepted = [dict()]
    F = np.asarray(fitness_fn(accepted), dtype=float)
    W = normalize_objectives_(F, goals)
    rank, crowding = _front_rank_crowding(W)
    return {"genomes": accepted, "F": F, "rank": rank, "crowding": crowding,
            "trajectory": trajectory}

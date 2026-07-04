"""
This is a script for the backend of the SeqOpt genome and evolutionary operators. A variant
genome is a sparse map ``{pos: to_aa}`` (1-based positions, distinct, size 1..n_mut_max) over
one wild-type sequence. All operators take an explicit seeded ``random.Random`` so the stream
is reproducible and matches the DEAP parity oracle, which reuses these same operators.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np


# I Helper Functions
def canonical(genome: Dict[int, str]) -> Tuple[Tuple[int, str], ...]:
    """Return the genome as a sorted, hashable tuple of (pos, to_aa) pairs (cache/label key)."""
    return tuple(sorted(genome.items()))


def apply_genome(wt_seq: str, genome: Dict[int, str]) -> str:
    """Apply all of a genome's point mutations to the wild-type sequence."""
    chars = list(wt_seq)
    for pos, to_aa in genome.items():
        chars[pos - 1] = to_aa
    return "".join(chars)


def variant_label(wt_seq: str, genome: Dict[int, str]) -> str:
    """Build the '+'-joined '<from><pos><to>' label (empty string for the wild-type)."""
    if len(genome) == 0:
        return ""
    return "+".join(f"{wt_seq[pos - 1]}{pos}{to_aa}"
                    for pos, to_aa in sorted(genome.items()))


def _weighted_positions(positions, weights):
    """Align a per-position weight vector to ``positions`` (uniform when weights is None)."""
    if weights is None:
        return None
    w = np.asarray([max(float(weights.get(p, 0.0)), 0.0) for p in positions], dtype=float)
    if w.sum() <= 0:
        return None
    return list(w)


def _pick_position(positions, weights, exclude, rng):
    """Pick one allowed position (importance-weighted when weights given), or None if exhausted."""
    pool = [p for p in positions if p not in exclude]
    if len(pool) == 0:
        return None
    w = _weighted_positions(pool, weights)
    if w is None:
        return pool[rng.randrange(len(pool))]
    return rng.choices(pool, weights=w, k=1)[0]


def _pick_aa(wt_seq, pos, alphabet, rng):
    """Pick a substitution AA at ``pos`` different from the wild-type residue."""
    wt_aa = wt_seq[pos - 1]
    pool = [aa for aa in alphabet if aa != wt_aa]
    if len(pool) == 0:
        pool = list(alphabet)
    return pool[rng.randrange(len(pool))]


# II Main Functions
def random_genome(wt_seq, positions, alphabet, n_mut_max, rng, weights=None) -> Dict[int, str]:
    """Draw one random genome: a random size in 1..n_mut_max, importance-weighted positions."""
    size = rng.randint(1, n_mut_max)
    genome: Dict[int, str] = {}
    for _ in range(size):
        pos = _pick_position(positions, weights, set(genome), rng)
        if pos is None:
            break
        genome[pos] = _pick_aa(wt_seq, pos, alphabet, rng)
    return genome


def init_population(pop_size, wt_seq, positions, alphabet, n_mut_max, rng,
                    weights=None, suggest_seeds: Optional[List[Dict[int, str]]] = None,
                    ) -> List[Dict[int, str]]:
    """Build the initial population (optionally warm-started from suggest seeds)."""
    pop: List[Dict[int, str]] = []
    if suggest_seeds is not None:
        for seed in suggest_seeds[:pop_size]:
            pop.append(dict(seed))
    while len(pop) < pop_size:
        pop.append(random_genome(wt_seq, positions, alphabet, n_mut_max, rng, weights))
    return pop


def repair(genome: Dict[int, str], n_mut_max, rng) -> Dict[int, str]:
    """Drop random positions until the genome respects ``n_mut_max`` (never empties it)."""
    if len(genome) <= n_mut_max:
        return genome
    keep = list(genome.items())
    while len(keep) > n_mut_max:
        keep.pop(rng.randrange(len(keep)))
    return dict(keep)


def crossover_uniform(g1, g2, n_mut_max, rng, cx_prob=0.5) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Uniform crossover over the union of parent positions; each gene swapped with prob cx_prob."""
    c1, c2 = dict(g1), dict(g2)
    for pos in sorted(set(g1) | set(g2)):
        a1 = c1.get(pos)
        a2 = c2.get(pos)
        if rng.random() < cx_prob:
            # Swap the per-position substitution between the two children.
            if a2 is None:
                c1.pop(pos, None)
            else:
                c1[pos] = a2
            if a1 is None:
                c2.pop(pos, None)
            else:
                c2[pos] = a1
    c1 = repair(c1, n_mut_max, rng)
    c2 = repair(c2, n_mut_max, rng)
    return c1, c2


def crossover_npoint(g1, g2, n_mut_max, rng, n_points=1) -> Tuple[Dict[int, str], Dict[int, str]]:
    """One/two-point crossover on the sorted union of parent positions."""
    union = sorted(set(g1) | set(g2))
    L = len(union)
    if L < 2:
        return dict(g1), dict(g2)
    cuts = sorted(rng.randrange(1, L) for _ in range(min(n_points, L - 1)))
    c1, c2 = {}, {}
    swap = False
    last = 0
    cuts = cuts + [L]
    for cut in cuts:
        for pos in union[last:cut]:
            src1, src2 = (g2, g1) if swap else (g1, g2)
            if pos in src1:
                c1[pos] = src1[pos]
            if pos in src2:
                c2[pos] = src2[pos]
        swap = not swap
        last = cut
    return repair(c1, n_mut_max, rng), repair(c2, n_mut_max, rng)


def mutate(genome, wt_seq, positions, alphabet, n_mut_max, rng,
           mutation="substitution", weights=None) -> Dict[int, str]:
    """Apply one in-place move (re-point / add / remove / shift) to a copy of the genome."""
    g = dict(genome)
    move = rng.random()
    if mutation == "shift" and len(g) > 0:
        # Move an existing mutation to a neighbouring free position (keeps the substitution).
        pos = list(g)[rng.randrange(len(g))]
        to_aa = g.pop(pos)
        for cand in (pos + 1, pos - 1, pos + 2, pos - 2):
            if cand in positions and cand not in g:
                g[cand] = to_aa
                return g
        g[pos] = to_aa
        return g
    # substitution family: re-point an existing position, or add / remove one.
    if len(g) == 0 or (move < 0.5 and len(g) < n_mut_max):
        pos = _pick_position(positions, weights, set(g), rng)
        if pos is not None:
            g[pos] = _pick_aa(wt_seq, pos, alphabet, rng)
    elif move < 0.8 or len(g) == 1:
        pos = list(g)[rng.randrange(len(g))]
        g[pos] = _pick_aa(wt_seq, pos, alphabet, rng)
    else:
        pos = list(g)[rng.randrange(len(g))]
        g.pop(pos)
    return g

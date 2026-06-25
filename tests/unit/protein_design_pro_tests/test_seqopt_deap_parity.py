"""This is a script to test parity of SeqOpt's NSGA-II core against the DEAP reference.

DEAP is a **dev/test-only** parity oracle (the shipped runtime is DEAP-free, pure-Python). On
fixed synthetic fitness sets these tests assert our selection core reproduces DEAP's
``sortNondominated`` / ``assignCrowdingDist`` / ``selNSGA2`` [Deb02]_:

* identical non-dominated **rank** (front membership) for every point,
* identical within-front **crowding ordering** + crowding values within ``atol``,
* identical ``selNSGA2`` survivor **set**,
* both SeqOpt engines (``exact`` / ``fast``) agree with DEAP (and each other).

Skipped when ``deap`` is not installed (core-only environments).
"""
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

deap = pytest.importorskip("deap")
from deap import base, creator, tools  # noqa: E402

from aaanalysis.protein_design_pro._backend.seqopt.nsga2 import (  # noqa: E402
    normalize_objectives_, fast_non_dominated_sort, fast_non_dominated_sort_vec,
    crowding_distance, select_nsga2, select_nsga2_engine)

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

# One maximization Fitness/Individual type reused across calls (we pre-normalize to all-max,
# so DEAP weights are all +1 and we never juggle per-test weight signatures).
if not hasattr(creator, "_SeqOptFitnessMax"):
    creator.create("_SeqOptFitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "_SeqOptInd"):
    creator.create("_SeqOptInd", list, fitness=creator._SeqOptFitnessMax)


# I Helper Functions
def _deap_inds(W):
    """Build DEAP individuals from a maximization-normalized objective array (weights all +1)."""
    creator._SeqOptFitnessMax.weights = (1.0,) * W.shape[1]
    inds = []
    for i, row in enumerate(W):
        ind = creator._SeqOptInd(row.tolist())
        ind.fitness.values = tuple(float(v) for v in row)
        ind.idx = i
        inds.append(ind)
    return inds


def _deap_rank(W):
    """DEAP non-dominated rank per point via tools.sortNondominated."""
    inds = _deap_inds(W)
    fronts = tools.sortNondominated(inds, len(inds))
    rank = np.zeros(len(inds), dtype=int)
    for f, front in enumerate(fronts):
        for ind in front:
            rank[ind.idx] = f
    return rank


def _deap_crowding(W):
    """DEAP crowding distance per point (assignCrowdingDist over each front)."""
    inds = _deap_inds(W)
    fronts = tools.sortNondominated(inds, len(inds))
    crowd = np.zeros(len(inds), dtype=float)
    for front in fronts:
        tools.emo.assignCrowdingDist(front)
        for ind in front:
            crowd[ind.idx] = ind.fitness.crowding_dist
    return crowd


def _deap_select(W, mu):
    """DEAP selNSGA2 survivor index set."""
    inds = _deap_inds(W)
    chosen = tools.selNSGA2(inds, mu)
    return {ind.idx for ind in chosen}


def _rand_W(seed, n, m):
    """Continuous random objectives (tie-free) — crowding + selNSGA2 parity is exact here."""
    rng = np.random.default_rng(seed)
    return rng.random((n, m))


def _rand_W_ties(seed, n, m):
    """Integer objectives with many ties — exercises non-dominated *rank* (front membership)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=(n, m)).astype(float)


# II Tests
class TestNonDominatedRankParity:
    @settings(max_examples=15, deadline=None)
    @given(seed=some.integers(0, 200), n=some.integers(3, 25), m=some.integers(2, 4))
    def test_rank_matches_deap(self, seed, n, m):
        W = _rand_W(seed, n, m)
        _, ours = fast_non_dominated_sort(W)
        assert ours.tolist() == _deap_rank(W).tolist()

    @settings(max_examples=10, deadline=None)
    @given(seed=some.integers(0, 200), n=some.integers(3, 25), m=some.integers(2, 4))
    def test_fast_sort_matches_deap(self, seed, n, m):
        W = _rand_W(seed, n, m)
        _, ours = fast_non_dominated_sort_vec(W)
        assert ours.tolist() == _deap_rank(W).tolist()

    def test_golden_three_point_front(self):
        W = normalize_objectives_(np.array([[1., 0.], [0., 1.], [0.5, 0.5], [0., 0.]]),
                                  ["max", "max"])
        assert fast_non_dominated_sort(W)[1].tolist() == _deap_rank(W).tolist()

    @settings(max_examples=15, deadline=None)
    @given(seed=some.integers(0, 200), n=some.integers(3, 25), m=some.integers(2, 4))
    def test_rank_matches_deap_with_ties(self, seed, n, m):
        # Heavy integer ties (duplicate fitness rows): non-dominated rank must still match DEAP.
        W = _rand_W_ties(seed, n, m)
        assert fast_non_dominated_sort(W)[1].tolist() == _deap_rank(W).tolist()


class TestCrowdingParity:
    @settings(max_examples=15, deadline=None)
    @given(seed=some.integers(0, 200), n=some.integers(4, 25), m=some.integers(2, 3))
    def test_crowding_order_and_values_match_deap(self, seed, n, m):
        W = _rand_W(seed, n, m)
        fronts, _ = fast_non_dominated_sort(W)
        ours = np.zeros(len(W))
        for front in fronts:
            d = crowding_distance(W, front)
            for idx, member in enumerate(front):
                ours[member] = d[idx]
        theirs = _deap_crowding(W)
        # inf at boundaries must agree; finite values within atol; per-front ordering identical
        assert np.array_equal(np.isinf(ours), np.isinf(theirs))
        fin = np.isfinite(ours) & np.isfinite(theirs)
        assert np.allclose(ours[fin], theirs[fin], atol=1e-9)
        for front in fronts:
            assert (np.argsort(ours[front], kind="mergesort").tolist()
                    == np.argsort(theirs[front], kind="mergesort").tolist())


class TestSelectNsga2Parity:
    @settings(max_examples=15, deadline=None)
    @given(seed=some.integers(0, 200), n=some.integers(6, 25), m=some.integers(2, 3))
    def test_survivor_set_matches_deap(self, seed, n, m):
        # selNSGA2 is identical UP TO crowding ties on the partial front (DEAP breaks ties by
        # front-appearance order, we by index — both valid). So we assert the selected
        # rank/crowding *profile* is identical: the multiset of (rank, crowding) of the
        # survivors matches, which is equivalence modulo arbitrary tie-breaks.
        W = _rand_W(seed, n, m)
        mu = max(2, n // 2)
        _, rank = fast_non_dominated_sort(W)
        crowd = np.zeros(len(W))
        for front in fast_non_dominated_sort(W)[0]:
            d = crowding_distance(W, front)
            for idx, member in enumerate(front):
                crowd[member] = d[idx]

        def profile(idxs):
            return sorted((int(rank[i]), round(float(crowd[i]), 9)) for i in idxs)

        theirs = profile(_deap_select(W, mu))
        assert profile(select_nsga2(W, mu)[0]) == theirs
        assert profile(select_nsga2_engine(W, mu, engine="fast")[0]) == theirs

    def test_engines_agree_on_survivor_order(self):
        W = _rand_W(7, 20, 2)
        ex = select_nsga2(W, 10)[0]
        fa = select_nsga2_engine(W, 10, engine="fast")[0]
        assert ex == fa

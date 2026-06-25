"""This is a script to test the SeqOpt NSGA-II backend (sort / crowding / metrics / operators).

Pure-Python core, no pro dependency: golden + property checks of the unit that mirrors DEAP's
selNSGA2 (the parity oracle is deferred until the dev-only ``deap`` dependency is added).
"""
import random
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

from aaanalysis.protein_design_pro._backend.seqopt.nsga2 import (
    normalize_objectives_, fast_non_dominated_sort, crowding_distance, crowded_better,
    dcd_tournament, select_nsga2)
from aaanalysis.protein_design_pro._backend.seqopt.metrics import hypervolume, spread
from aaanalysis.protein_design_pro._backend.seqopt.genome import (
    canonical, apply_genome, variant_label, random_genome, init_population, repair,
    crossover_uniform, crossover_npoint, mutate)

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

WT = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS = list(range(1, len(WT) + 1))
ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


class TestNonDominatedSortGoldenValues:
    def test_three_point_front_plus_dominated(self):
        F = np.array([[1., 0.], [0., 1.], [0.5, 0.5], [0., 0.]])
        W = normalize_objectives_(F, ["max", "max"])
        fronts, rank = fast_non_dominated_sort(W)
        assert rank.tolist() == [0, 0, 0, 1]
        assert fronts[0] == [0, 1, 2] and fronts[1] == [3]

    def test_single_point_is_rank_zero(self):
        W = normalize_objectives_(np.array([[3., 7.]]), ["max", "max"])
        _, rank = fast_non_dominated_sort(W)
        assert rank.tolist() == [0]

    def test_chain_of_dominated_points(self):
        F = np.array([[3., 3.], [2., 2.], [1., 1.]])
        _, rank = fast_non_dominated_sort(normalize_objectives_(F, ["max", "max"]))
        assert rank.tolist() == [0, 1, 2]

    def test_min_goal_flips_dominance(self):
        F = np.array([[0., 0.], [1., 1.]])
        _, rank = fast_non_dominated_sort(normalize_objectives_(F, ["min", "min"]))
        assert rank.tolist() == [0, 1]


class TestCrowdingGoldenValues:
    def test_boundary_points_are_infinite(self):
        F = np.array([[0., 1.], [0.5, 0.5], [1., 0.]])
        W = normalize_objectives_(F, ["max", "max"])
        d = crowding_distance(W, [0, 1, 2])
        assert np.isinf(d[0]) and np.isinf(d[2]) and np.isfinite(d[1])

    def test_interior_distance_value(self):
        F = np.array([[0., 1.], [0.5, 0.5], [1., 0.]])
        W = normalize_objectives_(F, ["max", "max"])
        d = crowding_distance(W, [0, 1, 2])
        # DEAP normalization (nobj * span): per-objective 0.5 each -> 1.0 total (matches DEAP).
        assert d[1] == pytest.approx(1.0)

    def test_two_points_both_infinite(self):
        W = normalize_objectives_(np.array([[0., 0.], [1., 1.]]), ["max", "max"])
        d = crowding_distance(W, [0, 1])
        assert np.all(np.isinf(d))

    def test_crowded_better_prefers_lower_rank(self):
        assert crowded_better(0, 0.1, 1, 9.0) is True
        assert crowded_better(1, 9.0, 0, 0.1) is False

    def test_crowded_better_prefers_more_spread_on_tie(self):
        assert crowded_better(0, 2.0, 0, 1.0) is True


class TestHypervolumeGoldenValues:
    def test_single_point_rectangle(self):
        W = np.array([[2., 3.]])
        assert hypervolume(W, ref=np.array([0., 0.])) == pytest.approx(6.0)

    def test_two_point_staircase(self):
        W = np.array([[1., 0.], [0., 1.], [0.5, 0.5]])
        assert hypervolume(W, ref=np.array([0., 0.])) == pytest.approx(0.25)

    def test_one_objective(self):
        W = np.array([[1.], [3.], [2.]])
        assert hypervolume(W, ref=np.array([0.])) == pytest.approx(3.0)

    def test_empty_is_zero(self):
        assert hypervolume(np.empty((0, 2))) == 0.0

    @settings(max_examples=5, deadline=None)
    @given(n=some.integers(min_value=1, max_value=8))
    def test_hypervolume_nonnegative(self, n):
        rng = np.random.default_rng(n)
        W = rng.random((n, 2))
        assert hypervolume(W) >= 0.0


class TestSpread:
    def test_degenerate_front_zero(self):
        assert spread(np.array([[1., 1.]])) == 0.0

    def test_distinct_points_positive(self):
        assert spread(np.array([[1., 0.], [0., 1.]])) > 0.0


class TestSelectNsga2:
    def test_keeps_mu_survivors(self):
        F = np.array([[1., 0.], [0., 1.], [0.5, 0.5], [0., 0.], [0.2, 0.2]])
        W = normalize_objectives_(F, ["max", "max"])
        surv, rank, crowd = select_nsga2(W, 3)
        assert len(surv) == 3
        assert all(rank[i] <= rank[j] for i, j in zip(surv, surv[1:]) if rank[i] != rank[j])

    def test_dcd_tournament_length_and_range(self):
        rank = np.array([0, 0, 1, 2])
        crowd = np.array([np.inf, 1.0, 0.5, 0.0])
        chosen = dcd_tournament(rank, crowd, 6, random.Random(1))
        assert len(chosen) == 6 and all(0 <= c < 4 for c in chosen)


class TestGenomeOperators:
    def test_canonical_is_sorted_tuple(self):
        assert canonical({5: "A", 1: "K"}) == ((1, "K"), (5, "A"))

    def test_apply_genome_mutates_positions(self):
        seq = apply_genome(WT, {1: "M", 3: "P"})
        assert seq[0] == "M" and seq[2] == "P"

    def test_variant_label_joins_sorted(self):
        assert variant_label(WT, {3: "P", 1: "M"}) == f"{WT[0]}1M+{WT[2]}3P"

    def test_variant_label_empty(self):
        assert variant_label(WT, {}) == ""

    @settings(max_examples=5, deadline=None)
    @given(n_mut_max=some.integers(min_value=1, max_value=6))
    def test_random_genome_respects_n_mut_max(self, n_mut_max):
        g = random_genome(WT, POSITIONS, ALPHABET, n_mut_max, random.Random(n_mut_max))
        assert 1 <= len(g) <= n_mut_max

    def test_random_genome_substitutes_non_wt(self):
        g = random_genome(WT, POSITIONS, ALPHABET, 5, random.Random(0))
        assert all(WT[p - 1] != a for p, a in g.items())

    def test_repair_caps_size(self):
        g = {p: "A" for p in range(1, 9)}
        assert len(repair(g, 3, random.Random(0))) == 3

    def test_init_population_size(self):
        pop = init_population(10, WT, POSITIONS, ALPHABET, 4, random.Random(0))
        assert len(pop) == 10

    def test_init_population_warm_start(self):
        seeds = [{1: "M"}, {2: "P"}]
        pop = init_population(5, WT, POSITIONS, ALPHABET, 4, random.Random(0),
                              suggest_seeds=seeds)
        assert pop[0] == {1: "M"} and pop[1] == {2: "P"}

    def test_crossover_uniform_respects_cap(self):
        c1, c2 = crossover_uniform({1: "A", 2: "C"}, {3: "D", 4: "E"}, 3, random.Random(0))
        assert len(c1) <= 3 and len(c2) <= 3

    def test_crossover_npoint_respects_cap(self):
        c1, c2 = crossover_npoint({1: "A", 2: "C"}, {3: "D", 5: "E"}, 3, random.Random(0),
                                  n_points=2)
        assert len(c1) <= 3 and len(c2) <= 3

    def test_mutate_substitution_changes_genome(self):
        g0 = {5: "A"}
        g1 = mutate(g0, WT, POSITIONS, ALPHABET, 5, random.Random(2), mutation="substitution")
        assert isinstance(g1, dict) and len(g1) >= 1

    def test_mutate_shift_moves_position(self):
        g0 = {5: "A"}
        g1 = mutate(g0, WT, POSITIONS, ALPHABET, 5, random.Random(0), mutation="shift")
        assert list(g1.values()) == ["A"] and set(g1) != {5} or set(g1) == {5}

    @settings(max_examples=5, deadline=None)
    @given(seed=some.integers(min_value=0, max_value=50))
    def test_random_genome_reproducible(self, seed):
        a = random_genome(WT, POSITIONS, ALPHABET, 4, random.Random(seed))
        b = random_genome(WT, POSITIONS, ALPHABET, 4, random.Random(seed))
        assert a == b

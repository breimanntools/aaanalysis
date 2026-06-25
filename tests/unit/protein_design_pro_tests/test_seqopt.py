"""This is a script to test SeqOpt.run / SeqOpt.eval and SeqOptPlot (**[pro]**).

Guarded by ``shap`` (the whole protein_design_pro subpackage imports ShapModel); skipped in a
core-only environment. Tiny deterministic wild-type + real-scale df_feat so the genuine ΔCPP /
SequenceFeature engine runs while staying fast.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

pytest.importorskip("shap")
from aaanalysis.protein_design_pro import SeqOpt, SeqOptPlot  # noqa: E402

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

OBJ = [("activity", "max", ut.COL_DELTA_PRED), ("parsimony", "min", ut.COL_N_MUT)]


class _StubModel2D:
    classes_ = np.array([0, 1])
    n_features_in_ = 4

    def predict_proba(self, X):
        x0 = np.asarray(X, dtype=float)[:, 0]
        p1 = 1.0 / (1.0 + np.exp(-x0))
        return np.column_stack([1.0 - p1, p1])


@pytest.fixture
def wt():
    return pd.DataFrame({
        ut.COL_ENTRY: ["P1"],
        ut.COL_SEQ: ["MKLAGTWYVFAILMVFWCGSTNQDEHKRPYLAGTWYVFAI"],
        ut.COL_TMD_START: [11], ut.COL_TMD_STOP: [20]})


@pytest.fixture
def df_feat():
    scales = list(ut.load_default_scales().columns[:4])
    return pd.DataFrame({
        ut.COL_FEATURE: [f"TMD-Segment(1,1)-{s}" for s in scales],
        ut.COL_CAT: ["Polarity", "ASA/Volume", "Polarity", "Energy"],
        ut.COL_SUBCAT: ["Hydrophobicity", "Volume", "Charge", "Free energy"],
        ut.COL_SCALE_NAME: scales,
        ut.COL_ABS_AUC: [.30, .25, .20, .10], ut.COL_ABS_MEAN_DIF: [.40, .30, .20, .10],
        ut.COL_MEAN_DIF: [.40, -.30, .20, -.10], ut.COL_STD_TEST: [.1] * 4,
        ut.COL_STD_REF: [.1] * 4, ut.COL_FEAT_IMPORT: [40., 30., 20., 10.]})


@pytest.fixture
def model():
    return _StubModel2D()


@pytest.fixture
def seqopt(model):
    return SeqOpt(mode="importance", model=model, random_state=7, verbose=False)


def _non_dominated(df):
    front = df[df[ut.COL_RANK] == 0]
    G = front[["activity", "parsimony"]].to_numpy(float).copy()
    G[:, 1] *= -1
    for i in range(len(G)):
        for j in range(len(G)):
            if i != j and np.all(G[j] >= G[i]) and np.any(G[j] > G[i]):
                return False
    return True


class TestSeqOptRun:
    def test_columns(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=8, n_gen=4,
                        n_mut_max=3, region="tmd")
        for c in (ut.COL_ENTRY, ut.COL_VARIANT, ut.COL_N_MUT, ut.COL_SEQ_MUT,
                  ut.COL_RANK, ut.COL_CROWDING, "activity", "parsimony"):
            assert c in df.columns

    def test_rank_zero_is_non_dominated(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=10, n_gen=5,
                        n_mut_max=3, region="tmd")
        assert _non_dominated(df)

    @settings(max_examples=4, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(pop_size=some.integers(min_value=4, max_value=12))
    def test_pop_size(self, model, wt, df_feat, pop_size):
        df = SeqOpt(mode="importance", model=model, random_state=1).run(
            df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=pop_size, n_gen=2,
            n_mut_max=2, region="tmd")
        assert len(df) >= 1

    @settings(max_examples=4, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(n_gen=some.integers(min_value=1, max_value=6))
    def test_n_gen(self, seqopt, wt, df_feat, n_gen):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, n_gen=n_gen, pop_size=6,
                        n_mut_max=2, region="tmd")
        assert len(seqopt.trajectory_) == n_gen + 1

    @settings(max_examples=4, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(n_mut_max=some.integers(min_value=1, max_value=4))
    def test_n_mut_max(self, seqopt, wt, df_feat, n_mut_max):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, n_mut_max=n_mut_max,
                        pop_size=6, n_gen=3, region="tmd")
        assert df[ut.COL_N_MUT].max() <= n_mut_max

    def test_algorithm_greedy(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, algorithm="greedy",
                        n_mut_max=3, region="tmd")
        assert len(df) >= 1

    def test_crossover_one_point(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, crossover="one_point",
                        pop_size=6, n_gen=3, n_mut_max=3, region="tmd")
        assert _non_dominated(df)

    def test_mutation_shift(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, mutation="shift",
                        pop_size=6, n_gen=3, n_mut_max=3, region="tmd")
        assert len(df) >= 1

    def test_survival_comma(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, survival="mu_comma_lambda",
                        pop_size=6, n_gen=3, n_mut_max=2, region="tmd")
        assert len(df) >= 1

    def test_cx_prob(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, cx_prob=0.9, pop_size=6,
                        n_gen=2, n_mut_max=2, region="tmd")
        assert len(df) >= 1

    def test_mut_prob(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, mut_prob=0.5, pop_size=6,
                        n_gen=2, n_mut_max=2, region="tmd")
        assert len(df) >= 1

    def test_init_suggest(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, init="suggest", pop_size=6,
                        n_gen=2, n_mut_max=2, region="tmd")
        assert len(df) >= 1

    def test_to_aa_restricts_alphabet(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, to_aa=["A", "V"],
                        pop_size=6, n_gen=2, n_mut_max=2, region="tmd")
        muts = "".join(df[ut.COL_VARIANT])
        assert len(df) >= 1

    def test_seed_reproducible(self, model, wt, df_feat):
        kw = dict(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=8, n_gen=4,
                  n_mut_max=3, region="tmd", seed=5)
        a = SeqOpt(mode="importance", model=model).run(**kw)
        b = SeqOpt(mode="importance", model=model).run(**kw)
        assert a.equals(b)

    def test_jmd_n_len(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, jmd_n_len=10, pop_size=6,
                        n_gen=2, n_mut_max=2, region="tmd")
        assert len(df) >= 1

    def test_jmd_c_len(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, jmd_c_len=10, pop_size=6,
                        n_gen=2, n_mut_max=2, region="tmd")
        assert len(df) >= 1

    # Negative cases
    def test_multi_entry_df_seq_raises(self, seqopt, df_feat):
        bad = pd.DataFrame({ut.COL_ENTRY: ["P1", "P2"],
                            ut.COL_SEQ: ["MKLAGTWYVFAILMVFWCGST", "ACDEFGHIKLMNPQRSTVWYA"],
                            ut.COL_TMD_START: [11, 11], ut.COL_TMD_STOP: [20, 20]})
        with pytest.raises(ValueError, match="exactly one"):
            seqopt.run(df_seq=bad, df_feat=df_feat, objectives=OBJ, region="tmd")

    def test_single_objective_raises(self, seqopt, wt, df_feat):
        with pytest.raises(ValueError, match="at least two"):
            seqopt.run(df_seq=wt, df_feat=df_feat,
                       objectives=[("activity", "max", ut.COL_DELTA_PRED)], region="tmd")

    def test_bad_goal_raises(self, seqopt, wt, df_feat):
        with pytest.raises(ValueError, match="goal"):
            seqopt.run(df_seq=wt, df_feat=df_feat,
                       objectives=[("a", "up", ut.COL_N_MUT), ("b", "min", ut.COL_N_MUT)],
                       region="tmd")

    def test_bad_algorithm_raises(self, seqopt, wt, df_feat):
        with pytest.raises(ValueError):
            seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, algorithm="cmaes",
                       region="tmd")

    def test_delta_pred_without_model_raises(self, wt, df_feat):
        so = SeqOpt(mode="importance", model=None)
        with pytest.raises(ValueError, match="model"):
            so.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, region="tmd")


class TestSeqOptInit:
    def test_impact_requires_reference(self, model):
        with pytest.raises(ValueError, match="mode='impact'"):
            SeqOpt(mode="impact", model=model, df_seq_ref=None, labels=None)

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError):
            SeqOpt(mode="random")

    def test_df_scales_accepted(self, model):
        so = SeqOpt(mode="importance", model=model, df_scales=ut.load_default_scales(),
                    target_class=1)
        assert so is not None

    def test_impact_mode_runs(self, wt, df_feat):
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier
        ref = pd.DataFrame({ut.COL_ENTRY: [f"R{i}" for i in range(8)],
                            ut.COL_SEQ: [wt_seq for wt_seq in
                                         (["MKLAGTWYVFAILMVFWCGSTNQDEHKRPYLAGTWYVFAI",
                                           "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"] * 4)],
                            ut.COL_TMD_START: [11] * 8, ut.COL_TMD_STOP: [20] * 8})
        labels = [1, 0] * 4
        sf = aa.SequenceFeature(verbose=False)
        X = np.asarray(sf.feature_matrix(features=list(df_feat[ut.COL_FEATURE]),
                                         df_parts=sf.get_df_parts(df_seq=ref),
                                         df_scales=ut.load_default_scales()), float)
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, labels)
        so = SeqOpt(mode="impact", model=rf, df_seq_ref=ref, labels=labels, random_state=3)
        df = so.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=6, n_gen=2,
                    n_mut_max=2, region="tmd")
        assert _non_dominated(df)

    def test_impact_mode_df_seq_ref_with_extra_columns(self, wt, df_feat):
        # Regression: a reference from load_dataset carries jmd_n/tmd/jmd_c/label columns; the
        # per-generation ShapModel refit must keep only the position-based columns (else the
        # appended variant row NaN-trips check_df_seq).
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier
        seqs = ["MKLAGTWYVFAILMVFWCGSTNQDEHKRPYLAGTWYVFAI",
                "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"] * 4
        ref = pd.DataFrame({ut.COL_ENTRY: [f"R{i}" for i in range(8)], ut.COL_SEQ: seqs,
                            ut.COL_TMD_START: [11] * 8, ut.COL_TMD_STOP: [20] * 8,
                            ut.COL_JMD_N: ["X" * 10] * 8, "label": [1, 0] * 4})
        labels = [1, 0] * 4
        sf = aa.SequenceFeature(verbose=False)
        X = np.asarray(sf.feature_matrix(features=list(df_feat[ut.COL_FEATURE]),
                                         df_parts=sf.get_df_parts(df_seq=ref),
                                         df_scales=ut.load_default_scales()), float)
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, labels)
        so = SeqOpt(mode="impact", model=rf, df_seq_ref=ref, labels=labels, random_state=3)
        df = so.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=6, n_gen=2,
                    n_mut_max=2, region="tmd")
        assert _non_dominated(df)


class TestSeqOptEval:
    def test_columns(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=10, n_gen=4,
                        n_mut_max=3, region="tmd")
        de = seqopt.eval(df_pareto=df)
        for c in (ut.COL_HYPERVOLUME, ut.COL_N_FRONT, ut.COL_SPREAD):
            assert c in de.columns

    def test_metrics_nonnegative(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=10, n_gen=4,
                        n_mut_max=3, region="tmd")
        de = seqopt.eval(df_pareto=df).iloc[0]
        assert de[ut.COL_HYPERVOLUME] >= 0 and de[ut.COL_N_FRONT] >= 1 and de[ut.COL_SPREAD] >= 0

    def test_ref_point(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=8, n_gen=3,
                        n_mut_max=3, region="tmd")
        de = seqopt.eval(df_pareto=df, ref_point=[0.0, -5.0])
        assert de.iloc[0][ut.COL_HYPERVOLUME] >= 0

    def test_missing_columns_raises(self, seqopt):
        with pytest.raises(ValueError):
            seqopt.eval(df_pareto=pd.DataFrame({"x": [1]}))


class TestSeqOptPlot:
    def test_pareto_front_returns_fig_ax(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=8, n_gen=3,
                        n_mut_max=3, region="tmd")
        res = SeqOptPlot().pareto_front(df_pareto=df, x="activity", y="parsimony")
        fig, ax = res
        assert ax is not None and fig is not None

    def test_pareto_front_only(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=8, n_gen=3,
                        n_mut_max=3, region="tmd")
        import matplotlib.pyplot as plt
        _, ax0 = plt.subplots()
        res = SeqOptPlot().pareto_front(df_pareto=df, x="activity", y="parsimony", ax=ax0,
                                        figsize=(4, 4), front_only=True)
        assert res[1] is not None

    def test_hypervolume_returns_fig_ax(self, seqopt, wt, df_feat):
        seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=8, n_gen=3,
                   n_mut_max=3, region="tmd")
        res = SeqOptPlot().hypervolume(trajectory=seqopt.trajectory_, figsize=(5, 3))
        assert res[1] is not None

    def test_bad_objective_column_raises(self, seqopt, wt, df_feat):
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=6, n_gen=2,
                        n_mut_max=2, region="tmd")
        with pytest.raises(ValueError):
            SeqOptPlot().pareto_front(df_pareto=df, x="nope", y="parsimony")

    def test_empty_trajectory_raises(self):
        with pytest.raises(ValueError):
            SeqOptPlot().hypervolume(trajectory=[])


OBJ3 = [("activity", "max", ut.COL_DELTA_PRED), ("shift", "min", ut.COL_SHIFT_SCORE),
        ("parsimony", "min", ut.COL_N_MUT)]


class TestSeqOptVisualization:
    """Convergence history + richer SeqOptPlot views (3-D, parallel coordinates)."""

    def _run3(self, seqopt, wt, df_feat):
        return seqopt.run(df_seq=wt, df_feat=df_feat, objectives=OBJ3, pop_size=12, n_gen=5,
                          n_mut_max=3, region="tmd", seed=2)

    def test_history_columns(self, seqopt, wt, df_feat):
        self._run3(seqopt, wt, df_feat)
        for c in (ut.COL_GENERATION, ut.COL_HYPERVOLUME, ut.COL_SPREAD,
                  "best_activity", "best_shift", "best_parsimony"):
            assert c in seqopt.history_.columns
        assert len(seqopt.history_) == 6      # n_gen + 1

    def test_convergence_returns_panels(self, seqopt, wt, df_feat):
        self._run3(seqopt, wt, df_feat)
        fig, axes = SeqOptPlot().convergence(history=seqopt.history_, figsize=(5, 6))
        assert len(axes) == 3

    def test_pareto_front_3d(self, seqopt, wt, df_feat):
        df = self._run3(seqopt, wt, df_feat)
        res = SeqOptPlot().pareto_front(df_pareto=df, x="activity", y="parsimony", z="shift")
        assert res[1] is not None

    def test_parallel_coordinates(self, seqopt, wt, df_feat):
        df = self._run3(seqopt, wt, df_feat)
        res = SeqOptPlot().parallel_coordinates(
            df_pareto=df, objectives=["activity", "shift", "parsimony"], front_only=True)
        assert res[1] is not None

    def test_parallel_coordinates_one_objective_raises(self, seqopt, wt, df_feat):
        df = self._run3(seqopt, wt, df_feat)
        with pytest.raises(ValueError):
            SeqOptPlot().parallel_coordinates(df_pareto=df, objectives=["activity"])

    def test_callable_sequence_objective(self, seqopt, wt, df_feat):
        # External-predictor objective: receives the variant SEQUENCE, returns a float.
        seen = {}

        def frac_tryptophan(sequence):
            seen[sequence] = sequence.count("W") / len(sequence)
            return seen[sequence]

        obj = [("activity", "max", ut.COL_DELTA_PRED), ("low_w", "min", frac_tryptophan)]
        df = seqopt.run(df_seq=wt, df_feat=df_feat, objectives=obj, pop_size=10, n_gen=4,
                        n_mut_max=3, region="tmd", seed=1)
        assert "low_w" in df.columns and len(seen) >= 1

    def test_callable_only_objectives_need_no_model(self, wt, df_feat):
        # A pure-callable multi-objective run works without any CPP model.
        so = SeqOpt(mode="importance", model=None)
        obj = [("len_w", "min", lambda s: s.count("W")), ("len_p", "max", lambda s: s.count("P"))]
        df = so.run(df_seq=wt, df_feat=df_feat, objectives=obj, pop_size=8, n_gen=3,
                    n_mut_max=2, region="tmd", seed=4)
        assert {"len_w", "len_p"}.issubset(df.columns)


class TestSeqOptCapabilities:
    """The DEAP-mapped operator/algorithm families, all pure-Python."""

    def _base(self, wt, df_feat):
        return dict(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=10, n_gen=4,
                    n_mut_max=3, region="tmd", seed=4)

    def test_variation_or(self, seqopt, wt, df_feat):
        df = seqopt.run(variation="or", cx_prob=0.5, mut_prob=0.3, **self._base(wt, df_feat))
        assert _non_dominated(df)

    def test_variation_or_prob_sum_gt_one_raises(self, seqopt, wt, df_feat):
        with pytest.raises(ValueError, match="cx_prob"):
            seqopt.run(variation="or", cx_prob=0.8, mut_prob=0.5, **self._base(wt, df_feat))

    def test_engine_fast(self, seqopt, wt, df_feat):
        df = seqopt.run(engine="fast", **self._base(wt, df_feat))
        assert _non_dominated(df)

    def test_engine_exact_equals_fast(self, model, wt, df_feat):
        kw = dict(df_seq=wt, df_feat=df_feat, objectives=OBJ, pop_size=10, n_gen=5,
                  n_mut_max=3, region="tmd", seed=11)
        ex = SeqOpt(mode="importance", model=model).run(engine="exact", **kw)
        fa = SeqOpt(mode="importance", model=model).run(engine="fast", **kw)
        assert set(ex[ut.COL_VARIANT]) == set(fa[ut.COL_VARIANT])

    def test_survival_ea_simple(self, seqopt, wt, df_feat):
        df = seqopt.run(survival="ea_simple", **self._base(wt, df_feat))
        assert len(df) >= 1

    def test_hall_of_fame_populated(self, seqopt, wt, df_feat):
        seqopt.run(hof_size=7, **self._base(wt, df_feat))
        assert 1 <= len(seqopt.hall_of_fame_) <= 7

    def test_constraints_delta_excludes_position(self, seqopt, wt, df_feat):
        df = seqopt.run(constraints=[lambda g: 11 not in g], penalty="delta",
                        **self._base(wt, df_feat))
        front = df[df[ut.COL_RANK] == 0]
        touches_11 = front[ut.COL_VARIANT].str.contains(r"\D11\D", regex=True).any()
        assert not touches_11 or (front[ut.COL_VARIANT] == "").all()

    def test_constraints_closest_valid(self, seqopt, wt, df_feat):
        df = seqopt.run(constraints=[lambda g: len(g) <= 2], penalty="closest_valid",
                        **self._base(wt, df_feat))
        assert len(df) >= 1

    def test_constraints_non_callable_raises(self, seqopt, wt, df_feat):
        with pytest.raises(ValueError, match="constraints"):
            seqopt.run(constraints=[123], **self._base(wt, df_feat))

    def test_eval_convergence_with_ref_front(self, seqopt, wt, df_feat):
        df = seqopt.run(**self._base(wt, df_feat))
        de = seqopt.eval(df_pareto=df, ref_front=[[5.0, 1.0], [3.0, 1.0]])
        assert ut.COL_CONVERGENCE in de.columns and de.iloc[0][ut.COL_CONVERGENCE] >= 0

    def test_bad_engine_raises(self, seqopt, wt, df_feat):
        with pytest.raises(ValueError):
            seqopt.run(engine="cuda", **self._base(wt, df_feat))

    def test_bad_variation_raises(self, seqopt, wt, df_feat):
        with pytest.raises(ValueError):
            seqopt.run(variation="xor", **self._base(wt, df_feat))

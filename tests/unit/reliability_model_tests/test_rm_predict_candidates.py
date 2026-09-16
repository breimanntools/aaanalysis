"""Unit tests for ReliabilityModel.predict_candidates (design-candidate entry point).

The method is thin glue: rebuild the candidate feature matrix with
``SequenceFeature.feature_matrix`` (wild-type TMD coordinates from ``df_seq``) and delegate
to ``ReliabilityModel.predict``. The tests therefore pin (a) the validation surface of every
public parameter, (b) exact agreement with a manual ``feature_matrix`` + ``predict``
round-trip, and (c) row alignment with the candidate table.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering import SequenceFeature

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

COLS_AD = [ut.COL_OOD_SCORE, ut.COL_IN_DOMAIN, ut.COL_AD_KNN, ut.COL_AD_MAHALANOBIS,
           ut.COL_AD_LEVERAGE, ut.COL_AD_STATUS, ut.COL_AD_NEAREST_TRAIN]
SEQ_P1 = "MKLAGTWYVFAILMVFWCGSTNQDEHKRPYLAGTWYVFAI"
SEQ_P2 = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"


# I Helper Functions
def _df_seq():
    """Position-based df_seq: 2 wild-types, TMD 11-20, length 40 (room for jmd_n/jmd_c=10)."""
    return pd.DataFrame({ut.COL_ENTRY: ["P1", "P2"],
                         ut.COL_SEQ: [SEQ_P1, SEQ_P2],
                         ut.COL_TMD_START: [11, 11],
                         ut.COL_TMD_STOP: [20, 20]})


def _scales(n=4):
    return list(ut.load_default_scales().columns[:n])


def _features(part="TMD", n=4):
    return [f"{part}-Segment(1,1)-{s}" for s in _scales(n)]


def _features_multipart():
    parts = ["JMD_N", "TMD", "JMD_C", "TMD"]
    return [f"{p}-Segment(1,1)-{s}" for p, s in zip(parts, _scales(4))]


def _df_feat(features=None):
    """Minimal but schema-valid df_feat for the given feature ids."""
    features = features if features is not None else _features()
    n = len(features)
    return pd.DataFrame({
        ut.COL_FEATURE: features,
        ut.COL_CAT: ["Polarity", "ASA/Volume", "Polarity", "Energy"][:n],
        ut.COL_SUBCAT: ["Hydrophobicity", "Volume", "Charge", "Free energy"][:n],
        ut.COL_SCALE_NAME: [f.split("-")[-1] for f in features],
        ut.COL_ABS_AUC: [0.30, 0.25, 0.20, 0.10][:n],
        ut.COL_ABS_MEAN_DIF: [0.40, 0.30, 0.20, 0.10][:n],
        ut.COL_MEAN_DIF: [0.40, -0.30, 0.20, -0.10][:n],
        ut.COL_STD_TEST: [0.10] * n,
        ut.COL_STD_REF: [0.10] * n})


def _train_seqs(n=24, seed=0):
    """Random single-substitution variants of the two wild-types, as a position-based df_seq."""
    rng = np.random.default_rng(seed)
    aa_list = list(ut.LIST_CANONICAL_AA)
    rows = []
    for i in range(n):
        seq = list(SEQ_P1 if i % 2 == 0 else SEQ_P2)
        for pos in rng.choice(range(10, 25), size=3, replace=False):
            seq[int(pos)] = aa_list[int(rng.integers(0, len(aa_list)))]
        rows.append((f"T{i}", "".join(seq), 11, 20))
    return pd.DataFrame(rows, columns=[ut.COL_ENTRY, ut.COL_SEQ, ut.COL_TMD_START,
                                       ut.COL_TMD_STOP])


def _matrix(features=None, df_seq=None, jmd_n_len=10, jmd_c_len=10, df_scales=None):
    """Build a feature matrix the manual way (the reference route of the round-trip)."""
    parts = sorted({f.split("-")[0].lower() for f in features})
    sf = SequenceFeature(verbose=False)
    X = sf.feature_matrix(features=features, df_seq=df_seq, df_scales=df_scales,
                          df_parts_kws=dict(list_parts=parts, jmd_n_len=jmd_n_len,
                                            jmd_c_len=jmd_c_len))
    return np.asarray(X, dtype=float)


@lru_cache(maxsize=4)
def _fitted(multipart=False):
    """Cached (rm, features) fitted on the training variants (kept cheap: no bootstrap)."""
    features = _features_multipart() if multipart else _features()
    df_train = _train_seqs()
    X = _matrix(features=features, df_seq=df_train)
    labels = [0, 1] * (len(df_train) // 2)
    rm = aa.ReliabilityModel(verbose=False, random_state=0).fit(X, labels, n_bootstrap=0,
                                                                calibrate=False)
    return rm, features


def _candidates(n=3):
    """A real SeqMut.mutate output (entry + sequence_mut + delta columns)."""
    entries = ["P1", "P1", "P2", "P2", "P1"][:n]
    positions = [12, 15, 13, 18, 11][:n]
    to_aa = ["K", "W", "D", "A", "G"][:n]
    mutations = pd.DataFrame({ut.COL_ENTRY: entries, ut.COL_POS: positions,
                              ut.COL_TO_AA: to_aa})
    return aa.SeqMut().mutate(df_seq=_df_seq(), mutations=mutations)


def _cand_df_seq(df_cand=None, df_seq=None, col_seq=ut.COL_SEQ_MUT):
    """The position-based df_seq the method builds internally (reference implementation)."""
    starts = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_TMD_START]))
    stops = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_TMD_STOP]))
    return pd.DataFrame({
        ut.COL_ENTRY: [f"{e}__{i}" for i, e in enumerate(df_cand[ut.COL_ENTRY])],
        ut.COL_SEQ: list(df_cand[col_seq]),
        ut.COL_TMD_START: [starts[e] for e in df_cand[ut.COL_ENTRY]],
        ut.COL_TMD_STOP: [stops[e] for e in df_cand[ut.COL_ENTRY]]})


@pytest.fixture(scope="module")
def rm_feat():
    return _fitted()


@pytest.fixture(scope="module")
def df_seq():
    return _df_seq()


@pytest.fixture(scope="module")
def df_cand():
    return _candidates()


# II Test Classes
class TestPredictCandidates:
    """One public parameter per test: positive cases first, negative cases after."""

    # --- df_cand
    @settings(max_examples=5)
    @given(n=some.integers(min_value=1, max_value=5))
    def test_df_cand_valid(self, n):
        rm, features = _fitted()
        df = rm.predict_candidates(df_cand=_candidates(n), df_seq=_df_seq(), features=features)
        assert isinstance(df, pd.DataFrame) and len(df) == n

    def test_df_cand_extra_columns_ignored(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        df_extra = df_cand.copy()
        df_extra["junk"] = "x"
        a = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        b = rm.predict_candidates(df_cand=df_extra, df_seq=df_seq, features=features)
        pd.testing.assert_frame_equal(a, b)
        assert "junk" not in b.columns

    def test_df_cand_invalid(self, rm_feat, df_seq):
        rm, features = rm_feat
        for bad in [None, "abc", 5, [1, 2]]:
            with pytest.raises(ValueError, match="'df_cand'"):
                rm.predict_candidates(df_cand=bad, df_seq=df_seq, features=features)

    def test_df_cand_empty_raises(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        with pytest.raises(ValueError, match="at least one candidate row"):
            rm.predict_candidates(df_cand=df_cand.head(0), df_seq=df_seq, features=features)

    def test_df_cand_missing_entry_column_raises(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        with pytest.raises(ValueError, match="missing required columns"):
            rm.predict_candidates(df_cand=df_cand.drop(columns=[ut.COL_ENTRY]), df_seq=df_seq,
                                  features=features)

    def test_df_cand_unknown_entry_raises(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        df_bad = df_cand.copy()
        df_bad[ut.COL_ENTRY] = "NOPE"
        with pytest.raises(ValueError, match=r"'df_cand' entry \('NOPE'\) is not in 'df_seq'"):
            rm.predict_candidates(df_cand=df_bad, df_seq=df_seq, features=features)

    def test_df_cand_non_string_sequence_raises(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        df_bad = df_cand.copy()
        df_bad[ut.COL_SEQ_MUT] = 5
        with pytest.raises(ValueError, match="sequence_mut"):
            rm.predict_candidates(df_cand=df_bad, df_seq=df_seq, features=features)

    # --- df_seq
    def test_df_seq_valid_subset(self, rm_feat, df_seq):
        rm, features = rm_feat
        df_cand = _candidates(2)                       # P1 candidates only
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq.head(1), features=features)
        assert len(df) == 2

    def test_df_seq_invalid(self, rm_feat, df_cand):
        rm, features = rm_feat
        for bad in [None, "abc", 5]:
            with pytest.raises(ValueError, match="'df_seq'"):
                rm.predict_candidates(df_cand=df_cand, df_seq=bad, features=features)

    def test_df_seq_not_position_based_raises(self, rm_feat, df_cand, df_seq):
        rm, features = rm_feat
        df_bad = df_seq.drop(columns=[ut.COL_TMD_START, ut.COL_TMD_STOP])
        with pytest.raises(ValueError, match="position-based format"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_bad, features=features)

    def test_df_seq_duplicate_entries_raise(self, rm_feat, df_cand, df_seq):
        rm, features = rm_feat
        df_bad = pd.concat([df_seq, df_seq.head(1)], ignore_index=True)
        with pytest.raises(ValueError, match="unique 'entry' values"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_bad, features=features)

    # --- features
    def test_features_as_list(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=list(features))
        assert list(df.columns) == ut.COLS_RELIABILITY

    def test_features_as_df_feat(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        a = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        b = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=_df_feat(features))
        pd.testing.assert_frame_equal(a, b)

    def test_features_invalid(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        for bad in [None, []]:
            with pytest.raises(ValueError, match="'features'"):
                rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=bad)

    def test_features_bad_grammar_raises(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        bad = list(features[:-1]) + ["NOT_A_FEATURE_ID"]
        with pytest.raises(ValueError, match="PART-SPLIT-SCALE"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=bad)

    def test_features_count_mismatch_raises(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        with pytest.raises(ValueError, match="features but the model was fit on"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features[:2])

    # --- df_scales
    def test_df_scales_valid(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features,
                                   df_scales=ut.load_default_scales())
        assert len(df) == len(df_cand)

    def test_df_scales_none_matches_default(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        a = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features,
                                  df_scales=None)
        b = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features,
                                  df_scales=ut.load_default_scales())
        pd.testing.assert_frame_equal(a, b)

    def test_df_scales_invalid(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        with pytest.raises(ValueError, match="df_scales"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features,
                                  df_scales="not a frame")

    # --- col_seq
    def test_col_seq_valid_default(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        a = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        b = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features,
                                  col_seq=ut.COL_SEQ_MUT)
        pd.testing.assert_frame_equal(a, b)

    def test_col_seq_scores_wild_type_column(self, rm_feat, df_seq):
        rm, features = rm_feat
        df = rm.predict_candidates(df_cand=df_seq, df_seq=df_seq, features=features,
                                   col_seq=ut.COL_SEQ)
        X_wt = _matrix(features=features, df_seq=df_seq)
        np.testing.assert_allclose(df[ut.COL_SCORE].to_numpy(),
                                   rm.predict(X_wt)[ut.COL_SCORE].to_numpy(), atol=0)

    def test_col_seq_invalid(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        for bad in [None, 5, ["sequence_mut"]]:
            with pytest.raises(ValueError, match="'col_seq'"):
                rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features,
                                      col_seq=bad)

    def test_col_seq_absent_column_raises(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        with pytest.raises(ValueError, match="missing required columns"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features,
                                  col_seq="no_such_column")

    # --- jmd_n_len / jmd_c_len
    @settings(max_examples=5)
    @given(jmd_n_len=some.integers(min_value=0, max_value=10))
    def test_jmd_n_len_valid(self, jmd_n_len):
        rm, features = _fitted()
        df = rm.predict_candidates(df_cand=_candidates(2), df_seq=_df_seq(), features=features,
                                   jmd_n_len=jmd_n_len)
        assert len(df) == 2 and df[ut.COL_AD_KNN].notna().all()

    @settings(max_examples=5)
    @given(jmd_c_len=some.integers(min_value=0, max_value=10))
    def test_jmd_c_len_valid(self, jmd_c_len):
        rm, features = _fitted()
        df = rm.predict_candidates(df_cand=_candidates(2), df_seq=_df_seq(), features=features,
                                   jmd_c_len=jmd_c_len)
        assert len(df) == 2 and df[ut.COL_AD_KNN].notna().all()

    @pytest.mark.parametrize("bad", [-1, 1.5, "a", None])
    def test_jmd_n_len_invalid(self, rm_feat, df_seq, df_cand, bad):
        rm, features = rm_feat
        with pytest.raises(ValueError, match="'jmd_n_len'"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features,
                                  jmd_n_len=bad)

    @pytest.mark.parametrize("bad", [-1, 1.5, "a", None])
    def test_jmd_c_len_invalid(self, rm_feat, df_seq, df_cand, bad):
        rm, features = rm_feat
        with pytest.raises(ValueError, match="'jmd_c_len'"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features,
                                  jmd_c_len=bad)

    # --- n_jobs
    @pytest.mark.parametrize("n_jobs", [1, 2, None, -1])
    def test_n_jobs_valid(self, rm_feat, df_seq, df_cand, n_jobs):
        rm, features = rm_feat
        a = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features, n_jobs=1)
        b = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features,
                                  n_jobs=n_jobs)
        pd.testing.assert_frame_equal(a, b)

    @pytest.mark.parametrize("bad", [0, -2, "a", 1.5])
    def test_n_jobs_invalid(self, rm_feat, df_seq, df_cand, bad):
        rm, features = rm_feat
        with pytest.raises(ValueError, match="n_jobs"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features, n_jobs=bad)

    # --- contract
    def test_columns_match_predict(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        assert list(df.columns) == ut.COLS_RELIABILITY
        assert all(c in df.columns for c in COLS_AD)

    def test_row_aligned_with_candidates(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        assert len(df) == len(df_cand)
        assert list(df.index) == list(df_cand.index)

    def test_before_fit_raises(self, df_seq, df_cand):
        with pytest.raises(RuntimeError, match="Call 'fit' before 'predict_candidates'"):
            aa.ReliabilityModel().predict_candidates(df_cand=df_cand, df_seq=df_seq,
                                                     features=_features())


class TestPredictCandidatesComplex:
    """Cross-parameter behaviour, real SeqMut / SeqOpt-shaped inputs, and the round-trip."""

    def test_round_trip_matches_manual_matrix(self, rm_feat, df_seq, df_cand):
        """Acceptance criterion: identical to a manual feature_matrix + predict round-trip."""
        rm, features = rm_feat
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        X = _matrix(features=features, df_seq=_cand_df_seq(df_cand=df_cand, df_seq=df_seq))
        pd.testing.assert_frame_equal(df.reset_index(drop=True), rm.predict(X))

    def test_seqmut_mutate_output_is_scored_and_aligned(self, rm_feat, df_seq):
        """Real SeqMut.mutate output: row-aligned, carrying the applicability-domain columns."""
        rm, features = rm_feat
        mutations = pd.DataFrame({ut.COL_ENTRY: ["P1", "P2", "P1"],
                                  ut.COL_POS: [12, 13, 19],
                                  ut.COL_TO_AA: ["K", "D", "W"]})
        df_mut = aa.SeqMut().mutate(df_seq=df_seq, mutations=mutations, df_feat=_df_feat(features))
        df_rel = rm.predict_candidates(df_cand=df_mut, df_seq=df_seq, features=features)
        assert len(df_rel) == len(df_mut) and list(df_rel.index) == list(df_mut.index)
        for col in COLS_AD:
            assert col in df_rel.columns and df_rel[col].notna().all()
        assert df_rel[ut.COL_AD_STATUS].isin(ut.LIST_AD_STATUS).all()
        joined = df_mut.join(df_rel)
        assert len(joined) == len(df_mut) and ut.COL_SEQ_MUT in joined.columns

    def test_seqmut_combine_output_shape(self, rm_feat, df_seq):
        """SeqMut.combine emits (entry, variant, n_mut, sequence_mut) — the same entry point."""
        rm, features = rm_feat
        variants = pd.DataFrame({ut.COL_ENTRY: ["P1"] * 4,
                                 ut.COL_VARIANT: ["a", "a", "b", "b"],
                                 ut.COL_POS: [12, 15, 12, 16],
                                 ut.COL_TO_AA: ["K", "W", "A", "P"]})
        df_var = aa.SeqMut().combine(df_seq=df_seq, variants=variants, df_feat=_df_feat(features))
        df_rel = rm.predict_candidates(df_cand=df_var, df_seq=df_seq, features=features)
        assert len(df_rel) == len(df_var) == 2
        assert ut.COL_SEQ_MUT in df_var.columns

    def test_multipart_features(self, df_seq):
        """Features spanning JMD-N / TMD / JMD-C are sliced into all referenced parts."""
        rm, features = _fitted(multipart=True)
        df_cand = _candidates(3)
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        X = _matrix(features=features, df_seq=_cand_df_seq(df_cand=df_cand, df_seq=df_seq))
        pd.testing.assert_frame_equal(df.reset_index(drop=True), rm.predict(X))

    def test_duplicate_candidates_scored_independently(self, rm_feat, df_seq):
        rm, features = rm_feat
        df_cand = _candidates(1)
        df_dup = pd.concat([df_cand, df_cand], ignore_index=True)
        df = rm.predict_candidates(df_cand=df_dup, df_seq=df_seq, features=features)
        assert len(df) == 2
        assert df[ut.COL_OOD_SCORE].iloc[0] == pytest.approx(df[ut.COL_OOD_SCORE].iloc[1],
                                                             abs=1e-12)

    def test_non_default_index_preserved(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        df_idx = df_cand.copy()
        df_idx.index = ["c1", "c2", "c3"][:len(df_idx)]
        df = rm.predict_candidates(df_cand=df_idx, df_seq=df_seq, features=features)
        assert list(df.index) == list(df_idx.index)

    def test_jmd_lengths_change_the_matrix(self, df_seq):
        rm, features = _fitted(multipart=True)
        df_cand = _candidates(3)
        a = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features, jmd_n_len=10)
        b = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features, jmd_n_len=4)
        assert not np.allclose(a[ut.COL_AD_KNN].to_numpy(), b[ut.COL_AD_KNN].to_numpy())

    def test_verbose_reports_the_candidate_count(self, df_seq, capsys):
        rm, features = _fitted()
        rm_verbose = aa.ReliabilityModel(verbose=True, random_state=0)
        rm_verbose.fit(rm._X_train, rm._y_train, n_bootstrap=0, calibrate=False)
        capsys.readouterr()
        rm_verbose.predict_candidates(df_cand=_candidates(3), df_seq=df_seq, features=features)
        out = capsys.readouterr().out
        assert "scored 3 candidate(s)" in out and "applicability domain" in out

    def test_heavily_mutated_candidate_leaves_the_domain(self, rm_feat, df_seq):
        """A candidate rewritten across the TMD is out-of-domain, which is the point."""
        rm, features = rm_feat
        wild = df_seq[ut.COL_SEQ].iloc[0]
        far = wild[:10] + "W" * 10 + wild[20:]
        df_cand = pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_SEQ_MUT: [far]})
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        assert df[ut.COL_OOD_SCORE].iloc[0] > 1.0
        assert not bool(df[ut.COL_IN_DOMAIN].iloc[0])
        assert df[ut.COL_AD_STATUS].iloc[0] in ("borderline", "outside")

    # Negative cross-parameter cases
    def test_invalid_col_seq_with_valid_rest_raises(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        with pytest.raises(ValueError, match="'col_seq'"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=_df_feat(features),
                                  col_seq=7, jmd_n_len=10, jmd_c_len=10, n_jobs=1)

    def test_unknown_entry_with_valid_features_raises(self, rm_feat, df_seq):
        rm, features = rm_feat
        df_cand = pd.DataFrame({ut.COL_ENTRY: ["P1", "GHOST"],
                                ut.COL_SEQ_MUT: [SEQ_P1, SEQ_P2]})
        with pytest.raises(ValueError, match="GHOST"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)

    def test_bad_grammar_inside_df_feat_raises(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        df_feat_bad = _df_feat(features).copy()
        df_feat_bad.loc[0, ut.COL_FEATURE] = "TMD-Segment(1,1)"
        with pytest.raises(ValueError, match="PART-SPLIT-SCALE"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=df_feat_bad)

    def test_feature_subset_with_valid_candidates_raises(self, rm_feat, df_seq, df_cand):
        rm, features = rm_feat
        with pytest.raises(ValueError, match="features but the model was fit on"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features[:3],
                                  jmd_n_len=10, jmd_c_len=10)

    def test_part_based_df_seq_with_valid_candidates_raises(self, rm_feat, df_cand):
        rm, features = rm_feat
        df_part = pd.DataFrame({ut.COL_ENTRY: ["P1", "P2"],
                                ut.COL_JMD_N: ["A" * 10] * 2,
                                ut.COL_TMD: ["L" * 10] * 2,
                                ut.COL_JMD_C: ["G" * 10] * 2})
        with pytest.raises(ValueError, match="position-based format"):
            rm.predict_candidates(df_cand=df_cand, df_seq=df_part, features=features)

    def test_before_fit_with_all_parameters_raises(self, df_seq, df_cand):
        with pytest.raises(RuntimeError, match="Call 'fit' before 'predict_candidates'"):
            aa.ReliabilityModel(verbose=False).predict_candidates(
                df_cand=df_cand, df_seq=df_seq, features=_features(),
                df_scales=ut.load_default_scales(), col_seq=ut.COL_SEQ_MUT,
                jmd_n_len=10, jmd_c_len=10, n_jobs=1)


class TestPredictCandidatesGoldenValues:
    """Hand-computed feature values and the documented applicability-domain identities."""

    def test_feature_value_is_the_mean_scale_over_the_tmd(self, rm_feat, df_seq):
        """A 'TMD-Segment(1,1)-<scale>' value is the plain mean of that scale over TMD 11-20."""
        rm, features = rm_feat
        df_cand = _candidates(2)
        df_scales = ut.load_default_scales()
        X_hand = np.array([[float(np.mean([df_scales.loc[a, f.split("-")[-1]]
                                           for a in seq[10:20]])) for f in features]
                           for seq in df_cand[ut.COL_SEQ_MUT]])
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), rm.predict(X_hand))

    def test_ood_score_equals_ad_knn_over_threshold(self, rm_feat, df_seq):
        rm, features = rm_feat
        df = rm.predict_candidates(df_cand=_candidates(5), df_seq=df_seq, features=features)
        np.testing.assert_allclose(df[ut.COL_OOD_SCORE].to_numpy(),
                                   df[ut.COL_AD_KNN].to_numpy() / rm.ad_threshold_, atol=1e-9)

    def test_in_domain_equals_status_inside(self, rm_feat, df_seq):
        rm, features = rm_feat
        df = rm.predict_candidates(df_cand=_candidates(5), df_seq=df_seq, features=features)
        assert (df[ut.COL_IN_DOMAIN] == (df[ut.COL_AD_STATUS] == "inside")).all()

    def test_nearest_train_of_a_training_sequence_is_its_own_row(self, rm_feat, df_seq):
        """A candidate that IS training row 0 must name row 0 as its nearest neighbour."""
        rm, features = rm_feat
        df_train = _train_seqs()
        df_cand = pd.DataFrame({ut.COL_ENTRY: ["P1"],
                                ut.COL_SEQ_MUT: [df_train[ut.COL_SEQ].iloc[0]]})
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        assert int(df[ut.COL_AD_NEAREST_TRAIN].iloc[0]) == 0
        assert float(df[ut.COL_AD_KNN].iloc[0]) >= 0.0

    def test_score_equals_the_model_probability_on_the_hand_matrix(self, rm_feat, df_seq):
        rm, features = rm_feat
        df_cand = _candidates(3)
        X = _matrix(features=features, df_seq=_cand_df_seq(df_cand=df_cand, df_seq=df_seq))
        proba = rm.model_.predict_proba(X)[:, list(rm.model_.classes_).index(rm.label_pos_)]
        df = rm.predict_candidates(df_cand=df_cand, df_seq=df_seq, features=features)
        np.testing.assert_allclose(df[ut.COL_SCORE].to_numpy(), proba, atol=1e-12)

    def test_nearest_train_is_a_valid_training_row_index(self, rm_feat, df_seq):
        rm, features = rm_feat
        df = rm.predict_candidates(df_cand=_candidates(5), df_seq=df_seq, features=features)
        nn = df[ut.COL_AD_NEAREST_TRAIN]
        assert np.issubdtype(nn.dtype, np.integer)
        assert nn.between(0, len(_train_seqs()) - 1).all()

"""This is a script to test branch arcs in CPP's filter / feature-utility backend
through the public API only (``aa.CPP``, ``aa.SequenceFeature``, ``aa.load_*``).

Each test targets a specific guard arm in the CPP filtering / feature-utility
backend (``check_feature.py``, ``cpp/utils_feature.py``) that the broader
``run`` / ``run_num`` happy-path suite does not exercise. Tests stay small
(``n`` 4-10) and assert the raise / warn / value the arm produces.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Helper functions
def _seg_features(df_scales, list_parts=("tmd",)):
    sf = aa.SequenceFeature()
    split_kws = sf.get_split_kws(split_types=["Segment"], n_split_min=1, n_split_max=1)
    return sf.get_features(list_parts=list(list_parts), split_kws=split_kws,
                           list_scales=list(df_scales))


class TestCheckFeatureBranches:
    """Reachable guard arms in ``check_feature.py`` driven through CPP /
    SequenceFeature construction and run."""

    def test_scale_id_missing_in_df_cat_raises(self):
        # check_match_df_scales_df_cat: a df_scales scale absent from df_cat raises.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=6)
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales().T.head(3).T
        df_cat = aa.load_scales(name="scales_cat")
        df_scales_extra = df_scales.copy()
        df_scales_extra["FAKE_SCALE_XYZ"] = df_scales_extra.iloc[:, 0]
        with pytest.raises(ValueError, match="missing in 'df_cat'"):
            aa.CPP(df_parts=df_parts, df_scales=df_scales_extra, df_cat=df_cat)

    def test_ext_len_exceeds_jmd_c_len_raises(self):
        # check_parts_len: ext_len <= jmd_n_len (skips the jmd_n guard) but
        # > jmd_c_len -> the jmd_c guard raises.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=4)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales().T.head(2).T
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        aa.options["ext_len"] = 5
        try:
            with pytest.raises(ValueError, match="length of jmd_c"):
                cpp.run(labels=labels, jmd_n_len=10, jmd_c_len=2, n_filter=5, n_jobs=1)
        finally:
            aa.options["ext_len"] = 0

    @settings(max_examples=5, deadline=None)
    @given(jmd_len=some.integers(min_value=3, max_value=6))
    def test_seq_based_df_seq_get_df_parts(self, jmd_len):
        # check_match_df_seq_jmd_len: a sequence-only df_seq (entry + sequence,
        # both jmd lens given) takes the seq_based arm computing tmd_start/stop.
        seq20 = "ACDEFGHIKLMNPQRSTVWY"
        df_seq = pd.DataFrame({"entry": ["A", "B"], "sequence": [seq20, seq20[::-1]]})
        df_parts = aa.SequenceFeature().get_df_parts(
            df_seq=df_seq, list_parts=["tmd"], jmd_n_len=jmd_len, jmd_c_len=jmd_len)
        assert df_parts.shape[0] == 2

    def test_pattern_feature_max_pos_split(self):
        # _get_max_pos_split Pattern arm: a Pattern feature passed to
        # get_feature_positions resolves n_max via the Pattern branch.
        sf = aa.SequenceFeature()
        feat = ["TMD-Pattern(N,1,3,5)-ANDN920101"]
        pos = sf.get_feature_positions(features=feat, tmd_seq="ACDEFGHIKLMNPQRSTVWY",
                                       jmd_n_seq="ACDEFGHIKL", jmd_c_seq="ACDEFGHIKL")
        assert "-" in pos[0]

    def test_seq_parts_too_short_for_feature_raises(self):
        # check_match_features_seq_parts else-arm (explicit tmd/jmd seqs provided):
        # a too-short tmd_seq for a Segment(5,5) feature raises.
        sf = aa.SequenceFeature()
        feat = ["TMD-Segment(5,5)-ANDN920101"]
        with pytest.raises(ValueError, match="too short"):
            sf.get_feature_positions(features=feat, tmd_seq="AC",
                                     jmd_n_seq="ACDEFGHIKL", jmd_c_seq="ACDEFGHIKL")

    def test_part_based_df_seq_get_df_parts(self):
        # check_match_df_seq_jmd_len: a part-based df_seq (jmd_n/tmd/jmd_c columns,
        # not pos-based) with both jmd lens takes the part_based-not-pos_based arm.
        seq = "ACDEFGHIKLMNPQRSTVWY"
        df_seq = pd.DataFrame({"entry": ["A", "B"],
                               "jmd_n": ["ACDEF", "FEDCA"],
                               "tmd": [seq, seq],
                               "jmd_c": ["GHIKL", "LKIHG"]})
        df_parts = aa.SequenceFeature().get_df_parts(
            df_seq=df_seq, list_parts=["tmd"], jmd_n_len=5, jmd_c_len=5)
        assert df_parts.shape[0] == 2


class TestUtilsFeatureBranches:
    """Reachable guard / value arms in ``cpp/utils_feature.py``."""

    def test_post_check_gap_nan_raises(self):
        # _feature_value -> post_check_vf_scale: accept_gaps=True over an all-gap
        # part yields NaN feature values and raises.
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=4)
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd"])
        df_scales = aa.load_scales().T.head(2).T
        feats = _seg_features(df_scales)
        df_parts_gap = df_parts.copy()
        df_parts_gap.iloc[0, 0] = "-" * len(str(df_parts_gap.iloc[0, 0]))
        with pytest.raises(ValueError, match="NaN feature values"):
            sf.feature_matrix(features=feats, df_parts=df_parts_gap,
                              df_scales=df_scales, accept_gaps=True, n_jobs=1)

    def test_get_feature_positions_as_str(self):
        # _get_positions(as_str=True) via the public get_feature_positions.
        sf = aa.SequenceFeature()
        df_scales = aa.load_scales().T.head(2).T
        feats = _seg_features(df_scales)
        pos = sf.get_feature_positions(features=feats)
        assert all(isinstance(p, str) for p in pos)
        assert "," in pos[0]

    def test_pos_based_extract_parts(self):
        # _extract_parts pos_based arm: a (sequence + tmd_start + tmd_stop) df_seq
        # with jmd lens slices parts via create_parts rather than reading columns.
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=4)
        df_pos_only = df_seq[["entry", "sequence", "tmd_start", "tmd_stop"]].copy()
        df_parts = sf.get_df_parts(df_seq=df_pos_only, list_parts=["tmd"],
                                   jmd_n_len=10, jmd_c_len=10)
        assert df_parts.shape[0] == len(df_seq)

    def test_get_df_pos_parts_mean_value_type(self):
        # get_df_pos_ + get_df_pos_parts_ value_type='mean' branch (col_val='mean_dif').
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(6).T
        df_feat = aa.CPP(df_parts=df_parts, df_scales=df_scales,
                         verbose=False).run(labels=labels, n_filter=15, n_jobs=1)
        df_pos = sf.get_df_pos(df_feat=df_feat, col_val="mean_dif",
                               list_parts=["tmd", "jmd_n", "jmd_c"])
        assert list(df_pos.columns) == ["tmd", "jmd_n", "jmd_c"]

    def test_get_df_pos_parts_sum_value_type(self):
        # get_df_pos_ + get_df_pos_parts_ value_type='sum' branch (col_val='feat_importance').
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(6).T
        df_feat = aa.CPP(df_parts=df_parts, df_scales=df_scales,
                         verbose=False).run(labels=labels, n_filter=15, n_jobs=1)
        df_feat = df_feat.copy()
        df_feat["feat_importance"] = np.linspace(0.1, 1.0, len(df_feat))
        df_pos_parts = sf.get_df_pos(df_feat=df_feat, col_val="feat_importance",
                                     list_parts=["tmd", "jmd_n"])
        df_pos = sf.get_df_pos(df_feat=df_feat, col_val="feat_importance")
        assert list(df_pos_parts.columns) == ["tmd", "jmd_n"]
        assert df_pos.shape[1] == 40

    def test_add_scale_info_drops_existing_cols(self):
        # add_scale_info_: feeding a df_feat that already carries scale-info
        # columns back through get_df_feat exercises the drop-existing-col arm.
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd"])
        df_scales = aa.load_scales().T.head(3).T
        feats = _seg_features(df_scales)
        df_feat = sf.get_df_feat(features=feats, df_parts=df_parts, labels=labels,
                                 df_scales=df_scales, n_jobs=1)
        assert "category" in df_feat.columns
        df_feat2 = sf.get_df_feat(features=df_feat, df_parts=df_parts, labels=labels,
                                  df_scales=df_scales, n_jobs=1)
        assert "category" in df_feat2.columns
        assert list(df_feat2["feature"]) == list(df_feat["feature"])


class TestParallelFilterBranches:
    """Multiprocessing arms of the assign / pre-filter-stats / progress stages
    (the n_jobs > 1 path through the shared Manager + chunked workers)."""

    def test_parallel_run_verbose_matches_serial(self):
        # n_jobs=2 over enough scales drives the chunked multiprocessing workers
        # in _assign / _stat_filter and the Manager-based progress in _progress.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(8).T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_serial = aa.CPP(df_parts=df_parts, df_scales=df_scales,
                               verbose=False).run(labels=labels, n_filter=10, n_jobs=1)
            df_parallel = aa.CPP(df_parts=df_parts, df_scales=df_scales,
                                 verbose=True).run(labels=labels, n_filter=10, n_jobs=2)
        assert set(df_serial["feature"]) == set(df_parallel["feature"])

    def test_parallel_scale_batched_run(self):
        # n_jobs>1 combined with scale-batching exercises the parallel assign /
        # pre-filter workers inside each batch.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(8).T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_feat = aa.CPP(df_parts=df_parts, df_scales=df_scales,
                             verbose=False).run(labels=labels, n_filter=10,
                                                n_jobs=2, n_batches=2)
        assert len(df_feat) == 10

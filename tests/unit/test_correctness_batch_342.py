"""Regression tests pinning a batch of low-risk correctness fixes so each defect
cannot silently return."""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


# load_dataset(non_canonical_aa="gap") must not corrupt the shared cache
def test_load_dataset_gap_does_not_corrupt_cache():
    name = "SEQ_CAPSID"  # contains non-canonical amino acids (B, U, X)
    df_gap = aa.load_dataset(name=name, non_canonical_aa="gap")
    assert df_gap[ut.COL_SEQ].str.contains(ut.STR_AA_GAP, regex=False).any(), \
        "test dataset must contain non-canonical AAs so the gap path is exercised"
    df_keep = aa.load_dataset(name=name, non_canonical_aa="keep")
    assert not df_keep[ut.COL_SEQ].str.contains(ut.STR_AA_GAP, regex=False).any(), \
        "'keep' returned gapped sequences -> the earlier 'gap' call corrupted the cache"


# load_features must return a fresh copy, not the shared cached object
def test_load_features_returns_independent_copy():
    d1 = aa.load_features(name="DOM_GSEC")
    d2 = aa.load_features(name="DOM_GSEC")
    assert d1 is not d2
    d1.iloc[0, 0] = "__SENTINEL__"
    d3 = aa.load_features(name="DOM_GSEC")
    assert d3.iloc[0, 0] != "__SENTINEL__"


# read_fasta -> clear ValueError on pre-header text; leading blank is skipped
def test_read_fasta_preheader_text_raises_valueerror(tmp_path):
    bad = tmp_path / "bad.fasta"
    bad.write_text("junk before header\n>A\nMKV\n")
    with pytest.raises(ValueError):
        aa.read_fasta(file_path=str(bad))


def test_read_fasta_leading_blank_line_ok(tmp_path):
    ok = tmp_path / "ok.fasta"
    ok.write_text("\n>A\nMKV\n>B\nAAA\n")
    df = aa.read_fasta(file_path=str(ok))
    assert len(df) == 2


# comp_seq_sim self-similarity diagonal on the [0, 100] scale
def test_comp_seq_sim_diagonal_is_100():
    pytest.importorskip("Bio")
    df_seq = pd.DataFrame({ut.COL_ENTRY: ["P1", "P2"],
                           ut.COL_SEQ: ["ACDEFGHIKL", "ACDEFGHIKM"]})
    res = aa.comp_seq_sim(df_seq=df_seq)
    diag = np.diag(np.asarray(res, dtype=float))
    assert np.allclose(diag, 100.0)


# get_best_n_clusters must not return 0 for a single-feature set (KMeans(0))
def test_get_best_n_clusters_single_feature():
    from aaanalysis.feature_engineering._backend.cpp.cpp_eval import get_best_n_clusters
    X = np.array([[0.1, 0.2, 0.3, 0.4]])  # one feature (row)
    assert get_best_n_clusters(X=X, min_th=0.3, random_state=0) >= 1


# wrong-length marker_size list -> ValueError, not a later IndexError
def test_check_marker_size_wrong_length_raises_valueerror():
    from aaanalysis._utils.plotting import _check_marker_size
    with pytest.raises(ValueError):
        _check_marker_size(marker_size=[10, 12], list_cat=["a", "b", "c"])
    assert _check_marker_size(marker_size=[10, 12, 14], list_cat=["a", "b", "c"]) == [10, 12, 14]


# display_df row/col selector is 0..n-1 (off-by-one)
def test_display_df_out_of_bounds_selector_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        aa.display_df(df=df, row_to_show=3)   # valid rows are 0..2


# check_match_X_n_clusters states the correct inequality
def test_aaclust_n_clusters_message_uses_leq():
    from aaanalysis.feature_engineering._aaclust import check_match_X_n_clusters
    X = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])  # 3 samples, 1 unique
    with pytest.raises(ValueError) as e:
        check_match_X_n_clusters(X=X, n_clusters=2)     # n_samples>=n_clusters, n_unique<n_clusters
    assert "<=" in str(e.value)


# check_metric message no longer advertises None
def test_check_metric_message_no_none():
    from aaanalysis.feature_engineering._backend.check_aaclust import check_metric
    with pytest.raises(ValueError) as e:
        check_metric(metric="not-a-metric")
    assert "None" not in str(e.value)


# options validation must check the incoming candidate, not the current global
def test_options_validation_not_bypassed_after_value_set():
    try:
        aa.options["verbose"] = True
        with pytest.raises(ValueError):
            aa.options["verbose"] = "garbage"
        assert aa.options["verbose"] is True  # garbage was rejected, not stored
        aa.options["random_state"] = 7
        with pytest.raises(ValueError):
            aa.options["random_state"] = "garbage"
        assert aa.options["random_state"] == 7
    finally:
        aa.options["verbose"] = "off"
        aa.options["random_state"] = "off"


# name_tmd must be validated like the other name_* options
def test_options_name_tmd_is_validated():
    try:
        with pytest.raises(ValueError):
            aa.options["name_tmd"] = 123
        assert aa.options["name_tmd"] == "TMD"
    finally:
        aa.options["name_tmd"] = "TMD"


# disabling then re-enabling allow_multiprocessing must restore (not lose) the
# user's own LOKY_MAX_CPU_COUNT, and never leave it stuck at "1"
def test_allow_multiprocessing_restores_user_loky_value():
    import os
    from aaanalysis import config as _cfg
    prev = os.environ.get("LOKY_MAX_CPU_COUNT")
    try:
        # Start from a clean, enabled state (reset the module cap flag).
        aa.options["allow_multiprocessing"] = True
        _cfg.check_n_jobs(n_jobs=1)
        os.environ["LOKY_MAX_CPU_COUNT"] = "4"            # the user's own cap
        aa.options["allow_multiprocessing"] = False
        _cfg.check_n_jobs(n_jobs=1)
        assert os.environ.get("LOKY_MAX_CPU_COUNT") == "1"   # our cap while disabled
        aa.options["allow_multiprocessing"] = True
        _cfg.check_n_jobs(n_jobs=1)
        assert os.environ.get("LOKY_MAX_CPU_COUNT") == "4"   # user's value restored, not lost
    finally:
        aa.options["allow_multiprocessing"] = True
        _cfg.check_n_jobs(n_jobs=1)
        if prev is None:
            os.environ.pop("LOKY_MAX_CPU_COUNT", None)
        else:
            os.environ["LOKY_MAX_CPU_COUNT"] = prev


# a value the user sets DURING the disabled window must not be clobbered
def test_allow_multiprocessing_reenable_preserves_user_change_during_disable():
    import os
    from aaanalysis import config as _cfg
    prev = os.environ.get("LOKY_MAX_CPU_COUNT")
    try:
        aa.options["allow_multiprocessing"] = True
        _cfg.check_n_jobs(n_jobs=1)                 # clean flag state
        aa.options["allow_multiprocessing"] = False
        _cfg.check_n_jobs(n_jobs=1)                 # our cap -> "1"
        os.environ["LOKY_MAX_CPU_COUNT"] = "32"     # user sets their own value while disabled
        aa.options["allow_multiprocessing"] = True
        _cfg.check_n_jobs(n_jobs=1)
        assert os.environ.get("LOKY_MAX_CPU_COUNT") == "32"   # user's value not clobbered
    finally:
        aa.options["allow_multiprocessing"] = True
        _cfg.check_n_jobs(n_jobs=1)
        if prev is None:
            os.environ.pop("LOKY_MAX_CPU_COUNT", None)
        else:
            os.environ["LOKY_MAX_CPU_COUNT"] = prev

"""Tests for the CPP.run redundancy criterion (``redundancy='legacy'|'exact'``)."""
import hashlib
import inspect
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


def _setup():
    df_seq = aa.load_dataset(name="DOM_GSEC")
    labels = df_seq[ut.COL_LABEL].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    return df_parts, labels


def _feature_sha(df_feat):
    feats = sorted(df_feat[ut.COL_FEATURE])
    return len(feats), hashlib.sha256("\n".join(feats).encode()).hexdigest()[:16]


# Frozen redundancy-reduced feature sets on the canonical DOM_GSEC cell
# (CPP(random_state=42).run(labels, redundancy=...), n_jobs=1). (n_features, sha256(sorted ids)[:16]).
# 'legacy' pins byte-identity to prior versions (guards the set(x) branch against silent drift);
# 'exact' is the T3 regression anchor for the exact-position redundancy filter, required because
# that filter changes output. Regression-marked so it runs in the nightly, not the blocking gate:
# exact-value anchors are pinned to one interpreter/OS cell and would be flaky across the matrix.
_ANCHORS = {"legacy": (100, "70eeb5b3b6633948"), "exact": (100, "5b7cb30d1112de5a")}


class TestCPPRedundancy:
    def test_legacy_is_the_default(self):
        # Routing check: an unset `redundancy` must route to the 'legacy' branch (default==explicit).
        # This does NOT prove byte-identity to prior versions on its own — the frozen anchor below does.
        df_parts, labels = _setup()
        cpp = aa.CPP(df_parts=df_parts, verbose=False, random_state=42)
        d_default = cpp.run(labels=labels, n_jobs=1).reset_index(drop=True)
        d_legacy = cpp.run(labels=labels, redundancy="legacy", n_jobs=1).reset_index(drop=True)
        pd.testing.assert_frame_equal(d_default, d_legacy)

    def test_exact_differs_from_legacy(self):
        # DOM_GSEC has multi-digit positions -> the two criteria select different features
        df_parts, labels = _setup()
        cpp = aa.CPP(df_parts=df_parts, verbose=False, random_state=42)
        legacy = set(cpp.run(labels=labels, redundancy="legacy", n_jobs=1)[ut.COL_FEATURE])
        exact = set(cpp.run(labels=labels, redundancy="exact", n_jobs=1)[ut.COL_FEATURE])
        assert legacy != exact
        assert 0 < len(exact) <= 100

    def test_exact_is_deterministic(self):
        df_parts, labels = _setup()
        cpp = aa.CPP(df_parts=df_parts, verbose=False, random_state=42)
        a = set(cpp.run(labels=labels, redundancy="exact", n_jobs=1)[ut.COL_FEATURE])
        b = set(cpp.run(labels=labels, redundancy="exact", n_jobs=1)[ut.COL_FEATURE])
        assert a == b

    def test_invalid_redundancy_raises(self):
        df_parts, labels = _setup()
        cpp = aa.CPP(df_parts=df_parts, verbose=False)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, redundancy="not-a-mode", n_jobs=1)

    def test_run_num_exposes_redundancy_default_legacy(self):
        params = inspect.signature(aa.CPP.run_num).parameters
        assert "redundancy" in params
        assert params["redundancy"].default == "legacy"

    @pytest.mark.regression
    @pytest.mark.parametrize("mode", ["legacy", "exact"])
    def test_frozen_feature_set_anchor(self, mode):
        # Pins the exact redundancy-reduced feature set per criterion; any silent reshuffle
        # (legacy byte-identity break, or exact drift) fails here. See _ANCHORS note above.
        df_parts, labels = _setup()
        cpp = aa.CPP(df_parts=df_parts, verbose=False, random_state=42)
        got = _feature_sha(cpp.run(labels=labels, redundancy=mode, n_jobs=1))
        assert got == _ANCHORS[mode], f"{mode} feature set drifted: {got} != {_ANCHORS[mode]}"

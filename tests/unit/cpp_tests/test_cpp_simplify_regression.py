"""This is a script for the CPP.simplify candidate_search='fast' T3 regression
anchor (ADR-0032, tier T3; extends the ADR-0015 pattern).

candidate_search='fast' is an opt-in heuristic that caps the candidates evaluated
per feature. It is tier T3 (statistically-equivalent): the kept-feature set MAY
differ from the exact path, but must stay within a documented quality band on the
canonical cell:

    kept-feature Jaccard >= 0.95   AND   ΔavgABS_AUC <= 0.005   (fast vs exact)

This anchor pins that band forever (the band assertion) AND freezes the measured
values (the frozen-value assertion) so a regression that stays *inside* the band
is still caught (the ADR-0032 rejected-alternative: "a later real regression
hides inside the band"). It runs in the non-gating nightly only, never in the
blocking matrix — banded/exact values are canonical-cell-specific and a platform
ULP drift must not block merges (ADR-0015 D2).

Frozen values below were captured on a dev machine (darwin/py3.13) to validate
the mechanics; the FIRST canonical-cell CI run re-verifies them and they are
re-frozen only on an intentional, reviewed change (e.g. a new _FAST_TOP_K).

Run locally (off the canonical cell) with: AAA_RUN_REGRESSION=1 pytest ...
"""
import os
import sys
import warnings

import pytest

import aaanalysis as aa

# Pin to the canonical cell; AAA_RUN_REGRESSION=1 forces it on any env (local check).
_CANONICAL_ENV = (
    os.environ.get("AAA_RUN_REGRESSION") == "1"
    or (sys.platform.startswith("linux") and sys.version_info[:2] == (3, 11))
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.skipif(
        not _CANONICAL_ENV,
        reason="banded/exact-value regression pinned to Linux/py3.11 (ADR-0015/0032); "
        "set AAA_RUN_REGRESSION=1 to force locally",
    ),
]

# --- Documented quality band (ADR-0032, T3) ---------------------------------
JACCARD_MIN = 0.95
DELTA_AVG_ABS_AUC_MAX = 0.005

# --- Frozen reference values (captured on darwin/py3.13; _FAST_TOP_K=10) ------
# Re-freeze ONLY on an intentional, reviewed change (e.g. a new _FAST_TOP_K).
FROZEN = {
    # strategy: (n_exact, top_feature, top_abs_auc_3dp, jaccard, delta)
    "greedy": (8, "TMD_C_JMD_C-Segment(1,2)-CHOP780201", 0.36, 1.0, 0.0),
    "consolidate": (9, "TMD_C_JMD_C-Segment(1,2)-CHOP780201", 0.36, 1.0, 0.0),
}


def _exact_and_fast(strategy):
    """Canonical DOM_GSEC simplify cell -> (df_exact, df_fast) for one strategy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aa.options["verbose"] = False
        aa.options["random_state"] = 0
        df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        split_kws = sf.get_split_kws(
            n_split_min=1, n_split_max=2, split_types=["Segment"]
        )
        cpp = aa.CPP(df_parts=df_parts, df_scales=aa.load_scales(top_explain_n=20),
                     split_kws=split_kws, random_state=0, verbose=False)
        df_feat = cpp.run(labels=labels, n_filter=10)
        df_exact = cpp.simplify(df_feat=df_feat, labels=labels, strategy=strategy,
                                candidate_search="exact", ml_cv=3)
        df_fast = cpp.simplify(df_feat=df_feat, labels=labels, strategy=strategy,
                               candidate_search="fast", ml_cv=3)
    return df_exact, df_fast


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


class TestSimplifyFastRegression:
    """Frozen T3 band for candidate_search='fast' vs 'exact' (ADR-0032)."""

    @pytest.mark.parametrize("strategy", ["greedy", "consolidate"])
    def test_fast_within_band(self, strategy):
        # The forever-true policy gate: fast stays inside the documented band.
        df_exact, df_fast = _exact_and_fast(strategy)
        jaccard = _jaccard(df_exact["feature"], df_fast["feature"])
        delta = abs(float(df_fast["abs_auc"].mean()) - float(df_exact["abs_auc"].mean()))
        assert jaccard >= JACCARD_MIN, (strategy, jaccard)
        assert delta <= DELTA_AVG_ABS_AUC_MAX, (strategy, delta)

    @pytest.mark.parametrize("strategy", ["greedy", "consolidate"])
    def test_frozen_values(self, strategy):
        # Catches drift INSIDE the band (a real regression the band alone misses).
        n_exact, top_feature, top_auc, frozen_j, frozen_d = FROZEN[strategy]
        df_exact, df_fast = _exact_and_fast(strategy)
        assert len(df_exact) == n_exact
        assert df_exact["feature"].iloc[0] == top_feature
        assert round(float(df_exact["abs_auc"].iloc[0]), 3) == top_auc
        jaccard = _jaccard(df_exact["feature"], df_fast["feature"])
        delta = abs(float(df_fast["abs_auc"].mean()) - float(df_exact["abs_auc"].mean()))
        assert round(jaccard, 3) == frozen_j
        assert round(delta, 4) == frozen_d

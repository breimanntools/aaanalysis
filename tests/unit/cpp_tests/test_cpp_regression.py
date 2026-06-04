"""This is a script for the CPP exact-value regression anchor (ADR-0015).

A seeded DOM_GSEC mini-pipeline whose top-ranked feature identity and frozen
``abs_auc`` (rounded to 3 dp) must not change. It guards the end-to-end
scientific result against silent drift during refactoring — what the parity
tests (builder-vs-builder agreement) cannot catch.

Exact-value freezing is only reproducible on a fixed environment, so this test
is PINNED to one canonical cell (Linux + py3.11) and self-skips elsewhere; the
py3.11-3.14 x Linux/Windows matrix has known ULP-level numeric divergence.
Frozen values below were captured on a dev machine (darwin/py3.13) to validate
the mechanics; the FIRST canonical-cell CI run re-verifies them and they are
re-frozen only on an intentional, reviewed change to CPP scoring.

Run locally (off the canonical cell) with: AAA_RUN_REGRESSION=1 pytest ...
"""
import os
import sys

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
        reason="exact-value regression pinned to Linux/py3.11 (ADR-0015); "
        "set AAA_RUN_REGRESSION=1 to force locally",
    ),
]

# --- Frozen reference values (ADR-0015) -------------------------------------
# Captured on darwin/py3.13; authoritative on the canonical Linux/py3.11 cell.
# Re-freeze ONLY on an intentional, reviewed change to CPP scoring.
FROZEN_TOP_FEATURE = "TMD_C_JMD_C-Segment(2,3)-CHOP780212"
FROZEN_TOP_ABS_AUC = 0.328
N_SAMPLES = 40
N_SCALES = 20
N_FILTER = 50


def _run_reference_pipeline():
    aa.options["verbose"] = False
    aa.options["random_state"] = 0
    df_seq = aa.load_dataset(name="DOM_GSEC", n=N_SAMPLES)
    labels = df_seq["label"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales(top60_n=20).T.head(N_SCALES).T
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False)
    return cpp.run(labels=labels, n_filter=N_FILTER, n_jobs=1)


class TestCPPRegression:
    """Frozen end-to-end CPP result (ADR-0015)."""

    def test_top_feature_identity_frozen(self):
        df_feat = _run_reference_pipeline()
        assert len(df_feat) == N_FILTER
        assert df_feat["feature"].iloc[0] == FROZEN_TOP_FEATURE

    def test_top_abs_auc_frozen(self):
        df_feat = _run_reference_pipeline()
        assert round(float(df_feat["abs_auc"].iloc[0]), 3) == FROZEN_TOP_ABS_AUC

"""Regression anchor for the ShapModel ``fuzzy_aggregation="interpolate"`` estimator.

Pins the unbiased interpolation estimator on a real DOM_GSEC fuzzy-labeling cell:
three proteins explained one at a time as a single fuzzy sample with invented
prediction scores — APP (``P05067``, p=0.85), CD44 (``P16070``, p=0.65), and a
non-substrate (``Q14802``, p=0.15).

Two guards, strongest first:

1. **Exact-p identity (platform-robust).** ``interpolate`` with ``n_rounds=1`` must
   equal ``p*S1 + (1-p)*S0`` to ``atol=1e-10`` — recomputed on the same machine, so
   no frozen value is involved and it cannot drift across platforms. This is the
   unbiased-by-construction guarantee; any regression in the estimator breaks it.
2. **Fit-count speed advantage (deterministic).** ``interpolate n_rounds=1`` does
   strictly fewer model fits than the ``threshold n_rounds=5`` default (2 vs 5 per
   fuzzy sample), captured via a fit-count spy — the wall-clock win measured against
   aaanalysis 1.0.3 (~2.15x faster on this cell) reduced to a noise-free invariant.
3. **Frozen signatures.** Compact per-protein signatures (row sum + max |value|) for
   both estimators. The threshold signatures were verified **byte-identical** to
   aaanalysis 1.0.3 on this cell (the no-regression guarantee for the default path);
   interpolate differs by design (exact-p vs the biased threshold grid).

Frozen values were captured on a dev machine (darwin/py3.13); the first canonical-cell
CI run (Linux/py3.11, nightly) re-verifies them and they are re-frozen only on an
intentional, reviewed change. Runs in the non-gating nightly only.

Run locally (off the canonical cell) with: AAA_RUN_REGRESSION=1 pytest <this file>
"""
import os
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

import aaanalysis as aa
from aaanalysis.explainable_ai_pro._backend.shap_model import shap_model_fit as B

aa.options["verbose"] = False

# Pin to the canonical cell; AAA_RUN_REGRESSION=1 forces it on any env (local check).
_CANONICAL_ENV = (
    os.environ.get("AAA_RUN_REGRESSION") == "1"
    or (sys.platform.startswith("linux") and sys.version_info[:2] == (3, 11))
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.skipif(
        not _CANONICAL_ENV,
        reason="exact-value regression pinned to Linux/py3.11; "
        "set AAA_RUN_REGRESSION=1 to force locally",
    ),
]

# Three proteins + invented prediction scores (substrate, substrate, non-substrate)
PROTEINS = {"P05067": 0.85, "P16070": 0.65, "Q14802": 0.15}
N_FEAT = 25
SEED = 42
ONE_MODEL = dict(list_model_classes=[RandomForestClassifier])

# Frozen (sum, max|value|) per protein; tolerant of cross-platform RF-SHAP drift.
# THRESHOLD == aaanalysis 1.0.3 (verified byte-identical on this cell).
SIG_THRESHOLD = {"P05067": (0.3335, 0.0533), "P16070": (0.1911, 0.0934), "Q14802": (-0.3002, 0.0701)}
SIG_INTERPOLATE_N1 = {"P05067": (0.3676, 0.0550), "P16070": (0.2222, 0.0876), "Q14802": (-0.2054, 0.0628)}
SIG_ATOL = 5e-3


def _build_cell():
    df_seq = aa.load_dataset(name="DOM_GSEC")
    df_feat = aa.load_features(name="DOM_GSEC").head(N_FEAT)
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)
    return X, df_seq["entry"].to_list(), df_seq["label"].to_list()


def _fuzzy_labels(base_labels, idx, score):
    labels = [float(v) for v in base_labels]
    labels[idx] = score
    return labels


def test_interpolate_exact_p_identity_on_dom_gsec():
    """interpolate (n_rounds=1) == p*S1 + (1-p)*S0 on the real 3-protein cell."""
    X, entries, base_labels = _build_cell()
    for entry, score in PROTEINS.items():
        i = entries.index(entry)
        labels = _fuzzy_labels(base_labels, i, score)
        sm = aa.ShapModel(**ONE_MODEL, verbose=False, random_state=SEED)
        sm.fit(X, labels=labels, fuzzy_labeling=True, fuzzy_aggregation="interpolate", n_rounds=1)
        # Reference: two seeded fits with the fuzzy sample pinned at 0 and at 1
        mk = dict(sm._list_model_kwargs[0])
        mk["random_state"] = SEED  # round 0 -> random_state + 0
        args = dict(list_model_classes=[RandomForestClassifier], list_model_kwargs=[mk],
                    explainer_class=sm._explainer_class, explainer_kwargs=sm._explainer_kwargs,
                    class_index=1, n_background_data=None)
        labels_0 = [0 if k == i else labels[k] for k in range(len(labels))]
        labels_1 = [1 if k == i else labels[k] for k in range(len(labels))]
        shap_0, _ = B._aggregate_shap_values(X, labels=labels_0, **args)
        shap_1, _ = B._aggregate_shap_values(X, labels=labels_1, **args)
        ref = score * shap_1 + (1 - score) * shap_0
        assert np.allclose(sm.shap_values, ref, atol=1e-10, rtol=0), entry


def test_interpolate_n1_fewer_fits_than_threshold_n5():
    """The wall-clock win vs 1.0.3 as a noise-free invariant: 2 fits vs 5 per fuzzy sample."""
    X, entries, base_labels = _build_cell()
    i = entries.index("P05067")
    labels = _fuzzy_labels(base_labels, i, PROTEINS["P05067"])
    orig = B._compute_shap_values
    counter = {"n": 0}

    def spy(*a, **k):
        counter["n"] += 1
        return orig(*a, **k)

    B._compute_shap_values = spy
    try:
        counter["n"] = 0
        aa.ShapModel(**ONE_MODEL, verbose=False, random_state=SEED).fit(
            X, labels=labels, fuzzy_labeling=True, fuzzy_aggregation="interpolate", n_rounds=1)
        n_interpolate = counter["n"]
        counter["n"] = 0
        aa.ShapModel(**ONE_MODEL, verbose=False, random_state=SEED).fit(
            X, labels=labels, fuzzy_labeling=True, fuzzy_aggregation="threshold", n_rounds=5)
        n_threshold = counter["n"]
    finally:
        B._compute_shap_values = orig
    assert n_interpolate == 2
    assert n_threshold == 5
    assert n_interpolate < n_threshold


@pytest.mark.parametrize("aggregation,n_rounds,frozen", [
    ("threshold", 5, SIG_THRESHOLD),
    ("interpolate", 1, SIG_INTERPOLATE_N1),
])
def test_frozen_signatures(aggregation, n_rounds, frozen):
    """Coarse per-protein anchors; threshold signatures == aaanalysis 1.0.3 on this cell."""
    X, entries, base_labels = _build_cell()
    for entry, score in PROTEINS.items():
        i = entries.index(entry)
        labels = _fuzzy_labels(base_labels, i, score)
        sm = aa.ShapModel(**ONE_MODEL, verbose=False, random_state=SEED)
        sm.fit(X, labels=labels, fuzzy_labeling=True, fuzzy_aggregation=aggregation, n_rounds=n_rounds)
        row = sm.shap_values[i]
        exp_sum, exp_maxabs = frozen[entry]
        assert np.isclose(row.sum(), exp_sum, atol=SIG_ATOL), f"{entry} sum"
        assert np.isclose(np.abs(row).max(), exp_maxabs, atol=SIG_ATOL), f"{entry} maxabs"

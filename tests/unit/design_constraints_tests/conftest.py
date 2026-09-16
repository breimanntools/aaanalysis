"""Shared fixtures for the DesignConstraints test suite.

A tiny deterministic parent sequence plus the position-based ``df_seq`` / real-scale
``df_feat`` the AAMut / SeqMut / SeqOpt wiring tests need, so the genuine ΔCPP engine runs
while the suite stays fast.
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis.utils as ut

# Parent used across the suite: 1-based positions 1..10 are M K L A G T W Y V F.
PARENT = "MKLAGTWYVF"


@pytest.fixture
def parent():
    """A 10-residue parent sequence with 1-based positions 1..10 = M K L A G T W Y V F."""
    return PARENT


@pytest.fixture
def df_seq_pos():
    """Position-based df_seq: one 40-residue protein with TMD 11-20."""
    return pd.DataFrame({
        ut.COL_ENTRY: ["P1"],
        ut.COL_SEQ: ["MKLAGTWYVFAILMVFWCGSTNQDEHKRPYLAGTWYVFAI"],
        ut.COL_TMD_START: [11],
        ut.COL_TMD_STOP: [20],
    })


@pytest.fixture
def df_feat():
    """Small df_feat over the TMD with real scales, mean_dif and feat_importance."""
    scales = list(ut.load_default_scales().columns[:4])
    return pd.DataFrame({
        ut.COL_FEATURE: [f"TMD-Segment(1,1)-{s}" for s in scales],
        ut.COL_CAT: ["Polarity", "ASA/Volume", "Polarity", "Energy"],
        ut.COL_SUBCAT: ["Hydrophobicity", "Volume", "Charge", "Free energy"],
        ut.COL_SCALE_NAME: scales,
        ut.COL_ABS_AUC: [0.30, 0.25, 0.20, 0.10],
        ut.COL_ABS_MEAN_DIF: [0.40, 0.30, 0.20, 0.10],
        ut.COL_MEAN_DIF: [0.40, -0.30, 0.20, -0.10],
        ut.COL_STD_TEST: [0.10] * 4,
        ut.COL_STD_REF: [0.10] * 4,
        ut.COL_FEAT_IMPORT: [40.0, 30.0, 20.0, 10.0],
    })


class _StubModel2D:
    """scikit-learn-style stub whose positive-class score is a logistic of the first feature."""
    classes_ = np.array([0, 1])
    n_features_in_ = 4

    def predict_proba(self, X):
        x0 = np.asarray(X, dtype=float)[:, 0]
        p1 = 1.0 / (1.0 + np.exp(-x0))
        return np.column_stack([1.0 - p1, p1])


@pytest.fixture
def model():
    """A fitted-classifier stub exposing ``predict_proba``."""
    return _StubModel2D()

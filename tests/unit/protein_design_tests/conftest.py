"""Shared fixtures for the protein_design (AAMut / SeqMut) test suite.

Tiny, deterministic position-based ``df_seq`` + a real-scale ``df_feat`` so the ΔCPP engine
runs through the genuine SequenceFeature builder (not a mock) while staying fast.
"""
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


def _scales():
    """First few real scale ids from the default scale set."""
    return list(ut.load_default_scales().columns[:4])


@pytest.fixture(scope="session")
def df_scales():
    return ut.load_default_scales()


@pytest.fixture
def df_seq_pos():
    """Position-based df_seq: 2 proteins, TMD 11-20, length 40 (room for jmd_n/jmd_c=10)."""
    return pd.DataFrame({
        ut.COL_ENTRY: ["P1", "P2"],
        ut.COL_SEQ: ["MKLAGTWYVFAILMVFWCGSTNQDEHKRPYLAGTWYVFAI",
                     "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"],
        ut.COL_TMD_START: [11, 11],
        ut.COL_TMD_STOP: [20, 20],
    })


@pytest.fixture
def df_feat():
    """Small df_feat over TMD with real scales, including mean_dif + feat_importance."""
    scales = _scales()
    return pd.DataFrame({
        ut.COL_FEATURE: [f"TMD-Segment(1,1)-{s}" for s in scales],
        ut.COL_CAT: ["Polarity", "ASA/Volume", "Polarity", "Energy"],
        ut.COL_SUBCAT: ["Hydrophobicity", "Volume", "Charge", "Free energy"],
        ut.COL_SCALE_NAME: scales,
        ut.COL_ABS_AUC: [0.30, 0.25, 0.20, 0.10],
        ut.COL_ABS_MEAN_DIF: [0.40, 0.30, 0.20, 0.10],
        ut.COL_MEAN_DIF: [0.40, -0.30, 0.20, -0.10],
        ut.COL_STD_TEST: [0.10, 0.10, 0.10, 0.10],
        ut.COL_STD_REF: [0.10, 0.10, 0.10, 0.10],
        ut.COL_FEAT_IMPORT: [40.0, 30.0, 20.0, 10.0],
    })


@pytest.fixture
def df_feat_multipart():
    """df_feat spanning jmd_n / tmd / jmd_c parts."""
    scales = _scales()
    parts = ["JMD_N", "TMD", "JMD_C", "TMD"]
    return pd.DataFrame({
        ut.COL_FEATURE: [f"{p}-Segment(1,1)-{s}" for p, s in zip(parts, scales)],
        ut.COL_CAT: ["Polarity", "ASA/Volume", "Polarity", "Energy"],
        ut.COL_SUBCAT: ["Hydrophobicity", "Volume", "Charge", "Free energy"],
        ut.COL_SCALE_NAME: scales,
        ut.COL_ABS_AUC: [0.30, 0.25, 0.20, 0.10],
        ut.COL_ABS_MEAN_DIF: [0.40, 0.30, 0.20, 0.10],
        ut.COL_MEAN_DIF: [0.40, -0.30, 0.20, -0.10],
        ut.COL_STD_TEST: [0.10, 0.10, 0.10, 0.10],
        ut.COL_STD_REF: [0.10, 0.10, 0.10, 0.10],
        ut.COL_FEAT_IMPORT: [40.0, 30.0, 20.0, 10.0],
    })


@pytest.fixture
def df_impact():
    """A small AAMut.run output for the plot/eval tests."""
    return aa.AAMut().run(from_aa=["M", "L", "K"], to_aa=["V", "A", "D"])


@pytest.fixture
def df_scan(df_seq_pos, df_feat):
    """A SeqMut.scan output over the TMD for the eval/plot tests."""
    return aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")

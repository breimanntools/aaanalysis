"""This is a script for the end-to-end integration test:
StructurePreprocessor.encode_dssp + encode_pdb → combine_dict_nums →
NumericalFeature.get_parts → CPP.run_num.

The test pre-populates ``df_seq`` with synthetic DSSP/PDB column data so the
encoders don't need the ``mkdssp`` binary, and uses small synthetic PDB
fixtures from ``aaanalysis/_data/pdb_test/`` for the ``encode_pdb`` step.
The goal is to prove that the ``(df_scales, df_cat, dict_num)`` triple
produced by ``StructurePreprocessor`` is drop-in compatible with
``CPP.run_num``.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

PDB_FIXTURES = Path(__file__).resolve().parents[3] / \
    "aaanalysis" / "_data" / "pdb_test"


def _build_df_seq_with_dssp(n_per_label=3, L=40):
    """Build a labeled df_seq with synthetic DSSP-style list columns + tmd boundaries.

    Each entry has a fixed-length sequence so ``len(ss) == L``.
    Two classes (label 0 vs 1) differ in their SS distribution so CPP has
    something to discriminate.
    """
    rng = np.random.default_rng(7)
    aa_letters = list(ut.LIST_CANONICAL_AA)
    rows = []
    for label in (0, 1):
        for k in range(n_per_label):
            entry = f"S{label}_{k}"
            sequence = "".join(rng.choice(aa_letters, size=L))
            ss_chars = ["H"] * L if label == 1 else ["E"] * L
            asa = [60.0 + 10 * label] * L
            phi = [-60.0 - 5 * label] * L
            psi = [-45.0 - 5 * label] * L
            rows.append({
                "entry": entry,
                "sequence": sequence,
                "label": label,
                "tmd_start": 11,
                "tmd_stop": 30,
                ut.COL_SS: ss_chars,
                "asa": asa,
                "phi": phi,
                "psi": psi,
                ut.COL_DSSP_OK: True,
            })
    return pd.DataFrame(rows)


def _build_synthetic_dict_pdb(df_seq, D=1):
    """Produce a deterministic dict_pdb without invoking encode_pdb."""
    rng = np.random.default_rng(11)
    return {row["entry"]: rng.standard_normal((len(row["sequence"]), D))
            for _, row in df_seq.iterrows()}


class TestStructuralRunNum:
    """End-to-end smoke: StructurePreprocessor → CPP.run_num runs without errors."""

    def test_encode_dssp_plus_combine_then_run_num(self):
        df_seq = _build_df_seq_with_dssp(n_per_label=3, L=40)
        stp = aa.StructurePreprocessor(verbose=False)
        dict_dssp, _ = stp.encode_dssp(
            df_seq=df_seq, pdb_folder=None,
            features=["ss3", "rasa", "phi_psi_sincos"])
        dict_pdb = _build_synthetic_dict_pdb(df_seq, D=1)
        dict_num = aa.combine_dict_nums(dict_nums=[dict_dssp, dict_pdb])
        # Shape contract: per-entry dict_num is (L_entry, D_total)
        assert all(v.shape == (40, 3 + 1 + 4 + 1) for v in dict_num.values())

        df_scales, df_cat = stp.build_scales(
            features=["ss3", "rasa", "phi_psi_sincos", "bfactor"])
        nf = aa.NumericalFeature()
        df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq,
                                                dict_num=dict_num)
        assert isinstance(df_parts, pd.DataFrame)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, df_cat=df_cat)
        labels = df_seq["label"].tolist()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_feat = cpp.run_num(dict_num_parts=dict_num_parts,
                                  labels=labels, n_filter=5, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        assert "feature" in df_feat.columns
        assert len(df_feat) >= 1

    def test_encode_pdb_fixture_to_dict_num(self):
        """encode_pdb against the small synthetic PDB fixtures → expected shape."""
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKLMNPQRS", "VLIMKRSTGADE"],
        })
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=df_seq,
                                       pdb_folder=str(PDB_FIXTURES),
                                       features=["bfactor"])
        assert d["P1"].shape == (16, 1)
        assert d["P2"].shape == (12, 1)

    def test_build_scales_matches_combined_D(self):
        df_seq = _build_df_seq_with_dssp(n_per_label=2, L=30)
        stp = aa.StructurePreprocessor(verbose=False)
        dict_dssp, _ = stp.encode_dssp(df_seq=df_seq, pdb_folder=None,
                                       features=["ss3", "rasa"])
        df_scales, df_cat = stp.build_scales(features=["ss3", "rasa"])
        D = next(iter(dict_dssp.values())).shape[1]
        assert D == len(df_scales.columns)
        assert D == len(df_cat)

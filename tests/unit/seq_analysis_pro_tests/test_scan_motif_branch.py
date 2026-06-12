"""This is a script to test branch arms of scan_motif() via the public API.

Targets the empty-result ``output_mode='sequences'`` arm (no FIMO hits survive
an impossibly strict threshold). Exercised only through ``aa.scan_motif``.
Requires the FIMO binary; skipped otherwise."""
import shutil

import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

fimo_required = pytest.mark.skipif(shutil.which("fimo") is None,
                                   reason="FIMO binary not on PATH")

SCHEMA_SEQUENCES = ["entry", "sequence", "labels"]


# I Helper Functions
def _pwm_for_a(window_size=5):
    """Alanine-dominant PWM (A=0.81, the other 19 AAs 0.01 each)."""
    pwm = pd.DataFrame(0.01, index=range(window_size),
                       columns=list(ut.LIST_CANONICAL_AA))
    pwm["A"] = 0.81
    return pwm


def _df_seq_with_aaa():
    """P1 positive (excluded); P2/P3 candidates."""
    return pd.DataFrame({
        "entry": ["P1", "P2", "P3"],
        "sequence": ["ACDEFGHIKLAAAAA", "ACDEFGHIKLAAAAA", "ACDEFGHIKLMNPQR"],
        "pos": [[5], [], []],
    })


# II Test Classes
@fimo_required
class TestScanMotifBranch:
    """Branch arms reachable through aa.scan_motif."""

    def test_valid_empty_result_sequences_mode(self):
        """No-hit + output_mode='sequences' arm: an impossibly strict p-value
        threshold yields zero hits, so the per-residue labels output is built
        with all-empty centers (one row per source protein)."""
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-300, output_mode="sequences")
        assert list(df.columns) == SCHEMA_SEQUENCES
        assert len(df) == 3
        # every label list is the same length as its sequence, all label_ref(0)
        for seq, labels in zip(df["sequence"], df["labels"]):
            assert len(labels) == len(seq)
        # no positive marked here because positives only mark via pos_col;
        # P1's positive position is still labelled label_test in sequences mode
        labels_p1 = df.loc[df["entry"] == "P1", "labels"].iloc[0]
        assert 1 in labels_p1

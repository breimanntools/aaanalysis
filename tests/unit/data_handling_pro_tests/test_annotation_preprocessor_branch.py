"""Branch-coverage tests for AnnotationPreprocessor reached through the
public ``aa.AnnotationPreprocessor`` API.

Targets un-hit branch arms in ``_annot_preproc.py``:
 * ``ingest`` optional ``score`` column unit-range guard (the
   ``if COL_SCORE in df_user.columns`` True arm).
 * ``to_df_seq`` residue-type collection with a positive position that lies
   outside the target sequence (the ``if 1 <= p <= len(seq)`` False arm).
"""
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


TDS_SEQ = "MKSTYACDESGHIKLSNPQRST"


# I Helper Functions
def _df_seq_one(entry="P1", seq=TDS_SEQ):
    return pd.DataFrame({ut.COL_ENTRY: [entry], ut.COL_SEQ: [seq]})


# II Test Classes
class TestIngestScoreColumn:
    """ingest: optional 'score' column triggers the unit-range guard."""

    def test_score_column_in_range_accepted(self):
        ap = aa.AnnotationPreprocessor(verbose=False)
        df_user = pd.DataFrame({
            ut.COL_PROTEIN_ID: ["P1", "P1"],
            ut.COL_START: [3, 7],
            ut.COL_FEATURE_TYPE: ["phospho", "phospho"],
            ut.COL_SCORE: [0.4, 1.0],
        })
        out = ap.ingest(df_user=df_user)
        assert ut.COL_SCORE in out.columns
        assert len(out) == 2

    def test_score_column_out_of_range_raises(self):
        ap = aa.AnnotationPreprocessor(verbose=False)
        df_user = pd.DataFrame({
            ut.COL_PROTEIN_ID: ["P1"],
            ut.COL_START: [3],
            ut.COL_FEATURE_TYPE: ["phospho"],
            ut.COL_SCORE: [1.5],   # out of [0, 1]
        })
        with pytest.raises(ValueError):
            ap.ingest(df_user=df_user)


class TestToDfSeqPositionOutOfRange:
    """to_df_seq: a positive annotation 'start' beyond the sequence length
    exercises the ``1 <= p <= len(seq)`` False arm in the residue-type pass."""

    def test_position_beyond_sequence_ignored_for_residue_type(self):
        ap = aa.AnnotationPreprocessor(verbose=False)
        df_seq = _df_seq_one()
        # 'start'=999 is past len(TDS_SEQ); the residue-type collector must
        # skip it without IndexError. Pair with an in-range positive so the
        # feature_type has at least one valid anchor.
        df_user = pd.DataFrame({
            ut.COL_PROTEIN_ID: ["P1", "P1"],
            ut.COL_START: [3, 999],
            ut.COL_FEATURE_TYPE: ["phospho", "phospho"],
            ut.COL_AA: ["S", ""],
        })
        df_annot = ap.ingest(df_user=df_user)
        # Must not raise IndexError despite the out-of-range position; the
        # residue-type collector skips p=999 (1 <= p <= len(seq) False arm)
        # while the in-range S at position 3 still drives residue matching.
        out = ap.to_df_seq(df_seq=df_seq, df_annot=df_annot,
                           feature_type="phospho", match_residue_type=True)
        assert len(out["aa_context"].iloc[0]) == len(TDS_SEQ)
        # Residue-type matching restricts eligible context to S residues only.
        ctx = out["aa_context"].iloc[0]
        elig = [i + 1 for i, c in enumerate(ctx) if c == "1"]
        assert all(TDS_SEQ[p - 1] == "S" for p in elig)

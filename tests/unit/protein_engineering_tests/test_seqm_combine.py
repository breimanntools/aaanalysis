"""Tests for SeqMut.combine (score combined multi-mutation variants)."""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


def _variants():
    """Two variants on P1: a double (11A, 12P) and a single (13K)."""
    return pd.DataFrame({
        ut.COL_ENTRY: ["P1", "P1", "P1"],
        ut.COL_VARIANT: ["v1", "v1", "v2"],
        ut.COL_POS: [11, 12, 13],
        ut.COL_TO_AA: ["A", "P", "K"],
    })


class TestSeqMutCombine:
    def test_columns_model_free(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        assert list(df.columns) == ut.COLS_SEQMUT_VARIANT

    def test_columns_with_model(self, df_seq_pos, df_feat, model_tuple):
        df = aa.SeqMut(model=model_tuple).combine(df_seq=df_seq_pos, variants=_variants(),
                                                  df_feat=df_feat)
        assert ut.COL_DELTA_PRED in df.columns

    def test_one_row_per_variant(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        assert len(df) == 2

    def test_variant_label_joins_mutations(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        labels = set(df[ut.COL_VARIANT])
        assert any("+" in lab and lab.count("+") == 1 for lab in labels)  # the double

    def test_n_mut_counts_mutations(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        assert set(df[ut.COL_N_MUT]) == {2, 1}

    def test_combined_sequence_carries_all_mutations(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        double = df[df[ut.COL_N_MUT] == 2].iloc[0]
        assert double[ut.COL_SEQ_MUT][10] == "A" and double[ut.COL_SEQ_MUT][11] == "P"

    def test_sorted_by_score(self, df_seq_pos, df_feat, model_tuple):
        df = aa.SeqMut(model=model_tuple).combine(df_seq=df_seq_pos, variants=_variants(),
                                                  df_feat=df_feat)
        assert np.all(np.diff(df[ut.COL_DELTA_PRED].to_numpy()) <= 1e-9)

    def test_jmd_n_len(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                                 jmd_n_len=8)
        assert len(df) == 2

    def test_jmd_c_len(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                                 jmd_c_len=8)
        assert len(df) == 2

    # Negative cases
    def test_duplicate_position_in_variant_raises(self, df_seq_pos, df_feat):
        bad = pd.DataFrame({ut.COL_ENTRY: ["P1", "P1"], ut.COL_VARIANT: ["v", "v"],
                            ut.COL_POS: [11, 11], ut.COL_TO_AA: ["A", "P"]})
        with pytest.raises(ValueError):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=bad, df_feat=df_feat)

    def test_missing_variant_column_raises(self, df_seq_pos, df_feat):
        bad = pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_POS: [11], ut.COL_TO_AA: ["A"]})
        with pytest.raises(ValueError):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=bad, df_feat=df_feat)

    def test_unknown_entry_raises(self, df_seq_pos, df_feat):
        bad = pd.DataFrame({ut.COL_ENTRY: ["NOPE"], ut.COL_VARIANT: ["v"],
                            ut.COL_POS: [11], ut.COL_TO_AA: ["A"]})
        with pytest.raises(ValueError):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=bad, df_feat=df_feat)


    # Candidate lineage (opt-in, purely additive)
    def test_lineage_default_leaves_the_output_unchanged(self, df_seq_pos, df_feat):
        seqm = aa.SeqMut()
        df = seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        assert list(df.columns) == ut.COLS_SEQMUT_VARIANT
        assert seqm.lineage_ is None

    def test_lineage_true_appends_only_candidate_id(self, df_seq_pos, df_feat):
        seqm = aa.SeqMut()
        df_off = seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        df_on = seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                             lineage=True)
        assert list(df_on.columns) == ut.COLS_SEQMUT_VARIANT + [ut.COL_CANDIDATE_ID]
        assert df_off.equals(df_on[df_off.columns])

    def test_lineage_records_are_row_aligned(self, df_seq_pos, df_feat):
        seqm = aa.SeqMut()
        df = seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                          lineage=True)
        assert list(df[ut.COL_CANDIDATE_ID]) == [r["candidate_id"] for r in seqm.lineage_]

    def test_lineage_records_objective_values(self, df_seq_pos, df_feat, model_tuple):
        seqm = aa.SeqMut(model=model_tuple)
        seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat, lineage=True)
        assert sorted(seqm.lineage_[0]["objective_values"]) == sorted(
            [ut.COL_DELTA_CPP, ut.COL_DELTA_PRED, ut.COL_SHIFT_SCORE])

    def test_lineage_method_names_the_generating_method(self, df_seq_pos, df_feat):
        seqm = aa.SeqMut()
        seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat, lineage=True)
        assert seqm.lineage_[0]["method"] == "SeqMut.combine"

    def test_lineage_without_constraints_has_no_digest(self, df_seq_pos, df_feat):
        seqm = aa.SeqMut()
        seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat, lineage=True)
        assert seqm.lineage_[0]["constraints_digest"] is None

    def test_lineage_with_constraints_keeps_the_feasibility_columns(self, df_seq_pos, df_feat):
        seqm = aa.SeqMut()
        df = seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                          constraints=aa.DesignConstraints(n_mut_max=1), lineage=True)
        assert {ut.COL_IS_FEASIBLE, ut.COL_REASONS, ut.COL_CANDIDATE_ID} <= set(df.columns)

    def test_lineage_false_is_the_documented_default(self, df_seq_pos, df_feat):
        seqm = aa.SeqMut()
        df = seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                          lineage=False)
        assert ut.COL_CANDIDATE_ID not in df.columns

    def test_lineage_string_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError, match="lineage"):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                                lineage="yes")

    def test_lineage_int_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError, match="lineage"):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                                lineage=2)

    def test_lineage_none_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError, match="lineage"):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                                lineage=None)

    def test_lineage_incomplete_record_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError, match="should carry every lineage field"):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                                lineage={"candidate_id": "sha256:0"})

    def test_lineage_empty_record_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError, match="should carry every lineage field"):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                                lineage={})


class TestSeqMutCombineGoldenValues:
    def test_single_variant_matches_mutate(self, df_seq_pos, df_feat, model_tuple):
        # A 1-mutation "variant" must score identically to the same point mutation via mutate.
        sm = aa.SeqMut(model=model_tuple)
        single = pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_VARIANT: ["v"],
                               ut.COL_POS: [11], ut.COL_TO_AA: ["A"]})
        df_c = sm.combine(df_seq=df_seq_pos, variants=single, df_feat=df_feat)
        muts = pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_POS: [11], ut.COL_TO_AA: ["A"]})
        df_m = sm.mutate(df_seq=df_seq_pos, mutations=muts, df_feat=df_feat)
        assert df_c[ut.COL_DELTA_PRED].iloc[0] == pytest.approx(df_m[ut.COL_DELTA_PRED].iloc[0])
        assert df_c[ut.COL_DELTA_CPP].iloc[0] == pytest.approx(df_m[ut.COL_DELTA_CPP].iloc[0])

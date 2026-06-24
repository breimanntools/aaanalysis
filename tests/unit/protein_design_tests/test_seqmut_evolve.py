"""Tests for SeqMut.evolve (greedy directed-evolution stacking)."""
import numpy as np
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


class TestSeqMutEvolve:
    def test_columns_model_free(self, df_seq_pos, df_feat):
        df = aa.SeqMut().evolve(df_seq=df_seq_pos, df_feat=df_feat, depth=2, region="tmd")
        assert list(df.columns) == ut.COLS_SEQMUT_EVOLVE

    def test_columns_with_model(self, df_seq_pos, df_feat, model_tuple):
        df = aa.SeqMut(model=model_tuple).evolve(df_seq=df_seq_pos, df_feat=df_feat, depth=2,
                                                 region="tmd")
        assert ut.COL_DELTA_PRED in df.columns

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_depth_controls_rounds_per_entry(self, df_seq_pos, df_feat, depth):
        df = aa.SeqMut().evolve(df_seq=df_seq_pos, df_feat=df_feat, depth=depth, region="tmd")
        for _entry, g in df.groupby(ut.COL_ENTRY):
            assert len(g) <= depth
            assert list(g[ut.COL_ROUND]) == list(range(1, len(g) + 1))

    def test_distinct_positions_per_entry(self, df_seq_pos, df_feat):
        df = aa.SeqMut().evolve(df_seq=df_seq_pos, df_feat=df_feat, depth=3, region="tmd")
        for _entry, g in df.groupby(ut.COL_ENTRY):
            positions = [int(m[1:-1]) for m in g[ut.COL_MUTATION]]
            assert len(set(positions)) == len(positions)

    def test_greedy_objective_non_decreasing_with_model(self, df_seq_pos, df_feat, model_tuple):
        # Cumulative delta_pred should not decrease as greedy rounds stack (each round adds the
        # best available marginal gain on a fresh background; with this monotone stub it climbs).
        df = aa.SeqMut(model=model_tuple).evolve(df_seq=df_seq_pos, df_feat=df_feat, depth=3,
                                                 region="tmd")
        for _entry, g in df.groupby(ut.COL_ENTRY):
            vals = g.sort_values(ut.COL_ROUND)[ut.COL_DELTA_PRED].to_numpy()
            assert np.all(np.diff(vals) >= -1e-9)

    def test_region_restriction(self, df_seq_pos, df_feat_multipart):
        df = aa.SeqMut().evolve(df_seq=df_seq_pos, df_feat=df_feat_multipart, depth=2,
                                region="tmd")
        positions = [int(m[1:-1]) for m in df[ut.COL_MUTATION]]
        assert all(11 <= p <= 20 for p in positions)

    def test_to_aa_restriction(self, df_seq_pos, df_feat):
        df = aa.SeqMut().evolve(df_seq=df_seq_pos, df_feat=df_feat, depth=2, region="tmd",
                                to_aa=["A", "G"])
        assert set(m[-1] for m in df[ut.COL_MUTATION]).issubset({"A", "G"})

    def test_jmd_lens(self, df_seq_pos, df_feat):
        df = aa.SeqMut().evolve(df_seq=df_seq_pos, df_feat=df_feat, depth=1, region="tmd",
                                jmd_n_len=8, jmd_c_len=8)
        assert len(df) >= 1

    # Negative cases
    def test_invalid_depth_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError):
            aa.SeqMut().evolve(df_seq=df_seq_pos, df_feat=df_feat, depth=0)

    def test_non_pos_df_seq_raises(self, df_feat):
        import pandas as pd
        bad = pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_SEQ: ["ACDEFGHIKL"]})
        with pytest.raises(ValueError):
            aa.SeqMut().evolve(df_seq=bad, df_feat=df_feat, depth=1)


class TestSeqMutEvolveGoldenValues:
    def test_first_round_matches_top_suggest(self, df_seq_pos, df_feat, model_tuple):
        # The first evolved mutation is the global single-mutation argmax = suggest's top hit.
        sm = aa.SeqMut(model=model_tuple)
        df_ev = sm.evolve(df_seq=df_seq_pos.head(1), df_feat=df_feat, depth=1, region="tmd")
        df_sug = sm.suggest(df_seq=df_seq_pos.head(1), df_feat=df_feat, n=1, region="tmd")
        assert df_ev[ut.COL_MUTATION].iloc[0] == df_sug[ut.COL_MUTATION].iloc[0]

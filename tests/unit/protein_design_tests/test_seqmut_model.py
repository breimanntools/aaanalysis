"""Tests for the model-aware (ML-guided) SeqMut: prediction-shift delta_pred on scan /
suggest / mutate, target_class handling, and model<->df_feat checks."""
import numpy as np
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

COLS_PRED = [ut.COL_DELTA_PRED, ut.COL_WT_PRED, ut.COL_WT_PRED_STD]


class TestSeqMutModelScan:
    def test_scan_adds_pred_columns(self, df_seq_pos, df_feat, model_tuple):
        df = aa.SeqMut(model=model_tuple).scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        for col in COLS_PRED:
            assert col in df.columns
        assert np.isfinite(df[ut.COL_DELTA_PRED]).all()

    def test_model_free_has_no_pred_columns(self, df_seq_pos, df_feat):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert list(df.columns) == ut.COLS_SEQMUT_SCAN
        assert ut.COL_DELTA_PRED not in df.columns

    def test_wt_pred_constant_per_entry(self, df_seq_pos, df_feat, model_tuple):
        df = aa.SeqMut(model=model_tuple).scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        for _entry, g in df.groupby(ut.COL_ENTRY):
            assert g[ut.COL_WT_PRED].nunique() == 1

    def test_tuple_model_reports_std(self, df_seq_pos, df_feat, model_tuple):
        df = aa.SeqMut(model=model_tuple).scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert np.isfinite(df[ut.COL_WT_PRED_STD]).all()

    def test_2d_model_has_nan_std(self, df_seq_pos, df_feat, model_2d):
        df = aa.SeqMut(model=model_2d).scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert df[ut.COL_WT_PRED_STD].isna().all()


class TestSeqMutModelSuggest:
    def test_suggest_ranks_by_delta_pred(self, df_seq_pos, df_feat, model_2d):
        df = aa.SeqMut(model=model_2d).suggest(df_seq=df_seq_pos, df_feat=df_feat, n=10,
                                               region="tmd")
        assert np.all(np.diff(df[ut.COL_DELTA_PRED].to_numpy()) <= 1e-9)

    def test_suggest_model_free_ranks_by_shift(self, df_seq_pos, df_feat):
        df = aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, n=10, region="tmd")
        assert np.all(np.diff(df[ut.COL_SHIFT_SCORE].to_numpy()) <= 1e-9)


class TestSeqMutModelMutate:
    def test_mutate_adds_delta_pred(self, df_seq_pos, df_feat, model_tuple):
        import pandas as pd
        muts = pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_POS: [11], ut.COL_TO_AA: ["A"]})
        df = aa.SeqMut(model=model_tuple).mutate(df_seq=df_seq_pos, mutations=muts, df_feat=df_feat)
        assert ut.COL_DELTA_PRED in df.columns


class TestSeqMutTargetClass:
    def test_target_class_zero_negates_delta(self, df_seq_pos, df_feat, model_2d):
        # For a 2-class model, P(class 0) = 1 - P(class 1), so the prediction shift flips sign.
        d1 = aa.SeqMut(model=model_2d, target_class=1).scan(df_seq=df_seq_pos, df_feat=df_feat,
                                                            region="tmd")
        d0 = aa.SeqMut(model=model_2d, target_class=0).scan(df_seq=df_seq_pos, df_feat=df_feat,
                                                            region="tmd")
        d1 = d1.set_index(ut.COL_MUTATION)[ut.COL_DELTA_PRED]
        d0 = d0.set_index(ut.COL_MUTATION)[ut.COL_DELTA_PRED]
        assert np.allclose(d0.loc[d1.index].to_numpy(), -d1.to_numpy(), atol=1e-9)

    def test_target_class_without_model_raises(self):
        with pytest.raises(ValueError):
            aa.SeqMut(target_class=1)

    def test_target_class_not_in_classes_raises(self, model_2d):
        with pytest.raises(ValueError):
            aa.SeqMut(model=model_2d, target_class=7)


class TestSeqMutModelChecks:
    def test_model_without_predict_proba_raises(self):
        with pytest.raises(ValueError):
            aa.SeqMut(model=object())

    def test_model_df_feat_n_features_mismatch_raises(self, df_seq_pos, df_feat, model_2d):
        df_feat2 = df_feat.head(3)  # model.n_features_in_ == 4 != 3
        with pytest.raises(ValueError):
            aa.SeqMut(model=model_2d).scan(df_seq=df_seq_pos, df_feat=df_feat2, region="tmd")


class TestSeqMutModelGoldenValues:
    def test_delta_pred_matches_independent_recompute(self, df_seq_pos, df_feat, model_tuple):
        # delta_pred must equal (P(mut) - P(wt)) * 100 recomputed via the public builder.
        sm = aa.SeqMut(model=model_tuple)
        df = sm.scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        row = df.iloc[0]
        entry, pos, to_aa = row[ut.COL_ENTRY], int(row[ut.COL_POS]), row[ut.COL_TO_AA]
        seq = df_seq_pos.set_index(ut.COL_ENTRY).loc[entry, ut.COL_SEQ]
        seq_mut = seq[:pos - 1] + to_aa + seq[pos:]
        import pandas as pd
        sf = aa.SequenceFeature()
        feats = list(df_feat[ut.COL_FEATURE])
        base = df_seq_pos[df_seq_pos[ut.COL_ENTRY] == entry]
        ts, te = int(base[ut.COL_TMD_START].iloc[0]), int(base[ut.COL_TMD_STOP].iloc[0])
        df_mut = pd.DataFrame({ut.COL_ENTRY: [entry], ut.COL_SEQ: [seq_mut],
                               ut.COL_TMD_START: [ts], ut.COL_TMD_STOP: [te]})
        ds = ut.load_default_scales()
        x_wt = sf.feature_matrix(features=feats, df_parts=sf.get_df_parts(df_seq=base), df_scales=ds)
        x_mut = sf.feature_matrix(features=feats, df_parts=sf.get_df_parts(df_seq=df_mut), df_scales=ds)
        p_wt, _ = model_tuple.predict_proba(np.asarray(x_wt, dtype=float))
        p_mut, _ = model_tuple.predict_proba(np.asarray(x_mut, dtype=float))
        expected = (p_mut[0] - p_wt[0]) * 100.0
        assert row[ut.COL_DELTA_PRED] == pytest.approx(expected, abs=1e-6)

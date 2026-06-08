"""This is a script to test the standardized CPP output schema (issue #18):
``CPP.run`` returns ``df_feat`` with a deterministic canonical column order
(``ut.LIST_COLS_FEAT``), the p-value column name tracks ``parametric``, and the
canonical order is a lower bound (downstream columns append, never displace).

A pure reorder must not change values, so this is a structural contract test;
byte-exact values are guarded separately by the ADR-0015 regression anchor.
"""
import warnings

import aaanalysis as aa
import aaanalysis.utils as ut


def _run(parametric=False, n_filter=10):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
    labels = df_seq["label"].to_list()
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    split_kws = sf.get_split_kws(n_split_min=1, n_split_max=2, split_types=["Segment"])
    df_scales = aa.load_scales().iloc[:, :15]
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws,
                 verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cpp.run(labels=labels, n_filter=n_filter, parametric=parametric)


class TestCppOutputSchema:
    def test_canonical_column_order(self):
        df_feat = _run(parametric=False)
        assert list(df_feat.columns) == list(ut.LIST_COLS_FEAT)

    def test_parametric_pval_column_name(self):
        df_feat = _run(parametric=True)
        # Same canonical order, but the p-value column name reflects the t-test,
        # occupying the exact slot the Mann-Whitney column would.
        expected = [ut.COL_PVAL_TTEST if c == ut.COL_PVAL_MW else c
                    for c in ut.LIST_COLS_FEAT]
        assert list(df_feat.columns) == expected

    def test_filter_stats_attrs_survive_reorder(self):
        df_feat = _run()
        assert "last_filter_stats" in df_feat.attrs
        assert df_feat.attrs["last_filter_stats"]["n_final"] == len(df_feat)

    def test_post_hoc_column_appends_after_positions(self):
        df_feat = _run()
        df_feat["feat_impact_APP"] = 0.0
        out = ut.sort_cols_feat(df_feat=df_feat)
        assert list(out.columns)[-1] == "feat_impact_APP"
        assert list(out.columns)[: len(ut.LIST_COLS_FEAT)] == list(ut.LIST_COLS_FEAT)

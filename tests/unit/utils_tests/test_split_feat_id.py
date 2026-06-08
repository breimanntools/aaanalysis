"""This is a script to test the canonical df_feat schema helpers in utils.py
(``split_feat_id`` / ``join_feat_id`` / ``sort_cols_feat``), introduced for the
standardized CPP output schema (issue #18).

These are pure helpers exposed via ``ut``; tested directly. ``sort_cols_feat`` is
the tolerant column-reorder that every CPP output passes through, so its
never-drop / stable-append contract is pinned here.
"""
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

# Component strings for a feature id: any non-empty text without the '-' delimiter.
_component = st.text(
    alphabet=st.characters(blacklist_characters="-", blacklist_categories=("Cs",)),
    min_size=1,
).filter(lambda s: s.strip() != "")


class TestSplitJoinFeatId:
    def test_golden_split(self):
        fid = "TMD-Segment(1,5)-Hydrophobicity_Eisenberg"
        assert ut.split_feat_id(feat_id=fid) == (
            "TMD", "Segment(1,5)", "Hydrophobicity_Eisenberg")

    def test_golden_join(self):
        out = ut.join_feat_id(
            part="TMD", split="Segment(1,5)", scale_id="Hydrophobicity_Eisenberg")
        assert out == "TMD-Segment(1,5)-Hydrophobicity_Eisenberg"

    def test_join_does_not_uppercase(self):
        # Casing is the caller's responsibility — round-trips must stay byte-identical.
        assert ut.join_feat_id(part="tmd", split="S", scale_id="x") == "tmd-S-x"

    @given(part=_component, split=_component, scale=_component)
    def test_round_trip(self, part, split, scale):
        feat_id = ut.join_feat_id(part=part, split=split, scale_id=scale)
        assert ut.split_feat_id(feat_id=feat_id) == (part, split, scale)

    def test_split_wrong_component_count_raises(self):
        # Pure split — exactly three components; structural validity is
        # checked separately in check_features.
        with pytest.raises(ValueError):
            ut.split_feat_id(feat_id="TMD-Segment(1,5)")


class TestSortColsFeat:
    def _canonical_frame(self):
        return pd.DataFrame(columns=list(ut.LIST_COLS_FEAT))

    def test_canonical_order_is_idempotent(self):
        df = self._canonical_frame()
        assert list(ut.sort_cols_feat(df_feat=df).columns) == list(ut.LIST_COLS_FEAT)

    def test_reorders_shuffled_known_columns(self):
        df = pd.DataFrame(columns=list(reversed(ut.LIST_COLS_FEAT)))
        assert list(ut.sort_cols_feat(df_feat=df).columns) == list(ut.LIST_COLS_FEAT)

    def test_ttest_pval_slots_into_canonical_position(self):
        # parametric=True swaps the p-value column NAME; it must occupy the MW slot.
        cols = [c for c in ut.LIST_COLS_FEAT if c != ut.COL_PVAL_MW]
        cols.insert(cols.index(ut.COL_PVAL_FDR), ut.COL_PVAL_TTEST)
        df = pd.DataFrame(columns=list(reversed(cols)))
        out = list(ut.sort_cols_feat(df_feat=df).columns)
        assert out.index(ut.COL_PVAL_TTEST) == out.index(ut.COL_PVAL_FDR) - 1

    def test_extra_columns_appended_stably_never_dropped(self):
        # Post-hoc + per-substrate SHAP columns must survive after 'positions'.
        extras = ["mean_dif_APP", "feat_importance", "feat_impact_APP"]
        df = pd.DataFrame(columns=extras[:1] + list(ut.LIST_COLS_FEAT) + extras[1:])
        out = list(ut.sort_cols_feat(df_feat=df).columns)
        assert out[: len(ut.LIST_COLS_FEAT)] == list(ut.LIST_COLS_FEAT)
        assert out[len(ut.LIST_COLS_FEAT):] == extras  # stable order, none dropped
        assert set(out) == set(df.columns)

    def test_substrate_mean_dif_does_not_collide_with_canonical(self):
        # 'mean_dif_APP' must NOT be matched as the canonical 'mean_dif'.
        df = pd.DataFrame(columns=["mean_dif_APP"] + list(ut.LIST_COLS_FEAT))
        out = list(ut.sort_cols_feat(df_feat=df).columns)
        assert out[-1] == "mean_dif_APP"
        assert out.index(ut.COL_MEAN_DIF) < out.index("mean_dif_APP")

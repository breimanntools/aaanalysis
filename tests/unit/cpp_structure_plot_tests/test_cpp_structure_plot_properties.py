"""Property/invariant + golden tests for the CPPStructurePlot family.

House-standard depth complementing the per-method files: hypothesis property tests for the
model-dependent paths (invariants stay true across scikit-learn / SHAP versions, unlike exact
SHAP values which the repo pins to the nightly ``regression`` anchor) and exact golden tests for
the deterministic, pure pieces (the residue-impact mapping and the feature-map column geometry).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck, strategies as st

# Pro-gated: structure parsing needs biopython, rendering needs py3Dmol, impact needs SHAP.
pytest.importorskip("Bio")
pytest.importorskip("py3Dmol")
pytest.importorskip("shap")

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering_pro._backend.cpp_struct.mapping import compute_residue_impact
from aaanalysis.feature_engineering_pro._backend.cpp_struct.predict import (
    build_builtin_predictor, resolve_estimators)
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

_AAS = list("ACDEFGHIKLMNPQRSTVWY")
_NO_FIXTURE_HC = [HealthCheck.function_scoped_fixture]


# --- fixtures ----------------------------------------------------------------
@pytest.fixture(scope="module")
def df_scales():
    return aa.load_scales(name="scales")


@pytest.fixture(scope="module")
def df_feat(df_scales):
    """A complete CPP-style df_feat (real scales/categories)."""
    df_cat = aa.load_scales(name="scales_cat")
    df_cat = df_cat[df_cat[ut.COL_SCALE_ID].isin(df_scales.columns)].head(4).reset_index(drop=True)
    ids = df_cat[ut.COL_SCALE_ID].tolist()
    splits = ["Segment(1,2)", "Segment(2,2)", "Segment(1,1)", "Pattern(C,1)"]
    parts = ["TMD", "TMD", "JMD_N", "TMD"]
    return pd.DataFrame({
        ut.COL_FEATURE: [f"{parts[i]}-{splits[i]}-{ids[i]}" for i in range(4)],
        ut.COL_CAT: df_cat[ut.COL_CAT], ut.COL_SUBCAT: df_cat[ut.COL_SUBCAT],
        ut.COL_SCALE_NAME: df_cat[ut.COL_SCALE_NAME],
        "abs_auc": [0.2, 0.15, 0.3, 0.1], "abs_mean_dif": [0.3, 0.2, 0.5, 0.4],
        "mean_dif": [0.3, -0.2, 0.5, -0.4], "std_test": 0.1, "std_ref": 0.1,
        "feat_impact": [0.8, -0.5, 1.2, -0.3]})   # present so plot_linked can render directly


@pytest.fixture(scope="module")
def predictor(df_feat, df_scales):
    rng = np.random.default_rng(0)
    df_seq = pd.DataFrame([{ut.COL_ENTRY: f"P{i:03d}",
                            ut.COL_SEQ: "".join(rng.choice(_AAS, size=60)),
                            ut.COL_TMD_START: 21, ut.COL_TMD_STOP: 30} for i in range(16)])
    return build_builtin_predictor(df_feat=df_feat, df_seq=df_seq, labels=[1, 0] * 8, tmd_len=10,
                                   jmd_n_len=10, jmd_c_len=10, df_scales=df_scales, random_state=42)


@pytest.fixture(scope="module")
def query_seq():
    return "".join(np.random.default_rng(1).choice(_AAS, size=60))


# --- golden: compute_residue_impact (pure, deterministic, version-stable) ----
class TestResidueImpactGolden:
    """Exact mapping of feature impact onto residues (no model, so exact values are safe)."""

    def test_whole_tmd_segment(self):
        df = pd.DataFrame({ut.COL_FEATURE: ["TMD-Segment(1,1)-X"], "feat_impact": [2.0]})
        di, mx, pos = compute_residue_impact(df_feat=df, col_imp="feat_impact", start=1,
                                             tmd_len=10, jmd_n_len=10, jmd_c_len=10,
                                             normalize_by_span=False)
        # TMD spans residues 11..20 (start=1, jmd_n_len=10); each gets the full +2.0.
        assert pos == list(range(11, 21))
        assert mx == 2.0
        assert all(di[r] == 2.0 for r in range(11, 21))
        assert all(di[r] == 0.0 for r in list(range(1, 11)) + list(range(21, 31)))

    def test_two_half_segments_opposite_sign(self):
        df = pd.DataFrame({ut.COL_FEATURE: ["TMD-Segment(1,2)-A", "TMD-Segment(2,2)-B"],
                           "feat_impact": [1.0, -3.0]})
        di, mx, _pos = compute_residue_impact(df_feat=df, col_imp="feat_impact", start=1,
                                              tmd_len=10, jmd_n_len=10, jmd_c_len=10,
                                              normalize_by_span=False)
        assert all(di[r] == 1.0 for r in range(11, 16))    # first half of the TMD
        assert all(di[r] == -3.0 for r in range(16, 21))   # second half
        assert mx == 3.0                                   # max(|+1|, |-3|)

    def test_start_offset_shifts_positions(self):
        df = pd.DataFrame({ut.COL_FEATURE: ["TMD-Segment(1,1)-X"], "feat_impact": [1.0]})
        _di, _mx, pos = compute_residue_impact(df_feat=df, col_imp="feat_impact", start=41,
                                               tmd_len=10, jmd_n_len=10, jmd_c_len=10,
                                               normalize_by_span=False)
        assert pos == list(range(51, 61))   # start 41 + jmd_n 10 -> TMD at 51..60


# --- property: compute_residue_impact invariants -----------------------------
class TestResidueImpactProperties:
    @settings(max_examples=25, deadline=None)
    @given(imp=st.lists(st.floats(min_value=-9, max_value=9, allow_nan=False),
                        min_size=1, max_size=4))
    def test_max_abs_matches_painted_residues(self, imp):
        feats = ["TMD-Segment(1,1)-X", "JMD_N-Segment(1,1)-Y", "TMD-Segment(1,2)-Z",
                 "TMD-Segment(2,2)-W"][:len(imp)]
        df = pd.DataFrame({ut.COL_FEATURE: feats, "feat_impact": imp})
        di, mx, pos = compute_residue_impact(df_feat=df, col_imp="feat_impact", start=1,
                                             tmd_len=10, jmd_n_len=10, jmd_c_len=10,
                                             normalize_by_span=False)
        painted = [v for v in di.values() if v != 0]
        assert mx == pytest.approx(max((abs(v) for v in painted), default=0.0))
        assert all(1 <= r <= 30 for r in pos)   # within the jmd_n+tmd+jmd_c window


# --- property: build_builtin_predictor invariants (robust to version drift) ---
class TestPredictorProperties:
    @settings(max_examples=8, deadline=None, suppress_health_check=_NO_FIXTURE_HC)
    @given(p1=st.integers(min_value=11, max_value=41))
    def test_impact_normalized_and_proba_bounded(self, predictor, query_seq, p1):
        out = predictor(query_seq, p1)
        assert ut.COL_FEAT_IMPACT in out.columns
        # normalized to percentages summing to 100 (per-feature rounding -> small tolerance)
        assert out[ut.COL_FEAT_IMPACT].abs().sum() == pytest.approx(100.0, abs=0.5)
        assert np.isfinite(out[ut.COL_FEAT_IMPACT]).all()
        assert 0.0 <= out.attrs["proba"] <= 1.0

    @settings(max_examples=6, deadline=None, suppress_health_check=_NO_FIXTURE_HC)
    @given(p1=st.integers(min_value=11, max_value=41))
    def test_deterministic_per_site(self, predictor, query_seq, p1):
        a, b = predictor(query_seq, p1), predictor(query_seq, p1)
        assert np.allclose(a[ut.COL_FEAT_IMPACT], b[ut.COL_FEAT_IMPACT])


# --- property: resolve_estimators --------------------------------------------
_MODEL_NAMES = [ut.MODEL_RF, ut.MODEL_SVM, ut.MODEL_LOG_REG, "extra_trees"]


class TestResolveEstimatorsProperties:
    @settings(max_examples=20, deadline=None)
    @given(names=st.lists(st.sampled_from(_MODEL_NAMES), min_size=1, max_size=4))
    def test_resolves_each_name(self, names):
        est = resolve_estimators(names, random_state=3)
        assert len(est) == len(names)
        assert all(isinstance(e, BaseEstimator) for e in est)

    @settings(max_examples=10, deadline=None)
    @given(rs=st.integers(min_value=0, max_value=9999))
    def test_random_state_injected(self, rs):
        assert resolve_estimators(ut.MODEL_RF, random_state=rs)[0].random_state == rs


# --- golden: feature-map column geometry (plot_linked) -----------------------
def _make_pdb(path, n=30, chain="A"):
    lines = []
    for i in range(n):
        x, y, z = i * 1.5, np.sin(i * 0.5) * 6, np.cos(i * 0.5) * 6
        lines.append(f"ATOM  {i + 1:5d}  CA  ALA {chain}{i + 1:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{40 + (i % 60):6.2f}           C")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def pdb_path(tmp_path_factory):
    return _make_pdb(tmp_path_factory.mktemp("struct_props") / "synthetic.pdb")


class TestColumnGeometry:
    """The feature map's per-column hover strips map column i -> residue start+i, in order."""

    def test_columns_cover_window_in_order(self, pdb_path, df_feat):
        import re
        view = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False).plot_linked(
            df_feat=df_feat, pdb=pdb_path, tmd_len=10, start=1, col_imp="feat_impact")
        resis = [int(x) for x in re.findall(r'data-resi="(\d+)"', view._repr_html_())]
        assert resis == list(range(1, 31))   # start=1 .. start + (jmd_n+tmd+jmd_c) - 1

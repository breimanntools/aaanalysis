"""This is a script to test AnnotationPreprocessor (PTM / functional-site
per-residue annotation layer): the UniProt JSON→schema mapper, the encoder
(residue-identity guard + 0.0/NaN fill + bond expansion), the user-ingestion
auto-register path, df_cat/colors registration, and the end-to-end
NumericalFeature.get_parts → CPP.run_num integration.

Network is never hit: the mapper is exercised against hand-built UniProtKB
JSON records, and the encoder/integration against hand-built df_annot tables.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.data_handling_pro import AnnotationPreprocessor
from aaanalysis.data_handling_pro._backend.annot_preproc._uniprot import (
    map_record_to_rows,
)

aa.options["verbose"] = False

SEQ = "MKSTYACDEFGHIKLCNPQRS"  # len 21; C at 7 and 16; S at 3, 21; T at 4


def _record(features):
    return {"sequence": {"value": SEQ}, "features": features}


def _feat(ftype, start, end, desc="", eco="ECO:0000269"):
    ev = [{"evidenceCode": eco}] if eco else []
    return {
        "type": ftype,
        "description": desc,
        "location": {"start": {"value": start}, "end": {"value": end}},
        "evidences": ev,
    }


def _df_seq():
    return pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_SEQ: [SEQ]})


def _annot_rows(rows):
    return pd.DataFrame(rows, columns=ut.COLS_ANNOT)


# ---------------------------------------------------------------------------
# Registry + build_cat / build_scales
# ---------------------------------------------------------------------------
class TestRegistryAndMetadata:
    def test_builtin_keys_present(self):
        ap = AnnotationPreprocessor(verbose=False)
        for key in [
            "phospho",
            "glyco_n",
            "glyco_o",
            "lipid",
            "disulfide",
            "crosslink",
            "signal_cleavage",
            "cleavage_site",
            "binding",
            "act_site",
            "dna_bind",
        ]:
            assert key in ap._registry

    def test_build_cat_categories(self):
        ap = AnnotationPreprocessor(verbose=False)
        df_cat = ap.build_cat(features=["phospho", "disulfide", "binding"])
        cats = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_CAT]))
        assert cats["phospho"] == "PTMs"
        assert cats["disulfide"] == "PTMs"
        assert cats["binding"] == "Functional sites"

    def test_build_cat_shape_and_columns(self):
        ap = AnnotationPreprocessor(verbose=False)
        df_cat = ap.build_cat(features=["phospho", "binding"])
        assert len(df_cat) == 2
        for col in (
            ut.COL_SCALE_ID,
            ut.COL_CAT,
            ut.COL_SUBCAT,
            ut.COL_SCALE_NAME,
            ut.COL_SCALE_DES,
        ):
            assert col in df_cat.columns

    def test_build_cat_invalid_feature_raises(self):
        ap = AnnotationPreprocessor(verbose=False)
        for bad in [["not_a_key"], [], "phospho", None]:
            with pytest.raises(ValueError):
                ap.build_cat(features=bad)

    def test_build_scales_per_aa_mean(self):
        ap = AnnotationPreprocessor(verbose=False)
        df_seq = _df_seq()
        df_annot = _annot_rows(
            [["P1", 3, 3, "S", "phospho", "PTMs", "UniProt", "", 1.0, None]]
        )
        dn = ap.encode(df_seq=df_seq, df_annot=df_annot, features=["phospho"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfs = ap.build_scales(
                df_seq=df_seq, dict_num=dn, features=["phospho"]
            )
        assert dfs.shape == (20, 1)
        # Two S residues (pos 3, 21); only pos 3 is phospho → mean 0.5
        assert abs(float(dfs.loc["S", "phospho"]) - 0.5) < 1e-9

    def test_build_scales_requires_corpus(self):
        ap = AnnotationPreprocessor(verbose=False)
        with pytest.raises(ValueError):
            ap.build_scales(df_seq=None, dict_num=None, features=["phospho"])


# ---------------------------------------------------------------------------
# UniProt JSON → df_annot mapper
# ---------------------------------------------------------------------------
class TestUniProtMapper:
    def test_phospho_routing(self):
        data = _record([_feat("Modified residue", 3, 3, "Phosphoserine")])
        df = _annot_rows(map_record_to_rows("P1", data, None, ut.LIST_ECO_MANUAL))
        assert df[ut.COL_FEATURE_TYPE].tolist() == ["phospho"]
        assert df[ut.COL_AA].tolist() == ["S"]

    def test_mod_res_other_routing(self):
        data = _record([_feat("Modified residue", 2, 2, "N6-acetyllysine")])
        df = _annot_rows(map_record_to_rows("P1", data, None, ut.LIST_ECO_MANUAL))
        assert df[ut.COL_FEATURE_TYPE].tolist() == ["mod_res_other"]

    def test_glyco_n_vs_o(self):
        data = _record(
            [
                _feat("Glycosylation", 8, 8, "N-linked (GlcNAc) asparagine"),
                _feat("Glycosylation", 3, 3, "O-linked (GalNAc) serine"),
            ]
        )
        df = _annot_rows(map_record_to_rows("P1", data, None, ut.LIST_ECO_MANUAL))
        assert set(df[ut.COL_FEATURE_TYPE]) == {"glyco_n", "glyco_o"}

    def test_disulfide_expands_to_two_endpoints_with_bond_id(self):
        data = _record([_feat("Disulfide bond", 7, 16)])
        df = _annot_rows(map_record_to_rows("P1", data, None, ut.LIST_ECO_MANUAL))
        dis = df[df[ut.COL_FEATURE_TYPE] == "disulfide"]
        assert len(dis) == 2
        assert sorted(dis[ut.COL_START].tolist()) == [7, 16]
        assert dis[ut.COL_BOND_ID].nunique() == 1
        assert dis[ut.COL_BOND_ID].iloc[0] is not None

    def test_signal_cleavage_uses_span_end(self):
        data = _record([_feat("Signal", 1, 5)])
        df = _annot_rows(map_record_to_rows("P1", data, None, ut.LIST_ECO_MANUAL))
        assert df[ut.COL_FEATURE_TYPE].tolist() == ["signal_cleavage"]
        assert df[ut.COL_START].tolist() == [5]

    def test_site_cleavage_regex_keeps_only_cleavage(self):
        data = _record(
            [
                _feat("Site", 10, 10, "Cleavage; by thrombin"),
                _feat("Site", 12, 12, "Important for catalytic activity"),
            ]
        )
        df = _annot_rows(map_record_to_rows("P1", data, None, ut.LIST_ECO_MANUAL))
        assert df[ut.COL_FEATURE_TYPE].tolist() == ["cleavage_site"]
        assert df[ut.COL_START].tolist() == [10]

    def test_evidence_filter_drops_by_similarity(self):
        # ECO:0000250 (by similarity) must be dropped under the manual set
        data = _record(
            [_feat("Modified residue", 4, 4, "Phosphothreonine", eco="ECO:0000250")]
        )
        df = _annot_rows(map_record_to_rows("P1", data, None, ut.LIST_ECO_MANUAL))
        assert len(df) == 0

    def test_evidence_filter_keeps_combinatorial(self):
        data = _record(
            [_feat("Modified residue", 3, 3, "Phosphoserine", eco="ECO:0007744")]
        )
        df = _annot_rows(map_record_to_rows("P1", data, None, ut.LIST_ECO_MANUAL))
        assert df[ut.COL_FEATURE_TYPE].tolist() == ["phospho"]

    def test_evidence_experimental_excludes_combinatorial(self):
        data = _record(
            [_feat("Modified residue", 3, 3, "Phosphoserine", eco="ECO:0007744")]
        )
        df = _annot_rows(map_record_to_rows("P1", data, None, ut.LIST_ECO_EXPERIMENTAL))
        assert len(df) == 0

    def test_evidence_all_disables_filter(self):
        data = _record(
            [_feat("Modified residue", 4, 4, "Phosphothreonine", eco="ECO:0000250")]
        )
        df = _annot_rows(map_record_to_rows("P1", data, None, None))
        assert df[ut.COL_FEATURE_TYPE].tolist() == ["phospho"]

    def test_allowed_features_filters_keys(self):
        data = _record(
            [
                _feat("Modified residue", 3, 3, "Phosphoserine"),
                _feat("Disulfide bond", 7, 16),
            ]
        )
        df = _annot_rows(
            map_record_to_rows("P1", data, ["phospho"], ut.LIST_ECO_MANUAL)
        )
        assert set(df[ut.COL_FEATURE_TYPE]) == {"phospho"}

    def test_binding_routed_to_functional(self):
        data = _record([_feat("Binding site", 9, 9, "")])
        df = _annot_rows(map_record_to_rows("P1", data, None, ut.LIST_ECO_MANUAL))
        assert df[ut.COL_FEATURE_TYPE].tolist() == ["binding"]
        assert df[ut.COL_CAT].tolist() == ["Functional sites"]


# ---------------------------------------------------------------------------
# encode: fill semantics, guard, bond profiling
# ---------------------------------------------------------------------------
class TestEncode:
    def test_shape_and_absent_is_zero(self):
        ap = AnnotationPreprocessor(verbose=False)
        df_annot = _annot_rows(
            [["P1", 3, 3, "S", "phospho", "PTMs", "UniProt", "", 1.0, None]]
        )
        dn = ap.encode(df_seq=_df_seq(), df_annot=df_annot, features=["phospho"])
        arr = dn["P1"]
        assert arr.shape == (21, 1)
        assert arr[2, 0] == 1.0  # pos 3 annotated
        assert arr[0, 0] == 0.0  # pos 1 absent → 0.0, not NaN
        assert not np.isnan(arr).any()

    def test_two_features_two_columns(self):
        ap = AnnotationPreprocessor(verbose=False)
        df_annot = _annot_rows(
            [
                ["P1", 3, 3, "S", "phospho", "PTMs", "UniProt", "", 1.0, None],
                ["P1", 7, 7, "C", "disulfide", "PTMs", "UniProt", "", 1.0, "b1"],
                ["P1", 16, 16, "C", "disulfide", "PTMs", "UniProt", "", 1.0, "b1"],
            ]
        )
        dn = ap.encode(
            df_seq=_df_seq(), df_annot=df_annot, features=["phospho", "disulfide"]
        )
        arr = dn["P1"]
        assert arr.shape == (21, 2)
        assert list(np.where(arr[:, 0] > 0)[0] + 1) == [3]
        assert list(np.where(arr[:, 1] > 0)[0] + 1) == [7, 16]

    def test_guard_raises_on_mismatch(self):
        ap = AnnotationPreprocessor(verbose=False)
        bad = _annot_rows(
            [["P1", 4, 4, "W", "phospho", "PTMs", "UniProt", "", 1.0, None]]
        )
        with pytest.raises(ValueError):
            ap.encode(
                df_seq=_df_seq(),
                df_annot=bad,
                features=["phospho"],
                on_mismatch="raise",
            )

    def test_guard_drop_skips_row(self):
        ap = AnnotationPreprocessor(verbose=False)
        bad = _annot_rows(
            [["P1", 4, 4, "W", "phospho", "PTMs", "UniProt", "", 1.0, None]]
        )
        dn = ap.encode(
            df_seq=_df_seq(), df_annot=bad, features=["phospho"], on_mismatch="drop"
        )
        assert dn["P1"].sum() == 0.0

    def test_guard_warn_emits_warning(self):
        ap = AnnotationPreprocessor(verbose=False)
        bad = _annot_rows(
            [["P1", 4, 4, "W", "phospho", "PTMs", "UniProt", "", 1.0, None]]
        )
        with pytest.warns(UserWarning):
            ap.encode(
                df_seq=_df_seq(), df_annot=bad, features=["phospho"], on_mismatch="warn"
            )

    def test_empty_aa_skips_guard(self):
        ap = AnnotationPreprocessor(verbose=False)
        # aa="" → no guard even though pos 4 is T, label here is arbitrary
        df_annot = _annot_rows(
            [["P1", 4, 4, "", "phospho", "PTMs", "user", "", 1.0, None]]
        )
        dn = ap.encode(df_seq=_df_seq(), df_annot=df_annot, features=["phospho"])
        assert dn["P1"][3, 0] == 1.0

    def test_score_value_preserved(self):
        ap = AnnotationPreprocessor(verbose=False)
        ap.register_feature(key="hotspot")
        df_annot = _annot_rows(
            [["P1", 5, 5, "", "hotspot", "Functional sites", "rfdiff", "", 0.42, None]]
        )
        dn = ap.encode(df_seq=_df_seq(), df_annot=df_annot, features=["hotspot"])
        assert abs(dn["P1"][4, 0] - 0.42) < 1e-9


# ---------------------------------------------------------------------------
# ingest: user path + auto-register
# ---------------------------------------------------------------------------
class TestIngest:
    def test_ingest_auto_registers_and_marks_functional(self):
        ap = AnnotationPreprocessor(verbose=False)
        df_user = pd.DataFrame(
            {
                ut.COL_PROTEIN_ID: ["P1", "P1"],
                ut.COL_START: [5, 9],
                ut.COL_FEATURE_TYPE: ["hotspot", "hotspot"],
                ut.COL_SCORE: [0.9, 0.4],
            }
        )
        da = ap.ingest(df_user)
        assert "hotspot" in ap._registry
        assert da[ut.COL_CAT].unique().tolist() == ["Functional sites"]
        assert set(ut.COLS_ANNOT).issubset(da.columns)

    def test_ingest_missing_columns_raises(self):
        ap = AnnotationPreprocessor(verbose=False)
        with pytest.raises(ValueError):
            ap.ingest(pd.DataFrame({ut.COL_PROTEIN_ID: ["P1"]}))

    def test_ingest_out_of_range_score_raises(self):
        ap = AnnotationPreprocessor(verbose=False)
        df_user = pd.DataFrame(
            {
                ut.COL_PROTEIN_ID: ["P1"],
                ut.COL_START: [5],
                ut.COL_FEATURE_TYPE: ["hotspot"],
                ut.COL_SCORE: [1.5],
            }
        )
        with pytest.raises(ValueError):
            ap.ingest(df_user)

    def test_ingest_defaults_source_and_score(self):
        ap = AnnotationPreprocessor(verbose=False)
        df_user = pd.DataFrame(
            {
                ut.COL_PROTEIN_ID: ["P1"],
                ut.COL_START: [5],
                ut.COL_FEATURE_TYPE: ["iface"],
            }
        )
        da = ap.ingest(df_user)
        assert da[ut.COL_SOURCE].iloc[0] == "user"
        assert da[ut.COL_SCORE].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Color / category registration
# ---------------------------------------------------------------------------
class TestColorRegistration:
    def test_both_categories_in_dict_color_cat(self):
        assert ut.DICT_COLOR_CAT["PTMs"] == "#B36BCB"
        assert ut.DICT_COLOR_CAT["Functional sites"] == "#2C6E9E"

    def test_categories_in_list_cat(self):
        assert "PTMs" in ut.LIST_CAT
        assert "Functional sites" in ut.LIST_CAT

    def test_no_color_collision_with_embeddings(self):
        assert ut.DICT_COLOR_CAT["Functional sites"] != ut.DICT_COLOR_CAT["Embeddings"]

    def test_cpp_plot_validator_accepts_new_categories(self):
        from aaanalysis.feature_engineering._cpp_plot import (
            check_match_dict_color_list_cat,
        )

        out = check_match_dict_color_list_cat(
            dict_color=ut.DICT_COLOR_CAT, list_cat=["PTMs", "Functional sites"]
        )
        assert set(out) == {"PTMs", "Functional sites"}


# ---------------------------------------------------------------------------
# End-to-end: encode → get_parts → CPP.run_num (sparse-presence corpus)
# ---------------------------------------------------------------------------
def _build_sparse_corpus(n_per_label=5, L=40, seed=0):
    """Labeled df_seq + df_annot where label-1 carries phospho on S/T in the
    TMD slice (11-30) far more often than label-0 — a sparse-presence signal
    that must survive run_num's max_std_test pre-filter."""
    rng = np.random.default_rng(seed)
    aas = list(ut.LIST_CANONICAL_AA)
    rows, annot = [], []
    for label in (0, 1):
        for k in range(n_per_label):
            entry = f"A{label}_{k}"
            seq = "".join(rng.choice(aas, size=L))
            rows.append(
                {
                    ut.COL_ENTRY: entry,
                    ut.COL_SEQ: seq,
                    "label": label,
                    "tmd_start": 11,
                    "tmd_stop": 30,
                }
            )
            for i, ch in enumerate(seq):
                pos = i + 1
                if ch in "ST" and 11 <= pos <= 30:
                    p = 0.8 if label == 1 else 0.1
                    if rng.random() < p:
                        annot.append(
                            [
                                entry,
                                pos,
                                pos,
                                ch,
                                "phospho",
                                "PTMs",
                                "user",
                                "",
                                1.0,
                                None,
                            ]
                        )
    df_seq = pd.DataFrame(rows)
    df_annot = pd.DataFrame(annot, columns=ut.COLS_ANNOT)
    return df_seq, df_annot


class TestEndToEndRunNum:
    def test_pipeline_produces_ptm_features(self):
        df_seq, df_annot = _build_sparse_corpus()
        ap = AnnotationPreprocessor(verbose=False)
        feats = ["phospho"]
        dn = ap.encode(df_seq=df_seq, df_annot=df_annot, features=feats)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = ap.build_scales(
                df_seq=df_seq, dict_num=dn, features=feats
            )
        df_cat = ap.build_cat(features=feats)
        # D invariant
        D = next(iter(dn.values())).shape[1]
        assert D == len(df_scales.columns) == len(df_cat)
        nf = aa.NumericalFeature()
        df_parts, dnp = nf.get_parts(df_seq=df_seq, dict_num=dn)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, df_cat=df_cat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_feat = cpp.run_num(
                dict_num_parts=dnp,
                labels=df_seq["label"].tolist(),
                n_filter=10,
                n_jobs=1,
            )
        assert isinstance(df_feat, pd.DataFrame)
        assert len(df_feat) >= 1
        assert (df_feat[ut.COL_CAT] == "PTMs").all()

    def test_combine_dict_nums_roundtrip(self):
        df_seq, df_annot = _build_sparse_corpus(n_per_label=2)
        ap = AnnotationPreprocessor(verbose=False)
        dn = ap.encode(df_seq=df_seq, df_annot=df_annot, features=["phospho"])
        combined = aa.combine_dict_nums([dn, dn])
        # Two stacked copies → 2 columns, same entry set and L
        for entry, arr in combined.items():
            assert arr.shape[1] == 2
            assert arr.shape[0] == len(
                df_seq.set_index(ut.COL_ENTRY).loc[entry, ut.COL_SEQ]
            )


# ---------------------------------------------------------------------------
# to_df_seq: window-split export for AAWindowSampler
# ---------------------------------------------------------------------------
# seq positions: M1 K2 S3 T4 Y5 A6 C7 D8 E9 S10 G11 H12 I13 K14 L15 S16 N17
#                P18 Q19 R20 S21 T22  → S at 3,10,16,21; T at 4,22; C at 7
TODFSEQ_SEQ = "MKSTYACDESGHIKLSNPQRST"


def _df_seq_tds():
    return pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_SEQ: [TODFSEQ_SEQ]})


def _annot_tds():
    # phospho at S3 (positive); glyco_o at T4 (other annotation); disulfide C7
    return _annot_rows(
        [
            ["P1", 3, 3, "S", "phospho", "PTMs", "UniProt", "", 1.0, None],
            ["P1", 4, 4, "T", "glyco_o", "PTMs", "UniProt", "", 1.0, None],
            ["P1", 7, 7, "C", "disulfide", "PTMs", "UniProt", "", 1.0, "b1"],
        ]
    )


def _eligible_positions(ctx):
    return [i + 1 for i, c in enumerate(ctx) if c == "1"]


class TestToDfSeq:
    def test_pos_column_holds_feature_positives(self):
        ap = AnnotationPreprocessor(verbose=False)
        out = ap.to_df_seq(_df_seq_tds(), _annot_tds(), feature_type="phospho")
        assert out[ut.COL_POS].iloc[0] == [3]

    def test_residue_type_matched_and_contamination_excluded(self):
        ap = AnnotationPreprocessor(verbose=False)
        out = ap.to_df_seq(_df_seq_tds(), _annot_tds(), feature_type="phospho")
        elig = _eligible_positions(out["aa_context"].iloc[0])
        # phospho positives are on S → only non-annotated S residues remain
        assert all(TODFSEQ_SEQ[p - 1] == "S" for p in elig)
        assert 3 not in elig  # the positive itself
        assert 4 not in elig  # glyco_o (other annotation), and not S
        assert set(elig) == {10, 16, 21}

    def test_no_match_residue_type_keeps_all_non_annotated(self):
        ap = AnnotationPreprocessor(verbose=False)
        out = ap.to_df_seq(
            _df_seq_tds(),
            _annot_tds(),
            feature_type="phospho",
            match_residue_type=False,
        )
        elig = _eligible_positions(out["aa_context"].iloc[0])
        assert all(p not in elig for p in (3, 4, 7))  # annotated excluded
        assert len(elig) == len(TODFSEQ_SEQ) - 3  # everything else

    def test_keep_other_annotations_when_disabled(self):
        ap = AnnotationPreprocessor(verbose=False)
        out = ap.to_df_seq(
            _df_seq_tds(),
            _annot_tds(),
            feature_type="phospho",
            match_residue_type=False,
            exclude_other_annotations=False,
        )
        elig = _eligible_positions(out["aa_context"].iloc[0])
        assert 4 in elig and 7 in elig  # glyco_o / disulfide now eligible
        assert 3 not in elig  # the positive stays excluded

    def test_context_mask_length_equals_sequence(self):
        ap = AnnotationPreprocessor(verbose=False)
        out = ap.to_df_seq(_df_seq_tds(), _annot_tds(), feature_type="phospho")
        assert len(out["aa_context"].iloc[0]) == len(TODFSEQ_SEQ)

    def test_column_collision_raises(self):
        ap = AnnotationPreprocessor(verbose=False)
        df_seq = _df_seq_tds()
        df_seq[ut.COL_POS] = [[1]]
        with pytest.raises(ValueError):
            ap.to_df_seq(df_seq, _annot_tds(), feature_type="phospho")

    def test_no_positives_warns_and_empty_pos(self):
        ap = AnnotationPreprocessor(verbose=False)
        with pytest.warns(UserWarning):
            out = ap.to_df_seq(_df_seq_tds(), _annot_tds(), feature_type="dna_bind")
        assert out[ut.COL_POS].iloc[0] == []

    def test_custom_column_names(self):
        ap = AnnotationPreprocessor(verbose=False)
        out = ap.to_df_seq(
            _df_seq_tds(),
            _annot_tds(),
            feature_type="phospho",
            pos_col="positives",
            aa_context_col="ctx",
        )
        assert "positives" in out.columns and "ctx" in out.columns

    def test_end_to_end_window_sampler_residue_matched(self):
        ap = AnnotationPreprocessor(verbose=False)
        out = ap.to_df_seq(_df_seq_tds(), _annot_tds(), feature_type="phospho")
        aws = aa.AAWindowSampler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_ref = aws.sample_same_protein(
                df_seq=out,
                n=10,
                pos_col=ut.COL_POS,
                window_size=5,
                aa_context_col="aa_context",
                context_in="1",
                min_distance_to_pos=1,
                seed=0,
            )
        anchors = [int(p) for p in df_ref["source_position"].tolist()]
        assert len(anchors) >= 1
        assert all(TODFSEQ_SEQ[p - 1] == "S" for p in anchors)  # type-matched
        assert 3 not in anchors  # not the positive

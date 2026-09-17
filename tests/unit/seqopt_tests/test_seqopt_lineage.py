"""This is a script to test SeqOpt.run(lineage=...) and SeqOpt.trace_lineage().

Tiny deterministic wild-type + real-scale df_feat, model-free objectives, and a very small
population so the genuine NSGA-II search runs while the suite stays fast.
"""
import json

import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

SEQ_WT = "MKLAGTWYVFAILMVFWCGSTNQDEHKRPYLAGTWYVFAI"
OBJ = [("magnitude", "max", ut.COL_DELTA_CPP), ("parsimony", "min", ut.COL_N_MUT)]
KWS_FAST = dict(pop_size=8, n_gen=2, n_mut_max=3, region="tmd")


@pytest.fixture
def wt():
    """Position-based df_seq with the single wild-type SeqOpt optimizes."""
    return pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_SEQ: [SEQ_WT],
                         ut.COL_TMD_START: [11], ut.COL_TMD_STOP: [20]})


@pytest.fixture
def df_feat():
    """Small df_feat over the TMD with real scales, mean_dif and feat_importance."""
    scales = list(ut.load_default_scales().columns[:4])
    return pd.DataFrame({
        ut.COL_FEATURE: [f"TMD-Segment(1,1)-{s}" for s in scales],
        ut.COL_CAT: ["Polarity", "ASA/Volume", "Polarity", "Energy"],
        ut.COL_SUBCAT: ["Hydrophobicity", "Volume", "Charge", "Free energy"],
        ut.COL_SCALE_NAME: scales,
        ut.COL_ABS_AUC: [.30, .25, .20, .10], ut.COL_ABS_MEAN_DIF: [.40, .30, .20, .10],
        ut.COL_MEAN_DIF: [.40, -.30, .20, -.10], ut.COL_STD_TEST: [.1] * 4,
        ut.COL_STD_REF: [.1] * 4, ut.COL_FEAT_IMPORT: [40., 30., 20., 10.]})


def _df_seq_of(sequence):
    """A one-row position-based df_seq carrying ``sequence`` (same TMD coordinates)."""
    return pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_SEQ: [sequence],
                         ut.COL_TMD_START: [11], ut.COL_TMD_STOP: [20]})


def _replay(sequence, list_lineage):
    """Apply every mutation of a root-first chain to ``sequence``."""
    for record in list_lineage:
        for mutation in record["mutations"]:
            pos = mutation[ut.COL_POS]
            assert sequence[pos - 1] == mutation[ut.COL_FROM_AA]
            sequence = sequence[:pos - 1] + mutation[ut.COL_TO_AA] + sequence[pos:]
    return sequence


class TestRunLineage:
    """Normal cases for the opt-in ``lineage`` parameter of SeqOpt.run."""

    def test_default_leaves_the_table_unchanged(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        df_pareto = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, **KWS_FAST)
        assert ut.COL_CANDIDATE_ID not in df_pareto.columns
        assert seqo.lineage_ is None

    def test_opt_in_appends_only_the_candidate_id_column(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        df_off = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, **KWS_FAST)
        seqo = aa.SeqOpt(random_state=42)
        df_on = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True, **KWS_FAST)
        assert list(df_on.columns) == list(df_off.columns) + [ut.COL_CANDIDATE_ID]
        assert df_off.equals(df_on[df_off.columns])

    def test_one_record_per_returned_variant(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        df_pareto = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True,
                             **KWS_FAST)
        assert len(seqo.lineage_) == len(df_pareto)

    def test_records_are_row_aligned(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        df_pareto = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True,
                             **KWS_FAST)
        assert list(df_pareto[ut.COL_CANDIDATE_ID]) == [r["candidate_id"]
                                                        for r in seqo.lineage_]

    def test_record_carries_every_field(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True, **KWS_FAST)
        assert list(seqo.lineage_[0]) == list(ut.LIST_CANDIDATE_LINEAGE)

    def test_objective_values_use_the_objective_names(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True, **KWS_FAST)
        assert sorted(seqo.lineage_[0]["objective_values"]) == ["magnitude", "parsimony"]

    def test_method_names_the_algorithm(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, algorithm="greedy",
                 n_mut_max=2, region="tmd", lineage=True)
        assert seqo.lineage_[0]["method"] == "SeqOpt.run:greedy"

    @settings(max_examples=3,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(seed=some.integers(min_value=0, max_value=10))
    def test_seed_records_the_effective_seed(self, seed, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, seed=seed, lineage=True,
                 **KWS_FAST)
        assert seqo.lineage_[0]["seed"] == seed

    def test_records_are_json_serializable(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True, **KWS_FAST)
        assert json.loads(json.dumps(seqo.lineage_)) == seqo.lineage_

    def test_parent_record_sets_the_parent_id(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True, **KWS_FAST)
        parent = seqo.lineage_[0]
        seqo_2 = aa.SeqOpt(random_state=1)
        seqo_2.run(df_seq=_df_seq_of(_replay(SEQ_WT, [parent])), df_feat=df_feat,
                   objectives=OBJ, lineage=parent, **KWS_FAST)
        assert {r["parent_id"] for r in seqo_2.lineage_} == {parent["candidate_id"]}

    # Negative cases
    def test_lineage_string_raises(self, wt, df_feat):
        with pytest.raises(ValueError, match="lineage"):
            aa.SeqOpt(random_state=42).run(df_seq=wt, df_feat=df_feat, objectives=OBJ,
                                           lineage="yes", **KWS_FAST)

    def test_lineage_int_raises(self, wt, df_feat):
        with pytest.raises(ValueError, match="lineage"):
            aa.SeqOpt(random_state=42).run(df_seq=wt, df_feat=df_feat, objectives=OBJ,
                                           lineage=1, **KWS_FAST)

    def test_lineage_none_raises(self, wt, df_feat):
        with pytest.raises(ValueError, match="lineage"):
            aa.SeqOpt(random_state=42).run(df_seq=wt, df_feat=df_feat, objectives=OBJ,
                                           lineage=None, **KWS_FAST)

    def test_lineage_incomplete_record_raises(self, wt, df_feat):
        with pytest.raises(ValueError, match="should carry every lineage field"):
            aa.SeqOpt(random_state=42).run(df_seq=wt, df_feat=df_feat, objectives=OBJ,
                                           lineage={"candidate_id": "sha256:0"}, **KWS_FAST)

    def test_lineage_list_of_records_raises(self, wt, df_feat):
        # A *list* is the tracer's input; run() takes one parent record.
        with pytest.raises(ValueError, match="lineage"):
            aa.SeqOpt(random_state=42).run(df_seq=wt, df_feat=df_feat, objectives=OBJ,
                                           lineage=[{"candidate_id": "sha256:0"}], **KWS_FAST)


class TestRunLineageComplex:
    """Multi-generation designs and interactions with the other run parameters."""

    def test_three_rounds_replay_the_full_path(self, wt, df_feat):
        records, sequence, parent = [], SEQ_WT, True
        for seed in [0, 1, 2]:
            seqo = aa.SeqOpt(random_state=seed)
            df_pareto = seqo.run(df_seq=_df_seq_of(sequence), df_feat=df_feat, objectives=OBJ,
                                 lineage=parent, **KWS_FAST)
            best = df_pareto[df_pareto[ut.COL_N_MUT] > 0].iloc[0]
            parent = [r for r in seqo.lineage_
                      if r["candidate_id"] == best[ut.COL_CANDIDATE_ID]][0]
            records += seqo.lineage_
            sequence = best[ut.COL_SEQ_MUT]
        chain = aa.SeqOpt.trace_lineage(lineage=records, candidate_id=parent["candidate_id"])
        # One record per round, plus the wild-type itself when the parsimony objective keeps
        # the unmutated variant on the front (a zero-mutation record, its own root).
        assert len([r for r in chain if r["mutations"]]) == 3
        assert _replay(SEQ_WT, chain) == sequence

    def test_constraints_digest_is_recorded(self, wt, df_feat):
        dc = aa.DesignConstraints(immutable_positions=[11], n_mut_max=3)
        seqo = aa.SeqOpt(random_state=42)
        seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, constraints=dc, region="tmd",
                 pop_size=8, n_gen=2, lineage=True)
        assert seqo.lineage_[0]["constraints_digest"].startswith("sha256:")

    def test_greedy_and_nsga2_agree_on_a_shared_candidate_id(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        df_greedy = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, algorithm="greedy",
                             n_mut_max=2, region="tmd", lineage=True)
        by_seq = dict(zip(df_greedy[ut.COL_SEQ_MUT], df_greedy[ut.COL_CANDIDATE_ID]))
        df_nsga = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True, **KWS_FAST)
        shared = [(s, c) for s, c in zip(df_nsga[ut.COL_SEQ_MUT], df_nsga[ut.COL_CANDIDATE_ID])
                  if s in by_seq]
        assert all(by_seq[s] == c for s, c in shared)

    def test_lineage_does_not_change_the_search(self, wt, df_feat):
        df_off = aa.SeqOpt(random_state=7).run(df_seq=wt, df_feat=df_feat, objectives=OBJ,
                                               **KWS_FAST)
        df_on = aa.SeqOpt(random_state=7).run(df_seq=wt, df_feat=df_feat, objectives=OBJ,
                                              lineage=True, **KWS_FAST)
        assert df_off.equals(df_on[df_off.columns])

    def test_wild_type_row_has_no_mutation(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        df_pareto = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True,
                             **KWS_FAST)
        for record, n_mut in zip(seqo.lineage_, df_pareto[ut.COL_N_MUT]):
            assert len(record["mutations"]) <= int(n_mut)

    # Negative cases
    def test_trace_with_an_unknown_candidate_raises(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True, **KWS_FAST)
        with pytest.raises(ValueError, match="should be the 'candidate_id' of"):
            aa.SeqOpt.trace_lineage(lineage=seqo.lineage_, candidate_id="sha256:nope")

    def test_trace_with_an_empty_export_raises(self):
        with pytest.raises(ValueError, match="non-empty list of lineage records"):
            aa.SeqOpt.trace_lineage(lineage=[], candidate_id="sha256:0")


class TestRunLineageGoldenValues:
    """Hand-computed identifiers over the optimizer's own output."""

    def test_candidate_id_matches_the_variant_label(self, wt, df_feat):
        import hashlib
        seqo = aa.SeqOpt(random_state=42)
        df_pareto = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True,
                             **KWS_FAST)
        row = df_pareto[df_pareto[ut.COL_N_MUT] > 0].iloc[0]
        payload = f"{SEQ_WT}|{row[ut.COL_VARIANT]}"
        expected = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert row[ut.COL_CANDIDATE_ID] == expected

    def test_mutations_reproduce_the_variant_sequence(self, wt, df_feat):
        seqo = aa.SeqOpt(random_state=42)
        df_pareto = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True,
                             **KWS_FAST)
        for record, seq_mut in zip(seqo.lineage_, df_pareto[ut.COL_SEQ_MUT]):
            assert _replay(SEQ_WT, [record]) == seq_mut

    def test_the_same_run_twice_assigns_the_same_ids(self, wt, df_feat):
        ids = []
        for _ in range(2):
            seqo = aa.SeqOpt(random_state=3)
            df_pareto = seqo.run(df_seq=wt, df_feat=df_feat, objectives=OBJ, lineage=True,
                                 **KWS_FAST)
            ids.append(list(df_pareto[ut.COL_CANDIDATE_ID]))
        assert ids[0] == ids[1]

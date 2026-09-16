"""This is a script to test SeqMut.trace_lineage() and the candidate-lineage record it walks."""
import hashlib
import json

import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

# The P1 sequence of the shared ``df_seq_pos`` fixture (1-based: 11=A, 12=I, 13=L, 14=M, 15=V).
SEQ_P1 = "MKLAGTWYVFAILMVFWCGSTNQDEHKRPYLAGTWYVFAI"


def _variants(list_pos=(11, 12), list_to_aa=("W", "P"), variant="v1", entry="P1"):
    """One combined variant applying ``list_to_aa`` at ``list_pos``."""
    return pd.DataFrame({ut.COL_ENTRY: [entry] * len(list_pos),
                         ut.COL_VARIANT: [variant] * len(list_pos),
                         ut.COL_POS: list(list_pos),
                         ut.COL_TO_AA: list(list_to_aa)})


def _round(df_seq, df_feat, variants, lineage=True):
    """Score one design round with lineage on; return (df_variant, records)."""
    seqm = aa.SeqMut()
    df_variant = seqm.combine(df_seq=df_seq, variants=variants, df_feat=df_feat,
                              lineage=lineage)
    return df_variant, seqm.lineage_


def _df_seq_of(sequence, entry="P1"):
    """A position-based one-row df_seq carrying ``sequence`` (TMD 11-20)."""
    return pd.DataFrame({ut.COL_ENTRY: [entry], ut.COL_SEQ: [sequence],
                         ut.COL_TMD_START: [11], ut.COL_TMD_STOP: [20]})


def _replay(sequence, list_lineage):
    """Apply every mutation of a root-first chain to ``sequence``."""
    for record in list_lineage:
        for mutation in record["mutations"]:
            pos = mutation[ut.COL_POS]
            assert sequence[pos - 1] == mutation[ut.COL_FROM_AA]
            sequence = sequence[:pos - 1] + mutation[ut.COL_TO_AA] + sequence[pos:]
    return sequence


def _digest(text):
    """The ``sha256:<hex>`` digest used by the candidate identifier."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestTraceLineage:
    """Normal cases: one parameter of SeqMut.trace_lineage per test."""

    def test_returns_list_of_records(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        out = aa.SeqMut.trace_lineage(lineage=records, candidate_id=records[0]["candidate_id"])
        assert isinstance(out, list) and all(isinstance(r, dict) for r in out)

    def test_single_round_chain_has_one_record(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        assert len(aa.SeqMut.trace_lineage(lineage=records,
                                           candidate_id=records[0]["candidate_id"])) == 1

    def test_chain_ends_at_the_requested_candidate(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        candidate_id = records[0]["candidate_id"]
        chain = aa.SeqMut.trace_lineage(lineage=records, candidate_id=candidate_id)
        assert chain[-1]["candidate_id"] == candidate_id

    def test_root_record_parent_is_its_source(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        chain = aa.SeqMut.trace_lineage(lineage=records,
                                        candidate_id=records[0]["candidate_id"])
        assert chain[0]["parent_id"] == chain[0]["source_seq_id"]

    def test_lineage_accepts_a_tuple(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        chain = aa.SeqMut.trace_lineage(lineage=tuple(records),
                                        candidate_id=records[0]["candidate_id"])
        assert len(chain) == 1

    def test_lineage_survives_a_json_round_trip(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        reloaded = json.loads(json.dumps(records))
        assert reloaded == records
        chain = aa.SeqMut.trace_lineage(lineage=reloaded,
                                        candidate_id=records[0]["candidate_id"])
        assert chain[-1]["mutations"] == records[0]["mutations"]

    def test_lineage_ignores_unrelated_records(self, df_seq_pos, df_feat):
        _, records_a = _round(df_seq_pos, df_feat, _variants())
        _, records_b = _round(df_seq_pos, df_feat,
                              _variants(list_pos=(14,), list_to_aa=("K",), variant="v2"))
        chain = aa.SeqMut.trace_lineage(lineage=records_a + records_b,
                                        candidate_id=records_a[0]["candidate_id"])
        assert [r["candidate_id"] for r in chain] == [records_a[0]["candidate_id"]]

    @settings(max_examples=5,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(pos=some.integers(min_value=11, max_value=20))
    def test_candidate_id_of_any_position(self, pos, df_seq_pos, df_feat):
        to_aa = "W" if SEQ_P1[pos - 1] != "W" else "A"
        _, records = _round(df_seq_pos, df_feat,
                            _variants(list_pos=(pos,), list_to_aa=(to_aa,)))
        chain = aa.SeqMut.trace_lineage(lineage=records,
                                        candidate_id=records[0]["candidate_id"])
        assert chain[-1]["mutations"][0][ut.COL_POS] == pos

    @settings(max_examples=5,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(to_aa=some.sampled_from([aa_ for aa_ in ut.LIST_CANONICAL_AA if aa_ != "A"]))
    def test_candidate_id_of_any_target_residue(self, to_aa, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat,
                            _variants(list_pos=(11,), list_to_aa=(to_aa,)))
        chain = aa.SeqMut.trace_lineage(lineage=records,
                                        candidate_id=records[0]["candidate_id"])
        assert chain[-1]["mutations"][0][ut.COL_TO_AA] == to_aa

    def test_every_documented_field_is_present(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        assert list(records[0]) == list(ut.LIST_CANDIDATE_LINEAGE)

    # Negative cases: one invalid parameter per test
    def test_lineage_none_raises(self):
        with pytest.raises(ValueError, match="lineage"):
            aa.SeqMut.trace_lineage(lineage=None, candidate_id="sha256:0")

    def test_lineage_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty list of lineage records"):
            aa.SeqMut.trace_lineage(lineage=[], candidate_id="sha256:0")

    def test_lineage_string_raises(self):
        with pytest.raises(ValueError, match="lineage"):
            aa.SeqMut.trace_lineage(lineage="records", candidate_id="sha256:0")

    def test_lineage_of_non_dicts_raises(self):
        with pytest.raises(ValueError, match="dictionary"):
            aa.SeqMut.trace_lineage(lineage=[1, 2], candidate_id="sha256:0")

    def test_lineage_record_missing_fields_raises(self):
        with pytest.raises(ValueError, match="should carry every lineage field"):
            aa.SeqMut.trace_lineage(lineage=[{"candidate_id": "sha256:0"}],
                                    candidate_id="sha256:0")

    def test_lineage_record_non_string_id_raises(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        broken = dict(records[0], candidate_id=1)
        with pytest.raises(ValueError, match="candidate_id"):
            aa.SeqMut.trace_lineage(lineage=[broken], candidate_id="sha256:0")

    def test_lineage_record_non_list_mutations_raises(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        broken = dict(records[0], mutations={"pos": 11})
        with pytest.raises(ValueError, match="mutations"):
            aa.SeqMut.trace_lineage(lineage=[broken],
                                    candidate_id=records[0]["candidate_id"])

    def test_candidate_id_none_raises(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        with pytest.raises(ValueError, match="candidate_id"):
            aa.SeqMut.trace_lineage(lineage=records, candidate_id=None)

    def test_candidate_id_unknown_raises(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        with pytest.raises(ValueError, match="should be the 'candidate_id' of"):
            aa.SeqMut.trace_lineage(lineage=records, candidate_id="sha256:deadbeef")

    def test_candidate_id_not_a_string_raises(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        with pytest.raises(ValueError, match="candidate_id"):
            aa.SeqMut.trace_lineage(lineage=records, candidate_id=42)


class TestTraceLineageComplex:
    """Combinations: branching, multi-generation chains, and merged exports."""

    def test_branching_candidates_share_one_parent(self, df_seq_pos, df_feat):
        variants = pd.concat([_variants(list_pos=(11, 12), list_to_aa=("W", "P"), variant="v1"),
                              _variants(list_pos=(13,), list_to_aa=("K",), variant="v2")])
        _, records = _round(df_seq_pos, df_feat, variants)
        assert len({r["parent_id"] for r in records}) == 1
        assert len({r["candidate_id"] for r in records}) == 2

    def test_two_generations_chain_up(self, df_seq_pos, df_feat):
        df_1, records_1 = _round(df_seq_pos, df_feat, _variants())
        seq_2 = df_1[ut.COL_SEQ_MUT].iloc[0]
        seqm = aa.SeqMut()
        seqm.combine(df_seq=_df_seq_of(seq_2), df_feat=df_feat,
                     variants=_variants(list_pos=(15,), list_to_aa=("C",), variant="w1"),
                     lineage=records_1[0])
        chain = aa.SeqMut.trace_lineage(lineage=records_1 + seqm.lineage_,
                                        candidate_id=seqm.lineage_[0]["candidate_id"])
        assert len(chain) == 2
        assert chain[0]["candidate_id"] == chain[1]["parent_id"]

    def test_three_generations_replay_the_full_path(self, df_seq_pos, df_feat):
        records, sequence, parent = [], SEQ_P1, True
        for pos, to_aa in [(11, "W"), (15, "C"), (18, "K")]:
            seqm = aa.SeqMut()
            df_variant = seqm.combine(df_seq=_df_seq_of(sequence), df_feat=df_feat,
                                      variants=_variants(list_pos=(pos,), list_to_aa=(to_aa,)),
                                      lineage=parent)
            records += seqm.lineage_
            sequence = df_variant[ut.COL_SEQ_MUT].iloc[0]
            parent = seqm.lineage_[0]
        chain = aa.SeqMut.trace_lineage(lineage=records, candidate_id=records[-1]["candidate_id"])
        assert len(chain) == 3
        assert _replay(SEQ_P1, chain) == sequence

    def test_chain_is_root_first(self, df_seq_pos, df_feat):
        df_1, records_1 = _round(df_seq_pos, df_feat, _variants())
        seqm = aa.SeqMut()
        seqm.combine(df_seq=_df_seq_of(df_1[ut.COL_SEQ_MUT].iloc[0]), df_feat=df_feat,
                     variants=_variants(list_pos=(15,), list_to_aa=("C",), variant="w1"),
                     lineage=records_1[0])
        chain = aa.SeqMut.trace_lineage(lineage=seqm.lineage_ + records_1,
                                        candidate_id=seqm.lineage_[0]["candidate_id"])
        assert chain[0]["candidate_id"] == records_1[0]["candidate_id"]

    def test_partial_export_stops_at_the_first_missing_ancestor(self, df_seq_pos, df_feat):
        df_1, records_1 = _round(df_seq_pos, df_feat, _variants())
        seqm = aa.SeqMut()
        seqm.combine(df_seq=_df_seq_of(df_1[ut.COL_SEQ_MUT].iloc[0]), df_feat=df_feat,
                     variants=_variants(list_pos=(15,), list_to_aa=("C",), variant="w1"),
                     lineage=records_1[0])
        chain = aa.SeqMut.trace_lineage(lineage=seqm.lineage_,
                                        candidate_id=seqm.lineage_[0]["candidate_id"])
        assert len(chain) == 1

    # Negative / degenerate compositions
    def test_cyclic_records_raise(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        record_a = dict(records[0], candidate_id="sha256:a", parent_id="sha256:b")
        record_b = dict(records[0], candidate_id="sha256:b", parent_id="sha256:a")
        with pytest.raises(RuntimeError, match="cyclic"):
            aa.SeqMut.trace_lineage(lineage=[record_a, record_b], candidate_id="sha256:a")

    def test_self_rooted_record_ends_the_chain(self, df_seq_pos, df_feat):
        # A candidate identical to its source is its own root: no mutation to walk back.
        _, records = _round(df_seq_pos, df_feat, _variants())
        self_root = dict(records[0], parent_id=records[0]["candidate_id"])
        chain = aa.SeqMut.trace_lineage(lineage=[self_root],
                                        candidate_id=self_root["candidate_id"])
        assert len(chain) == 1

    def test_parent_record_with_two_source_sequences_raises(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        variants = pd.concat([_variants(list_pos=(11,), list_to_aa=("W",), variant="v1"),
                              _variants(list_pos=(11,), list_to_aa=("W",), variant="v2",
                                        entry="P2")])
        with pytest.raises(ValueError, match="one source sequence"):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=variants, df_feat=df_feat,
                                lineage=records[0])

    def test_duplicate_records_are_indexed_once(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        chain = aa.SeqMut.trace_lineage(lineage=records + records,
                                        candidate_id=records[0]["candidate_id"])
        assert len(chain) == 1

    def test_broken_parent_link_in_a_long_export_raises_nothing(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        orphan = dict(records[0], candidate_id="sha256:orphan", parent_id="sha256:missing")
        chain = aa.SeqMut.trace_lineage(lineage=records + [orphan],
                                        candidate_id="sha256:orphan")
        assert len(chain) == 1 and chain[0]["candidate_id"] == "sha256:orphan"

    def test_records_of_two_classes_merge_into_one_chain(self, df_seq_pos, df_feat):
        df_1, records_1 = _round(df_seq_pos, df_feat, _variants())
        seqo = aa.SeqOpt(random_state=0)
        seqo.run(df_seq=_df_seq_of(df_1[ut.COL_SEQ_MUT].iloc[0]), df_feat=df_feat,
                 objectives=[("magnitude", "max", ut.COL_DELTA_CPP),
                             ("parsimony", "min", ut.COL_N_MUT)],
                 pop_size=6, n_gen=2, n_mut_max=2, region="tmd", lineage=records_1[0])
        chain = aa.SeqMut.trace_lineage(lineage=records_1 + seqo.lineage_,
                                        candidate_id=seqo.lineage_[0]["candidate_id"])
        assert len(chain) == 2
        assert chain[0]["method"] == "SeqMut.combine"
        assert chain[1]["method"].startswith("SeqOpt.run")


class TestTraceLineageGoldenValues:
    """Hand-computed identifiers and replayed paths."""

    def test_candidate_id_is_the_hand_computed_hash(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        # P1 position 11 is 'A' and 12 is 'I', so the payload is "<sequence>|A11W+I12P".
        assert records[0]["candidate_id"] == _digest(f"{SEQ_P1}|A11W+I12P")

    def test_source_seq_id_is_the_hash_of_the_unmutated_source(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat, _variants())
        assert records[0]["source_seq_id"] == _digest(f"{SEQ_P1}|")

    def test_mutations_are_parent_relative_and_position_ordered(self, df_seq_pos, df_feat):
        _, records = _round(df_seq_pos, df_feat,
                            _variants(list_pos=(13, 11), list_to_aa=("K", "W")))
        assert records[0]["mutations"] == [{ut.COL_POS: 11, ut.COL_FROM_AA: "A",
                                            ut.COL_TO_AA: "W"},
                                           {ut.COL_POS: 13, ut.COL_FROM_AA: "L",
                                            ut.COL_TO_AA: "K"}]

    def test_two_runs_assign_the_same_candidate_id(self, df_seq_pos, df_feat):
        _, records_a = _round(df_seq_pos, df_feat, _variants())
        # The same mutant reached in the other position order, in a second run.
        _, records_b = _round(df_seq_pos, df_feat,
                              _variants(list_pos=(12, 11), list_to_aa=("P", "W"),
                                        variant="other"))
        assert records_a[0]["candidate_id"] == records_b[0]["candidate_id"]

    def test_duplicate_detection_over_a_candidate_set(self, df_seq_pos, df_feat):
        _, records_a = _round(df_seq_pos, df_feat, _variants())
        _, records_b = _round(df_seq_pos, df_feat,
                              _variants(list_pos=(11,), list_to_aa=("W",), variant="v2"))
        merged = records_a + records_b + records_a
        assert len({r["candidate_id"] for r in merged}) == 2

    def test_a_different_mutant_gets_a_different_id(self, df_seq_pos, df_feat):
        _, records_a = _round(df_seq_pos, df_feat, _variants())
        _, records_b = _round(df_seq_pos, df_feat,
                              _variants(list_pos=(11, 12), list_to_aa=("W", "K")))
        assert records_a[0]["candidate_id"] != records_b[0]["candidate_id"]

    def test_a_no_op_edit_is_not_recorded(self, df_seq_pos, df_feat):
        # Position 11 already carries 'A', so requesting 'A' changes nothing.
        _, records = _round(df_seq_pos, df_feat,
                            _variants(list_pos=(11, 12), list_to_aa=("A", "P")))
        assert records[0]["mutations"] == [{ut.COL_POS: 12, ut.COL_FROM_AA: "I",
                                            ut.COL_TO_AA: "P"}]
        assert records[0]["candidate_id"] == _digest(f"{SEQ_P1}|I12P")

    def test_replaying_one_record_rebuilds_the_candidate(self, df_seq_pos, df_feat):
        df_variant, records = _round(df_seq_pos, df_feat, _variants())
        assert _replay(SEQ_P1, records) == df_variant[ut.COL_SEQ_MUT].iloc[0]

    def test_constraints_digest_is_stable_and_specific(self, df_seq_pos, df_feat):
        dc = aa.DesignConstraints(n_mut_max=2)
        seqm = aa.SeqMut()
        seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                     constraints=dc, lineage=True)
        digest = seqm.lineage_[0]["constraints_digest"]
        seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                     constraints=aa.DesignConstraints(n_mut_max=2), lineage=True)
        assert seqm.lineage_[0]["constraints_digest"] == digest
        seqm.combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                     constraints=aa.DesignConstraints(n_mut_max=3), lineage=True)
        assert seqm.lineage_[0]["constraints_digest"] != digest

"""This is a script to test DesignConstraints.check() and DesignConstraints.as_predicate()."""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis.utils as ut
from aaanalysis.protein_engineering import DesignConstraints

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

# 1-based positions 1..10 are M K L A G T W Y V F.
PARENT = "MKLAGTWYVF"


def _mutate(parent, dict_mut):
    """Apply a ``{1-based pos: to_aa}`` mapping to ``parent`` (the test-side genome helper)."""
    chars = list(parent)
    for pos, to_aa in dict_mut.items():
        chars[pos - 1] = to_aa
    return "".join(chars)


class TestCheck:
    """Normal cases: one constraint field per test, plus the two parameters of check()."""

    def test_candidate_equal_to_parent_is_feasible(self, parent):
        ok, reasons = DesignConstraints(parent=parent).check(candidate=parent)
        assert ok is True and reasons == []

    @settings(max_examples=5)
    @given(pos=some.integers(min_value=1, max_value=10))
    def test_candidate(self, pos):
        # The module-level PARENT is used instead of the fixture: hypothesis does not reset
        # function-scoped fixtures between generated inputs.
        candidate = _mutate(PARENT, {pos: "A"})
        ok, reasons = DesignConstraints(parent=PARENT).check(candidate=candidate)
        assert ok is True and reasons == []

    def test_parent_per_call_overrides_the_stored_parent(self):
        dc = DesignConstraints(immutable_positions=[1], parent="AAAAA")
        ok, _reasons = dc.check(candidate="MKLAG", parent="MKLAG")
        assert ok is True

    def test_immutable_positions(self, parent):
        dc = DesignConstraints(immutable_positions=[1], parent=parent)
        ok, reasons = dc.check(candidate=_mutate(parent, {1: "A"}))
        assert ok is False and len(reasons) == 1 and reasons[0].startswith("immutable_positions")

    def test_mutable_positions(self, parent):
        dc = DesignConstraints(mutable_positions=[1, 2], parent=parent)
        ok, reasons = dc.check(candidate=_mutate(parent, {5: "A"}))
        assert ok is False and reasons[0].startswith("mutable_positions")

    def test_mutable_positions_part_name_is_not_checked_here(self, parent):
        # A part name needs TMD coordinates, so it is applied by the calling class.
        dc = DesignConstraints(mutable_positions="tmd", parent=parent)
        ok, reasons = dc.check(candidate=_mutate(parent, {5: "A"}))
        assert ok is True and reasons == []

    def test_permitted_substitutions_list(self, parent):
        dc = DesignConstraints(permitted_substitutions=["A"], parent=parent)
        ok, reasons = dc.check(candidate=_mutate(parent, {1: "W"}))
        assert ok is False and reasons[0].startswith("permitted_substitutions")

    def test_permitted_substitutions_dict_only_binds_listed_positions(self, parent):
        dc = DesignConstraints(permitted_substitutions={1: ["A"]}, parent=parent)
        assert dc.check(candidate=_mutate(parent, {2: "W"}))[0] is True
        assert dc.check(candidate=_mutate(parent, {1: "W"}))[0] is False

    def test_forbidden_substitutions(self, parent):
        dc = DesignConstraints(forbidden_substitutions=["P"], parent=parent)
        ok, reasons = dc.check(candidate=_mutate(parent, {1: "P"}))
        assert ok is False and reasons[0].startswith("forbidden_substitutions")

    def test_n_mut_max(self, parent):
        dc = DesignConstraints(n_mut_max=1, parent=parent)
        assert dc.check(candidate=_mutate(parent, {1: "A"}))[0] is True
        assert dc.check(candidate=_mutate(parent, {1: "A", 2: "A"}))[0] is False

    def test_min_identity(self, parent):
        dc = DesignConstraints(min_identity=0.95, parent=parent)
        ok, reasons = dc.check(candidate=_mutate(parent, {1: "A"}))
        assert ok is False and reasons[0].startswith("min_identity")

    def test_max_identity(self, parent):
        dc = DesignConstraints(max_identity=0.95, parent=parent)
        ok, reasons = dc.check(candidate=parent)
        assert ok is False and reasons[0].startswith("max_identity")

    def test_forbidden_motifs(self, parent):
        dc = DesignConstraints(forbidden_motifs=["WY"], parent=parent)
        ok, reasons = dc.check(candidate=parent)
        assert ok is False and reasons[0].startswith("forbidden_motifs")

    def test_required_motifs(self, parent):
        dc = DesignConstraints(required_motifs=["KKK"], parent=parent)
        ok, reasons = dc.check(candidate=parent)
        assert ok is False and reasons[0].startswith("required_motifs")

    def test_no_constraint_never_rejects(self, parent):
        ok, reasons = DesignConstraints(parent=parent).check(candidate="AAAAAAAAAA")
        assert ok is True and reasons == []

    # Negative cases
    def test_candidate_longer_than_parent_raises(self, parent):
        with pytest.raises(ValueError, match="same length"):
            DesignConstraints(parent=parent).check(candidate=parent + "A")

    def test_candidate_shorter_than_parent_raises(self, parent):
        with pytest.raises(ValueError, match="same length"):
            DesignConstraints(parent=parent).check(candidate=parent[:-1])

    def test_candidate_none_raises(self, parent):
        with pytest.raises(ValueError, match="'candidate'"):
            DesignConstraints(parent=parent).check(candidate=None)

    def test_candidate_empty_raises(self, parent):
        with pytest.raises(ValueError, match="non-empty protein sequence"):
            DesignConstraints(parent=parent).check(candidate="")

    def test_candidate_lowercase_raises(self, parent):
        with pytest.raises(ValueError, match="upper-case one-letter residue codes"):
            DesignConstraints(parent=parent).check(candidate=parent.lower())

    def test_candidate_not_a_string_raises(self, parent):
        with pytest.raises(ValueError, match="'candidate'"):
            DesignConstraints(parent=parent).check(candidate=list(parent))

    def test_missing_parent_raises(self):
        with pytest.raises(ValueError, match="'parent'"):
            DesignConstraints(n_mut_max=1).check(candidate="MKLAG")

    def test_bad_per_call_parent_raises(self):
        with pytest.raises(ValueError, match="'parent'"):
            DesignConstraints().check(candidate="MKLAG", parent="mklag")

    def test_parent_not_a_string_raises(self):
        with pytest.raises(ValueError, match="'parent'"):
            DesignConstraints().check(candidate="MKLAG", parent=12345)


class TestCheckComplex:
    """Combinations and edge interactions across several constraint fields."""

    def test_feasible_under_every_field_at_once(self, parent):
        dc = DesignConstraints(immutable_positions=[10], mutable_positions=[1, 2, 3],
                               permitted_substitutions=["A", "V"],
                               forbidden_substitutions={3: ["V"]}, n_mut_max=2,
                               min_identity=0.7, max_identity=0.95,
                               forbidden_motifs=["PP"], required_motifs=["WY"],
                               parent=parent)
        ok, reasons = dc.check(candidate=_mutate(parent, {1: "A", 2: "V"}))
        assert ok is True and reasons == []

    def test_identity_bounds_bracket_an_acceptable_candidate(self, parent):
        dc = DesignConstraints(min_identity=0.75, max_identity=0.85, parent=parent)
        assert dc.check(candidate=_mutate(parent, {1: "A", 2: "A"}))[0] is True

    def test_identity_lower_bound_is_inclusive(self, parent):
        dc = DesignConstraints(min_identity=0.9, parent=parent)
        assert dc.check(candidate=_mutate(parent, {1: "A"}))[0] is True

    def test_a_silent_no_op_substitution_is_not_a_mutation(self, parent):
        dc = DesignConstraints(immutable_positions=[1], n_mut_max=1, parent=parent)
        # Writing the wild-type residue back at position 1 leaves the sequence unchanged.
        assert dc.check(candidate=_mutate(parent, {1: "M", 2: "A"}))[0] is True

    def test_forbidden_wins_over_permitted_at_the_same_position(self, parent):
        dc = DesignConstraints(permitted_substitutions={1: ["A", "V"]},
                               forbidden_substitutions={1: ["V"]}, parent=parent)
        assert dc.check(candidate=_mutate(parent, {1: "A"}))[0] is True
        ok, reasons = dc.check(candidate=_mutate(parent, {1: "V"}))
        assert ok is False and len(reasons) == 1 and reasons[0].startswith("forbidden_substitutions")

    def test_a_substitution_can_create_a_forbidden_motif(self, parent):
        dc = DesignConstraints(forbidden_motifs=["AA"], parent=parent)
        assert dc.check(candidate=parent)[0] is True
        assert dc.check(candidate=_mutate(parent, {5: "A"}))[0] is False   # A4 + A5

    # Negative cross-field cases
    def test_two_fields_violated_at_once(self, parent):
        dc = DesignConstraints(immutable_positions=[1], n_mut_max=1, parent=parent)
        ok, reasons = dc.check(candidate=_mutate(parent, {1: "A", 2: "A"}))
        assert ok is False and len(reasons) == 2

    def test_position_and_residue_fields_violated_at_once(self, parent):
        dc = DesignConstraints(mutable_positions=[1], permitted_substitutions=["A"],
                               parent=parent)
        ok, reasons = dc.check(candidate=_mutate(parent, {5: "W"}))
        assert ok is False and len(reasons) == 2

    def test_both_identity_bounds_cannot_be_violated_together(self, parent):
        dc = DesignConstraints(min_identity=0.5, max_identity=0.95, parent=parent)
        ok, reasons = dc.check(candidate="AAAAAAAAAA")
        assert ok is False and len(reasons) == 1 and reasons[0].startswith("min_identity")

    def test_both_motif_fields_violated_at_once(self, parent):
        dc = DesignConstraints(forbidden_motifs=["WY"], required_motifs=["KKK"], parent=parent)
        ok, reasons = dc.check(candidate=parent)
        assert ok is False and len(reasons) == 2

    def test_budget_and_motif_violated_at_once(self, parent):
        dc = DesignConstraints(n_mut_max=1, required_motifs=["WY"], parent=parent)
        ok, reasons = dc.check(candidate=_mutate(parent, {7: "A", 8: "A"}))
        assert ok is False and len(reasons) == 2

    def test_a_bad_candidate_raises_before_any_reason_is_collected(self, parent):
        dc = DesignConstraints(immutable_positions=[1], n_mut_max=1, parent=parent)
        with pytest.raises(ValueError, match="same length"):
            dc.check(candidate="AA")


class TestCheckGoldenValues:
    """Hand-computed reason strings, counts and ordering."""

    def test_exactly_two_violations_returns_exactly_those_two_reasons_in_order(self, parent):
        # parent  = M K L A G T W Y V F
        # candidate 'AALAGTWYVF' mutates position 1 (M->A) and position 2 (K->A).
        # Position 1 is immutable -> reason 1; two mutations exceed n_mut_max=1 -> reason 5.
        # The reported order follows ut.LIST_DESIGN_CONSTRAINTS, so immutable comes first.
        dc = DesignConstraints(immutable_positions=[1], n_mut_max=1, parent=parent)
        ok, reasons = dc.check(candidate="AALAGTWYVF")
        assert ok is False
        assert reasons == ["immutable_positions: position(s) [1] are immutable",
                           "n_mut_max: 2 mutations exceed the maximum of 1"]

    def test_four_violations_follow_the_documented_field_order(self, parent):
        # candidate 'AALAGTWYAF' mutates 1 (M->A), 2 (K->A) and 9 (V->A).
        #   immutable_positions=[9]          -> position 9 is immutable
        #   mutable_positions=[1, 2, 3]      -> position 9 is outside the mutable set
        #   permitted_substitutions=['A']    -> every target is 'A', so NOT violated
        #   forbidden_substitutions={2:['A']}-> K2A is forbidden
        #   n_mut_max=1                      -> 3 mutations exceed 1
        dc = DesignConstraints(immutable_positions=[9], mutable_positions=[1, 2, 3],
                               permitted_substitutions=["A"],
                               forbidden_substitutions={2: ["A"]}, n_mut_max=1, parent=parent)
        ok, reasons = dc.check(candidate="AALAGTWYAF")
        assert ok is False
        assert reasons == [
            "immutable_positions: position(s) [9] are immutable",
            "mutable_positions: position(s) [9] are outside the mutable set",
            "forbidden_substitutions: substitution(s) ['K2A'] are forbidden",
            "n_mut_max: 3 mutations exceed the maximum of 1"]
        assert [r.split(":")[0] for r in reasons] == [
            f for f in ut.LIST_DESIGN_CONSTRAINTS if f.split(":")[0] in
            {"immutable_positions", "mutable_positions", "forbidden_substitutions", "n_mut_max"}]

    def test_permitted_substitution_reason_names_the_substitution(self, parent):
        dc = DesignConstraints(permitted_substitutions=["A"], parent=parent)
        ok, reasons = dc.check(candidate="WKLAGTWYVF")   # M1W
        assert reasons == ["permitted_substitutions: substitution(s) ['M1W'] are not permitted"]

    def test_min_identity_reason_carries_the_computed_identity(self, parent):
        # One substitution over a 10-residue parent -> identity 9/10 = 0.9000.
        dc = DesignConstraints(min_identity=0.95, parent=parent)
        ok, reasons = dc.check(candidate="AKLAGTWYVF")
        assert reasons == ["min_identity: identity 0.9000 is below the minimum of 0.9500"]

    def test_max_identity_reason_carries_the_computed_identity(self, parent):
        dc = DesignConstraints(max_identity=0.9000, parent=parent)
        ok, reasons = dc.check(candidate=parent)          # identity 10/10 = 1.0000
        assert reasons == ["max_identity: identity 1.0000 is above the maximum of 0.9000"]

    def test_motif_reasons_list_every_offending_motif(self, parent):
        dc = DesignConstraints(forbidden_motifs=["WY", "KL"], required_motifs=["PP", "GG"],
                               parent=parent)
        ok, reasons = dc.check(candidate=parent)
        assert reasons == ["forbidden_motifs: motif(s) ['WY', 'KL'] occur in the candidate",
                           "required_motifs: motif(s) ['PP', 'GG'] are absent from the candidate"]

    def test_every_reason_is_a_non_empty_field_prefixed_string(self, parent):
        dc = DesignConstraints(immutable_positions=[1], mutable_positions=[2],
                               permitted_substitutions=["V"], forbidden_substitutions=["A"],
                               n_mut_max=1, min_identity=0.99, max_identity=0.995,
                               forbidden_motifs=["AA"], required_motifs=["PPP"], parent=parent)
        ok, reasons = dc.check(candidate="AALAGTWYVF")
        assert ok is False and len(reasons) >= 1
        for reason in reasons:
            field = reason.split(":")[0]
            assert field in ut.LIST_DESIGN_CONSTRAINTS and len(reason) > len(field) + 2


class TestAsPredicate:
    """The genome-shaped adapter consumed by SeqOpt.run."""

    def test_returns_a_callable(self, parent):
        assert callable(DesignConstraints(parent=parent).as_predicate())

    @settings(max_examples=5)
    @given(pos=some.integers(min_value=2, max_value=10))
    def test_feasible_genome(self, pos):
        predicate = DesignConstraints(immutable_positions=[1], parent=PARENT).as_predicate()
        assert predicate({pos: "A"}) is True

    def test_infeasible_genome(self, parent):
        predicate = DesignConstraints(immutable_positions=[1], parent=parent).as_predicate()
        assert predicate({1: "A"}) is False

    def test_empty_genome_is_the_parent(self, parent):
        predicate = DesignConstraints(n_mut_max=1, parent=parent).as_predicate()
        assert predicate({}) is True

    def test_parent_per_call_overrides_the_stored_parent(self, parent):
        predicate = DesignConstraints(forbidden_motifs=["AA"],
                                      parent="AAAAAAAAAA").as_predicate(parent=parent)
        assert predicate({}) is True

    def test_budget_is_enforced(self, parent):
        predicate = DesignConstraints(n_mut_max=2, parent=parent).as_predicate()
        assert predicate({1: "A", 2: "A"}) is True
        assert predicate({1: "A", 2: "A", 3: "A"}) is False

    def test_agrees_with_check(self, parent):
        dc = DesignConstraints(immutable_positions=[1], n_mut_max=1, parent=parent)
        predicate = dc.as_predicate()
        for genome in ({1: "A"}, {2: "A"}, {2: "A", 3: "A"}):
            candidate = _mutate(parent, genome)
            assert predicate(genome) == dc.check(candidate=candidate)[0]

    def test_missing_parent_raises(self):
        with pytest.raises(ValueError, match="'parent'"):
            DesignConstraints(n_mut_max=1).as_predicate()

    def test_bad_parent_raises(self):
        with pytest.raises(ValueError, match="'parent'"):
            DesignConstraints().as_predicate(parent="mklag")

    def test_predicate_is_a_snapshot_of_the_limits(self, parent):
        dc = DesignConstraints(n_mut_max=1, parent=parent)
        predicate = dc.as_predicate()
        dc.n_mut_max = 5   # mutating the object afterwards must not change the built predicate
        assert predicate({1: "A", 2: "A"}) is False

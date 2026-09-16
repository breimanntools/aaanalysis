"""This is a script to test DesignConstraints() construction, to_dict() and from_dict()."""
import json

import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis.utils as ut
from aaanalysis.protein_engineering import DesignConstraints

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

AA = some.sampled_from(list(ut.LIST_CANONICAL_AA))
MOTIFS = some.text(alphabet=list(ut.LIST_CANONICAL_AA), min_size=1, max_size=4)


class TestDesignConstraints:
    """Normal cases: one parameter of the constructor per test."""

    @settings(max_examples=5)
    @given(immutable_positions=some.lists(some.integers(min_value=1, max_value=50),
                                          min_size=1, max_size=5))
    def test_immutable_positions(self, immutable_positions):
        dc = DesignConstraints(immutable_positions=immutable_positions)
        assert dc.immutable_positions == [int(p) for p in immutable_positions]

    @settings(max_examples=5)
    @given(mutable_positions=some.lists(some.integers(min_value=1, max_value=50),
                                        min_size=1, max_size=5))
    def test_mutable_positions_list(self, mutable_positions):
        dc = DesignConstraints(mutable_positions=mutable_positions)
        assert dc.mutable_positions == [int(p) for p in mutable_positions]

    @pytest.mark.parametrize("part", ut.COLS_SEQ_PARTS)
    def test_mutable_positions_part_name(self, part):
        assert DesignConstraints(mutable_positions=part.upper()).mutable_positions == part

    @settings(max_examples=5)
    @given(permitted_substitutions=some.lists(AA, min_size=1, max_size=6))
    def test_permitted_substitutions_list(self, permitted_substitutions):
        dc = DesignConstraints(permitted_substitutions=permitted_substitutions)
        assert dc.permitted_substitutions == list(permitted_substitutions)

    @settings(max_examples=5)
    @given(pos=some.integers(min_value=1, max_value=50))
    def test_permitted_substitutions_dict(self, pos):
        dc = DesignConstraints(permitted_substitutions={pos: ["A", "V"]})
        assert dc.permitted_substitutions == {pos: ["A", "V"]}

    @settings(max_examples=5)
    @given(forbidden_substitutions=some.lists(AA, min_size=1, max_size=6))
    def test_forbidden_substitutions(self, forbidden_substitutions):
        dc = DesignConstraints(forbidden_substitutions=forbidden_substitutions)
        assert dc.forbidden_substitutions == list(forbidden_substitutions)

    @settings(max_examples=5)
    @given(n_mut_max=some.integers(min_value=1, max_value=50))
    def test_n_mut_max(self, n_mut_max):
        assert DesignConstraints(n_mut_max=n_mut_max).n_mut_max == n_mut_max

    @settings(max_examples=5)
    @given(min_identity=some.floats(min_value=0, max_value=1))
    def test_min_identity(self, min_identity):
        assert DesignConstraints(min_identity=min_identity).min_identity == min_identity

    @settings(max_examples=5)
    @given(max_identity=some.floats(min_value=0, max_value=1))
    def test_max_identity(self, max_identity):
        assert DesignConstraints(max_identity=max_identity).max_identity == max_identity

    @settings(max_examples=5)
    @given(forbidden_motifs=some.lists(MOTIFS, min_size=1, max_size=3))
    def test_forbidden_motifs(self, forbidden_motifs):
        dc = DesignConstraints(forbidden_motifs=forbidden_motifs)
        assert dc.forbidden_motifs == list(forbidden_motifs)

    @settings(max_examples=5)
    @given(required_motifs=some.lists(MOTIFS, min_size=1, max_size=3))
    def test_required_motifs(self, required_motifs):
        dc = DesignConstraints(required_motifs=required_motifs)
        assert dc.required_motifs == list(required_motifs)

    @settings(max_examples=5)
    @given(parent=some.text(alphabet=list(ut.LIST_CANONICAL_AA), min_size=1, max_size=20))
    def test_parent(self, parent):
        assert DesignConstraints(parent=parent).parent == parent

    def test_defaults_are_all_none(self):
        dc = DesignConstraints()
        assert all(getattr(dc, field) is None for field in ut.LIST_DESIGN_CONSTRAINTS)
        assert dc.parent is None

    def test_non_canonical_parent_is_accepted(self):
        # A wild-type may legitimately carry X / U / B / Z; only motifs and substitution
        # rules are restricted to the 20 canonical amino acids.
        assert DesignConstraints(parent="MKXLAG").parent == "MKXLAG"

    # Negative cases: one invalid parameter per test
    def test_immutable_positions_below_one_raises(self):
        with pytest.raises(ValueError, match="immutable_positions"):
            DesignConstraints(immutable_positions=[0])

    def test_immutable_positions_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty list of 1-based positions"):
            DesignConstraints(immutable_positions=[])

    def test_immutable_positions_not_list_raises(self):
        with pytest.raises(ValueError, match="immutable_positions"):
            DesignConstraints(immutable_positions="abc")

    def test_mutable_positions_unknown_part_raises(self):
        with pytest.raises(ValueError, match="should be one of"):
            DesignConstraints(mutable_positions="middle")

    def test_permitted_substitutions_non_canonical_raises(self):
        with pytest.raises(ValueError, match="canonical amino acids"):
            DesignConstraints(permitted_substitutions=["Z"])

    def test_permitted_substitutions_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty list of canonical amino acids"):
            DesignConstraints(permitted_substitutions=[])

    def test_permitted_substitutions_dict_bad_key_raises(self):
        with pytest.raises(ValueError, match="1-based integer position"):
            DesignConstraints(permitted_substitutions={"tmd": ["A"]})

    def test_permitted_substitutions_empty_dict_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            DesignConstraints(permitted_substitutions={})

    def test_forbidden_substitutions_non_canonical_raises(self):
        with pytest.raises(ValueError, match="canonical amino acids"):
            DesignConstraints(forbidden_substitutions=["B"])

    def test_n_mut_max_zero_raises(self):
        with pytest.raises(ValueError, match="'n_mut_max'"):
            DesignConstraints(n_mut_max=0)

    def test_n_mut_max_float_raises(self):
        with pytest.raises(ValueError, match="'n_mut_max'"):
            DesignConstraints(n_mut_max=2.5)

    def test_min_identity_above_one_raises(self):
        with pytest.raises(ValueError, match="'min_identity'"):
            DesignConstraints(min_identity=1.5)

    def test_max_identity_below_zero_raises(self):
        with pytest.raises(ValueError, match="'max_identity'"):
            DesignConstraints(max_identity=-0.1)

    def test_forbidden_motifs_non_canonical_raises(self):
        with pytest.raises(ValueError, match="non-empty string of"):
            DesignConstraints(forbidden_motifs=["1A"])

    def test_required_motifs_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty list of amino acid motifs"):
            DesignConstraints(required_motifs=[])

    def test_parent_lowercase_raises(self):
        with pytest.raises(ValueError, match="upper-case one-letter residue codes"):
            DesignConstraints(parent="mklag")

    def test_parent_not_a_string_raises(self):
        with pytest.raises(ValueError, match="'parent'"):
            DesignConstraints(parent=123)


class TestDesignConstraintsComplex:
    """Combinations and edge interactions across constructor parameters."""

    def test_full_constraint_set(self):
        dc = DesignConstraints(immutable_positions=[1], mutable_positions=[2, 3, 4],
                               permitted_substitutions=["A", "V"],
                               forbidden_substitutions={3: ["V"]}, n_mut_max=2,
                               min_identity=0.5, max_identity=0.99,
                               forbidden_motifs=["WW"], required_motifs=["KL"],
                               parent="MKLAGTWYVF")
        assert dc.n_mut_max == 2 and dc.forbidden_substitutions == {3: ["V"]}

    def test_disjoint_mutable_and_immutable(self):
        dc = DesignConstraints(mutable_positions=[1, 2], immutable_positions=[3, 4])
        assert dc.mutable_positions == [1, 2] and dc.immutable_positions == [3, 4]

    def test_part_name_mutable_with_immutable_positions(self):
        # A part name cannot overlap-check against explicit positions, so both are kept.
        dc = DesignConstraints(mutable_positions="tmd", immutable_positions=[1])
        assert dc.mutable_positions == "tmd" and dc.immutable_positions == [1]

    def test_identity_bounds_may_be_equal(self):
        dc = DesignConstraints(min_identity=0.8, max_identity=0.8)
        assert dc.min_identity == dc.max_identity == 0.8

    def test_permitted_and_forbidden_partial_overlap(self):
        dc = DesignConstraints(permitted_substitutions=["A", "V", "P"],
                               forbidden_substitutions=["P"])
        assert dc.permitted_substitutions == ["A", "V", "P"]

    def test_dict_rules_never_trigger_the_global_overlap_guard(self):
        dc = DesignConstraints(permitted_substitutions={2: ["A"]},
                               forbidden_substitutions={2: ["A"]})
        assert dc.permitted_substitutions == {2: ["A"]}

    # Negative cross-parameter cases
    def test_mutable_and_immutable_overlap_raises(self):
        with pytest.raises(ValueError, match="should not overlap"):
            DesignConstraints(mutable_positions=[1, 2, 3], immutable_positions=[3])

    def test_min_identity_above_max_identity_raises(self):
        with pytest.raises(ValueError, match="should be <= 'max_identity'"):
            DesignConstraints(min_identity=0.9, max_identity=0.5)

    def test_forbidden_excludes_every_permitted_raises(self):
        with pytest.raises(ValueError, match="should leave at least one"):
            DesignConstraints(permitted_substitutions=["A", "V"],
                              forbidden_substitutions=["A", "V", "P"])

    def test_full_set_with_one_bad_field_raises(self):
        with pytest.raises(ValueError, match="'n_mut_max'"):
            DesignConstraints(immutable_positions=[1], permitted_substitutions=["A"],
                              n_mut_max=-1, forbidden_motifs=["WW"])

    def test_dict_rule_with_bad_amino_acid_raises(self):
        with pytest.raises(ValueError, match="canonical amino acids"):
            DesignConstraints(mutable_positions=[1, 2],
                              permitted_substitutions={2: ["A", "z"]})

    def test_overlap_guard_runs_after_each_field_is_normalized(self):
        # numpy-free tuples are normalized to lists first, then the overlap guard fires.
        with pytest.raises(ValueError, match="should not overlap"):
            DesignConstraints(mutable_positions=(5, 6), immutable_positions=(6,))


class TestDesignConstraintsGoldenValues:
    """Hand-computed expectations for normalization, export and round-tripping."""

    def test_to_dict_has_exactly_the_documented_fields(self):
        dict_constraints = DesignConstraints().to_dict()
        assert list(dict_constraints) == list(ut.LIST_DESIGN_CONSTRAINTS) + ["parent"]

    def test_field_order_is_the_documented_reason_order(self):
        assert ut.LIST_DESIGN_CONSTRAINTS == [
            "immutable_positions", "mutable_positions", "permitted_substitutions",
            "forbidden_substitutions", "n_mut_max", "min_identity", "max_identity",
            "forbidden_motifs", "required_motifs"]

    def test_to_dict_values(self):
        dc = DesignConstraints(immutable_positions=[3, 1], n_mut_max=2, parent="MKLAG")
        dict_constraints = dc.to_dict()
        assert dict_constraints["immutable_positions"] == [3, 1]   # input order preserved
        assert dict_constraints["n_mut_max"] == 2
        assert dict_constraints["parent"] == "MKLAG"
        assert dict_constraints["min_identity"] is None

    def test_to_dict_returns_copies(self):
        dc = DesignConstraints(immutable_positions=[1], permitted_substitutions={2: ["A"]})
        dict_constraints = dc.to_dict()
        dict_constraints["immutable_positions"].append(99)
        dict_constraints["permitted_substitutions"][2].append("V")
        assert dc.immutable_positions == [1] and dc.permitted_substitutions == {2: ["A"]}

    def test_from_dict_round_trips_to_an_equal_object(self):
        dc = DesignConstraints(immutable_positions=[1], mutable_positions="tmd",
                               permitted_substitutions=["A", "V"], n_mut_max=3,
                               min_identity=0.8, forbidden_motifs=["WW"], parent="MKLAGTWYVF")
        assert DesignConstraints.from_dict(dict_constraints=dc.to_dict()) == dc

    def test_from_dict_round_trips_through_json(self):
        dc = DesignConstraints(permitted_substitutions={3: ["A", "V"]}, n_mut_max=2,
                               max_identity=0.95, required_motifs=["KL"], parent="MKLAG")
        # JSON turns the integer position keys into digit strings; from_dict converts back.
        dict_json = json.loads(json.dumps(dc.to_dict()))
        assert list(dict_json["permitted_substitutions"]) == ["3"]
        assert DesignConstraints.from_dict(dict_constraints=dict_json) == dc

    def test_from_dict_accepts_a_partial_dict(self):
        dc = DesignConstraints.from_dict(dict_constraints={"n_mut_max": 4})
        assert dc.n_mut_max == 4 and dc.parent is None

    def test_from_dict_rejects_an_unknown_field(self):
        with pytest.raises(ValueError, match="should contain only the DesignConstraints fields"):
            DesignConstraints.from_dict(dict_constraints={"n_mut": 4})

    def test_from_dict_rejects_a_non_dict(self):
        with pytest.raises(ValueError, match="'dict_constraints'"):
            DesignConstraints.from_dict(dict_constraints=[("n_mut_max", 4)])

    def test_from_dict_revalidates_every_field(self):
        with pytest.raises(ValueError, match="'n_mut_max'"):
            DesignConstraints.from_dict(dict_constraints={"n_mut_max": 0})

    def test_equality_is_field_based(self):
        assert DesignConstraints(n_mut_max=2) == DesignConstraints(n_mut_max=2)
        assert DesignConstraints(n_mut_max=2) != DesignConstraints(n_mut_max=3)
        assert DesignConstraints(n_mut_max=2) != "n_mut_max=2"

    def test_repr_lists_only_the_set_fields(self):
        assert repr(DesignConstraints(n_mut_max=2)) == "DesignConstraints(n_mut_max=2)"
        assert repr(DesignConstraints()) == "DesignConstraints()"

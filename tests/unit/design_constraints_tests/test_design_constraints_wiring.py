"""This is a script to test the DesignConstraints wiring of AAMut, SeqMut and SeqOpt.

The same object must drive all three classes, the ``region`` / ``to_aa`` / ``n_mut_max``
shorthands must keep their meaning by building one internally, and the published
``SeqOpt.run(constraints=[...])`` list of feasibility callables must keep working.
"""
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.protein_engineering import DesignConstraints

OBJECTIVES = [("activity", "max", ut.COL_DELTA_PRED), ("parsimony", "min", ut.COL_N_MUT)]


def _variants():
    """Two variants on P1: a genuine double (I12P, L13K) and a single (M14A).

    The wild-type residue at position 11 is already ``A``, so a ``11 -> A`` row would be a
    silent no-op rather than a mutation; positions 12-14 are used to keep the mutation counts
    unambiguous.
    """
    return pd.DataFrame({
        ut.COL_ENTRY: ["P1", "P1", "P1"],
        ut.COL_VARIANT: ["v1", "v1", "v2"],
        ut.COL_POS: [12, 13, 14],
        ut.COL_TO_AA: ["P", "K", "A"],
    })


class TestDesignConstraintsWiring:
    """One class / one method per test: the object and the scalars agree."""

    def test_aamut_object_matches_the_to_aa_shorthand(self):
        aam = aa.AAMut()
        df_scalar = aam.run(from_aa="M", to_aa=["V", "A"])
        df_object = aam.run(from_aa="M",
                            constraints=DesignConstraints(permitted_substitutions=["V", "A"]))
        assert df_scalar.equals(df_object)

    def test_aamut_forbidden_substitutions(self):
        df_impact = aa.AAMut().run(
            from_aa="M", constraints=DesignConstraints(forbidden_substitutions=["P"]))
        assert "P" not in set(df_impact[ut.COL_TO_AA])

    def test_seqmut_scan_object_matches_the_region_and_to_aa_shorthands(self, df_seq_pos, df_feat):
        seqm = aa.SeqMut()
        df_scalar = seqm.scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd",
                              to_aa=["A", "L", "V", "P"])
        df_object = seqm.scan(df_seq=df_seq_pos, df_feat=df_feat,
                              constraints=DesignConstraints(
                                  mutable_positions="tmd",
                                  permitted_substitutions=["A", "L", "V", "P"]))
        assert df_scalar.equals(df_object)

    def test_seqmut_scan_drops_immutable_positions(self, df_seq_pos, df_feat):
        df_scan = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat,
                                   constraints=DesignConstraints(mutable_positions="tmd",
                                                                 immutable_positions=[11]))
        assert 11 not in set(df_scan[ut.COL_POS]) and len(df_scan) > 0

    def test_seqmut_scan_drops_forbidden_substitutions(self, df_seq_pos, df_feat):
        df_scan = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat,
                                   constraints=DesignConstraints(mutable_positions="tmd",
                                                                 forbidden_substitutions=["P"]))
        assert "P" not in set(df_scan[ut.COL_TO_AA])

    def test_seqmut_suggest_object_matches_the_shorthands(self, df_seq_pos, df_feat):
        seqm = aa.SeqMut()
        df_scalar = seqm.suggest(df_seq=df_seq_pos, df_feat=df_feat, n=5, region="tmd",
                                 to_aa=["A", "L", "V"])
        df_object = seqm.suggest(df_seq=df_seq_pos, df_feat=df_feat, n=5,
                                 constraints=DesignConstraints(
                                     mutable_positions="tmd",
                                     permitted_substitutions=["A", "L", "V"]))
        assert df_scalar.equals(df_object)

    def test_seqmut_combine_appends_the_feasibility_columns(self, df_seq_pos, df_feat):
        df_variant = aa.SeqMut().combine(
            df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
            constraints=DesignConstraints(n_mut_max=1))
        assert list(df_variant.columns)[-2:] == [ut.COL_IS_FEASIBLE, ut.COL_REASONS]

    def test_seqmut_combine_without_constraints_is_unchanged(self, df_seq_pos, df_feat):
        df_variant = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        assert list(df_variant.columns) == ut.COLS_SEQMUT_VARIANT

    def test_seqopt_object_matches_the_scalars(self, df_seq_pos, df_feat, model):
        kwargs = dict(df_seq=df_seq_pos, df_feat=df_feat, objectives=OBJECTIVES,
                      pop_size=8, n_gen=3)
        df_scalar = aa.SeqOpt(model=model, random_state=7).run(n_mut_max=3, region="tmd", **kwargs)
        df_object = aa.SeqOpt(model=model, random_state=7).run(
            constraints=DesignConstraints(n_mut_max=3, mutable_positions="tmd"), **kwargs)
        assert df_scalar.equals(df_object)

    def test_seqopt_still_accepts_the_published_callable_list(self, df_seq_pos, df_feat, model):
        df_pareto = aa.SeqOpt(model=model, random_state=7).run(
            df_seq=df_seq_pos, df_feat=df_feat, objectives=OBJECTIVES, pop_size=8, n_gen=3,
            n_mut_max=3, region="tmd", constraints=[lambda genome: 11 not in genome])
        assert len(df_pareto) >= 1

    def test_seqopt_honours_immutable_positions(self, df_seq_pos, df_feat, model):
        df_pareto = aa.SeqOpt(model=model, random_state=7).run(
            df_seq=df_seq_pos, df_feat=df_feat, objectives=OBJECTIVES, pop_size=8, n_gen=3,
            constraints=DesignConstraints(n_mut_max=3, mutable_positions="tmd",
                                          immutable_positions=[11]))
        assert not df_pareto[ut.COL_VARIANT].str.contains(r"\D11\D", regex=True).any()

    # Negative cases
    def test_aamut_conflicting_to_aa_raises(self):
        with pytest.raises(ValueError, match="one place only"):
            aa.AAMut().run(from_aa="M", to_aa=["W"],
                           constraints=DesignConstraints(permitted_substitutions=["A"]))

    def test_aamut_non_design_constraints_raises(self):
        with pytest.raises(ValueError, match="'constraints'"):
            aa.AAMut().run(from_aa="M", constraints={"n_mut_max": 2})

    def test_aamut_constraints_excluding_every_target_raises(self):
        with pytest.raises(ValueError, match="at least one target amino acid"):
            aa.AAMut().run(from_aa="M", constraints=DesignConstraints(
                forbidden_substitutions=list(ut.LIST_CANONICAL_AA)))

    def test_seqmut_scan_conflicting_region_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError, match="one place only"):
            aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="jmd_n",
                             constraints=DesignConstraints(mutable_positions="tmd"))

    def test_seqmut_scan_conflicting_to_aa_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError, match="one place only"):
            aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, to_aa=["A"],
                             constraints=DesignConstraints(permitted_substitutions=["V"]))

    def test_seqmut_scan_constraints_excluding_everything_raises(self, df_seq_pos, df_feat):
        # Position 11 is the only mutable one and its single permitted target is forbidden there,
        # so the scan plan is emptied by the constraints rather than by 'region'.
        with pytest.raises(ValueError, match="at least one substitution"):
            aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat,
                             constraints=DesignConstraints(mutable_positions=[11],
                                                           immutable_positions=[12],
                                                           permitted_substitutions=["V"],
                                                           forbidden_substitutions={11: ["V"]}))

    def test_seqmut_suggest_non_design_constraints_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError, match="'constraints'"):
            aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, n=5, constraints="tmd")

    def test_seqmut_combine_non_design_constraints_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError, match="'constraints'"):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                                constraints=[lambda genome: True])

    def test_seqopt_conflicting_n_mut_max_raises(self, df_seq_pos, df_feat, model):
        with pytest.raises(ValueError, match="one place only"):
            aa.SeqOpt(model=model).run(df_seq=df_seq_pos, df_feat=df_feat, objectives=OBJECTIVES,
                                       pop_size=8, n_gen=2, n_mut_max=4,
                                       constraints=DesignConstraints(n_mut_max=3))

    def test_seqopt_non_callable_list_element_raises(self, df_seq_pos, df_feat, model):
        with pytest.raises(ValueError, match="constraints"):
            aa.SeqOpt(model=model).run(df_seq=df_seq_pos, df_feat=df_feat, objectives=OBJECTIVES,
                                       pop_size=8, n_gen=2, constraints=[123])


class TestDesignConstraintsWiringComplex:
    """One constraint set driving several classes, and cross-parameter failures."""

    def test_one_object_drives_seqmut_and_seqopt_with_the_same_limits(self, df_seq_pos,
                                                                     df_feat, model):
        constraints = DesignConstraints(mutable_positions="tmd", immutable_positions=[11],
                                        forbidden_substitutions=["P"], n_mut_max=3)
        df_scan = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, constraints=constraints)
        df_pareto = aa.SeqOpt(model=model, random_state=7).run(
            df_seq=df_seq_pos, df_feat=df_feat, objectives=OBJECTIVES, pop_size=8, n_gen=3,
            constraints=constraints)
        assert 11 not in set(df_scan[ut.COL_POS]) and "P" not in set(df_scan[ut.COL_TO_AA])
        assert not df_pareto[ut.COL_VARIANT].str.contains(r"\D11\D", regex=True).any()
        assert df_pareto[ut.COL_N_MUT].max() <= 3

    def test_scalar_fills_a_field_the_object_leaves_open(self, df_seq_pos, df_feat):
        seqm = aa.SeqMut()
        df_scalar = seqm.scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd", to_aa=["A", "V"])
        df_merged = seqm.scan(df_seq=df_seq_pos, df_feat=df_feat, to_aa=["A", "V"],
                              constraints=DesignConstraints(mutable_positions="tmd"))
        assert df_scalar.equals(df_merged)

    def test_repeating_a_scalar_identically_is_not_a_conflict(self, df_seq_pos, df_feat):
        df_scan = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd",
                                   constraints=DesignConstraints(mutable_positions="tmd"))
        assert len(df_scan) > 0

    def test_default_n_mut_max_beside_an_object_is_not_a_conflict(self, df_seq_pos, df_feat,
                                                                  model):
        df_pareto = aa.SeqOpt(model=model, random_state=7).run(
            df_seq=df_seq_pos, df_feat=df_feat, objectives=OBJECTIVES, pop_size=8, n_gen=3,
            n_mut_max=5, constraints=DesignConstraints(n_mut_max=2, mutable_positions="tmd"))
        assert df_pareto[ut.COL_N_MUT].max() <= 2

    def test_combine_reasons_are_empty_exactly_for_feasible_variants(self, df_seq_pos, df_feat):
        df_variant = aa.SeqMut().combine(
            df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
            constraints=DesignConstraints(n_mut_max=1, forbidden_substitutions=["P"]))
        for feasible, reasons in zip(df_variant[ut.COL_IS_FEASIBLE], df_variant[ut.COL_REASONS]):
            assert (reasons == "") is bool(feasible)
        assert not df_variant[ut.COL_IS_FEASIBLE].all()   # the double violates both limits

    def test_combine_reports_every_violated_limit_of_a_rejected_variant(self, df_seq_pos,
                                                                        df_feat):
        df_variant = aa.SeqMut().combine(
            df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
            constraints=DesignConstraints(n_mut_max=1, forbidden_substitutions=["P"]))
        row = df_variant[~df_variant[ut.COL_IS_FEASIBLE]].iloc[0]
        assert "forbidden_substitutions" in row[ut.COL_REASONS]
        assert "n_mut_max" in row[ut.COL_REASONS]

    # Negative cross-class cases
    def test_seqopt_empty_search_space_raises(self, df_seq_pos, df_feat, model):
        with pytest.raises(ValueError, match="search space is empty"):
            aa.SeqOpt(model=model).run(
                df_seq=df_seq_pos, df_feat=df_feat, objectives=OBJECTIVES, pop_size=8, n_gen=2,
                constraints=DesignConstraints(mutable_positions=[11], immutable_positions=[11 + 1],
                                              forbidden_substitutions=list(ut.LIST_CANONICAL_AA)))

    def test_seqopt_conflicting_region_and_object_raises(self, df_seq_pos, df_feat, model):
        with pytest.raises(ValueError, match="one place only"):
            aa.SeqOpt(model=model).run(
                df_seq=df_seq_pos, df_feat=df_feat, objectives=OBJECTIVES, pop_size=8, n_gen=2,
                region="jmd_c", constraints=DesignConstraints(mutable_positions="tmd"))

    def test_seqmut_scan_conflicting_region_list_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError, match="one place only"):
            aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region=[11, 12],
                             constraints=DesignConstraints(mutable_positions=[11, 13]))

    def test_aamut_position_keyed_rule_is_not_applicable(self):
        # AAMut is residue-level: a position-keyed rule cannot bind, so nothing is filtered.
        df_scalar = aa.AAMut().run(from_aa="M", to_aa=["V", "A"])
        df_object = aa.AAMut().run(from_aa="M", to_aa=["V", "A"],
                                   constraints=DesignConstraints(
                                       forbidden_substitutions={3: ["V"]}))
        assert df_scalar.equals(df_object)

    def test_combine_with_an_impossible_identity_bound_rejects_everything(self, df_seq_pos,
                                                                         df_feat):
        df_variant = aa.SeqMut().combine(
            df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
            constraints=DesignConstraints(min_identity=1.0))
        assert not df_variant[ut.COL_IS_FEASIBLE].any()
        assert df_variant[ut.COL_REASONS].str.startswith("min_identity").all()

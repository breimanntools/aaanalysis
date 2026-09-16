"""
This is a script for the frontend of the DesignConstraints class: the one validated container
of protein-design limits that AAMut, SeqMut and SeqOpt all consume, so a constraint set written
for one of them can be handed to the others and every rejected candidate carries its reasons.
"""
from typing import Optional, List, Dict, Tuple, Union, Callable, Any
import numpy as np

import aaanalysis.utils as ut
from ._backend.design_constraints import apply_genome, comp_reasons


# I Helper Functions
def check_sequence(name=None, val=None, accept_none=True):
    """Check a protein sequence: a non-empty string of upper-case one-letter residue codes.

    Non-canonical codes (``X``, ``U``, ``B``, ``Z``) are accepted, because a wild-type sequence
    may legitimately carry them; the canonical-alphabet restriction applies to the substitution
    rules and motifs, which are always written in the 20 canonical amino acids.
    """
    if val is None:
        if not accept_none:
            raise ValueError(f"'{name}' (None) should be a protein sequence string.")
        return None
    ut.check_str(name=name, val=val, accept_none=False)
    if len(val) == 0:
        raise ValueError(f"'{name}' ('') should be a non-empty protein sequence string.")
    wrong = sorted({aa for aa in val if not ("A" <= aa <= "Z")})
    if len(wrong) > 0:
        raise ValueError(f"'{name}' ({wrong}) should contain only upper-case one-letter "
                         f"residue codes.")
    return val


def check_positions(name=None, val=None):
    """Check a list of 1-based positions; the order of the input list is preserved."""
    if val is None:
        return None
    val = ut.check_list_like(name=name, val=val)
    if len(val) == 0:
        raise ValueError(f"'{name}' ([]) should be a non-empty list of 1-based positions or None.")
    for pos in val:
        ut.check_number_range(name=f"'{name}' position", val=pos, min_val=1, just_int=True)
    return [int(pos) for pos in val]


def check_mutable_positions(name="mutable_positions", val=None):
    """Check the mutable span: a sequence-part name or a list of 1-based positions."""
    if val is None:
        return None
    if isinstance(val, str):
        if val.lower() not in ut.COLS_SEQ_PARTS:
            raise ValueError(f"'{name}' ({val}) should be one of {ut.COLS_SEQ_PARTS} "
                             f"or a list of 1-based positions.")
        return val.lower()
    return check_positions(name=name, val=val)


def check_aa_set(name=None, val=None):
    """Check a non-empty set of canonical amino acids; the input order is preserved."""
    val = ut.check_list_like(name=name, val=val, accept_str=True)
    if len(val) == 0:
        raise ValueError(f"'{name}' ([]) should be a non-empty list of canonical amino acids.")
    wrong = [aa for aa in val if aa not in ut.LIST_CANONICAL_AA]
    if len(wrong) > 0:
        raise ValueError(f"'{name}' ({wrong}) should be canonical amino acids from "
                         f"{ut.LIST_CANONICAL_AA}.")
    return [str(aa) for aa in val]


def check_substitutions(name=None, val=None):
    """Check a substitution rule: an amino acid list, or a {1-based position: list} dict."""
    if val is None:
        return None
    if not isinstance(val, dict):
        return check_aa_set(name=name, val=val)
    if len(val) == 0:
        raise ValueError(f"'{name}' ({{}}) should be a non-empty {{position: amino acids}} dict.")
    rule = {}
    for pos, list_aa in val.items():
        # JSON round-trips integer keys as strings, so a digit string is accepted here.
        is_int = isinstance(pos, (int, np.integer)) and not isinstance(pos, bool)
        if not (is_int or (isinstance(pos, str) and pos.isdigit())):
            raise ValueError(f"'{name}' key ({pos}) should be a 1-based integer position "
                             f"(a digit string from a JSON round-trip is accepted).")
        ut.check_number_range(name=f"'{name}' position", val=int(pos), min_val=1, just_int=True)
        rule[int(pos)] = check_aa_set(name=f"{name}[{int(pos)}]", val=list_aa)
    return rule


def check_motifs(name=None, val=None):
    """Check a list of literal amino acid motifs (substrings, not regular expressions)."""
    if val is None:
        return None
    val = ut.check_list_like(name=name, val=val, accept_str=True)
    if len(val) == 0:
        raise ValueError(f"'{name}' ([]) should be a non-empty list of amino acid motifs.")
    for motif in val:
        ut.check_str(name=f"'{name}' motif", val=motif, accept_none=False)
        wrong = sorted({aa for aa in motif if aa not in ut.LIST_CANONICAL_AA})
        if len(motif) == 0 or len(wrong) > 0:
            raise ValueError(f"'{name}' motif ('{motif}') should be a non-empty string of "
                             f"canonical amino acids from {ut.LIST_CANONICAL_AA}.")
    return [str(motif) for motif in val]


def check_match_identity_bounds(min_identity=None, max_identity=None):
    """Check that the sequence-identity bounds are fractions in [0, 1] with min <= max."""
    for name, val in [("min_identity", min_identity), ("max_identity", max_identity)]:
        if val is not None:
            ut.check_number_range(name=name, val=val, min_val=0, max_val=1, just_int=False)
    if min_identity is not None and max_identity is not None and min_identity > max_identity:
        raise ValueError(f"'min_identity' ({min_identity}) should be <= 'max_identity' "
                         f"({max_identity}); the bounds would admit no candidate.")


def check_match_mutable_immutable(mutable_positions=None, immutable_positions=None):
    """Check that no position is declared both mutable and immutable."""
    if isinstance(mutable_positions, list) and immutable_positions is not None:
        overlap = sorted(set(mutable_positions) & set(immutable_positions))
        if len(overlap) > 0:
            raise ValueError(f"'immutable_positions' ({overlap}) should not overlap "
                             f"'mutable_positions' ({mutable_positions}); a position is either "
                             f"mutable or immutable.")


def check_match_permitted_forbidden(permitted_substitutions=None, forbidden_substitutions=None):
    """Check that the global substitution rules leave at least one amino acid allowed."""
    if isinstance(permitted_substitutions, list) and isinstance(forbidden_substitutions, list):
        left = [aa for aa in permitted_substitutions if aa not in set(forbidden_substitutions)]
        if len(left) == 0:
            raise ValueError(f"'forbidden_substitutions' ({forbidden_substitutions}) should leave "
                             f"at least one of 'permitted_substitutions' "
                             f"({permitted_substitutions}) allowed.")


def check_match_candidate_parent(candidate=None, parent=None):
    """Check that a candidate is a same-length substitution variant of its parent."""
    check_sequence(name="candidate", val=candidate, accept_none=False)
    if len(candidate) != len(parent):
        raise ValueError(f"'candidate' (len={len(candidate)}) should have the same length as "
                         f"'parent' (len={len(parent)}); DesignConstraints compares position by "
                         f"position, so insertions and deletions are out of scope.")


def check_constraints(name="constraints", val=None):
    """Check that a value is a :class:`DesignConstraints` object (or None)."""
    if val is None:
        return None
    if not isinstance(val, DesignConstraints):
        raise ValueError(f"'{name}' ({type(val).__name__}) should be a DesignConstraints object "
                         f"or None.")
    return val


def resolve_constraints(constraints=None, region=None, to_aa=None, n_mut_max=None,
                        n_mut_max_default=None):
    """Fold the scalar shorthands into one :class:`DesignConstraints` and reject conflicts.

    ``region``, ``to_aa`` and ``n_mut_max`` are shorthand for ``mutable_positions``,
    ``permitted_substitutions`` and ``n_mut_max``. They are folded into the object and read back
    out of it, so a limit has exactly one definition. Passing a scalar *and* an object that sets
    the same limit to a different value raises.

    Returns ``(constraints, region, to_aa, n_mut_max)`` with the three scalars taken from the
    resolved object.
    """
    dc = check_constraints(name="constraints", val=constraints)
    region = check_mutable_positions(name="region", val=region)
    to_aa = None if to_aa is None else check_aa_set(name="to_aa", val=to_aa)
    if n_mut_max is not None:
        ut.check_number_range(name="n_mut_max", val=n_mut_max, min_val=1, just_int=True)
    if dc is None:
        dc = DesignConstraints(mutable_positions=region, permitted_substitutions=to_aa,
                               n_mut_max=n_mut_max)
        return dc, region, to_aa, n_mut_max
    if region is not None and dc.mutable_positions is not None and region != dc.mutable_positions:
        raise ValueError(f"'region' ({region}) should be None or equal to "
                         f"'constraints.mutable_positions' ({dc.mutable_positions}); a design "
                         f"limit is defined in one place only.")
    if (to_aa is not None and dc.permitted_substitutions is not None
            and to_aa != dc.permitted_substitutions):
        raise ValueError(f"'to_aa' ({to_aa}) should be None or equal to "
                         f"'constraints.permitted_substitutions' ({dc.permitted_substitutions}); "
                         f"a design limit is defined in one place only.")
    # A scalar left at its documented default is not a second definition of the limit.
    n_mut_scalar = None if n_mut_max == n_mut_max_default else n_mut_max
    if n_mut_scalar is not None and dc.n_mut_max is not None and n_mut_scalar != dc.n_mut_max:
        raise ValueError(f"'n_mut_max' ({n_mut_max}) should be left at its default or equal to "
                         f"'constraints.n_mut_max' ({dc.n_mut_max}); a design limit is defined "
                         f"in one place only.")
    # Merge: a scalar fills only a field the object leaves open.
    args = dc.to_dict()
    if args["mutable_positions"] is None:
        args["mutable_positions"] = region
    if args["permitted_substitutions"] is None:
        args["permitted_substitutions"] = to_aa
    if args["n_mut_max"] is None:
        args["n_mut_max"] = n_mut_max
    dc = DesignConstraints.from_dict(dict_constraints=args)
    to_aa = dc.permitted_substitutions if isinstance(dc.permitted_substitutions, list) else None
    return dc, dc.mutable_positions, to_aa, dc.n_mut_max


# II Main Functions
class DesignConstraints:
    """
    Design Constraints (DesignConstraints) container for the sequence-design limits shared by
    :class:`AAMut`, :class:`SeqMut` and :class:`SeqOpt` [Breimann24a]_.

    A design campaign expresses the same limits over and over: which residues may change, how
    many substitutions are allowed, which target residues are off-limits, how close a variant
    must stay to its parent. ``DesignConstraints`` is the single place those limits are written
    down, so a constraint set built for one class is accepted by the other two, and a discarded
    candidate can say *why* it was discarded instead of vanishing silently.

    The primary contract is :meth:`DesignConstraints.check`, which answers ``(ok, reasons)`` for
    a candidate sequence; it is the only shape all three classes share.
    :meth:`DesignConstraints.as_predicate` adapts the same limits to the genome-shaped
    feasibility callable :meth:`SeqOpt.run` already consumes, and
    :meth:`DesignConstraints.to_dict` / :meth:`DesignConstraints.from_dict` round-trip a
    constraint set through JSON.

    The object subsumes the limits that already ship: the ``region`` and ``to_aa`` of
    :meth:`SeqMut.scan` and the ``n_mut_max`` / ``region`` / ``to_aa`` of :meth:`SeqOpt.run` stay
    as shorthand and build a ``DesignConstraints`` internally, so a limit is never expressed by
    two independent mechanisms. Passing both an object and a conflicting scalar raises.

    **Coordinate convention.** Every position is a **1-based position in the parent (wild-type)
    sequence** -- the same convention as the ``region`` parameter of :meth:`SeqMut.scan` and the
    ``pos`` column of the ``SeqMut`` / ``SeqOpt`` mutation tables. A candidate is a
    substitution-only variant and therefore has the parent's length; insertions and deletions are
    out of scope.

    Attributes
    ----------
    immutable_positions : list of int or None
        Positions that must keep their parent residue.
    mutable_positions : str or list of int or None
        The span a substitution may fall in (a part name or explicit positions).
    permitted_substitutions : list of str or dict or None
        The only target residues allowed, globally or per position.
    forbidden_substitutions : list of str or dict or None
        Target residues that are never allowed, globally or per position.
    n_mut_max : int or None
        Maximum number of substitutions relative to the parent.
    min_identity : float or None
        Lowest sequence identity to the parent a candidate may have.
    max_identity : float or None
        Highest sequence identity to the parent a candidate may have.
    forbidden_motifs : list of str or None
        Amino acid motifs that must not occur in a candidate.
    required_motifs : list of str or None
        Amino acid motifs that must occur in a candidate.
    parent : str or None
        Default parent (wild-type) sequence the limits are evaluated against.

    .. versionadded:: 1.2.0

    """
    def __init__(self,
                 *, immutable_positions: Optional[List[int]] = None,
                 mutable_positions: Optional[Union[str, List[int]]] = None,
                 permitted_substitutions: Optional[Union[List[str], Dict[int, List[str]]]] = None,
                 forbidden_substitutions: Optional[Union[List[str], Dict[int, List[str]]]] = None,
                 n_mut_max: Optional[int] = None,
                 min_identity: Optional[float] = None,
                 max_identity: Optional[float] = None,
                 forbidden_motifs: Optional[List[str]] = None,
                 required_motifs: Optional[List[str]] = None,
                 parent: Optional[str] = None,
                 ):
        """
        Parameters
        ----------
        immutable_positions : list of int, optional
            1-based parent positions that must keep their wild-type residue. A candidate that
            substitutes any of them is rejected, and :meth:`SeqMut.scan` / :meth:`SeqOpt.run`
            drop them from the search space. ``None`` (default) leaves every position mutable.
        mutable_positions : str or list of int, optional
            The span a substitution may fall in: a sequence-part name (``'jmd_n'``, ``'tmd'`` or
            ``'jmd_c'``) or a list of 1-based parent positions. This is the object form of the
            ``region`` shorthand. ``None`` (default) places no span limit. A part name needs the
            TMD coordinates of a ``df_seq`` and is therefore applied by the calling class, not by
            :meth:`DesignConstraints.check`, which evaluates an explicit position list only.
        permitted_substitutions : list of str or dict, optional
            The only target amino acids a substitution may introduce: a list of canonical amino
            acids (applies everywhere) or a ``{1-based position: list of amino acids}`` dict
            (applies only to the positions it lists; an unlisted position stays unrestricted).
            This is the object form of the ``to_aa`` shorthand. ``None`` (default) permits all 20
            canonical amino acids.
        forbidden_substitutions : list of str or dict, optional
            Target amino acids a substitution may never introduce, in the same two shapes as
            ``permitted_substitutions``. Applied on top of it, so a residue that is permitted and
            forbidden is forbidden. ``None`` (default) forbids nothing.
        n_mut_max : int, optional
            Maximum number of substitutions a candidate may carry relative to the parent
            (``>= 1``). This is the object form of the ``n_mut_max`` shorthand of
            :meth:`SeqOpt.run`. ``None`` (default) places no limit.
        min_identity : float, optional
            Lowest sequence identity to the parent a candidate may have, as a fraction in
            ``[0, 1]`` (identity = fraction of positions carrying the parent residue, so a
            candidate with ``k`` substitutions over a parent of length ``L`` has identity
            ``1 - k / L``). ``None`` (default) places no lower bound.
        max_identity : float, optional
            Highest sequence identity to the parent a candidate may have, as a fraction in
            ``[0, 1]``. Use it to force a minimum amount of change. ``None`` (default) places no
            upper bound.
        forbidden_motifs : list of str, optional
            Literal amino acid motifs (plain substrings, not regular expressions) that must not
            occur anywhere in a candidate, e.g. a protease site the design must avoid creating.
            ``None`` (default) forbids no motif.
        required_motifs : list of str, optional
            Literal amino acid motifs that must occur in a candidate, e.g. a binding epitope the
            design must retain. ``None`` (default) requires no motif.
        parent : str, optional
            Default parent (wild-type) sequence the limits are evaluated against, so
            :meth:`DesignConstraints.check` can be called with the candidate alone. ``None``
            (default) leaves it unset, and the parent is then passed per call.

        Raises
        ------
        ValueError
            If a position is not a 1-based integer, an amino acid is not canonical, a motif is
            empty or non-canonical, ``n_mut_max < 1``, an identity bound falls outside ``[0, 1]``
            or ``min_identity > max_identity``, a position is both mutable and immutable, or the
            global substitution rules leave no amino acid allowed.

        See Also
        --------
        * :meth:`DesignConstraints.check`: the primary ``(ok, reasons)`` contract.
        * :meth:`SeqMut.scan`: whose ``region`` / ``to_aa`` this object subsumes.
        * :meth:`SeqOpt.run`: whose ``constraints`` accepts this object directly.
        """
        # Validate
        self.immutable_positions = check_positions(name="immutable_positions",
                                                   val=immutable_positions)
        self.mutable_positions = check_mutable_positions(name="mutable_positions",
                                                         val=mutable_positions)
        self.permitted_substitutions = check_substitutions(name="permitted_substitutions",
                                                           val=permitted_substitutions)
        self.forbidden_substitutions = check_substitutions(name="forbidden_substitutions",
                                                           val=forbidden_substitutions)
        if n_mut_max is not None:
            ut.check_number_range(name="n_mut_max", val=n_mut_max, min_val=1, just_int=True)
        check_match_identity_bounds(min_identity=min_identity, max_identity=max_identity)
        self.forbidden_motifs = check_motifs(name="forbidden_motifs", val=forbidden_motifs)
        self.required_motifs = check_motifs(name="required_motifs", val=required_motifs)
        self.parent = check_sequence(name="parent", val=parent, accept_none=True)
        check_match_mutable_immutable(mutable_positions=self.mutable_positions,
                                      immutable_positions=self.immutable_positions)
        check_match_permitted_forbidden(permitted_substitutions=self.permitted_substitutions,
                                        forbidden_substitutions=self.forbidden_substitutions)
        # Store the scalars
        self.n_mut_max = None if n_mut_max is None else int(n_mut_max)
        self.min_identity = None if min_identity is None else float(min_identity)
        self.max_identity = None if max_identity is None else float(max_identity)

    # Helper methods
    def _resolve_parent(self, parent=None):
        """Return the effective parent sequence (a per-call parent overrides the stored one)."""
        if parent is not None:
            return check_sequence(name="parent", val=parent, accept_none=False)
        if self.parent is None:
            raise ValueError("'parent' (None) should be a sequence, passed here or set on the "
                             "DesignConstraints object; the limits are relative to it.")
        return self.parent

    def __eq__(self, other):
        """Two constraint sets are equal when every normalized field is equal."""
        if not isinstance(other, DesignConstraints):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        """Render the constraint set as its non-default constructor call."""
        args = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items() if v is not None)
        return f"DesignConstraints({args})"

    # Main methods
    def check(self,
              candidate: str,
              parent: Optional[str] = None,
              ) -> Tuple[bool, List[str]]:
        """
        Check one candidate sequence against every limit and report why it was rejected.

        This is the primary contract of the class and the one shape :class:`AAMut`,
        :class:`SeqMut` and :class:`SeqOpt` share: a candidate is either feasible, or it comes
        back with one explicit reason per violated limit, so "why was this candidate discarded?"
        is answerable without re-running the design loop.

        Parameters
        ----------
        candidate : str
            Candidate sequence: a substitution-only variant of the parent, of the same length
            and written in canonical amino acids.
        parent : str, optional
            Parent (wild-type) sequence the candidate is compared against, using 1-based
            positions. ``None`` (default) uses the ``parent`` given to the constructor.

        Returns
        -------
        ok : bool
            ``True`` when the candidate satisfies every limit (``reasons`` is then empty).
        reasons : list of str
            One ``"<field>: <detail>"`` string per violated limit, always in the field order
            ``immutable_positions``, ``mutable_positions``, ``permitted_substitutions``,
            ``forbidden_substitutions``, ``n_mut_max``, ``min_identity``, ``max_identity``,
            ``forbidden_motifs``, ``required_motifs``. A candidate violating n limits therefore
            reports exactly n reasons, in that order.

        Raises
        ------
        ValueError
            If ``candidate`` is not a non-empty canonical amino acid string, if its length
            differs from the parent's (insertions and deletions are out of scope), or if no
            parent is available (neither here nor on the object).

        See Also
        --------
        * :meth:`DesignConstraints.as_predicate`: the same limits as a ``SeqOpt`` feasibility callable.
        * :meth:`SeqMut.combine`: which appends the ``is_feasible`` / ``reasons`` columns from this check.

        .. versionadded:: 1.2.0

        """
        # Validate
        parent = self._resolve_parent(parent=parent)
        check_match_candidate_parent(candidate=candidate, parent=parent)
        # Collect one reason per violated limit
        reasons = comp_reasons(parent=parent, candidate=candidate, spec=self.to_dict())
        return len(reasons) == 0, reasons

    def as_predicate(self,
                     parent: Optional[str] = None,
                     ) -> Callable[[Dict[int, str]], bool]:
        """
        Adapt the limits to the genome-shaped feasibility callable the optimizer consumes.

        :meth:`SeqOpt.run` scores a variant as a *genome*, a sparse ``{1-based position: target
        amino acid}`` mapping over one wild-type, and feeds each feasibility predicate into the
        penalty term of its fitness function. That genome format is the optimizer's internal
        shape, so it stays out of :meth:`DesignConstraints.check`; this adapter bridges the two
        by applying the genome to the parent and checking the resulting sequence.

        Parameters
        ----------
        parent : str, optional
            Parent (wild-type) sequence the genomes are applied to. ``None`` (default) uses the
            ``parent`` given to the constructor.

        Returns
        -------
        predicate : callable
            A ``genome -> bool`` function returning ``True`` when the variant the genome encodes
            satisfies every limit, matching the ``constraints=[...]`` element type of
            :meth:`SeqOpt.run`.

        Raises
        ------
        ValueError
            If no parent is available (neither here nor on the object).

        See Also
        --------
        * :meth:`DesignConstraints.check`: the primary contract this predicate wraps.
        * :meth:`SeqOpt.run`: whose ``constraints`` list consumes the returned callable.

        .. versionadded:: 1.2.0

        """
        # Validate
        parent = self._resolve_parent(parent=parent)
        # Build the genome-shaped adapter over the sequence-level check
        spec = self.to_dict()

        def predicate(genome):
            candidate = apply_genome(parent=parent, genome=genome)
            return len(comp_reasons(parent=parent, candidate=candidate, spec=spec)) == 0

        return predicate

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the constraint set as a plain, JSON-serializable dictionary.

        The returned dictionary holds one key per constraint field plus ``parent``, with copies
        of the stored containers, so mutating it never changes the object. It is the input of
        :meth:`DesignConstraints.from_dict`, which reconstructs an equal object.

        Returns
        -------
        dict_constraints : dict
            One key per field of ``ut.LIST_DESIGN_CONSTRAINTS`` plus ``parent``; unset fields
            are ``None``.

        See Also
        --------
        * :meth:`DesignConstraints.from_dict`: the inverse.

        .. versionadded:: 1.2.0

        """
        def _copy(val):
            if isinstance(val, dict):
                return {pos: list(list_aa) for pos, list_aa in val.items()}
            if isinstance(val, list):
                return list(val)
            return val

        dict_constraints = {field: _copy(getattr(self, field))
                            for field in ut.LIST_DESIGN_CONSTRAINTS}
        dict_constraints["parent"] = self.parent
        return dict_constraints

    @classmethod
    def from_dict(cls, dict_constraints: Dict[str, Any]) -> "DesignConstraints":
        """
        Rebuild a constraint set from the dictionary produced by :meth:`DesignConstraints.to_dict`.

        Every field is re-validated, so a dictionary that has been through JSON is accepted:
        integer position keys that JSON turned into digit strings are converted back, and a
        malformed field raises the same message the constructor would.

        Parameters
        ----------
        dict_constraints : dict
            Constraint fields, as produced by :meth:`DesignConstraints.to_dict`. Missing keys
            default to ``None``; unknown keys are rejected.

        Returns
        -------
        constraints : DesignConstraints
            A new object equal to the one ``dict_constraints`` was exported from.

        Raises
        ------
        ValueError
            If ``dict_constraints`` is not a dictionary, carries a key that is not a constraint
            field, or holds a value the constructor rejects.

        See Also
        --------
        * :meth:`DesignConstraints.to_dict`: the inverse.

        .. versionadded:: 1.2.0

        """
        # Validate
        ut.check_dict(name="dict_constraints", val=dict_constraints, accept_none=False)
        fields = list(ut.LIST_DESIGN_CONSTRAINTS) + ["parent"]
        wrong = [key for key in dict_constraints if key not in fields]
        if len(wrong) > 0:
            raise ValueError(f"'dict_constraints' ({wrong}) should contain only the "
                             f"DesignConstraints fields {fields}.")
        # Rebuild (the constructor re-validates every field)
        return cls(**{field: dict_constraints.get(field) for field in fields})

"""
This is a script for the backend of the DesignConstraints container shared by the AAMut,
SeqMut and SeqOpt frontends.

It turns a (parent, candidate) sequence pair into an ordered list of machine-readable
rejection reasons, and applies the same limits structurally to a SeqMut mutation plan, an
AAMut substitution-pair set, or a SeqOpt search space. Positions are 1-based over the parent
sequence (the ``region`` convention). The frontend validates every field; these helpers trust
the normalized spec dict it passes in.
"""
from typing import Any, Dict, List, Tuple

import aaanalysis.utils as ut

# Identity bounds are compared with a small tolerance so a hand-set bound such as 0.9 is not
# missed by the binary representation of n_same / len(parent).
TOL_IDENTITY = 1e-9


# I Helper Functions
def rule_at_(rule=None, pos=None):
    """Return the amino acids a substitution rule constrains at 1-based ``pos``, or None.

    A list rule applies globally; a dict rule applies only to the positions it lists, so an
    unlisted position is unrestricted (``None``).
    """
    if rule is None:
        return None
    if isinstance(rule, dict):
        return rule.get(pos)
    return rule


def comp_mutations(parent=None, candidate=None):
    """Return the ``(pos, from_aa, to_aa)`` triples where ``candidate`` differs from ``parent``."""
    return [(i + 1, a, b) for i, (a, b) in enumerate(zip(parent, candidate)) if a != b]


def comp_identity(parent=None, candidate=None):
    """Return the fraction of positions at which ``candidate`` matches ``parent`` (in [0, 1])."""
    if len(parent) == 0:
        raise RuntimeError("'parent' is empty; sequence identity to the parent is undefined.")
    n_same = sum(1 for a, b in zip(parent, candidate) if a == b)
    return n_same / len(parent)


def has_sequence_limits(spec=None):
    """True if the spec carries a limit that can only be checked on a full candidate sequence."""
    return any(spec.get(field) is not None for field in ut.LIST_DESIGN_SEQ_LIMITS)


def is_allowed_substitution(pos=None, to_aa=None, spec=None):
    """True if substituting 1-based ``pos`` by ``to_aa`` satisfies the position/residue limits."""
    immutable = spec.get("immutable_positions")
    if immutable is not None and pos in set(immutable):
        return False
    mutable = spec.get("mutable_positions")
    if isinstance(mutable, list) and pos not in set(mutable):
        return False
    allowed = rule_at_(rule=spec.get("permitted_substitutions"), pos=pos)
    if allowed is not None and to_aa not in set(allowed):
        return False
    banned = rule_at_(rule=spec.get("forbidden_substitutions"), pos=pos)
    if banned is not None and to_aa in set(banned):
        return False
    return True


# II Main Functions
def apply_genome(parent=None, genome=None):
    """Apply a ``{1-based pos: to_aa}`` genome to ``parent`` and return the variant sequence."""
    chars = list(parent)
    for pos, to_aa in genome.items():
        chars[pos - 1] = to_aa
    return "".join(chars)


def comp_reasons(parent=None, candidate=None, spec=None):
    """Return the ordered rejection reasons a candidate collects (an empty list = feasible).

    Parameters
    ----------
    parent : str
        Reference (wild-type) sequence.
    candidate : str
        Variant sequence of the same length (substitutions only).
    spec : dict
        Normalized constraint fields (``DesignConstraints.to_dict()``).

    Returns
    -------
    reasons : list of str
        One ``"<field>: <detail>"`` string per violated limit, in ``ut.LIST_DESIGN_CONSTRAINTS``
        order, so a candidate violating n limits reports exactly n reasons.
    """
    reasons: List[str] = []
    muts = comp_mutations(parent=parent, candidate=candidate)
    positions = [pos for pos, _from_aa, _to_aa in muts]
    # 1) immutable_positions
    immutable = spec.get("immutable_positions")
    if immutable is not None:
        hit = [p for p in positions if p in set(immutable)]
        if len(hit) > 0:
            reasons.append(f"immutable_positions: position(s) {hit} are immutable")
    # 2) mutable_positions — a part name is applied structurally by the calling class, so only
    # an explicit position list can be checked here.
    mutable = spec.get("mutable_positions")
    if isinstance(mutable, list):
        hit = [p for p in positions if p not in set(mutable)]
        if len(hit) > 0:
            reasons.append(f"mutable_positions: position(s) {hit} are outside the mutable set")
    # 3) permitted_substitutions
    permitted = spec.get("permitted_substitutions")
    if permitted is not None:
        hit = []
        for pos, from_aa, to_aa in muts:
            allowed = rule_at_(rule=permitted, pos=pos)
            if allowed is not None and to_aa not in set(allowed):
                hit.append(f"{from_aa}{pos}{to_aa}")
        if len(hit) > 0:
            reasons.append(f"permitted_substitutions: substitution(s) {hit} are not permitted")
    # 4) forbidden_substitutions
    forbidden = spec.get("forbidden_substitutions")
    if forbidden is not None:
        hit = []
        for pos, from_aa, to_aa in muts:
            banned = rule_at_(rule=forbidden, pos=pos)
            if banned is not None and to_aa in set(banned):
                hit.append(f"{from_aa}{pos}{to_aa}")
        if len(hit) > 0:
            reasons.append(f"forbidden_substitutions: substitution(s) {hit} are forbidden")
    # 5) n_mut_max
    n_mut_max = spec.get("n_mut_max")
    if n_mut_max is not None and len(muts) > n_mut_max:
        reasons.append(f"n_mut_max: {len(muts)} mutations exceed the maximum of {n_mut_max}")
    # 6, 7) identity bounds
    min_identity = spec.get("min_identity")
    max_identity = spec.get("max_identity")
    if min_identity is not None or max_identity is not None:
        identity = comp_identity(parent=parent, candidate=candidate)
        if min_identity is not None and identity < min_identity - TOL_IDENTITY:
            reasons.append(f"min_identity: identity {identity:.4f} is below the minimum "
                           f"of {min_identity:.4f}")
        if max_identity is not None and identity > max_identity + TOL_IDENTITY:
            reasons.append(f"max_identity: identity {identity:.4f} is above the maximum "
                           f"of {max_identity:.4f}")
    # 8) forbidden_motifs
    forbidden_motifs = spec.get("forbidden_motifs")
    if forbidden_motifs is not None:
        hit = [m for m in forbidden_motifs if m in candidate]
        if len(hit) > 0:
            reasons.append(f"forbidden_motifs: motif(s) {hit} occur in the candidate")
    # 9) required_motifs
    required_motifs = spec.get("required_motifs")
    if required_motifs is not None:
        hit = [m for m in required_motifs if m not in candidate]
        if len(hit) > 0:
            reasons.append(f"required_motifs: motif(s) {hit} are absent from the candidate")
    return reasons


def filter_scan_plan(df_plan=None, spec=None):
    """Drop the SeqMut plan rows whose ``(pos, to_aa)`` a spec's position/residue limits forbid."""
    keep = [is_allowed_substitution(pos=int(pos), to_aa=to_aa, spec=spec)
            for pos, to_aa in zip(df_plan[ut.COL_POS], df_plan[ut.COL_TO_AA])]
    return df_plan[keep].reset_index(drop=True)


def filter_substitution_pairs(list_from=None, list_to=None, spec=None):
    """Restrict an AAMut from/to alphabet to the residue-level substitution limits.

    Only the global (list-form) permitted / forbidden substitutions apply: AAMut is
    residue-level and carries no positions, so position-keyed rules are not applicable.
    """
    permitted = spec.get("permitted_substitutions")
    forbidden = spec.get("forbidden_substitutions")
    if isinstance(permitted, list):
        list_to = [aa for aa in list_to if aa in set(permitted)]
    if isinstance(forbidden, list):
        list_to = [aa for aa in list_to if aa not in set(forbidden)]
    return list(list_from), list(list_to)


def filter_search_space(positions=None, alphabet=None, spec=None):
    """Restrict a SeqOpt search space to the limits enforceable by the genome operators."""
    immutable = spec.get("immutable_positions")
    if immutable is not None:
        positions = [p for p in positions if p not in set(immutable)]
    forbidden = spec.get("forbidden_substitutions")
    if isinstance(forbidden, list):
        alphabet = [aa for aa in alphabet if aa not in set(forbidden)]
    return list(positions), list(alphabet)

"""
This is a script for the backend of the candidate-lineage record shared by the SeqMut and
SeqOpt frontends.

It turns a (source, candidate) sequence pair into the ordered, parent-relative mutations and
the content hash that identifies the candidate, and it walks a list of records back to its
root. Positions are 1-based over the source (parent) sequence, the same convention as the
``region`` argument and the ``pos`` column. The frontend validates every field; these helpers
trust the normalized arguments it passes in.
"""
from typing import Any, Dict, List, Optional
import hashlib
import json

import aaanalysis.utils as ut
from .design_constraints import comp_mutations


# I Helper Functions
def comp_digest(text=None):
    """Return the stable ``sha256:<hex>`` digest of a text payload (process-independent)."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def comp_variant_label(mutations=None):
    """Return the ``'+'``-joined mutation labels (``''`` when there is no mutation)."""
    return "+".join(f"{m[ut.COL_FROM_AA]}{m[ut.COL_POS]}{m[ut.COL_TO_AA]}" for m in mutations)


# II Main Functions
def build_mutations(source_seq=None, candidate_seq=None) -> List[Dict[str, Any]]:
    """Return the ordered, source-relative mutations of a candidate.

    One ``{pos, from_aa, to_aa}`` dict per differing position, ordered by ascending 1-based
    position. The order is canonical rather than historical: a substitution-only candidate is
    independent of the order its mutations were applied in, so ordering by position is what
    makes the content hash agree between two runs.
    """
    return [{ut.COL_POS: int(pos), ut.COL_FROM_AA: str(from_aa), ut.COL_TO_AA: str(to_aa)}
            for pos, from_aa, to_aa in comp_mutations(parent=source_seq, candidate=candidate_seq)]


def comp_candidate_id(source_seq=None, mutations=None) -> str:
    """Return the content hash over the source sequence and its ordered mutations.

    The hashed payload is ``"<source_seq>|<'+'-joined mutation labels>"``, e.g.
    ``"MKLAGTWYVF|L3A+G5W"``, so the identifier depends on nothing but the content: two runs
    that build the same mutant from the same source assign the same identifier.
    """
    return comp_digest(text=f"{source_seq}|{comp_variant_label(mutations=mutations)}")


def comp_constraints_digest(spec=None) -> Optional[str]:
    """Return the digest of an applied constraint spec, or ``None`` when none was applied.

    The spec is the ``DesignConstraints.to_dict()`` export, serialized with sorted keys so two
    equal constraint sets digest equally regardless of how they were built.
    """
    if spec is None:
        return None
    return comp_digest(text=json.dumps(spec, sort_keys=True, default=str))


def build_record(source_seq=None, candidate_seq=None, method=None, objective_values=None,
                 seed=None, constraints_digest=None, parent_id=None) -> Dict[str, Any]:
    """Return one plain, JSON-serializable lineage record for a candidate sequence."""
    mutations = build_mutations(source_seq=source_seq, candidate_seq=candidate_seq)
    source_seq_id = comp_candidate_id(source_seq=source_seq, mutations=[])
    return {"candidate_id": comp_candidate_id(source_seq=source_seq, mutations=mutations),
            "parent_id": source_seq_id if parent_id is None else parent_id,
            "source_seq_id": source_seq_id,
            "mutations": mutations,
            "method": method,
            "objective_values": {str(k): float(v) for k, v in objective_values.items()},
            "seed": None if seed is None else int(seed),
            "constraints_digest": constraints_digest}


def trace_records(lineage=None, candidate_id=None) -> List[Dict[str, Any]]:
    """Return the root-first chain of records from the root ancestor to ``candidate_id``.

    The walk follows ``parent_id`` and stops at the first identifier no record carries (the
    root of the exported chain), or at a record that is its own parent (a candidate identical
    to its source has no mutation to walk back through). Identifiers are content hashes,
    so a record that appears twice (the same candidate from two runs) is indexed once, by
    its first occurrence.
    """
    records_by_id = {}
    for record in lineage:
        records_by_id.setdefault(record["candidate_id"], record)
    chain, seen, current = [], set(), candidate_id
    while current in records_by_id:
        if current in seen:
            raise RuntimeError(f"'lineage' is cyclic at candidate_id '{current}'; a candidate "
                               f"cannot be its own ancestor.")
        seen.add(current)
        record = records_by_id[current]
        chain.append(record)
        if record["parent_id"] == current:
            # A candidate identical to its source is its own root (no mutation to walk back).
            break
        current = record["parent_id"]
    chain.reverse()
    return chain

"""
This is a script for the frontend of the candidate-lineage record: the thin, plain-dict,
JSON-serializable record of how one designed candidate was made, shared by SeqMut and SeqOpt.

The record answers "where did this mutant come from?" without changing any return type: it is
opt-in, and it carries the candidate's content hash, its parent, the ordered parent-relative
mutations, the generating method, the objective values, the effective seed and a digest of the
applied design limits. It is the design-side companion of the run-level provenance record.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import aaanalysis.utils as ut
from ._backend.candidate_lineage import build_record, comp_constraints_digest


# I Helper Functions
def check_lineage_record(name: str = "lineage", val: Any = None) -> Dict[str, Any]:
    """Check one lineage record: a dict carrying every field of ``ut.LIST_CANDIDATE_LINEAGE``."""
    ut.check_dict(name=name, val=val, accept_none=False)
    missing = [field for field in ut.LIST_CANDIDATE_LINEAGE if field not in val]
    if len(missing) > 0:
        raise ValueError(f"'{name}' ({missing}) should carry every lineage field "
                         f"{list(ut.LIST_CANDIDATE_LINEAGE)}.")
    for field in ["candidate_id", "parent_id", "source_seq_id"]:
        ut.check_str(name=f"'{name}' {field}", val=val[field], accept_none=False)
    ut.check_list_like(name=f"'{name}' mutations", val=val["mutations"],
                       check_all_non_none=True)
    return {str(key): value for key, value in val.items()}


def check_lineage(name: str = "lineage", val: Any = None) -> List[Dict[str, Any]]:
    """Check a list of lineage records, as exported by a design round (or several)."""
    val = ut.check_list_like(name=name, val=val, accept_none=False)
    if len(val) == 0:
        raise ValueError(f"'{name}' ([]) should be a non-empty list of lineage records.")
    return [check_lineage_record(name=f"{name}[{i}]", val=record)
            for i, record in enumerate(val)]


def resolve_lineage(lineage: Union[bool, Dict[str, Any]] = False,
                    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Resolve the opt-in ``lineage`` argument into ``(enabled, parent record or None)``.

    ``False`` / ``True`` switch the record off / on with the source treated as the root of the
    chain; a record dict switches it on *and* names the parent this round descends from.
    """
    if isinstance(lineage, bool):
        return lineage, None
    if isinstance(lineage, dict):
        return True, check_lineage_record(name="lineage", val=lineage)
    raise ValueError(f"'lineage' ({type(lineage).__name__}) should be a bool, or the lineage "
                     f"record of the parent this design round starts from.")


def check_match_lineage_candidate_id(lineage: List[Dict[str, Any]],
                                     candidate_id: Any = None) -> None:
    """Check that ``candidate_id`` names a record of ``lineage``."""
    ut.check_str(name="candidate_id", val=candidate_id, accept_none=False)
    list_id = [record["candidate_id"] for record in lineage]
    if candidate_id not in list_id:
        raise ValueError(f"'candidate_id' ('{candidate_id}') should be the 'candidate_id' of "
                         f"one of the {len(list_id)} records in 'lineage'.")


def check_match_parent_source(parent: Optional[Dict[str, Any]],
                              list_source_seq: List[str]) -> None:
    """Check that a parent record is not shared by candidates of different source sequences."""
    if parent is None:
        return None
    n_source = len(set(list_source_seq))
    if n_source > 1:
        raise ValueError(f"'lineage' (a parent record) should be passed for candidates of one "
                         f"source sequence, but {n_source} were scored; a candidate descends "
                         f"from exactly one parent.")
    return None


# II Main Functions
def build_lineage(list_source_seq: List[str],
                  list_candidate_seq: List[str],
                  method: str,
                  list_objective_values: List[Dict[str, Any]],
                  seed: Optional[int] = None,
                  spec: Optional[Dict[str, Any]] = None,
                  parent: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return one lineage record per candidate, aligned to the rows of the returned table."""
    constraints_digest = comp_constraints_digest(spec=spec)
    parent_id = None if parent is None else parent["candidate_id"]
    return [build_record(source_seq=source_seq, candidate_seq=candidate_seq, method=method,
                         objective_values=objective_values, seed=seed,
                         constraints_digest=constraints_digest, parent_id=parent_id)
            for source_seq, candidate_seq, objective_values
            in zip(list_source_seq, list_candidate_seq, list_objective_values)]

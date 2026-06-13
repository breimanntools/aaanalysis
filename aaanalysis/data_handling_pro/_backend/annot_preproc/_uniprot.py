"""
This is a script for the backend of the AnnotationPreprocessor: fetching
UniProtKB entries from the public JSON endpoint and mapping their ``features``
array into the canonical per-residue ``df_annot`` schema.

Mapping rules:

- Bond features (Disulfide bond, Cross-link) expand to TWO single-residue
  endpoint rows sharing a ``bond_id``.
- Processing features (Signal, Propeptide, Transit peptide) contribute their
  span END as the cleavage P1 anchor (Schechter-Berger convention) — a single
  residue, no description parsing.
- ``Site`` is a grab-bag: only descriptions matching a cleavage pattern are
  routed to ``cleavage_site``; everything else is dropped (never blanket-dumped
  into PTM positives).
- ``Modified residue`` / ``Glycosylation`` are description-routed to the
  registry key (phospho / glyco_n / glyco_o / mod_res_other).
- Range features (non-bond, start != end) expand to one row per residue.
- The evidence allow-set filters by ANY of a feature's ECO codes; ``None``
  disables filtering.
- ``aa`` is taken from the fetched UniProt canonical sequence and is later
  verified against the user's target sequence at encode time.

Backend trusts the frontend for argument validation; it raises ``RuntimeError``
on network / response failures (derived invariants).
"""

import re
from typing import Dict, List, Optional

import pandas as pd
import requests

import aaanalysis.utils as ut
from .feature_registry import REGISTRY
from .._fetch import http_get_, run_in_order_

# I Helper Functions
UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"

# Raw UniProt JSON feature ``type`` strings → handling. JSON uses long names
# (not the TXT/GFF short codes MOD_RES / CARBOHYD / ...).
_TYPE_MOD_RES = "Modified residue"
_TYPE_CARBOHYD = "Glycosylation"
_TYPE_LIPID = "Lipidation"
_TYPE_DISULFID = "Disulfide bond"
_TYPE_CROSSLNK = "Cross-link"
_TYPE_SIGNAL = "Signal"
_TYPE_PROPEP = "Propeptide"
_TYPE_TRANSIT = "Transit peptide"
_TYPE_SITE = "Site"
_TYPE_BINDING = "Binding site"
_TYPE_ACT_SITE = "Active site"
_TYPE_DNA_BIND = "DNA binding"

_BOND_TYPES = {_TYPE_DISULFID: "disulfide", _TYPE_CROSSLNK: "crosslink"}
_PROCESSING_TYPES = {
    _TYPE_SIGNAL: "signal_cleavage",
    _TYPE_PROPEP: "propep_cleavage",
    _TYPE_TRANSIT: "transit_cleavage",
}
_FUNC_TYPES = {
    _TYPE_BINDING: "binding",
    _TYPE_ACT_SITE: "act_site",
    _TYPE_DNA_BIND: "dna_bind",
}

# SITE descriptions that denote a cleavage event (case-insensitive).
_CLEAVAGE_SITE_RE = re.compile(r"cleav", re.IGNORECASE)


def fetch_uniprot_json(accession: str, timeout: float = 30.0) -> dict:
    """GET the UniProtKB JSON record for ``accession``.

    Raises
    ------
    RuntimeError
        On any non-200 response or transport error.
    """
    url = UNIPROT_JSON_URL.format(acc=accession)
    try:
        resp = http_get_(url, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"UniProt request for '{accession}' failed: {e}") from e
    if resp.status_code != 200:
        raise RuntimeError(
            f"UniProt request for '{accession}' returned HTTP "
            f"{resp.status_code} (expected 200)"
        )
    return resp.json()


def _get_sequence(data: dict) -> str:
    """Extract the canonical sequence string from a UniProtKB JSON record."""
    seq = data.get("sequence", {}).get("value")
    if not seq:
        raise RuntimeError(
            "UniProt record is missing 'sequence.value' — cannot map "
            "positions to residues"
        )
    return seq


def _evidence_codes(feature: dict) -> List[str]:
    return [e.get("evidenceCode", "") for e in feature.get("evidences", [])]


def _passes_evidence(feature: dict, allowed: Optional[List[str]]) -> bool:
    """True if filtering is off, or any feature ECO code is in ``allowed``."""
    if allowed is None:
        return True
    return any(code in allowed for code in _evidence_codes(feature))


def _primary_evidence(feature: dict) -> str:
    """Representative ECO code stored in the schema (empty if none)."""
    codes = _evidence_codes(feature)
    return codes[0] if codes else ""


def _classify_mod_res(description: str) -> str:
    return "phospho" if description.lower().startswith("phospho") else "mod_res_other"


def _classify_glyco(description: str) -> str:
    return "glyco_n" if "n-linked" in description.lower() else "glyco_o"


def _location_bounds(feature: dict):
    """Return ``(start, end)`` 1-based ints, or ``(None, None)`` if unknown."""
    loc = feature.get("location", {})
    start = loc.get("start", {}).get("value")
    end = loc.get("end", {}).get("value")
    return start, end


def _feature_key(ftype: str, description: str) -> Optional[str]:
    """Map a raw UniProt feature to a registry key, or None to drop it."""
    if ftype == _TYPE_MOD_RES:
        return _classify_mod_res(description)
    if ftype == _TYPE_CARBOHYD:
        return _classify_glyco(description)
    if ftype == _TYPE_LIPID:
        return "lipid"
    if ftype in _FUNC_TYPES:
        return _FUNC_TYPES[ftype]
    if ftype == _TYPE_SITE:
        return "cleavage_site" if _CLEAVAGE_SITE_RE.search(description) else None
    return None


def _row(protein_id, pos, aa, feature_key, source, evidence, score, bond_id):
    """Positional row in ``ut.COLS_ANNOT`` order."""
    category = REGISTRY[feature_key]["category"]
    return [
        protein_id,
        pos,
        pos,
        aa,
        feature_key,
        category,
        source,
        evidence,
        score,
        bond_id,
    ]


# II Main Functions
def map_record_to_rows(
    accession: str,
    data: dict,
    allowed_features: Optional[List[str]],
    evidence_codes: Optional[List[str]],
) -> List[list]:
    """Map one UniProtKB JSON record to a list of positional ``df_annot`` rows.

    Parameters
    ----------
    accession : str
        The protein_id stored on every emitted row.
    data : dict
        Parsed UniProtKB JSON record.
    allowed_features : list of str or None
        Registry keys to keep; ``None`` keeps every built-in key.
    evidence_codes : list of str or None
        Evidence allow-set; ``None`` disables evidence filtering.
    """
    seq = _get_sequence(data)
    seq_len = len(seq)
    rows: List[list] = []
    bond_counter = 0

    def keep(key: str) -> bool:
        return allowed_features is None or key in allowed_features

    def aa_at(pos: int) -> str:
        # 1-based position into the canonical sequence; guarded by bounds.
        return seq[pos - 1] if 1 <= pos <= seq_len else ""

    for feature in data.get("features", []):
        ftype = feature.get("type", "")
        description = feature.get("description", "") or ""
        if not _passes_evidence(feature, evidence_codes):
            continue
        evidence = _primary_evidence(feature)
        start, end = _location_bounds(feature)
        if start is None or end is None:
            continue

        # Bond features → two endpoints + shared bond_id
        if ftype in _BOND_TYPES:
            key = _BOND_TYPES[ftype]
            if not keep(key):
                continue
            bond_counter += 1
            bond_id = f"{accession}_{key}_{bond_counter}"
            for pos in (start, end):
                rows.append(
                    _row(
                        accession,
                        pos,
                        aa_at(pos),
                        key,
                        "UniProt",
                        evidence,
                        1.0,
                        bond_id,
                    )
                )
            continue

        # Processing features → span END is the cleavage P1 anchor
        if ftype in _PROCESSING_TYPES:
            key = _PROCESSING_TYPES[ftype]
            if not keep(key):
                continue
            rows.append(
                _row(accession, end, aa_at(end), key, "UniProt", evidence, 1.0, None)
            )
            continue

        # Description-routed / direct features → one row per residue in span
        key = _feature_key(ftype, description)
        if key is None or not keep(key):
            continue
        for pos in range(start, end + 1):
            rows.append(
                _row(accession, pos, aa_at(pos), key, "UniProt", evidence, 1.0, None)
            )

    return rows


def _fetch_uniprot_one(
    entry: str,
    allowed_features: Optional[List[str]],
    evidence_codes: Optional[List[str]],
    timeout: float,
) -> List[list]:
    """Fetch + map one entry to its ``df_annot`` rows (network work only).

    No logging, so it is safe to run on a worker thread; the caller emits the
    verbose prints from the main thread in input order.
    """
    data = fetch_uniprot_json(entry, timeout=timeout)
    return map_record_to_rows(entry, data, allowed_features, evidence_codes)


def fetch_and_map(
    entries: List[str],
    allowed_features: Optional[List[str]],
    evidence_codes: Optional[List[str]],
    timeout: float = 30.0,
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch every entry and concatenate the mapped rows into a ``df_annot``.

    With ``max_workers`` greater than 1 the per-entry fetches run on a thread
    pool; rows are concatenated in input order, so the ``df_annot`` is identical
    to the sequential path regardless of worker count. The verbose prints are
    emitted from the main thread.

    Raises
    ------
    RuntimeError
        Propagated from :func:`fetch_uniprot_json` on network failure.
    """
    results = run_in_order_(
        lambda entry: _fetch_uniprot_one(entry, allowed_features,
                                         evidence_codes, timeout),
        entries, max_workers=max_workers)
    all_rows: List[list] = []
    for entry, rows in zip(entries, results):
        if verbose:
            ut.print_out(f"Fetching UniProt features for '{entry}'")
        all_rows.extend(rows)
    return pd.DataFrame(all_rows, columns=ut.COLS_ANNOT)

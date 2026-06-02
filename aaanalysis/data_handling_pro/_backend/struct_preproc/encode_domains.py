"""
This is a script for the backend of the StructurePreprocessor: per-feature
encoders that turn a parsed domain segmentation (list of domains, each
domain is a list of (start, end) 1-based inclusive segments) into ``(L, 1)``
per-residue numerical tensors normalized to ``[0, 1]`` per the recipes in
``feature_registry.NORMALIZATION_RECIPES``.

Four feature keys are exposed (v1.2):

  domain_boundary           (L, 1)   binary {0, 1}, N or C-term of any domain
  domain_relative_position  (L, 1)   fraction-of-way through residue's domain
                                     (0 = N-term of domain, 1 = C-term)
  domain_size               (L, 1)   total residues in residue's domain / 200
  n_domains_in_protein      (L, 1)   count of domains in protein / 10
                                     (broadcast per residue — constant within a protein)

Residues not assigned to any domain in the chopping string get NaN in all
four columns.
"""
from typing import List, Tuple

import numpy as np

from .feature_registry import normalize


# I Helper Functions
def _residue_to_domain_index(L: int,
                             domains: List[List[Tuple[int, int]]]) -> np.ndarray:
    """Build a (L,) int array mapping each residue (0-indexed) to its domain
    index, or -1 if unassigned.
    """
    out = np.full(L, -1, dtype=np.int64)
    for di, segments in enumerate(domains):
        for start_1, end_1 in segments:
            start_0 = max(0, start_1 - 1)
            end_0 = min(L, end_1)   # end_1 is inclusive 1-based → exclusive 0-based
            for i in range(start_0, end_0):
                if out[i] == -1:
                    out[i] = di
    return out


def _domain_extents(domains: List[List[Tuple[int, int]]]
                    ) -> List[Tuple[int, int, int]]:
    """For each domain: (min_residue_1based, max_residue_1based, total_size).

    Discontinuous domains: min is min-of-all-segments, max is max-of-all,
    size is the sum of segment lengths (not max-min+1).
    """
    out: List[Tuple[int, int, int]] = []
    for segments in domains:
        if not segments:
            out.append((0, 0, 0))
            continue
        mins = [s[0] for s in segments]
        maxs = [s[1] for s in segments]
        size = sum(e - s + 1 for s, e in segments)
        out.append((min(mins), max(maxs), size))
    return out


# II Main Functions
def encode_domain_boundary(L: int,
                           domains: List[List[Tuple[int, int]]]) -> np.ndarray:
    """Per-residue binary boundary mask as ``(L, 1)`` in ``{0, 1}`` (NaN if unassigned).

    A boundary residue is the first or last residue of any segment of any
    domain (covers domain N-term / C-term plus internal segment endpoints
    of discontinuous domains).
    """
    out = np.full((L, 1), np.nan, dtype=np.float64)
    domain_id = _residue_to_domain_index(L, domains)
    for i in range(L):
        if domain_id[i] == -1:
            continue
        out[i, 0] = 0.0
    for segments in domains:
        for start_1, end_1 in segments:
            for endpoint_1 in (start_1, end_1):
                idx = endpoint_1 - 1
                if 0 <= idx < L:
                    out[idx, 0] = 1.0
    return normalize("domain_boundary", out)


def encode_domain_relative_position(
    L: int, domains: List[List[Tuple[int, int]]]
) -> np.ndarray:
    """Per-residue fraction-of-way-through-domain as ``(L, 1)`` in ``[0, 1]``.

    For each residue assigned to a domain: 0.0 at the domain's first
    residue (smallest residue index across all its segments), 1.0 at the
    last, linear in-between. Domains of size 1 emit 0.5 (mid-point). NaN
    for unassigned residues.
    """
    out = np.full((L, 1), np.nan, dtype=np.float64)
    extents = _domain_extents(domains)
    domain_id = _residue_to_domain_index(L, domains)
    for i in range(L):
        di = int(domain_id[i])
        if di == -1:
            continue
        d_min, d_max, _size = extents[di]
        if d_max == d_min:
            out[i, 0] = 0.5
        else:
            out[i, 0] = (i + 1 - d_min) / (d_max - d_min)
    return normalize("domain_relative_position", out)


def encode_domain_size(L: int,
                       domains: List[List[Tuple[int, int]]]) -> np.ndarray:
    """Per-residue domain size as ``(L, 1)``: residues in this residue's domain.

    Recipe normalizes by /200 (clipped to [0, 1]); domains larger than
    200 residues saturate at 1.0. NaN for unassigned residues.
    """
    out = np.full((L, 1), np.nan, dtype=np.float64)
    extents = _domain_extents(domains)
    domain_id = _residue_to_domain_index(L, domains)
    for i in range(L):
        di = int(domain_id[i])
        if di == -1:
            continue
        out[i, 0] = float(extents[di][2])
    return normalize("domain_size", out)


def encode_n_domains_in_protein(L: int,
                                domains: List[List[Tuple[int, int]]]
                                ) -> np.ndarray:
    """Per-residue (broadcast) domain count as ``(L, 1)`` in ``[0, 1]``.

    Recipe normalizes by /10 (clipped); proteins with more than 10
    domains saturate. Unlike the other domain features, this is constant
    across all assigned residues of a protein (the per-protein domain
    count). Unassigned residues are NaN.
    """
    out = np.full((L, 1), np.nan, dtype=np.float64)
    n = float(len(domains))
    domain_id = _residue_to_domain_index(L, domains)
    for i in range(L):
        if domain_id[i] == -1:
            continue
        out[i, 0] = n
    return normalize("n_domains_in_protein", out)

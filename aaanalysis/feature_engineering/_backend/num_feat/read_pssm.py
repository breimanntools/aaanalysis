"""
This is a script for the backend of the NumericalFeature.from_pssm method.

Parses PSI-BLAST ASCII position-specific scoring matrices (``-out_ascii_pssm``) into
per-residue ``(L, 20)`` arrays whose columns follow ``ut.LIST_CANONICAL_AA`` order.
PSI-BLAST writes its columns in the ``ARNDCQEGHILKMFPSTWYV`` order, so every matrix is
explicitly permuted by the column header read from the file.
"""
import os
import numpy as np

import aaanalysis.utils as ut
from ..cpp.sequence_feature import get_composition_scales_

N_AA = 20


# I Helper Functions
def _is_header_line(tokens):
    """Column header: at least 20 single upper-case letters and nothing else."""
    return len(tokens) >= N_AA and all(len(t) == 1 and t.isalpha() and t.isupper() for t in tokens)


def _is_data_line(tokens):
    """Data row: '<position> <residue> <20 log-odds> [<20 percentages> <info> <weight>]'."""
    return len(tokens) >= 2 + N_AA and tokens[0].isdigit() and len(tokens[1]) == 1 and tokens[1].isalpha()


def _get_permutation(header=None, file=None):
    """Column indices mapping the file's header order onto ``ut.LIST_CANONICAL_AA`` order."""
    order = header[:N_AA]
    if sorted(order) != sorted(ut.LIST_CANONICAL_AA):
        raise ValueError(f"PSSM file '{file}' has an invalid column header {order}; "
                         f"it should list the 20 canonical amino acids "
                         f"(PSI-BLAST order: {''.join(ut.LIST_PSSM_AA_ORDER)}).")
    return np.array([order.index(aa) for aa in ut.LIST_CANONICAL_AA])


def _normalize(arr=None, values=None):
    """Map raw PSSM values onto [0, 1]: sigmoid for log-odds, /100 for percentages."""
    if values == "log_odds":
        return 1.0 / (1.0 + np.exp(-arr))
    return arr / 100.0


# II Main Functions
def read_pssm_file_(file=None):
    """Parse one PSI-BLAST ASCII PSSM file.

    Returns ``(residues, log_odds, frequencies)``: the query residue string, the ``(L, 20)``
    log-odds block and the ``(L, 20)`` weighted-observed-percentage block (``None`` if the
    file carries no percentage block), both permuted to ``ut.LIST_CANONICAL_AA`` order.
    """
    with open(file, "r") as f:
        lines = f.read().splitlines()
    perm = None
    residues, rows_lo, rows_freq = [], [], []
    for line in lines:
        tokens = line.split()
        if perm is None:
            if _is_header_line(tokens):
                perm = _get_permutation(header=tokens, file=file)
            continue
        if _is_data_line(tokens):
            residues.append(tokens[1].upper())
            rows_lo.append([float(x) for x in tokens[2:2 + N_AA]])
            if len(tokens) >= 2 + 2 * N_AA:
                rows_freq.append([float(x) for x in tokens[2 + N_AA:2 + 2 * N_AA]])
        elif residues:
            break  # First non-data line after the matrix ends it (K / Lambda footer follows)
    if perm is None or not residues:
        raise ValueError(f"PSSM file '{file}' could not be parsed: no PSI-BLAST column header "
                         f"or no matrix rows found (expected ASCII output of 'psiblast -out_ascii_pssm').")
    log_odds = np.asarray(rows_lo, dtype=np.float64)[:, perm]
    frequencies = None
    if len(rows_freq) == len(rows_lo):
        frequencies = np.asarray(rows_freq, dtype=np.float64)[:, perm]
    return "".join(residues), log_odds, frequencies


def load_pssm_(dict_source=None, values="log_odds", normalize=True):
    """Convert ``{entry: file path or (L, 20) array}`` into ``dict_num`` and per-entry residues.

    Arrays are taken as already in ``ut.LIST_CANONICAL_AA`` column order; their residues
    are unknown (``None``).
    """
    dict_num, dict_residues = {}, {}
    for entry, src in dict_source.items():
        if isinstance(src, (str, os.PathLike)):
            residues, log_odds, frequencies = read_pssm_file_(file=src)
            if values == "frequencies" and frequencies is None:
                raise ValueError(f"PSSM file '{src}' (entry '{entry}') has no weighted observed "
                                 f"percentage block; use values='log_odds'.")
            arr = log_odds if values == "log_odds" else frequencies
        else:
            residues = None
            arr = np.asarray(src, dtype=np.float64)
        if normalize:
            arr = _normalize(arr=arr, values=values)
        dict_num[entry] = arr
        dict_residues[entry] = residues
    return dict_num, dict_residues


def get_pssm_scales_(values="log_odds"):
    """20-column ``(df_scales, df_cat)`` naming the PSSM dimensions (``PSSM_<AA>``).

    Reuses the one-hot amino acid composition scale set: ``df_scales`` is the ``(20, 20)``
    identity (row values are unused in numerical mode) and ``df_cat`` groups each column by
    the physicochemical class of its amino acid.
    """
    df_scales, df_cat = get_composition_scales_(k=1)
    scale_ids = [f"{ut.STR_PSSM_SCALE_PREFIX}{aa}" for aa in ut.LIST_CANONICAL_AA]
    df_scales = df_scales.copy()
    df_scales.columns = scale_ids
    df_cat = df_cat.copy()
    str_value = "log-odds score" if values == "log_odds" else "weighted observed percentage"
    df_cat[ut.COL_SCALE_ID] = scale_ids
    df_cat[ut.COL_SCALE_NAME] = [f"PSSM {aa}" for aa in ut.LIST_CANONICAL_AA]
    df_cat[ut.COL_SCALE_DES] = [f"PSSM {str_value} for amino acid {aa}" for aa in ut.LIST_CANONICAL_AA]
    return df_scales, df_cat

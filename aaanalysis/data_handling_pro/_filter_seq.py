"""
This is a script for a wrapper function called filter_seq that provides a Python interface to the
redundancy-reduction algorithms CD-Hit and MMseqs2.
"""
from typing import Optional, List, Literal
import shutil
import os
import pandas as pd

from aaanalysis import utils as ut
from ._backend.cd_hit import run_cd_hit
from ._backend.mmseq2 import run_mmseqs2


# I Helper functions
def check_is_tool(name=None):
    """Check whether `name` is on PATH and marked as executable."""
    if not shutil.which(name):
        raise ValueError(f"{name} is not installed or not in the PATH.")


def check_match_identity_coverage(global_identity=True, coverage_short=0.0, coverage_long=0.0):
    """Check if identity and coverage match"""
    if not global_identity and coverage_short == coverage_long == 0:
        raise ValueError(f"If 'global_identity' is False, 'coverage_short' ({coverage_short}) "
                         f"or 'coverage_long' ({coverage_long}) should be >0.0")


def check_seq_len(df_seq=None, len_min=11):
    """ Check if the length of each sequence in the specified column is at least `len_min`"""
    mask_seq_len_is_fine = df_seq[ut.COL_SEQ].str.len() >= len_min
    list_seq_too_short = df_seq[~mask_seq_len_is_fine][ut.COL_ENTRY].to_list()
    if len(list_seq_too_short) > 0:
        raise ValueError(f"Minimum requiered length (n>={len_min}) for sequences in '{ut.COL_SEQ}'"
                         f"is not meet by the following entries: {list_seq_too_short}")


def check_seq_gaps(df_seq):
    """Check if sequences in the specified column contain gaps ('-')."""
    mask_has_gaps = df_seq[ut.COL_SEQ].str.contains("-")
    list_seq_with_gaps = df_seq[mask_has_gaps][ut.COL_ENTRY].to_list()
    if list_seq_with_gaps:
        raise ValueError(f"The following sequences in '{ut.COL_SEQ}' should not "
                         f"contain gaps ('-'): {list_seq_with_gaps}")


# II Main function
def filter_seq(df_seq: pd.DataFrame = None,
               method: Literal['cd-hit', 'mmseqs'] = "cd-hit",
               similarity_threshold: float = 0.9,
               word_size: Optional[int] = None,
               global_identity: bool = True,
               coverage_long: Optional[float] = None,
               coverage_short: Optional[float] = None,
               sort_clusters: bool = False,
               n_jobs: int = 1,
               verbose: bool = False
               ) -> pd.DataFrame:
    """
    Redundancy reduction of sequences using clustering-based algorithms.

    This functions performs redundancy reduction of sequences by clustering and selecting representative sequences using
    the CD-HIT ([Li06]_) or MMseqs2 ([Steinegger17]_) algorithms locally. It allows for adjustable filtering strictness:

    * Strict filtering results in smaller, more homogeneous clusters, suitable when high sequence similarity is required.
    * Non-strict filtering creates larger, more diverse clusters, enhancing sequence representation.

    CD-HIT and MMseq2 are standalone software tools, each requiring separate installation. CD-Hit is more
    resource-efficient and easier to install, while MMseq2 is a larger multi-purpose tool. Pairwise sequence similarities
    for the MMseq2 clustering results were computed using the Biopython :class:`Bio.Align.PairwiseAligner` class.

    Parameters
    ----------
    df_seq : pd.DataFrame, shape (n_samples, n>=1)
        DataFrame containing an ``entry`` and ''sequence'' column for unique identifiers and sequences.
        Sequence length must be at least 11 amino acids and the sequence should not contain any gaps ('-').
    method : {'cd-hit', 'mmseqs'}, default='cd-hit'
        Specifies the clustering algorithm to use:

        - ``cd-hit``: Efficiently clusters sequences, ideal for small- to medium-sized datasets.
        - ``mmseqs``: Advanced algorithm designed for large-scale sequence analysis, offering high accuracy.

    similarity_threshold : float, default=0.9
        Defines the minimum sequence identity [0.4-1.0] for clustering. Higher values increase strictness.
    word_size : int, optional
        The size of the 'word' (in CD-HIT, [2-5]) or 'k-mer' (in MMseqs, [5-7]) used for the initial screening step in clustering.
        Effect on strictness is dataset-dependent. If ``None``, optimized based on ``similarity_threshold`` (CD-Hit).
    global_identity : bool, default=True
        Whether to use global (True) or local (False) sequence identity for 'cd-hit' clustering. Global is stricter.
        80%-coverage is used for local 'cd-hit' clustering if not specified. MMseq2 uses only local alignments.
    coverage_long : float, optional
        Minimum percentage [0.1-1.0] of the longer sequence that must be included in the alignment.
        Higher values increase strictness.
    coverage_short : float, optional
        Minimum percentage [0.1-1.0] of the shorter sequence that must be included in the alignment.
        Higher values increase strictness.
    sort_clusters : bool, default=False
        If ``True``, sort clusters by the number of contained sequences.
    n_jobs : int, None, or -1, default=1
        Number of CPU cores used for multiprocessing. If ``-1`` or ``None``, the number is set to all available cores.
    verbose : bool, default=False
        If ``True``, verbose outputs are enabled.

    Returns
    -------
    df_clust : pd.DataFrame
        A DataFrame with clustering results.

    Notes
    -----
    * **CD-HIT** and **MMseqs2** use different approaches for clustering sequences:

      - **CD-HIT** sorts sequences by length and clusters them using global or local alignments against the longest sequence.
      - **MMseqs2** employs an index-based approach and optimized algorithms for faster and more sensitive data handling.

    * Parameter Comparison:
    +--------------------------+---------------------------------+----------------------------------------+
    | Parameter                | CD-HIT                          | MMseqs2                                |
    +==========================+=================================+========================================+
    | **similarity_threshold** | `-c` (sequence identity)        | `--min-seq-id` (minimum sequence id)   |
    +--------------------------+---------------------------------+----------------------------------------+
    | **word_size**            | `-n` (word length)              | `-k` (k-mer size, auto-optimized)      |
    +--------------------------+---------------------------------+----------------------------------------+
    | **coverage_Long**        | `-aL` (coverage of longer seq)  | `--cov-mode 0 -c` (bidirectional)      |
    +--------------------------+---------------------------------+----------------------------------------+
    | **coverage_short**       | `-aS` (coverage of shorter seq) | `--cov-mode 1 -c` (target coverage)    |
    +--------------------------+---------------------------------+----------------------------------------+

    See Also
    --------
    * `CD-HIT Documentation <https://github.com/weizhongli/cdhit/wiki>`_.
    * MMseqs2 `GitHub ReadMe <https://github.com/soedinglab/MMseqs2>`_
      and `GitHub Wiki <https://github.com/soedinglab/mmseqs2/wiki>`_.
    * Comparison of CD-Hit and MMseqs2 parameters under
      `Frequently Asked Questions <https://github.com/soedinglab/mmseqs2/wiki#how-do-parameters-of-cd-hit-relate-to-mmseqs2>`_.

    Warnings
    --------
    * This function requires `biopython`, which is automatically installed via `pip install aaanalysis[pro]`.
    * CD-HIT and MMseq2 must be installed separately.
    * CD-HIT is not available for Windows.

    Examples
    --------
    .. include:: examples/filter_seq.rst
    """
    # Check input
    ut.check_df(name="df_seq", df=df_seq,
                cols_requiered=[ut.COL_ENTRY, ut.COL_SEQ],
                cols_nan_check=[ut.COL_ENTRY, ut.COL_SEQ])
    check_seq_len(df_seq=df_seq, len_min=11)
    check_seq_gaps(df_seq=df_seq)
    ut.check_str_options(name="method", val=method, accept_none=False,
                         list_str_options=["cd-hit", "mmseqs"])
    check_is_tool(name=method)
    ut.check_number_range(name="similarity_threshold", val=similarity_threshold,
                          min_val=0.4, max_val=1, accept_none=False, just_int=False)
    ut.check_number_range(name="word_size", val=word_size,
                          min_val=2 if method == "cd-hit" else 5,
                          max_val=5 if method == "cd-hit" else 7,
                          accept_none=True, just_int=True,
                          str_add=f"For the '{method}' method.")
    ut.check_bool(name="global_identity", val=global_identity)
    ut.check_number_range(name="coverage_long", val=coverage_long, min_val=0.1, max_val=1,
                          accept_none=True, just_int=False)
    ut.check_number_range(name="coverage_short", val=coverage_short, min_val=0.1, max_val=1,
                          accept_none=True, just_int=False)
    n_jobs = ut.check_n_jobs(n_jobs=n_jobs if n_jobs is not None else -1)
    ut.check_bool(name="sort_clusters", val=sort_clusters)
    ut.check_bool(name="verbose", val=sort_clusters)
    check_match_identity_coverage(global_identity=global_identity,
                                  coverage_short=coverage_short,
                                  coverage_long=coverage_long)
    # Run filtering
    args = dict(similarity_threshold=similarity_threshold, word_size=word_size,
                coverage_long=coverage_long, coverage_short=coverage_short,
                n_jobs=n_jobs, sort_clusters=sort_clusters, verbose=verbose)
    if method == "cd-hit":
        df_clust = run_cd_hit(df_seq=df_seq, global_identity=global_identity, **args)
    else:
        df_clust = run_mmseqs2(df_seq=df_seq, **args)
    return df_clust

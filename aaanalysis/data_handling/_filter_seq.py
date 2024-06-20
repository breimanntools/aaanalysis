"""
This is a script for a wrapper function called filter_seq that provides an Python interface to the
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
def check_valid_method(name=None):
    """Check if method name is valid"""
    list_valid_methods = ['cd-hit', 'mmseqs']
    if name not in list_valid_methods:
        raise ValueError(f"'method' ('{name}') should be one of following: {list_valid_methods}")


def check_is_tool(name=None):
    """Check whether `name` is on PATH and marked as executable."""
    if not shutil.which(name):
        raise ValueError(f"{name} is not installed or not in the PATH.")


def check_match_identity_coverage(global_identity=True, coverage_short=0.0, coverage_long=0.0):
    """Check if identity and coverage match"""
    if not global_identity and coverage_short == coverage_long == 0:
        raise ValueError(f"If 'global_identity' is False, 'coverage_short' ({coverage_short}) "
                         f"or 'coverage_long' ({coverage_long}) should be >0.0")


# II Main function
# TODO examples, tests
def filter_seq(df_seq: pd.DataFrame = None,
               method: Literal['cd-hit', 'mmseqs'] = "cd-hit",
               similarity_threshold: float = 0.9,
               word_size: Optional[int] = None,
               global_identity: bool = True,
               coverage_long: float = None,
               coverage_short: float = None,
               n_jobs: int = 1,
               sort_clusters: bool = False,
               verbose: bool = False
               ) -> pd.DataFrame:
    """
    UNDER CONSTRUCTION: Redundancy reduction of sequences using clustering-based algorithms.

    This functions performs redundancy reduction of sequences by clustering and selecting representative sequences using
    the CD-HIT ([Li06]_) or MMseqs2 ([Steinegger17]_) algorithms locally. It allows for adjustable filtering strictness:

    * Strict filtering results in smaller, more homogeneous clusters, suitable when high sequence similarity is required.
    * Non-strict filtering creates larger, more diverse clusters, enhancing sequence representation.

    CD-Hit and MMseq2 are standalone software tools, each requiring separate installation. CD-Hit is more
    resource-efficient and easier to install, while MMseq2 is a larger multi-purpose tool. Pairwise sequence similarities
    for the MMseq2 clustering results were computed using the Biopython :class:`Bio.Align.PairwiseAligner` class.

    Parameters
    ----------
    df_seq : pd.DataFrame, shape (n_samples, n>=1)
        DataFrame containing an ``entry`` and ''sequence'' column for unique identifiers and sequences.
    method : {'cd-hit', 'mmseqs'}, default='cd-hit'
        Specifies the clustering algorithm to use:

        - ``cd-hit``: Efficiently clusters sequences, ideal for small- to medium-sized datasets.
        - ``mmseqs``: Advanced algorithm designed for large-scale sequence analysis, offering high accuracy.

    similarity_threshold : float, default=0.9
        Defines the minimum sequence identity [0.0-1.0] for clustering. Higher values increase strictness.
    word_size : int, optional
        The size (>=2) of the 'word' (in CD-Hit) or 'k-mer' (in MMseqs) used for the initial screening step in clustering.
        Effect on strictness is dataset-dependent. If ``None``, optimized based on ``similarity_threshold`` (CD-Hit).
    global_identity : bool, default=True
        Whether to use global (True) or local (False) sequence identity for clustering. Global is stricter.
        Only relevant for 'cd-hit' method. MMseq2 uses only local alignments.
    coverage_long : float, optional
        Minimum percentage [0.0-1.0] of the longer sequence that must be included in the alignment.
        Higher values increase strictness.
    coverage_short : float, optional
        Minimum percentage [0.0-1.0] of the shorter sequence that must be included in the alignment.
        Higher values increase strictness.
    n_jobs : int, default=1
        Number of CPU threads for processing.
    sort_clusters : bool, default=False
        If ``True``, sort clusters by the number of contained sequences.
    verbose : bool, default=False
       If ``True``, verbose outputs are enabled.

    Returns
    -------
    Optional[pd.DataFrame]
        A DataFrame with clustering results if 'file_output' is not specified, otherwise None.

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

    Examples
    --------

    """
    # Check input
    ut.check_df(name="df_seq", df=df_seq,
                cols_requiered=[ut.COL_ENTRY, ut.COL_SEQ],
                cols_nan_check=[ut.COL_ENTRY, ut.COL_SEQ])
    ut.check_str(name="method", val=method, accept_none=False)
    check_valid_method(name=method)
    check_is_tool(name=method)
    ut.check_number_range(name="similarity_threshold", val=similarity_threshold,
                          min_val=0, max_val=1, accept_none=False, just_int=False)
    ut.check_number_range(name="word_size", val=word_size, min_val=2, accept_none=True, just_int=True)
    ut.check_bool(name="global_identity", val=global_identity)
    ut.check_number_range(name="coverage_long", val=coverage_long, min_val=0, max_val=1,
                          accept_none=True, just_int=False)
    ut.check_number_range(name="coverage_short", val=coverage_short, min_val=0, max_val=1,
                          accept_none=True, just_int=False)
    ut.check_number_range(name="n_jobs", val=n_jobs, min_val=1, accept_none=False, just_int=True)
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
        df = run_cd_hit(df_seq=df_seq, global_identity=global_identity, **args)
    else:
        df = run_mmseqs2(df_seq=df_seq, **args)
    return df

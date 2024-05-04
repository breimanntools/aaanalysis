"""
This is a script for a wrapper function called filter_seq that provides an Python interface to the
redundancy-reduction algorithms CD-Hit and MMSeq2.
"""
from typing import Optional, List, Literal
import shutil
import os
import pandas as pd

from aaanalysis import utils as ut
from .backend.cd_hit import run_cd_hit, get_df_clust_from_cd_hit
from .backend.mmseq2 import run_mmseq2


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
# TODO test, adjust, finish
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

    This functions performs redundancy reduction of sequences by clustering and selecting representative sequences
    using the CD-HIT or MMSeq2 algorithms. It allows for adjustable filtering strictness:

    * Strict filtering results in smaller, more homogeneous clusters, suitable for analyses requiring high sequence similarity.
    * Non-strict filtering creates larger, more diverse clusters, enhancing sequence representation.

    Parameters
    ----------
    df_seq : pd.DataFrame, shape (n_samples, n>=1)
        DataFrame containing an ``entry`` and ''sequence'' column for unique identifiers and sequences.
    method : str
        Specifies the clustering algorithm to use ('cd-hit' or 'mmseqs').
    similarity_threshold : float, default=0.9
        Defines the minimum sequence identity for clustering. Higher values increase strictness.
    word_size : int, optional
        The size (>=2) of the 'word' (in CD-Hit) or 'k-mer' (in MMseqs) used for the initial screening step in clustering.
        Effect on strictness is dataset-dependent. If None, Optimized based on ``similarity_threshold``.
    global_identity : bool, default=True
        Whether to use global (True) or local (False) sequence identity for clustering. Global is stricter.
        Only relevant for 'cd-hit' method. MMseq uses only local alignments.
    coverage_long : float, optional
        Minimum percentage [0.0-1.0] of the longer sequence that must be included in the alignment.
        Higher values increase strictness.
    coverage_short : float, optional
        Minimum percentage [0.0-1.0] of the shorter sequence that must be included in the alignment.
        Higher values increase strictness.
    n_jobs : int, default=1
        Number of CPU threads for processing.
    sort_clusters : bool, default=False
        If True, sort clusters by the number of contained sequences.
    verbose : bool, default=False
        If True, enable detailed output.

    Returns
    -------
    Optional[pd.DataFrame]
        A DataFrame with clustering results if 'file_output' is not specified, otherwise None.

    Notes
    -----
    * **CD-HIT** and **MMSeqs2** use different methods for clustering sequences:

      - **CD-HIT** sorts sequences by length and clusters them using global or local alignments against the longest sequence.
      - **MMSeqs2** employs an index-based approach and optimized algorithms for faster and more sensitive data handling.

    * While **CD-HIT** is quick and efficient for small to medium-sized datasets, **MMSeqs2** offers
      higher accuracy and is suitable for any dataset size.

    * Parameter Comparison:
    +---------------------+---------------------------------+----------------------------------------+
    | Parameter           | CD-HIT                          | MMSeqs2                                |
    +=====================+=================================+========================================+
    | Similarity Threshold| `-c` (sequence identity)        | `--min-seq-id` (minimum sequence id)   |
    +---------------------+---------------------------------+----------------------------------------+
    | Word Size           | `-n` (word length)              | `-k` (k-mer size, auto-optimized)      |
    +---------------------+---------------------------------+----------------------------------------+
    | Coverage Long       | `-aL` (coverage of longer seq)  | `--cov-mode 0 -c` (bidirectional)      |
    +---------------------+---------------------------------+----------------------------------------+
    | Coverage Short      | `-aS` (coverage of shorter seq) | `--cov-mode 1 -c` (target coverage)    |
    +---------------------+---------------------------------+----------------------------------------+

    See Also
    --------
    * `CD-HIT Documentation <https://github.com/weizhongli/cdhit/wiki>`_
    * MMSeq2 `GitHub Readme <https://github.com/soedinglab/MMseqs2>`_
      and `GitHub Wiki <https://github.com/soedinglab/mmseqs2/wiki>`_

    Examples
    --------

    """
    # Check input
    ut.check_df(name="df_seq", df=df_seq, cols_requiered=[ut.COL_ENTRY, ut.COL_SEQ],
                cols_nan_check=[ut.COL_ENTRY, ut.COL_SEQ])
    check_valid_method(name=method)
    check_is_tool(name=method)
    check_match_identity_coverage(global_identity=global_identity,
                                  coverage_short=coverage_short,
                                  coverage_long=coverage_long)
    ut.check_number_range(name="word_size", val=word_size, min_val=2, just_int=True, accept_none=True)

    # Run filtering
    args = dict(similarity_threshold=similarity_threshold, word_size=word_size,
                coverage_long=coverage_long, coverage_short=coverage_short,
                n_jobs=n_jobs, sort_clusters=sort_clusters, verbose=verbose)
    if method == "cd-hit":
        df = run_cd_hit(df_seq=df_seq, global_identity=global_identity, **args)
    else:
        df = run_mmseq2(df_seq=df_seq, **args)
    return df

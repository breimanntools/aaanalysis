"""
This is a script for the frontend of the SeqMut class for CPP-guided sequence mutation and
ΔCPP analysis.
"""
from typing import Optional, List, Union
import pandas as pd

import aaanalysis.utils as ut
from aaanalysis.feature_engineering._sequence_feature import SequenceFeature
from ._backend.seqmut.seqmut import (build_scan_plan, comp_delta_x, comp_scan_scores,
                                     build_scan_output, eval_disruptive, classify_region)


# I Helper Functions
def check_match_df_seq_pos_based(df_seq=None) -> None:
    """Check that df_seq is in the position-based format (sequence + tmd_start + tmd_stop)."""
    ut.check_df_seq(df_seq=df_seq)
    missing = [c for c in ut.COLS_SEQ_POS if c not in df_seq.columns]
    if len(missing) > 0:
        raise ValueError(f"'df_seq' should be in the position-based format with columns "
                         f"{ut.COLS_SEQ_POS}; missing: {missing}. SeqMut needs full sequences "
                         f"and TMD coordinates to mutate residues and recompute parts.")


def check_match_mutations_df_seq(mutations=None, df_seq=None):
    """Check the df_mut(entry, pos, to_aa) table against df_seq and derive from_aa."""
    cols_required = [ut.COL_ENTRY, ut.COL_POS, ut.COL_TO_AA]
    ut.check_df(df=mutations, name="mutations", cols_required=cols_required)
    seq_by_entry = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ]))
    list_from = []
    for entry, pos, to_aa in zip(mutations[ut.COL_ENTRY], mutations[ut.COL_POS],
                                 mutations[ut.COL_TO_AA]):
        if entry not in seq_by_entry:
            raise ValueError(f"'mutations' entry ('{entry}') is not in 'df_seq'. Available entries: "
                             f"{ut.preview_options(seq_by_entry)}.")
        seq = seq_by_entry[entry]
        pos = int(pos)
        if not 1 <= pos <= len(seq):
            raise ValueError(f"'pos' ({pos}) for entry '{entry}' should be in [1, {len(seq)}].")
        if to_aa not in ut.LIST_CANONICAL_AA:
            raise ValueError(f"'to_aa' ({to_aa}) should be a canonical amino acid.")
        list_from.append(seq[pos - 1])
    return list_from


def check_region(region=None):
    """Check the region argument (None, a part name, or a list of 1-based positions)."""
    if region is None or isinstance(region, str):
        if isinstance(region, str) and region.lower() not in ut.COLS_SEQ_PARTS:
            raise ValueError(f"'region' ({region}) should be one of {ut.COLS_SEQ_PARTS} "
                             f"or a list of 1-based positions.")
        return region
    region = ut.check_list_like(name="region", val=region)
    for p in region:
        ut.check_number_range(name="region position", val=p, min_val=1, just_int=True)
    return region


def check_to_aa_set(to_aa=None):
    """Check the substitution alphabet; default to all canonical amino acids."""
    if to_aa is None:
        return list(ut.LIST_CANONICAL_AA)
    to_aa = ut.check_list_like(name="to_aa", val=to_aa, accept_str=True)
    wrong = [aa for aa in to_aa if aa not in ut.LIST_CANONICAL_AA]
    if len(wrong) > 0:
        raise ValueError(f"'to_aa' ({wrong}) should be canonical amino acids.")
    return to_aa


def get_weight_vec(df_feat=None, weight=None):
    """Return the per-feature weight vector for the shift score, or None."""
    if weight is None:
        return None
    if weight not in ut.LIST_SHIFT_WEIGHTS:
        raise ValueError(f"'weight' ({weight}) should be one of {ut.LIST_SHIFT_WEIGHTS} or None.")
    if weight not in df_feat.columns:
        raise ValueError(f"'weight' ({weight}) column is not in 'df_feat'.")
    return df_feat[weight].to_numpy(dtype=float)


# II Main Functions
class SeqMut:
    """
    Sequence Mutator (SeqMut) class for CPP-guided sequence mutation and ΔCPP analysis
    [Breimann24a]_.

    ``SeqMut`` is the **CPP-aware** counterpart of :class:`AAMut`: it applies point mutations
    to protein sequences and measures the deterministic, model-free change they induce in a set
    of CPP features (``ΔCPP``). It inverts the CPP prediction direction — instead of asking
    *what distinguishes two groups*, it asks *how a mutation moves a sequence's feature profile*
    — supporting residue/region mutation, exhaustive ΔCPP scanning, and target-shift suggestion.

    .. versionadded:: 1.0.0

    """
    def __init__(self,
                 verbose: bool = False,
                 df_scales: Optional[pd.DataFrame] = None,
                 ):
        """
        Parameters
        ----------
        verbose : bool, default=False
            If ``True``, verbose outputs are enabled.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of amino acid scales (index = canonical amino acids, columns = scale ids).
            Default from :func:`load_scales`.

        See Also
        --------
        * :class:`AAMut` for the residue-level, CPP-agnostic substitution analysis.
        * :class:`CPP` whose ``df_feat`` defines the features mutated against.
        """
        self._verbose = ut.check_verbose(verbose)
        if df_scales is None:
            df_scales = ut.load_default_scales()
        self.df_scales = df_scales
        self._sf = SequenceFeature(verbose=False)

    # Helper methods
    def _delta_table(self, df_plan=None, df_seq=None, df_feat=None, jmd_n_len=10, jmd_c_len=10,
                     weight=None):
        """Run the ΔCPP engine for a mutation plan and return the scored scan output."""
        features = list(df_feat[ut.COL_FEATURE])
        mean_dif = df_feat[ut.COL_MEAN_DIF].to_numpy(dtype=float)
        weight_vec = get_weight_vec(df_feat=df_feat, weight=weight)
        dX = comp_delta_x(df_plan=df_plan, df_seq=df_seq, features=features,
                          df_scales=self.df_scales, sf=self._sf,
                          jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        delta_cpp, shift_score = comp_scan_scores(dX=dX, mean_dif=mean_dif, weight_vec=weight_vec)
        return build_scan_output(df_plan=df_plan, delta_cpp=delta_cpp, shift_score=shift_score)

    # Main methods
    def mutate(self,
               df_seq: pd.DataFrame,
               mutations: pd.DataFrame,
               df_feat: Optional[pd.DataFrame] = None,
               jmd_n_len: int = 10,
               jmd_c_len: int = 10,
               ) -> pd.DataFrame:
        """
        Apply specific point mutations to sequences and (optionally) measure their ΔCPP.

        Each row of ``mutations`` edits one residue of its ``entry``'s sequence; the mutated
        sequence and a human-readable ``mutation`` label are always returned, and the
        feature-space change is added when a ``df_feat`` is supplied.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers, in the
            **position-based** format (``sequence``, ``tmd_start``, ``tmd_stop``). See
            :meth:`SequenceFeature.get_df_parts` for the full ``df_seq`` format specification.
        mutations : pd.DataFrame, shape (n_mutations, >=3)
            Tidy mutation table with columns ``entry``, ``pos`` (1-based position in the full
            sequence), and ``to_aa`` (target amino acid). ``from_aa`` is derived and checked.
        df_feat : pd.DataFrame, optional
            CPP feature set (output of :meth:`CPP.run`). If given, the per-mutation ΔCPP
            (``delta_cpp``) and ``shift_score`` toward the test-class profile are added.
        jmd_n_len : int, default=10
            Length of JMD-N in number of amino acids.
        jmd_c_len : int, default=10
            Length of JMD-C in number of amino acids.

        Returns
        -------
        df_mut : pd.DataFrame, shape (n_mutations, n_info)
            The ``mutations`` table augmented with ``from_aa``, ``mutation`` (``"<from><pos><to>"``),
            ``sequence_mut`` (the mutated sequence), and — when ``df_feat`` is given — ``delta_cpp``
            and ``shift_score``.

        Examples
        --------
        .. include:: examples/seqmut_mutate.rst
        """
        # Validate
        check_match_df_seq_pos_based(df_seq=df_seq)
        list_from = check_match_mutations_df_seq(mutations=mutations, df_seq=df_seq)
        ut.check_number_range(name="jmd_n_len", val=jmd_n_len, min_val=0, just_int=True)
        ut.check_number_range(name="jmd_c_len", val=jmd_c_len, min_val=0, just_int=True)
        # Build mutated sequences + history table
        seq_by_entry = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ]))
        tmd_start = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_TMD_START]))
        tmd_stop = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_TMD_STOP]))
        df_mut = mutations.reset_index(drop=True).copy()
        df_mut[ut.COL_FROM_AA] = list_from
        seq_mut, mut_label = [], []
        for entry, pos, from_aa, to_aa in zip(df_mut[ut.COL_ENTRY], df_mut[ut.COL_POS],
                                              df_mut[ut.COL_FROM_AA], df_mut[ut.COL_TO_AA]):
            pos = int(pos)
            seq = seq_by_entry[entry]
            seq_mut.append(seq[:pos - 1] + to_aa + seq[pos:])
            mut_label.append(f"{from_aa}{pos}{to_aa}")
        df_mut[ut.COL_MUTATION] = mut_label
        df_mut[ut.COL_SEQ_MUT] = seq_mut
        if df_feat is not None:
            df_feat = ut.check_df_feat(df_feat=df_feat)
            df_plan = df_mut[[ut.COL_ENTRY, ut.COL_POS, ut.COL_FROM_AA, ut.COL_TO_AA]].copy()
            df_plan[ut.COL_TMD_START] = [tmd_start[e] for e in df_plan[ut.COL_ENTRY]]
            df_plan[ut.COL_TMD_STOP] = [tmd_stop[e] for e in df_plan[ut.COL_ENTRY]]
            df_plan[ut.COL_REGION] = [classify_region(pos=int(p), tmd_start=int(ts), tmd_stop=int(te))
                                      for p, ts, te in zip(df_plan[ut.COL_POS],
                                                           df_plan[ut.COL_TMD_START],
                                                           df_plan[ut.COL_TMD_STOP])]
            df_scored = self._delta_table(df_plan=df_plan, df_seq=df_seq, df_feat=df_feat,
                                          jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
            df_scored = df_scored.set_index(ut.COL_MUTATION)
            df_mut[ut.COL_DELTA_CPP] = df_scored.loc[df_mut[ut.COL_MUTATION], ut.COL_DELTA_CPP].to_numpy()
            df_mut[ut.COL_SHIFT_SCORE] = df_scored.loc[df_mut[ut.COL_MUTATION], ut.COL_SHIFT_SCORE].to_numpy()
        return df_mut

    def scan(self,
             df_seq: pd.DataFrame,
             df_feat: pd.DataFrame,
             region: Optional[Union[str, List[int]]] = None,
             to_aa: Optional[List[str]] = None,
             jmd_n_len: int = 10,
             jmd_c_len: int = 10,
             ) -> pd.DataFrame:
        """
        Run an exhaustive single-position mutational scan and rank mutations by |ΔCPP|.

        For every scannable position and every substitution, the change in the CPP feature
        vector is measured and aggregated into ``delta_cpp`` (the L1 magnitude ``Sum|ΔX|``).

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers, in the
            **position-based** format (``sequence``, ``tmd_start``, ``tmd_stop``). See
            :meth:`SequenceFeature.get_df_parts` for the full ``df_seq`` format specification.
        df_feat : pd.DataFrame
            CPP feature set (output of :meth:`CPP.run`) defining which features ΔCPP is measured over.
        region : str or list of int, optional
            Restrict the scan: ``None`` covers the full JMD-N + TMD + JMD-C span, a part name
            (``'jmd_n'`` / ``'tmd'`` / ``'jmd_c'``) restricts to that part, and a list restricts
            to those 1-based positions.
        to_aa : list of str, optional
            Substitution alphabet. If ``None``, every canonical amino acid (except the wild-type
            residue) is tried at each position.
        jmd_n_len : int, default=10
            Length of JMD-N in number of amino acids.
        jmd_c_len : int, default=10
            Length of JMD-C in number of amino acids.

        Returns
        -------
        df_scan : pd.DataFrame, shape (n_mutations, 8)
            Tidy mutation landscape with columns ``entry``, ``pos``, ``from_aa``, ``to_aa``,
            ``mutation``, ``region``, ``delta_cpp``, and ``shift_score``, sorted by descending
            ``delta_cpp``.

        Examples
        --------
        .. include:: examples/seqmut_scan.rst
        """
        # Validate
        check_match_df_seq_pos_based(df_seq=df_seq)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        region = check_region(region=region)
        to_aa = check_to_aa_set(to_aa=to_aa)
        ut.check_number_range(name="jmd_n_len", val=jmd_n_len, min_val=0, just_int=True)
        ut.check_number_range(name="jmd_c_len", val=jmd_c_len, min_val=0, just_int=True)
        # Scan
        df_plan = build_scan_plan(df_seq=df_seq, region=region, to_aa=to_aa,
                                  jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        if len(df_plan) == 0:
            raise ValueError("No scannable positions for the given 'region'.")
        df_scan = self._delta_table(df_plan=df_plan, df_seq=df_seq, df_feat=df_feat,
                                    jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        if self._verbose:
            ut.print_out(f"SeqMut scanned {len(df_scan)} mutations over {len(df_seq)} sequence(s).")
        return df_scan

    def suggest(self,
                df_seq: pd.DataFrame,
                df_feat: pd.DataFrame,
                n: int = 10,
                region: Optional[Union[str, List[int]]] = None,
                to_aa: Optional[List[str]] = None,
                weight: Optional[str] = None,
                jmd_n_len: int = 10,
                jmd_c_len: int = 10,
                ) -> pd.DataFrame:
        """
        Suggest the top mutations that shift a sequence toward the test-class CPP profile.

        Mutations are ranked by ``shift_score`` = ``Sum sign(mean_dif) * ΔX`` (optionally weighted
        by a ``df_feat`` column), i.e. how strongly they move features in the direction by which
        the test class differs from the reference class. This is the single-objective design
        primitive; multi-objective / library generation is out of scope (issues #57-#60).

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers, in the
            **position-based** format (``sequence``, ``tmd_start``, ``tmd_stop``). See
            :meth:`SequenceFeature.get_df_parts` for the full ``df_seq`` format specification.
        df_feat : pd.DataFrame
            CPP feature set (output of :meth:`CPP.run`); its signed ``mean_dif`` defines the target direction.
        n : int, default=10
            Number of top mutations to return.
        region : str or list of int, optional
            Restrict the scan (see :meth:`SeqMut.scan`).
        to_aa : list of str, optional
            Substitution alphabet (see :meth:`SeqMut.scan`).
        weight : str, optional
            Optionally weight the shift score by a ``df_feat`` column (``'feat_importance'`` or
            ``'abs_auc'``). If ``None``, all features contribute equally.
        jmd_n_len : int, default=10
            Length of JMD-N in number of amino acids.
        jmd_c_len : int, default=10
            Length of JMD-C in number of amino acids.

        Returns
        -------
        df_suggest : pd.DataFrame, shape (n, 8)
            The top-``n`` mutations sorted by descending ``shift_score``.

        Examples
        --------
        .. include:: examples/seqmut_suggest.rst
        """
        # Validate
        check_match_df_seq_pos_based(df_seq=df_seq)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        region = check_region(region=region)
        to_aa = check_to_aa_set(to_aa=to_aa)
        ut.check_number_range(name="n", val=n, min_val=1, just_int=True)
        get_weight_vec(df_feat=df_feat, weight=weight)  # validate weight early
        ut.check_number_range(name="jmd_n_len", val=jmd_n_len, min_val=0, just_int=True)
        ut.check_number_range(name="jmd_c_len", val=jmd_c_len, min_val=0, just_int=True)
        # Scan + rank by shift toward the test-class profile
        df_plan = build_scan_plan(df_seq=df_seq, region=region, to_aa=to_aa,
                                  jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        if len(df_plan) == 0:
            raise ValueError("No scannable positions for the given 'region'.")
        df_scan = self._delta_table(df_plan=df_plan, df_seq=df_seq, df_feat=df_feat,
                                    jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, weight=weight)
        df_suggest = df_scan.sort_values(ut.COL_SHIFT_SCORE, ascending=False).head(n).reset_index(drop=True)
        return df_suggest

    def eval(self,
             df_scan: pd.DataFrame,
             th: Optional[float] = None,
             ) -> pd.DataFrame:
        """
        Evaluate a mutational scan: tag mutations stable/disruptive and summarize per region.

        A mutation is disruptive when its ``|ΔCPP|`` reaches the threshold; the per-region
        disruptive rate shows where in the sequence (JMD-N / TMD / JMD-C) substitutions move
        the CPP profile most.

        Parameters
        ----------
        df_scan : pd.DataFrame
            Mutational landscape produced by :meth:`SeqMut.scan`.
        th : float, optional
            ``|ΔCPP|`` threshold above which a mutation is disruptive. If ``None``, the upper
            tertile (2/3 quantile) of the observed ``delta_cpp`` distribution is used.

        Returns
        -------
        df_eval : pd.DataFrame, shape (n_entry_region, 6)
            One row per ``entry`` x ``region`` with ``n_mut``, ``n_disruptive``,
            ``frac_disruptive``, and ``mean_delta_cpp``.

        Examples
        --------
        .. include:: examples/seqmut_eval.rst
        """
        # Validate
        ut.check_df(df=df_scan, name="df_scan", cols_required=ut.COLS_SEQMUT_SCAN)
        if th is not None:
            ut.check_number_range(name="th", val=th, min_val=0, accept_none=False, just_int=False)
        # Evaluate
        df_eval, th_used = eval_disruptive(df_scan=df_scan, th=th)
        if self._verbose:
            ut.print_out(f"SeqMut.eval used disruptive threshold |delta_cpp| >= {th_used:.4f}.")
        return df_eval

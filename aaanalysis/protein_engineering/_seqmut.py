"""
This is a script for the frontend of the SeqMut class for CPP-guided sequence mutation and
ΔCPP / model prediction-shift analysis.
"""
from typing import Optional, List, Union, Any
import pandas as pd

import aaanalysis.utils as ut
from aaanalysis.feature_engineering._sequence_feature import SequenceFeature
from ._backend.seqmut.seqmut import (build_scan_plan, comp_feature_matrices, comp_scan_scores,
                                     comp_pred_scores, comp_seq_scores, build_scan_output,
                                     eval_disruptive, classify_region)


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


def check_model(model=None):
    """Check that ``model`` is a fitted classifier exposing ``predict_proba`` (or None)."""
    if model is None:
        return None
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError(f"'model' ({type(model).__name__}) should be a fitted classifier "
                         f"with a 'predict_proba' method, or None.")
    return model


def check_target_class(model=None, target_class=None):
    """Check ``target_class`` against the bound model; require a model when it is set."""
    if target_class is None:
        return None
    if model is None:
        raise ValueError("'target_class' was given without a 'model'; pass a fitted model "
                         "or leave 'target_class' as None.")
    classes = getattr(model, "classes_", None)
    if classes is not None and target_class not in list(classes):
        raise ValueError(f"'target_class' ({target_class}) should be one of the model "
                         f"classes {list(classes)}.")
    return target_class


def check_match_model_df_feat(model=None, df_feat=None):
    """Check the bound model was fitted on the same number of features as ``df_feat``."""
    if model is None:
        return None
    n_feat = len(df_feat)
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != n_feat:
        raise ValueError(f"'model' was fitted on {int(n_in)} features but 'df_feat' has "
                         f"{n_feat}; the model and df_feat must describe the same feature set.")
    return None


def check_match_variants_df_seq(variants=None, df_seq=None):
    """Check the variants table (entry, variant, pos, to_aa) and derive from_aa per row."""
    cols_required = [ut.COL_ENTRY, ut.COL_VARIANT, ut.COL_POS, ut.COL_TO_AA]
    ut.check_df(df=variants, name="variants", cols_required=cols_required)
    list_from = check_match_mutations_df_seq(mutations=variants, df_seq=df_seq)
    # No two mutations of the same variant may touch the same position.
    for (entry, var), g in variants.groupby([ut.COL_ENTRY, ut.COL_VARIANT], sort=False):
        positions = [int(p) for p in g[ut.COL_POS]]
        if len(set(positions)) != len(positions):
            raise ValueError(f"variant '{var}' (entry '{entry}') has two mutations at the "
                             f"same position; a variant must mutate distinct positions.")
    return list_from


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
                 model: Optional[Any] = None,
                 target_class: Optional[Any] = None,
                 ):
        """
        Parameters
        ----------
        verbose : bool, default=False
            If ``True``, verbose outputs are enabled.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of amino acid scales (index = canonical amino acids, columns = scale ids).
            Default from :func:`load_scales`.
        model : object, optional
            A fitted classifier exposing ``predict_proba`` (e.g. :class:`TreeModel` or any
            scikit-learn classifier) trained on the CPP feature matrix of the ``df_feat`` used
            at call time. When given, the methods add a model prediction-shift column
            ``delta_pred`` (the change of the predicted score a mutation induces, in percentage
            points) and :meth:`SeqMut.suggest` is guided by it. When ``None`` (default) the class
            stays deterministic and model-free.
        target_class : int or str, optional
            Class whose predicted probability ``delta_pred`` tracks. ``None`` (default) selects
            the positive class. A class label is matched against ``model.classes_`` when
            available. Requires ``model``.

        See Also
        --------
        * :class:`AAMut` for the residue-level, CPP-agnostic substitution analysis.
        * :class:`CPP` whose ``df_feat`` defines the features mutated against.
        * :class:`TreeModel` whose ``predict_proba`` provides the prediction score.
        """
        self._verbose = ut.check_verbose(verbose)
        if df_scales is None:
            df_scales = ut.load_default_scales()
        self.df_scales = df_scales
        self._sf = SequenceFeature(verbose=False)
        self._model = check_model(model=model)
        self._target_class = check_target_class(model=self._model, target_class=target_class)

    # Helper methods
    def _delta_table(self, df_plan=None, df_seq=None, df_feat=None, jmd_n_len=10, jmd_c_len=10,
                     weight=None, sort=True):
        """Run the ΔCPP (+ model ΔP) engine for a mutation plan and return the scored output."""
        features = list(df_feat[ut.COL_FEATURE])
        mean_dif = df_feat[ut.COL_MEAN_DIF].to_numpy(dtype=float)
        weight_vec = get_weight_vec(df_feat=df_feat, weight=weight)
        X_wt, X_mut, wt_rows = comp_feature_matrices(
            df_plan=df_plan, df_seq=df_seq, features=features, df_scales=self.df_scales,
            sf=self._sf, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        dX = X_mut - X_wt[wt_rows]
        delta_cpp, shift_score = comp_scan_scores(dX=dX, mean_dif=mean_dif, weight_vec=weight_vec)
        if self._model is not None:
            delta_pred, wt_pred, wt_pred_std = comp_pred_scores(
                X_wt=X_wt, X_mut=X_mut, wt_rows=wt_rows, model=self._model,
                target_class=self._target_class)
            return build_scan_output(df_plan=df_plan, delta_cpp=delta_cpp, shift_score=shift_score,
                                     delta_pred=delta_pred, wt_pred=wt_pred, wt_pred_std=wt_pred_std,
                                     sort=sort)
        return build_scan_output(df_plan=df_plan, delta_cpp=delta_cpp, shift_score=shift_score, sort=sort)

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
            (``delta_cpp``) and ``shift_score`` toward the test-class profile are added; when a
            ``model`` is bound to this :class:`SeqMut`, the model prediction-shift ``delta_pred``
            is added too.
        jmd_n_len : int, default=10
            Length of JMD-N in number of amino acids.
        jmd_c_len : int, default=10
            Length of JMD-C in number of amino acids.

        Returns
        -------
        df_mut : pd.DataFrame, shape (n_mutations, n_info)
            The ``mutations`` table augmented with ``from_aa``, ``mutation`` (``"<from><pos><to>"``),
            ``sequence_mut`` (the mutated sequence), and — when ``df_feat`` is given — ``delta_cpp``
            and ``shift_score`` (plus ``delta_pred`` when a model is bound).

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
            check_match_model_df_feat(model=self._model, df_feat=df_feat)
            df_plan = df_mut[[ut.COL_ENTRY, ut.COL_POS, ut.COL_FROM_AA, ut.COL_TO_AA]].copy()
            df_plan[ut.COL_TMD_START] = [tmd_start[e] for e in df_plan[ut.COL_ENTRY]]
            df_plan[ut.COL_TMD_STOP] = [tmd_stop[e] for e in df_plan[ut.COL_ENTRY]]
            df_plan[ut.COL_REGION] = [classify_region(pos=int(p), tmd_start=int(ts), tmd_stop=int(te))
                                      for p, ts, te in zip(df_plan[ut.COL_POS],
                                                           df_plan[ut.COL_TMD_START],
                                                           df_plan[ut.COL_TMD_STOP])]
            # Score in mutation order (sort=False) so results align row-for-row with df_mut;
            # no label re-join, so duplicate mutation rows can never desync or crash.
            df_scored = self._delta_table(df_plan=df_plan, df_seq=df_seq, df_feat=df_feat,
                                          jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, sort=False)
            df_mut[ut.COL_DELTA_CPP] = df_scored[ut.COL_DELTA_CPP].to_numpy()
            df_mut[ut.COL_SHIFT_SCORE] = df_scored[ut.COL_SHIFT_SCORE].to_numpy()
            if self._model is not None:
                df_mut[ut.COL_DELTA_PRED] = df_scored[ut.COL_DELTA_PRED].to_numpy()
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
            ``delta_cpp``. When a ``model`` is bound to this :class:`SeqMut`, the model
            prediction-shift columns ``delta_pred`` (ΔP, percentage points), ``wt_pred`` and
            ``wt_pred_std`` are appended — this is the data behind the mutation-scan heatmap.

        Examples
        --------
        .. include:: examples/seqmut_scan.rst
        """
        # Validate
        check_match_df_seq_pos_based(df_seq=df_seq)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        check_match_model_df_feat(model=self._model, df_feat=df_feat)
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
        Suggest the top mutations that move a sequence toward the desired CPP / model outcome.

        Without a bound ``model``, mutations are ranked by ``shift_score`` = ``Sum sign(mean_dif)
        * ΔX`` (optionally weighted by a ``df_feat`` column), i.e. how strongly they move features
        in the direction by which the test class differs from the reference class. With a bound
        ``model`` the ranking switches to the model prediction-shift ``delta_pred`` (the ML-guided
        objective), so the suggested mutations are those predicted to raise the target-class score
        most. This is the single-objective design primitive; combining several mutations into one
        variant is :meth:`SeqMut.combine`.

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
            ``'abs_auc'``). If ``None``, all features contribute equally. Ignored when a ``model``
            is bound (the ranking then uses ``delta_pred``).
        jmd_n_len : int, default=10
            Length of JMD-N in number of amino acids.
        jmd_c_len : int, default=10
            Length of JMD-C in number of amino acids.

        Returns
        -------
        df_suggest : pd.DataFrame, shape (n, 8)
            The top-``n`` mutations sorted by descending ``shift_score`` — or by descending
            ``delta_pred`` when a ``model`` is bound (the table then also carries the model
            prediction-shift columns).

        Examples
        --------
        .. include:: examples/seqmut_suggest.rst
        """
        # Validate
        check_match_df_seq_pos_based(df_seq=df_seq)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        check_match_model_df_feat(model=self._model, df_feat=df_feat)
        region = check_region(region=region)
        to_aa = check_to_aa_set(to_aa=to_aa)
        ut.check_number_range(name="n", val=n, min_val=1, just_int=True)
        get_weight_vec(df_feat=df_feat, weight=weight)  # validate weight early
        ut.check_number_range(name="jmd_n_len", val=jmd_n_len, min_val=0, just_int=True)
        ut.check_number_range(name="jmd_c_len", val=jmd_c_len, min_val=0, just_int=True)
        # Scan + rank by the ML-guided prediction shift (model) or the target-shift (model-free)
        df_plan = build_scan_plan(df_seq=df_seq, region=region, to_aa=to_aa,
                                  jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        if len(df_plan) == 0:
            raise ValueError("No scannable positions for the given 'region'.")
        df_scan = self._delta_table(df_plan=df_plan, df_seq=df_seq, df_feat=df_feat,
                                    jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, weight=weight)
        rank_col = ut.COL_DELTA_PRED if self._model is not None else ut.COL_SHIFT_SCORE
        df_suggest = df_scan.sort_values(rank_col, ascending=False).head(n).reset_index(drop=True)
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

    def combine(self,
                df_seq: pd.DataFrame,
                variants: pd.DataFrame,
                df_feat: pd.DataFrame,
                jmd_n_len: int = 10,
                jmd_c_len: int = 10,
                ) -> pd.DataFrame:
        """
        Score combined (multi-mutation) variants by applying their mutations together.

        Each variant groups several point mutations that are applied to the **same** sequence,
        yielding one combined sequence whose ΔCPP (and, with a bound ``model``, prediction shift
        ``delta_pred``) is measured once. This is how 2-3 mutations are combined, in contrast to
        :meth:`SeqMut.mutate`, which scores every point mutation independently against the
        wild-type.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers, in the
            **position-based** format (``sequence``, ``tmd_start``, ``tmd_stop``). See
            :meth:`SequenceFeature.get_df_parts` for the full ``df_seq`` format specification.
        variants : pd.DataFrame, shape (n_mutations, >=4)
            Tidy table with columns ``entry``, ``variant`` (a grouping id), ``pos`` (1-based) and
            ``to_aa``. Rows sharing the same ``entry`` and ``variant`` are applied together as one
            combined variant; ``from_aa`` is derived and checked, and a variant must mutate
            distinct positions.
        df_feat : pd.DataFrame
            CPP feature set (output of :meth:`CPP.run`) defining the features measured over.
        jmd_n_len : int, default=10
            Length of JMD-N in number of amino acids.
        jmd_c_len : int, default=10
            Length of JMD-C in number of amino acids.

        Returns
        -------
        df_variant : pd.DataFrame, shape (n_variants, 6)
            One row per combined variant with ``entry``, ``variant`` (the ``'+'``-joined single
            mutations, e.g. ``"R20K+K27P"``), ``n_mut``, ``sequence_mut``, ``delta_cpp`` and
            ``shift_score`` — plus ``delta_pred`` when a model is bound — sorted by descending
            ``delta_pred`` (model) or ``shift_score`` (model-free).

        Examples
        --------
        .. include:: examples/seqmut_combine.rst
        """
        # Validate
        check_match_df_seq_pos_based(df_seq=df_seq)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        check_match_model_df_feat(model=self._model, df_feat=df_feat)
        list_from = check_match_variants_df_seq(variants=variants, df_seq=df_seq)
        ut.check_number_range(name="jmd_n_len", val=jmd_n_len, min_val=0, just_int=True)
        ut.check_number_range(name="jmd_c_len", val=jmd_c_len, min_val=0, just_int=True)
        # Build one combined sequence per (entry, variant)
        variants = variants.reset_index(drop=True).copy()
        variants[ut.COL_FROM_AA] = list_from
        seq_by_entry = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ]))
        tmd_start = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_TMD_START]))
        tmd_stop = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_TMD_STOP]))
        rows = []
        for (entry, _var), g in variants.groupby([ut.COL_ENTRY, ut.COL_VARIANT], sort=False):
            g = g.sort_values(ut.COL_POS)
            mut_seq, labels = seq_by_entry[entry], []
            for pos, from_aa, to_aa in zip(g[ut.COL_POS], g[ut.COL_FROM_AA], g[ut.COL_TO_AA]):
                pos = int(pos)
                mut_seq = mut_seq[:pos - 1] + to_aa + mut_seq[pos:]
                labels.append(f"{from_aa}{pos}{to_aa}")
            rows.append((entry, "+".join(labels), len(g), mut_seq,
                         tmd_start[entry], tmd_stop[entry]))
        df_var = pd.DataFrame(rows, columns=[ut.COL_ENTRY, ut.COL_VARIANT, ut.COL_N_MUT,
                                             ut.COL_SEQ_MUT, ut.COL_TMD_START, ut.COL_TMD_STOP])
        # Score the combined variants
        features = list(df_feat[ut.COL_FEATURE])
        mean_dif = df_feat[ut.COL_MEAN_DIF].to_numpy(dtype=float)
        scores = comp_seq_scores(df_var=df_var, df_seq=df_seq, features=features,
                                 mean_dif=mean_dif, df_scales=self.df_scales, sf=self._sf,
                                 model=self._model, target_class=self._target_class,
                                 jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        df_variant = df_var[ut.COLS_SEQMUT_VARIANT[:4]].copy()
        df_variant[ut.COL_DELTA_CPP] = scores[ut.COL_DELTA_CPP]
        df_variant[ut.COL_SHIFT_SCORE] = scores[ut.COL_SHIFT_SCORE]
        rank_col = ut.COL_SHIFT_SCORE
        if self._model is not None:
            df_variant[ut.COL_DELTA_PRED] = scores[ut.COL_DELTA_PRED]
            rank_col = ut.COL_DELTA_PRED
        df_variant = df_variant.sort_values(rank_col, ascending=False).reset_index(drop=True)
        return df_variant

"""
This is a script for the frontend of the AAWindowSampler class.
"""
from typing import Optional, List, Tuple, Union, Dict, Callable
import math
import numbers
import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.aa_window_sampler._utils import (parse_pos_col,
                                                 collect_test_windows,
                                                 window_offsets)
from ._backend.aa_window_sampler.sample_same_protein import sample_same_protein
from ._backend.aa_window_sampler.sample_different_protein import sample_different_protein
from ._backend.aa_window_sampler.sample_synthetic import (sample_synthetic,
                                                            LIST_SYNTH_GENERATORS,
                                                            PRESETS)
from ._backend.aa_window_sampler.sample_motif_matched import sample_motif_matched
from ._backend.aa_window_sampler.build_output import (build_segments_output,
                                                       build_sequences_output)


# I Helper Functions
def check_similarity_threshold(name, val):
    """Validate a similarity threshold in [0, 1]; ``None`` is accepted."""
    if val is None:
        return None
    ut.check_number_range(name=name, val=val, min_val=0, max_val=1,
                          just_int=False, accept_none=False)
    return float(val)


def check_output_mode(output_mode):
    """Validate ``output_mode`` is one of the allowed modes."""
    ut.check_str_options(name="output_mode", val=output_mode,
                         list_str_options=ut.LIST_OUTPUT_MODES)


def check_distance_to_pos(min_distance_to_pos, max_distance_to_pos):
    """Validate the ``(min, max)`` distance-to-positive band.

    Each bound is either a non-negative ``int`` or ``None`` (meaning "no bound").
    If both are set, ``min_distance_to_pos <= max_distance_to_pos``.
    """
    if min_distance_to_pos is not None:
        ut.check_number_range(name="min_distance_to_pos", val=min_distance_to_pos,
                              min_val=0, just_int=True)
    if max_distance_to_pos is not None:
        ut.check_number_range(name="max_distance_to_pos", val=max_distance_to_pos,
                              min_val=0, just_int=True)
    if (min_distance_to_pos is not None and max_distance_to_pos is not None
            and min_distance_to_pos > max_distance_to_pos):
        raise ValueError(
            f"'min_distance_to_pos' ({min_distance_to_pos}) should be "
            f"<= 'max_distance_to_pos' ({max_distance_to_pos})")


def check_synth_generator(generator):
    """Validate the polymorphic ``generator`` parameter of :meth:`AAWindowSampler.sample_synthetic`.

    Accepts three shapes:

    * ``str``: a built-in generator (``uniform``, ``global_freq``, ``position_specific``,
      ``scrambled``) or an AAontology preset name from :data:`PRESETS`.
    * ``list[str]`` or ``tuple[str, ...]``: at least two **distinct** preset
      names from :data:`PRESETS`. The backend computes the multiplicative mix
      (see ``_mix_preset_aa_freq`` and [LiuDeber99]_).
    * ``dict[str, Real]``: at least two single-character keys mapped to
      non-negative probabilities that sum to ``1.0`` (within ``1e-6``). Keys
      define a custom alphabet, so the sampler is not restricted to amino
      acids; keys are case-sensitive (``'A'`` and ``'a'`` are distinct symbols).

    Returns ``generator`` unchanged on success.
    """
    if isinstance(generator, str):
        ut.check_str_options(name="generator", val=generator,
                             list_str_options=LIST_SYNTH_GENERATORS)
        return generator
    if isinstance(generator, (list, tuple)):
        if len(generator) < 2:
            raise ValueError(f"'generator' (sequence of length {len(generator)}) should "
                             f"have at least 2 components")
        for m in generator:
            if not isinstance(m, str):
                raise ValueError(f"'generator' (sequence element {m!r}) should be a string")
            if m not in PRESETS:
                raise ValueError(f"'generator' (sequence element {m!r}) should be "
                                 f"one of {sorted(PRESETS)}")
        if len(set(generator)) != len(generator):
            raise ValueError(f"'generator' ({list(generator)}) should not contain "
                             f"duplicate components")
        return generator
    if isinstance(generator, dict):
        if len(generator) < 2:
            raise ValueError(f"'generator' (dict of size {len(generator)}) should have "
                             f"at least 2 entries")
        for k, v in generator.items():
            if not isinstance(k, str) or len(k) != 1:
                raise ValueError(f"'generator' (dict key {k!r}) should be a "
                                 f"single-character string")
            if (isinstance(v, bool)
                    or not isinstance(v, numbers.Real)
                    or math.isnan(float(v))
                    or math.isinf(float(v))
                    or v < 0):
                raise ValueError(f"'generator' (dict value {v!r} for key {k!r}) "
                                 f"should be a finite non-negative number")
        total = float(sum(generator.values()))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"'generator' (dict values summing to {total}) should "
                             f"sum to 1.0 (±1e-6)")
        return generator
    raise ValueError(f"'generator' (type {type(generator).__name__}) should be a str, "
                     f"list/tuple of str, or dict[str, Real]")


def check_pwm(motif_pwm, window_size):
    """Validate ``motif_pwm`` and return it reindexed to ``ut.LIST_CANONICAL_AA``.

    Returns an ``np.ndarray`` of shape ``(window_size, len(ut.LIST_CANONICAL_AA))``
    with columns ordered by ``ut.LIST_CANONICAL_AA`` (alphabetical,
    ``ACDEFGHIKLMNPQRSTVWY``). ``motif_pwm`` must be a ``pd.DataFrame`` whose
    columns are the 20 canonical AA letters in any order (reindexed internally);
    non-canonical or missing canonical columns are rejected (the canonical
    protein alphabet is required by both the PWM-sum scan and the FIMO engine).
    Raises ``ValueError`` for malformed inputs.
    """
    if not isinstance(motif_pwm, pd.DataFrame):
        raise ValueError(
            f"'motif_pwm' (type {type(motif_pwm).__name__}) should be a "
            f"pd.DataFrame with the 20 canonical AAs as columns "
            f"({ut.LIST_CANONICAL_AA}).")
    canonical = set(ut.LIST_CANONICAL_AA)
    cols = set(motif_pwm.columns)
    missing = sorted(canonical - cols)
    extra = sorted(cols - canonical)
    if missing or extra:
        raise ValueError(
            f"'motif_pwm' (DataFrame columns {sorted(cols)}) should have "
            f"exactly the 20 canonical AAs as columns "
            f"({ut.LIST_CANONICAL_AA}); missing={missing}, extra={extra}")
    pwm = motif_pwm.reindex(columns=ut.LIST_CANONICAL_AA).to_numpy(dtype=float)
    expected_shape = (window_size, len(ut.LIST_CANONICAL_AA))
    if pwm.shape != expected_shape:
        raise ValueError(f"'motif_pwm' (shape {pwm.shape}) should be "
                         f"{expected_shape} (window_size, n_aa).")
    return pwm


def check_motif_args(motif_pwm, motif_score_threshold, motif_match, window_size):
    """Validate the motif-filter parameter triplet (or pair, when ``motif_match`` is ``None``).

    Returns the validated PWM as a ``np.ndarray`` of shape
    ``(window_size, len(ut.LIST_CANONICAL_AA))`` with columns ordered by
    ``ut.LIST_CANONICAL_AA`` (alphabetical, ``ACDEFGHIKLMNPQRSTVWY``), or
    ``None`` when no motif filter is requested (PWM validation delegated to
    :func:`check_pwm`).

    Pass ``motif_match=None`` from callers that do not expose ``motif_match``
    (e.g. :meth:`AAWindowSampler.sample_motif_matched`, which only scores and
    keeps high-scoring windows). Raises ``ValueError`` for inconsistent or
    malformed inputs.
    """
    if motif_pwm is None and motif_score_threshold is None:
        return None
    if motif_pwm is None:
        raise ValueError("'motif_score_threshold' was given without 'motif_pwm'.")
    if motif_score_threshold is None:
        raise ValueError("'motif_pwm' was given without 'motif_score_threshold'.")
    pwm = check_pwm(motif_pwm, window_size)
    ut.check_number_range(name="motif_score_threshold",
                          val=motif_score_threshold,
                          accept_none=False, just_int=False)
    if motif_match is not None:
        ut.check_str_options(name="motif_match", val=motif_match,
                             list_str_options=ut.LIST_MOTIF_MATCHES)
    return pwm


def check_context_args(aa_context_col, context_in, context_out):
    """Guard against silent-drop foot-gun: ``context_in``/``context_out`` require ``aa_context_col``."""
    if (context_in is not None or context_out is not None) and aa_context_col is None:
        raise ValueError("'context_in'/'context_out' require 'aa_context_col' to be set.")


def check_custom_filter(custom_filter):
    """Validate the ``custom_filter`` hook is a callable (or ``None``)."""
    if custom_filter is not None and not callable(custom_filter):
        raise ValueError(f"'custom_filter' (type {type(custom_filter).__name__}) "
                         f"should be a callable (window, entry, source_position) "
                         f"-> bool, or None")
    return custom_filter


def check_arms(arms):
    """Validate the ``arms`` mapping passed to :meth:`AAWindowSampler.sample_benchmark_set`.

    Each value must be a dict carrying a ``"method"`` key whose value is one of
    :data:`ut.LIST_STRATEGIES`; the remaining keys forward as kwargs to that
    ``sample_*`` method. Keys managed by ``sample_benchmark_set`` itself
    (``df_seq``, ``seed``, ``output_mode``) must not appear in an arm config.
    """
    if not isinstance(arms, dict) or len(arms) == 0:
        raise ValueError(f"'arms' ({arms!r}) should be a non-empty dict mapping an "
                         f"arm name to a {{'method': ..., **kwargs}} config")
    reserved = {"df_seq", "seed", "output_mode"}
    for name, spec in arms.items():
        if not isinstance(name, str):
            raise ValueError(f"'arms' key ({name!r}) should be a string arm name")
        if not isinstance(spec, dict):
            raise ValueError(f"'arms[{name!r}]' (type {type(spec).__name__}) should be "
                             f"a dict with a 'method' key")
        if "method" not in spec:
            raise ValueError(f"'arms[{name!r}]' (keys {sorted(spec)}) should include a "
                             f"'method' key")
        method = spec["method"]
        if method not in ut.LIST_STRATEGIES:
            raise ValueError(f"'arms[{name!r}][\"method\"]' ({method!r}) should be one "
                             f"of {ut.LIST_STRATEGIES}")
        bad = sorted(reserved.intersection(spec))
        if bad:
            raise ValueError(f"'arms[{name!r}]' (reserved keys {bad}) should not set "
                             f"keys managed by sample_benchmark_set "
                             f"({sorted(reserved)})")


def _filter_aa_context(df_seq=None, aa_context_col=None, context_in=None, context_out=None):
    """Return per-row 1-based positions whose per-residue context tag passes the filter.

    Parameters
    ----------
    df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
        DataFrame containing an ``entry`` column with unique protein identifiers and a
        ``sequence`` column with full protein sequences. Used here as the source of
        per-residue context tags via ``aa_context_col``.
    aa_context_col : str
        Column with a per-residue context tag for every row. Each cell is a string
        or sequence (list / tuple / array) whose length equals the row's sequence
        length; element ``i`` is the context tag for residue ``i+1`` (1-based).
        Empty / missing cells (``None`` / ``NaN``) are treated as "no eligible
        residues" for that row.
    context_in : value or list-like, optional
        Whitelist: residues whose context tag is in ``context_in`` are kept.
    context_out : value or list-like, optional
        Blacklist: residues whose context tag is in ``context_out`` are dropped.

    Returns
    -------
    list of list of int
        For each ``df_seq`` row, the 1-based positions that satisfy the filter.

    Raises
    ------
    ValueError
        If ``aa_context_col`` is missing from ``df_seq``, neither ``context_in``
        nor ``context_out`` is provided, or any cell length disagrees with the
        sequence length.
    """
    if aa_context_col not in df_seq.columns:
        raise ValueError(f"'aa_context_col' ('{aa_context_col}') is not a column "
                         f"of 'df_seq'. Columns: {list(df_seq.columns)}")
    if context_in is None and context_out is None:
        raise ValueError("'_filter_aa_context' requires at least one of 'context_in' "
                         "or 'context_out'.")

    def _to_set(val):
        if val is None:
            return None
        if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
            return set(val)
        return {val}

    in_set = _to_set(context_in)
    out_set = _to_set(context_out)
    allowed = []
    for entry, seq, ctx in zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ],
                                df_seq[aa_context_col]):
        if ctx is None or (isinstance(ctx, float) and np.isnan(ctx)):
            allowed.append([])
            continue
        if len(ctx) != len(seq):
            raise ValueError(f"'{aa_context_col}' for entry '{entry}' has length "
                             f"{len(ctx)} but sequence length is {len(seq)}.")
        positions = []
        for i, tag in enumerate(ctx, start=1):
            if in_set is not None and tag not in in_set:
                continue
            if out_set is not None and tag in out_set:
                continue
            positions.append(i)
        allowed.append(positions)
    return allowed


# II Main Function
class AAWindowSampler:
    """
    Utility class for sampling amino-acid windows / segments from full protein sequences.

    Four sampling strategies are provided:

    - :meth:`sample_same_protein` — windows from proteins that contain at least one test
      position [Boyd10Cascleave]_, [Song12]_.
    - :meth:`sample_different_protein` — windows from proteins outside the test set;
      naturally suited as the unlabeled pool ``U`` for positive-unlabeled learning
      [ElkanNoto08]_, [BekkerDavis20]_.
    - :meth:`sample_synthetic` — synthetic control windows from built-in priors,
      AAontology presets [Rawlings16]_, multiplicative preset mixes [LiuDeber99]_, or
      custom-alphabet frequency tables.
    - :meth:`sample_motif_matched` — in-memory FIMO (Find Individual Motif Occurrences)-style
      scan against a user-supplied Position Weight Matrix (PWM); a Command-Line
      Interface (CLI) parity wrapper that delegates to ``fimo`` lives at
      :func:`aaanalysis.scan_motif`.

    Output modes (per method, except :meth:`sample_synthetic` which is segments-only):

    * ``"segments"`` — one row per sampled window, schema
      ``[entry_win, entry, sequence, window, source_position, label, role, strategy]``.
      ``entry_win = <entry>_<start_pos>-<end_pos>`` (1-based inclusive); the same biological
      window across calls produces the same ``entry_win``, so
      ``drop_duplicates(subset="entry_win")`` is the natural cross-call dedupe primitive.
      Synthetic outputs use ``entry_win = "synth_{i}"`` with a per-call counter — concatenating
      multiple :meth:`sample_synthetic` outputs may collide; deduplicate on the ``window``
      column instead.
    * ``"sequences"`` — one row per source protein with a ``labels`` list of length
      ``len(sequence)`` carrying ``label_test`` at known test positions, ``label_ref`` at
      sampled positions, and ``None`` elsewhere.

    .. versionadded:: 1.1.0

    Notes
    -----
    Class type:
        This is a utility class — it does not implement ``.fit`` / ``.run`` / ``.eval``.
        Compute output-quality metrics from the returned DataFrame using
        :func:`aaanalysis.metrics.comp_kld` or the backend ``window_identity`` helper.

    Identity-based similarity:
        Two filters operate on per-position residue identity of equal-length windows
        (no alignment needed):

        * ``max_similarity_to_test`` — drop sampled windows whose identity to any
          known test window exceeds the threshold (anti-leakage).
        * ``max_similarity_within_ref`` — greedily drop sampled windows whose identity
          to a previously kept sampled window exceeds the threshold (redundancy
          reduction). In :meth:`sample_same_protein`, this filter spans protein
          boundaries; protein iteration order is randomized under the seed so output
          depends only on ``df_seq`` content + seed, not row order.

    Iterative filtering:
        If filtering shrinks the candidate pool below the target, additional draws are
        performed up to ``max_sampling_attempts``. If still insufficient, a warning is
        emitted and the available samples are returned.

    Anchoring convention:
        Positions in ``pos_col`` and the emitted ``source_position`` are interpreted as
        **P1**-style residue anchors under Schechter–Berger cleavage nomenclature
        [Rawlings16]_. For window length ``L``, the window covers ``(L - 1) // 2``
        residues upstream of the anchor, the anchor itself, and ``L // 2`` residues
        downstream — right-heavy for even ``L``.

    See Also
    --------
    * :class:`aaanalysis.SequencePreprocessor` for ``aa_window`` extraction primitives.
    * :class:`aaanalysis.SequenceFeature` for canonical ``df_seq`` formats and conventions.
    """

    def __init__(self,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
                 max_similarity_to_test: Optional[float] = None,
                 max_similarity_within_ref: Optional[float] = None,
                 filter_iteratively: bool = True,
                 max_sampling_attempts: int = 10,
                 custom_filter: Optional[Callable[[str, str, int], bool]] = None,
                 ):
        """
        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, prints sampling progress and warnings when fewer windows
            than requested are returned.
        random_state : int, optional
            Default seed for all sampling methods. A per-call ``seed`` overrides it.
        max_similarity_to_test : float in [0, 1], optional
            Drop sampled windows whose per-position identity to any test window exceeds
            this threshold.
        max_similarity_within_ref : float in [0, 1], optional
            Greedily drop sampled windows whose per-position identity to a previously kept
            sampled window exceeds this threshold.
        filter_iteratively : bool, default=True
            Iteratively re-draw if filtering reduces the candidate pool below the target.
        max_sampling_attempts : int, default=10
            Cap on iterative re-draw attempts.
        custom_filter : callable, optional
            User-supplied keep-predicate ``(window, entry, source_position) -> bool``
            applied to every sampled window across all ``sample_*`` methods; a window
            is kept only when it returns ``True``. ``window`` is the window string,
            ``entry`` its source protein, and ``source_position`` the 1-based P1 anchor.
            The escape hatch for structure- / domain-specific decoy rules. Synthetic
            windows have no source protein, so it is called with ``entry=""`` and
            ``source_position=-1``. If the predicate raises during sampling, the error
            surfaces as a ``RuntimeError`` naming the offending window (the original
            exception is chained).
        """
        self.verbose = ut.check_verbose(verbose)
        self._random_state = ut.check_random_state(random_state=random_state)
        self._max_similarity_to_test = check_similarity_threshold(
            "max_similarity_to_test", max_similarity_to_test)
        self._max_similarity_within_ref = check_similarity_threshold(
            "max_similarity_within_ref", max_similarity_within_ref)
        ut.check_bool(name="filter_iteratively", val=filter_iteratively)
        self._filter_iteratively = filter_iteratively
        ut.check_number_range(name="max_sampling_attempts", val=max_sampling_attempts,
                              min_val=1, just_int=True)
        self._max_sampling_attempts = max_sampling_attempts
        self._custom_filter = check_custom_filter(custom_filter)

    # Internal helpers
    def _rng(self, seed):
        if seed is not None:
            ut.check_number_range(name="seed", val=seed, min_val=0, just_int=True)
            return np.random.default_rng(seed)
        return np.random.default_rng(self._random_state)

    # Sampling methods
    def sample_same_protein(self,
                            df_seq: pd.DataFrame = None,
                            n: int = 100,
                            window_size: int = 9,
                            pos_col: str = ut.COL_POS,
                            min_distance_to_pos: Optional[int] = None,
                            max_distance_to_pos: Optional[int] = None,
                            label_test: Union[int, float] = 1,
                            label_ref: Union[int, float] = 0,
                            role: str = ut.ROLE_NEG,
                            output_mode: str = ut.OUT_SEGMENTS,
                            aa_context_col: Optional[str] = None,
                            context_in: Optional[Union[str, List]] = None,
                            context_out: Optional[Union[str, List]] = None,
                            motif_pwm: Optional[pd.DataFrame] = None,
                            motif_score_threshold: Optional[float] = None,
                            motif_match: str = "in",
                            seed: Optional[int] = None,
                            ) -> pd.DataFrame:
        """Sample windows from proteins that contain at least one test position.

        Draws up to ``n`` reference windows from the same proteins that carry a labeled
        test position, making it the natural source for within-protein hard negatives.
        Windows are distributed roughly uniformly across eligible proteins and filtered
        by the similarity thresholds set on :class:`AAWindowSampler`. Complement this
        method with :meth:`sample_different_protein` when an unlabeled cross-protein
        pool is also needed.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers
            and a ``sequence`` column with full protein sequences. See Notes.
        n : int, default=100
            Maximum total number of sampled windows across all eligible proteins.
            ``n`` is split roughly uniformly across eligible source proteins (each
            protein gets ~``n / n_proteins`` windows); shortfalls from proteins with
            small candidate pools are redistributed round-robin. Fewer than ``n``
            are returned (with a warning) if the eligible space cannot supply.
        window_size : int, default=9
            Length of each sampled window in residues.
        pos_col : str, default='pos'
            Column with per-row 1-based positive positions. See Notes.
        min_distance_to_pos : int, optional
            Minimum residue distance from the nearest positive on the same protein
            (a sampled P1 anchor ``c`` is admitted only if
            ``min(|c - p| for p in positives) >= min_distance_to_pos``).
            ``None`` (default) drops this lower bound — sampled windows are allowed
            to overlap positive windows.
        max_distance_to_pos : int, optional
            Maximum residue distance from the nearest positive on the same protein
            (a sampled P1 anchor ``c`` is admitted only if
            ``min(|c - p| for p in positives) <= max_distance_to_pos``).
            ``None`` (default) drops this upper bound — sampled windows may sit
            anywhere on the protein.
        label_test : int or float, default=1
            Label assigned to positives in ``output_mode='sequences'``.
        label_ref : int or float, default=0
            Label assigned to sampled reference positions / rows.
        role : str, default='Negative'
            Role tag stored in the output's ``role`` column.
        output_mode : {'segments', 'sequences'}, default='segments'
            Output schema. See Notes.
        aa_context_col : str, optional
            Per-residue context column used with ``context_in`` / ``context_out``.
        context_in : value or list-like, optional
            Whitelist of ``aa_context_col`` tag values for eligible residues.
        context_out : value or list-like, optional
            Blacklist of ``aa_context_col`` tag values for excluded residues.
        motif_pwm : pd.DataFrame, optional
            Position Weight Matrix (PWM) of shape ``(window_size, 20)`` whose columns
            are the 20 canonical amino acid (AA) letters in any order (reindexed
            internally to ``ut.LIST_CANONICAL_AA``). Required together with
            ``motif_score_threshold`` when motif filtering is desired.
        motif_score_threshold : float, optional
            PWM score threshold; required when ``motif_pwm`` is set.
        motif_match : {'in', 'out'}, default='in'
            ``'in'`` keeps windows with score ``>=`` threshold; ``'out'`` keeps the rest.
        seed : int, optional
            Per-call seed; falls back to the class-level ``random_state``.

        Returns
        -------
        df_seq_out : pd.DataFrame
            Sampled windows; one row per window with ``entry``, ``sequence``,
            ``role``, ``strategy``, and ``entry_win`` columns.

        Notes
        -----
        Each row of ``df_seq`` whose ``pos_col`` cell is a non-empty list / tuple /
        array of 1-based integer positions is a "positive" row; rows with empty /
        ``None`` / ``NaN`` cells are skipped. Sampled windows are drawn from the
        same proteins as the positives; the positive windows themselves drive the
        ``max_similarity_to_test`` filter. The
        ``(min_distance_to_pos, max_distance_to_pos)`` band is exposed only on this
        method; :meth:`sample_different_protein` and :meth:`sample_motif_matched`
        sample from proteins with no listed positives, so the band has nothing to
        act on.

        With the default ``None`` / ``None`` band, sampled centers can sit directly
        on or adjacent to positive anchors, producing windows that overlap positive
        windows by up to ``window_size - 1`` residues. For hard-negative-style
        sampling that excludes positional overlap, set
        ``min_distance_to_pos=window_size``; to constrain sampled windows to a
        defined neighborhood of positives (e.g. local hard negatives), pair with a
        finite ``max_distance_to_pos``. Content-level overlap is controlled
        separately by ``max_similarity_to_test``.

        Protein iteration order is randomized under the seed; output is independent
        of ``df_seq`` row order.

        ``output_mode='segments'`` returns one row per sampled window with schema
        ``[entry_win, entry, sequence, window, source_position, label, role, strategy]``.
        ``output_mode='sequences'`` returns one row per source protein with a
        ``labels`` list of length ``len(sequence)`` carrying ``label_test`` at
        positives, ``label_ref`` at sampled positions, and ``None`` elsewhere.

        Examples
        --------
        .. include:: examples/aws_sample_same_protein.rst
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        ut.check_str(name="pos_col", val=pos_col, accept_none=False)
        ut.check_number_range(name="n", val=n, min_val=1, just_int=True)
        ut.check_number_range(name="window_size", val=window_size, min_val=1, just_int=True)
        check_distance_to_pos(min_distance_to_pos, max_distance_to_pos)
        check_output_mode(output_mode)
        ut.check_str(name="role", val=role, accept_none=False)
        check_context_args(aa_context_col, context_in, context_out)
        motif_pwm = check_motif_args(motif_pwm, motif_score_threshold,
                                      motif_match, window_size)
        # Build pool
        positions = parse_pos_col(df_seq, pos_col)
        test_windows = collect_test_windows(df_seq, pos_col, window_size)
        allowed_positions = (_filter_aa_context(df_seq=df_seq,
                                                 aa_context_col=aa_context_col,
                                                 context_in=context_in,
                                                 context_out=context_out)
                             if aa_context_col is not None else None)
        rng = self._rng(seed)
        rows, source_indices, sampled_centers = sample_same_protein(
            df_seq=df_seq, positions=positions,
            n=n, window_size=window_size,
            min_distance_to_pos=min_distance_to_pos,
            max_distance_to_pos=max_distance_to_pos,
            test_windows=test_windows, allowed_positions=allowed_positions,
            max_similarity_to_test=self._max_similarity_to_test,
            max_similarity_within_ref=self._max_similarity_within_ref,
            motif_pwm=motif_pwm,
            motif_score_threshold=motif_score_threshold,
            motif_match=motif_match,
            max_sampling_attempts=self._max_sampling_attempts,
            filter_iteratively=self._filter_iteratively,
            rng=rng, verbose=self.verbose,
            custom_filter=self._custom_filter,
        )
        # Build output
        if output_mode == ut.OUT_SEGMENTS:
            return build_segments_output(rows, strategy=ut.STRATEGY_SAME, role=role,
                                          label_value=label_ref,
                                          window_size=window_size)
        return build_sequences_output(df_seq, positions, sampled_centers,
                                       label_test=label_test, label_ref=label_ref,
                                       mark_test=True)

    def sample_different_protein(self,
                                 df_seq: pd.DataFrame = None,
                                 n: int = 100,
                                 window_size: int = 9,
                                 pos_col: str = ut.COL_POS,
                                 candidate_proteins: Optional[List[str]] = None,
                                 label_test: Union[int, float] = 1,
                                 label_ref: Union[int, float] = 0,
                                 role: str = ut.ROLE_UNL,
                                 output_mode: str = ut.OUT_SEGMENTS,
                                 aa_context_col: Optional[str] = None,
                                 context_in: Optional[Union[str, List]] = None,
                                 context_out: Optional[Union[str, List]] = None,
                                 motif_pwm: Optional[pd.DataFrame] = None,
                                 motif_score_threshold: Optional[float] = None,
                                 motif_match: str = "in",
                                 seed: Optional[int] = None,
                                 ) -> pd.DataFrame:
        """Sample windows from proteins outside the test set (proteins with no test positions).

        Draws up to ``n`` reference windows exclusively from proteins that carry no
        labeled positive positions, making them naturally unlabeled candidates for
        positive-unlabeled learning [ElkanNoto08]_, [BekkerDavis20]_. Use this method
        alongside :meth:`sample_same_protein` to build a combined reference pool that
        covers both within- and cross-protein negatives.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers
            and a ``sequence`` column with full protein sequences. See Notes.
        n : int, default=100
            Maximum total number of sampled windows. Fewer are returned (with a
            warning) if the eligible space cannot supply.
        window_size : int, default=9
            Length of each sampled window in residues.
        pos_col : str, default='pos'
            Column with per-row 1-based positive positions. See Notes.
        candidate_proteins : list of str, optional
            Restrict the candidate pool to these entries.
        label_test : int or float, default=1
            Label assigned to positives in ``output_mode='sequences'``.
        label_ref : int or float, default=0
            Label assigned to sampled positions / rows.
        role : str, default='Unlabeled'
            Role tag stored in the output's ``role`` column.
        output_mode : {'segments', 'sequences'}, default='segments'
            Output schema. See Notes.
        aa_context_col : str, optional
            Per-residue context column used with ``context_in`` / ``context_out``.
        context_in : value or list-like, optional
            Whitelist of ``aa_context_col`` tag values for eligible residues.
        context_out : value or list-like, optional
            Blacklist of ``aa_context_col`` tag values for excluded residues.
        motif_pwm : pd.DataFrame, optional
            Position Weight Matrix (PWM) of shape ``(window_size, 20)`` whose columns
            are the 20 canonical amino acid (AA) letters in any order (reindexed
            internally to ``ut.LIST_CANONICAL_AA``).
        motif_score_threshold : float, optional
            PWM score threshold; required when ``motif_pwm`` is set.
        motif_match : {'in', 'out'}, default='in'
            ``'in'`` keeps windows with score ``>=`` threshold; ``'out'`` keeps the rest.
        seed : int, optional
            Per-call seed; falls back to the class-level ``random_state``.

        Returns
        -------
        df_seq_out : pd.DataFrame
            Sampled windows; one row per window with ``entry``, ``sequence``,
            ``role``, ``strategy``, and ``entry_win`` columns.

        Notes
        -----
        ``df_seq`` plays a dual role: rows whose ``pos_col`` cell is a non-empty
        list / tuple / array of 1-based positions are *positive* rows — they are
        excluded from the candidate pool and contribute their windows only to the
        ``max_similarity_to_test`` filter. Rows with empty / ``None`` / ``NaN``
        cells form the candidate pool from which the returned windows are drawn.

        ``output_mode='segments'`` returns one row per sampled window with schema
        ``[entry_win, entry, sequence, window, source_position, label, role, strategy]``;
        ``output_mode='sequences'`` returns one row per protein with a per-residue
        ``labels`` list carrying ``label_test`` at the positives of positive proteins,
        ``label_ref`` at sampled positions in candidate proteins, and ``None``
        elsewhere — a single mergeable per-residue label vector across calls.

        Examples
        --------
        .. include:: examples/aws_sample_different_protein.rst
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        ut.check_str(name="pos_col", val=pos_col, accept_none=False)
        ut.check_number_range(name="n", val=n, min_val=1, just_int=True)
        ut.check_number_range(name="window_size", val=window_size, min_val=1, just_int=True)
        check_output_mode(output_mode)
        ut.check_str(name="role", val=role, accept_none=False)
        if candidate_proteins is not None:
            candidate_proteins = ut.check_list_like(name="candidate_proteins",
                                                    val=candidate_proteins,
                                                    accept_str=True, accept_none=False,
                                                    convert=True)
        check_context_args(aa_context_col, context_in, context_out)
        motif_pwm = check_motif_args(motif_pwm, motif_score_threshold,
                                      motif_match, window_size)
        # Build pool
        positions = parse_pos_col(df_seq, pos_col)
        test_windows = collect_test_windows(df_seq, pos_col, window_size)
        allowed_positions = (_filter_aa_context(df_seq=df_seq,
                                                 aa_context_col=aa_context_col,
                                                 context_in=context_in,
                                                 context_out=context_out)
                             if aa_context_col is not None else None)
        rng = self._rng(seed)
        rows, source_indices, sampled_centers = sample_different_protein(
            df_seq=df_seq, positions=positions, n=n, window_size=window_size,
            candidate_proteins=candidate_proteins, test_windows=test_windows,
            allowed_positions=allowed_positions,
            max_similarity_to_test=self._max_similarity_to_test,
            max_similarity_within_ref=self._max_similarity_within_ref,
            motif_pwm=motif_pwm,
            motif_score_threshold=motif_score_threshold,
            motif_match=motif_match,
            max_sampling_attempts=self._max_sampling_attempts,
            filter_iteratively=self._filter_iteratively,
            rng=rng, verbose=self.verbose,
            custom_filter=self._custom_filter,
        )
        # Build output
        if output_mode == ut.OUT_SEGMENTS:
            return build_segments_output(rows, strategy=ut.STRATEGY_DIFF, role=role,
                                          label_value=label_ref,
                                          window_size=window_size)
        return build_sequences_output(df_seq, positions, sampled_centers,
                                       label_test=label_test, label_ref=label_ref,
                                       mark_test=True)

    def sample_synthetic(self,
                         df_seq: pd.DataFrame = None,
                         n: int = 100,
                         window_size: int = 9,
                         generator: Union[str, List[str], Tuple[str, ...],
                                          Dict[str, float]] = ut.MODE_GLOBAL_FREQ,
                         pos_col: Optional[str] = None,
                         label_ref: Union[int, float] = 0,
                         role: str = ut.ROLE_CTRL,
                         seed: Optional[int] = None,
                         ) -> pd.DataFrame:
        """Generate synthetic control windows. Always returns ``output_mode='segments'``.

        Synthetic windows have no source protein, so ``output_mode`` is not exposed
        (no per-residue view exists). Synthetic rows use ``entry_win = "synth_{i}"``
        with a per-call counter — concatenating multiple :meth:`sample_synthetic`
        outputs may collide on ``entry_win``; deduplicate on the ``window`` column
        instead.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers
            and a ``sequence`` column with full protein sequences. See Notes.
        n : int, default=100
            Maximum total number of synthetic windows. Fewer are returned (with
            a warning) if the filters cannot supply.
        window_size : int, default=9
            Length of each synthetic window in residues.
        generator : str, list/tuple of str, or dict, default='global_freq'
            Synthesis recipe. Accepts three shapes:

            * ``str``: a single built-in generator (``'uniform'``, ``'global_freq'``,
              ``'position_specific'``, ``'scrambled'``) or an AAontology preset
              name (see Notes).
            * ``list[str]`` or ``tuple[str, ...]``: at least two **distinct**
              AAontology preset names. Their priors are combined into a
              multiplicative joint prior over the 20 canonical amino acids (AAs)
              (see Notes / [LiuDeber99]_). Duplicate components are rejected.
            * ``dict[str, Real]``: a custom frequency table. Keys are
              single-character symbols and values are non-negative
              probabilities summing to ``1.0`` (within ``1e-6``). Keys define
              the alphabet, so the sampler is **not** restricted to amino
              acids; any single-character symbols (e.g. nucleotides) work.
              Keys are case-sensitive (``'A'`` and ``'a'`` are distinct).
        pos_col : str, optional
            Column with per-row 1-based positive positions. See Notes.
        label_ref : int or float, default=0
            Label assigned to the synthetic rows.
        role : str, default='Control'
            Role tag stored in the output's ``role`` column.
        seed : int, optional
            Per-call seed; falls back to the class-level ``random_state``.

        Returns
        -------
        df_seq_out : pd.DataFrame
            Sampled windows; one row per window with ``entry``, ``sequence``,
            ``role``, ``strategy``, and ``entry_win`` columns.

        Notes
        -----
        ``df_seq`` is consumed differently by each generator:

        * ``generator='global_freq'``: source of empirical amino-acid frequencies
          across all sequences.
        * ``generator='position_specific'`` / ``'scrambled'``: source of *test windows*
          extracted at the 1-based positions in ``pos_col``. ``pos_col`` is
          required for these two generators.
        * ``generator='uniform'``, AAontology preset generators, list-mix generators,
          and custom dict generators: ``df_seq`` is **not** consumed for synthesis
          itself; ``pos_col`` is still optional and only used as the source of
          test windows for the ``max_similarity_to_test`` filter.

        **Built-in generators**

        * ``'uniform'``: each residue drawn uniformly from the 20 canonical AAs.
        * ``'global_freq'``: residues drawn from the empirical AA frequency in ``df_seq``.
        * ``'position_specific'``: per-position frequency of the test windows.
        * ``'scrambled'``: shuffle a randomly chosen test window.

        **AAontology preset generators** load a curated scale via
        :func:`aaanalysis.load_scales` and normalize its per-amino-acid values
        into a probability distribution. Composition presets are true AA-frequency
        distributions; conformation presets are normalized propensities used as
        physicochemically-biased priors.

        *Composition (3)*

        * ``'aa_composition'``: Dayhoff 1978a (canonical baseline)
        * ``'aa_composition_surface'``: Fukuchi-Nishikawa 2001 (surface composition)
        * ``'aa_composition_mp'``: Cedano 1997 (membrane proteins)

        *Conformation (7)*

        * ``'alpha_helix'``: Chou-Fasman 1978b
        * ``'beta_sheet'``: Chou-Fasman 1978b
        * ``'beta_strand'``: Lifson-Sander 1979
        * ``'beta_turn'``: Chou-Fasman 1978b
        * ``'coil'``: Nagano 1973
        * ``'linker'``: George-Heringa 2003 (medium 6-14 AA)
        * ``'pi_helix'``: Fodje-Al-Karadaghi 2002

        **Mixed-prior generator (list of preset names)** combines the per-AA
        probability vectors of the listed presets via element-wise product
        followed by renormalization, producing a Bayesian-style joint prior
        (e.g. ``generator=['aa_composition_mp', 'alpha_helix']`` for a
        membrane-helix prior). Combining hydrophobicity-like composition with
        helicity-like conformation as the basis for transmembrane
        characterization is supported by [LiuDeber99]_.

        **Custom alphabet generator (dict)** lets the user supply an arbitrary
        character-to-frequency table. Keys must be single characters (any
        symbol), values must be non-negative and sum to ``1.0``. The sampler
        is then no longer restricted to amino acids.

        **Single polymorphic ``generator``** — the three accepted shapes
        (built-in / preset ``str``, ``list[str]`` for a multiplicative preset
        mix, ``dict[str, float]`` for a custom-alphabet frequency table) all
        answer the same conceptual question ("recipe for one window"), so a
        single parameter is preferred over three mutually-exclusive named
        parameters. The dispatch-on-shape complexity is absorbed by the
        ``check_synth_generator`` validator.

        Examples
        --------
        .. include:: examples/aws_sample_synthetic.rst
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        ut.check_number_range(name="n", val=n, min_val=1, just_int=True)
        ut.check_number_range(name="window_size", val=window_size, min_val=1, just_int=True)
        check_synth_generator(generator)
        ut.check_str(name="role", val=role, accept_none=False)
        if pos_col is not None:
            ut.check_str(name="pos_col", val=pos_col, accept_none=False)
        # Build pool
        rng = self._rng(seed)
        test_windows = (collect_test_windows(df_seq, pos_col, window_size)
                        if pos_col is not None else [])
        windows = sample_synthetic(
            df_seq=df_seq, n=n, window_size=window_size, generator=generator,
            pos_col=pos_col, test_windows=test_windows,
            max_similarity_to_test=self._max_similarity_to_test,
            max_similarity_within_ref=self._max_similarity_within_ref,
            max_sampling_attempts=self._max_sampling_attempts,
            filter_iteratively=self._filter_iteratively,
            rng=rng, verbose=self.verbose,
            custom_filter=self._custom_filter,
        )
        # Build output — synthetic has no source protein, so we build inline
        # instead of routing through build_segments_output (entry_win = synth_{i},
        # per-call counter; source_position = -1 sentinel).
        if isinstance(generator, (list, tuple)):
            strategy_tag = f"{ut.STR_SYNTH_MIX}:{'+'.join(sorted(generator))}"
        elif isinstance(generator, dict):
            strategy_tag = f"{ut.STR_SYNTH_CUSTOM}:{'+'.join(sorted(generator))}"
        else:
            strategy_tag = generator
        strategy = f"{ut.STRATEGY_SYNTH_PREFIX}:{strategy_tag}"
        df = pd.DataFrame({
            ut.COL_ENTRY_WIN: [f"synth_{i}" for i in range(len(windows))],
            ut.COL_ENTRY: "",
            ut.COL_SEQ: "",
            ut.COL_WINDOW: windows,
            ut.COL_SOURCE_POS: -1,
            ut.COL_LABEL: label_ref,
            ut.COL_ROLE: role,
            ut.COL_STRATEGY: strategy,
        })
        return df[ut.COLS_SEGMENTS].copy()

    def sample_motif_matched(self,
                              df_seq: pd.DataFrame = None,
                              n: int = 100,
                              window_size: int = 9,
                              motif_pwm: Optional[pd.DataFrame] = None,
                              motif_score_threshold: Optional[float] = None,
                              pos_col: str = ut.COL_POS,
                              label_test: Union[int, float] = 1,
                              label_ref: Union[int, float] = 0,
                              role: str = ut.ROLE_NEG,
                              output_mode: str = ut.OUT_SEGMENTS,
                              aa_context_col: Optional[str] = None,
                              context_in: Optional[Union[str, List]] = None,
                              context_out: Optional[Union[str, List]] = None,
                              seed: Optional[int] = None,
                              ) -> pd.DataFrame:
        """Scan candidate proteins for windows matching a user-supplied Position Weight
        Matrix (PWM); a Find Individual Motif Occurrences (FIMO) equivalent.

        Useful for hard-negative mining: candidates that look like positives at
        the local-motif level but were not labeled positive.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers
            and a ``sequence`` column with full protein sequences. See Notes.
        n : int, default=100
            Maximum number of motif-matched windows to return.
        window_size : int, default=9
            Window length; must equal the first dimension of ``motif_pwm``.
        motif_pwm : pd.DataFrame
            Position Weight Matrix of shape ``(window_size, 20)`` whose columns are
            the 20 canonical amino acid (AA) letters in any order (reindexed internally
            to ``ut.LIST_CANONICAL_AA``). Required.
        motif_score_threshold : float
            Score threshold (sum of per-position PWM values). Required.
        pos_col : str, default='pos'
            Column with per-row 1-based positive positions. See Notes.
        label_test : int or float, default=1
            Label assigned to positives in ``output_mode='sequences'``.
        label_ref : int or float, default=0
            Label assigned to sampled motif-matched positions / rows.
        role : str, default='Negative'
            Role tag stored in the output's ``role`` column.
        output_mode : {'segments', 'sequences'}, default='segments'
            Output schema; see :meth:`sample_same_protein` Notes.
        aa_context_col : str, optional
            Per-residue context column used with ``context_in`` / ``context_out``.
        context_in : value or list-like, optional
            Whitelist of ``aa_context_col`` tag values for eligible residues.
        context_out : value or list-like, optional
            Blacklist of ``aa_context_col`` tag values for excluded residues.
        seed : int, optional
            Per-call seed; falls back to the class-level ``random_state``.

        Returns
        -------
        df_seq_out : pd.DataFrame
            Sampled windows; one row per window with ``entry``, ``sequence``,
            ``role``, ``strategy``, and ``entry_win`` columns.
            An additional ``motif_score`` column is appended when
            ``output_mode='segments'``; it is absent in ``'sequences'`` mode.

        Notes
        -----
        Rows of ``df_seq`` whose ``pos_col`` cell is a non-empty list / tuple /
        array of 1-based positions are *positive* rows — they are excluded from
        the scan and contribute their windows only to the
        ``max_similarity_to_test`` filter. Rows with empty / ``None`` / ``NaN``
        cells form the candidate pool, where every position with a fully-fitting
        window is scored against ``motif_pwm`` (sum of per-position values;
        non-canonical residues contribute zero). Positions with score
        ``>= motif_score_threshold`` are returned, ranked by descending score
        among those that survive the identity / context filters, and capped at
        ``n``.

        Unlike :meth:`sample_same_protein` and :meth:`sample_different_protein`,
        this method does not accept ``motif_match`` — it always returns
        high-scoring matches. For the inverse operation ("sample windows that do
        NOT match a motif"), call :meth:`sample_different_protein` with the same
        ``motif_pwm`` and ``motif_match='out'``.

        Examples
        --------
        .. include:: examples/aws_sample_motif_matched.rst
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        ut.check_str(name="pos_col", val=pos_col, accept_none=False)
        ut.check_number_range(name="n", val=n, min_val=1, just_int=True)
        ut.check_number_range(name="window_size", val=window_size, min_val=1, just_int=True)
        check_output_mode(output_mode)
        ut.check_str(name="role", val=role, accept_none=False)
        if motif_pwm is None:
            raise ValueError("'motif_pwm' is required for sample_motif_matched.")
        if motif_score_threshold is None:
            raise ValueError("'motif_score_threshold' is required for sample_motif_matched.")
        # `motif_match` is not exposed by sample_motif_matched (the scan only
        # keeps high-scoring windows); pass None so the validator skips that check.
        motif_pwm = check_motif_args(motif_pwm, motif_score_threshold,
                                      None, window_size)
        check_context_args(aa_context_col, context_in, context_out)
        # Build pool
        positions = parse_pos_col(df_seq, pos_col)
        test_windows = collect_test_windows(df_seq, pos_col, window_size)
        allowed_positions = (_filter_aa_context(df_seq=df_seq,
                                                 aa_context_col=aa_context_col,
                                                 context_in=context_in,
                                                 context_out=context_out)
                             if aa_context_col is not None else None)
        rng = self._rng(seed)
        rows, source_indices, sampled_centers, sampled_scores = sample_motif_matched(
            df_seq=df_seq, positions=positions, n=n, window_size=window_size,
            motif_pwm=motif_pwm, motif_score_threshold=motif_score_threshold,
            test_windows=test_windows, allowed_positions=allowed_positions,
            max_similarity_to_test=self._max_similarity_to_test,
            max_similarity_within_ref=self._max_similarity_within_ref,
            max_sampling_attempts=self._max_sampling_attempts,
            filter_iteratively=self._filter_iteratively,
            rng=rng, verbose=self.verbose,
            custom_filter=self._custom_filter,
        )
        # Build output
        if output_mode == ut.OUT_SEGMENTS:
            df_out = build_segments_output(rows, strategy=ut.STRATEGY_MOTIF_MATCHED,
                                            role=role, label_value=label_ref,
                                            window_size=window_size)
            df_out["motif_score"] = sampled_scores
            return df_out
        return build_sequences_output(df_seq, positions, sampled_centers,
                                       label_test=label_test, label_ref=label_ref,
                                       mark_test=True)

    def sample_benchmark_set(self,
                             df_seq: pd.DataFrame = None,
                             arms: Dict[str, Dict] = None,
                             seed: Optional[int] = None,
                             ) -> pd.DataFrame:
        """Run several named sampling arms and concatenate them into one benchmark set.

        Thin multi-arm orchestrator over the individual ``sample_*`` methods: it
        adds no new sampling behavior. Each arm is one ordinary ``sample_*`` call
        in ``'segments'`` mode, tagged with its arm name in an extra ``arm``
        column so a downstream benchmark can consume any mix of arms uniformly.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers
            and a ``sequence`` column with full protein sequences. Passed to every
            arm.
        arms : dict
            Mapping ``{arm_name: {"method": <strategy>, **kwargs}}``. ``method`` is
            one of ``'same_protein'``, ``'different_protein'``, ``'synthetic'``,
            ``'motif_matched'`` (``ut.LIST_STRATEGIES``); the remaining keys forward
            as keyword arguments to the matching ``sample_*`` method. The reserved
            keys ``df_seq``, ``seed``, and ``output_mode`` are managed here and must
            not appear in an arm config.
        seed : int, optional
            Master seed; falls back to the class-level ``random_state``. Per-arm
            sub-seeds are derived deterministically via :class:`numpy.random.SeedSequence`,
            so identical ``seed`` values reproduce identical benchmark sets.

        Returns
        -------
        df_seq_out : pd.DataFrame
            Row-wise concatenation of every arm's ``'segments'`` output with an
            added ``arm`` column. No automatic cross-arm dedupe — every sampled
            row is preserved. Deduplicate protein-sourced windows on ``entry_win``
            and synthetic windows on ``window`` if needed.

        Notes
        -----
        ``role`` and ``strategy`` tags set by each arm are preserved through the
        concatenation; together with ``arm`` they carry full row provenance. A
        ``motif_matched`` arm adds a ``motif_score`` column, which is ``NaN`` for
        rows from other arms.

        Examples
        --------
        .. include:: examples/aws_sample_benchmark_set.rst
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        check_arms(arms)
        if seed is not None:
            ut.check_number_range(name="seed", val=seed, min_val=0, just_int=True)
        # Derive deterministic per-arm sub-seeds from the effective master seed.
        arm_names = list(arms.keys())
        master = seed if seed is not None else self._random_state
        if master is not None:
            sub_seeds = [int(s) for s in
                         np.random.SeedSequence(master).generate_state(len(arm_names))]
        else:
            sub_seeds = [None] * len(arm_names)
        dispatch = {
            ut.STRATEGY_SAME: self.sample_same_protein,
            ut.STRATEGY_DIFF: self.sample_different_protein,
            ut.STRATEGY_SYNTH_PREFIX: self.sample_synthetic,
            ut.STRATEGY_MOTIF_MATCHED: self.sample_motif_matched,
        }
        # Run each arm and tag it.
        frames = []
        for arm_name, sub_seed in zip(arm_names, sub_seeds):
            spec = dict(arms[arm_name])
            method = spec.pop("method")
            df_arm = dispatch[method](df_seq=df_seq, seed=sub_seed, **spec).copy()
            df_arm[ut.COL_ARM] = arm_name
            frames.append(df_arm)
        return pd.concat(frames, ignore_index=True)

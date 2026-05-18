"""
This is a script for a wrapper around the FIMO CLI (MEME suite). It returns
the same ``df_seq``-shaped output as
:meth:`aaanalysis.AAWindowSampler.sample_motif_matched`, with strict parity
on the returned hit set.

Strategy: run ``fimo --thresh 1.0`` to enumerate every motif occurrence in
the candidate rows, then re-score each occurrence in Python with the raw
PWM-sum used by the in-memory method. The user's ``motif_score_threshold``
is then applied to the Python-side score, so both paths use the same scoring
formula and the same filter.

Parity contract: same ``df_seq`` + same ``motif_pwm`` + same
``motif_score_threshold`` ⇒ same set of ``(entry, source_position)`` and
identical ``motif_score`` values. ``AAWindowSampler``'s class-level
``max_similarity_to_test`` / ``max_similarity_within_ref`` filters are *not*
applied by the wrapper (it has no class state); for parity, leave those
defaults.
"""
from typing import Optional, List, Union
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from aaanalysis.data_handling._aa_window_sampler import (check_motif_args,
                                                          check_context_cols)
from aaanalysis.data_handling._backend.aa_window_sampler._utils import (
    parse_pos_col, score_window_pwm_, window_offsets)
from aaanalysis.data_handling._backend.aa_window_sampler.build_output import (
    build_segments_output, build_sequences_output)


# I Helper Functions
def check_fimo_installed():
    """Raise ``RuntimeError`` if the ``fimo`` binary is not on PATH."""
    if not shutil.which("fimo"):
        raise RuntimeError(
            "'fimo' is not installed or not in PATH. Install the MEME suite "
            "(e.g. `conda install -c bioconda meme`) to use this wrapper. "
            "For the pure-Python equivalent see "
            "`aaanalysis.AAWindowSampler.sample_motif_matched`."
        )

def _pwm_to_meme(motif_pwm, motif_name="aaws_motif"):
    """Render a PWM as a MEME-format text block.

    The input PWM has columns ordered by ``ut.LIST_CANONICAL_AA`` (the package
    convention). MEME's built-in protein alphabet expects strict alphabetic
    order (``ut.STR_MEME_PROTEIN_ALPHABET``); the columns are therefore
    remapped before serialization. Rows are renormalized to sum to 1.0; users
    supplying log-odds matrices should convert beforehand.
    """
    pwm = np.asarray(motif_pwm, dtype=float)
    pwm = np.clip(pwm, 0.0, None)
    row_sums = pwm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    probs = pwm / row_sums
    # Remap from ut.LIST_CANONICAL_AA order → MEME's alphabetic order.
    src_idx = {aa: i for i, aa in enumerate(ut.LIST_CANONICAL_AA)}
    permutation = [src_idx[aa] for aa in sorted(ut.LIST_CANONICAL_AA)]
    probs_meme = probs[:, permutation]
    width = probs_meme.shape[0]
    n_aa = probs_meme.shape[1]
    lines = ["MEME version 4", "",
             "strands: +", "",
             f"MOTIF {motif_name}", "",
             f"letter-probability matrix: alength= {n_aa} w= {width} "
             f"nsites= 1 E= 0"]
    for row in probs_meme:
        lines.append(" " + " ".join(f"{v:.6f}" for v in row))
    return "\n".join(lines) + "\n"


def _df_seq_to_fasta(df_seq, out_path):
    """Write the sequences in ``df_seq`` to ``out_path`` as a minimal FASTA."""
    with open(out_path, "w") as fh:
        for entry, seq in zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ]):
            fh.write(f">{entry}\n{seq}\n")


def _run_fimo(motif_path, fasta_path, *,
               max_stored_scores=None, bg_file=None, motif_pseudo=None):
    """Invoke FIMO with a permissive p-value threshold and return all parsed hits.

    The wrapper later re-scores each returned position with the raw PWM-sum
    used by :meth:`AAWindowSampler.sample_motif_matched`, so the FIMO score
    itself is discarded; we only use FIMO as a position scanner. Running with
    ``--thresh 1.0`` ensures every motif occurrence is reported (p-values are
    always in [0, 1]).

    The ``--text``, ``--thresh 1.0``, and ``--no-qvalue`` flags are
    parity-critical and are always set by the wrapper. The optional
    ``max_stored_scores`` / ``bg_file`` / ``motif_pseudo`` arguments map to
    the corresponding FIMO flags; see :func:`scan_motif` for the user-facing
    surface.

    Returns
    -------
    list of dict
        ``{entry, start, stop, fimo_score, matched_sequence}``; positions are
        1-based as emitted by FIMO.
    """
    cmd = ["fimo", "--text", "--thresh", "1.0", "--no-qvalue"]
    if max_stored_scores is not None:
        cmd.extend(["--max-stored-scores", str(int(max_stored_scores))])
    if bg_file is not None:
        cmd.extend(["--bgfile", str(bg_file)])
    if motif_pseudo is not None:
        cmd.extend(["--motif-pseudo", str(float(motif_pseudo))])
    cmd.extend([str(motif_path), str(fasta_path)])
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    hits = []
    for line in proc.stdout.splitlines():
        if not line or line.startswith("#") or line.startswith("motif_id"):
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        try:
            start = int(parts[3])
            stop = int(parts[4])
            fimo_score = float(parts[6])
        except (ValueError, IndexError):
            continue
        hits.append({
            "entry": parts[2],
            "start": start,
            "stop": stop,
            "fimo_score": fimo_score,
            "matched_sequence": parts[-1],
        })
    return hits


# II Main Function
def scan_motif(df_seq: pd.DataFrame = None,
                                  pos_col: str = "pos",
                                  n: int = 100,
                                  window_size: int = 9,
                                  motif_pwm: Optional[np.ndarray] = None,
                                  motif_score_threshold: Optional[float] = None,
                                  label_test: Union[int, float] = 1,
                                  label_ref: Union[int, float] = 0,
                                  role: str = ut.ROLE_NEG,
                                  output_mode: str = ut.OUT_SEGMENTS,
                                  context_cols: Optional[List[str]] = None,
                                  max_stored_scores: Optional[int] = None,
                                  bg_file: Optional[Union[str, Path]] = None,
                                  motif_pseudo: Optional[float] = None,
                                  ) -> pd.DataFrame:
    """Wrapper around the FIMO CLI [Bailey09]_, [Grant11]_; returns the same hits as
    :meth:`AAWindowSampler.sample_motif_matched`.

    Raises ``RuntimeError`` if ``fimo`` is not on PATH. The output schema
    matches :meth:`AAWindowSampler.sample_motif_matched` (including the
    ``motif_score`` column). Strict parity: same input ⇒ same hit set ⇒ same
    ``motif_score`` values; see module docstring for the implementation
    technique.

    Parameters
    ----------
    df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
        DataFrame containing an ``entry`` column with unique protein
        identifiers and a ``sequence`` column with full protein sequences.
        Rows are split into *positive* and *candidate* rows by ``pos_col``;
        candidate rows feed the FIMO scan.
    pos_col : str, default='pos'
        Column with per-row 1-based positive positions; non-empty cells mark
        positive rows (excluded from the scan), empty / ``None`` / ``NaN``
        cells mark candidate rows.
    n : int, default=100
        Maximum number of motif-matched windows to return.
    window_size : int, default=9
        Window length; must equal the first dimension of ``motif_pwm``.
    motif_pwm : np.ndarray
        Position-weight matrix of shape ``(window_size, 20)``, columns ordered
        by ``ut.LIST_CANONICAL_AA``. Required.
    motif_score_threshold : float
        Score threshold (sum of per-position PWM values). Required.
    label_test : int or float, default=1
        Label assigned to positives in ``output_mode='sequences'``.
    label_ref : int or float, default=0
        Label assigned to sampled motif-matched rows.
    role : str, default='Negative'
        Role tag stored in the output's ``role`` column.
    output_mode : {'segments', 'sequences'}, default='segments'
        Output schema; same as :meth:`AAWindowSampler.sample_motif_matched`.
    context_cols : list of str, optional
        Extra ``df_seq`` columns to copy through to the output (provenance).
    max_stored_scores : int, optional
        Maximum number of motif occurrences FIMO may store internally before
        truncating. FIMO's default is 100 000; raise this only when scanning
        very large candidate sets and FIMO reports truncation.
    bg_file : str or pathlib.Path, optional
        Path to a MEME-format background amino-acid frequency file. When
        omitted, FIMO uses its built-in protein background.
    motif_pseudo : float, optional
        Pseudocount applied to the motif before scanning (FIMO's default is
        ``0.1``). Pass ``0.0`` to disable smoothing.

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    * Candidate sequences are written to a temporary FASTA and passed to
      ``fimo`` via ``subprocess``.
    * The PWM is written in MEME letter-probability format (column order
      remapped to ``ut.STR_MEME_PROTEIN_ALPHABET``) and ``fimo`` runs in
      ``--text`` mode with ``--thresh 1.0`` so every motif occurrence is
      reported.
    * Each FIMO hit is re-scored with the raw PWM-sum used by
      :meth:`AAWindowSampler.sample_motif_matched`; only positions with
      ``score >= motif_score_threshold`` are kept.
    * Surviving hits are ranked by descending score (deterministic tiebreak
      by ``entry`` then 0-based center) and capped at ``n``.
    * Protein-only: this wrapper passes the 20 canonical amino acids to MEME
      as the alphabet; gapped or non-protein alphabets are not supported.

    The wrapper sets ``--text``, ``--thresh 1.0``, and ``--no-qvalue``
    unconditionally because they are required for the parity contract with
    :meth:`AAWindowSampler.sample_motif_matched`. The other AAanalysis
    parameters above are passed through to FIMO as follows:

    ===================  ======================
    AAanalysis parameter  FIMO flag
    ===================  ======================
    ``max_stored_scores`` ``--max-stored-scores``
    ``bg_file``           ``--bgfile``
    ``motif_pseudo``      ``--motif-pseudo``
    ===================  ======================

    See Also
    --------
    * MEME Suite `documentation <https://meme-suite.org/meme/doc/overview.html>`__
      and FIMO `manual <https://meme-suite.org/meme/doc/fimo.html>`__.
    * :meth:`AAWindowSampler.sample_motif_matched` for the pure-Python
      equivalent (no FIMO binary required).
    """
    check_fimo_installed()
    # Validate
    ut.check_df_seq(df_seq=df_seq)
    ut.check_str(name="pos_col", val=pos_col, accept_none=False)
    ut.check_number_range(name="n", val=n, min_val=1, just_int=True)
    ut.check_number_range(name="window_size", val=window_size, min_val=1, just_int=True)
    ut.check_str(name="role", val=role, accept_none=False)
    ut.check_str_options(name="output_mode", val=output_mode,
                         list_str_options=ut.LIST_OUTPUT_MODES)
    if motif_pwm is None:
        raise ValueError("'motif_pwm' is required for scan_motif.")
    if motif_score_threshold is None:
        raise ValueError("'motif_score_threshold' is required for "
                         "scan_motif.")
    # `motif_match` is not exposed (scan-only mode); pass None.
    motif_pwm = check_motif_args(motif_pwm, motif_score_threshold,
                                  None, window_size)
    if max_stored_scores is not None:
        ut.check_number_range(name="max_stored_scores", val=max_stored_scores,
                              min_val=1, just_int=True)
    if motif_pseudo is not None:
        ut.check_number_range(name="motif_pseudo", val=motif_pseudo,
                              min_val=0, just_int=False)
    if bg_file is not None:
        bg_path = Path(bg_file)
        if not bg_path.is_file():
            raise ValueError(f"'bg_file' ({bg_file!r}) should be a path to "
                             f"an existing file")
    reserved = ut.COLS_SEGMENTS if output_mode == ut.OUT_SEGMENTS else ut.COLS_SEQUENCES
    context_cols = check_context_cols(df_seq, context_cols, reserved)
    # Identify candidate rows
    positions = parse_pos_col(df_seq, pos_col)
    eligible_idx = [i for i, p in enumerate(positions) if not p]
    if not eligible_idx:
        raise ValueError("No eligible candidate proteins (rows with no test "
                         "positions) for scan_motif.")
    candidate_df = df_seq.iloc[eligible_idx].reset_index(drop=True)
    # Run FIMO with a permissive threshold; re-score every reported position
    # in Python so the wrapper returns the same hits as sample_motif_matched.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fasta_path = tmp / "candidates.fasta"
        motif_path = tmp / "motif.meme"
        _df_seq_to_fasta(candidate_df, fasta_path)
        motif_path.write_text(_pwm_to_meme(motif_pwm))
        raw_hits = _run_fimo(motif_path, fasta_path,
                              max_stored_scores=max_stored_scores,
                              bg_file=bg_file,
                              motif_pseudo=motif_pseudo)
    # Re-score with the raw PWM-sum and filter by the user's score threshold.
    seq_by_entry = {df_seq.iloc[i][ut.COL_ENTRY]: df_seq.iloc[i][ut.COL_SEQ]
                    for i in eligible_idx}
    hits = []
    for h in raw_hits:
        seq = seq_by_entry.get(h["entry"])
        if seq is None:
            continue
        start_0 = h["start"] - 1  # FIMO reports 1-based start
        window = seq[start_0:start_0 + window_size]
        if len(window) != window_size:
            continue
        score = score_window_pwm_(window, motif_pwm)
        if score >= motif_score_threshold:
            h_copy = dict(h)
            h_copy["score"] = score
            hits.append(h_copy)
    if not hits:
        if output_mode == ut.OUT_SEGMENTS:
            df_out = build_segments_output(
                [], strategy=ut.STRATEGY_MOTIF_MATCHED, role=role,
                label_value=label_ref, df_seq=df_seq,
                window_size=window_size,
                source_indices=None, context_cols=context_cols)
            df_out["motif_score"] = []
            return df_out
        return build_sequences_output(df_seq, positions,
                                       [[] for _ in range(len(df_seq))],
                                       label_test=label_test, label_ref=label_ref,
                                       mark_test=True, context_cols=context_cols)
    # Rank, cap at n, build output schema
    hits.sort(key=lambda h: (-h["score"], h["entry"], h["start"]))
    hits = hits[:n]
    entry_to_idx = {df_seq.iloc[i][ut.COL_ENTRY]: i for i in eligible_idx}
    half_left, _ = window_offsets(window_size)
    rows, source_indices, sampled_scores = [], [], []
    sampled_centers = [[] for _ in range(len(df_seq))]
    for h in hits:
        entry = h["entry"]
        if entry not in entry_to_idx:
            continue
        i = entry_to_idx[entry]
        # Convert FIMO 1-based start to 0-based P1-anchor center.
        center = (h["start"] - 1) + half_left
        seq_i = df_seq.iloc[i][ut.COL_SEQ]
        window = seq_i[center - half_left:center - half_left + window_size]
        rows.append([entry, seq_i, window, center + 1])
        source_indices.append(i)
        sampled_centers[i].append(center)
        sampled_scores.append(h["score"])
    if output_mode == ut.OUT_SEGMENTS:
        df_out = build_segments_output(rows, strategy=ut.STRATEGY_MOTIF_MATCHED,
                                        role=role, label_value=label_ref,
                                        df_seq=df_seq,
                                        window_size=window_size,
                                        source_indices=source_indices,
                                        context_cols=context_cols)
        df_out["motif_score"] = sampled_scores
        return df_out
    return build_sequences_output(df_seq, positions, sampled_centers,
                                   label_test=label_test, label_ref=label_ref,
                                   mark_test=True, context_cols=context_cols)

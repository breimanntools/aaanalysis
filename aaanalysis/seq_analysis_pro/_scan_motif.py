"""
This is a script for the frontend of the scan_motif function, a wrapper
around the FIMO CLI (MEME suite) that mirrors
:meth:`aaanalysis.AAWindowSampler.sample_motif_matched`.
"""
from typing import Optional, Union
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from aaanalysis.seq_analysis._aa_window_sampler import check_pwm
from aaanalysis.seq_analysis._backend.aa_window_sampler._utils import (
    parse_pos_col, window_offsets)
from aaanalysis.seq_analysis._backend.aa_window_sampler.build_output import (
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


def _run_fimo(motif_path, fasta_path, *, pvalue_threshold,
              max_stored_scores=None, bg_file=None, motif_pseudo=None):
    """Invoke FIMO at ``pvalue_threshold`` and return its significant hits.

    FIMO performs the match selection itself: ``--thresh <pvalue_threshold>``
    reports only occurrences whose match p-value (computed against the
    background model from the motif's letter probabilities) is below the
    threshold. Both FIMO's ``score`` (log-odds) and ``p-value`` are kept and
    surfaced by :func:`scan_motif`; unlike the pure-Python
    :meth:`AAWindowSampler.sample_motif_matched`, no PWM-sum re-scoring is
    applied.

    ``--text`` streams the hit table (no q-values are computed in this mode).
    The optional ``max_stored_scores`` / ``bg_file`` / ``motif_pseudo``
    arguments map to the corresponding FIMO flags and genuinely affect the
    reported hits; see :func:`scan_motif` for the user-facing surface.

    Returns
    -------
    list of dict
        ``{entry, start, stop, fimo_score, p_value, matched_sequence}``;
        positions are 1-based as emitted by FIMO.
    """
    cmd = ["fimo", "--text", "--thresh", str(float(pvalue_threshold)),
           "--no-qvalue"]
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
            p_value = float(parts[7])
        except (ValueError, IndexError):
            continue
        hits.append({
            "entry": parts[2],
            "start": start,
            "stop": stop,
            "fimo_score": fimo_score,
            "p_value": p_value,
            "matched_sequence": parts[-1],
        })
    return hits


# II Main Function
def scan_motif(df_seq: pd.DataFrame = None,
                                  pos_col: str = "pos",
                                  n: int = 100,
                                  window_size: int = 9,
                                  motif_pwm: pd.DataFrame = None,
                                  pvalue_threshold: float = 1e-4,
                                  label_test: Union[int, float] = 1,
                                  label_ref: Union[int, float] = 0,
                                  role: str = ut.ROLE_NEG,
                                  output_mode: str = ut.OUT_SEGMENTS,
                                  max_stored_scores: Optional[int] = None,
                                  bg_file: Optional[Union[str, Path]] = None,
                                  motif_pseudo: Optional[float] = None,
                                  ) -> pd.DataFrame:
    """
    Scan candidate proteins for statistically significant occurrences of a PWM using the FIMO CLI.

    CLI wrapper around FIMO [Bailey09]_, [Grant11]_ from the MEME suite. Unlike
    the pure-Python :meth:`AAWindowSampler.sample_motif_matched` (which keeps
    windows whose raw per-position PWM-sum is ``>= motif_score_threshold``),
    ``scan_motif`` lets FIMO perform its own probabilistic matching: each window
    is scored against the background model implied by the PWM and kept only when
    its match p-value is below ``pvalue_threshold``. The two thus select
    *different* windows and are complementary ways to mine motif-matched training
    data. The output schema matches :meth:`AAWindowSampler.sample_motif_matched`,
    with ``motif_score`` holding FIMO's log-odds score and an added ``p_value``
    column (segments mode).

    .. versionadded:: 1.1.0

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
    motif_pwm : pd.DataFrame
        Position-weight matrix of shape ``(window_size, 20)`` whose columns are
        the 20 canonical AA letters in any order (reindexed internally to
        ``ut.LIST_CANONICAL_AA``). Required.
    pvalue_threshold : float, default=1e-4
        FIMO match-p-value cutoff (maps to ``fimo --thresh``); only occurrences
        with a match p-value below this are reported. Smaller is stricter.
    label_test : int or float, default=1
        Label assigned to positives in ``output_mode='sequences'``.
    label_ref : int or float, default=0
        Label assigned to sampled motif-matched rows.
    role : str, default='Negative'
        Role tag stored in the output's ``role`` column.
    output_mode : {'segments', 'sequences'}, default='segments'
        Output schema; same as :meth:`AAWindowSampler.sample_motif_matched`.
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
    df_hits : pd.DataFrame
        Significant motif hits, one row per matched window, ranked by ascending
        match p-value. In ``output_mode='segments'`` the schema of
        :meth:`AAWindowSampler.sample_motif_matched` is extended with FIMO's
        ``motif_score`` and ``p_value`` columns.

    Raises
    ------
    RuntimeError
        If the ``fimo`` binary is not on PATH.
    ValueError
        If ``motif_pwm`` is not provided, if ``bg_file`` is set but does not
        point to an existing file, or if ``df_seq`` contains no eligible
        candidate proteins (rows without test positions).

    Notes
    -----
    * Candidate sequences are written to a temporary FASTA and the PWM to a
      temporary MEME letter-probability file (column order remapped to
      ``ut.STR_MEME_PROTEIN_ALPHABET``); ``fimo`` runs in ``--text`` mode.
    * FIMO selects and scores the hits at ``--thresh pvalue_threshold``;
      AAanalysis only ranks them by ascending p-value (deterministic tiebreak
      by descending score, then ``entry`` and start) and caps at ``n``.
      ``motif_score`` is FIMO's log-odds score, not a PWM-sum.
    * ``max_stored_scores`` / ``bg_file`` / ``motif_pseudo`` map to the FIMO
      flags ``--max-stored-scores`` / ``--bgfile`` / ``--motif-pseudo`` and
      genuinely change the reported hits (they tune FIMO's scoring).
    * Protein-only: the 20 canonical amino acids are passed to MEME as the
      alphabet; gapped or non-protein alphabets are not supported.

    See Also
    --------
    * MEME Suite `documentation <https://meme-suite.org/meme/doc/overview.html>`__
      and FIMO `manual <https://meme-suite.org/meme/doc/fimo.html>`__.
    * :meth:`AAWindowSampler.sample_motif_matched` for the pure-Python PWM-sum
      sampler (no FIMO binary required) that selects windows by a different
      criterion.

    Examples
    --------
    .. include:: examples/scan_motif.rst
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
    ut.check_number_range(name="pvalue_threshold", val=pvalue_threshold,
                          min_val=0, max_val=1, accept_none=False,
                          just_int=False)
    motif_pwm = check_pwm(motif_pwm, window_size)
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
    # Identify candidate rows
    positions = parse_pos_col(df_seq, pos_col)
    eligible_idx = [i for i, p in enumerate(positions) if not p]
    if not eligible_idx:
        raise ValueError("No eligible candidate proteins (rows with no test "
                         "positions) for scan_motif.")
    candidate_df = df_seq.iloc[eligible_idx].reset_index(drop=True)
    # Let FIMO select and score the significant hits directly (no re-scoring).
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fasta_path = tmp / "candidates.fasta"
        motif_path = tmp / "motif.meme"
        _df_seq_to_fasta(candidate_df, fasta_path)
        motif_path.write_text(_pwm_to_meme(motif_pwm))
        raw_hits = _run_fimo(motif_path, fasta_path,
                             pvalue_threshold=pvalue_threshold,
                             max_stored_scores=max_stored_scores,
                             bg_file=bg_file,
                             motif_pseudo=motif_pseudo)
    # Keep only hits whose window fits fully inside the candidate sequence.
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
        hits.append(h)
    if not hits:
        if output_mode == ut.OUT_SEGMENTS:
            df_out = build_segments_output(
                [], strategy=ut.STRATEGY_MOTIF_MATCHED, role=role,
                label_value=label_ref, window_size=window_size)
            df_out["motif_score"] = []
            df_out["p_value"] = []
            return df_out
        return build_sequences_output(df_seq, positions,
                                       [[] for _ in range(len(df_seq))],
                                       label_test=label_test, label_ref=label_ref,
                                       mark_test=True)
    # Rank by ascending p-value (most significant first); cap at n.
    hits.sort(key=lambda h: (h["p_value"], -h["fimo_score"], h["entry"],
                             h["start"]))
    hits = hits[:n]
    entry_to_idx = {df_seq.iloc[i][ut.COL_ENTRY]: i for i in eligible_idx}
    half_left, _ = window_offsets(window_size)
    rows, sampled_scores, sampled_pvalues = [], [], []
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
        sampled_centers[i].append(center)
        sampled_scores.append(h["fimo_score"])
        sampled_pvalues.append(h["p_value"])
    if output_mode == ut.OUT_SEGMENTS:
        df_out = build_segments_output(rows, strategy=ut.STRATEGY_MOTIF_MATCHED,
                                        role=role, label_value=label_ref,
                                        window_size=window_size)
        df_out["motif_score"] = sampled_scores
        df_out["p_value"] = sampled_pvalues
        return df_out
    return build_sequences_output(df_seq, positions, sampled_centers,
                                   label_test=label_test, label_ref=label_ref,
                                   mark_test=True)

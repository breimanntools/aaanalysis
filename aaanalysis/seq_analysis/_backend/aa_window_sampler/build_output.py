"""
This is a script for assembling AAWindowSampler outputs in 'segments' and 'sequences' modes.
"""
import pandas as pd

import aaanalysis.utils as ut

from ._utils import window_offsets


# I Helper Functions
# (no helpers — constants and main builders only)


# II Main Functions
def build_segments_output(rows, *, strategy, role, label_value, window_size):
    """Build the segments-mode DataFrame.

    Parameters
    ----------
    rows : list of [entry, sequence, window, source_position]
        Row tuples produced by the sampling backends. ``source_position`` is the
        1-based P1 anchor of the window.
    strategy : str
    role : str
    label_value : int or float
    window_size : int
        Used to derive the inclusive ``[start_pos, end_pos]`` span for
        ``entry_win``.

    Notes
    -----
    ``entry_win`` is constructed as ``<entry>_<start_pos>-<end_pos>`` with
    1-based inclusive coordinates derived from ``source_position`` and
    ``window_size``. The same biological window across calls produces the same
    ``entry_win`` — making ``drop_duplicates(subset="entry_win")`` the natural
    cross-call dedupe primitive. Synthetic outputs do not call this builder;
    they construct their own DataFrame with ``entry_win="synth_{i}"``.
    """
    df = pd.DataFrame(rows, columns=[ut.COL_ENTRY, ut.COL_SEQ,
                                      ut.COL_WINDOW, ut.COL_SOURCE_POS])
    df[ut.COL_LABEL] = label_value
    df[ut.COL_ROLE] = role
    df[ut.COL_STRATEGY] = strategy
    half_left, _ = window_offsets(window_size)
    start = df[ut.COL_SOURCE_POS] - half_left
    end = df[ut.COL_SOURCE_POS] + (window_size - half_left) - 1
    df[ut.COL_ENTRY_WIN] = (df[ut.COL_ENTRY].astype(str) + "_"
                            + start.astype(str) + "-" + end.astype(str))
    return df[ut.COLS_SEGMENTS].copy()


def build_sequences_output(df_seq, positions, sampled_centers, *,
                             label_test, label_ref, mark_test):
    """Build the sequences-mode DataFrame.

    For each row of ``df_seq``, build a ``labels`` list of length ``len(sequence)``:

    - ``label_test`` at known test positions (only if ``mark_test`` is ``True``)
    - ``label_ref`` at sampled centers
    - ``None`` elsewhere
    """
    labels_per_entry = []
    for seq, pos_list, centers in zip(df_seq[ut.COL_SEQ], positions, sampled_centers):
        labels = [None] * len(seq)
        if mark_test:
            for p in pos_list:
                if 1 <= p <= len(seq):
                    labels[p - 1] = label_test
        for c in centers:
            if 0 <= c < len(seq):
                labels[c] = label_ref
        labels_per_entry.append(labels)
    df = pd.DataFrame({
        ut.COL_ENTRY: df_seq[ut.COL_ENTRY].tolist(),
        ut.COL_SEQ: df_seq[ut.COL_SEQ].tolist(),
        ut.COL_LABELS: labels_per_entry,
    })
    return df[ut.COLS_SEQUENCES].copy()

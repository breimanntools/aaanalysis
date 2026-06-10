"""
This is a script for the backend of the SeqMut class: the CPP-guided ΔCPP mutation engine.

The engine is deterministic and model-free. For a set of point mutations it rebuilds the
sequence parts, recomputes the CPP feature matrix ``X`` via the SequenceFeature builder, and
returns the per-feature change ``ΔX = X_mut - X_wt`` aggregated into a scalar ``delta_cpp``
(Sum|ΔX|) and a signed ``shift_score`` toward the test-class profile.
"""
import numpy as np
import pandas as pd

import aaanalysis.utils as ut


# I Helper Functions
def get_needed_parts(features=None):
    """Return the sorted unique lower-case parts referenced by a list of feature ids."""
    parts = sorted({ut.split_feat_id(feat_id=f)[0].lower() for f in features})
    return parts


def classify_region(pos=None, tmd_start=None, tmd_stop=None):
    """Classify a 1-based position into 'jmd_n' / 'tmd' / 'jmd_c'."""
    if pos < tmd_start:
        return ut.COL_JMD_N
    if pos > tmd_stop:
        return ut.COL_JMD_C
    return ut.COL_TMD


def get_region_positions(region=None, tmd_start=None, tmd_stop=None, jmd_n_len=10,
                         jmd_c_len=10, seq_len=None):
    """Return the sorted 1-based positions to scan for one sequence.

    ``region=None`` covers the full jmd_n + tmd + jmd_c span (every position a standard CPP
    feature can touch). A part name restricts to that part; a list restricts to those positions.
    """
    lo_full = max(1, tmd_start - jmd_n_len)
    hi_full = min(seq_len, tmd_stop + jmd_c_len)
    if region is None:
        positions = range(lo_full, hi_full + 1)
    elif isinstance(region, str):
        part = region.lower()
        if part == ut.COL_TMD:
            positions = range(tmd_start, tmd_stop + 1)
        elif part == ut.COL_JMD_N:
            positions = range(lo_full, tmd_start)
        elif part == ut.COL_JMD_C:
            positions = range(tmd_stop + 1, hi_full + 1)
        else:
            positions = range(lo_full, hi_full + 1)
    else:
        positions = [int(p) for p in region]
    return [p for p in positions if 1 <= p <= seq_len]


def build_scan_plan(df_seq=None, region=None, to_aa=None, jmd_n_len=10, jmd_c_len=10):
    """Enumerate every (entry, pos, from_aa, to_aa) point mutation for a scan."""
    rows = []
    for _, r in df_seq.iterrows():
        entry, seq = r[ut.COL_ENTRY], r[ut.COL_SEQ]
        ts, te = int(r[ut.COL_TMD_START]), int(r[ut.COL_TMD_STOP])
        positions = get_region_positions(region=region, tmd_start=ts, tmd_stop=te,
                                         jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, seq_len=len(seq))
        for pos in positions:
            from_aa = seq[pos - 1]
            for aa in to_aa:
                if aa == from_aa:
                    continue
                rows.append((entry, pos, from_aa, aa, ts, te,
                             classify_region(pos=pos, tmd_start=ts, tmd_stop=te)))
    cols = [ut.COL_ENTRY, ut.COL_POS, ut.COL_FROM_AA, ut.COL_TO_AA,
            ut.COL_TMD_START, ut.COL_TMD_STOP, ut.COL_REGION]
    return pd.DataFrame(rows, columns=cols)


def _apply_point_mutation(seq=None, pos=None, to_aa=None):
    """Return ``seq`` with the residue at 1-based ``pos`` replaced by ``to_aa``."""
    return seq[:pos - 1] + to_aa + seq[pos:]


def comp_delta_x(df_plan=None, df_seq=None, features=None, df_scales=None, sf=None,
                 jmd_n_len=10, jmd_c_len=10):
    """Compute the per-feature ΔX matrix for every mutation row in ``df_plan``.

    Two heavy calls total (wild-type once, all mutants once), regardless of how many
    mutations are scanned. Returns ``dX`` of shape ``(len(df_plan), n_features)``.
    """
    list_parts = get_needed_parts(features=features)
    # Wild-type feature matrix, one row per entry.
    df_seq_wt = df_seq[[ut.COL_ENTRY, ut.COL_SEQ, ut.COL_TMD_START, ut.COL_TMD_STOP]].copy()
    df_parts_wt = sf.get_df_parts(df_seq=df_seq_wt, list_parts=list_parts,
                                  jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    X_wt = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts_wt,
                                        df_scales=df_scales), dtype=float)
    entry_to_row = {e: i for i, e in enumerate(df_seq_wt[ut.COL_ENTRY])}
    seq_by_entry = dict(zip(df_seq_wt[ut.COL_ENTRY], df_seq_wt[ut.COL_SEQ]))
    # Mutant feature matrix: build a df_seq with unique synthetic entries (parts depend only on
    # sequence + tmd coordinates, so a renamed entry is safe and preserves row order).
    mut_seqs = [_apply_point_mutation(seq=seq_by_entry[e], pos=int(p), to_aa=a)
                for e, p, a in zip(df_plan[ut.COL_ENTRY], df_plan[ut.COL_POS], df_plan[ut.COL_TO_AA])]
    df_seq_mut = pd.DataFrame({
        ut.COL_ENTRY: [f"{e}__{i}" for i, e in enumerate(df_plan[ut.COL_ENTRY])],
        ut.COL_SEQ: mut_seqs,
        ut.COL_TMD_START: df_plan[ut.COL_TMD_START].to_numpy(),
        ut.COL_TMD_STOP: df_plan[ut.COL_TMD_STOP].to_numpy(),
    })
    df_parts_mut = sf.get_df_parts(df_seq=df_seq_mut, list_parts=list_parts,
                                   jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    X_mut = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts_mut,
                                         df_scales=df_scales), dtype=float)
    wt_rows = np.asarray([entry_to_row[e] for e in df_plan[ut.COL_ENTRY]], dtype=int)
    dX = X_mut - X_wt[wt_rows]
    return dX


def comp_scan_scores(dX=None, mean_dif=None, weight_vec=None):
    """Aggregate a ΔX matrix into per-mutation ``delta_cpp`` (Sum|ΔX|) and ``shift_score``."""
    delta_cpp = np.abs(dX).sum(axis=1)
    direction = np.sign(np.asarray(mean_dif, dtype=float))
    if weight_vec is not None:
        direction = direction * np.asarray(weight_vec, dtype=float)
    shift_score = dX @ direction
    return delta_cpp, shift_score


def build_scan_output(df_plan=None, delta_cpp=None, shift_score=None):
    """Assemble the tidy scan output DataFrame, sorted by descending |ΔCPP|."""
    df_out = df_plan[[ut.COL_ENTRY, ut.COL_POS, ut.COL_FROM_AA, ut.COL_TO_AA, ut.COL_REGION]].copy()
    df_out[ut.COL_MUTATION] = [f"{f}{int(p)}{t}" for f, p, t in
                               zip(df_out[ut.COL_FROM_AA], df_out[ut.COL_POS], df_out[ut.COL_TO_AA])]
    df_out[ut.COL_DELTA_CPP] = delta_cpp
    df_out[ut.COL_SHIFT_SCORE] = shift_score
    df_out = df_out[ut.COLS_SEQMUT_SCAN]
    df_out = df_out.sort_values(ut.COL_DELTA_CPP, ascending=False).reset_index(drop=True)
    return df_out


def eval_disruptive(df_scan=None, th=None):
    """Tag mutations stable/disruptive by a |ΔCPP| threshold and aggregate per entry+region."""
    df = df_scan.copy()
    if th is None:
        th = float(np.quantile(df[ut.COL_DELTA_CPP], 2 / 3)) if len(df) else 0.0
    df[ut.COL_IS_DISRUPTIVE] = df[ut.COL_DELTA_CPP] >= th
    grouped = df.groupby([ut.COL_ENTRY, ut.COL_REGION], sort=False)
    df_eval = grouped.agg(
        **{ut.COL_N_MUT: (ut.COL_DELTA_CPP, "size"),
           ut.COL_N_DISRUPTIVE: (ut.COL_IS_DISRUPTIVE, "sum"),
           ut.COL_MEAN_DELTA_CPP: (ut.COL_DELTA_CPP, "mean")}).reset_index()
    df_eval[ut.COL_FRAC_DISRUPTIVE] = df_eval[ut.COL_N_DISRUPTIVE] / df_eval[ut.COL_N_MUT]
    df_eval = df_eval[ut.COLS_SEQMUT_EVAL]
    return df_eval, th

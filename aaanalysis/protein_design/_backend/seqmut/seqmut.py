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


def comp_feature_matrices(df_plan=None, df_seq=None, features=None, df_scales=None, sf=None,
                          jmd_n_len=10, jmd_c_len=10):
    """Build the wild-type and mutant CPP feature matrices for a mutation plan.

    Two heavy calls total (wild-type once, all mutants once), regardless of how many
    mutations are scanned. Returns ``(X_wt, X_mut, wt_rows)`` where ``X_wt`` has one row
    per ``df_seq`` entry, ``X_mut`` one row per ``df_plan`` mutation, and ``wt_rows`` maps
    each mutation row to its wild-type row in ``X_wt``.
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
    return X_wt, X_mut, wt_rows


def comp_delta_x(df_plan=None, df_seq=None, features=None, df_scales=None, sf=None,
                 jmd_n_len=10, jmd_c_len=10):
    """Compute the per-feature ΔX matrix (``X_mut - X_wt``) for every mutation row."""
    X_wt, X_mut, wt_rows = comp_feature_matrices(
        df_plan=df_plan, df_seq=df_seq, features=features, df_scales=df_scales, sf=sf,
        jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    return X_mut - X_wt[wt_rows]


def predict_target_scores(model=None, X=None, target_class=None):
    """Return ``(pred, pred_std)`` for the target class over the rows of ``X``.

    Duck-typed on ``predict_proba``. A model that returns a ``(pred, pred_std)`` tuple
    (e.g. ``TreeModel``, positive class) is used directly; a standard sklearn classifier
    returning a 2-D probability matrix is indexed at the target-class column (``pred_std``
    is then ``None``). ``pred`` / ``pred_std`` are probabilities in ``[0, 1]``.
    """
    out = model.predict_proba(X)
    if isinstance(out, tuple):
        pred, pred_std = out
        return np.asarray(pred, dtype=float), np.asarray(pred_std, dtype=float)
    proba = np.asarray(out, dtype=float)
    if proba.ndim == 1:
        return proba, None
    col = resolve_target_col(model=model, target_class=target_class, n_classes=proba.shape[1])
    return proba[:, col], None


def resolve_target_col(model=None, target_class=None, n_classes=None):
    """Resolve ``target_class`` to a probability-matrix column index.

    ``None`` selects the positive class (the last column). A class *label* is mapped via
    ``model.classes_`` when available; otherwise ``target_class`` is treated as the column
    index itself.
    """
    if target_class is None:
        return n_classes - 1
    classes = getattr(model, "classes_", None)
    if classes is not None:
        matches = np.where(np.asarray(classes) == target_class)[0]
        if len(matches) == 0:
            raise ValueError(f"'target_class' ({target_class}) should be one of the model "
                             f"classes {list(classes)}.")
        return int(matches[0])
    return int(target_class)


def comp_pred_scores(X_wt=None, X_mut=None, wt_rows=None, model=None, target_class=None):
    """Compute the model prediction-score delta (ΔP%) and the wild-type score per row.

    Returns ``(delta_pred, wt_pred_row, wt_pred_std_row)`` as percentages: ``delta_pred`` is
    ``(P(mut) - P(wt)) * 100`` for each mutation, and ``wt_pred_row`` / ``wt_pred_std_row``
    repeat the per-entry wild-type score along the mutation rows (std is NaN if unavailable).
    """
    wt_pred, wt_pred_std = predict_target_scores(model=model, X=X_wt, target_class=target_class)
    mut_pred, _ = predict_target_scores(model=model, X=X_mut, target_class=target_class)
    delta_pred = (mut_pred - wt_pred[wt_rows]) * 100.0
    wt_pred_row = wt_pred[wt_rows] * 100.0
    if wt_pred_std is None:
        wt_pred_std_row = np.full(len(wt_rows), np.nan)
    else:
        wt_pred_std_row = wt_pred_std[wt_rows] * 100.0
    return delta_pred, wt_pred_row, wt_pred_std_row


def comp_scan_scores(dX=None, mean_dif=None, weight_vec=None):
    """Aggregate a ΔX matrix into per-mutation ``delta_cpp`` (Sum|ΔX|) and ``shift_score``."""
    delta_cpp = np.abs(dX).sum(axis=1)
    direction = np.sign(np.asarray(mean_dif, dtype=float))
    if weight_vec is not None:
        direction = direction * np.asarray(weight_vec, dtype=float)
    shift_score = dX @ direction
    return delta_cpp, shift_score


def build_scan_output(df_plan=None, delta_cpp=None, shift_score=None,
                      delta_pred=None, wt_pred=None, wt_pred_std=None):
    """Assemble the tidy scan output DataFrame, sorted by descending |ΔCPP|.

    When ``delta_pred`` is given (a model is bound), the model prediction-shift columns
    (``delta_pred``, ``wt_pred``, ``wt_pred_std``) are appended after the model-free ones.
    """
    df_out = df_plan[[ut.COL_ENTRY, ut.COL_POS, ut.COL_FROM_AA, ut.COL_TO_AA, ut.COL_REGION]].copy()
    df_out[ut.COL_MUTATION] = [f"{f}{int(p)}{t}" for f, p, t in
                               zip(df_out[ut.COL_FROM_AA], df_out[ut.COL_POS], df_out[ut.COL_TO_AA])]
    df_out[ut.COL_DELTA_CPP] = delta_cpp
    df_out[ut.COL_SHIFT_SCORE] = shift_score
    cols = list(ut.COLS_SEQMUT_SCAN)
    if delta_pred is not None:
        df_out[ut.COL_DELTA_PRED] = delta_pred
        df_out[ut.COL_WT_PRED] = wt_pred
        df_out[ut.COL_WT_PRED_STD] = wt_pred_std
        cols = cols + [ut.COL_DELTA_PRED, ut.COL_WT_PRED, ut.COL_WT_PRED_STD]
    df_out = df_out[cols]
    df_out = df_out.sort_values(ut.COL_DELTA_CPP, ascending=False).reset_index(drop=True)
    return df_out


def comp_matrices_for_seqs(df_var=None, df_seq=None, features=None, df_scales=None, sf=None,
                           jmd_n_len=10, jmd_c_len=10):
    """Build ``(X_wt, X_mut, wt_rows)`` for explicitly-provided mutated sequences.

    ``df_var`` carries one row per variant / evolved sequence with columns ``entry``,
    ``sequence_mut``, ``tmd_start`` and ``tmd_stop``. ``X_wt`` has one row per ``df_seq``
    entry; ``X_mut`` one row per ``df_var`` row; ``wt_rows`` maps each variant to its
    wild-type row. Unlike :func:`comp_feature_matrices`, the mutant sequence is taken
    verbatim (it may carry several point mutations), so combined variants are supported.
    """
    list_parts = get_needed_parts(features=features)
    df_seq_wt = df_seq[[ut.COL_ENTRY, ut.COL_SEQ, ut.COL_TMD_START, ut.COL_TMD_STOP]].copy()
    df_parts_wt = sf.get_df_parts(df_seq=df_seq_wt, list_parts=list_parts,
                                  jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    X_wt = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts_wt,
                                        df_scales=df_scales), dtype=float)
    entry_to_row = {e: i for i, e in enumerate(df_seq_wt[ut.COL_ENTRY])}
    df_seq_mut = pd.DataFrame({
        ut.COL_ENTRY: [f"{e}__{i}" for i, e in enumerate(df_var[ut.COL_ENTRY])],
        ut.COL_SEQ: list(df_var[ut.COL_SEQ_MUT]),
        ut.COL_TMD_START: df_var[ut.COL_TMD_START].to_numpy(),
        ut.COL_TMD_STOP: df_var[ut.COL_TMD_STOP].to_numpy(),
    })
    df_parts_mut = sf.get_df_parts(df_seq=df_seq_mut, list_parts=list_parts,
                                   jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    X_mut = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts_mut,
                                         df_scales=df_scales), dtype=float)
    wt_rows = np.asarray([entry_to_row[e] for e in df_var[ut.COL_ENTRY]], dtype=int)
    return X_wt, X_mut, wt_rows


def comp_seq_scores(df_var=None, df_seq=None, features=None, mean_dif=None, df_scales=None,
                    sf=None, weight_vec=None, model=None, target_class=None,
                    jmd_n_len=10, jmd_c_len=10):
    """Score explicitly-provided mutated sequences (variants / evolved seqs) vs wild-type.

    Returns a dict of per-row arrays: ``delta_cpp`` and ``shift_score`` always, plus
    ``delta_pred`` / ``wt_pred`` / ``wt_pred_std`` when a ``model`` is bound.
    """
    X_wt, X_mut, wt_rows = comp_matrices_for_seqs(
        df_var=df_var, df_seq=df_seq, features=features, df_scales=df_scales, sf=sf,
        jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    dX = X_mut - X_wt[wt_rows]
    delta_cpp, shift_score = comp_scan_scores(dX=dX, mean_dif=mean_dif, weight_vec=weight_vec)
    out = {ut.COL_DELTA_CPP: delta_cpp, ut.COL_SHIFT_SCORE: shift_score}
    if model is not None:
        delta_pred, wt_pred, wt_pred_std = comp_pred_scores(
            X_wt=X_wt, X_mut=X_mut, wt_rows=wt_rows, model=model, target_class=target_class)
        out[ut.COL_DELTA_PRED] = delta_pred
        out[ut.COL_WT_PRED] = wt_pred
        out[ut.COL_WT_PRED_STD] = wt_pred_std
    return out


def greedy_evolve(df_seq=None, df_feat=None, df_scales=None, sf=None, model=None,
                  target_class=None, weight_vec=None, depth=3, region=None, to_aa=None,
                  jmd_n_len=10, jmd_c_len=10):
    """Greedy directed-evolution: per entry, fix the best single mutation each round.

    Each round scans the *current* (already-mutated) sequence, picks the substitution with
    the highest objective (``delta_pred`` when a model is bound, else ``shift_score``),
    fixes it into the running background, and records the running variant's cumulative
    score vs the original wild-type. Positions already mutated are not revisited.
    """
    features = list(df_feat[ut.COL_FEATURE])
    mean_dif = df_feat[ut.COL_MEAN_DIF].to_numpy(dtype=float)
    use_model = model is not None
    rows = []
    for _, r in df_seq.iterrows():
        entry, wt_seq = r[ut.COL_ENTRY], r[ut.COL_SEQ]
        ts, te = int(r[ut.COL_TMD_START]), int(r[ut.COL_TMD_STOP])
        df_seq_wt = pd.DataFrame({ut.COL_ENTRY: [entry], ut.COL_SEQ: [wt_seq],
                                  ut.COL_TMD_START: [ts], ut.COL_TMD_STOP: [te]})
        cur_seq, used_pos = wt_seq, set()
        for round_idx in range(1, depth + 1):
            df_seq_cur = pd.DataFrame({ut.COL_ENTRY: [entry], ut.COL_SEQ: [cur_seq],
                                       ut.COL_TMD_START: [ts], ut.COL_TMD_STOP: [te]})
            df_plan = build_scan_plan(df_seq=df_seq_cur, region=region, to_aa=to_aa,
                                      jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
            df_plan = df_plan[~df_plan[ut.COL_POS].isin(used_pos)].reset_index(drop=True)
            if len(df_plan) == 0:
                break
            # Marginal scores vs the current background.
            X_wt_c, X_mut_c, wt_rows_c = comp_feature_matrices(
                df_plan=df_plan, df_seq=df_seq_cur, features=features, df_scales=df_scales,
                sf=sf, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
            dX = X_mut_c - X_wt_c[wt_rows_c]
            _, shift_c = comp_scan_scores(dX=dX, mean_dif=mean_dif, weight_vec=weight_vec)
            if use_model:
                obj, _, _ = comp_pred_scores(X_wt=X_wt_c, X_mut=X_mut_c, wt_rows=wt_rows_c,
                                             model=model, target_class=target_class)
            else:
                obj = shift_c
            best = int(np.argmax(obj))
            best_pos = int(df_plan[ut.COL_POS].iloc[best])
            best_from = df_plan[ut.COL_FROM_AA].iloc[best]
            best_to = df_plan[ut.COL_TO_AA].iloc[best]
            cur_seq = _apply_point_mutation(seq=cur_seq, pos=best_pos, to_aa=best_to)
            used_pos.add(best_pos)
            mutation_label = f"{best_from}{best_pos}{best_to}"
            # Cumulative score of the running variant vs the original wild-type.
            df_var = pd.DataFrame({ut.COL_ENTRY: [entry], ut.COL_SEQ_MUT: [cur_seq],
                                   ut.COL_TMD_START: [ts], ut.COL_TMD_STOP: [te]})
            cum = comp_seq_scores(df_var=df_var, df_seq=df_seq_wt, features=features,
                                  mean_dif=mean_dif, df_scales=df_scales, sf=sf,
                                  weight_vec=weight_vec, model=model, target_class=target_class,
                                  jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
            row = [entry, round_idx, mutation_label, cur_seq,
                   float(cum[ut.COL_DELTA_CPP][0]), float(cum[ut.COL_SHIFT_SCORE][0])]
            if use_model:
                row.append(float(cum[ut.COL_DELTA_PRED][0]))
            rows.append(row)
    cols = list(ut.COLS_SEQMUT_EVOLVE)
    if use_model:
        cols = cols + [ut.COL_DELTA_PRED]
    return pd.DataFrame(rows, columns=cols)


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

"""
This is a script for the backend of the SequenceFeature() object,
a supportive class for the CPP feature engineering.
"""
import pandas as pd
import numpy as np

import aaanalysis.utils as ut
from .utils_feature import get_positions_, get_feature_matrix_, get_amino_acids_, add_scale_info_
from ._split import SplitRange
from ._utils_feature_stat import add_stat_


# I Helper Functions


# II Main Functions
# Parts and splits
def get_split_kws_(n_split_min=1, n_split_max=15, steps_pattern=None, n_min=2, n_max=4, len_max=15,
                   steps_periodicpattern=None, split_types=None):
    """Get split kws for CPP class"""
    if split_types is None:
        split_types = ut.LIST_SPLIT_TYPES
    if steps_pattern is None:
        # Differences between interacting amino acids in helix (without gaps) include 6, 7 ,8 to include gaps
        steps_pattern = [3, 4]
    if steps_periodicpattern is None:
        steps_periodicpattern = [3, 4]  # Differences between interacting amino acids in helix (without gaps)
    split_kws = {ut.STR_SEGMENT: dict(n_split_min=n_split_min, n_split_max=n_split_max),
                 ut.STR_PATTERN: dict(steps=steps_pattern, n_min=n_min, n_max=n_max, len_max=len_max),
                 ut.STR_PERIODIC_PATTERN: dict(steps=steps_periodicpattern)}
    split_kws = {x: split_kws[x] for x in split_types}
    return split_kws


# Features
def get_features_(list_parts=None, split_kws=None, list_scales=None):
    """Create list of all feature ids for given Parts, Splits, and Scales"""
    spr = SplitRange(split_type_str=True)
    features = []
    for split_type in split_kws:
        args = split_kws[split_type]
        labels_s = getattr(spr, "labels_" + split_type.lower())(**args)
        features.extend([ut.join_feat_id(part=part.upper(), split=split, scale_id=scale)
                         for part in list_parts for split in labels_s for scale in list_scales])
    return features


def get_feature_names_(features=None, df_cat=None, tmd_len=20, jmd_c_len=10, jmd_n_len=10, start=1):
    """Convert feature ids (PART-SPLIT-SCALE) into feature names (scale name [positions])."""
    feat_positions = get_positions_(features=features, tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                    start=start)
    dict_scales = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_SUBCAT]))
    feat_names = []
    for feat_id, pos in zip(features, feat_positions):
        part, split, scale = ut.split_feat_id(feat_id=feat_id)
        split_type = split.split("(")[0]
        if split_type == ut.STR_SEGMENT and len(pos.split(",")) > 2:
            pos = pos.split(",")[0] + "-" + pos.split(",")[-1]
        if split_type == ut.STR_PERIODIC_PATTERN:
            step = split.split("+")[1].split(",")[0]
            pos = pos.split(",")[0] + ".." + step + ".." + pos.split(",")[-1]
        feat_names.append(f"{dict_scales[scale]} [{pos}]")
    return feat_names


def _split_to_phrase(split=None):
    """Turn a SPLIT token into a readable phrase; flag whether its positions are contiguous."""
    # Parse the SPLIT string locally (like get_feature_names_ above) rather than reusing the
    # file-private _get_split_info in utils_feature.py: it lives in another module and only the
    # display wording — not the parsed ints — is needed here.
    split_type = split.split("(")[0]
    args = split.split("(")[1].replace(")", "")
    if split_type == ut.STR_SEGMENT:
        i_th, n_split = [int(x) for x in args.split(",")]
        return f"segment {i_th} of {n_split}", True
    terminus = args.split(",")[0]
    term = "N-terminus" if terminus == "N" else "C-terminus"
    if split_type == ut.STR_PERIODIC_PATTERN:
        steps = split.split("i+")[1].split(",")[0]  # e.g. '3/4'
        return f"periodic pattern (steps {steps} from {term})", False
    return f"pattern (from {term})", False


def get_feature_descriptions_(features=None, df_cat=None, tmd_len=20, jmd_c_len=10, jmd_n_len=10, start=1):
    """Build one standardized, human-readable description per feature id (PART-SPLIT-SCALE)."""
    feat_positions = get_positions_(features=features, tmd_len=tmd_len, jmd_n_len=jmd_n_len,
                                    jmd_c_len=jmd_c_len, start=start)
    dict_name = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_SCALE_NAME]))
    dict_cat = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_CAT]))
    dict_subcat = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_SUBCAT]))
    descriptions = []
    for feat_id, pos in zip(features, feat_positions):
        part, split, scale = ut.split_feat_id(feat_id=feat_id)
        part_label = ut.DICT_PART_LABEL[part.lower()]
        phrase, contiguous = _split_to_phrase(split=split)
        list_pos = pos.split(",")
        pos_str = f"{list_pos[0]}-{list_pos[-1]}" if (contiguous and len(list_pos) > 2) else ", ".join(list_pos)
        des = (f"{part_label}, {phrase} (positions {pos_str}) — "
               f"{dict_name[scale]} [{dict_cat[scale]}: {dict_subcat[scale]}]")
        descriptions.append(des)
    return descriptions


def get_df_feat_(features=None, df_parts=None, labels=None,
                 label_test=1, label_ref=0, df_scales=None, df_cat=None,
                 accept_gaps=False, parametric=False,
                 start=1, tmd_len=20, jmd_c_len=10, jmd_n_len=10, n_jobs=1):
    """Create df feature for comparing groups and or samples"""
    X = get_feature_matrix_(features=features, df_parts=df_parts, df_scales=df_scales,
                            accept_gaps=accept_gaps, n_jobs=n_jobs)
    mask_test = [x == label_test for x in labels]
    mask_ref = [x == label_ref for x in labels]
    mean_dif = X[mask_test].mean(axis=0) - X[mask_ref].mean(axis=0)
    abs_mean_dif = abs(mean_dif)
    std_test = X[mask_test].std(axis=0)
    df = pd.DataFrame(zip(features, abs_mean_dif, std_test),
                      columns=[ut.COL_FEATURE, ut.COL_ABS_MEAN_DIF, ut.COL_STD_TEST])
    df = add_stat_(df=df, X=X, labels=labels, parametric=parametric,
                   label_test=label_test, label_ref=label_ref, n_jobs=n_jobs)
    df = add_scale_info_(df_feat=df, df_cat=df_cat)
    df[ut.COL_POSITION] = get_positions_(features=features, start=start, jmd_n_len=jmd_n_len, tmd_len=tmd_len,
                                         jmd_c_len=jmd_c_len)
    if sum(mask_test) == 1:
        position = np.where(labels == label_test)[0][0]
        jmd_n_seq, tmd_seq, jmd_c_seq = df_parts[[ut.COL_JMD_N, ut.COL_TMD, ut.COL_JMD_C]].iloc[position].values
        df[ut.COL_AA_TEST] = get_amino_acids_(features=features, jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq,
                                              jmd_c_seq=jmd_c_seq)
    if sum(mask_ref) == 1:
        position = np.where(labels == label_ref)[0][0]
        jmd_n_seq, tmd_seq, jmd_c_seq = df_parts[[ut.COL_JMD_N, ut.COL_TMD, ut.COL_JMD_C]].iloc[position].values
        df[ut.COL_AA_REF] = get_amino_acids_(features=features, jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq,
                                             jmd_c_seq=jmd_c_seq)
    # Standardize the df_feat column order (issue #18); amino_acids_* append last.
    df = ut.sort_cols_feat(df_feat=df)
    return df


# Scale-composition baseline featurization
def get_scale_composition_(df_parts=None, df_scales=None):
    """Per-sequence scale average over the concatenated sequence parts (vectorized).

    For each row of ``df_parts`` the part strings are concatenated into one span; every
    residue that is not a (single-letter) index label of ``df_scales`` (gaps ``'-'`` and
    any other non-canonical symbol) is dropped, and the remaining residues' scale rows are
    averaged column-wise into one value per scale, giving the ``(n_seq, n_scales)`` matrix.
    A row whose span has no scored residue (empty / all-non-canonical) becomes all-``NaN``;
    ``n_kept`` reports the number of averaged residues per row so the frontend can warn.

    Implementation: all residues of all spans are flattened into one byte array and mapped
    to scale rows via a 256-entry lookup table; a single ``np.bincount`` tallies a small
    per-sequence residue-count matrix ``(n_seq, n_letters)``, and one BLAS matmul against
    the scale matrix yields the per-sequence sums — no per-sequence Python loop or
    ``DataFrame.loc``, and the wide part scales with ``n_scales`` as a matmul, not a scatter.
    """
    n_seq = len(df_parts)
    scales_arr = df_scales.to_numpy(dtype=float)                    # (n_letters, n_scales)
    n_letters = scales_arr.shape[0]
    # Byte -> scale-row lookup (-1 = non-scored); only single-character labels can match a residue
    lut = np.full(256, -1, dtype=np.intp)
    for row, aa in enumerate(df_scales.index):
        if len(aa) == 1 and ord(aa) < 256:
            lut[ord(aa)] = row
    # Flatten every residue of every span, tracking its owning sequence
    # (df_parts columns are guaranteed str by the frontend's get_df_parts)
    spans = df_parts.agg("".join, axis=1).to_list()
    lengths = np.fromiter((len(s) for s in spans), dtype=np.intp, count=n_seq)
    codes = np.frombuffer("".join(spans).encode("latin-1", "replace"), dtype=np.uint8)
    seq_ids = np.repeat(np.arange(n_seq), lengths)
    rows = lut[codes]                                               # scale row per residue
    valid = rows >= 0
    seq_ids, rows = seq_ids[valid], rows[valid]
    # Per-sequence residue-count matrix (n_seq, n_letters) from one bincount, then sums via matmul
    counts = np.bincount(seq_ids * n_letters + rows,
                         minlength=n_seq * n_letters).reshape(n_seq, n_letters)
    n_kept = counts.sum(axis=1).astype(int)
    sums = counts @ scales_arr                                     # (n_seq, n_scales)
    with np.errstate(invalid="ignore"):
        X = sums / n_kept[:, None]                                 # 0 / 0 -> NaN (empty rows)
    return X, n_kept


def _canonical_codes_(df_parts=None):
    """Flatten every residue of every span to a canonical index (0..19), dropping non-canonical.

    The part strings of each row are concatenated into one span (in ``df_parts`` column order,
    matching :func:`get_scale_composition_`), every residue is mapped through a 256-entry byte
    lookup to its position in ``ut.LIST_CANONICAL_AA``, and gaps / non-canonical symbols (byte
    lookup ``-1``, incl. codepoints above 255 which ``latin-1`` "replace" turns into ``'?'``)
    are dropped. Returns the per-residue owning-sequence ids ``seq_ids`` and canonical indices
    ``aa_idx`` (both 1-D, aligned) plus the sequence count ``n_seq`` — the shared vectorized
    front half of the AAC and DPC composition featurizers (no per-sequence Python loop).
    """
    n_seq = len(df_parts)
    list_aa = list(ut.LIST_CANONICAL_AA)
    # Byte -> canonical index lookup (-1 = non-canonical)
    lut = np.full(256, -1, dtype=np.intp)
    for idx, aa in enumerate(list_aa):
        lut[ord(aa)] = idx
    # Flatten every residue of every span, tracking its owning sequence
    # (df_parts columns are guaranteed str by the frontend's get_df_parts)
    spans = df_parts.agg("".join, axis=1).to_list()
    lengths = np.fromiter((len(s) for s in spans), dtype=np.intp, count=n_seq)
    codes = np.frombuffer("".join(spans).encode("latin-1", "replace"), dtype=np.uint8)
    seq_ids = np.repeat(np.arange(n_seq), lengths)
    aa_idx = lut[codes]                                             # canonical index per residue
    valid = aa_idx >= 0
    return seq_ids[valid], aa_idx[valid], n_seq


def get_aa_composition_(df_parts=None):
    """Per-sequence amino-acid composition over the concatenated sequence parts (vectorized).

    For each row of ``df_parts`` the parts are concatenated into one span, non-canonical
    residues are dropped, and the 20 canonical amino acids (``ut.LIST_CANONICAL_AA`` order)
    are tallied and divided by the kept-residue count, giving the ``(n_seq, 20)`` fraction
    matrix ``X`` (each row sums to 1). A span with no canonical residue (empty / all
    non-canonical) becomes an all-``NaN`` row; ``n_kept`` reports the number of counted
    residues per row so the frontend can warn. One ``np.bincount`` over ``20 * seq + aa``
    builds the whole count matrix, so there is no per-sequence Python loop.
    """
    seq_ids, aa_idx, n_seq = _canonical_codes_(df_parts=df_parts)
    counts = np.bincount(seq_ids * 20 + aa_idx,
                         minlength=n_seq * 20).reshape(n_seq, 20).astype(float)
    n_kept = counts.sum(axis=1).astype(int)
    with np.errstate(invalid="ignore"):
        X = counts / n_kept[:, None]                               # 0 / 0 -> NaN (empty rows)
    return X, n_kept


def get_dipeptide_composition_(df_parts=None):
    """Per-sequence dipeptide composition over the concatenated sequence parts (vectorized).

    For each row of ``df_parts`` the parts are concatenated into one span and non-canonical
    residues are dropped; the remaining canonical residues form one contiguous chain over
    which the 400 ordered adjacent pairs (``AA, AC, ..., YY`` in ``ut.LIST_CANONICAL_AA``
    order, pair code ``20 * first + second``) are counted and divided by the pair count,
    giving the ``(n_seq, 400)`` fraction matrix ``X`` (each row with >= 2 canonical residues
    sums to 1). Pairs are formed on the concatenated, gap-free span, so an adjacency spans a
    dropped non-canonical residue and crosses a part boundary. A span with fewer than two
    canonical residues has no pair and becomes an all-``NaN`` row; ``n_pairs`` reports the
    counted pairs per row so the frontend can warn.

    Fully vectorized: consecutive flattened residues form candidate pairs, a same-sequence
    mask (``seq_ids[:-1] == seq_ids[1:]``) drops the cross-sequence junctions, and one
    ``np.bincount`` over ``400 * seq + pair`` builds the whole count matrix — no per-sequence
    Python loop.
    """
    seq_ids, aa_idx, n_seq = _canonical_codes_(df_parts=df_parts)
    # Candidate adjacent pairs are consecutive flattened residues; keep only same-sequence ones
    same = seq_ids[:-1] == seq_ids[1:]
    pair_seq = seq_ids[:-1][same]
    pair_code = (aa_idx[:-1][same] * 20 + aa_idx[1:][same])         # 0..399
    counts = np.bincount(pair_seq * 400 + pair_code,
                         minlength=n_seq * 400).reshape(n_seq, 400).astype(float)
    n_pairs = counts.sum(axis=1).astype(int)
    with np.errstate(invalid="ignore"):
        X = counts / n_pairs[:, None]                              # 0 / 0 -> NaN (< 2 residues)
    return X, n_pairs


# Multi-class / regression label conversion
def get_labels_ovr_(labels=None, label_test=1, label_ref=0):
    """One-vs-rest: per class, a full-length binary array (class -> test, rest -> ref)."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    return {int(c): np.where(labels == c, label_test, label_ref).astype(int) for c in classes}


def get_labels_ovo_(labels=None, label_test=1, label_ref=0):
    """One-vs-one: per class pair (a, b), a row mask + binary labels for the masked subset."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    out = {}
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            mask = np.isin(labels, [a, b])
            binary = np.where(labels[mask] == a, label_test, label_ref).astype(int)
            out[(int(a), int(b))] = (mask, binary)
    return out


def get_labels_quantile_(targets=None, q=0.5, label_test=1, label_ref=0):
    """Single-threshold split of continuous targets into a binary array (>= q-quantile -> test)."""
    targets = np.asarray(targets, dtype=float)
    cut = float(np.quantile(targets, q))
    return np.where(targets >= cut, label_test, label_ref).astype(int)


def get_labels_tiered_(targets=None, q_pos=0.8, list_q_neg=(0.8, 0.5, 0.3), label_test=1, label_ref=0):
    """Tiered thresholds: fixed positive set, stepwise-lower negative cuts, middle dropped.

    Positives are ``targets >= Q(q_pos)`` (fixed across tiers); for each ``q_neg``
    the negatives are ``targets <= Q(q_neg)`` (positives take precedence on ties),
    and samples in between are dropped. Returns ``{q_neg: (row_mask, binary)}``.
    Raises ``ValueError`` if a tier yields only one class.
    """
    targets = np.asarray(targets, dtype=float)
    cut_pos = float(np.quantile(targets, q_pos))
    pos = targets >= cut_pos
    out = {}
    for q_neg in list_q_neg:
        cut_neg = float(np.quantile(targets, q_neg))
        neg = (targets <= cut_neg) & ~pos
        mask = pos | neg
        if pos.sum() == 0 or neg.sum() == 0:
            raise ValueError(
                f"tier q_neg={q_neg} yields a single class "
                f"(n_pos={int(pos.sum())}, n_neg={int(neg.sum())}); "
                f"choose q_pos/q_neg that keep both groups non-empty."
            )
        binary = np.where(pos[mask], label_test, label_ref).astype(int)
        out[float(q_neg)] = (mask, binary)
    return out

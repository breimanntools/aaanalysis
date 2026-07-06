"""
This is a script for the backend of the SequenceFeature() object,
a supportive class for the CPP feature engineering.
"""
import itertools
import pandas as pd
import numpy as np

import aaanalysis.utils as ut
from .utils_feature import get_positions_, get_feature_matrix_, get_amino_acids_, add_scale_info_
from ._split import SplitRange
from ._utils_feature_stat import add_stat_

# Physicochemical grouping of the 20 canonical amino acids — the small, colorable category structure
# for one-hot AAC scales and k-mer composition maps (independent of the AAontology scale categories).
_DICT_AA_CLASS = {aa: cls for cls, aas in
                  {"Nonpolar": "GAVLIPM", "Aromatic": "FWY", "Polar": "STCNQ",
                   "Positive": "KRH", "Negative": "DE"}.items() for aa in aas}


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


def get_kmer_composition_(df_parts=None, k=1):
    """Per-sequence k-mer composition over the concatenated sequence parts (vectorized).

    Generalizes :func:`get_aa_composition_` (``k=1``, AAC) and
    :func:`get_dipeptide_composition_` (``k=2``, DPC): for each row of ``df_parts`` the parts
    are concatenated into one span, non-canonical residues are dropped, and the ``20 ** k``
    ordered overlapping k-mers of adjacent canonical residues are counted and divided by the
    k-mer count, giving the ``(n_seq, 20 ** k)`` fraction matrix ``X`` (each row with at least
    ``k`` canonical residues sums to 1). A k-mer of residues ``r_0 r_1 ... r_{k-1}`` has the
    base-20 code ``sum_j r_j * 20 ** (k - 1 - j)`` (``ut.LIST_CANONICAL_AA`` order), so column
    order matches ``itertools.product(LIST_CANONICAL_AA, repeat=k)``. k-mers are formed on the
    concatenated, gap-free span (so a window spans dropped non-canonical residues and crosses
    part boundaries); a same-sequence mask (``seq_ids[i] == seq_ids[i + k - 1]``) drops windows
    straddling two sequences. A span with fewer than ``k`` canonical residues has no k-mer and
    becomes an all-``NaN`` row; ``n_kmers`` reports the counted k-mers per row so the frontend
    can warn. One ``np.bincount`` over ``20 ** k * seq + code`` builds the whole count matrix,
    so there is no per-sequence Python loop. ``k`` is validated (int, in range) by the frontend.
    """
    seq_ids, aa_idx, n_seq = _canonical_codes_(df_parts=df_parts)
    n_codes = 20 ** k
    m = aa_idx.shape[0]
    if m < k:                                                      # no window fits in any span
        return np.full((n_seq, n_codes), np.nan), np.zeros(n_seq, dtype=int)
    # Overlapping windows of k consecutive flattened residues; keep only same-sequence ones.
    starts = np.arange(m - k + 1)
    same = seq_ids[starts] == seq_ids[starts + k - 1]
    starts = starts[same]
    win_seq = seq_ids[starts]
    code = np.zeros(starts.shape[0], dtype=np.int64)
    for j in range(k):
        code = code * 20 + aa_idx[starts + j]                     # base-20 k-mer code (0 .. 20**k-1)
    counts = np.bincount(win_seq * n_codes + code,
                         minlength=n_seq * n_codes).reshape(n_seq, n_codes).astype(float)
    n_kmers = counts.sum(axis=1).astype(int)
    with np.errstate(invalid="ignore"):
        X = counts / n_kmers[:, None]                             # 0 / 0 -> NaN (< k residues)
    return X, n_kmers


def get_composition_scales_(k=1):
    """Scale set + category table for k-mer composition (feeds ``CPP.run`` / colors the composition map).

    ``k == 1``: the ``(20, 20)`` one-hot identity ``df_scales`` (scale ``<AA>`` = 1 on residue ``AA``,
    0 elsewhere) plus a ``df_cat`` mapping each amino acid to its physicochemical class — feed both to
    :meth:`CPP.run` with the whole-part ``Segment(1,1)`` split to obtain amino-acid composition as a
    real ``df_feat`` / feature map. ``k >= 2``: ``df_scales`` is ``None`` (a k-mer is a property of an
    adjacent tuple, not a per-residue scale, so it cannot be a CPP scale) and ``df_cat`` categorizes
    each of the ``20 ** k`` k-mers by its residues' classes — for grouping / labeling the composition
    map, not for :meth:`CPP.run`.
    """
    aa = list(ut.LIST_CANONICAL_AA)
    kmers = ["".join(p) for p in itertools.product(aa, repeat=k)]
    subcat = ["-".join(_DICT_AA_CLASS[c] for c in km) for km in kmers]
    df_cat = pd.DataFrame({
        ut.COL_SCALE_ID: kmers,
        ut.COL_CAT: [_DICT_AA_CLASS[km[0]] for km in kmers],       # first-residue class (coarse)
        ut.COL_SUBCAT: subcat,                                     # per-residue class tuple
        ut.COL_SCALE_NAME: kmers,
        ut.COL_SCALE_DES: [f"{k}-mer {km} ({sc})" if k > 1 else f"Amino acid {km} indicator"
                           for km, sc in zip(kmers, subcat)],
    })
    df_scales = pd.DataFrame(np.eye(len(aa)), index=aa, columns=aa) if k == 1 else None
    return df_scales, df_cat


def get_aa_composition_(df_parts=None):
    """Per-sequence amino-acid composition (AAC) — the ``k=1`` case of :func:`get_kmer_composition_`.

    Concatenates each row's parts into one span, drops non-canonical residues, and returns the
    ``(n_seq, 20)`` fraction matrix ``X`` (``ut.LIST_CANONICAL_AA`` order, each row sums to 1)
    plus the kept-residue count per row (all-``NaN`` row when a span has no canonical residue).
    """
    return get_kmer_composition_(df_parts=df_parts, k=1)


def get_dipeptide_composition_(df_parts=None):
    """Per-sequence dipeptide composition (DPC) — the ``k=2`` case of :func:`get_kmer_composition_`.

    Concatenates each row's parts into one gap-free span, drops non-canonical residues, and
    returns the ``(n_seq, 400)`` fraction matrix ``X`` of ordered adjacent pairs (``AA, AC, ...,
    YY`` in ``ut.LIST_CANONICAL_AA`` order, each row with >= 2 canonical residues sums to 1)
    plus the pair count per row (all-``NaN`` row when a span has fewer than two canonical residues).
    """
    return get_kmer_composition_(df_parts=df_parts, k=2)


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

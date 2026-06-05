"""
This is a script for the backend of the AAWindowSampler.sample_synthetic() method.

Synthetic windows are produced by a ``generator`` that selects one of:

1. Independently of any source distribution (``uniform``).
2. An empirical frequency derived at call time from ``df_seq``
   (``global_freq``).
3. A curated AAontology preset — a per-amino-acid scale (loaded via
   :func:`aa.load_scales`) normalized into a probability distribution.
4. The per-position frequencies of the test windows
   (``position_specific``).
5. Shuffling within a randomly chosen test window (``scrambled``).
6. A multiplicative mix of multiple presets (list/tuple of preset names).
7. A user-supplied frequency table over an arbitrary single-character alphabet
   (dict[str, float]).
"""
import warnings
import numpy as np

import aaanalysis.utils as ut
from aaanalysis.data_handling._load_scales import load_scales
from ._utils import (collect_test_windows,
                     make_safe_custom_predicate_,
                     sample_pool_iteratively_)


# AAontology-backed presets.
#
# Each preset selects one canonical AAontology scale (loaded via aa.load_scales).
# The scale's per-amino-acid values are normalized into a probability distribution
# and used as the prior for synthetic-window generation.
#
# Composition presets are *true* AA-frequency distributions; conformation presets
# are normalized propensities and act as physicochemically-biased priors.
PRESETS = {
    # Composition (3) — true AA-frequency distributions
    "aa_composition":           ("DAYM780101", "Dayhoff et al., 1978a",
                                  "AA composition (canonical baseline)"),
    "aa_composition_surface":   ("FUKS010102", "Fukuchi-Nishikawa, 2001",
                                  "AA composition at the protein surface"),
    "aa_composition_mp":        ("CEDJ970103", "Cedano et al., 1997",
                                  "AA composition in membrane proteins"),
    # Conformation (7) — secondary-structure propensities
    "alpha_helix":              ("CHOP780201", "Chou-Fasman, 1978b",
                                  "α-helix propensity"),
    "beta_sheet":               ("CHOP780202", "Chou-Fasman, 1978b",
                                  "β-sheet propensity"),
    "beta_strand":              ("LIFS790101", "Lifson-Sander, 1979",
                                  "β-strand propensity"),
    "beta_turn":                ("CHOP780203", "Chou-Fasman, 1978b",
                                  "β-turn propensity"),
    "coil":                     ("NAGK730103", "Nagano, 1973",
                                  "Coil propensity"),
    "linker":                   ("GEOR030106", "George-Heringa, 2003",
                                  "Medium-length (6-14 AA) linker propensity"),
    "pi_helix":                 ("FODM020101", "Fodje-Al-Karadaghi, 2002",
                                  "π-helix propensity"),
}
LIST_PRESET_GENERATORS = list(PRESETS.keys())
LIST_SYNTH_GENERATORS = ut.LIST_SYNTH_MODES_BUILTIN + LIST_PRESET_GENERATORS


# I Helper Functions
def _global_aa_freq(df_seq, aa_list, aa_index):
    counts = np.zeros(len(aa_list), dtype=np.int64)
    for s in df_seq[ut.COL_SEQ]:
        for c in s:
            j = aa_index.get(c)
            if j is not None:
                counts[j] += 1
    total = counts.sum()
    if total == 0:
        raise ValueError("No canonical amino acids found in df_seq sequences.")
    return counts / total


def _load_raw_scales():
    """Load the raw AAontology scales table (not min-max normalized).

    Raw values preserve true composition / propensity ratios — e.g. the Dayhoff
    composition scale gives Trp ≈ 1.3% rather than 0 (which min-max would
    assign to whichever AA is least frequent).
    """
    return load_scales(name=ut.STR_SCALES_RAW)


def _preset_aa_freq(scale_id, aa_list, df_scales=None):
    """Load a raw AAontology scale and sum-normalize to a probability distribution.

    Uses the **raw** AAontology scale rather than the min-max normalized one,
    so true frequency / propensity ratios are preserved (e.g. Trp gets ~1.3%
    mass under Dayhoff rather than 0). All :data:`PRESETS` entries are
    all-positive raw composition or secondary-structure propensity scales,
    so a single sum-normalization step is sufficient. Negative-containing
    scales are rejected loudly because no curated preset uses them.

    Pass ``df_scales`` to skip the per-call ``_load_raw_scales`` copy when
    this helper is invoked in a tight loop (see ``_mix_preset_aa_freq``).
    """
    if df_scales is None:
        df_scales = _load_raw_scales()
    if scale_id not in df_scales.columns:
        raise ValueError(f"Scale '{scale_id}' not found in raw AAontology "
                         f"(scales_raw).")
    vals = np.array([float(df_scales.loc[a, scale_id]) for a in aa_list], dtype=float)
    if (vals < 0).any():
        raise ValueError(f"Scale '{scale_id}' has negative values; only "
                         f"all-positive composition / propensity scales are "
                         f"supported as preset priors.")
    total = vals.sum()
    if total == 0:
        raise ValueError(f"Scale '{scale_id}' sums to zero; cannot normalize.")
    return vals / total


def _mix_preset_aa_freq(preset_names, aa_list):
    """Multiplicative mix of multiple AAontology presets, renormalized.

    Implements a Bayesian-style joint prior over the 20 canonical AAs:
    each component contributes a normalized (20,) vector, the element-wise
    product is taken, and the result is renormalized. Motivated by the
    combined-property approach of Liu & Deber 1999 (hydrophobicity *and*
    helicity for transmembrane characterization).
    """
    df_scales = _load_raw_scales()
    arrays = [_preset_aa_freq(PRESETS[m][0], aa_list, df_scales=df_scales)
              for m in preset_names]
    product = np.prod(arrays, axis=0)
    total = product.sum()
    if total == 0:
        raise ValueError(f"Mixed preset prior over {list(preset_names)} sums to "
                         f"zero; no amino acid has positive probability under "
                         f"all components.")
    return product / total


def _custom_alphabet_probs(mode_dict):
    """Return ``(aa_list, probs)`` from a user-supplied frequency table.

    Keys define the alphabet (sorted for reproducibility). Validation of
    key/value shapes and sum-to-1 lives in the frontend.
    """
    aa_list = sorted(mode_dict.keys())
    probs = np.array([float(mode_dict[a]) for a in aa_list], dtype=float)
    return aa_list, probs


def _position_aa_freq(test_windows, window_size, aa_list, aa_index):
    arr = np.array([list(w) for w in test_windows])
    position_probs = []
    for col in range(window_size):
        counts = np.zeros(len(aa_list), dtype=np.int64)
        for c in arr[:, col]:
            j = aa_index.get(c)
            if j is not None:
                counts[j] += 1
        total = counts.sum()
        if total == 0:
            raise ValueError(f"Position {col} of test windows contains no "
                             f"canonical amino acids.")
        position_probs.append(counts / total)
    return position_probs


def _build_synthetic_drawer(*, df_seq, generator, pos_col, window_size,
                             aa_list, aa_index, rng):
    """Return a no-arg function that generates one synthetic window per call.

    ``generator`` may be a string (built-in mode or AAontology preset), a list
    of preset names (multiplicative mix, see ``_mix_preset_aa_freq``), or a
    dict mapping single-character symbols to non-negative probabilities
    (custom alphabet, not necessarily protein). Validation lives in the
    frontend.
    """
    if isinstance(generator, dict):
        custom_alpha, probs = _custom_alphabet_probs(generator)
        return lambda: "".join(rng.choice(custom_alpha, size=window_size,
                                          p=probs, replace=True))
    if isinstance(generator, (list, tuple)):
        probs = _mix_preset_aa_freq(generator, aa_list)
        return lambda: "".join(rng.choice(aa_list, size=window_size,
                                          p=probs, replace=True))
    if generator == ut.MODE_UNIFORM:
        return lambda: "".join(rng.choice(aa_list, size=window_size, replace=True))
    if generator == ut.MODE_GLOBAL_FREQ:
        probs = _global_aa_freq(df_seq, aa_list, aa_index)
        return lambda: "".join(rng.choice(aa_list, size=window_size, p=probs, replace=True))
    if generator in PRESETS:
        scale_id = PRESETS[generator][0]
        probs = _preset_aa_freq(scale_id, aa_list)
        return lambda: "".join(rng.choice(aa_list, size=window_size, p=probs, replace=True))
    if pos_col is None:
        raise ValueError(f"'pos_col' is required for generator='{generator}'.")
    test_windows_for_gen = collect_test_windows(df_seq, pos_col, window_size)
    if not test_windows_for_gen:
        raise ValueError(f"No test windows of size {window_size} found in df_seq.")
    if generator == ut.MODE_POSITION_SPECIFIC:
        position_probs = _position_aa_freq(test_windows_for_gen, window_size,
                                            aa_list, aa_index)

        def draw_pos_specific():
            return "".join(rng.choice(aa_list, p=position_probs[col])
                           for col in range(window_size))

        return draw_pos_specific
    pool = list(test_windows_for_gen)

    def draw_scrambled():
        base = pool[rng.integers(0, len(pool))]
        chars = list(base)
        rng.shuffle(chars)
        return "".join(chars)

    return draw_scrambled


# II Main Functions
def sample_synthetic(*, df_seq, n, window_size, generator, pos_col,
                       test_windows, max_similarity_to_test, max_similarity_within_ref,
                       max_sampling_attempts, filter_iteratively, rng, verbose,
                       custom_filter=None):
    """Generate ``n`` synthetic windows for the requested ``generator``.

    Returns
    -------
    list of str
        Up to ``n`` accepted synthetic windows.
    """
    aa_list = list(ut.LIST_CANONICAL_AA)
    aa_index = {a: i for i, a in enumerate(aa_list)}
    draw_one = _build_synthetic_drawer(df_seq=df_seq, generator=generator,
                                        pos_col=pos_col, window_size=window_size,
                                        aa_list=aa_list, aa_index=aa_index, rng=rng)

    def draw_batch(needed):
        return [(draw_one(), None) for _ in range(needed)]

    # Synthetic windows have no source protein: the user filter sees
    # entry="" and source_position=-1 (composition-only context).
    predicate = None
    if custom_filter is not None:
        predicate = make_safe_custom_predicate_(
            custom_filter, lambda _p: ("", -1))
    accepted = sample_pool_iteratively_(
        draw_batch=draw_batch, target_n=n, test_windows=test_windows,
        max_similarity_to_test=max_similarity_to_test,
        max_similarity_within_ref=max_similarity_within_ref,
        motif_pwm=None, motif_score_threshold=None, motif_match=None,
        max_attempts=max_sampling_attempts,
        filter_iteratively=filter_iteratively,
        custom_predicate=predicate,
    )
    if len(accepted) < n and verbose:
        warnings.warn(f"Only {len(accepted)}/{n} synthetic windows kept after filtering.",
                      RuntimeWarning)
    return [w for w, _ in accepted]

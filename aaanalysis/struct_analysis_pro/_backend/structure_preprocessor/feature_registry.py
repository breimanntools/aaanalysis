"""
This is a script for the backend of the StructurePreprocessor: a central
registry mapping each canonical feature_key to its source method, output
dimensionality, default dimension names, category / subcategory labels, and
the min-max normalization recipe applied at the encoder.

Every feature key declares ``category='Structure'`` — the top-level color /
redundancy-bucket grouping that pairs with ``ut.DICT_COLOR_CAT['Structure']``.
Fine-grained semantics live in ``subcategory``. The shared
``NORMALIZATION_RECIPES`` dict is the source of truth for the inverse
formulas published in the ``StructurePreprocessor`` class docstring.
"""
from typing import Callable, Dict, List, Optional

import numpy as np

# Single top-level category for everything emitted by StructurePreprocessor.
# Pairs with ``ut.DICT_COLOR_CAT['Structure']`` (color "#2E6E5E"). The
# redundancy filter's ``check_cat=True`` arm therefore groups all Structure
# features into one bucket; per-AA-mean df_scales values let the cor gate
# discriminate within the bucket.
CATEGORY_STRUCTURE = "Structure"


# I Helper Functions
def _identity(x):
    return x


def _clip_unit(x):
    return np.clip(x, 0.0, 1.0)


def _div(divisor: float) -> Callable:
    def recipe(x, _d=divisor):
        return np.clip(x / _d, 0.0, 1.0)
    return recipe


def _shift_half(x):
    # (x + 1) / 2; pairs with sin/cos inputs in [-1, 1].
    return (x + 1.0) / 2.0


# II Main Functions
ENCODER_DSSP = "encode_dssp"
ENCODER_PDB = "encode_pdb"
ENCODER_PAE = "encode_pae"   # used by v1.1 PAE keys (commit 3)


# Normalization recipes — keyed by feature_key. Each value is a callable
# applied to the raw per-residue tensor after extraction. The class
# docstring's "raw range -> recipe -> inverse" table is generated from this
# dict; do not duplicate the constants in encoder code.
NORMALIZATION_RECIPES: Dict[str, Callable] = {
    "ss3":                    _identity,        # one-hot ∈ {0, 1}
    "ss8":                    _identity,        # one-hot ∈ {0, 1}
    "rasa":                   _clip_unit,       # Tien table can overshoot slightly
    "phi_psi_sincos":         _shift_half,      # [-1, 1] → [0, 1]
    "bfactor":                _div(100.0),      # saturates at 100 Å²
    "depth":                  _div(15.0),       # saturates at 15 Å
    # AF model-file features (commit 2)
    "plddt":                  _div(100.0),      # pLDDT ∈ [0, 100] → [0, 1]
    "plddt_disorder":         _identity,        # boolean ∈ {0, 1}
    "plddt_tier":             _identity,        # one-hot ∈ {0, 1}
    "chi1_sincos":            _shift_half,      # [-1, 1] → [0, 1]
    "chi2_sincos":            _shift_half,      # [-1, 1] → [0, 1]
    "ca_centroid_dist":       _div(40.0),       # saturates at 40 Å
    "ca_centroid_dist_norm":  _div(2.0),        # saturates at 2 Rg
    "contact_count_8A":       _div(30.0),       # saturates at 30
    "contact_count_12A":      _div(80.0),       # saturates at 80
    "hse":                    _div(30.0),       # half-sphere counts, ~30 ceiling
    # AF PAE sidecar features (commit 3). AF documents the PAE saturation
    # cap at 31.75 Å; we divide by that and clip. ``pae_asymmetry`` is
    # bounded much lower in practice (asymmetry << absolute PAE) so we use
    # an empirical cap of 10 Å.
    "pae_row_mean":           _div(31.75),
    "pae_row_min":            _div(31.75),
    "pae_row_max":            _div(31.75),
    "pae_local_mean":         _div(31.75),
    "pae_distal_mean":        _div(31.75),
    "pae_asymmetry":          _div(10.0),
    "pae_band_means":         _div(31.75),
}


# Human-readable inverse recipes for the class docstring (units + formulas).
# Kept in sync with NORMALIZATION_RECIPES.
INVERSE_FORMULAS: Dict[str, str] = {
    "ss3":                    "identity (one-hot)",
    "ss8":                    "identity (one-hot)",
    "rasa":                   "identity (clipped at 1.0)",
    "phi_psi_sincos":         "x * 2 - 1   (in [-1, 1])",
    "bfactor":                "x * 100     (lossy when ≥1, B-factors > 100 Å² are clipped)",
    "depth":                  "x * 15      (lossy when ≥1, depths > 15 Å are clipped)",
    "plddt":                  "x * 100     (AlphaFold pLDDT ∈ [0, 100])",
    "plddt_disorder":         "identity (boolean from pLDDT < threshold)",
    "plddt_tier":             "identity (4-dim one-hot over <50/50-70/70-90/≥90)",
    "chi1_sincos":            "x * 2 - 1   (in [-1, 1])",
    "chi2_sincos":            "x * 2 - 1   (in [-1, 1])",
    "ca_centroid_dist":       "x * 40      (lossy when ≥1, distances > 40 Å are clipped)",
    "ca_centroid_dist_norm":  "x * 2       (lossy when ≥1, > 2 Rg are clipped)",
    "contact_count_8A":       "x * 30      (lossy when ≥1, counts > 30 are clipped)",
    "contact_count_12A":      "x * 80      (lossy when ≥1, counts > 80 are clipped)",
    "hse":                    "x * 30      (HSE half-sphere counts; lossy when ≥1)",
    "pae_row_mean":           "x * 31.75   (Å, AF PAE saturation cap)",
    "pae_row_min":            "x * 31.75   (Å)",
    "pae_row_max":            "x * 31.75   (Å)",
    "pae_local_mean":         "x * 31.75   (Å)",
    "pae_distal_mean":        "x * 31.75   (Å)",
    "pae_asymmetry":          "x * 10      (Å, lossy when ≥1; clipped at 10 Å)",
    "pae_band_means":         "x * 31.75   (Å, applied per band)",
}


REGISTRY: Dict[str, Dict] = {
    # ``subcategory`` follows the AAontology naming convention
    # (descriptive name, source / detail in parentheses) so the
    # CPPPlot.feature_map y-axis label reads cleanly. ``dim_names`` stay
    # short — they become per-column labels in df_scales and the per-row
    # labels of the feature map's y-axis when CPPPlot uses scale_name.
    "ss3": {
        "method": ENCODER_DSSP, "num_dims": 3,
        "dim_names": ["ss_helix", "ss_strand", "ss_coil"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "Secondary structure (3-state)",
    },
    "ss8": {
        "method": ENCODER_DSSP, "num_dims": 8,
        "dim_names": ["ss_H", "ss_B", "ss_E", "ss_G",
                      "ss_I", "ss_T", "ss_S", "ss_blank"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "Secondary structure (8-state)",
    },
    "rasa": {
        "method": ENCODER_DSSP, "num_dims": 1,
        "dim_names": ["rasa"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "Relative ASA (Tien)",
    },
    "phi_psi_sincos": {
        "method": ENCODER_DSSP, "num_dims": 4,
        "dim_names": ["phi_sin", "phi_cos", "psi_sin", "psi_cos"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "Backbone dihedral (sin/cos)",
    },
    "bfactor": {
        "method": ENCODER_PDB, "num_dims": 1,
        "dim_names": ["bfactor"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "B-factor (CA mean)",
    },
    "depth": {
        "method": ENCODER_PDB, "num_dims": 1,
        "dim_names": ["depth"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "Residue depth (MSMS)",
    },
    # AF model-file features. ``plddt`` and ``bfactor`` are intentionally
    # separate keys — they share arithmetic (B-factor column read) but
    # carry different subcategory labels so the redundancy filter / user
    # can tell them apart.
    "plddt": {
        "method": ENCODER_PDB, "num_dims": 1,
        "dim_names": ["plddt"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "AlphaFold pLDDT (raw)",
    },
    "plddt_disorder": {
        "method": ENCODER_PDB, "num_dims": 1,
        "dim_names": ["plddt_low"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "AlphaFold disorder (pLDDT<70)",
    },
    "plddt_tier": {
        "method": ENCODER_PDB, "num_dims": 4,
        "dim_names": ["plddt_very_low", "plddt_low_tier",
                      "plddt_confident", "plddt_very_high"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "AlphaFold pLDDT (4-tier)",
    },
    "chi1_sincos": {
        "method": ENCODER_PDB, "num_dims": 2,
        "dim_names": ["chi1_sin", "chi1_cos"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "Side-chain chi1 (sin/cos)",
    },
    "chi2_sincos": {
        "method": ENCODER_PDB, "num_dims": 2,
        "dim_names": ["chi2_sin", "chi2_cos"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "Side-chain chi2 (sin/cos)",
    },
    "ca_centroid_dist": {
        "method": ENCODER_PDB, "num_dims": 1,
        "dim_names": ["centroid_dist"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "CA centroid distance",
    },
    "ca_centroid_dist_norm": {
        "method": ENCODER_PDB, "num_dims": 1,
        "dim_names": ["centroid_dist_norm"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "CA centroid distance (Rg-norm)",
    },
    "contact_count_8A": {
        "method": ENCODER_PDB, "num_dims": 1,
        "dim_names": ["contacts_8A"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "CA-CA contacts (8 A)",
    },
    "contact_count_12A": {
        "method": ENCODER_PDB, "num_dims": 1,
        "dim_names": ["contacts_12A"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "CA-CA contacts (12 A)",
    },
    "hse": {
        "method": ENCODER_PDB, "num_dims": 2,
        "dim_names": ["hse_up", "hse_down"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "Half-sphere exposure (HSE-CA)",
    },
    # AF PAE sidecar features.
    "pae_row_mean": {
        "method": ENCODER_PAE, "num_dims": 1,
        "dim_names": ["pae_mean"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "AlphaFold PAE (row mean)",
    },
    "pae_row_min": {
        "method": ENCODER_PAE, "num_dims": 1,
        "dim_names": ["pae_min"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "AlphaFold PAE (row min)",
    },
    "pae_row_max": {
        "method": ENCODER_PAE, "num_dims": 1,
        "dim_names": ["pae_max"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "AlphaFold PAE (row max)",
    },
    "pae_local_mean": {
        "method": ENCODER_PAE, "num_dims": 1,
        "dim_names": ["pae_local"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "AlphaFold PAE (local mean)",
    },
    "pae_distal_mean": {
        "method": ENCODER_PAE, "num_dims": 1,
        "dim_names": ["pae_distal"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "AlphaFold PAE (distal mean)",
    },
    "pae_asymmetry": {
        "method": ENCODER_PAE, "num_dims": 1,
        "dim_names": ["pae_asymmetry"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "AlphaFold PAE (asymmetry)",
    },
    "pae_band_means": {
        "method": ENCODER_PAE, "num_dims": 3,
        "dim_names": ["pae_band_close", "pae_band_mid", "pae_band_far"],
        "category": CATEGORY_STRUCTURE,
        "subcategory": "AlphaFold PAE (3-band means)",
    },
}

VALID_FEATURE_KEYS = sorted(REGISTRY.keys())


def validate_feature_keys(features: List[str],
                          allowed_method: Optional[str] = None) -> None:
    """Validate that every entry in ``features`` is a registered feature_key.

    Parameters
    ----------
    features : list of str
        Feature keys to validate.
    allowed_method : str or None
        If provided (``'encode_dssp'`` / ``'encode_pdb'`` / ``'encode_pae'``),
        every key must belong to that encoder; mixed lists raise ``ValueError``.
    """
    if not isinstance(features, list) or len(features) == 0:
        raise ValueError(
            f"'features' ({features!r}) should be a non-empty list of "
            f"feature keys from {VALID_FEATURE_KEYS}")
    bad = [f for f in features if f not in REGISTRY]
    if bad:
        raise ValueError(
            f"'features' ({bad}) should be a subset of {VALID_FEATURE_KEYS}")
    if allowed_method is not None:
        wrong = [f for f in features if REGISTRY[f]["method"] != allowed_method]
        if wrong:
            raise ValueError(
                f"'features' ({wrong}) should be encoded by '{allowed_method}'; "
                f"use the matching StructurePreprocessor method for each "
                f"source group")


def normalize(feature_key: str, values: np.ndarray) -> np.ndarray:
    """Apply the registered min-max recipe for ``feature_key``.

    Returns the recipe output without copying when the recipe is identity;
    otherwise returns the numpy ndarray produced by the recipe. NaNs pass
    through (recipes use ``np.clip`` which preserves NaN).
    """
    if feature_key not in NORMALIZATION_RECIPES:
        raise RuntimeError(
            f"Internal: no NORMALIZATION_RECIPES entry for feature key "
            f"{feature_key!r}")
    recipe = NORMALIZATION_RECIPES[feature_key]
    return recipe(values)


def get_total_dims(features: List[str]) -> int:
    """Sum of ``num_dims`` across the supplied feature keys."""
    return sum(REGISTRY[f]["num_dims"] for f in features)


def get_dim_names(features: List[str]) -> List[str]:
    """Flat list of default dimension names in feature-key order."""
    names: List[str] = []
    for f in features:
        names.extend(REGISTRY[f]["dim_names"])
    return names


def get_categories(features: List[str]) -> List[str]:
    """One category label per dimension, in feature-key order.

    With the v1.1 palette, this is always ``'Structure'`` for every dim, but
    the helper signature stays per-dim for parity with
    ``get_subcategories`` and ``get_dim_names``.
    """
    cats: List[str] = []
    for f in features:
        cats.extend([REGISTRY[f]["category"]] * REGISTRY[f]["num_dims"])
    return cats


def get_subcategories(features: List[str]) -> List[str]:
    """One subcategory label per dimension, in feature-key order."""
    subs: List[str] = []
    for f in features:
        subs.extend([REGISTRY[f]["subcategory"]] * REGISTRY[f]["num_dims"])
    return subs

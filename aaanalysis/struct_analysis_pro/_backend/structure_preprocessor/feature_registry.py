"""
This is a script for the backend of the StructurePreprocessor: a central
registry mapping each canonical feature_key to its source method, output
dimensionality, default dimension names, and AAontology-style category /
subcategory labels. The frontend uses this registry both to dispatch the
encoders (``encode_dssp`` / ``encode_pdb``) and to assemble the
``(df_scales, df_cat)`` metadata pair from ``build_scales``.
"""
from typing import Dict, List, Optional

import aaanalysis.utils as ut


# I Helper Functions
# (none)


# II Main Functions
ENCODER_DSSP = "encode_dssp"
ENCODER_PDB = "encode_pdb"

REGISTRY: Dict[str, Dict] = {
    "ss3": {
        "method": ENCODER_DSSP, "num_dims": 3,
        "dim_names": ["ss_helix", "ss_strand", "ss_coil"],
        "category": "DSSP_SS", "subcategory": "3-state",
    },
    "ss8": {
        "method": ENCODER_DSSP, "num_dims": 8,
        "dim_names": ["ss_H", "ss_B", "ss_E", "ss_G",
                      "ss_I", "ss_T", "ss_S", "ss_blank"],
        "category": "DSSP_SS", "subcategory": "8-state",
    },
    "asa": {
        "method": ENCODER_DSSP, "num_dims": 1,
        "dim_names": ["asa"],
        "category": "DSSP_ASA", "subcategory": "absolute",
    },
    "rasa": {
        "method": ENCODER_DSSP, "num_dims": 1,
        "dim_names": ["rasa"],
        "category": "DSSP_ASA", "subcategory": "relative",
    },
    "phi_psi": {
        "method": ENCODER_DSSP, "num_dims": 2,
        "dim_names": ["phi", "psi"],
        "category": "Geometry", "subcategory": "dihedral_raw",
    },
    "phi_psi_sincos": {
        "method": ENCODER_DSSP, "num_dims": 4,
        "dim_names": ["phi_sin", "phi_cos", "psi_sin", "psi_cos"],
        "category": "Geometry", "subcategory": "dihedral_sincos",
    },
    "bfactor": {
        "method": ENCODER_PDB, "num_dims": 1,
        "dim_names": ["bfactor"],
        "category": "Flexibility", "subcategory": "bfactor_mean",
    },
    "depth": {
        "method": ENCODER_PDB, "num_dims": 1,
        "dim_names": ["depth"],
        "category": "Geometry", "subcategory": "residue_depth",
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
        If provided (``'encode_dssp'`` or ``'encode_pdb'``), every key must
        belong to that encoder; mixed lists raise ``ValueError``.
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
    """One category label per dimension, in feature-key order."""
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

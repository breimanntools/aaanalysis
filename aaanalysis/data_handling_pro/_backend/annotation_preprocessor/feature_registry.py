"""
This is a script for the backend of the AnnotationPreprocessor: a central
registry mapping each canonical annotation feature_key to its top-level
category, output dimensionality, default dimension names, subcategory label,
and the min-max normalization recipe applied at the encoder.

Two top-level categories pair with ``ut.DICT_COLOR_CAT``:

- ``'PTMs'`` (``'#B36BCB'``) — closed UniProt PTM/Processing vocabulary
  (MOD_RES, CARBOHYD, LIPID, DISULFID, CROSSLNK, and the SIGNAL / PROPEP /
  TRANSIT / SITE cleavage sources). Disulfide stays in PTMs.
- ``'Functional sites'`` (``'#2C6E9E'``) — UniProt BINDING / ACT_SITE /
  DNA_BIND seeds **plus** an open vocabulary of user/predictor keys registered
  at runtime (RFdiffusion hotspots, BindCraft interface residues, custom).

Every annotation feature is a single per-residue channel (``num_dims=1``)
holding a score in ``[0, 1]`` (presence features use ``1.0`` / ``0.0``;
predictor features carry the supplied confidence). The shared ``_clip_unit``
recipe is the source of truth: values are already in range, the clip only
guards against tiny float overshoot and preserves NaN for unresolved
positions.

The built-in keys are immutable. Open-vocabulary Functional-sites keys are
added per-instance (``AnnotationPreprocessor`` copies ``REGISTRY`` and mutates
its own copy), so global state never leaks between instances.
"""

from typing import Callable, Dict, List, Optional

import numpy as np

# Top-level category buckets. Pair with ``ut.DICT_COLOR_CAT``; the redundancy
# filter's ``check_cat=True`` arm groups by these. Fine-grained semantics live
# in ``subcategory``.
CATEGORY_PTM = "PTMs"
CATEGORY_FUNC = "Functional sites"

# Single logical encoder for every annotation key (parity with
# StructurePreprocessor's ENCODER_* constants).
ENCODER_ANNOT = "encode"


# I Helper Functions
def _clip_unit(x):
    # Annotation values are constructed in [0, 1]; clip guards float overshoot
    # of user scores and preserves NaN (np.clip passes NaN through).
    return np.clip(x, 0.0, 1.0)


# II Main Functions
NORMALIZATION_RECIPES: Dict[str, Callable] = {
    # PTMs (closed vocabulary)
    "phospho": _clip_unit,
    "glyco_n": _clip_unit,
    "glyco_o": _clip_unit,
    "lipid": _clip_unit,
    "disulfide": _clip_unit,
    "crosslink": _clip_unit,
    "mod_res_other": _clip_unit,
    "signal_cleavage": _clip_unit,
    "propep_cleavage": _clip_unit,
    "transit_cleavage": _clip_unit,
    "cleavage_site": _clip_unit,
    # Functional sites (built-in seeds)
    "binding": _clip_unit,
    "act_site": _clip_unit,
    "dna_bind": _clip_unit,
}


REGISTRY: Dict[str, Dict] = {
    # ---- PTMs ---------------------------------------------------------
    "phospho": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["phospho"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_phosphorylation",
    },
    "glyco_n": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["glyco_n"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_glycosylation_N",
    },
    "glyco_o": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["glyco_o"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_glycosylation_O",
    },
    "lipid": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["lipid"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_lipidation",
    },
    "disulfide": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["disulfide"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_disulfide",
    },
    "crosslink": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["crosslink"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_crosslink",
    },
    "mod_res_other": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["mod_res_other"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_mod_res_other",
    },
    "signal_cleavage": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["signal_cleavage"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_cleavage_signal",
    },
    "propep_cleavage": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["propep_cleavage"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_cleavage_propeptide",
    },
    "transit_cleavage": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["transit_cleavage"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_cleavage_transit",
    },
    "cleavage_site": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["cleavage_site"],
        "category": CATEGORY_PTM,
        "subcategory": "PTM_cleavage_site",
    },
    # ---- Functional sites (built-in seeds) ----------------------------
    "binding": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["binding"],
        "category": CATEGORY_FUNC,
        "subcategory": "FUNC_binding",
    },
    "act_site": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["act_site"],
        "category": CATEGORY_FUNC,
        "subcategory": "FUNC_active_site",
    },
    "dna_bind": {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": ["dna_bind"],
        "category": CATEGORY_FUNC,
        "subcategory": "FUNC_dna_binding",
    },
}

# Mapping from raw UniProt feature ``type`` to the registry key(s) it can feed.
# MOD_RES / CARBOHYD / SITE are description-routed by the mapper, so they list
# the candidate keys rather than a single target.
PTM_FEATURE_TYPES = (
    "MOD_RES",
    "CARBOHYD",
    "LIPID",
    "DISULFID",
    "CROSSLNK",
    "SIGNAL",
    "PROPEP",
    "TRANSIT",
    "SITE",
)
FUNC_FEATURE_TYPES = ("BINDING", "ACT_SITE", "DNA_BIND")

# Bond features expand to two single-residue endpoints sharing a bond_id.
BOND_FEATURE_TYPES = ("DISULFID", "CROSSLNK")
# Processing features whose span END is the cleavage P1 anchor.
PROCESSING_FEATURE_TYPES = ("SIGNAL", "PROPEP", "TRANSIT")

BUILTIN_FEATURE_KEYS = sorted(REGISTRY.keys())


def make_instance_registry() -> Dict[str, Dict]:
    """Return a deep-enough copy of the built-in REGISTRY for per-instance use.

    The values are flat dicts of immutables; a shallow copy of each entry is
    sufficient to keep instance-local mutations (open-vocabulary registration)
    from touching the module-global built-ins.
    """
    return {k: dict(v) for k, v in REGISTRY.items()}


def register_functional_key(
    registry: Dict[str, Dict],
    recipes: Dict[str, Callable],
    key: str,
    subcategory: Optional[str] = None,
    normalization: Optional[Callable] = None,
) -> None:
    """Add an open-vocabulary Functional-sites key to an instance registry.

    Parameters
    ----------
    registry : dict
        The instance-local registry to mutate (from :func:`make_instance_registry`).
    recipes : dict
        The instance-local normalization-recipe map to mutate.
    key : str
        The new feature key (the user/predictor ``label``).
    subcategory : str, optional
        Fine-grained label; defaults to ``'FUNC_<key>'``.
    normalization : callable, optional
        Recipe applied to the raw per-residue values; defaults to ``_clip_unit``
        (values must already lie in ``[0, 1]``).
    """
    registry[key] = {
        "method": ENCODER_ANNOT,
        "num_dims": 1,
        "dim_names": [key],
        "category": CATEGORY_FUNC,
        "subcategory": subcategory if subcategory is not None else f"FUNC_{key}",
    }
    recipes[key] = normalization if normalization is not None else _clip_unit


def validate_feature_keys(
    features: List[str], registry: Optional[Dict[str, Dict]] = None
) -> None:
    """Validate that every entry in ``features`` is a registered feature key.

    Parameters
    ----------
    features : list of str
        Feature keys to validate.
    registry : dict, optional
        Registry to validate against; defaults to the built-in ``REGISTRY``.
        Pass the instance registry to allow auto-registered Functional keys.
    """
    reg = REGISTRY if registry is None else registry
    valid = sorted(reg.keys())
    if not isinstance(features, list) or len(features) == 0:
        raise ValueError(
            f"'features' ({features!r}) should be a non-empty list of "
            f"feature keys from {valid}"
        )
    bad = [f for f in features if f not in reg]
    if bad:
        raise ValueError(f"'features' ({bad}) should be a subset of {valid}")


def normalize(
    feature_key: str, values: np.ndarray, recipes: Optional[Dict[str, Callable]] = None
) -> np.ndarray:
    """Apply the registered min-max recipe for ``feature_key``.

    NaNs pass through (recipes use ``np.clip`` which preserves NaN).
    """
    rec = NORMALIZATION_RECIPES if recipes is None else recipes
    if feature_key not in rec:
        raise RuntimeError(
            f"Internal: no normalization recipe for feature key " f"{feature_key!r}"
        )
    return rec[feature_key](values)


def get_total_dims(
    features: List[str], registry: Optional[Dict[str, Dict]] = None
) -> int:
    """Sum of ``num_dims`` across the supplied feature keys."""
    reg = REGISTRY if registry is None else registry
    return sum(reg[f]["num_dims"] for f in features)


def get_dim_names(
    features: List[str], registry: Optional[Dict[str, Dict]] = None
) -> List[str]:
    """Flat list of default dimension names in feature-key order."""
    reg = REGISTRY if registry is None else registry
    names: List[str] = []
    for f in features:
        names.extend(reg[f]["dim_names"])
    return names


def get_categories(
    features: List[str], registry: Optional[Dict[str, Dict]] = None
) -> List[str]:
    """One category label per dimension, in feature-key order."""
    reg = REGISTRY if registry is None else registry
    cats: List[str] = []
    for f in features:
        cats.extend([reg[f]["category"]] * reg[f]["num_dims"])
    return cats


def get_subcategories(
    features: List[str], registry: Optional[Dict[str, Dict]] = None
) -> List[str]:
    """One subcategory label per dimension, in feature-key order."""
    reg = REGISTRY if registry is None else registry
    subs: List[str] = []
    for f in features:
        subs.extend([reg[f]["subcategory"]] * reg[f]["num_dims"])
    return subs

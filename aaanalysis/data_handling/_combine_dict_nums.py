"""
This is a script for the frontend of ``combine_dict_nums``, a source-agnostic
top-level utility that concatenates multiple ``dict_num`` inputs (e.g.
``dict_dssp``, ``dict_pdb``, ``dict_embeddings``) into a single ``dict_num``
along the D axis per entry. The result feeds
:meth:`NumericalFeature.get_parts` and downstream :meth:`CPP.run_num`.
"""
from typing import Dict, List

import numpy as np


# I Helper Functions
def _check_list_of_dicts(dict_nums) -> None:
    """Reject anything that isn't a non-empty list of dicts."""
    if not isinstance(dict_nums, list):
        raise ValueError(
            f"'dict_nums' ({type(dict_nums).__name__}) should be a list of "
            f"dict[str, np.ndarray] inputs")
    if len(dict_nums) == 0:
        raise ValueError(
            f"'dict_nums' (len=0) should be a non-empty list of "
            f"dict[str, np.ndarray] inputs")
    for i, d in enumerate(dict_nums):
        if not isinstance(d, dict):
            raise ValueError(
                f"'dict_nums[{i}]' ({type(d).__name__}) should be a dict "
                f"mapping entry to np.ndarray of shape (L, D)")


def _check_entry_sets_match(dict_nums: List[Dict[str, np.ndarray]]) -> None:
    """All inputs must share the exact same entry set."""
    first_keys = set(dict_nums[0].keys())
    for i, d in enumerate(dict_nums[1:], start=1):
        keys = set(d.keys())
        if keys != first_keys:
            missing = first_keys - keys
            extra = keys - first_keys
            raise ValueError(
                f"'dict_nums[{i}]' (missing={sorted(missing)[:5]}, "
                f"extra={sorted(extra)[:5]}) should share the same entry "
                f"set as 'dict_nums[0]'")


def _check_arrays_shape_per_entry(
    dict_nums: List[Dict[str, np.ndarray]]
) -> None:
    """Per entry, all inputs must share the same L; each array is 2D (L, D)."""
    for entry in dict_nums[0]:
        L_ref = None
        for i, d in enumerate(dict_nums):
            arr = d[entry]
            if not isinstance(arr, np.ndarray):
                raise ValueError(
                    f"'dict_nums[{i}][{entry!r}]' "
                    f"({type(arr).__name__}) should be np.ndarray "
                    f"of shape (L, D)")
            if arr.ndim != 2:
                raise ValueError(
                    f"'dict_nums[{i}][{entry!r}]' (ndim={arr.ndim}) "
                    f"should be 2D of shape (L, D)")
            if L_ref is None:
                L_ref = arr.shape[0]
            elif arr.shape[0] != L_ref:
                raise ValueError(
                    f"'dict_nums[{i}][{entry!r}].shape[0]' "
                    f"({arr.shape[0]}) should equal L="
                    f"{L_ref} (from 'dict_nums[0][{entry!r}]')")


# II Main Functions
def combine_dict_nums(
    dict_nums: List[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Concatenate multiple per-residue ``dict_num`` inputs along the D axis.

    Source-agnostic: works with any combination of dicts whose values are
    ``(L_entry, D_i)`` ndarrays — typically
    :meth:`StructurePreprocessor.encode_dssp` /
    :meth:`StructurePreprocessor.encode_pdb` outputs, but also user-supplied
    per-residue embeddings or other per-residue numerical representations.

    Parameters
    ----------
    dict_nums : list of dict[str, np.ndarray]
        Each input is a ``{entry: (L_entry, D_i) ndarray}``. All inputs must
        share the same entry set; per entry, all inputs must share the same
        ``L``. The D axis is concatenated; output's D equals the sum of
        input Ds.

    Returns
    -------
    dict_num : dict[str, np.ndarray]
        ``{entry: (L_entry, D_total) ndarray}`` ready for
        :meth:`NumericalFeature.get_parts`.

    Raises
    ------
    ValueError
        If ``dict_nums`` is not a non-empty list of dicts, if the entry sets
        diverge across inputs, if any value is not a 2D ``np.ndarray``, or
        if the per-entry ``L`` differs across inputs.

    Examples
    --------
    .. include:: examples/combine_dict_nums.rst
    """
    # Validate
    _check_list_of_dicts(dict_nums)
    _check_entry_sets_match(dict_nums)
    _check_arrays_shape_per_entry(dict_nums)
    # Concatenate
    return {entry: np.concatenate([d[entry] for d in dict_nums], axis=1)
            for entry in dict_nums[0]}

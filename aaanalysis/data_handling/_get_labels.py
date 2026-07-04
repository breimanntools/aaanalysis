"""
This is a script for the frontend of the get_labels function, deriving a binary
label vector from a sequence DataFrame's label column.
"""
from typing import Any
import numpy as np
import pandas as pd

import aaanalysis.utils as ut


# I Helper Functions
def check_match_df_positive_label(df=None, col_label=None, positive_label=None) -> None:
    """Check that the positive label value is present in the label column."""
    present = set(df[col_label].tolist())
    if positive_label not in present:
        raise ValueError(f"'positive_label' ({positive_label}) is not among the values of "
                         f"column '{col_label}' ({sorted(present, key=str)}).")


# II Main Functions
def get_labels(df: pd.DataFrame,
               positive_label: Any = 1,
               col_label: str = "label",
               ) -> np.ndarray:
    """
    Derive a binary ``int`` label vector from a column of a sequence DataFrame.

    Maps the value flagged as positive (``positive_label``) onto ``1`` and every other
    value onto ``0``, the binary encoding consumed across the package (e.g. by
    :meth:`CPP.run`, :class:`TreeModel`, and the ``labels`` argument of most tools).
    This is the single-call form of the recurring ``(df[col] == x).astype(int).to_numpy()``
    expression.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    df : pd.DataFrame, shape (n_samples, n_seq_info)
        Sequence DataFrame (``df_seq``) containing the label column ``col_label``.
    positive_label : int or str, default=1
        Value in ``col_label`` marking the positive class. All rows equal to it become
        ``1``; all remaining rows become ``0``. Must be present in ``col_label``.
    col_label : str, default='label'
        Name of the column holding the (multi-value or already binary) labels.

    Returns
    -------
    labels : array-like, shape (n_samples,)
        Binary ``int`` label vector (``1`` = positive, ``0`` = otherwise), row-aligned
        to ``df``.

    Notes
    -----
    * The result equals ``(df[col_label] == positive_label).astype(int).to_numpy()``.
    * Pass the resulting vector directly as the ``labels`` argument of CPP, TreeModel,
      or other tools. For Positive-Unlabeled mining keep the package ``1`` (positive) /
      ``2`` (unlabeled) markers instead and pass ``X_pos`` / ``X_unlabeled`` to :meth:`dPULearn.fit`.

    Examples
    --------
    .. include:: examples/get_labels.rst
    """
    # Check input
    ut.check_str(name="col_label", val=col_label, accept_none=False)
    ut.check_df(name="df", df=df, cols_required=col_label)
    check_match_df_positive_label(df=df, col_label=col_label, positive_label=positive_label)
    # Derive binary int label vector
    labels = (df[col_label] == positive_label).astype(int).to_numpy()
    return labels

"""
This is a script for filtering scales from df_scales, such as using correlation or subcategory coverage.
"""

from typing import Literal

import pandas as pd

import aaanalysis.utils as ut

LIST_DATASETS_WITH_FEATURES = ["DOM_GSEC"]
FOLDER_FEATURES = ut.FOLDER_DATA + "features" + ut.SEP


# I Helper Functions
def check_name(name=None) -> None:
    """Check provided names of dataset"""
    if name not in LIST_DATASETS_WITH_FEATURES:
        raise ValueError(f"'name' should be one of: {LIST_DATASETS_WITH_FEATURES}")


# II Main Functions
def load_features(name: Literal["DOM_GSEC"] = "DOM_GSEC") -> pd.DataFrame:
    """
    Load feature sets for protein benchmarking datasets.

    Features are only provided for in-depth analyzed datasets available from the :func:`load_dataset`
    function. These are as follows:

        - 'DOM_GSEC' ([Breimann25]_)

    .. versionadded:: 0.1.3

    Parameters
    ----------
    name : str, default='DOM_GSEC'
        The name of the dataset for which features are loaded.

    Returns
    -------
    df_feat : pd.DataFrame, shape (n_features, n_feature_info)
        Feature DataFrame with one row per selected feature and the same
        columns emitted by :meth:`CPP.run` (feature id, statistical measures,
        and feature importance). See :meth:`CPP.run` for column definitions.

    Examples
    --------
    .. include:: examples/load_features.rst
    """
    # Check input
    check_name(name=name)
    # Load features
    df_feat = ut.read_csv_cached(FOLDER_FEATURES + f"FEATURES_{name}.{ut.STR_FILE_TYPE}")
    return df_feat

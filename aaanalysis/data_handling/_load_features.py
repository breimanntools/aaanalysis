"""
This is a script for filtering scales from df_scales, such as using correlation or subcategory coverage.
"""

import aaanalysis.utils as ut

LIST_DATASETS_WITH_FEATURES = ["DOM_GSEC"]
FOLDER_FEATURES = ut.FOLDER_DATA + "features" + ut.SEP


# I Helper Functions
def check_name(name=None):
    """Check provided names of dataset"""
    if name not in LIST_DATASETS_WITH_FEATURES:
        raise ValueError(f"'name' should be one of: {LIST_DATASETS_WITH_FEATURES}")


# II Main Functions
def load_features(name="DOM_GSEC"):
    """
    Load feature sets for protein benchmarking datasets.

    Features are only provided for in-depth analyzed datasets available from the :func:`load_dataset`
    function. These are as follows:

        - 'DOM_GSEC' ([Breimann25a]_)

    Parameters
    ----------
    name : str, default='DOM_GSEC'
        The name of the dataset for which features are loaded.

    Returns
    -------
    df_feat : pd.DataFrame, shape (n_features, n_feature_info)
        A DataFrame with features and their statistical measures as provided :meth:`CPP.run` method,
        including their feature importance.

    Examples
    --------
    .. include:: examples/load_features.rst
    """
    # Check input
    check_name(name=name)
    # Load features
    df_feat = ut.read_csv_cached(FOLDER_FEATURES + f"FEATURES_{name}.{ut.STR_FILE_TYPE}")
    return df_feat

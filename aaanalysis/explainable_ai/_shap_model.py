"""
This is a script for the processing SHAP values, primarily for the combination of SHAP with CPP.
SHAP models are not included due to instability of SHAP package development. ShapModel should solely work with
SHAP value matrix.
"""
import time
import pandas as pd
import numpy as np

import aaanalysis.utils as ut

# Settings


# I Helper Functions
# Get info for COL_FEAT_IMPORTANCE = "feat_importance"
# COO_FEAT_IMP_STD = "feat_importance_std"
# COL_FEAT_IMPACT = "feat_impact"

# II Main Functions
class ShapModel:
    """A wrapper for Tree explainable_ai from SHAP package"""
    def __init__(self, model=None):
        """"""

    def fit(self, n_epochs=10, return_models=False):
        """Fit provided tree based model n_epochs time and compute average feature importance"""

    def eval(self):
        """"""

    # TODO rename and add other functions (e.g., fuzzy labeling)
    @staticmethod
    def add_feat_impact(df_feat=None, col_shap="shap_value", name_feat_impact="feat_impact"):
        """
        Convert SHAP values into feature impact/importance and add to DataFrame.

        Parameters
        ----------
        df_feat: :class:`pandas.DataFrame`
            Feature DataFrame to which the feature impact will be added.
        col_shap: str, default = 'shap_value'
            Column name of `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ values in the feature DataFrame.
        name_feat_impact: str, default = 'feat_impact'
            Column name of feature impact or feature importance that will be added to the feature DataFrame.

        Returns
        -------
        df_feat: :class:`pandas.DataFrame`
            Feature DataFrame including feature impact.

        Notes
        -----
        - SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.
        - SHAP values represent a feature's responsibility for a change in the model output.
        - Missing values are accepted in SHAP values.

        """

        # Check input
        df_feat = df_feat.copy()
        ut.check_str(name="name_feat_impact", val=name_feat_impact)
        ut.check_str(name="col_shap", val=col_shap)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        #ut.check_col_in_df(df=df_feat, name_df="df_feat", col=col_shap, col_type=[float, int])
        #ut.check_col_in_df(df=df_feat, name_df="df_feat", col=name_feat_impact, error_if_exists=True)

        # Compute feature impact (accepting missing values)
        shap_values = np.array(df_feat[col_shap])
        feat_impact = shap_values / np.nansum(np.abs(shap_values)) * 100
        shap_loc = df_feat.columns.get_loc(col_shap)
        df_feat.insert(shap_loc + 1, name_feat_impact, feat_impact)
        return df_feat

    @staticmethod
    def fuzzly_labeling():
        """Perform fuzzy labeling for selected sample"""


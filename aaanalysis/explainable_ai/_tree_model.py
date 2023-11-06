"""
This is a script for the frontend of the TreeModel class used to obtain feature importance reproducibly.
To this end, random forest models are trained over multiple rounds with their results being averaged.
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
class TreeModel:
    """A wrapper for Tree based prediction models and Tree explainable_ai from SHAP package to
    explain prediction (typically binary classification) results at global and individual level"""
    def __init__(self, model=None):
        """"""

    def fit(self, n_epochs=10, rcf=True, return_models=False):
        """Fit provided tree based model n_epochs time and compute average feature importance"""

    def eval(self):
        """"""

    def add_feat_import(self, df_feat=None):
        """"""

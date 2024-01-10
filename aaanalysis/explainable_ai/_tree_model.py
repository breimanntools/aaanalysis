"""
This is a script for the frontend of the TreeModel class used to obtain feature importance reproducibly.
To this end, random forest models are trained over multiple rounds with their results being averaged.
"""
import time
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.base import clone

import aaanalysis.utils as ut

# Settings


# I Helper Functions
# Get info for COL_FEAT_IMPORTANCE = "feat_importance"
# COO_FEAT_IMP_STD = "feat_importance_std"
# COL_FEAT_IMPACT = "feat_impact"

# II Main Functions
class TreeModel:
    """
    Tree Model class: A wrapper for tree-based prediction models to obtain feature importance.
    """
    def __init__(self,
                 model_class_rcf=None,
                 list_model_classes_fit=None):
        self.model_class_rcf = model_class_rcf
        self.list_model_classes_fit = list_model_classes_fit
        """"""

    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D = None,
            label_test : int = 1,
            label_ref : int = 0,
            n_rounds=10,
            n_folds=5,
            rcf=True,
            n_feat_min=25,
            n_feat_max=50,
            eval_score="auc"

            ):
        """Fit provided tree based model n_rounds time and compute average feature importance."""
        # Check input
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_requiered=[label_test, label_ref],
                                 len_requiered=len(X), allow_other_vals=False)

        # Initialize arrays to store feature importances
        importances = {model_class.__name__: [] for model_class in self.list_model_classes_fit}

        # Initialize arrays to store feature importances for each fold and round
        importances = {model_class.__name__: [] for model_class in self.list_model_classes_fit}

        for _ in range(n_rounds):
            kf = KFold(n_splits=n_folds, shuffle=True)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = labels[train_index], labels[test_index]

                # Feature selection using RFE if enabled
                if rcf:
                    selector = RFE(self.model_class_rcf(), n_features_to_select=n_feat_min)
                    X_train_selected = selector.fit_transform(X_train, y_train)
                    X_test_selected = selector.transform(X_test)
                else:
                    X_train_selected, X_test_selected = X_train, X_test

                # Fit the models and collect feature importances for each fold
                for model_class in self.list_model_classes_fit:
                    model = clone(model_class()).fit(X_train_selected, y_train)
                    importances[model_class.__name__].append(model.feature_importances_)

        # Calculating average and standard deviation of feature importances
        avg_importances = {model: np.mean(importances, axis=0) for model, importances in importances.items()}
        std_importances = {model: np.std(importances, axis=0) for model, importances in importances.items()}

        return avg_importances, std_importances
    def eval(self):
        """"""

    def add_feat_import(self, df_feat=None):
        """"""

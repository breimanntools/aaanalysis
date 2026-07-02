"""
This is a script for the backend of the AAPred class: fitting models and obtaining prediction scores.
"""
import numpy as np

import aaanalysis.utils as ut


# I Helper Functions
def _positive_proba(model, X, label_pos):
    """Return the positive-class probability column for a fitted model."""
    proba = np.asarray(model.predict_proba(X))
    classes = list(model.classes_)
    idx_pos = classes.index(label_pos)
    return proba[:, idx_pos]


# Minimal built-in hyperparameter grids, keyed by estimator class name. Used when
# hyperparameter optimization is requested without an explicit grid. Models absent
# here are fit directly (no search).
_DEFAULT_PARAM_GRIDS = {
    "SVC": {"C": [0.1, 1.0, 10.0]},
    "RandomForestClassifier": {"n_estimators": [100, 300], "max_depth": [None, 10]},
    "ExtraTreesClassifier": {"n_estimators": [100, 300], "max_depth": [None, 10]},
    "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
    "KNeighborsClassifier": {"n_neighbors": [3, 5, 7]},
    "GradientBoostingClassifier": {"learning_rate": [0.05, 0.1], "n_estimators": [100, 300]},
    "MLPClassifier": {"alpha": [1e-4, 1e-3]},
}


# II Main Functions
def fit_models(X, labels, list_estimators=None, list_param_grids=None,
               optimize_hyperparams=False, n_cv=5, random_state=None):
    """Fit every configured estimator on the full data and return the fitted estimators.

    Each estimator in ``list_estimators`` is cloned before fitting (so the stored
    configuration is never mutated). When ``optimize_hyperparams`` is set, each model is
    tuned by ``GridSearchCV`` over its entry in ``list_param_grids`` (or a built-in default
    grid), and the best estimator is kept. Models with no available grid are fit directly.
    """
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    list_models = []
    for i, estimator in enumerate(list_estimators):
        model = clone(estimator)
        grid = None
        if optimize_hyperparams:
            if list_param_grids is not None and list_param_grids[i]:
                grid = list_param_grids[i]
            else:
                grid = _DEFAULT_PARAM_GRIDS.get(type(estimator).__name__)
        if grid:
            cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)
            search = GridSearchCV(model, grid, cv=cv)
            search.fit(X, labels)
            model = search.best_estimator_
        else:
            model.fit(X, labels)
        list_models.append(model)
    return list_models


def predict_proba_models(X, list_models=None, label_pos=1):
    """Average positive-class probability across fitted models.

    Returns the mean positive-class score per sample and its std across models
    (0 for a single model).
    """
    probas = np.vstack([_positive_proba(model, X, label_pos=label_pos) for model in list_models])
    pred = probas.mean(axis=0)
    pred_std = probas.std(axis=0) if len(list_models) > 1 else np.zeros(len(pred))
    return pred, pred_std

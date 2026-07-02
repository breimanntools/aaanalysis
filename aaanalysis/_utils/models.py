"""Shared classifier registry: map a model-name string to a configured estimator.

One source of truth for the ``name -> sklearn estimator`` mapping used across the
package (AAPred, ``predict_samples``, ``find_features``), so the name list does not
drift between call sites. The roster is kept deliberately small — the four standard
families the package already uses. Any other estimator (MLP, gradient boosting,
xgboost, a voting/stacking ensemble, a full ``Pipeline``, ...) is used by passing a
configured sklearn estimator instance instead of a name; only names route here. The
default (``svm``) reproduces the linear-SVM recipe used throughout the γ-secretase
analysis.
"""
from .. import _constants as const


# I Helper Functions
def _svm(random_state=None):
    from sklearn.svm import SVC
    return SVC(kernel="linear", probability=True, random_state=random_state)


def _rf(random_state=None):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=random_state)


def _extra_trees(random_state=None):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(random_state=random_state)


def _log_reg(random_state=None):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, random_state=random_state)


_FACTORIES = {
    const.MODEL_SVM: _svm,
    const.MODEL_RF: _rf,
    const.MODEL_EXTRA_TREES: _extra_trees,
    const.MODEL_LOG_REG: _log_reg,
}


# II Main Functions
def get_cv_model_(name=None, random_state=None):
    """Return a fresh configured estimator for a registry ``name``.

    ``name`` is one of ``ut.LIST_PRED_MODELS``. ``random_state`` is injected where the
    estimator supports it. Raises ``ValueError`` for an unknown name; pass a configured
    sklearn estimator instance instead of a name to use any other model.
    """
    if name not in _FACTORIES:
        valid = ", ".join(list(_FACTORIES))
        raise ValueError(f"'model' name '{name}' is not in the registry. Valid names: {valid}. "
                         f"A configured sklearn estimator instance may be passed instead.")
    return _FACTORIES[name](random_state=random_state)

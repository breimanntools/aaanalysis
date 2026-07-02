"""Shared classifier registry: map a model-name string to a configured estimator.

One source of truth for the ``name -> sklearn estimator`` mapping used across the
package (AAPred, ``predict_samples``, ``find_features``, ``CPP.simplify``), so the
name list does not drift between call sites. A model can always be passed as a
configured sklearn estimator instance instead of a name; only names route here.

The roster is the standard sklearn families plus two meta-ensembles (voting /
stacking of RF + SVM + logistic regression). The default (``svm``) reproduces the
linear-SVM recipe used throughout the γ-secretase analysis. ``xgboost`` is a
reserved name that raises a clear install hint if the optional dependency is
absent — it is never imported at module load, keeping the core install light.
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


def _mlp(random_state=None):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(max_iter=500, random_state=random_state)


def _tree(random_state=None):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=random_state)


def _lda(random_state=None):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis()


def _gradient_boosting(random_state=None):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(random_state=random_state)


def _knn(random_state=None):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier()


def _voting(random_state=None):
    from sklearn.ensemble import VotingClassifier
    estimators = [(const.MODEL_RF, _rf(random_state)),
                  (const.MODEL_SVM, _svm(random_state)),
                  (const.MODEL_LOG_REG, _log_reg(random_state))]
    return VotingClassifier(estimators=estimators, voting="soft")


def _stacking(random_state=None):
    from sklearn.ensemble import StackingClassifier
    estimators = [(const.MODEL_RF, _rf(random_state)),
                  (const.MODEL_SVM, _svm(random_state)),
                  (const.MODEL_LOG_REG, _log_reg(random_state))]
    return StackingClassifier(estimators=estimators,
                              final_estimator=_log_reg(random_state))


def _xgboost(random_state=None):
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            f"Model '{const.MODEL_XGBOOST}' requires the optional 'xgboost' package. "
            f"Install it with 'pip install xgboost'."
        ) from e
    return XGBClassifier(random_state=random_state, use_label_encoder=False,
                         eval_metric="logloss")


_FACTORIES = {
    const.MODEL_SVM: _svm,
    const.MODEL_RF: _rf,
    const.MODEL_EXTRA_TREES: _extra_trees,
    const.MODEL_LOG_REG: _log_reg,
    const.MODEL_MLP: _mlp,
    const.MODEL_TREE: _tree,
    const.MODEL_LDA: _lda,
    const.MODEL_GRADIENT_BOOSTING: _gradient_boosting,
    const.MODEL_KNN: _knn,
    const.MODEL_VOTING: _voting,
    const.MODEL_STACKING: _stacking,
    const.MODEL_XGBOOST: _xgboost,
}


# II Main Functions
def get_cv_model_(name=None, random_state=None):
    """Return a fresh configured estimator for a registry ``name``.

    ``name`` is one of ``ut.LIST_PRED_MODELS`` (or the optional ``"xgboost"``).
    ``random_state`` is injected where the estimator supports it. Raises
    ``ValueError`` for an unknown name; ``ImportError`` (with an install hint) for
    an optional model whose dependency is missing.
    """
    if name not in _FACTORIES:
        valid = ", ".join(list(_FACTORIES))
        raise ValueError(f"'model' name '{name}' is not in the registry. Valid names: {valid}. "
                         f"A configured sklearn estimator instance may be passed instead.")
    return _FACTORIES[name](random_state=random_state)

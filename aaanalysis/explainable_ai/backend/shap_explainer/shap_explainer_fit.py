"""
This is a script for the backend of the ShapModel.fit() method.
"""
import numpy as np


# I Helper Functions
def _get_shap_values(shap_values):
    """Retrieve SHAP values from SHAP output."""
    shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    return shap_values


def _compute_shap_values(X, labels, model_class=None, model_kwargs=None,
                         explainer_class=None, explainer_kwargs=None):
    """Fit a model and compute SHAP values using the provided SHAP Explainer.."""
    model = model_class(**model_kwargs).fit(X, labels)
    explainer = explainer_class(model, X, **explainer_kwargs)
    shap_values_ = explainer.shap_values(X, y=labels)
    shap_values = _get_shap_values(shap_values=shap_values_)
    expected_value = explainer.expected_value
    expected_value = expected_value if np.isscalar(expected_value) else expected_value[1]
    return shap_values, expected_value


def _aggregate_shap_values(X, labels=None, list_model_classes=None, list_model_kwargs=None,
                           explainer_class=None, explainer_kwargs=None, ):
    """Aggregate SHAP values across multiple models."""
    n_samples, n_features = X.shape
    n_models = len(list_model_classes)
    shap_values_rounds = np.empty(shape=(n_samples, n_features, n_models))
    list_expected_value = []
    for i, model_class in enumerate(list_model_classes):
        model_kwargs = list_model_kwargs[i]
        shap_values, expected_value = _compute_shap_values(X, labels=labels,
                                                           model_class=model_class, model_kwargs=model_kwargs,
                                                           explainer_class=explainer_class,
                                                           explainer_kwargs=explainer_kwargs)
        shap_values_rounds[:, :, i] = shap_values
        list_expected_value.append(expected_value)
    shap_values = shap_values_rounds.mean(axis=2)
    exp_val = np.mean(list_expected_value)
    return shap_values, exp_val



# II Main Functions
def monte_carlo_shap_estimation(X, labels=None, list_model_classes=None, list_model_kwargs=None,
                                explainer_class=None, explainer_kwargs=None, n_rounds=5,
                                is_selected=None, fuzzy_labeling=False):
    """
    Compute Monte Carlo estimates of SHAP values for multiple models and feature selections.
    """
    n_samples, n_features = X.shape
    n_selection_rounds = len(is_selected)
    mc_shap_values = np.zeros(shape=(n_samples, n_features, n_rounds, n_selection_rounds))
    list_expected_value = []
    for j in range(n_rounds):
        for i, selected_features in enumerate(is_selected):
            labels_ = labels
            # Adjust fuzzy labels (labels between 0 and 1, e.g., 0.5 -> 50% 1 and 50% 0)
            if fuzzy_labeling:
                threshold = (i * (j + 1)) / (n_rounds * n_selection_rounds)
                labels_ = [int(x >= threshold) for x in labels]
            X_selected = X[:, selected_features]
            _shap_values, _exp_val = _aggregate_shap_values(X_selected, labels=labels_,
                                                            list_model_classes=list_model_classes,
                                                            list_model_kwargs=list_model_kwargs,
                                                            explainer_class=explainer_class,
                                                            explainer_kwargs=explainer_kwargs)
            mc_shap_values[:, selected_features, j, i] = _shap_values
            list_expected_value.append(_exp_val)
    # Averaging over rounds and selections
    shap_values = mc_shap_values.mean(axis=(2, 3))
    exp_val = np.mean(list_expected_value)
    return shap_values, exp_val

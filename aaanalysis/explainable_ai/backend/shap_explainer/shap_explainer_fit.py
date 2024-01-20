"""
This is a script for the backend of the ShapModel.fit() method.
"""
import numpy as np
import shap

import aaanalysis.utils as ut

# I Helper Functions
def _get_shap_values(shap_values):
    """Retrieve SHAP values from SHAP output."""
    # Return SHAP values for general SHAP model
    shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    return shap_values

"""
def _get_shap_values(shap_output):
    # Return SHAP values for general SHAP model
    if isinstance(shap_output, shap.Explanation):
        print("hit")
        shap_values = shap_output.values
        # Handle multi-class outputs (binary classification included)
        if len(shap_values.shape) == 3:
            # Selecting the SHAP values for the positive class
            shap_values = shap_values[..., 1]  # For binary classification
    else:
        shap_values = shap_output[1] if isinstance(shap_output, list) else shap_output
    print(shap_values.shape)
    return shap_values
"""

def _compute_shap_values(X, labels, model_class=None, model_kwargs=None,
                         explainer_class=None, explainer_kwargs=None):
    """Fit a model and compute SHAP values using the provided SHAP Explainer"""
    # Ensure that model_kwargs and explainer_kwargs are dictionaries
    if model_kwargs is None:
        model_kwargs = {}
    if explainer_kwargs is None:
        explainer_kwargs = {}
    # Fit the model
    model = model_class(**model_kwargs).fit(X, labels)
    # Determine the correct input for the explainer
    if explainer_class is None or explainer_class == shap.Explainer:
        # Use the General Explainer
        explainer = shap.Explainer(model)
        shap_output = explainer(X)
    else:
        # Use a specific explainer
        if explainer_class.__name__ in ['KernelExplainer', 'OtherExplainerNeedingPredict']:
            model_input = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
        else:
            model_input = model
        explainer = explainer_class(model_input, X, **explainer_kwargs)
        if 'y' in explainer.shap_values.__code__.co_varnames:
            shap_output = explainer.shap_values(X, y=labels)
        else:
            shap_output = explainer.shap_values(X)
    # Process SHAP values
    shap_values = _get_shap_values(shap_output)
    expected_value = explainer.expected_value if not isinstance(shap_output, shap.Explanation) else shap_output.base_values
    return shap_values, expected_value



def _aggregate_shap_values(X, labels=None, list_model_classes=None, list_model_kwargs=None,
                           explainer_class=None, explainer_kwargs=None):
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
                                is_selected=None, fuzzy_labeling=False, verbose=False):
    """Compute Monte Carlo estimates of SHAP values for multiple models and feature selections."""
    n_samples, n_features = X.shape
    n_selection_rounds = len(is_selected)
    mc_shap_values = np.zeros(shape=(n_samples, n_features, n_rounds, n_selection_rounds))
    list_expected_value = []
    if verbose:
        ut.print_start_progress(start_message=f"ShapExplainer starts Monte Carlo estimation of shap values over {n_rounds} rounds.")
    for i in range(n_rounds):
        for j, selected_features in enumerate(is_selected):
            if verbose:
                pct_progress = j / len(is_selected)
                ut.print_progress(i=i+pct_progress, n=n_rounds)
            labels_ = labels
            # Adjust fuzzy labels (labels between 0 and 1, e.g., 0.5 -> 50% 1 and 50% 0)
            if fuzzy_labeling:
                threshold = (j * (i + 1)) / (n_rounds * n_selection_rounds)
                f = lambda x: x if x in [0, 1] else int(x >= threshold)
                labels_ = [f(x) for x in labels]
            X_selected = X[:, selected_features]
            _shap_values, _exp_val = _aggregate_shap_values(X_selected, labels=labels_,
                                                            list_model_classes=list_model_classes,
                                                            list_model_kwargs=list_model_kwargs,
                                                            explainer_class=explainer_class,
                                                            explainer_kwargs=explainer_kwargs)
            mc_shap_values[:, selected_features, i, j] = _shap_values
            list_expected_value.append(_exp_val)
    # Averaging over rounds and selections
    if verbose:
        ut.print_end_progress(end_message=f"ShapExplainer finished Monte Carlo estimation and saved results.")
    shap_values = mc_shap_values.mean(axis=(2, 3))
    exp_val = np.mean(list_expected_value)
    return shap_values, exp_val

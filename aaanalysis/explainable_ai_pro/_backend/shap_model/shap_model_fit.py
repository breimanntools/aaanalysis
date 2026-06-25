"""
This is a script for the backend of the ShapModel.fit() method.
"""
import numpy as np
import shap

import aaanalysis.utils as ut

LIST_VERBOSE_shap_modelS = [shap.KernelExplainer]


# I Helper Functions
def _get_background_data(X, explainer_class=None, n_background_data=None):
    """Sample background data for Kernel explainer"""
    X = X.copy()
    if explainer_class == shap.KernelExplainer and n_background_data is not None:
        # Summarize background data using k-means if specified
        background_data = shap.kmeans(X, n_background_data)
    else:
        background_data = X
    return background_data


def _get_shap_values(shap_output, class_index=1):
    """Retrieve SHAP values for the specified class index from SHAP output."""
    if isinstance(shap_output, list):
        if len(shap_output) > class_index:
            shap_values = np.array(shap_output[class_index])
        else:
            # Fallback: Return entire list as a NumPy array
            shap_values = np.array(shap_output)
    elif isinstance(shap_output, np.ndarray):
        n_dim = shap_output.ndim
        if n_dim >= 3 and shap_output.shape[-1] > class_index:
            # Multi-class: Retrieve SHAP values for the specified class
            shap_values = shap_output[..., class_index]
            shap_values = np.expand_dims(shap_values, axis=-1)
        elif n_dim == 2:
            # Binary classification or regression
            shap_values = np.expand_dims(shap_output, axis=-1)
        else:
            # Fallback: Return input as np.array
            shap_values = shap_output
    else:
        # Fallback: Unsupported input type
        shap_values = np.array(shap_output)
    return shap_values


def _class_index_from_labels(labels, label_target_class=1):
    """Map the target class label to its index among the integer (non-fuzzy) classes."""
    label_classes = sorted(list(dict.fromkeys([x for x in labels if x == int(x)])))
    return label_classes.index(label_target_class)


def _compute_shap_values(X, labels, model_class=None, model_kwargs=None,
                         explainer_class=None, explainer_kwargs=None,
                         class_index=1, n_background_data=None):
    """Fit a model and compute SHAP values using the provided SHAP Explainer."""
    # Fit the model
    model = model_class(**model_kwargs).fit(X, labels)

    # Instantiate explainer class
    if explainer_class in [shap.KernelExplainer, shap.DeepExplainer, shap.GradientExplainer]:
        background_data = _get_background_data(X, explainer_class=explainer_class, n_background_data=n_background_data)
        model_input = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
        explainer = explainer_class(model_input, background_data, **explainer_kwargs)
    elif explainer_class == shap.LinearExplainer:
        explainer = explainer_class(model, X, **explainer_kwargs)
    else:
        explainer = explainer_class(model, **explainer_kwargs)

    # Compute SHAP values
    if 'y' in explainer.shap_values.__code__.co_varnames:
        shap_output = explainer.shap_values(X, y=labels)
    else:
        shap_output = explainer.shap_values(X)

    # Process SHAP values
    shap_values = _get_shap_values(shap_output, class_index=class_index)
    # Handle expected value for classification models
    expected_value = explainer.expected_value
    if not np.isscalar(expected_value):
        expected_value = expected_value[class_index] if len(expected_value) > class_index else expected_value[-1]
    return shap_values, expected_value


def _aggregate_shap_values(X, labels=None, list_model_classes=None, list_model_kwargs=None,
                           explainer_class=None, explainer_kwargs=None,
                           class_index=1, n_background_data=None):
    """Aggregate SHAP values across multiple models."""
    n_samples, n_features = X.shape
    n_models = len(list_model_classes)
    shap_values_rounds = np.empty(shape=(n_samples, n_features, n_models))
    list_expected_value = []
    for i, model_class in enumerate(list_model_classes):
        model_kwargs = list_model_kwargs[i]
        args = dict(model_class=model_class, model_kwargs=model_kwargs,
                    explainer_class=explainer_class, explainer_kwargs=explainer_kwargs,
                    class_index=class_index, n_background_data=n_background_data)
        shap_values, expected_value = _compute_shap_values(X, labels=labels, **args)

        # Ensure that shap_values has the expected shape (n_samples, n_features)
        if shap_values.shape != (X.shape[0], X.shape[1]):
            shap_values = shap_values[:, :, 0]

        shap_values_rounds[:, :, i] = shap_values
        list_expected_value.append(expected_value)
    shap_values = np.mean(shap_values_rounds, axis=2)
    exp_val = np.mean(list_expected_value)
    return shap_values, exp_val


def _seed_model_kwargs(list_model_kwargs, random_state=None, round_idx=0):
    """Derive per-round-seeded copies of the model kwargs.

    With a fixed ``random_state`` each round uses ``random_state + round_idx`` so the rounds
    differ (Monte-Carlo averaging) yet stay reproducible. With ``random_state=None`` the kwargs
    are returned unchanged (``random_state`` already ``None``), so every fit re-draws fresh entropy.
    """
    if random_state is None:
        return [dict(model_kwargs) for model_kwargs in list_model_kwargs]
    seeded = []
    for model_kwargs in list_model_kwargs:
        model_kwargs = dict(model_kwargs)
        model_kwargs["random_state"] = random_state + round_idx
        seeded.append(model_kwargs)
    return seeded


# II Main Functions
@ut.catch_backend_processing_error()
def monte_carlo_shap_estimation(X, labels=None, list_model_classes=None, list_model_kwargs=None,
                                explainer_class=None, explainer_kwargs=None, n_rounds=5,
                                is_selected=None, fuzzy_labeling=False, verbose=False,
                                label_target_class=1, n_background_data=None):
    """Compute Monte Carlo estimates of SHAP values for multiple models and feature selections."""
    # Get class index
    class_index = _class_index_from_labels(labels, label_target_class)
    # Create empty SHAP value matrix
    n_samples, n_features = X.shape
    n_selection_rounds = len(is_selected)
    mc_shap_values = np.zeros(shape=(n_samples, n_features, n_rounds, n_selection_rounds))
    # Compute SHAP values
    list_expected_value = []
    if verbose:
        ut.print_start_progress(start_message=f"ShapModel starts Monte Carlo estimation of SHAP values over {n_rounds} rounds.")

    for i in range(n_rounds):
        for j, selected_features in enumerate(is_selected):
            if verbose:
                pct_progress = j / len(is_selected)
                add_new_line = explainer_class in LIST_VERBOSE_shap_modelS
                ut.print_progress(i=i+pct_progress, n_total=n_rounds, add_new_line=add_new_line)
            # Adjust fuzzy labels (labels between 0 and 1, e.g., 0.5 -> 50% 1 and 50% 0)
            if fuzzy_labeling:
                threshold = ((j + 1) * (i + 1)) / (n_rounds * n_selection_rounds)
                labels_ = [x if x in [0, 1] else int(x >= threshold) for x in labels]
            else:
                labels_ = labels
            X_selected = X[:, selected_features]
            args = dict(list_model_classes=list_model_classes, list_model_kwargs=list_model_kwargs,
                        explainer_class=explainer_class, explainer_kwargs=explainer_kwargs,
                        class_index=class_index, n_background_data=n_background_data)
            _shap_values, _exp_val = _aggregate_shap_values(X_selected, labels=labels_, **args)
            mc_shap_values[:, selected_features, i, j] = _shap_values
            list_expected_value.append(_exp_val)

    # Averaging over rounds and selections
    if verbose:
        ut.print_end_progress(end_message=f"ShapModel finished Monte Carlo estimation and saved results.")
    shap_values = np.mean(mc_shap_values, axis=(2, 3))
    exp_val = np.mean(list_expected_value)
    return shap_values, exp_val


@ut.catch_backend_processing_error()
def interpolate_fuzzy_shap_estimation(X, labels=None, list_model_classes=None, list_model_kwargs=None,
                                      explainer_class=None, explainer_kwargs=None, n_rounds=5,
                                      is_selected=None, verbose=False, label_target_class=1,
                                      n_background_data=None, random_state=None):
    """Compute unbiased exact-``p`` SHAP estimates for fuzzy labels by interpolating between 0/1 fits.

    Each fuzzy sample with soft label ``p`` is weighted by exactly ``p``: the model is fit twice
    (fuzzy sample at 0 -> ``S0``, at 1 -> ``S1``) and the per-feature attributions are blended as
    ``p * S1 + (1 - p) * S0``. Each fuzzy protein is explained independently against the fixed
    balanced 0/1 core, with the other fuzzy proteins excluded from that run's training data. With
    ``n_rounds=1`` this is exactly two fits per fuzzy sample; ``n_rounds > 1`` averages per-round
    re-seeded fits (reproducible for a fixed ``random_state``).
    """
    labels = list(labels)
    # Get class index (fuzzy float labels are excluded; classes come from the 0/1 core)
    class_index = _class_index_from_labels(labels, label_target_class)
    n_samples, n_features = X.shape
    n_selection_rounds = len(is_selected)
    n_cells = n_rounds * n_selection_rounds
    # Partition into the fixed 0/1 core and the fuzzy samples explained one at a time
    fuzzy_idx = [i for i, label in enumerate(labels) if label not in (0, 1)]
    core_idx = [i for i, label in enumerate(labels) if label in (0, 1)]
    core_labels = [labels[i] for i in core_idx]
    # A single fuzzy protein shares the full sample set, so the two blended fits already cover
    # every row (no separate baseline needed) -> exactly two fits per round and selection.
    single_fuzzy = len(fuzzy_idx) == 1
    acc_shap_values = np.zeros(shape=(n_samples, n_features))
    list_expected_value = []
    if verbose:
        ut.print_start_progress(start_message=f"ShapModel starts interpolation estimation of SHAP values over {n_rounds} rounds.")
    for i in range(n_rounds):
        _list_model_kwargs = _seed_model_kwargs(list_model_kwargs, random_state=random_state, round_idx=i)
        for j, selected_features in enumerate(is_selected):
            if verbose:
                pct_progress = j / len(is_selected)
                add_new_line = explainer_class in LIST_VERBOSE_shap_modelS
                ut.print_progress(i=i+pct_progress, n_total=n_rounds, add_new_line=add_new_line)
            X_selected = X[:, selected_features]
            args = dict(list_model_classes=list_model_classes, list_model_kwargs=_list_model_kwargs,
                        explainer_class=explainer_class, explainer_kwargs=explainer_kwargs,
                        class_index=class_index, n_background_data=n_background_data)
            if single_fuzzy:
                f = fuzzy_idx[0]
                p = labels[f]
                labels_0 = [0 if k == f else labels[k] for k in range(n_samples)]
                labels_1 = [1 if k == f else labels[k] for k in range(n_samples)]
                shap_0, exp_0 = _aggregate_shap_values(X_selected, labels=labels_0, **args)
                shap_1, exp_1 = _aggregate_shap_values(X_selected, labels=labels_1, **args)
                cell = p * shap_1 + (1 - p) * shap_0
                list_expected_value.append(p * exp_1 + (1 - p) * exp_0)
            else:
                cell = np.zeros(shape=(n_samples, X_selected.shape[1]))
                # Non-fuzzy core rows come from a single baseline fit on the core
                shap_core, exp_core = _aggregate_shap_values(X_selected[core_idx], labels=core_labels, **args)
                cell[core_idx] = shap_core
                list_expected_value.append(exp_core)
                # Each fuzzy protein is explained against core + itself (others excluded)
                for f in fuzzy_idx:
                    p = labels[f]
                    sub_idx = core_idx + [f]
                    X_sub = X_selected[sub_idx]
                    shap_0, exp_0 = _aggregate_shap_values(X_sub, labels=core_labels + [0], **args)
                    shap_1, exp_1 = _aggregate_shap_values(X_sub, labels=core_labels + [1], **args)
                    cell[f] = p * shap_1[-1] + (1 - p) * shap_0[-1]
                    list_expected_value.append(p * exp_1 + (1 - p) * exp_0)
            full_cell = np.zeros(shape=(n_samples, n_features))
            full_cell[:, selected_features] = cell
            acc_shap_values += full_cell
    if verbose:
        ut.print_end_progress(end_message=f"ShapModel finished interpolation estimation and saved results.")
    shap_values = acc_shap_values / n_cells
    exp_val = np.mean(list_expected_value)
    return shap_values, exp_val

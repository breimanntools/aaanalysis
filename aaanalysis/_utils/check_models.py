"""This is a script for scikit-learn model-specific check functions"""
import inspect
from inspect import isclass
import numpy as np

from ._utils import add_str
# Helper functions


# Main functions
def check_mode_class(model_class=None, str_add=None):
    """Check if the provided object is a class and callable, typically used for validating model classes."""
    # Check if model_class is actually a class and not an instance
    if not isclass(model_class):
        str_error = add_str(str_error=f"'model_class' ('{model_class}') is not a model class. "
                                      f"Please provide a valid model class.",
                            str_add=str_add)
        raise ValueError(str_error)
    # Check if model is callable
    if not callable(getattr(model_class, "__call__", None)):
        str_error = add_str(str_error=f"'model_class' ('{model_class}') is not a callable model.",
                            str_add=str_add)
        raise ValueError(str_error)


def check_model_kwargs(model_class=None, model_kwargs=None, name_model_class="model_class",
                       param_to_check=None, method_to_check=None, attribute_to_check=None,
                       random_state=None, str_add=None):
    """
    Check if the provided model class contains specific parameters and methods. Filters 'model_kwargs' to include only
    valid parameters for the model class.

    Parameters:
        model_class: The class of the model to check.
        model_kwargs: A dictionary of keyword arguments for the model.
        name_model_class: Name of model class for model class kwargs
        param_to_check: A specific parameter to check in the model class.
        method_to_check: A specific method to check in the model class.
        attribute_to_check: A specific attribute to check in model class
        random_state: random state
        str_add: additional error string

    Returns:
        model_kwargs: A filtered dictionary of model_kwargs containing only valid parameters for the model class.
    """
    model_kwargs = model_kwargs or {}
    if model_class is None:
        str_error = add_str(str_error=f"'{name_model_class}' must be provided.", str_add=str_add)
        raise ValueError(str_error)
    valid_args = list(inspect.signature(model_class).parameters.keys())
    # Check if 'param_to_check' is a parameter of the model
    if param_to_check is not None and param_to_check not in valid_args:
        str_error = add_str(str_error=f"'{param_to_check}' should be an argument in the given '{name_model_class}' ({model_class}).",
                            str_add=str_add)
        raise ValueError(str_error)
    # Check if 'method_to_check' is a method of the model
    if method_to_check is not None and not hasattr(model_class, method_to_check):
        str_error = add_str(str_error=f"'{method_to_check}' should be a method in the given '{name_model_class}' ({model_class}).",
                            str_add=str_add)
        raise ValueError(str_error)
    # Check if 'attribute_to_check' is an attribute of the model
    if attribute_to_check is not None and not hasattr(model_class, attribute_to_check):
        str_error = add_str(str_error=f"'{attribute_to_check}' should be an attribute in the given '{name_model_class}' ({model_class}).",
                            str_add=str_add)
        raise ValueError(str_error)
    # Check if model_kwargs contain invalid parameters for the model
    invalid_kwargs = [x for x in model_kwargs if x not in valid_args]
    if len(invalid_kwargs):
        str_error = add_str(str_error=f"'model_kwargs' (for '{model_class}') contains invalid arguments: {invalid_kwargs}",
                            str_add=str_add)
        raise ValueError(str_error)
    if "random_state" not in model_kwargs and "random_state" in valid_args:
        model_kwargs.update(dict(random_state=random_state))
    return model_kwargs


def check_match_list_model_classes_kwargs(list_model_classes=None, list_model_kwargs=None):
    """Check length match of list_model_classes and list_model_kwargs"""
    n_models = len(list_model_classes)
    n_args = len(list_model_kwargs)
    if n_models != n_args:
        raise ValueError(f"Length of 'list_model_kwargs' (n={n_args}) should match to 'list_model_classes' (n{n_models}")

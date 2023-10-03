"""This is a script for scikit-learn model specific check functions"""
import inspect
from inspect import isclass

# Helper functions

# Main functions
def check_mode_class(model_class=None):
    """"""
    # Check if model_class is actually a class and not an instance
    if not isclass(model_class):
        raise ValueError(f"'{model_class}' is not a model class. Please provide a valid model class.")
    # Check if model is callable
    if not callable(getattr(model_class, "__call__", None)):
        raise ValueError(f"'{model_class}' is not a callable model.")
    return model_class

def check_model_kwargs(model_class=None, model_kwargs=None, param_to_check="n_clusters"):
    """
    Check if the provided model has 'n_clusters' as a parameter.
    Filter the model_kwargs to only include keys that are valid parameters for the model.
    """
    model_kwargs = model_kwargs or {}
    if model_class is None:
        raise ValueError("'model_class' must be provided.")
    valid_args = list(inspect.signature(model_class).parameters.keys())
    # Check if 'param_to_check' is a parameter of the model
    if param_to_check not in valid_args:
        raise ValueError(f"'n_clusters' should be an argument in the given 'model' ({model_class}).")
    # Filter model_kwargs to only include valid parameters for the model
    invalid_kwargs = [x for x in model_kwargs if x not in valid_args]
    if len(invalid_kwargs):
        raise ValueError(f"'model_kwargs' contains non valid arguments: {invalid_kwargs}")
    return model_kwargs

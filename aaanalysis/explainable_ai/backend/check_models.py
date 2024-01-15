"""
This is a script for common check function for the explainable AI models.
"""


def check_match_list_model_classes_kwargs(list_model_classes=None, list_model_kwargs=None):
    """Check length match of list_model_classes and list_model_kwargs"""
    n_models = len(list_model_classes)
    n_args = len(list_model_kwargs)
    if n_models != n_args:
        raise ValueError(f"Length of 'list_model_kwargs' (n={n_args}) should match to 'list_model_classes' (n{n_models}")

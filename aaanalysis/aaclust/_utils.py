#! /usr/bin/python3
"""
Config with folder structure
"""
import inspect

# Default column names for scales and categories
METRIC_CORRELATION = "correlation"
LIST_METRICS = [METRIC_CORRELATION, "manhattan",  "euclidean", "cosine"]


# Check functions
def check_model(model=None, model_kwargs=None, except_None=True):
    """"""
    if except_None:
        return model_kwargs
    list_model_args = list(inspect.signature(model).parameters.keys())
    if "n_clusters" not in list_model_args:
        error = f"'n_clusters' should be argument in given clustering 'model' ({model})."
        raise ValueError(error)
    model_kwargs = {x: model_kwargs[x] for x in model_kwargs if x in list_model_args}
    return model_kwargs


def check_min_th(min_th=None, min_val=0, max_val=0.9):
    """Check if value of given name variable is non-negative integer"""
    check_types = [float, int]
    str_check = "non-negative float or int"
    error = f"'min_th' ({min_th}) should be {str_check} n, where {min_val}<=n<={max_val}"
    if type(min_th) not in check_types:
        raise ValueError(error)
    if not min_val <= min_th <= max_val:
        raise ValueError(error)


def check_merge_metric(merge_metric=None):
    """"""
    if merge_metric is not None and merge_metric not in LIST_METRICS:
        error = f"'merge_metric' should be None or one of following: {LIST_METRICS}"
        raise ValueError(error)
    return merge_metric

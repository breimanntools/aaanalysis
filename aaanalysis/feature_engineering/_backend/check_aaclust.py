"""
This is a script for the common AAclust checking functions
"""
import aaanalysis.utils as ut


# I Helper Functions


# II Main Functions
def check_metric(metric=None):
    """"""
    if metric not in ut.LIST_METRICS:
        error = f"'metric' should be None or one of following: {ut.LIST_METRICS}"
        raise ValueError(error)



"""This is a script for utility functions for utility functions."""
import numpy as np

# Constants
VALID_INT_TYPES = (
    int,
    np.int_, np.intc, np.intp, np.integer,  # np.integer covers all standard integer types
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.longlong  # Equivalent to np.int64 on most platforms
)

VALID_FLOAT_TYPES = (
    float,
    np.float_, np.float16, np.float32, np.float64, np.longdouble  # np.longdouble is the highest precision float available
)

VALID_INT_FLOAT_TYPES = VALID_INT_TYPES + VALID_FLOAT_TYPES


# Helper functions
def add_str(str_error=None, str_add=None):
    """Add additional error message 'str_add' to default error message ('add_str')"""
    if str_add:
        str_error += "\n " + str_add
    return str_error

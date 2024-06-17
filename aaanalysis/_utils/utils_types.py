from typing import Union, NewType, Sequence
import numpy as np
import pandas as pd

# Define the basic numeric type
NumericType = Union[int, float]

# Define the 1D and 2D types using Union
ArrayLike1DUnion = Union[Sequence[NumericType], np.ndarray, pd.Series]
ArrayLike2DUnion = Union[Sequence[Sequence[NumericType]], np.ndarray, pd.DataFrame]

# A 1D array-like object. Can be a sequence (e.g., list or tuple) of ints/floats, numpy ndarray, or pandas Series.
ArrayLike1D = NewType("ArrayLike1D", ArrayLike1DUnion)

# A 2D array-like object. Can be a sequence of sequence of ints/floats, numpy ndarray, or pandas DataFrame.
ArrayLike2D = NewType("ArrayLike2D", ArrayLike2DUnion)

# Numeric type lists
VALID_INT_TYPES = (
    int,
    np.intc, np.intp, np.integer,  # np.integer covers all standard integer types
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.longlong  # Equivalent to np.int64 on most platforms
)

VALID_FLOAT_TYPES = (
    float,
    np.float16, np.float32, np.float64, np.longdouble  # np.longdouble is the highest precision float available
)

VALID_INT_FLOAT_TYPES = VALID_INT_TYPES + VALID_FLOAT_TYPES

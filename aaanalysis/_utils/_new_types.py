from typing import Union, NewType, Sequence
import numpy as np
import pandas as pd

# Define the basic numeric type
NumericType = Union[int, float]

# Define the 1D and 2D types using Union
ArrayLike1DUnion = Union[Sequence[NumericType], np.ndarray, pd.Series]
ArrayLike2DUnion = Union[Sequence[Sequence[NumericType]], np.ndarray, pd.DataFrame]

# Now, we'll create distinct named types using NewType.
# This won't change runtime behavior but will be recognized by static type checkers and can be documented.

# A 1D array-like object. Can be a sequence (e.g., list or tuple) of ints/floats, numpy ndarray, or pandas Series.
ArrayLike1D = NewType("ArrayLike1D", ArrayLike1DUnion)

# A 2D array-like object. Can be a sequence of sequence of ints/floats, numpy ndarray, or pandas DataFrame.
ArrayLike2D = NewType("ArrayLike2D", ArrayLike2DUnion)

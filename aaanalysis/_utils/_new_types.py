"""Define new data types"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Any, Protocol, TypeVar, Generic, Optional
from sklearn.base import BaseEstimator

# Array-like Types
"""
ArrayLike = Union[
    List[Union[int, float, Any]],
    Tuple[Union[int, float, Any], ...],
    np.ndarray,
    pd.DataFrame,
    pd.Series
]

ArrayLikeInt = Union[List[int], List[int], Tuple[int], np.ndarray]
ArrayLikeFloat = Union[List[Union[float, int]], Tuple[Union[float, int]], np.ndarray]
ArrayLikeAny = Union[List[Any], Tuple[Any], np.ndarray]
ArrayLikeBool = Union[List[bool], Tuple[bool], np.ndarray]
"""

# Array-like Types until depth 3
# General ArrayLike (only numbers)
# 1D
NumericType = Union[int, float]
ArrayLike1D = Union[List[NumericType], Tuple[NumericType], pd.Series]

# 2D
ArrayLike = Union[List[List[NumericType]], Tuple[Tuple[NumericType]], np.ndarray, pd.DataFrame]

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
NestedGeneral2 = Union[List[Union[int, float]], Tuple[Union[int, float]]]
NestedGeneral3 = Union[List[NestedGeneral2], Tuple[NestedGeneral2]]
NestedGeneral4 = Union[List[NestedGeneral3], Tuple[NestedGeneral3]]
ArrayLike = Union[NestedGeneral2, NestedGeneral3, NestedGeneral4, np.ndarray, pd.DataFrame, pd.Series]

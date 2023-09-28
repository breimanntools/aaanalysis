"""Define new data types"""
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Any


# Types
ArrayLike = Union[
    List[Union[int, float, Any]],
    Tuple[Union[int, float, Any], ...],
    np.ndarray,
    pd.DataFrame,
    pd.Series
]
ArrayLikeInt = Union[List[int], Tuple[int], np.ndarray]
ArrayLikeFloat = Union[List[Union[float, int]], Tuple[Union[float, int]], np.ndarray]
ArrayLikeAny = Union[List[Any], Tuple[Any], np.ndarray]
ArrayLikeBool = Union[List[bool], Tuple[bool], np.ndarray]
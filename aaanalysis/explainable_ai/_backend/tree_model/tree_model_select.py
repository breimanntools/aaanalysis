"""This is a script for the backend of the TreeModel.select_features() method."""
import numpy as np

import aaanalysis.utils as ut


# I Helper Functions
def select_top_k_(feat_importance=None, n=None):
    """Keep the ``n`` features with the highest Monte Carlo importance."""
    # Ties are broken by lowest index (stable) via argsort on negated importance.
    order = np.argsort(-np.asarray(feat_importance), kind="stable")
    is_selected = np.zeros(len(feat_importance), dtype=bool)
    is_selected[order[:n]] = True
    return is_selected


def select_by_threshold_(feat_importance=None, threshold=None):
    """Keep features whose Monte Carlo importance is at least ``threshold``."""
    is_selected = np.asarray(feat_importance) >= threshold
    return is_selected


def select_by_frequency_(is_selected_rounds=None, min_freq=None):
    """Keep features selected in at least ``min_freq`` fraction of RFE rounds."""
    frequency = np.mean(np.asarray(is_selected_rounds, dtype=float), axis=0)
    is_selected = frequency >= min_freq
    return is_selected


# II Main Functions
def get_feature_selection_mask(strategy=None, param=None, feat_importance=None,
                               is_selected_rounds=None):
    """Dispatch to the requested selection strategy and return a 1D boolean mask."""
    if strategy == ut.STRATEGY_TOP_K:
        is_selected = select_top_k_(feat_importance=feat_importance, n=param)
    elif strategy == ut.STRATEGY_THRESHOLD:
        is_selected = select_by_threshold_(feat_importance=feat_importance, threshold=param)
    elif strategy == ut.STRATEGY_FREQUENCY:
        is_selected = select_by_frequency_(is_selected_rounds=is_selected_rounds, min_freq=param)
    else:
        # Frontend validates ``strategy``; reaching here is an internal bug.
        raise RuntimeError(f"Unknown selection 'strategy' reached backend: {strategy}")
    # Derived invariant: a selection that keeps nothing is not actionable downstream.
    if not np.any(is_selected):
        raise ValueError(f"No features selected for 'strategy'='{strategy}' with 'param'={param}. "
                         f"Relax the selection (e.g. lower 'threshold'/'min_freq' or raise 'top_k').")
    return is_selected

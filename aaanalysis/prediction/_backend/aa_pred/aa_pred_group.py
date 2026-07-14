"""
This is a script for the backend of the AAPred score-to-group confidence-band assignment.
"""
import numpy as np


# I Helper Functions


# II Main Functions
def assign_band_index(scores, sorted_thresholds):
    """Map scores to 0-based confidence-band indices (low score -> high score).

    Single source of truth for the band boundary convention shared by
    ``AAPred.score_to_group`` and the ``AAPredPlot.predict_group(band=True)`` colouring: each
    threshold is an **inclusive lower bound**, so a score equal to a threshold falls in the band
    *above* it (right-open bands ``[t_{i-1}, t_i)``). For sorted thresholds ``t_0 < ... < t_{k-1}``
    a score ``s`` maps to ``sum(t_i <= s)`` — band ``0`` is ``s < t_0`` and band ``k`` is
    ``s >= t_{k-1}``. Accepts a scalar or an array and returns the matching shape. ``NaN`` scores
    map to ``k`` (searchsorted orders NaN last); callers that treat missing scores specially mask
    them separately.
    """
    return np.searchsorted(np.asarray(sorted_thresholds, dtype=float), scores, side="right")

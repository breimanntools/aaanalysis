"""
This is a script for the backend of CPP's numerical-mode redundancy-reduction
stage. Ported verbatim from ``_filters._redundancy_filter`` (greedy
descending-AUC selection with position-overlap + scale-correlation gating).

# DEV: ADR-0001 records this verbatim port. Potential future optimization
# (precompute the (n_pre_filter, n_pre_filter) overlap matrix and the
# (D, D) scale-correlation matrix as numpy upfront, replacing the per-pair
# dict lookups and set ops inside the greedy loop with O(1) numpy reads).
# Greedy selection is inherently sequential; the precompute-vectorize pass
# was estimated at ~10x and deferred to a follow-up PR.
"""
from .._filters._redundancy_filter import filtering, filtering_info_  # noqa: F401

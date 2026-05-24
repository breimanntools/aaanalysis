"""
Cython-accelerated kernels for CPP feature-matrix construction.

Built via ``setup_inner.py`` (in this directory) or the project's
``build_cython_inplace.sh`` helper. The compiled extension lives next to
``_inner.pyx``. Import path:

    from aaanalysis.feature_engineering._backend.cpp._filters_c._inner import (
        compute_segment_mean, compute_pattern_n_mean, compute_pattern_c_mean,
    )

When the extension is not built (no compiler / pure-Python install), the
import raises ``ImportError`` and ``CPP.run_c`` is gated behind a
``missing_feature_stub`` so non-pro users still get a friendly hint.
"""

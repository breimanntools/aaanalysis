"""
This is a script for the backend of CPP's numerical-mode pre-filtering threshold
stage. Ported verbatim from ``_filters._pre_filter`` — the threshold + top-K
logic is independent of the upstream value source, so the same module is reused.
"""
from .._filters._pre_filter import pre_filtering  # noqa: F401

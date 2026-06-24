"""
``aaanalysis.pipe`` (aap) — high-level convenience pipelines (golden pipelines).

A second, opt-in API on top of the AAanalysis primitives: stateless, thin wrappers that chain
the existing classes into one call so users get a result without hand-wiring every step. Import it
explicitly under the conventional alias::

    import aaanalysis.pipe as aap

The wrappers add no algorithm of their own — their defaults are byte-identical to the explicit
primitive path (e.g. :meth:`SequenceFeature.feature_matrix` → :class:`TreeModel`), they return plain
numpy/pandas objects, and they thread ``random_state`` / ``n_jobs`` through. Machine-readable tool
contracts (MCP, verb/tool schemas) are intentionally **not** here; this layer is human- and
sklearn-idiomatic convenience only.

Public objects
--------------
* :func:`obtain_samples` — turn a described sampling situation into a balanced training set.
* :func:`predict` — build the feature matrix from ``df_feat`` and fit + evaluate a :class:`TreeModel`.

See Also
--------
* :mod:`aaanalysis.seq_analysis` — :class:`AAWindowSampler`, the sampler :func:`obtain_samples` wraps.
* :mod:`aaanalysis.feature_engineering` — the primitives these pipelines wrap.
* :mod:`aaanalysis.explainable_ai` — :class:`TreeModel`, the predictor used by :func:`predict`.
"""
from ._obtain_samples import obtain_samples
from ._pipelines import predict

__all__ = ["obtain_samples", "predict"]

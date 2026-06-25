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
* :func:`find_features` — staged CPP AutoML search that sweeps the feature space, selects the
  best configuration by cross-validated model performance, and draws the feature map.
* :func:`predict_samples` — train and compare predictors across one or more feature sets and
  scikit-learn models, returning the fitted predictors keyed by ``(feature_set, model)`` plus a
  cross-validated comparison table as the ``(predictors, None, df_eval)`` triple.
* :func:`plot_eval` — ``viridis`` evaluation-grid plot of a :func:`find_features` sweep that
  adapts to the number of swept axes (line / heatmap / faceted small-multiples).
* :func:`explain_features` (*pro*) — compute per-sample SHAP impact for a feature set and draw the
  SHAP-coloured feature map, returning the ``(df_feat_shap, ax, None)`` triple.

See Also
--------
* :mod:`aaanalysis.seq_analysis` — :class:`AAWindowSampler`, the sampler :func:`obtain_samples` wraps.
* :mod:`aaanalysis.feature_engineering` — the primitives these pipelines wrap.
* :mod:`aaanalysis.explainable_ai` — :class:`TreeModel`, used by :func:`find_features` to score
  feature importances (and a SHAP-ready predictor option for :func:`predict_samples`).
* :mod:`aaanalysis.explainable_ai_pro` — :class:`ShapModel`, the explainer :func:`explain_features` wraps.
"""
from ._obtain_samples import obtain_samples
from ._find_features import find_features
from ._pipelines import predict_samples
from ._eval_plot import plot_eval

__all__ = ["obtain_samples", "find_features", "predict_samples", "plot_eval"]

# explain_features is pro-gated (needs SHAP via ShapModel): degrade to a friendly install-hint stub
# when aaanalysis[pro] is absent, mirroring the top-level pro-import pattern (pro-core-boundary).
try:
    from ._explain_features import explain_features
except ImportError as e:
    from aaanalysis import missing_feature_stub
    explain_features = missing_feature_stub("explain_features", e, mode="pro")
__all__.append("explain_features")

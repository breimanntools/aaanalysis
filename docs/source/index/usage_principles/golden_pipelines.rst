.. _golden_pipelines:

Golden Pipelines
================

The golden pipelines in :mod:`aaanalysis.pipe` are a second, opt-in way to use AAanalysis: stateless
one-call wrappers that chain the existing classes into a complete step, so a first result does not
require wiring every primitive by hand. They add no algorithm of their own. Each one is a thin
facade over a path you could write explicitly, and that promise is kept honest by tests: where a
pipeline has an explicit equivalent, its defaults are pinned to reproduce that path byte for byte.
Import the module under its conventional alias:

.. code-block:: python

    import aaanalysis as aa
    import aaanalysis.pipe as ap

The pipeline spine
------------------

A typical workflow moves through up to four pipelines, each consuming what the previous one returns:

- :func:`~aaanalysis.pipe.obtain_samples`: turn a described sampling situation (known positive
  positions plus a negative sampling strategy) into a balanced, labeled training set.
- :func:`~aaanalysis.pipe.find_features`: search the CPP configuration space and return the best
  feature set ``df_feat`` together with the sweep table ``df_eval``.
- :func:`~aaanalysis.pipe.predict_samples`: cross-validate and refit one or more scikit-learn
  models on one or more feature sets and return the fitted predictors with a comparison table.
- :func:`~aaanalysis.pipe.explain_features` (requires the ``[pro]`` extra): compute per-sample SHAP
  impact for a feature set and draw the SHAP-coloured feature map.

:func:`~aaanalysis.pipe.plot_eval` complements the spine by turning a
:func:`~aaanalysis.pipe.find_features` sweep table into publication-ready evaluation figures.

The four spine pipelines above return the same three-slot shape, ``(result, plot, df_eval)``: the
primary result (a DataFrame or a dictionary of predictors), the figure handle or ``None`` when
``plot=False``, and a tidy evaluation table. Keeping one return shape means a script or a coding
agent can chain them without special-casing each one.
:func:`~aaanalysis.pipe.plot_eval` is the exception: it is a pure plotting helper that consumes a
sweep table rather than producing one, and it returns a plain ``list`` of matplotlib ``Figure``
objects, one per evaluation panel.

From sequences to a cross-validated score
-----------------------------------------

The whole path from a labeled dataset to a cross-validated balanced accuracy fits in a handful of
statements:

.. code-block:: python

    import aaanalysis as aa
    import aaanalysis.pipe as ap

    df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
    labels = df_seq["label"].to_list()
    df_feat, _, _ = ap.find_features(labels=labels, df_seq=df_seq, search="fast", plot=False, random_state=0)
    predictors, _, df_eval = ap.predict_samples(list_df_feat=df_feat, df_seq=df_seq, labels=labels,
                                                plot=False, random_state=0)
    score = df_eval["balanced_accuracy_mean"].max()

This snippet is part of the test suite: it is executed on every run, and its length is checked
against a budget of at most ten statements, so the ergonomics of the pipeline layer cannot erode
unnoticed.

Parity with the explicit path
-----------------------------

A convenience layer is only trustworthy if it does not quietly compute something different from
the primitives it wraps. Two parity anchors pin this:

- :func:`~aaanalysis.pipe.find_features` with ``search="fast"`` runs no search. Its ``df_feat`` is
  byte-identical to the explicit chain of :meth:`~aaanalysis.SequenceFeature.get_df_parts`,
  :meth:`~aaanalysis.CPP.run`, :meth:`~aaanalysis.CPP.simplify`, and the
  :class:`~aaanalysis.TreeModel` importance ranking.
- :func:`~aaanalysis.pipe.predict_samples` at its defaults (with a fixed ``random_state``) is
  byte-identical to building ``X`` with :meth:`~aaanalysis.SequenceFeature.feature_matrix`, then
  scoring each default estimator with scikit-learn's ``cross_validate`` (five folds) and refitting
  it on all samples. Both the comparison table and the fitted predictors match exactly.

The explicit equivalent of ``predict_samples`` is:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    from sklearn.svm import SVC

    sf = aa.SequenceFeature()
    X = sf.feature_matrix(features=df_feat["feature"], df_parts=sf.get_df_parts(df_seq=df_seq))
    models = {"RandomForest": RandomForestClassifier(random_state=0),
              "ExtraTrees": ExtraTreesClassifier(random_state=0),
              "SVM": SVC(class_weight="balanced", probability=True, random_state=0),
              "LogReg": LogisticRegression(max_iter=1000, random_state=0)}
    scoring = ["balanced_accuracy", "accuracy", "f1", "precision", "recall", "roc_auc"]
    for name, model in models.items():
        cv_results = cross_validate(model, X, y=labels, cv=5, scoring=scoring)
        predictor = model.fit(X, labels)

The staged searches of :func:`~aaanalysis.pipe.find_features` (``search="balanced"`` and
``search="exhaustive"``) select among many configurations, so they have no single explicit chain to
compare against; the parity anchors cover ``search="fast"`` only. With a fixed ``random_state`` the
staged searches are still reproducible.

Documented by example
---------------------

Each pipeline has an example notebook that demonstrates every public parameter by name. This is
enforced by the same parameter-coverage check that guards the example notebooks of the core API,
with no permitted gaps for :mod:`aaanalysis.pipe`. The failure behaviour of the pipelines on invalid
input is documented separately in :ref:`Failure Contracts of the Golden Pipelines <error_contracts>`.

.. note::

    The :mod:`aaanalysis.pipe` layer is experimental: signatures, defaults, and return objects may
    change between minor releases without the usual deprecation cycle. Pin a version if you depend
    on the current behaviour. The parity anchors above guard what the pipelines compute, not their
    call signatures.

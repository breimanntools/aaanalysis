.. _error_contracts:

Failure Contracts of the Golden Pipelines
=========================================

The golden pipelines (:func:`~aaanalysis.pipe.find_features`,
:func:`~aaanalysis.pipe.predict_samples`, :func:`~aaanalysis.pipe.explain_features`) and the
core :class:`~aaanalysis.CPP` to :class:`~aaanalysis.AAPred` path are called from scripts and
coding agents as well as notebooks, so their behaviour on the *unhappy* path is a documented
contract, not an accident. Every invalid call raises a **bare** :class:`ValueError` or
:class:`RuntimeError` (or, for an unavailable optional dependency, an :class:`ImportError` with
an install hint), and the message names the offending input in the package's own voice. A caller
can therefore tell "I called it wrong" from "the library broke" from the message alone, without
reading a traceback. AAanalysis deliberately keeps these as plain built-in exceptions: there is
no custom exception hierarchy and no error-code taxonomy, since mapping a raised error to a
structured, recoverable result is a downstream adapter's job.

The worked pairs below show one correct and one incorrect invocation per pipeline.

find_features
-------------

Correct, with binary labels aligned to the ``df_seq`` rows:

.. code-block:: python

    import aaanalysis as aa
    from aaanalysis import pipe as ap

    df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
    labels = df_seq["label"].to_list()
    df_feat, ax, df_eval = ap.find_features(labels=labels, df_seq=df_seq, plot=False)

Incorrect, a single-class label vector cannot define a test-versus-reference contrast:

.. code-block:: python

    ap.find_features(labels=[1] * len(df_seq), df_seq=df_seq, plot=False)
    # ValueError: 'labels' should contain more than one different value ({1}).

predict_samples
---------------

Correct, one or more CPP feature sets with labels aligned to ``df_seq``:

.. code-block:: python

    models, ax, df_eval = ap.predict_samples(list_df_feat=[df_feat], df_seq=df_seq,
                                             labels=labels, plot=False)

Incorrect, an empty feature set has nothing to score:

.. code-block:: python

    ap.predict_samples(list_df_feat=[], df_seq=df_seq, labels=labels, plot=False)
    # ValueError: 'list_df_feat' should contain at least one feature DataFrame.

explain_features (pro)
----------------------

:func:`~aaanalysis.pipe.explain_features` needs the ``[pro]`` extra (SHAP). Correct, with
``[pro]`` installed:

.. code-block:: python

    df_impact, ax, _ = ap.explain_features(df_feat=df_feat, df_seq=df_seq, labels=labels, plot=False)

Incorrect, without the ``[pro]`` extra the function degrades to an install hint:

.. code-block:: python

    ap.explain_features(df_feat=df_feat, df_seq=df_seq, labels=labels)
    # ImportError: 'explain_features' needs additional dependencies. Install via:
    #     pip install 'aaanalysis[pro]'

CPP to AAPred
-------------

``X`` below is the CPP feature matrix (for example from
:meth:`~aaanalysis.SequenceFeature.feature_matrix`). Correct, a fitted
:class:`~aaanalysis.AAPred` scores a matrix of the same width used in fit:

.. code-block:: python

    aap = aa.AAPred().fit(X=X, labels=labels)
    df_pred = aap.predict_proba(X=X)

Incorrect, scoring a matrix whose width does not match the fitted model, or scoring before
fitting:

.. code-block:: python

    aap.predict_proba(X=X[:, :3])
    # ValueError: 'X' n_features (3) should match the fitted model's n_features (20); ...

    aa.AAPred().predict_proba(X=X)
    # ValueError: 'AAPred' is not fitted; call 'AAPred.fit' first.

Contract summary
----------------

- Validation errors are :class:`ValueError`; runtime-state errors (such as calling ``eval``
  before ``run``) are :class:`RuntimeError`.
- An unavailable optional ``[pro]`` dependency raises :class:`ImportError` with a
  ``pip install 'aaanalysis[pro]'`` hint.
- Messages quote the offending input name, so they are self-explaining.
- The contract is regression-guarded by an integration failure-contract test suite and, from an
  installed distribution, by the packaging smoke check.

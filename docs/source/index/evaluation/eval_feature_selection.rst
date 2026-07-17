.. _eval_feature_selection:

The Four Evaluation Regimes
===========================
A CPP or model score can mean four very different things, and a small-dataset workflow
easily conflates them. Because AAanalysis targets biological problems with only a few
dozen to a few hundred labeled proteins, the difference matters: a number that ranks
features well is not the same as a number that estimates how the model will generalize.
This chapter separates the four regimes so that, for any score you report, you can say
which one produced it and never mistake an exploratory ranking signal for a
generalization estimate. The single knob that moves feature selection in and out of the
cross-validation is :func:`~aaanalysis.pipe.find_features`'s ``selection_scope``, and the
regimes below are keyed to it.


1. Exploratory feature discovery
--------------------------------
Run :class:`~aaanalysis.CPP` or :func:`~aaanalysis.pipe.find_features` with
``selection_scope="global"`` (the default) to surface and rank candidate physicochemical
determinants. Feature selection runs once on the **full** labeled set, and the model is
then cross-validated on that fixed feature matrix. Every score in the resulting
``df_eval`` sweep is therefore **post-selection** and in-sample optimistic: selection has
already seen every fold, so the number is a valid *relative* ranking signal for comparing
configurations, but a misleading *absolute* estimate of generalization. This is the home
of determinant discovery, and it is the correct regime when the goal is to understand
*which* features distinguish the classes.

Use the score to rank, not to claim accuracy. Reporting a ``"global"`` sweep score as a
held-out or generalization number is the most common misread this chapter guards against.


2. Honest (nested-CV) evaluation
--------------------------------
Pass ``selection_scope="fold"`` for a leakage-free, held-out estimate of generalization.
Within every fold of every configuration score, CPP re-selects features on the **train
split only**, the model is fit on those fold-specific train features, and it is scored on
the held-out test fold. Because the test fold never informs selection, the ``df_eval``
scores are honest generalization estimates and are typically lower than the ``"global"``
numbers for the same data. Nesting applies to the configuration-selection scores; the
winning configuration's second-step refinement still runs on all data, so no refinement
capability is lost. This regime re-runs CPP per fold and is much more expensive, so pair
it with ``search="fast"`` or ``"balanced"``.

A hand-rolled nested cross-validation gives the same guarantee; ``selection_scope="fold"``
simply builds the nesting into the pipeline.


3. Final model fitting
----------------------
Once a configuration is chosen, refit it on **all** the data to obtain the deployable
model. This is exactly what :func:`~aaanalysis.pipe.find_features` returns: its ``df_feat``
is always the winning configuration refit on the full set (outer-CV semantics), and
:meth:`~aaanalysis.TreeModel.fit` on the complete data produces the model you ship. The
training score of that refit model is an artifact of fitting, never an evaluation number.
Fitting on all available data maximizes the quality of the final model; its resubstitution
score says nothing about generalization and should not be reported as performance.


4. External test-set evaluation
-------------------------------
Score the final model on data held out from **all** selection and cross-validation. Such
an external test set gives the strongest generalization claim, because no part of it
touched feature selection, configuration choice, or model fitting. It is also the regime
that small biological datasets can rarely afford, since setting aside a representative,
untouched test set costs labeled examples that are already scarce. When one is available,
it is the number to trust; when it is not, the honest nested-CV estimate of regime 2 is the
best available proxy, and the exploratory scores of regime 1 must not be dressed up as its
substitute.


Choosing the right regime
-------------------------
The table below summarizes which regime answers which question and what its score does
and does not license:

.. list-table::
   :header-rows: 1
   :widths: 26 24 26 24

   * - Regime
     - How to obtain it
     - The score is
     - The score is not
   * - Exploratory feature discovery
     - ``selection_scope="global"`` (default)
     - a relative ranking of configurations and features
     - a generalization estimate
   * - Honest (nested-CV) evaluation
     - ``selection_scope="fold"`` (or a hand-rolled nested CV)
     - a leakage-free held-out generalization estimate
     - a substitute for an external test set
   * - Final model fitting
     - refit on all data (``find_features`` ``df_feat``; :meth:`~aaanalysis.TreeModel.fit`)
     - the deployable model
     - an evaluation; its training score is an artifact of fitting
   * - External test-set evaluation
     - the final model scored on fully held-out data
     - the strongest generalization claim
     - an option small biological datasets can usually afford

For the canonical short definitions of ``selection_scope`` and the four regimes, see the
project glossary. For the mechanism itself, see :func:`~aaanalysis.pipe.find_features` and
its ``selection_scope`` parameter.

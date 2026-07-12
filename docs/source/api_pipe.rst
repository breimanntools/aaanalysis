.. currentmodule:: aaanalysis.pipe

.. _api_pipe:

===============
API (Pipelines)
===============

AAanalysis exposes the same analysis through two interfaces:

.. code-block:: python

    import aaanalysis as aa            # explicit interface (the building blocks)
    import aaanalysis.pipe as ap      # implicit interface (the golden pipelines)

The :ref:`building blocks <api>` (``aa``) are the explicit objects and functions you
compose for full control over every step. The **golden pipelines** (``ap``) chain the
standard ``load`` to :class:`~aaanalysis.CPP` to ``model`` to ``explain`` to ``plot`` workflow into a
single call, much as ``pyplot`` sits over Matplotlib's ``Axes`` and ``Figure``. They are
stateless wrappers whose defaults match the explicit path, and they live in their own
module under a separate alias (``import aaanalysis.pipe as ap``). Reach for ``aa`` when
you want control over each step, and ``ap`` when you want a sensible default workflow in
one call.

Each pipeline below documents its inputs, outputs, and the building blocks it composes:

.. autosummary::
    :toctree: generated/

    obtain_samples
    find_features
    predict_samples
    plot_eval
    explain_features

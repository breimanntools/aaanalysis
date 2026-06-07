..
   Developer Notes:
   The paths to protocols are relative to ensure compatibility with the Sphinx referencing system
   used throughout the documentation. Protocol notebooks live in ``tutorials/`` (flat) and are
   auto-converted to ``generated/`` by ``create_notebooks_docs.py`` (called from ``conf.py``).
..


.. _protocols:

Protocols
=========

Task-oriented, **pipeline-ordered** recipes that answer a single biological question
end-to-end — *when to use it, what goes in, the minimal code, what comes out, how to
interpret it, common mistakes,* and *what to do next*. Unlike the tutorials (which teach
one function at a time), protocols teach **workflows**: each one is an executable notebook
you can run and adapt. This is a living catalog that grows along the AAanalysis pipeline;
see the `Protocols epic <https://github.com/breimanntools/aaanalysis/issues/35>`_.

The catalog opens with the two canonical protocols (the CPP signature and the prediction
levels), then follows the data-flow pipeline from constructing sets through to validating
your result.

.. toctree::
   :maxdepth: 1

   generated/protocol1_cpp_signature
   generated/protocol2_prediction_tasks
   generated/protocol3_sampling
   generated/protocol4_engineer_features
   generated/protocol5_compositional_positional
   generated/protocol6_feature_selection
   generated/protocol7_classifier
   generated/protocol8_interpretability
   generated/protocol9_validation

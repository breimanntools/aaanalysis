.. _usage_principles_aaclust:

AAclust: Selecting Redundancy-Reduced Scale Sets
================================================

AAclust is a clustering wrapper framework. introduced in [Breimann24a]_.

Taking any a set of amino acid scales as input, it estimates and optimizes the number of clusters to obtain a
redundancy-reduced scale set:

.. image :: /_artwork/schemes/scheme_AAclust1.png

.. image :: /_artwork/schemes/scheme_AAclust1tp.png


AAclust provides two selection modi: k-optimized and k-based selection:

.. image :: /_artwork/schemes/scheme_AAclust2.png

Three different clustering quality measures can be used to evaluate the the resulting clustering results:

.. image :: /_artwork/schemes/scheme_AAclust3.png
.. _usage_principles_aaclust:

AAclust: Selecting Redundancy-Reduced Scale Sets
================================================

**AAclust** (Amino Acid scale Clustering) is a clustering wrapper framework for selecting redundancy-reduced sets
of amino acid scales, introduced in [Breimann24a]_. Using Pearson correlation, AAclust optimizes the number of clusters
(*k*) and selects one representative scale per cluster. This process is illustrated in the figure below:


.. figure:: /_artwork/schemes/scheme_AAclust1.png

   Scheme of AAclust algorithm with clustering of amino acid scales and scale selection from [Breimann24a]_.

To accommodate varying user needs and preferences, AAclust offers two distinct approaches for setting the number of clusters

    - *k*-optimized: Here, *k* is automatically optimized by AAclust, ensuring an automated and refined selection process
    - *k*-based: In this mode, *k* is directly specified by the user.

The following scheme compares both methods and their respective options:


.. figure:: /_artwork/schemes/scheme_AAclust2.png

   Two operational modes (*k*-optimized and *k*-based) of AAclust for determining the number of clusters.
   from [Breimann24a]_.

See our :doc:`AAclust Tutorial <generated/tutorial3a_aaclust>` for hands-on examples.

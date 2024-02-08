.. _eval_aaclust:

Evaluating Clustering
=====================
Clustering algorithms aim to organize data points into meaningful groups based on their similarities. They play a crucial
role in identifying inherent patterns within unlabeled datasets. The key challenge is defining a relevant measure of
"similarity" that aligns with the specific context of the application.

Without a "ground truth" of predefined labels, evaluating clustering results becomes inherently subjective.
The difficulty lies in ensuring that the clusters formed are reflecting the underlying data distribution
and not just algorithmic artifacts or random noise.


Challenges in Clustering Evaluation
-----------------------------------
Clustering evaluation faces several challenges:

- **Defining Ground Truth**: Without labeled data, it's challenging to assess the quality of the clustering objectively.
- **Choice of Similarity Measure**: Different measures can yield different clustering results, affecting evaluation.
- **Determining the Number of Clusters**: There is often no clear criterion for choosing the number of clusters.
- **Algorithm Sensitivity**: Clustering algorithms can be sensitive to initial conditions or outliers in the data.

To address these challenges, several internal and external evaluation measures have been developed.


Clustering Quality Measures
---------------------------
AAnalysis focuses on three common clustering quality measures to evaluate clustering without ground truth:

- **Silhouette Coefficient (SC)**: Measures how similar an object is to its own cluster compared to other clusters.
  A high SC score indicates well-clustered data points, whereas a low score suggests overlapping clusters.
- **Calinski-Harabasz Score (CH)**: Compares the variance of data points within a cluster to the variance between
  clusters. A higher score generally indicates better-defined clusters.
- **Bayesian Information Criterion (BIC)**: A model-based criterion that evaluates the goodness-of-fit of
  the clustering model against its complexity.

These measures are essential in assessing the distinct aspects of clustering quality, such as within-cluster cohesion
and between-cluster separation, depicted below:

.. image:: /_artwork/schemes/scheme_AAclust3.png

   [Caption: Illustration of clustering quality measures: Silhouette Coefficient,
    Calinski-Harabasz Score, and Bayesian Information Criterion.]

To learn more about clustering and its evaluation in detail, you may refer to the
`scikit-learn clustering <https://scikit-learn.org/stable/modules/clustering.html>`_ documentation.

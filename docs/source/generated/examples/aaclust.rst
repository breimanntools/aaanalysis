The ``AAclust`` clustering wrapper framework can utilize any clustering
model that uses the ``n_clusters`` parameter:

.. code:: ipython2

    from sklearn.cluster import (KMeans, AgglomerativeClustering, MiniBatchKMeans, SpectralClustering)
    import aaanalysis as aa
    
    # AAclust with KMens (default)
    aac = aa.AAclust(model_class=KMeans)
    # AAclust with MiniBatchKMeans
    aac = aa.AAclust(model_class=MiniBatchKMeans)
    # AAclust with SpectralClustering
    aac = aa.AAclust(model_class=SpectralClustering)

The hierarchical agglomerative clustering model utilizes four different
distance measures, which can be provided to ``AAclust``\ by its
``model_kwargs`` parameter:

.. code:: ipython2

    # AAclust using AgglomerativeClustering with Euclidean distance
    aac = aa.AAclust(model_class=AgglomerativeClustering, model_kwargs=dict(metric='euclidean'))
    # Other recommended metrics are 'manhattan', 'cosine'

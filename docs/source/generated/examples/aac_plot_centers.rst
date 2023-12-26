We first create an example dataset:

.. code:: ipython2

    from sklearn.decomposition import PCA, KernelPCA, FastICA, TruncatedSVD, NMF
    import matplotlib.pyplot as plt
    import aaanalysis as aa
    aa.options["verbose"] = False
    # Obtain example scale dataset 
    df_scales = aa.load_scales()
    X = df_scales.T
    # Fit AAclust model retrieve labels to compute centers
    aac = aa.AAclust()
    labels = aac.fit(X, n_clusters=5).labels_
    centers, labels_centers = aac.comp_centers(X=X, labels=labels)

All data points are visualized in the PCA plot including the cluster
centers highlighted by an ‘x’:

.. code:: ipython2

    aac_plot = aa.AAclustPlot(model_class=PCA)
    aa.plot_settings()
    aac_plot.centers(X, labels=labels)
    plt.show()



.. image:: examples/aac_plot_centers_1_output_3_0.png


Compression of different Transformer models can be compared:

.. code:: ipython2

    list_models = [KernelPCA, FastICA, TruncatedSVD, NMF]


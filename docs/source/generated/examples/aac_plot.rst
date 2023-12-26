The ``AAclustPlot`` object utilizes Transformer models, such as
Principal Component Analysis (PCA), to visualize the ``AAclust``
clustering results. Valid models can be provided via the ``model_class``
parameter (default=PCA):

.. code:: ipython2

    import aaanalysis as aa
    # Valid transformer models
    from sklearn.decomposition import PCA, KernelPCA, FastICA, TruncatedSVD, NMF
    from sklearn.manifold import LocallyLinearEmbedding, Isomap
    # Initialize AAclustPlot with PCA
    aac_plot = aa.AAclustPlot(model_class=PCA)

Arguments of the transformer model can be set using the ``model_kwargs``
parameters:

.. code:: ipython2

    aac_plot = aa.AAclustPlot(model_class=PCA, model_kwargs=dict(svd_solver="full"))

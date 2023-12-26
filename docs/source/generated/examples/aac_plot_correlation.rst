To showcase the ``AAclustPlot().correlation()`` method, we create an
example dataset and obtained a DataFrame with pairwise correlations
(``df_corr``) using the ``AAclust().correlation()`` method:

.. code:: ipython2

    import matplotlib.pyplot as plt
    import aaanalysis as aa
    aa.options["verbose"] = False
    # Obtain example scale dataset 
    df_scales = aa.load_scales(unclassified_out=True).T.sample(50).T
    df_cat = aa.load_scales(name="scales_cat")
    dict_scale_name = dict(zip(df_cat["scale_id"], df_cat["subcategory"]))
    names = [dict_scale_name[s] for s in list(df_scales)]
    X = df_scales.T
    # Fit AAclust model and retrieve labels, cluster names, and df_corr
    aac = aa.AAclust()
    labels = aac.fit(X, n_clusters=10).labels_
    df_corr, labels_sorted = aac.comp_correlation(X=X, labels=labels)

The pair-wise Pearson correlation can now be visualized using the
``AAclustPlot().correlation()`` method. Provide the labels sorted as in
``df_corr``.

.. code:: ipython2

    aac_plot = aa.AAclustPlot()
    aa.plot_settings(font_scale=0.7, weight_bold=False, no_ticks=True)
    aac_plot.correlation(df_corr=df_corr, labels=labels_sorted)
    plt.show()



.. image:: examples/aac_plot_correlation_1_output_3_0.png


Gray bars indicate the clusters. To change their position or provide
multiple bars, use the ``bar_position`` parameter and adjust their width
and spacing by using ``bar_width_x``, ``bar_width_y``,
``bar_spacing_x``, and ``bar_spacing_y``

.. code:: ipython2

    aac_plot.correlation(df_corr=df_corr, labels=labels_sorted, bar_position=["left", "top"],
                         bar_width_x=1, bar_width_y=0.5, bar_spacing_x=1, bar_spacing_y=0.5)
    plt.show()



.. image:: examples/aac_plot_correlation_2_output_5_0.png


To obtain the correlation between each scale (y-axis) and the medoids
(x-axis), we obtain the medoids using the ``AAclust().comp_medoids()``
and ``AAclust().comp_correlation()`` methods.

.. code:: ipython2

    X_ref, labels_ref = aac.comp_medoids(X, labels=labels)
    # Creat correlation DataFrane between scales and medoids
    df_corr, labels_sorted = aac.comp_correlation(X=X, labels=labels, X_ref=X_ref, labels_ref=labels_ref)
    # Plot correlation
    aac_plot.correlation(df_corr=df_corr, labels=labels_sorted)
    plt.tight_layout()
    plt.show()



.. image:: examples/aac_plot_correlation_3_output_7_0.png


We can re-clustered the x-axis values be setting ``cluster_x=True``. The
``scipy.cluster.hierarchy.linkage`` method is internally used, for which
the linkage method can be selected by the ``method`` parameter
(default=\ ``average``):

.. code:: ipython2

    aac_plot.correlation(df_corr=df_corr, labels=labels_sorted, cluster_x=True, method="ward")
    plt.tight_layout()
    plt.show()



.. image:: examples/aac_plot_correlation_4_output_9_0.png


To show the respective scale and cluster names, provide them to the
``AAclust().comp_correlation()`` method and use the
``xtick_label_rotation`` parameter to rotate the x-ticks:

.. code:: ipython2

    # Creat correlation DataFrane between scales and medoids
    cluster_names = aac.name_clusters(X, labels=labels, names=names)
    dict_cluster = dict(zip(labels, cluster_names))
    names_ref = [dict_cluster[i] for i in labels_ref]
    df_corr, labels_sorted = aac.comp_correlation(X=X, labels=labels, X_ref=X_ref, labels_ref=labels_ref, names=names, names_ref=names_ref)
    # Plot correlation
    aac_plot.correlation(df_corr=df_corr, labels=labels_sorted, xtick_label_rotation=45)
    plt.tight_layout()
    plt.show()



.. image:: examples/aac_plot_correlation_5_output_11_0.png


The clusters can be colored using the ``bar_colors`` parameter:

.. code:: ipython2

    # Plot correlation
    n_clusters = len(set(labels_sorted))
    colors = aa.plot_get_clist(n_colors=n_clusters)
    aac_plot.correlation(df_corr=df_corr, labels=labels_sorted, xtick_label_rotation=45,
                         bar_colors=colors, bar_position=["left", "bottom"], bar_width_x=1, bar_width_y=0.2)
    plt.tight_layout()
    plt.show()



.. image:: examples/aac_plot_correlation_6_output_13_0.png


While ``vmin``, ``vmax``, anx ``cmap`` can be directly adjusted, further
keyword arguments for the ``sns.heatmap()`` function can be provided by
the ``kwargs_heatmap`` argument:

.. code:: ipython2

    # Plot correlation
    aac_plot.correlation(df_corr=df_corr, labels=labels_sorted, xtick_label_rotation=45,
                         vmin=-0.5, vmax=0.5, cmap="cividis", kwargs_heatmap=dict(linecolor="black"))
    plt.tight_layout()
    plt.show()



.. image:: examples/aac_plot_correlation_7_output_15_0.png


We first create an example dataset for the ``AAclustPlot().medoids()``,
which visualizes ‘medoids’ as obtained by the
``AAclust().comp_medoids()`` method:

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

All data points are visualized in the PCA plot including the cluster
representative samples (‘medoids’) highlighted by a bigger dot:

.. code:: ipython2

    aac_plot = aa.AAclustPlot(model_class=PCA)
    aa.plot_settings()
    ax, df_components = aac_plot.medoids(X, labels=labels)
    plt.show()
    # DataFrame for respective components are returned
    aa.display_df(df_components, n_rows=10, show_shape=True)



.. image:: examples/aac_plot_medoids_1_output_3_0.png


.. parsed-literal::

    DataFrame shape: (586, 2)



.. raw:: html

    <style type="text/css">
    #T_280f0 thead th {
      background-color: white;
      color: black;
    }
    #T_280f0 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_280f0 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_280f0 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_280f0  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_280f0 table {
      font-size: 12px;
    }
    </style>
    <table id="T_280f0" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_280f0_level0_col0" class="col_heading level0 col0" >PC1 (33.6%)</th>
          <th id="T_280f0_level0_col1" class="col_heading level0 col1" >PC2 (17.7%)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_280f0_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_280f0_row0_col0" class="data row0 col0" >-0.181000</td>
          <td id="T_280f0_row0_col1" class="data row0 col1" >0.580000</td>
        </tr>
        <tr>
          <th id="T_280f0_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_280f0_row1_col0" class="data row1 col0" >0.824000</td>
          <td id="T_280f0_row1_col1" class="data row1 col1" >-0.592000</td>
        </tr>
        <tr>
          <th id="T_280f0_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_280f0_row2_col0" class="data row2 col0" >0.724000</td>
          <td id="T_280f0_row2_col1" class="data row2 col1" >-0.838000</td>
        </tr>
        <tr>
          <th id="T_280f0_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_280f0_row3_col0" class="data row3 col0" >0.861000</td>
          <td id="T_280f0_row3_col1" class="data row3 col1" >-0.746000</td>
        </tr>
        <tr>
          <th id="T_280f0_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_280f0_row4_col0" class="data row4 col0" >0.645000</td>
          <td id="T_280f0_row4_col1" class="data row4 col1" >0.481000</td>
        </tr>
        <tr>
          <th id="T_280f0_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_280f0_row5_col0" class="data row5 col0" >1.266000</td>
          <td id="T_280f0_row5_col1" class="data row5 col1" >-0.149000</td>
        </tr>
        <tr>
          <th id="T_280f0_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_280f0_row6_col0" class="data row6 col0" >-0.753000</td>
          <td id="T_280f0_row6_col1" class="data row6 col1" >0.413000</td>
        </tr>
        <tr>
          <th id="T_280f0_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_280f0_row7_col0" class="data row7 col0" >-1.074000</td>
          <td id="T_280f0_row7_col1" class="data row7 col1" >0.348000</td>
        </tr>
        <tr>
          <th id="T_280f0_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_280f0_row8_col0" class="data row8 col0" >0.501000</td>
          <td id="T_280f0_row8_col1" class="data row8 col1" >0.262000</td>
        </tr>
        <tr>
          <th id="T_280f0_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_280f0_row9_col0" class="data row9 col0" >1.304000</td>
          <td id="T_280f0_row9_col1" class="data row9 col1" >-0.139000</td>
        </tr>
      </tbody>
    </table>



Select other PCs using the ``component_x`` and ``component_y``
parameters:

.. code:: ipython2

    aac_plot.medoids(X, labels=labels, component_x=3, component_y=4)
    plt.show()



.. image:: examples/aac_plot_medoids_2_output_5_0.png


Medoids can be obtained using different kinds of ``metric``. To compare
them in a single plot, you can use the ``ax`` and ``legend`` parameters:

.. code:: ipython2

    list_metrics = ["correlation", "euclidean", "manhattan", "cosine"]
    fig, axes = plt.subplots(4, 1, figsize=(7, 14), sharex=True, sharey=True)
    for i, metric in enumerate(list_metrics):
        ax = axes[i]
        # Set legend only for first subplot
        aac_plot.medoids(X, labels=labels, ax=ax, legend=i==0, metric=metric)
        ax.set_title(metric)
    plt.tight_layout()
    plt.show()
    plt.close()
        



.. image:: examples/aac_plot_medoids_3_output_7_0.png


Adjust the style of the scatter plot using the ``dot_size`` and
``dot_alpha`` arguments to change the size of the dots and their
transparency:

.. code:: ipython2

    aac_plot = aa.AAclustPlot(model_class=PCA)
    aac_plot.medoids(X, labels=labels, dot_size=50, dot_alpha=1)
    plt.show()



.. image:: examples/aac_plot_medoids_4_output_9_0.png


The cluster colors can be adjusted by the ``palette`` argument by
providing either a list of colors or a color map:

.. code:: ipython2

    colors = aa.plot_get_clist(n_colors=5)
    aac_plot.medoids(X, labels=labels, palette=colors)
    plt.show()
    aac_plot.medoids(X, labels=labels, palette="viridis")
    plt.show()



.. image:: examples/aac_plot_medoids_5_output_11_0.png



.. image:: examples/aac_plot_medoids_6_output_11_1.png


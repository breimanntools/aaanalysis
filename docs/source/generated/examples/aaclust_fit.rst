We load an example scale dataset to showcase the ``AAclust.fit()``
method:

.. code:: ipython2

    import aaanalysis as aa
    aa.options["verbose"] = False
    # Create test dataset of 25 amino acid scales
    df_scales = aa.load_scales().T.sample(25).T
    X = df_scales.T

By fitting ``AAclust``, its three-step algorithm is performed to select
an optimized ``n_clusters`` (k). The three steps involve (1) an
estimation of lower bound of k, (2) refinement of k, and (3) an optional
clustering merging. Various results are saved as attributes:

.. code:: ipython2

    # Fit clustering model
    aac = aa.AAclust()
    aac.fit(X)
    # Get output parameters
    n_clusters = aac.n_clusters
    print("n_clusters: ", n_clusters)
    labels = aac.labels_
    print("Labels: ", labels)
    centers = aac.centers_ # Cluster centers (average scales for each cluster)
    labels_centers = aac.labels_centers_
    medoids = aac.medoids_ # Representative scale for each cluster
    labels_medoids = aac.labels_medoids_
    print("Labels of medoids: ", labels_medoids)
    is_medoid = aac.is_medoid_
    df_scales_medoids = df_scales.T[is_medoid].T
    aa.display_df(df_scales_medoids)


.. parsed-literal::

    n_clusters:  4
    Labels:  [2 2 1 1 1 2 3 0 1 0 0 0 0 1 3 3 3 2 0 1 1 0 0 1 0]
    Labels of medoids:  [2 1 3 0]



.. raw:: html

    <style type="text/css">
    #T_52c2f thead th {
      background-color: white;
      color: black;
    }
    #T_52c2f tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_52c2f tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_52c2f th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_52c2f  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_52c2f table {
      font-size: 12px;
    }
    </style>
    <table id="T_52c2f" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_52c2f_level0_col0" class="col_heading level0 col0" >LINS030117</th>
          <th id="T_52c2f_level0_col1" class="col_heading level0 col1" >KOEH090110</th>
          <th id="T_52c2f_level0_col2" class="col_heading level0 col2" >CHOP780201</th>
          <th id="T_52c2f_level0_col3" class="col_heading level0 col3" >CASG920101</th>
        </tr>
        <tr>
          <th class="index_name level0" >AA</th>
          <th class="blank col0" >&nbsp;</th>
          <th class="blank col1" >&nbsp;</th>
          <th class="blank col2" >&nbsp;</th>
          <th class="blank col3" >&nbsp;</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_52c2f_level0_row0" class="row_heading level0 row0" >A</th>
          <td id="T_52c2f_row0_col0" class="data row0 col0" >0.186000</td>
          <td id="T_52c2f_row0_col1" class="data row0 col1" >0.140000</td>
          <td id="T_52c2f_row0_col2" class="data row0 col2" >0.904000</td>
          <td id="T_52c2f_row0_col3" class="data row0 col3" >0.514000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row1" class="row_heading level0 row1" >C</th>
          <td id="T_52c2f_row1_col0" class="data row1 col0" >0.000000</td>
          <td id="T_52c2f_row1_col1" class="data row1 col1" >0.285000</td>
          <td id="T_52c2f_row1_col2" class="data row1 col2" >0.138000</td>
          <td id="T_52c2f_row1_col3" class="data row1 col3" >1.000000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row2" class="row_heading level0 row2" >D</th>
          <td id="T_52c2f_row2_col0" class="data row2 col0" >0.186000</td>
          <td id="T_52c2f_row2_col1" class="data row2 col1" >0.919000</td>
          <td id="T_52c2f_row2_col2" class="data row2 col2" >0.468000</td>
          <td id="T_52c2f_row2_col3" class="data row2 col3" >0.057000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row3" class="row_heading level0 row3" >E</th>
          <td id="T_52c2f_row3_col0" class="data row3 col0" >0.349000</td>
          <td id="T_52c2f_row3_col1" class="data row3 col1" >0.913000</td>
          <td id="T_52c2f_row3_col2" class="data row3 col2" >1.000000</td>
          <td id="T_52c2f_row3_col3" class="data row3 col3" >0.086000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row4" class="row_heading level0 row4" >F</th>
          <td id="T_52c2f_row4_col0" class="data row4 col0" >0.326000</td>
          <td id="T_52c2f_row4_col1" class="data row4 col1" >0.029000</td>
          <td id="T_52c2f_row4_col2" class="data row4 col2" >0.596000</td>
          <td id="T_52c2f_row4_col3" class="data row4 col3" >0.743000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row5" class="row_heading level0 row5" >G</th>
          <td id="T_52c2f_row5_col0" class="data row5 col0" >0.023000</td>
          <td id="T_52c2f_row5_col1" class="data row5 col1" >0.221000</td>
          <td id="T_52c2f_row5_col2" class="data row5 col2" >0.000000</td>
          <td id="T_52c2f_row5_col3" class="data row5 col3" >0.429000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row6" class="row_heading level0 row6" >H</th>
          <td id="T_52c2f_row6_col0" class="data row6 col0" >0.419000</td>
          <td id="T_52c2f_row6_col1" class="data row6 col1" >0.651000</td>
          <td id="T_52c2f_row6_col2" class="data row6 col2" >0.457000</td>
          <td id="T_52c2f_row6_col3" class="data row6 col3" >0.571000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row7" class="row_heading level0 row7" >I</th>
          <td id="T_52c2f_row7_col0" class="data row7 col0" >0.140000</td>
          <td id="T_52c2f_row7_col1" class="data row7 col1" >0.029000</td>
          <td id="T_52c2f_row7_col2" class="data row7 col2" >0.543000</td>
          <td id="T_52c2f_row7_col3" class="data row7 col3" >0.857000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row8" class="row_heading level0 row8" >K</th>
          <td id="T_52c2f_row8_col0" class="data row8 col0" >1.000000</td>
          <td id="T_52c2f_row8_col1" class="data row8 col1" >1.000000</td>
          <td id="T_52c2f_row8_col2" class="data row8 col2" >0.628000</td>
          <td id="T_52c2f_row8_col3" class="data row8 col3" >0.000000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row9" class="row_heading level0 row9" >L</th>
          <td id="T_52c2f_row9_col0" class="data row9 col0" >0.186000</td>
          <td id="T_52c2f_row9_col1" class="data row9 col1" >0.000000</td>
          <td id="T_52c2f_row9_col2" class="data row9 col2" >0.681000</td>
          <td id="T_52c2f_row9_col3" class="data row9 col3" >0.600000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row10" class="row_heading level0 row10" >M</th>
          <td id="T_52c2f_row10_col0" class="data row10 col0" >0.372000</td>
          <td id="T_52c2f_row10_col1" class="data row10 col1" >0.180000</td>
          <td id="T_52c2f_row10_col2" class="data row10 col2" >0.936000</td>
          <td id="T_52c2f_row10_col3" class="data row10 col3" >0.600000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row11" class="row_heading level0 row11" >N</th>
          <td id="T_52c2f_row11_col0" class="data row11 col0" >0.093000</td>
          <td id="T_52c2f_row11_col1" class="data row11 col1" >0.599000</td>
          <td id="T_52c2f_row11_col2" class="data row11 col2" >0.106000</td>
          <td id="T_52c2f_row11_col3" class="data row11 col3" >0.314000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row12" class="row_heading level0 row12" >P</th>
          <td id="T_52c2f_row12_col0" class="data row12 col0" >0.698000</td>
          <td id="T_52c2f_row12_col1" class="data row12 col1" >0.570000</td>
          <td id="T_52c2f_row12_col2" class="data row12 col2" >0.000000</td>
          <td id="T_52c2f_row12_col3" class="data row12 col3" >0.171000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row13" class="row_heading level0 row13" >Q</th>
          <td id="T_52c2f_row13_col0" class="data row13 col0" >0.256000</td>
          <td id="T_52c2f_row13_col1" class="data row13 col1" >0.733000</td>
          <td id="T_52c2f_row13_col2" class="data row13 col2" >0.574000</td>
          <td id="T_52c2f_row13_col3" class="data row13 col3" >0.143000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row14" class="row_heading level0 row14" >R</th>
          <td id="T_52c2f_row14_col0" class="data row14 col0" >0.372000</td>
          <td id="T_52c2f_row14_col1" class="data row14 col1" >1.000000</td>
          <td id="T_52c2f_row14_col2" class="data row14 col2" >0.436000</td>
          <td id="T_52c2f_row14_col3" class="data row14 col3" >0.257000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row15" class="row_heading level0 row15" >S</th>
          <td id="T_52c2f_row15_col0" class="data row15 col0" >0.186000</td>
          <td id="T_52c2f_row15_col1" class="data row15 col1" >0.384000</td>
          <td id="T_52c2f_row15_col2" class="data row15 col2" >0.213000</td>
          <td id="T_52c2f_row15_col3" class="data row15 col3" >0.257000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row16" class="row_heading level0 row16" >T</th>
          <td id="T_52c2f_row16_col0" class="data row16 col0" >0.419000</td>
          <td id="T_52c2f_row16_col1" class="data row16 col1" >0.250000</td>
          <td id="T_52c2f_row16_col2" class="data row16 col2" >0.277000</td>
          <td id="T_52c2f_row16_col3" class="data row16 col3" >0.343000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row17" class="row_heading level0 row17" >V</th>
          <td id="T_52c2f_row17_col0" class="data row17 col0" >0.163000</td>
          <td id="T_52c2f_row17_col1" class="data row17 col1" >0.076000</td>
          <td id="T_52c2f_row17_col2" class="data row17 col2" >0.521000</td>
          <td id="T_52c2f_row17_col3" class="data row17 col3" >0.657000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row18" class="row_heading level0 row18" >W</th>
          <td id="T_52c2f_row18_col0" class="data row18 col0" >0.349000</td>
          <td id="T_52c2f_row18_col1" class="data row18 col1" >0.244000</td>
          <td id="T_52c2f_row18_col2" class="data row18 col2" >0.543000</td>
          <td id="T_52c2f_row18_col3" class="data row18 col3" >0.914000</td>
        </tr>
        <tr>
          <th id="T_52c2f_level0_row19" class="row_heading level0 row19" >Y</th>
          <td id="T_52c2f_row19_col0" class="data row19 col0" >0.349000</td>
          <td id="T_52c2f_row19_col1" class="data row19 col1" >0.413000</td>
          <td id="T_52c2f_row19_col2" class="data row19 col2" >0.128000</td>
          <td id="T_52c2f_row19_col3" class="data row19 col3" >0.600000</td>
        </tr>
      </tbody>
    </table>



``names`` can be provided to the ``AAclust().fit()`` method to retrieve
the names of the medoids:

.. code:: ipython2

    names = [f"scale {i+1}" for i in range(len(df_scales.T))]
    aac.fit(X, names=names)
    medoid_names = aac.medoid_names_
    print(medoid_names)


.. parsed-literal::

    ['scale 7', 'scale 23', 'scale 3', 'scale 18']


The ``n_clusters`` parameter can as well be pre-defined:

.. code:: ipython2

    aac.fit(X, n_clusters=5, names=names)
    medoid_names = aac.medoid_names_
    print(medoid_names)


.. parsed-literal::

    ['scale 16', 'scale 23', 'scale 3', 'scale 6', 'scale 17']


The second step of the ``AAclust`` algorithm (recursive k optimization)
can be adjusted using the ``min_th`` and ``on_center`` parameters:

.. code:: ipython2

    # Pearson correlation within all cluster members >= 0.5
    aac.fit(X, on_center=False, min_th=0.5)
    print(aac.n_clusters)
    # Pearson correlation between all cluster members and the respective center >= 0.5
    aac.fit(X, on_center=True, min_th=0.5)
    print(aac.n_clusters)
    # The latter is less strict, leading to bigger and thus fewer clusters 


.. parsed-literal::

    14
    4


The third and optional merging step can be adjusted using the ``metric``
parameter and disabled setting ``merge=False``. The attributes can be
directly retrieved since the ``AAclust.fit()`` method returns the fitted
clustering model:

.. code:: ipython2

    # Load over 500 scales
    X = aa.load_scales().T
    n_with_merging_euclidean = aac.fit(X).n_clusters
    n_with_merging_cosine = aac.fit(X, metric="cosine").n_clusters
    n_without_merging = aac.fit(X, merge=False).n_clusters
    print(n_with_merging_euclidean)
    print(n_with_merging_cosine)
    print(n_without_merging)


.. parsed-literal::

    56
    56
    53


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

    n_clusters:  5
    Labels:  [1 1 1 3 0 1 4 3 0 2 0 4 2 0 0 2 1 0 1 0 2 3 3 2 4]
    Labels of medoids:  [1 3 0 4 2]



.. raw:: html

    <style type="text/css">
    #T_c5112 thead th {
      background-color: white;
      color: black;
    }
    #T_c5112 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_c5112 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_c5112 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_c5112  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_c5112 table {
      font-size: 12px;
    }
    </style>
    <table id="T_c5112" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_c5112_level0_col0" class="col_heading level0 col0" >KARS160117</th>
          <th id="T_c5112_level0_col1" class="col_heading level0 col1" >AURR980110</th>
          <th id="T_c5112_level0_col2" class="col_heading level0 col2" >LINS030107</th>
          <th id="T_c5112_level0_col3" class="col_heading level0 col3" >COHE430101</th>
          <th id="T_c5112_level0_col4" class="col_heading level0 col4" >PTIO830101</th>
        </tr>
        <tr>
          <th class="index_name level0" >AA</th>
          <th class="blank col0" >&nbsp;</th>
          <th class="blank col1" >&nbsp;</th>
          <th class="blank col2" >&nbsp;</th>
          <th class="blank col3" >&nbsp;</th>
          <th class="blank col4" >&nbsp;</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_c5112_level0_row0" class="row_heading level0 row0" >A</th>
          <td id="T_c5112_row0_col0" class="data row0 col0" >0.082000</td>
          <td id="T_c5112_row0_col1" class="data row0 col1" >1.000000</td>
          <td id="T_c5112_row0_col2" class="data row0 col2" >0.200000</td>
          <td id="T_c5112_row0_col3" class="data row0 col3" >0.500000</td>
          <td id="T_c5112_row0_col4" class="data row0 col4" >0.870000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row1" class="row_heading level0 row1" >C</th>
          <td id="T_c5112_row1_col0" class="data row1 col0" >0.344000</td>
          <td id="T_c5112_row1_col1" class="data row1 col1" >0.242000</td>
          <td id="T_c5112_row1_col2" class="data row1 col2" >0.000000</td>
          <td id="T_c5112_row1_col3" class="data row1 col3" >0.033000</td>
          <td id="T_c5112_row1_col4" class="data row1 col4" >0.739000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row2" class="row_heading level0 row2" >D</th>
          <td id="T_c5112_row2_col0" class="data row2 col0" >0.443000</td>
          <td id="T_c5112_row2_col1" class="data row2 col1" >0.455000</td>
          <td id="T_c5112_row2_col2" class="data row2 col2" >0.800000</td>
          <td id="T_c5112_row2_col3" class="data row2 col3" >0.000000</td>
          <td id="T_c5112_row2_col4" class="data row2 col4" >0.478000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row3" class="row_heading level0 row3" >E</th>
          <td id="T_c5112_row3_col0" class="data row3 col0" >0.541000</td>
          <td id="T_c5112_row3_col1" class="data row3 col1" >0.958000</td>
          <td id="T_c5112_row3_col2" class="data row3 col2" >0.911000</td>
          <td id="T_c5112_row3_col3" class="data row3 col3" >0.200000</td>
          <td id="T_c5112_row3_col4" class="data row3 col4" >0.783000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row4" class="row_heading level0 row4" >F</th>
          <td id="T_c5112_row4_col0" class="data row4 col0" >0.672000</td>
          <td id="T_c5112_row4_col1" class="data row4 col1" >0.491000</td>
          <td id="T_c5112_row4_col2" class="data row4 col2" >0.067000</td>
          <td id="T_c5112_row4_col3" class="data row4 col3" >0.567000</td>
          <td id="T_c5112_row4_col4" class="data row4 col4" >0.870000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row5" class="row_heading level0 row5" >G</th>
          <td id="T_c5112_row5_col0" class="data row5 col0" >0.000000</td>
          <td id="T_c5112_row5_col1" class="data row5 col1" >0.103000</td>
          <td id="T_c5112_row5_col2" class="data row5 col2" >0.422000</td>
          <td id="T_c5112_row5_col3" class="data row5 col3" >0.133000</td>
          <td id="T_c5112_row5_col4" class="data row5 col4" >0.435000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row6" class="row_heading level0 row6" >H</th>
          <td id="T_c5112_row6_col0" class="data row6 col0" >0.656000</td>
          <td id="T_c5112_row6_col1" class="data row6 col1" >0.188000</td>
          <td id="T_c5112_row6_col2" class="data row6 col2" >0.467000</td>
          <td id="T_c5112_row6_col3" class="data row6 col3" >0.233000</td>
          <td id="T_c5112_row6_col4" class="data row6 col4" >0.652000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row7" class="row_heading level0 row7" >I</th>
          <td id="T_c5112_row7_col0" class="data row7 col0" >0.377000</td>
          <td id="T_c5112_row7_col1" class="data row7 col1" >0.570000</td>
          <td id="T_c5112_row7_col2" class="data row7 col2" >0.022000</td>
          <td id="T_c5112_row7_col3" class="data row7 col3" >1.000000</td>
          <td id="T_c5112_row7_col4" class="data row7 col4" >0.870000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row8" class="row_heading level0 row8" >K</th>
          <td id="T_c5112_row8_col0" class="data row8 col0" >0.492000</td>
          <td id="T_c5112_row8_col1" class="data row8 col1" >0.661000</td>
          <td id="T_c5112_row8_col2" class="data row8 col2" >1.000000</td>
          <td id="T_c5112_row8_col3" class="data row8 col3" >0.733000</td>
          <td id="T_c5112_row8_col4" class="data row8 col4" >0.783000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row9" class="row_heading level0 row9" >L</th>
          <td id="T_c5112_row9_col0" class="data row9 col0" >0.377000</td>
          <td id="T_c5112_row9_col1" class="data row9 col1" >0.800000</td>
          <td id="T_c5112_row9_col2" class="data row9 col2" >0.044000</td>
          <td id="T_c5112_row9_col3" class="data row9 col3" >1.000000</td>
          <td id="T_c5112_row9_col4" class="data row9 col4" >1.000000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row10" class="row_heading level0 row10" >M</th>
          <td id="T_c5112_row10_col0" class="data row10 col0" >0.541000</td>
          <td id="T_c5112_row10_col1" class="data row10 col1" >0.903000</td>
          <td id="T_c5112_row10_col2" class="data row10 col2" >0.089000</td>
          <td id="T_c5112_row10_col3" class="data row10 col3" >0.500000</td>
          <td id="T_c5112_row10_col4" class="data row10 col4" >0.913000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row11" class="row_heading level0 row11" >N</th>
          <td id="T_c5112_row11_col0" class="data row11 col0" >0.426000</td>
          <td id="T_c5112_row11_col1" class="data row11 col1" >0.352000</td>
          <td id="T_c5112_row11_col2" class="data row11 col2" >0.733000</td>
          <td id="T_c5112_row11_col3" class="data row11 col3" >0.033000</td>
          <td id="T_c5112_row11_col4" class="data row11 col4" >0.609000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row12" class="row_heading level0 row12" >P</th>
          <td id="T_c5112_row12_col0" class="data row12 col0" >0.279000</td>
          <td id="T_c5112_row12_col1" class="data row12 col1" >0.000000</td>
          <td id="T_c5112_row12_col2" class="data row12 col2" >0.733000</td>
          <td id="T_c5112_row12_col3" class="data row12 col3" >0.533000</td>
          <td id="T_c5112_row12_col4" class="data row12 col4" >0.000000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row13" class="row_heading level0 row13" >Q</th>
          <td id="T_c5112_row13_col0" class="data row13 col0" >0.525000</td>
          <td id="T_c5112_row13_col1" class="data row13 col1" >0.497000</td>
          <td id="T_c5112_row13_col2" class="data row13 col2" >0.778000</td>
          <td id="T_c5112_row13_col3" class="data row13 col3" >0.233000</td>
          <td id="T_c5112_row13_col4" class="data row13 col4" >0.783000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row14" class="row_heading level0 row14" >R</th>
          <td id="T_c5112_row14_col0" class="data row14 col0" >0.623000</td>
          <td id="T_c5112_row14_col1" class="data row14 col1" >0.958000</td>
          <td id="T_c5112_row14_col2" class="data row14 col2" >0.711000</td>
          <td id="T_c5112_row14_col3" class="data row14 col3" >0.333000</td>
          <td id="T_c5112_row14_col4" class="data row14 col4" >0.739000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row15" class="row_heading level0 row15" >S</th>
          <td id="T_c5112_row15_col0" class="data row15 col0" >0.246000</td>
          <td id="T_c5112_row15_col1" class="data row15 col1" >0.315000</td>
          <td id="T_c5112_row15_col2" class="data row15 col2" >0.556000</td>
          <td id="T_c5112_row15_col3" class="data row15 col3" >0.267000</td>
          <td id="T_c5112_row15_col4" class="data row15 col4" >0.565000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row16" class="row_heading level0 row16" >T</th>
          <td id="T_c5112_row16_col0" class="data row16 col0" >0.328000</td>
          <td id="T_c5112_row16_col1" class="data row16 col1" >0.333000</td>
          <td id="T_c5112_row16_col2" class="data row16 col2" >0.511000</td>
          <td id="T_c5112_row16_col3" class="data row16 col3" >0.333000</td>
          <td id="T_c5112_row16_col4" class="data row16 col4" >0.565000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row17" class="row_heading level0 row17" >V</th>
          <td id="T_c5112_row17_col0" class="data row17 col0" >0.279000</td>
          <td id="T_c5112_row17_col1" class="data row17 col1" >0.400000</td>
          <td id="T_c5112_row17_col2" class="data row17 col2" >0.044000</td>
          <td id="T_c5112_row17_col3" class="data row17 col3" >0.867000</td>
          <td id="T_c5112_row17_col4" class="data row17 col4" >0.739000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row18" class="row_heading level0 row18" >W</th>
          <td id="T_c5112_row18_col0" class="data row18 col0" >1.000000</td>
          <td id="T_c5112_row18_col1" class="data row18 col1" >0.321000</td>
          <td id="T_c5112_row18_col2" class="data row18 col2" >0.156000</td>
          <td id="T_c5112_row18_col3" class="data row18 col3" >0.467000</td>
          <td id="T_c5112_row18_col4" class="data row18 col4" >0.870000</td>
        </tr>
        <tr>
          <th id="T_c5112_level0_row19" class="row_heading level0 row19" >Y</th>
          <td id="T_c5112_row19_col0" class="data row19 col0" >0.803000</td>
          <td id="T_c5112_row19_col1" class="data row19 col1" >0.461000</td>
          <td id="T_c5112_row19_col2" class="data row19 col2" >0.244000</td>
          <td id="T_c5112_row19_col3" class="data row19 col3" >0.367000</td>
          <td id="T_c5112_row19_col4" class="data row19 col4" >0.870000</td>
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

    ['scale 24', 'scale 3', 'scale 14', 'scale 11']


The ``n_clusters`` parameter can as well be pre-defined:

.. code:: ipython2

    aac.fit(X, n_clusters=5, names=names)
    medoid_names = aac.medoid_names_
    print(medoid_names)


.. parsed-literal::

    ['scale 5', 'scale 17', 'scale 23', 'scale 10', 'scale 25']


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

    17
    7


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

    49
    57
    59


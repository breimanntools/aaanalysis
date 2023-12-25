The cluster centers can be computed using the
``AAclust().comp_centers()`` method:

.. code:: ipython2

    import aaanalysis as aa
    import pandas as pd
    # Create example dataset comprising 100 scales
    df_scales = aa.load_scales().T.sample(100).T
    X = df_scales.T
    # Fit AAclust model and obtain clustering centers for 5 clusters
    aac = aa.AAclust()
    labels = aac.fit(X, n_clusters=5).labels_
    centers, labels_centers = aac.comp_centers(X=X, labels=labels)
    # Create DataFrame with cluster centers
    columns = [f"Cluster {i}" for i in labels_centers]
    df_centers = pd.DataFrame(centers.T, columns=columns, index=df_scales.index)
    aa.display_df(df_centers)



.. raw:: html

    <style type="text/css">
    #T_3d623 thead th {
      background-color: white;
      color: black;
    }
    #T_3d623 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_3d623 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_3d623 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_3d623  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_3d623 table {
      font-size: 12px;
    }
    </style>
    <table id="T_3d623" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_3d623_level0_col0" class="col_heading level0 col0" >Cluster 4</th>
          <th id="T_3d623_level0_col1" class="col_heading level0 col1" >Cluster 0</th>
          <th id="T_3d623_level0_col2" class="col_heading level0 col2" >Cluster 2</th>
          <th id="T_3d623_level0_col3" class="col_heading level0 col3" >Cluster 3</th>
          <th id="T_3d623_level0_col4" class="col_heading level0 col4" >Cluster 1</th>
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
          <th id="T_3d623_level0_row0" class="row_heading level0 row0" >A</th>
          <td id="T_3d623_row0_col0" class="data row0 col0" >0.239000</td>
          <td id="T_3d623_row0_col1" class="data row0 col1" >0.230000</td>
          <td id="T_3d623_row0_col2" class="data row0 col2" >0.404000</td>
          <td id="T_3d623_row0_col3" class="data row0 col3" >0.526000</td>
          <td id="T_3d623_row0_col4" class="data row0 col4" >0.496000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row1" class="row_heading level0 row1" >C</th>
          <td id="T_3d623_row1_col0" class="data row1 col0" >0.370000</td>
          <td id="T_3d623_row1_col1" class="data row1 col1" >0.353000</td>
          <td id="T_3d623_row1_col2" class="data row1 col2" >0.524000</td>
          <td id="T_3d623_row1_col3" class="data row1 col3" >0.434000</td>
          <td id="T_3d623_row1_col4" class="data row1 col4" >0.266000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row2" class="row_heading level0 row2" >D</th>
          <td id="T_3d623_row2_col0" class="data row2 col0" >0.615000</td>
          <td id="T_3d623_row2_col1" class="data row2 col1" >0.373000</td>
          <td id="T_3d623_row2_col2" class="data row2 col2" >0.173000</td>
          <td id="T_3d623_row2_col3" class="data row2 col3" >0.511000</td>
          <td id="T_3d623_row2_col4" class="data row2 col4" >0.755000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row3" class="row_heading level0 row3" >E</th>
          <td id="T_3d623_row3_col0" class="data row3 col0" >0.446000</td>
          <td id="T_3d623_row3_col1" class="data row3 col1" >0.434000</td>
          <td id="T_3d623_row3_col2" class="data row3 col2" >0.193000</td>
          <td id="T_3d623_row3_col3" class="data row3 col3" >0.636000</td>
          <td id="T_3d623_row3_col4" class="data row3 col4" >0.687000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row4" class="row_heading level0 row4" >F</th>
          <td id="T_3d623_row4_col0" class="data row4 col0" >0.306000</td>
          <td id="T_3d623_row4_col1" class="data row4 col1" >0.484000</td>
          <td id="T_3d623_row4_col2" class="data row4 col2" >0.753000</td>
          <td id="T_3d623_row4_col3" class="data row4 col3" >0.767000</td>
          <td id="T_3d623_row4_col4" class="data row4 col4" >0.178000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row5" class="row_heading level0 row5" >G</th>
          <td id="T_3d623_row5_col0" class="data row5 col0" >0.640000</td>
          <td id="T_3d623_row5_col1" class="data row5 col1" >0.157000</td>
          <td id="T_3d623_row5_col2" class="data row5 col2" >0.321000</td>
          <td id="T_3d623_row5_col3" class="data row5 col3" >0.167000</td>
          <td id="T_3d623_row5_col4" class="data row5 col4" >0.700000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row6" class="row_heading level0 row6" >H</th>
          <td id="T_3d623_row6_col0" class="data row6 col0" >0.376000</td>
          <td id="T_3d623_row6_col1" class="data row6 col1" >0.478000</td>
          <td id="T_3d623_row6_col2" class="data row6 col2" >0.399000</td>
          <td id="T_3d623_row6_col3" class="data row6 col3" >0.698000</td>
          <td id="T_3d623_row6_col4" class="data row6 col4" >0.476000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row7" class="row_heading level0 row7" >I</th>
          <td id="T_3d623_row7_col0" class="data row7 col0" >0.168000</td>
          <td id="T_3d623_row7_col1" class="data row7 col1" >0.326000</td>
          <td id="T_3d623_row7_col2" class="data row7 col2" >0.859000</td>
          <td id="T_3d623_row7_col3" class="data row7 col3" >0.611000</td>
          <td id="T_3d623_row7_col4" class="data row7 col4" >0.239000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row8" class="row_heading level0 row8" >K</th>
          <td id="T_3d623_row8_col0" class="data row8 col0" >0.304000</td>
          <td id="T_3d623_row8_col1" class="data row8 col1" >0.665000</td>
          <td id="T_3d623_row8_col2" class="data row8 col2" >0.236000</td>
          <td id="T_3d623_row8_col3" class="data row8 col3" >0.708000</td>
          <td id="T_3d623_row8_col4" class="data row8 col4" >0.772000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row9" class="row_heading level0 row9" >L</th>
          <td id="T_3d623_row9_col0" class="data row9 col0" >0.151000</td>
          <td id="T_3d623_row9_col1" class="data row9 col1" >0.399000</td>
          <td id="T_3d623_row9_col2" class="data row9 col2" >0.749000</td>
          <td id="T_3d623_row9_col3" class="data row9 col3" >0.745000</td>
          <td id="T_3d623_row9_col4" class="data row9 col4" >0.279000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row10" class="row_heading level0 row10" >M</th>
          <td id="T_3d623_row10_col0" class="data row10 col0" >0.167000</td>
          <td id="T_3d623_row10_col1" class="data row10 col1" >0.421000</td>
          <td id="T_3d623_row10_col2" class="data row10 col2" >0.590000</td>
          <td id="T_3d623_row10_col3" class="data row10 col3" >0.775000</td>
          <td id="T_3d623_row10_col4" class="data row10 col4" >0.177000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row11" class="row_heading level0 row11" >N</th>
          <td id="T_3d623_row11_col0" class="data row11 col0" >0.491000</td>
          <td id="T_3d623_row11_col1" class="data row11 col1" >0.362000</td>
          <td id="T_3d623_row11_col2" class="data row11 col2" >0.223000</td>
          <td id="T_3d623_row11_col3" class="data row11 col3" >0.528000</td>
          <td id="T_3d623_row11_col4" class="data row11 col4" >0.715000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row12" class="row_heading level0 row12" >P</th>
          <td id="T_3d623_row12_col0" class="data row12 col0" >0.484000</td>
          <td id="T_3d623_row12_col1" class="data row12 col1" >0.272000</td>
          <td id="T_3d623_row12_col2" class="data row12 col2" >0.332000</td>
          <td id="T_3d623_row12_col3" class="data row12 col3" >0.403000</td>
          <td id="T_3d623_row12_col4" class="data row12 col4" >0.650000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row13" class="row_heading level0 row13" >Q</th>
          <td id="T_3d623_row13_col0" class="data row13 col0" >0.365000</td>
          <td id="T_3d623_row13_col1" class="data row13 col1" >0.451000</td>
          <td id="T_3d623_row13_col2" class="data row13 col2" >0.212000</td>
          <td id="T_3d623_row13_col3" class="data row13 col3" >0.638000</td>
          <td id="T_3d623_row13_col4" class="data row13 col4" >0.624000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row14" class="row_heading level0 row14" >R</th>
          <td id="T_3d623_row14_col0" class="data row14 col0" >0.233000</td>
          <td id="T_3d623_row14_col1" class="data row14 col1" >0.624000</td>
          <td id="T_3d623_row14_col2" class="data row14 col2" >0.249000</td>
          <td id="T_3d623_row14_col3" class="data row14 col3" >0.819000</td>
          <td id="T_3d623_row14_col4" class="data row14 col4" >0.683000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row15" class="row_heading level0 row15" >S</th>
          <td id="T_3d623_row15_col0" class="data row15 col0" >0.483000</td>
          <td id="T_3d623_row15_col1" class="data row15 col1" >0.274000</td>
          <td id="T_3d623_row15_col2" class="data row15 col2" >0.322000</td>
          <td id="T_3d623_row15_col3" class="data row15 col3" >0.418000</td>
          <td id="T_3d623_row15_col4" class="data row15 col4" >0.642000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row16" class="row_heading level0 row16" >T</th>
          <td id="T_3d623_row16_col0" class="data row16 col0" >0.317000</td>
          <td id="T_3d623_row16_col1" class="data row16 col1" >0.308000</td>
          <td id="T_3d623_row16_col2" class="data row16 col2" >0.424000</td>
          <td id="T_3d623_row16_col3" class="data row16 col3" >0.470000</td>
          <td id="T_3d623_row16_col4" class="data row16 col4" >0.539000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row17" class="row_heading level0 row17" >V</th>
          <td id="T_3d623_row17_col0" class="data row17 col0" >0.158000</td>
          <td id="T_3d623_row17_col1" class="data row17 col1" >0.274000</td>
          <td id="T_3d623_row17_col2" class="data row17 col2" >0.780000</td>
          <td id="T_3d623_row17_col3" class="data row17 col3" >0.558000</td>
          <td id="T_3d623_row17_col4" class="data row17 col4" >0.292000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row18" class="row_heading level0 row18" >W</th>
          <td id="T_3d623_row18_col0" class="data row18 col0" >0.207000</td>
          <td id="T_3d623_row18_col1" class="data row18 col1" >0.675000</td>
          <td id="T_3d623_row18_col2" class="data row18 col2" >0.594000</td>
          <td id="T_3d623_row18_col3" class="data row18 col3" >0.766000</td>
          <td id="T_3d623_row18_col4" class="data row18 col4" >0.183000</td>
        </tr>
        <tr>
          <th id="T_3d623_level0_row19" class="row_heading level0 row19" >Y</th>
          <td id="T_3d623_row19_col0" class="data row19 col0" >0.336000</td>
          <td id="T_3d623_row19_col1" class="data row19 col1" >0.587000</td>
          <td id="T_3d623_row19_col2" class="data row19 col2" >0.613000</td>
          <td id="T_3d623_row19_col3" class="data row19 col3" >0.763000</td>
          <td id="T_3d623_row19_col4" class="data row19 col4" >0.412000</td>
        </tr>
      </tbody>
    </table>



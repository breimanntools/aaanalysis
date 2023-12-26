The representative samples for each cluster (called ‘medoids’) can be
obtained using the ``AAclust().comp_medoids()`` method:

.. code:: ipython2

    import aaanalysis as aa
    import pandas as pd
    # Create example dataset comprising 100 scales
    df_scales = aa.load_scales().T.sample(100).T
    X = df_scales.T
    
    # Fit AAclust model and obtain clustering centers for 5 clusters
    aac = aa.AAclust()
    labels = aac.fit(X, n_clusters=5).labels_
    centers, labels_centers = aac.comp_medoids(X=X, labels=labels)
    
    # Create DataFrame with cluster centers
    columns = [f"Cluster {i}" for i in labels_centers]
    df_medoids = pd.DataFrame(centers.T, columns=columns, index=df_scales.index)
    aa.display_df(df_medoids)



.. raw:: html

    <style type="text/css">
    #T_20950 thead th {
      background-color: white;
      color: black;
    }
    #T_20950 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_20950 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_20950 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_20950  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_20950 table {
      font-size: 12px;
    }
    </style>
    <table id="T_20950" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_20950_level0_col0" class="col_heading level0 col0" >Cluster 4</th>
          <th id="T_20950_level0_col1" class="col_heading level0 col1" >Cluster 2</th>
          <th id="T_20950_level0_col2" class="col_heading level0 col2" >Cluster 0</th>
          <th id="T_20950_level0_col3" class="col_heading level0 col3" >Cluster 1</th>
          <th id="T_20950_level0_col4" class="col_heading level0 col4" >Cluster 3</th>
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
          <th id="T_20950_level0_row0" class="row_heading level0 row0" >A</th>
          <td id="T_20950_row0_col0" class="data row0 col0" >0.838000</td>
          <td id="T_20950_row0_col1" class="data row0 col1" >0.661000</td>
          <td id="T_20950_row0_col2" class="data row0 col2" >0.147000</td>
          <td id="T_20950_row0_col3" class="data row0 col3" >0.149000</td>
          <td id="T_20950_row0_col4" class="data row0 col4" >0.240000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row1" class="row_heading level0 row1" >C</th>
          <td id="T_20950_row1_col0" class="data row1 col0" >0.362000</td>
          <td id="T_20950_row1_col1" class="data row1 col1" >0.792000</td>
          <td id="T_20950_row1_col2" class="data row1 col2" >0.429000</td>
          <td id="T_20950_row1_col3" class="data row1 col3" >0.000000</td>
          <td id="T_20950_row1_col4" class="data row1 col4" >0.267000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row2" class="row_heading level0 row2" >D</th>
          <td id="T_20950_row2_col0" class="data row2 col0" >0.488000</td>
          <td id="T_20950_row2_col1" class="data row2 col1" >0.276000</td>
          <td id="T_20950_row2_col2" class="data row2 col2" >0.216000</td>
          <td id="T_20950_row2_col3" class="data row2 col3" >0.809000</td>
          <td id="T_20950_row2_col4" class="data row2 col4" >0.667000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row3" class="row_heading level0 row3" >E</th>
          <td id="T_20950_row3_col0" class="data row3 col0" >1.000000</td>
          <td id="T_20950_row3_col1" class="data row3 col1" >0.140000</td>
          <td id="T_20950_row3_col2" class="data row3 col2" >0.315000</td>
          <td id="T_20950_row3_col3" class="data row3 col3" >0.894000</td>
          <td id="T_20950_row3_col4" class="data row3 col4" >0.387000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row4" class="row_heading level0 row4" >F</th>
          <td id="T_20950_row4_col0" class="data row4 col0" >0.575000</td>
          <td id="T_20950_row4_col1" class="data row4 col1" >0.957000</td>
          <td id="T_20950_row4_col2" class="data row4 col2" >0.811000</td>
          <td id="T_20950_row4_col3" class="data row4 col3" >0.000000</td>
          <td id="T_20950_row4_col4" class="data row4 col4" >0.120000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row5" class="row_heading level0 row5" >G</th>
          <td id="T_20950_row5_col0" class="data row5 col0" >0.000000</td>
          <td id="T_20950_row5_col1" class="data row5 col1" >0.513000</td>
          <td id="T_20950_row5_col2" class="data row5 col2" >0.000000</td>
          <td id="T_20950_row5_col3" class="data row5 col3" >0.298000</td>
          <td id="T_20950_row5_col4" class="data row5 col4" >0.820000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row6" class="row_heading level0 row6" >H</th>
          <td id="T_20950_row6_col0" class="data row6 col0" >0.875000</td>
          <td id="T_20950_row6_col1" class="data row6 col1" >0.670000</td>
          <td id="T_20950_row6_col2" class="data row6 col2" >0.575000</td>
          <td id="T_20950_row6_col3" class="data row6 col3" >0.489000</td>
          <td id="T_20950_row6_col4" class="data row6 col4" >0.180000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row7" class="row_heading level0 row7" >I</th>
          <td id="T_20950_row7_col0" class="data row7 col0" >0.300000</td>
          <td id="T_20950_row7_col1" class="data row7 col1" >0.821000</td>
          <td id="T_20950_row7_col2" class="data row7 col2" >0.588000</td>
          <td id="T_20950_row7_col3" class="data row7 col3" >0.000000</td>
          <td id="T_20950_row7_col4" class="data row7 col4" >0.067000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row8" class="row_heading level0 row8" >K</th>
          <td id="T_20950_row8_col0" class="data row8 col0" >0.750000</td>
          <td id="T_20950_row8_col1" class="data row8 col1" >0.000000</td>
          <td id="T_20950_row8_col2" class="data row8 col2" >0.325000</td>
          <td id="T_20950_row8_col3" class="data row8 col3" >1.000000</td>
          <td id="T_20950_row8_col4" class="data row8 col4" >0.367000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row9" class="row_heading level0 row9" >L</th>
          <td id="T_20950_row9_col0" class="data row9 col0" >0.838000</td>
          <td id="T_20950_row9_col1" class="data row9 col1" >0.843000</td>
          <td id="T_20950_row9_col2" class="data row9 col2" >0.626000</td>
          <td id="T_20950_row9_col3" class="data row9 col3" >0.021000</td>
          <td id="T_20950_row9_col4" class="data row9 col4" >0.113000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row10" class="row_heading level0 row10" >M</th>
          <td id="T_20950_row10_col0" class="data row10 col0" >0.862000</td>
          <td id="T_20950_row10_col1" class="data row10 col1" >0.880000</td>
          <td id="T_20950_row10_col2" class="data row10 col2" >0.680000</td>
          <td id="T_20950_row10_col3" class="data row10 col3" >0.043000</td>
          <td id="T_20950_row10_col4" class="data row10 col4" >0.000000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row11" class="row_heading level0 row11" >N</th>
          <td id="T_20950_row11_col0" class="data row11 col0" >0.338000</td>
          <td id="T_20950_row11_col1" class="data row11 col1" >0.330000</td>
          <td id="T_20950_row11_col2" class="data row11 col2" >0.250000</td>
          <td id="T_20950_row11_col3" class="data row11 col3" >0.745000</td>
          <td id="T_20950_row11_col4" class="data row11 col4" >0.580000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row12" class="row_heading level0 row12" >P</th>
          <td id="T_20950_row12_col0" class="data row12 col0" >0.000000</td>
          <td id="T_20950_row12_col1" class="data row12 col1" >0.433000</td>
          <td id="T_20950_row12_col2" class="data row12 col2" >0.186000</td>
          <td id="T_20950_row12_col3" class="data row12 col3" >0.319000</td>
          <td id="T_20950_row12_col4" class="data row12 col4" >1.000000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row13" class="row_heading level0 row13" >Q</th>
          <td id="T_20950_row13_col0" class="data row13 col0" >0.512000</td>
          <td id="T_20950_row13_col1" class="data row13 col1" >0.202000</td>
          <td id="T_20950_row13_col2" class="data row13 col2" >0.348000</td>
          <td id="T_20950_row13_col3" class="data row13 col3" >0.787000</td>
          <td id="T_20950_row13_col4" class="data row13 col4" >0.380000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row14" class="row_heading level0 row14" >R</th>
          <td id="T_20950_row14_col0" class="data row14 col0" >0.375000</td>
          <td id="T_20950_row14_col1" class="data row14 col1" >0.342000</td>
          <td id="T_20950_row14_col2" class="data row14 col2" >0.614000</td>
          <td id="T_20950_row14_col3" class="data row14 col3" >0.745000</td>
          <td id="T_20950_row14_col4" class="data row14 col4" >0.313000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row15" class="row_heading level0 row15" >S</th>
          <td id="T_20950_row15_col0" class="data row15 col0" >0.188000</td>
          <td id="T_20950_row15_col1" class="data row15 col1" >0.365000</td>
          <td id="T_20950_row15_col2" class="data row15 col2" >0.140000</td>
          <td id="T_20950_row15_col3" class="data row15 col3" >0.489000</td>
          <td id="T_20950_row15_col4" class="data row15 col4" >0.607000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row16" class="row_heading level0 row16" >T</th>
          <td id="T_20950_row16_col0" class="data row16 col0" >0.212000</td>
          <td id="T_20950_row16_col1" class="data row16 col1" >0.382000</td>
          <td id="T_20950_row16_col2" class="data row16 col2" >0.270000</td>
          <td id="T_20950_row16_col3" class="data row16 col3" >0.404000</td>
          <td id="T_20950_row16_col4" class="data row16 col4" >0.420000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row17" class="row_heading level0 row17" >V</th>
          <td id="T_20950_row17_col0" class="data row17 col0" >0.400000</td>
          <td id="T_20950_row17_col1" class="data row17 col1" >0.869000</td>
          <td id="T_20950_row17_col2" class="data row17 col2" >0.483000</td>
          <td id="T_20950_row17_col3" class="data row17 col3" >0.021000</td>
          <td id="T_20950_row17_col4" class="data row17 col4" >0.040000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row18" class="row_heading level0 row18" >W</th>
          <td id="T_20950_row18_col0" class="data row18 col0" >0.500000</td>
          <td id="T_20950_row18_col1" class="data row18 col1" >1.000000</td>
          <td id="T_20950_row18_col2" class="data row18 col2" >1.000000</td>
          <td id="T_20950_row18_col3" class="data row18 col3" >0.191000</td>
          <td id="T_20950_row18_col4" class="data row18 col4" >0.233000</td>
        </tr>
        <tr>
          <th id="T_20950_level0_row19" class="row_heading level0 row19" >Y</th>
          <td id="T_20950_row19_col0" class="data row19 col0" >0.100000</td>
          <td id="T_20950_row19_col1" class="data row19 col1" >0.598000</td>
          <td id="T_20950_row19_col2" class="data row19 col2" >0.710000</td>
          <td id="T_20950_row19_col3" class="data row19 col3" >0.319000</td>
          <td id="T_20950_row19_col4" class="data row19 col4" >0.427000</td>
        </tr>
      </tbody>
    </table>



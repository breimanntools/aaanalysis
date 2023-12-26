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
    #T_cf376 thead th {
      background-color: white;
      color: black;
    }
    #T_cf376 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_cf376 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_cf376 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_cf376  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_cf376 table {
      font-size: 12px;
    }
    </style>
    <table id="T_cf376" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_cf376_level0_col0" class="col_heading level0 col0" >Cluster 0</th>
          <th id="T_cf376_level0_col1" class="col_heading level0 col1" >Cluster 3</th>
          <th id="T_cf376_level0_col2" class="col_heading level0 col2" >Cluster 4</th>
          <th id="T_cf376_level0_col3" class="col_heading level0 col3" >Cluster 1</th>
          <th id="T_cf376_level0_col4" class="col_heading level0 col4" >Cluster 2</th>
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
          <th id="T_cf376_level0_row0" class="row_heading level0 row0" >A</th>
          <td id="T_cf376_row0_col0" class="data row0 col0" >0.243000</td>
          <td id="T_cf376_row0_col1" class="data row0 col1" >0.598000</td>
          <td id="T_cf376_row0_col2" class="data row0 col2" >0.361000</td>
          <td id="T_cf376_row0_col3" class="data row0 col3" >0.467000</td>
          <td id="T_cf376_row0_col4" class="data row0 col4" >0.955000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row1" class="row_heading level0 row1" >C</th>
          <td id="T_cf376_row1_col0" class="data row1 col0" >0.360000</td>
          <td id="T_cf376_row1_col1" class="data row1 col1" >0.456000</td>
          <td id="T_cf376_row1_col2" class="data row1 col2" >0.238000</td>
          <td id="T_cf376_row1_col3" class="data row1 col3" >0.673000</td>
          <td id="T_cf376_row1_col4" class="data row1 col4" >0.099000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row2" class="row_heading level0 row2" >D</th>
          <td id="T_cf376_row2_col0" class="data row2 col0" >0.528000</td>
          <td id="T_cf376_row2_col1" class="data row2 col1" >0.483000</td>
          <td id="T_cf376_row2_col2" class="data row2 col2" >0.755000</td>
          <td id="T_cf376_row2_col3" class="data row2 col3" >0.123000</td>
          <td id="T_cf376_row2_col4" class="data row2 col4" >0.556000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row3" class="row_heading level0 row3" >E</th>
          <td id="T_cf376_row3_col0" class="data row3 col0" >0.310000</td>
          <td id="T_cf376_row3_col1" class="data row3 col1" >0.675000</td>
          <td id="T_cf376_row3_col2" class="data row3 col2" >0.765000</td>
          <td id="T_cf376_row3_col3" class="data row3 col3" >0.119000</td>
          <td id="T_cf376_row3_col4" class="data row3 col4" >0.600000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row4" class="row_heading level0 row4" >F</th>
          <td id="T_cf376_row4_col0" class="data row4 col0" >0.337000</td>
          <td id="T_cf376_row4_col1" class="data row4 col1" >0.707000</td>
          <td id="T_cf376_row4_col2" class="data row4 col2" >0.125000</td>
          <td id="T_cf376_row4_col3" class="data row4 col3" >0.799000</td>
          <td id="T_cf376_row4_col4" class="data row4 col4" >0.327000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row5" class="row_heading level0 row5" >G</th>
          <td id="T_cf376_row5_col0" class="data row5 col0" >0.607000</td>
          <td id="T_cf376_row5_col1" class="data row5 col1" >0.145000</td>
          <td id="T_cf376_row5_col2" class="data row5 col2" >0.409000</td>
          <td id="T_cf376_row5_col3" class="data row5 col3" >0.341000</td>
          <td id="T_cf376_row5_col4" class="data row5 col4" >0.864000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row6" class="row_heading level0 row6" >H</th>
          <td id="T_cf376_row6_col0" class="data row6 col0" >0.349000</td>
          <td id="T_cf376_row6_col1" class="data row6 col1" >0.664000</td>
          <td id="T_cf376_row6_col2" class="data row6 col2" >0.449000</td>
          <td id="T_cf376_row6_col3" class="data row6 col3" >0.304000</td>
          <td id="T_cf376_row6_col4" class="data row6 col4" >0.133000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row7" class="row_heading level0 row7" >I</th>
          <td id="T_cf376_row7_col0" class="data row7 col0" >0.214000</td>
          <td id="T_cf376_row7_col1" class="data row7 col1" >0.617000</td>
          <td id="T_cf376_row7_col2" class="data row7 col2" >0.192000</td>
          <td id="T_cf376_row7_col3" class="data row7 col3" >0.840000</td>
          <td id="T_cf376_row7_col4" class="data row7 col4" >0.483000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row8" class="row_heading level0 row8" >K</th>
          <td id="T_cf376_row8_col0" class="data row8 col0" >0.261000</td>
          <td id="T_cf376_row8_col1" class="data row8 col1" >0.727000</td>
          <td id="T_cf376_row8_col2" class="data row8 col2" >0.765000</td>
          <td id="T_cf376_row8_col3" class="data row8 col3" >0.135000</td>
          <td id="T_cf376_row8_col4" class="data row8 col4" >0.683000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row9" class="row_heading level0 row9" >L</th>
          <td id="T_cf376_row9_col0" class="data row9 col0" >0.141000</td>
          <td id="T_cf376_row9_col1" class="data row9 col1" >0.745000</td>
          <td id="T_cf376_row9_col2" class="data row9 col2" >0.192000</td>
          <td id="T_cf376_row9_col3" class="data row9 col3" >0.759000</td>
          <td id="T_cf376_row9_col4" class="data row9 col4" >0.867000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row10" class="row_heading level0 row10" >M</th>
          <td id="T_cf376_row10_col0" class="data row10 col0" >0.212000</td>
          <td id="T_cf376_row10_col1" class="data row10 col1" >0.791000</td>
          <td id="T_cf376_row10_col2" class="data row10 col2" >0.248000</td>
          <td id="T_cf376_row10_col3" class="data row10 col3" >0.675000</td>
          <td id="T_cf376_row10_col4" class="data row10 col4" >0.120000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row11" class="row_heading level0 row11" >N</th>
          <td id="T_cf376_row11_col0" class="data row11 col0" >0.533000</td>
          <td id="T_cf376_row11_col1" class="data row11 col1" >0.531000</td>
          <td id="T_cf376_row11_col2" class="data row11 col2" >0.576000</td>
          <td id="T_cf376_row11_col3" class="data row11 col3" >0.135000</td>
          <td id="T_cf376_row11_col4" class="data row11 col4" >0.473000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row12" class="row_heading level0 row12" >P</th>
          <td id="T_cf376_row12_col0" class="data row12 col0" >0.711000</td>
          <td id="T_cf376_row12_col1" class="data row12 col1" >0.273000</td>
          <td id="T_cf376_row12_col2" class="data row12 col2" >0.523000</td>
          <td id="T_cf376_row12_col3" class="data row12 col3" >0.234000</td>
          <td id="T_cf376_row12_col4" class="data row12 col4" >0.502000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row13" class="row_heading level0 row13" >Q</th>
          <td id="T_cf376_row13_col0" class="data row13 col0" >0.321000</td>
          <td id="T_cf376_row13_col1" class="data row13 col1" >0.673000</td>
          <td id="T_cf376_row13_col2" class="data row13 col2" >0.588000</td>
          <td id="T_cf376_row13_col3" class="data row13 col3" >0.214000</td>
          <td id="T_cf376_row13_col4" class="data row13 col4" >0.397000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row14" class="row_heading level0 row14" >R</th>
          <td id="T_cf376_row14_col0" class="data row14 col0" >0.298000</td>
          <td id="T_cf376_row14_col1" class="data row14 col1" >0.616000</td>
          <td id="T_cf376_row14_col2" class="data row14 col2" >0.744000</td>
          <td id="T_cf376_row14_col3" class="data row14 col3" >0.199000</td>
          <td id="T_cf376_row14_col4" class="data row14 col4" >0.552000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row15" class="row_heading level0 row15" >S</th>
          <td id="T_cf376_row15_col0" class="data row15 col0" >0.575000</td>
          <td id="T_cf376_row15_col1" class="data row15 col1" >0.394000</td>
          <td id="T_cf376_row15_col2" class="data row15 col2" >0.483000</td>
          <td id="T_cf376_row15_col3" class="data row15 col3" >0.266000</td>
          <td id="T_cf376_row15_col4" class="data row15 col4" >0.830000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row16" class="row_heading level0 row16" >T</th>
          <td id="T_cf376_row16_col0" class="data row16 col0" >0.468000</td>
          <td id="T_cf376_row16_col1" class="data row16 col1" >0.410000</td>
          <td id="T_cf376_row16_col2" class="data row16 col2" >0.466000</td>
          <td id="T_cf376_row16_col3" class="data row16 col3" >0.411000</td>
          <td id="T_cf376_row16_col4" class="data row16 col4" >0.597000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row17" class="row_heading level0 row17" >V</th>
          <td id="T_cf376_row17_col0" class="data row17 col0" >0.226000</td>
          <td id="T_cf376_row17_col1" class="data row17 col1" >0.535000</td>
          <td id="T_cf376_row17_col2" class="data row17 col2" >0.217000</td>
          <td id="T_cf376_row17_col3" class="data row17 col3" >0.802000</td>
          <td id="T_cf376_row17_col4" class="data row17 col4" >0.700000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row18" class="row_heading level0 row18" >W</th>
          <td id="T_cf376_row18_col0" class="data row18 col0" >0.288000</td>
          <td id="T_cf376_row18_col1" class="data row18 col1" >0.662000</td>
          <td id="T_cf376_row18_col2" class="data row18 col2" >0.214000</td>
          <td id="T_cf376_row18_col3" class="data row18 col3" >0.643000</td>
          <td id="T_cf376_row18_col4" class="data row18 col4" >0.035000</td>
        </tr>
        <tr>
          <th id="T_cf376_level0_row19" class="row_heading level0 row19" >Y</th>
          <td id="T_cf376_row19_col0" class="data row19 col0" >0.440000</td>
          <td id="T_cf376_row19_col1" class="data row19 col1" >0.608000</td>
          <td id="T_cf376_row19_col2" class="data row19 col2" >0.360000</td>
          <td id="T_cf376_row19_col3" class="data row19 col3" >0.531000</td>
          <td id="T_cf376_row19_col4" class="data row19 col4" >0.331000</td>
        </tr>
      </tbody>
    </table>



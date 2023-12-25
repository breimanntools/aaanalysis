We load a default scale dataset to showcase the ``AAclust.fit()``
method:

.. code:: ipython2

    import aaanalysis as aa
    aa.options["verbose"] = False
    # Create test dataset of 25 amino acid scales
    df_scales = aa.load_scales().T.sample(15).T
    aa.display_df(df_scales)
    X = df_scales.T



.. raw:: html

    <style type="text/css">
    #T_0b62e thead th {
      background-color: white;
      color: black;
    }
    #T_0b62e tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_0b62e tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_0b62e th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_0b62e  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_0b62e table {
      font-size: 12px;
    }
    </style>
    <table id="T_0b62e" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_0b62e_level0_col0" class="col_heading level0 col0" >WERD780102</th>
          <th id="T_0b62e_level0_col1" class="col_heading level0 col1" >NAKH900112</th>
          <th id="T_0b62e_level0_col2" class="col_heading level0 col2" >KARS160121</th>
          <th id="T_0b62e_level0_col3" class="col_heading level0 col3" >GEOR030102</th>
          <th id="T_0b62e_level0_col4" class="col_heading level0 col4" >ROBB790101</th>
          <th id="T_0b62e_level0_col5" class="col_heading level0 col5" >GEIM800110</th>
          <th id="T_0b62e_level0_col6" class="col_heading level0 col6" >QIAN880102</th>
          <th id="T_0b62e_level0_col7" class="col_heading level0 col7" >NADH010102</th>
          <th id="T_0b62e_level0_col8" class="col_heading level0 col8" >RADA880107</th>
          <th id="T_0b62e_level0_col9" class="col_heading level0 col9" >TANS770109</th>
          <th id="T_0b62e_level0_col10" class="col_heading level0 col10" >NAKH900106</th>
          <th id="T_0b62e_level0_col11" class="col_heading level0 col11" >LIFS790101</th>
          <th id="T_0b62e_level0_col12" class="col_heading level0 col12" >LINS030107</th>
          <th id="T_0b62e_level0_col13" class="col_heading level0 col13" >ROBB760101</th>
          <th id="T_0b62e_level0_col14" class="col_heading level0 col14" >GEIM800108</th>
        </tr>
        <tr>
          <th class="index_name level0" >AA</th>
          <th class="blank col0" >&nbsp;</th>
          <th class="blank col1" >&nbsp;</th>
          <th class="blank col2" >&nbsp;</th>
          <th class="blank col3" >&nbsp;</th>
          <th class="blank col4" >&nbsp;</th>
          <th class="blank col5" >&nbsp;</th>
          <th class="blank col6" >&nbsp;</th>
          <th class="blank col7" >&nbsp;</th>
          <th class="blank col8" >&nbsp;</th>
          <th class="blank col9" >&nbsp;</th>
          <th class="blank col10" >&nbsp;</th>
          <th class="blank col11" >&nbsp;</th>
          <th class="blank col12" >&nbsp;</th>
          <th class="blank col13" >&nbsp;</th>
          <th class="blank col14" >&nbsp;</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_0b62e_level0_row0" class="row_heading level0 row0" >A</th>
          <td id="T_0b62e_row0_col0" class="data row0 col0" >0.522000</td>
          <td id="T_0b62e_row0_col1" class="data row0 col1" >0.292000</td>
          <td id="T_0b62e_row0_col2" class="data row0 col2" >0.248000</td>
          <td id="T_0b62e_row0_col3" class="data row0 col3" >0.250000</td>
          <td id="T_0b62e_row0_col4" class="data row0 col4" >0.038000</td>
          <td id="T_0b62e_row0_col5" class="data row0 col5" >0.507000</td>
          <td id="T_0b62e_row0_col6" class="data row0 col6" >1.000000</td>
          <td id="T_0b62e_row0_col7" class="data row0 col7" >0.749000</td>
          <td id="T_0b62e_row0_col8" class="data row0 col8" >0.820000</td>
          <td id="T_0b62e_row0_col9" class="data row0 col9" >0.292000</td>
          <td id="T_0b62e_row0_col10" class="data row0 col10" >0.237000</td>
          <td id="T_0b62e_row0_col11" class="data row0 col11" >0.369000</td>
          <td id="T_0b62e_row0_col12" class="data row0 col12" >0.200000</td>
          <td id="T_0b62e_row0_col13" class="data row0 col13" >0.921000</td>
          <td id="T_0b62e_row0_col14" class="data row0 col14" >0.306000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row1" class="row_heading level0 row1" >C</th>
          <td id="T_0b62e_row1_col0" class="data row1 col0" >0.368000</td>
          <td id="T_0b62e_row1_col1" class="data row1 col1" >0.020000</td>
          <td id="T_0b62e_row1_col2" class="data row1 col2" >0.776000</td>
          <td id="T_0b62e_row1_col3" class="data row1 col3" >0.246000</td>
          <td id="T_0b62e_row1_col4" class="data row1 col4" >0.635000</td>
          <td id="T_0b62e_row1_col5" class="data row1 col5" >0.471000</td>
          <td id="T_0b62e_row1_col6" class="data row1 col6" >0.349000</td>
          <td id="T_0b62e_row1_col7" class="data row1 col7" >1.000000</td>
          <td id="T_0b62e_row1_col8" class="data row1 col8" >0.919000</td>
          <td id="T_0b62e_row1_col9" class="data row1 col9" >0.285000</td>
          <td id="T_0b62e_row1_col10" class="data row1 col10" >0.303000</td>
          <td id="T_0b62e_row1_col11" class="data row1 col11" >0.539000</td>
          <td id="T_0b62e_row1_col12" class="data row1 col12" >0.000000</td>
          <td id="T_0b62e_row1_col13" class="data row1 col13" >0.445000</td>
          <td id="T_0b62e_row1_col14" class="data row1 col14" >0.324000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row2" class="row_heading level0 row2" >D</th>
          <td id="T_0b62e_row2_col0" class="data row2 col0" >0.302000</td>
          <td id="T_0b62e_row2_col1" class="data row2 col1" >0.008000</td>
          <td id="T_0b62e_row2_col2" class="data row2 col2" >0.683000</td>
          <td id="T_0b62e_row2_col3" class="data row2 col3" >0.091000</td>
          <td id="T_0b62e_row2_col4" class="data row2 col4" >0.000000</td>
          <td id="T_0b62e_row2_col5" class="data row2 col5" >0.735000</td>
          <td id="T_0b62e_row2_col6" class="data row2 col6" >0.825000</td>
          <td id="T_0b62e_row2_col7" class="data row2 col7" >0.371000</td>
          <td id="T_0b62e_row2_col8" class="data row2 col8" >0.573000</td>
          <td id="T_0b62e_row2_col9" class="data row2 col9" >0.478000</td>
          <td id="T_0b62e_row2_col10" class="data row2 col10" >0.000000</td>
          <td id="T_0b62e_row2_col11" class="data row2 col11" >0.057000</td>
          <td id="T_0b62e_row2_col12" class="data row2 col12" >0.800000</td>
          <td id="T_0b62e_row2_col13" class="data row2 col13" >0.555000</td>
          <td id="T_0b62e_row2_col14" class="data row2 col14" >0.759000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row3" class="row_heading level0 row3" >E</th>
          <td id="T_0b62e_row3_col0" class="data row3 col0" >0.187000</td>
          <td id="T_0b62e_row3_col1" class="data row3 col1" >0.057000</td>
          <td id="T_0b62e_row3_col2" class="data row3 col2" >0.710000</td>
          <td id="T_0b62e_row3_col3" class="data row3 col3" >0.404000</td>
          <td id="T_0b62e_row3_col4" class="data row3 col4" >0.096000</td>
          <td id="T_0b62e_row3_col5" class="data row3 col5" >0.728000</td>
          <td id="T_0b62e_row3_col6" class="data row3 col6" >0.921000</td>
          <td id="T_0b62e_row3_col7" class="data row3 col7" >0.263000</td>
          <td id="T_0b62e_row3_col8" class="data row3 col8" >0.614000</td>
          <td id="T_0b62e_row3_col9" class="data row3 col9" >0.326000</td>
          <td id="T_0b62e_row3_col10" class="data row3 col10" >0.090000</td>
          <td id="T_0b62e_row3_col11" class="data row3 col11" >0.149000</td>
          <td id="T_0b62e_row3_col12" class="data row3 col12" >0.911000</td>
          <td id="T_0b62e_row3_col13" class="data row3 col13" >1.000000</td>
          <td id="T_0b62e_row3_col14" class="data row3 col14" >0.361000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row4" class="row_heading level0 row4" >F</th>
          <td id="T_0b62e_row4_col0" class="data row4 col0" >0.297000</td>
          <td id="T_0b62e_row4_col1" class="data row4 col1" >0.346000</td>
          <td id="T_0b62e_row4_col2" class="data row4 col2" >0.842000</td>
          <td id="T_0b62e_row4_col3" class="data row4 col3" >0.536000</td>
          <td id="T_0b62e_row4_col4" class="data row4 col4" >0.769000</td>
          <td id="T_0b62e_row4_col5" class="data row4 col5" >0.140000</td>
          <td id="T_0b62e_row4_col6" class="data row4 col6" >0.778000</td>
          <td id="T_0b62e_row4_col7" class="data row4 col7" >0.915000</td>
          <td id="T_0b62e_row4_col8" class="data row4 col8" >0.919000</td>
          <td id="T_0b62e_row4_col9" class="data row4 col9" >0.130000</td>
          <td id="T_0b62e_row4_col10" class="data row4 col10" >0.724000</td>
          <td id="T_0b62e_row4_col11" class="data row4 col11" >0.603000</td>
          <td id="T_0b62e_row4_col12" class="data row4 col12" >0.067000</td>
          <td id="T_0b62e_row4_col13" class="data row4 col13" >0.622000</td>
          <td id="T_0b62e_row4_col14" class="data row4 col14" >0.130000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row5" class="row_heading level0 row5" >G</th>
          <td id="T_0b62e_row5_col0" class="data row5 col0" >0.346000</td>
          <td id="T_0b62e_row5_col1" class="data row5 col1" >0.210000</td>
          <td id="T_0b62e_row5_col2" class="data row5 col2" >0.000000</td>
          <td id="T_0b62e_row5_col3" class="data row5 col3" >0.000000</td>
          <td id="T_0b62e_row5_col4" class="data row5 col4" >0.288000</td>
          <td id="T_0b62e_row5_col5" class="data row5 col5" >0.654000</td>
          <td id="T_0b62e_row5_col6" class="data row5 col6" >0.000000</td>
          <td id="T_0b62e_row5_col7" class="data row5 col7" >0.561000</td>
          <td id="T_0b62e_row5_col8" class="data row5 col8" >0.803000</td>
          <td id="T_0b62e_row5_col9" class="data row5 col9" >1.000000</td>
          <td id="T_0b62e_row5_col10" class="data row5 col10" >0.259000</td>
          <td id="T_0b62e_row5_col11" class="data row5 col11" >0.149000</td>
          <td id="T_0b62e_row5_col12" class="data row5 col12" >0.422000</td>
          <td id="T_0b62e_row5_col13" class="data row5 col13" >0.000000</td>
          <td id="T_0b62e_row5_col14" class="data row5 col14" >0.861000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row6" class="row_heading level0 row6" >H</th>
          <td id="T_0b62e_row6_col0" class="data row6 col0" >0.335000</td>
          <td id="T_0b62e_row6_col1" class="data row6 col1" >0.034000</td>
          <td id="T_0b62e_row6_col2" class="data row6 col2" >0.683000</td>
          <td id="T_0b62e_row6_col3" class="data row6 col3" >0.201000</td>
          <td id="T_0b62e_row6_col4" class="data row6 col4" >0.442000</td>
          <td id="T_0b62e_row6_col5" class="data row6 col5" >0.324000</td>
          <td id="T_0b62e_row6_col6" class="data row6 col6" >0.746000</td>
          <td id="T_0b62e_row6_col7" class="data row6 col7" >0.439000</td>
          <td id="T_0b62e_row6_col8" class="data row6 col8" >0.600000</td>
          <td id="T_0b62e_row6_col9" class="data row6 col9" >0.081000</td>
          <td id="T_0b62e_row6_col10" class="data row6 col10" >0.401000</td>
          <td id="T_0b62e_row6_col11" class="data row6 col11" >0.376000</td>
          <td id="T_0b62e_row6_col12" class="data row6 col12" >0.467000</td>
          <td id="T_0b62e_row6_col13" class="data row6 col13" >0.598000</td>
          <td id="T_0b62e_row6_col14" class="data row6 col14" >0.296000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row7" class="row_heading level0 row7" >I</th>
          <td id="T_0b62e_row7_col0" class="data row7 col0" >0.330000</td>
          <td id="T_0b62e_row7_col1" class="data row7 col1" >0.588000</td>
          <td id="T_0b62e_row7_col2" class="data row7 col2" >0.604000</td>
          <td id="T_0b62e_row7_col3" class="data row7 col3" >0.161000</td>
          <td id="T_0b62e_row7_col4" class="data row7 col4" >1.000000</td>
          <td id="T_0b62e_row7_col5" class="data row7 col5" >0.191000</td>
          <td id="T_0b62e_row7_col6" class="data row7 col6" >0.540000</td>
          <td id="T_0b62e_row7_col7" class="data row7 col7" >0.909000</td>
          <td id="T_0b62e_row7_col8" class="data row7 col8" >1.000000</td>
          <td id="T_0b62e_row7_col9" class="data row7 col9" >0.155000</td>
          <td id="T_0b62e_row7_col10" class="data row7 col10" >0.697000</td>
          <td id="T_0b62e_row7_col11" class="data row7 col11" >1.000000</td>
          <td id="T_0b62e_row7_col12" class="data row7 col12" >0.022000</td>
          <td id="T_0b62e_row7_col13" class="data row7 col13" >0.561000</td>
          <td id="T_0b62e_row7_col14" class="data row7 col14" >0.065000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row8" class="row_heading level0 row8" >K</th>
          <td id="T_0b62e_row8_col0" class="data row8 col0" >0.368000</td>
          <td id="T_0b62e_row8_col1" class="data row8 col1" >0.035000</td>
          <td id="T_0b62e_row8_col2" class="data row8 col2" >0.660000</td>
          <td id="T_0b62e_row8_col3" class="data row8 col3" >0.195000</td>
          <td id="T_0b62e_row8_col4" class="data row8 col4" >0.058000</td>
          <td id="T_0b62e_row8_col5" class="data row8 col5" >0.390000</td>
          <td id="T_0b62e_row8_col6" class="data row8 col6" >0.778000</td>
          <td id="T_0b62e_row8_col7" class="data row8 col7" >0.000000</td>
          <td id="T_0b62e_row8_col8" class="data row8 col8" >0.224000</td>
          <td id="T_0b62e_row8_col9" class="data row8 col9" >0.293000</td>
          <td id="T_0b62e_row8_col10" class="data row8 col10" >0.127000</td>
          <td id="T_0b62e_row8_col11" class="data row8 col11" >0.213000</td>
          <td id="T_0b62e_row8_col12" class="data row8 col12" >1.000000</td>
          <td id="T_0b62e_row8_col13" class="data row8 col13" >0.665000</td>
          <td id="T_0b62e_row8_col14" class="data row8 col14" >0.222000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row9" class="row_heading level0 row9" >L</th>
          <td id="T_0b62e_row9_col0" class="data row9 col0" >0.192000</td>
          <td id="T_0b62e_row9_col1" class="data row9 col1" >1.000000</td>
          <td id="T_0b62e_row9_col2" class="data row9 col2" >0.604000</td>
          <td id="T_0b62e_row9_col3" class="data row9 col3" >0.513000</td>
          <td id="T_0b62e_row9_col4" class="data row9 col4" >0.615000</td>
          <td id="T_0b62e_row9_col5" class="data row9 col5" >0.081000</td>
          <td id="T_0b62e_row9_col6" class="data row9 col6" >0.556000</td>
          <td id="T_0b62e_row9_col7" class="data row9 col7" >0.901000</td>
          <td id="T_0b62e_row9_col8" class="data row9 col8" >0.878000</td>
          <td id="T_0b62e_row9_col9" class="data row9 col9" >0.198000</td>
          <td id="T_0b62e_row9_col10" class="data row9 col10" >0.905000</td>
          <td id="T_0b62e_row9_col11" class="data row9 col11" >0.638000</td>
          <td id="T_0b62e_row9_col12" class="data row9 col12" >0.044000</td>
          <td id="T_0b62e_row9_col13" class="data row9 col13" >0.720000</td>
          <td id="T_0b62e_row9_col14" class="data row9 col14" >0.009000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row10" class="row_heading level0 row10" >M</th>
          <td id="T_0b62e_row10_col0" class="data row10 col0" >0.000000</td>
          <td id="T_0b62e_row10_col1" class="data row10 col1" >0.318000</td>
          <td id="T_0b62e_row10_col2" class="data row10 col2" >1.000000</td>
          <td id="T_0b62e_row10_col3" class="data row10 col3" >0.151000</td>
          <td id="T_0b62e_row10_col4" class="data row10 col4" >0.577000</td>
          <td id="T_0b62e_row10_col5" class="data row10 col5" >0.206000</td>
          <td id="T_0b62e_row10_col6" class="data row10 col6" >0.587000</td>
          <td id="T_0b62e_row10_col7" class="data row10 col7" >0.813000</td>
          <td id="T_0b62e_row10_col8" class="data row10 col8" >0.837000</td>
          <td id="T_0b62e_row10_col9" class="data row10 col9" >0.334000</td>
          <td id="T_0b62e_row10_col10" class="data row10 col10" >1.000000</td>
          <td id="T_0b62e_row10_col11" class="data row10 col11" >0.560000</td>
          <td id="T_0b62e_row10_col12" class="data row10 col12" >0.089000</td>
          <td id="T_0b62e_row10_col13" class="data row10 col13" >0.848000</td>
          <td id="T_0b62e_row10_col14" class="data row10 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row11" class="row_heading level0 row11" >N</th>
          <td id="T_0b62e_row11_col0" class="data row11 col0" >1.000000</td>
          <td id="T_0b62e_row11_col1" class="data row11 col1" >0.067000</td>
          <td id="T_0b62e_row11_col2" class="data row11 col2" >0.644000</td>
          <td id="T_0b62e_row11_col3" class="data row11 col3" >0.277000</td>
          <td id="T_0b62e_row11_col4" class="data row11 col4" >0.096000</td>
          <td id="T_0b62e_row11_col5" class="data row11 col5" >0.853000</td>
          <td id="T_0b62e_row11_col6" class="data row11 col6" >0.540000</td>
          <td id="T_0b62e_row11_col7" class="data row11 col7" >0.354000</td>
          <td id="T_0b62e_row11_col8" class="data row11 col8" >0.519000</td>
          <td id="T_0b62e_row11_col9" class="data row11 col9" >0.421000</td>
          <td id="T_0b62e_row11_col10" class="data row11 col10" >0.381000</td>
          <td id="T_0b62e_row11_col11" class="data row11 col11" >0.142000</td>
          <td id="T_0b62e_row11_col12" class="data row11 col12" >0.733000</td>
          <td id="T_0b62e_row11_col13" class="data row11 col13" >0.213000</td>
          <td id="T_0b62e_row11_col14" class="data row11 col14" >0.981000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row12" class="row_heading level0 row12" >P</th>
          <td id="T_0b62e_row12_col0" class="data row12 col0" >0.110000</td>
          <td id="T_0b62e_row12_col1" class="data row12 col1" >0.146000</td>
          <td id="T_0b62e_row12_col2" class="data row12 col2" >0.842000</td>
          <td id="T_0b62e_row12_col3" class="data row12 col3" >1.000000</td>
          <td id="T_0b62e_row12_col4" class="data row12 col4" >0.308000</td>
          <td id="T_0b62e_row12_col5" class="data row12 col5" >1.000000</td>
          <td id="T_0b62e_row12_col6" class="data row12 col6" >0.460000</td>
          <td id="T_0b62e_row12_col7" class="data row12 col7" >0.368000</td>
          <td id="T_0b62e_row12_col8" class="data row12 col8" >0.919000</td>
          <td id="T_0b62e_row12_col9" class="data row12 col9" >0.108000</td>
          <td id="T_0b62e_row12_col10" class="data row12 col10" >0.403000</td>
          <td id="T_0b62e_row12_col11" class="data row12 col11" >0.000000</td>
          <td id="T_0b62e_row12_col12" class="data row12 col12" >0.733000</td>
          <td id="T_0b62e_row12_col13" class="data row12 col13" >0.055000</td>
          <td id="T_0b62e_row12_col14" class="data row12 col14" >1.000000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row13" class="row_heading level0 row13" >Q</th>
          <td id="T_0b62e_row13_col0" class="data row13 col0" >0.132000</td>
          <td id="T_0b62e_row13_col1" class="data row13 col1" >0.037000</td>
          <td id="T_0b62e_row13_col2" class="data row13 col2" >0.693000</td>
          <td id="T_0b62e_row13_col3" class="data row13 col3" >0.478000</td>
          <td id="T_0b62e_row13_col4" class="data row13 col4" >0.212000</td>
          <td id="T_0b62e_row13_col5" class="data row13 col5" >0.294000</td>
          <td id="T_0b62e_row13_col6" class="data row13 col6" >0.381000</td>
          <td id="T_0b62e_row13_col7" class="data row13 col7" >0.225000</td>
          <td id="T_0b62e_row13_col8" class="data row13 col8" >0.400000</td>
          <td id="T_0b62e_row13_col9" class="data row13 col9" >0.171000</td>
          <td id="T_0b62e_row13_col10" class="data row13 col10" >0.203000</td>
          <td id="T_0b62e_row13_col11" class="data row13 col11" >0.390000</td>
          <td id="T_0b62e_row13_col12" class="data row13 col12" >0.778000</td>
          <td id="T_0b62e_row13_col13" class="data row13 col13" >0.585000</td>
          <td id="T_0b62e_row13_col14" class="data row13 col14" >0.333000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row14" class="row_heading level0 row14" >R</th>
          <td id="T_0b62e_row14_col0" class="data row14 col0" >0.324000</td>
          <td id="T_0b62e_row14_col1" class="data row14 col1" >0.000000</td>
          <td id="T_0b62e_row14_col2" class="data row14 col2" >0.710000</td>
          <td id="T_0b62e_row14_col3" class="data row14 col3" >0.549000</td>
          <td id="T_0b62e_row14_col4" class="data row14 col4" >0.288000</td>
          <td id="T_0b62e_row14_col5" class="data row14 col5" >0.382000</td>
          <td id="T_0b62e_row14_col6" class="data row14 col6" >0.365000</td>
          <td id="T_0b62e_row14_col7" class="data row14 col7" >0.178000</td>
          <td id="T_0b62e_row14_col8" class="data row14 col8" >0.000000</td>
          <td id="T_0b62e_row14_col9" class="data row14 col9" >0.000000</td>
          <td id="T_0b62e_row14_col10" class="data row14 col10" >0.061000</td>
          <td id="T_0b62e_row14_col11" class="data row14 col11" >0.376000</td>
          <td id="T_0b62e_row14_col12" class="data row14 col12" >0.711000</td>
          <td id="T_0b62e_row14_col13" class="data row14 col13" >0.470000</td>
          <td id="T_0b62e_row14_col14" class="data row14 col14" >0.389000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row15" class="row_heading level0 row15" >S</th>
          <td id="T_0b62e_row15_col0" class="data row15 col0" >0.429000</td>
          <td id="T_0b62e_row15_col1" class="data row15 col1" >0.303000</td>
          <td id="T_0b62e_row15_col2" class="data row15 col2" >0.512000</td>
          <td id="T_0b62e_row15_col3" class="data row15 col3" >0.168000</td>
          <td id="T_0b62e_row15_col4" class="data row15 col4" >0.000000</td>
          <td id="T_0b62e_row15_col5" class="data row15 col5" >0.529000</td>
          <td id="T_0b62e_row15_col6" class="data row15 col6" >0.603000</td>
          <td id="T_0b62e_row15_col7" class="data row15 col7" >0.523000</td>
          <td id="T_0b62e_row15_col8" class="data row15 col8" >0.664000</td>
          <td id="T_0b62e_row15_col9" class="data row15 col9" >0.390000</td>
          <td id="T_0b62e_row15_col10" class="data row15 col10" >0.450000</td>
          <td id="T_0b62e_row15_col11" class="data row15 col11" >0.298000</td>
          <td id="T_0b62e_row15_col12" class="data row15 col12" >0.556000</td>
          <td id="T_0b62e_row15_col13" class="data row15 col13" >0.287000</td>
          <td id="T_0b62e_row15_col14" class="data row15 col14" >0.602000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row16" class="row_heading level0 row16" >T</th>
          <td id="T_0b62e_row16_col0" class="data row16 col0" >0.462000</td>
          <td id="T_0b62e_row16_col1" class="data row16 col1" >0.399000</td>
          <td id="T_0b62e_row16_col2" class="data row16 col2" >0.545000</td>
          <td id="T_0b62e_row16_col3" class="data row16 col3" >0.344000</td>
          <td id="T_0b62e_row16_col4" class="data row16 col4" >0.135000</td>
          <td id="T_0b62e_row16_col5" class="data row16 col5" >0.346000</td>
          <td id="T_0b62e_row16_col6" class="data row16 col6" >0.048000</td>
          <td id="T_0b62e_row16_col7" class="data row16 col7" >0.591000</td>
          <td id="T_0b62e_row16_col8" class="data row16 col8" >0.678000</td>
          <td id="T_0b62e_row16_col9" class="data row16 col9" >0.251000</td>
          <td id="T_0b62e_row16_col10" class="data row16 col10" >0.619000</td>
          <td id="T_0b62e_row16_col11" class="data row16 col11" >0.511000</td>
          <td id="T_0b62e_row16_col12" class="data row16 col12" >0.511000</td>
          <td id="T_0b62e_row16_col13" class="data row16 col13" >0.366000</td>
          <td id="T_0b62e_row16_col14" class="data row16 col14" >0.426000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row17" class="row_heading level0 row17" >V</th>
          <td id="T_0b62e_row17_col0" class="data row17 col0" >0.181000</td>
          <td id="T_0b62e_row17_col1" class="data row17 col1" >0.277000</td>
          <td id="T_0b62e_row17_col2" class="data row17 col2" >0.545000</td>
          <td id="T_0b62e_row17_col3" class="data row17 col3" >0.151000</td>
          <td id="T_0b62e_row17_col4" class="data row17 col4" >0.500000</td>
          <td id="T_0b62e_row17_col5" class="data row17 col5" >0.125000</td>
          <td id="T_0b62e_row17_col6" class="data row17 col6" >0.619000</td>
          <td id="T_0b62e_row17_col7" class="data row17 col7" >0.915000</td>
          <td id="T_0b62e_row17_col8" class="data row17 col8" >0.949000</td>
          <td id="T_0b62e_row17_col9" class="data row17 col9" >0.099000</td>
          <td id="T_0b62e_row17_col10" class="data row17 col10" >0.183000</td>
          <td id="T_0b62e_row17_col11" class="data row17 col11" >1.000000</td>
          <td id="T_0b62e_row17_col12" class="data row17 col12" >0.044000</td>
          <td id="T_0b62e_row17_col13" class="data row17 col13" >0.610000</td>
          <td id="T_0b62e_row17_col14" class="data row17 col14" >0.019000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row18" class="row_heading level0 row18" >W</th>
          <td id="T_0b62e_row18_col0" class="data row18 col0" >0.253000</td>
          <td id="T_0b62e_row18_col1" class="data row18 col1" >0.080000</td>
          <td id="T_0b62e_row18_col2" class="data row18 col2" >0.916000</td>
          <td id="T_0b62e_row18_col3" class="data row18 col3" >0.066000</td>
          <td id="T_0b62e_row18_col4" class="data row18 col4" >0.808000</td>
          <td id="T_0b62e_row18_col5" class="data row18 col5" >0.154000</td>
          <td id="T_0b62e_row18_col6" class="data row18 col6" >0.571000</td>
          <td id="T_0b62e_row18_col7" class="data row18 col7" >0.801000</td>
          <td id="T_0b62e_row18_col8" class="data row18 col8" >0.719000</td>
          <td id="T_0b62e_row18_col9" class="data row18 col9" >0.207000</td>
          <td id="T_0b62e_row18_col10" class="data row18 col10" >0.707000</td>
          <td id="T_0b62e_row18_col11" class="data row18 col11" >0.809000</td>
          <td id="T_0b62e_row18_col12" class="data row18 col12" >0.156000</td>
          <td id="T_0b62e_row18_col13" class="data row18 col13" >0.598000</td>
          <td id="T_0b62e_row18_col14" class="data row18 col14" >0.083000</td>
        </tr>
        <tr>
          <th id="T_0b62e_level0_row19" class="row_heading level0 row19" >Y</th>
          <td id="T_0b62e_row19_col0" class="data row19 col0" >0.203000</td>
          <td id="T_0b62e_row19_col1" class="data row19 col1" >0.102000</td>
          <td id="T_0b62e_row19_col2" class="data row19 col2" >0.864000</td>
          <td id="T_0b62e_row19_col3" class="data row19 col3" >0.110000</td>
          <td id="T_0b62e_row19_col4" class="data row19 col4" >0.635000</td>
          <td id="T_0b62e_row19_col5" class="data row19 col5" >0.000000</td>
          <td id="T_0b62e_row19_col6" class="data row19 col6" >0.127000</td>
          <td id="T_0b62e_row19_col7" class="data row19 col7" >0.632000</td>
          <td id="T_0b62e_row19_col8" class="data row19 col8" >0.573000</td>
          <td id="T_0b62e_row19_col9" class="data row19 col9" >0.273000</td>
          <td id="T_0b62e_row19_col10" class="data row19 col10" >0.425000</td>
          <td id="T_0b62e_row19_col11" class="data row19 col11" >0.801000</td>
          <td id="T_0b62e_row19_col12" class="data row19 col12" >0.244000</td>
          <td id="T_0b62e_row19_col13" class="data row19 col13" >0.250000</td>
          <td id="T_0b62e_row19_col14" class="data row19 col14" >0.315000</td>
        </tr>
      </tbody>
    </table>



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

    n_clusters:  3
    Labels:  [0 1 1 1 2 0 1 2 2 0 2 2 0 1 0]
    Labels of medoids:  [0 1 2]



.. raw:: html

    <style type="text/css">
    #T_9c58c thead th {
      background-color: white;
      color: black;
    }
    #T_9c58c tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_9c58c tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_9c58c th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_9c58c  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_9c58c table {
      font-size: 12px;
    }
    </style>
    <table id="T_9c58c" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_9c58c_level0_col0" class="col_heading level0 col0" >NADH010102</th>
          <th id="T_9c58c_level0_col1" class="col_heading level0 col1" >ROBB760101</th>
          <th id="T_9c58c_level0_col2" class="col_heading level0 col2" >GEIM800108</th>
        </tr>
        <tr>
          <th class="index_name level0" >AA</th>
          <th class="blank col0" >&nbsp;</th>
          <th class="blank col1" >&nbsp;</th>
          <th class="blank col2" >&nbsp;</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_9c58c_level0_row0" class="row_heading level0 row0" >A</th>
          <td id="T_9c58c_row0_col0" class="data row0 col0" >0.749000</td>
          <td id="T_9c58c_row0_col1" class="data row0 col1" >0.921000</td>
          <td id="T_9c58c_row0_col2" class="data row0 col2" >0.306000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row1" class="row_heading level0 row1" >C</th>
          <td id="T_9c58c_row1_col0" class="data row1 col0" >1.000000</td>
          <td id="T_9c58c_row1_col1" class="data row1 col1" >0.445000</td>
          <td id="T_9c58c_row1_col2" class="data row1 col2" >0.324000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row2" class="row_heading level0 row2" >D</th>
          <td id="T_9c58c_row2_col0" class="data row2 col0" >0.371000</td>
          <td id="T_9c58c_row2_col1" class="data row2 col1" >0.555000</td>
          <td id="T_9c58c_row2_col2" class="data row2 col2" >0.759000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row3" class="row_heading level0 row3" >E</th>
          <td id="T_9c58c_row3_col0" class="data row3 col0" >0.263000</td>
          <td id="T_9c58c_row3_col1" class="data row3 col1" >1.000000</td>
          <td id="T_9c58c_row3_col2" class="data row3 col2" >0.361000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row4" class="row_heading level0 row4" >F</th>
          <td id="T_9c58c_row4_col0" class="data row4 col0" >0.915000</td>
          <td id="T_9c58c_row4_col1" class="data row4 col1" >0.622000</td>
          <td id="T_9c58c_row4_col2" class="data row4 col2" >0.130000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row5" class="row_heading level0 row5" >G</th>
          <td id="T_9c58c_row5_col0" class="data row5 col0" >0.561000</td>
          <td id="T_9c58c_row5_col1" class="data row5 col1" >0.000000</td>
          <td id="T_9c58c_row5_col2" class="data row5 col2" >0.861000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row6" class="row_heading level0 row6" >H</th>
          <td id="T_9c58c_row6_col0" class="data row6 col0" >0.439000</td>
          <td id="T_9c58c_row6_col1" class="data row6 col1" >0.598000</td>
          <td id="T_9c58c_row6_col2" class="data row6 col2" >0.296000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row7" class="row_heading level0 row7" >I</th>
          <td id="T_9c58c_row7_col0" class="data row7 col0" >0.909000</td>
          <td id="T_9c58c_row7_col1" class="data row7 col1" >0.561000</td>
          <td id="T_9c58c_row7_col2" class="data row7 col2" >0.065000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row8" class="row_heading level0 row8" >K</th>
          <td id="T_9c58c_row8_col0" class="data row8 col0" >0.000000</td>
          <td id="T_9c58c_row8_col1" class="data row8 col1" >0.665000</td>
          <td id="T_9c58c_row8_col2" class="data row8 col2" >0.222000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row9" class="row_heading level0 row9" >L</th>
          <td id="T_9c58c_row9_col0" class="data row9 col0" >0.901000</td>
          <td id="T_9c58c_row9_col1" class="data row9 col1" >0.720000</td>
          <td id="T_9c58c_row9_col2" class="data row9 col2" >0.009000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row10" class="row_heading level0 row10" >M</th>
          <td id="T_9c58c_row10_col0" class="data row10 col0" >0.813000</td>
          <td id="T_9c58c_row10_col1" class="data row10 col1" >0.848000</td>
          <td id="T_9c58c_row10_col2" class="data row10 col2" >0.000000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row11" class="row_heading level0 row11" >N</th>
          <td id="T_9c58c_row11_col0" class="data row11 col0" >0.354000</td>
          <td id="T_9c58c_row11_col1" class="data row11 col1" >0.213000</td>
          <td id="T_9c58c_row11_col2" class="data row11 col2" >0.981000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row12" class="row_heading level0 row12" >P</th>
          <td id="T_9c58c_row12_col0" class="data row12 col0" >0.368000</td>
          <td id="T_9c58c_row12_col1" class="data row12 col1" >0.055000</td>
          <td id="T_9c58c_row12_col2" class="data row12 col2" >1.000000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row13" class="row_heading level0 row13" >Q</th>
          <td id="T_9c58c_row13_col0" class="data row13 col0" >0.225000</td>
          <td id="T_9c58c_row13_col1" class="data row13 col1" >0.585000</td>
          <td id="T_9c58c_row13_col2" class="data row13 col2" >0.333000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row14" class="row_heading level0 row14" >R</th>
          <td id="T_9c58c_row14_col0" class="data row14 col0" >0.178000</td>
          <td id="T_9c58c_row14_col1" class="data row14 col1" >0.470000</td>
          <td id="T_9c58c_row14_col2" class="data row14 col2" >0.389000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row15" class="row_heading level0 row15" >S</th>
          <td id="T_9c58c_row15_col0" class="data row15 col0" >0.523000</td>
          <td id="T_9c58c_row15_col1" class="data row15 col1" >0.287000</td>
          <td id="T_9c58c_row15_col2" class="data row15 col2" >0.602000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row16" class="row_heading level0 row16" >T</th>
          <td id="T_9c58c_row16_col0" class="data row16 col0" >0.591000</td>
          <td id="T_9c58c_row16_col1" class="data row16 col1" >0.366000</td>
          <td id="T_9c58c_row16_col2" class="data row16 col2" >0.426000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row17" class="row_heading level0 row17" >V</th>
          <td id="T_9c58c_row17_col0" class="data row17 col0" >0.915000</td>
          <td id="T_9c58c_row17_col1" class="data row17 col1" >0.610000</td>
          <td id="T_9c58c_row17_col2" class="data row17 col2" >0.019000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row18" class="row_heading level0 row18" >W</th>
          <td id="T_9c58c_row18_col0" class="data row18 col0" >0.801000</td>
          <td id="T_9c58c_row18_col1" class="data row18 col1" >0.598000</td>
          <td id="T_9c58c_row18_col2" class="data row18 col2" >0.083000</td>
        </tr>
        <tr>
          <th id="T_9c58c_level0_row19" class="row_heading level0 row19" >Y</th>
          <td id="T_9c58c_row19_col0" class="data row19 col0" >0.632000</td>
          <td id="T_9c58c_row19_col1" class="data row19 col1" >0.250000</td>
          <td id="T_9c58c_row19_col2" class="data row19 col2" >0.315000</td>
        </tr>
      </tbody>
    </table>



``names`` can be provided to the ``AAclust.fit()`` method to retrieve
the names of the medoids:

.. code:: ipython2

    names = [f"scale {i+1}" for i in range(len(df_scales.T))]
    aac.fit(X, names=names)
    medoid_names = aac.medoid_names_
    print(medoid_names)


.. parsed-literal::

    ['scale 15', 'scale 11', 'scale 14', 'scale 5']


The ``n_clusters`` parameter can as well be pre-defined:

.. code:: ipython2

    aac.fit(X, n_clusters=5, names=names)
    medoid_names = aac.medoid_names_
    print(medoid_names)


.. parsed-literal::

    ['scale 10', 'scale 14', 'scale 6', 'scale 5', 'scale 8']


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

    8
    3


The third and optional merging step can be adjusted using the ``metric``
paramter and disabled setting ``merge=False``. The attributes can be
directly retrieved since the ``AAclust.fit()`` method returns the fitted
clustering model:

.. code:: ipython2

    # over 500 scales
    X = aa.load_scales().T
    n_with_merging_euclidean = aac.fit(X).n_clusters
    n_with_merging_cosine = aac.fit(X, metric="cosine").n_clusters
    n_without_merging = aac.fit(X, merge=False).n_clusters
    print(n_with_merging_euclidean)
    print(n_with_merging_cosine)
    print(n_without_merging)


.. parsed-literal::

    49
    47
    54




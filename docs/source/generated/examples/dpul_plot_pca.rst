To get insights into the identification process by ``dPULearn.fit``, you
can create a Principal Component Analysis (PCA) plot for identified
negative, positive, and unlabeled dataset groups. To this end, we load
an example dataset and perform a ``PCA-based identification`` of
negatives:

.. code:: ipython2

    import matplotlib.pyplot as plt
    import aaanalysis as aa
    aa.options["verbose"] = False
    # Dataset with positive (γ-secretase substrates)
    # and unlabeled data (proteins with unknown substrate status)
    df_seq = aa.load_dataset(name="DOM_GSEC_PU")
    labels = df_seq["label"].to_numpy()
    n_pos = sum([x == 1 for x in labels])
    df_feat = aa.load_features(name="DOM_GSEC")
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)
    # PCA-based identification of 'n_pos' negatives
    dpul = aa.dPULearn().fit(X, labels=labels, n_unl_to_neg=n_pos)
    df_pu = dpul.df_pu_
    labels = dpul.labels_
    aa.display_df(df_pu)



.. raw:: html

    <style type="text/css">
    #T_78b05 thead th {
      background-color: white;
      color: black;
    }
    #T_78b05 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_78b05 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_78b05 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_78b05  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_78b05 table {
      font-size: 12px;
    }
    </style>
    <table id="T_78b05" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_78b05_level0_col0" class="col_heading level0 col0" >selection_via</th>
          <th id="T_78b05_level0_col1" class="col_heading level0 col1" >PC1 (56.2%)</th>
          <th id="T_78b05_level0_col2" class="col_heading level0 col2" >PC2 (7.4%)</th>
          <th id="T_78b05_level0_col3" class="col_heading level0 col3" >PC3 (2.9%)</th>
          <th id="T_78b05_level0_col4" class="col_heading level0 col4" >PC4 (2.8%)</th>
          <th id="T_78b05_level0_col5" class="col_heading level0 col5" >PC5 (2.1%)</th>
          <th id="T_78b05_level0_col6" class="col_heading level0 col6" >PC6 (1.7%)</th>
          <th id="T_78b05_level0_col7" class="col_heading level0 col7" >PC7 (1.6%)</th>
          <th id="T_78b05_level0_col8" class="col_heading level0 col8" >PC1 (56.2%)_abs_dif</th>
          <th id="T_78b05_level0_col9" class="col_heading level0 col9" >PC2 (7.4%)_abs_dif</th>
          <th id="T_78b05_level0_col10" class="col_heading level0 col10" >PC3 (2.9%)_abs_dif</th>
          <th id="T_78b05_level0_col11" class="col_heading level0 col11" >PC4 (2.8%)_abs_dif</th>
          <th id="T_78b05_level0_col12" class="col_heading level0 col12" >PC5 (2.1%)_abs_dif</th>
          <th id="T_78b05_level0_col13" class="col_heading level0 col13" >PC6 (1.7%)_abs_dif</th>
          <th id="T_78b05_level0_col14" class="col_heading level0 col14" >PC7 (1.6%)_abs_dif</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_78b05_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_78b05_row0_col0" class="data row0 col0" >None</td>
          <td id="T_78b05_row0_col1" class="data row0 col1" >0.052400</td>
          <td id="T_78b05_row0_col2" class="data row0 col2" >-0.039300</td>
          <td id="T_78b05_row0_col3" class="data row0 col3" >0.066300</td>
          <td id="T_78b05_row0_col4" class="data row0 col4" >-0.020600</td>
          <td id="T_78b05_row0_col5" class="data row0 col5" >-0.002600</td>
          <td id="T_78b05_row0_col6" class="data row0 col6" >-0.022800</td>
          <td id="T_78b05_row0_col7" class="data row0 col7" >0.040900</td>
          <td id="T_78b05_row0_col8" class="data row0 col8" >0.006900</td>
          <td id="T_78b05_row0_col9" class="data row0 col9" >0.009500</td>
          <td id="T_78b05_row0_col10" class="data row0 col10" >0.035500</td>
          <td id="T_78b05_row0_col11" class="data row0 col11" >0.002700</td>
          <td id="T_78b05_row0_col12" class="data row0 col12" >0.001900</td>
          <td id="T_78b05_row0_col13" class="data row0 col13" >0.018900</td>
          <td id="T_78b05_row0_col14" class="data row0 col14" >0.042600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_78b05_row1_col0" class="data row1 col0" >None</td>
          <td id="T_78b05_row1_col1" class="data row1 col1" >0.052300</td>
          <td id="T_78b05_row1_col2" class="data row1 col2" >-0.019300</td>
          <td id="T_78b05_row1_col3" class="data row1 col3" >0.046000</td>
          <td id="T_78b05_row1_col4" class="data row1 col4" >-0.045800</td>
          <td id="T_78b05_row1_col5" class="data row1 col5" >0.015000</td>
          <td id="T_78b05_row1_col6" class="data row1 col6" >0.004000</td>
          <td id="T_78b05_row1_col7" class="data row1 col7" >0.019600</td>
          <td id="T_78b05_row1_col8" class="data row1 col8" >0.006700</td>
          <td id="T_78b05_row1_col9" class="data row1 col9" >0.010600</td>
          <td id="T_78b05_row1_col10" class="data row1 col10" >0.015300</td>
          <td id="T_78b05_row1_col11" class="data row1 col11" >0.027900</td>
          <td id="T_78b05_row1_col12" class="data row1 col12" >0.015600</td>
          <td id="T_78b05_row1_col13" class="data row1 col13" >0.007900</td>
          <td id="T_78b05_row1_col14" class="data row1 col14" >0.021300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_78b05_row2_col0" class="data row2 col0" >None</td>
          <td id="T_78b05_row2_col1" class="data row2 col1" >0.044900</td>
          <td id="T_78b05_row2_col2" class="data row2 col2" >-0.058200</td>
          <td id="T_78b05_row2_col3" class="data row2 col3" >0.000100</td>
          <td id="T_78b05_row2_col4" class="data row2 col4" >-0.086900</td>
          <td id="T_78b05_row2_col5" class="data row2 col5" >-0.012700</td>
          <td id="T_78b05_row2_col6" class="data row2 col6" >-0.007200</td>
          <td id="T_78b05_row2_col7" class="data row2 col7" >0.032200</td>
          <td id="T_78b05_row2_col8" class="data row2 col8" >0.000600</td>
          <td id="T_78b05_row2_col9" class="data row2 col9" >0.028400</td>
          <td id="T_78b05_row2_col10" class="data row2 col10" >0.030600</td>
          <td id="T_78b05_row2_col11" class="data row2 col11" >0.069000</td>
          <td id="T_78b05_row2_col12" class="data row2 col12" >0.012100</td>
          <td id="T_78b05_row2_col13" class="data row2 col13" >0.003300</td>
          <td id="T_78b05_row2_col14" class="data row2 col14" >0.033900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_78b05_row3_col0" class="data row3 col0" >None</td>
          <td id="T_78b05_row3_col1" class="data row3 col1" >0.052000</td>
          <td id="T_78b05_row3_col2" class="data row3 col2" >-0.042900</td>
          <td id="T_78b05_row3_col3" class="data row3 col3" >0.042300</td>
          <td id="T_78b05_row3_col4" class="data row3 col4" >-0.011500</td>
          <td id="T_78b05_row3_col5" class="data row3 col5" >0.002000</td>
          <td id="T_78b05_row3_col6" class="data row3 col6" >0.006500</td>
          <td id="T_78b05_row3_col7" class="data row3 col7" >0.025200</td>
          <td id="T_78b05_row3_col8" class="data row3 col8" >0.006400</td>
          <td id="T_78b05_row3_col9" class="data row3 col9" >0.013100</td>
          <td id="T_78b05_row3_col10" class="data row3 col10" >0.011500</td>
          <td id="T_78b05_row3_col11" class="data row3 col11" >0.006400</td>
          <td id="T_78b05_row3_col12" class="data row3 col12" >0.002700</td>
          <td id="T_78b05_row3_col13" class="data row3 col13" >0.010300</td>
          <td id="T_78b05_row3_col14" class="data row3 col14" >0.027000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_78b05_row4_col0" class="data row4 col0" >None</td>
          <td id="T_78b05_row4_col1" class="data row4 col1" >0.052200</td>
          <td id="T_78b05_row4_col2" class="data row4 col2" >-0.051300</td>
          <td id="T_78b05_row4_col3" class="data row4 col3" >0.061700</td>
          <td id="T_78b05_row4_col4" class="data row4 col4" >0.004100</td>
          <td id="T_78b05_row4_col5" class="data row4 col5" >-0.027500</td>
          <td id="T_78b05_row4_col6" class="data row4 col6" >0.009100</td>
          <td id="T_78b05_row4_col7" class="data row4 col7" >0.034800</td>
          <td id="T_78b05_row4_col8" class="data row4 col8" >0.006600</td>
          <td id="T_78b05_row4_col9" class="data row4 col9" >0.021500</td>
          <td id="T_78b05_row4_col10" class="data row4 col10" >0.031000</td>
          <td id="T_78b05_row4_col11" class="data row4 col11" >0.022000</td>
          <td id="T_78b05_row4_col12" class="data row4 col12" >0.026800</td>
          <td id="T_78b05_row4_col13" class="data row4 col13" >0.013000</td>
          <td id="T_78b05_row4_col14" class="data row4 col14" >0.036600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_78b05_row5_col0" class="data row5 col0" >None</td>
          <td id="T_78b05_row5_col1" class="data row5 col1" >0.049200</td>
          <td id="T_78b05_row5_col2" class="data row5 col2" >-0.045500</td>
          <td id="T_78b05_row5_col3" class="data row5 col3" >0.029800</td>
          <td id="T_78b05_row5_col4" class="data row5 col4" >0.086200</td>
          <td id="T_78b05_row5_col5" class="data row5 col5" >0.036500</td>
          <td id="T_78b05_row5_col6" class="data row5 col6" >0.028600</td>
          <td id="T_78b05_row5_col7" class="data row5 col7" >0.000800</td>
          <td id="T_78b05_row5_col8" class="data row5 col8" >0.003600</td>
          <td id="T_78b05_row5_col9" class="data row5 col9" >0.015700</td>
          <td id="T_78b05_row5_col10" class="data row5 col10" >0.001000</td>
          <td id="T_78b05_row5_col11" class="data row5 col11" >0.104100</td>
          <td id="T_78b05_row5_col12" class="data row5 col12" >0.037200</td>
          <td id="T_78b05_row5_col13" class="data row5 col13" >0.032500</td>
          <td id="T_78b05_row5_col14" class="data row5 col14" >0.002600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_78b05_row6_col0" class="data row6 col0" >None</td>
          <td id="T_78b05_row6_col1" class="data row6 col1" >0.035900</td>
          <td id="T_78b05_row6_col2" class="data row6 col2" >0.039500</td>
          <td id="T_78b05_row6_col3" class="data row6 col3" >0.034900</td>
          <td id="T_78b05_row6_col4" class="data row6 col4" >-0.014500</td>
          <td id="T_78b05_row6_col5" class="data row6 col5" >0.047800</td>
          <td id="T_78b05_row6_col6" class="data row6 col6" >-0.061600</td>
          <td id="T_78b05_row6_col7" class="data row6 col7" >-0.073300</td>
          <td id="T_78b05_row6_col8" class="data row6 col8" >0.009600</td>
          <td id="T_78b05_row6_col9" class="data row6 col9" >0.069400</td>
          <td id="T_78b05_row6_col10" class="data row6 col10" >0.004200</td>
          <td id="T_78b05_row6_col11" class="data row6 col11" >0.003400</td>
          <td id="T_78b05_row6_col12" class="data row6 col12" >0.048400</td>
          <td id="T_78b05_row6_col13" class="data row6 col13" >0.057700</td>
          <td id="T_78b05_row6_col14" class="data row6 col14" >0.071600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_78b05_row7_col0" class="data row7 col0" >None</td>
          <td id="T_78b05_row7_col1" class="data row7 col1" >0.050200</td>
          <td id="T_78b05_row7_col2" class="data row7 col2" >-0.060300</td>
          <td id="T_78b05_row7_col3" class="data row7 col3" >0.043400</td>
          <td id="T_78b05_row7_col4" class="data row7 col4" >-0.014000</td>
          <td id="T_78b05_row7_col5" class="data row7 col5" >-0.022800</td>
          <td id="T_78b05_row7_col6" class="data row7 col6" >-0.014900</td>
          <td id="T_78b05_row7_col7" class="data row7 col7" >0.034400</td>
          <td id="T_78b05_row7_col8" class="data row7 col8" >0.004700</td>
          <td id="T_78b05_row7_col9" class="data row7 col9" >0.030500</td>
          <td id="T_78b05_row7_col10" class="data row7 col10" >0.012600</td>
          <td id="T_78b05_row7_col11" class="data row7 col11" >0.003900</td>
          <td id="T_78b05_row7_col12" class="data row7 col12" >0.022100</td>
          <td id="T_78b05_row7_col13" class="data row7 col13" >0.011000</td>
          <td id="T_78b05_row7_col14" class="data row7 col14" >0.036100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_78b05_row8_col0" class="data row8 col0" >None</td>
          <td id="T_78b05_row8_col1" class="data row8 col1" >0.053600</td>
          <td id="T_78b05_row8_col2" class="data row8 col2" >-0.064700</td>
          <td id="T_78b05_row8_col3" class="data row8 col3" >0.028500</td>
          <td id="T_78b05_row8_col4" class="data row8 col4" >0.001200</td>
          <td id="T_78b05_row8_col5" class="data row8 col5" >0.049100</td>
          <td id="T_78b05_row8_col6" class="data row8 col6" >0.001500</td>
          <td id="T_78b05_row8_col7" class="data row8 col7" >-0.006700</td>
          <td id="T_78b05_row8_col8" class="data row8 col8" >0.008100</td>
          <td id="T_78b05_row8_col9" class="data row8 col9" >0.034900</td>
          <td id="T_78b05_row8_col10" class="data row8 col10" >0.002300</td>
          <td id="T_78b05_row8_col11" class="data row8 col11" >0.019100</td>
          <td id="T_78b05_row8_col12" class="data row8 col12" >0.049700</td>
          <td id="T_78b05_row8_col13" class="data row8 col13" >0.005400</td>
          <td id="T_78b05_row8_col14" class="data row8 col14" >0.005000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_78b05_row9_col0" class="data row9 col0" >None</td>
          <td id="T_78b05_row9_col1" class="data row9 col1" >0.043800</td>
          <td id="T_78b05_row9_col2" class="data row9 col2" >-0.056500</td>
          <td id="T_78b05_row9_col3" class="data row9 col3" >0.010900</td>
          <td id="T_78b05_row9_col4" class="data row9 col4" >0.015400</td>
          <td id="T_78b05_row9_col5" class="data row9 col5" >0.024700</td>
          <td id="T_78b05_row9_col6" class="data row9 col6" >-0.021900</td>
          <td id="T_78b05_row9_col7" class="data row9 col7" >-0.015800</td>
          <td id="T_78b05_row9_col8" class="data row9 col8" >0.001800</td>
          <td id="T_78b05_row9_col9" class="data row9 col9" >0.026600</td>
          <td id="T_78b05_row9_col10" class="data row9 col10" >0.019800</td>
          <td id="T_78b05_row9_col11" class="data row9 col11" >0.033300</td>
          <td id="T_78b05_row9_col12" class="data row9 col12" >0.025400</td>
          <td id="T_78b05_row9_col13" class="data row9 col13" >0.018000</td>
          <td id="T_78b05_row9_col14" class="data row9 col14" >0.014100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row10" class="row_heading level0 row10" >11</th>
          <td id="T_78b05_row10_col0" class="data row10 col0" >None</td>
          <td id="T_78b05_row10_col1" class="data row10 col1" >0.046400</td>
          <td id="T_78b05_row10_col2" class="data row10 col2" >-0.056200</td>
          <td id="T_78b05_row10_col3" class="data row10 col3" >0.068900</td>
          <td id="T_78b05_row10_col4" class="data row10 col4" >-0.025800</td>
          <td id="T_78b05_row10_col5" class="data row10 col5" >0.012000</td>
          <td id="T_78b05_row10_col6" class="data row10 col6" >0.044100</td>
          <td id="T_78b05_row10_col7" class="data row10 col7" >-0.018500</td>
          <td id="T_78b05_row10_col8" class="data row10 col8" >0.000800</td>
          <td id="T_78b05_row10_col9" class="data row10 col9" >0.026400</td>
          <td id="T_78b05_row10_col10" class="data row10 col10" >0.038200</td>
          <td id="T_78b05_row10_col11" class="data row10 col11" >0.007900</td>
          <td id="T_78b05_row10_col12" class="data row10 col12" >0.012700</td>
          <td id="T_78b05_row10_col13" class="data row10 col13" >0.048000</td>
          <td id="T_78b05_row10_col14" class="data row10 col14" >0.016800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row11" class="row_heading level0 row11" >12</th>
          <td id="T_78b05_row11_col0" class="data row11 col0" >None</td>
          <td id="T_78b05_row11_col1" class="data row11 col1" >0.043600</td>
          <td id="T_78b05_row11_col2" class="data row11 col2" >0.017300</td>
          <td id="T_78b05_row11_col3" class="data row11 col3" >0.058800</td>
          <td id="T_78b05_row11_col4" class="data row11 col4" >0.059400</td>
          <td id="T_78b05_row11_col5" class="data row11 col5" >-0.069200</td>
          <td id="T_78b05_row11_col6" class="data row11 col6" >0.019900</td>
          <td id="T_78b05_row11_col7" class="data row11 col7" >-0.024500</td>
          <td id="T_78b05_row11_col8" class="data row11 col8" >0.001900</td>
          <td id="T_78b05_row11_col9" class="data row11 col9" >0.047100</td>
          <td id="T_78b05_row11_col10" class="data row11 col10" >0.028000</td>
          <td id="T_78b05_row11_col11" class="data row11 col11" >0.077300</td>
          <td id="T_78b05_row11_col12" class="data row11 col12" >0.068500</td>
          <td id="T_78b05_row11_col13" class="data row11 col13" >0.023800</td>
          <td id="T_78b05_row11_col14" class="data row11 col14" >0.022800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row12" class="row_heading level0 row12" >13</th>
          <td id="T_78b05_row12_col0" class="data row12 col0" >None</td>
          <td id="T_78b05_row12_col1" class="data row12 col1" >0.042100</td>
          <td id="T_78b05_row12_col2" class="data row12 col2" >-0.018500</td>
          <td id="T_78b05_row12_col3" class="data row12 col3" >0.024500</td>
          <td id="T_78b05_row12_col4" class="data row12 col4" >-0.058100</td>
          <td id="T_78b05_row12_col5" class="data row12 col5" >-0.017600</td>
          <td id="T_78b05_row12_col6" class="data row12 col6" >-0.034700</td>
          <td id="T_78b05_row12_col7" class="data row12 col7" >0.004100</td>
          <td id="T_78b05_row12_col8" class="data row12 col8" >0.003500</td>
          <td id="T_78b05_row12_col9" class="data row12 col9" >0.011300</td>
          <td id="T_78b05_row12_col10" class="data row12 col10" >0.006300</td>
          <td id="T_78b05_row12_col11" class="data row12 col11" >0.040200</td>
          <td id="T_78b05_row12_col12" class="data row12 col12" >0.017000</td>
          <td id="T_78b05_row12_col13" class="data row12 col13" >0.030800</td>
          <td id="T_78b05_row12_col14" class="data row12 col14" >0.005800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row13" class="row_heading level0 row13" >14</th>
          <td id="T_78b05_row13_col0" class="data row13 col0" >None</td>
          <td id="T_78b05_row13_col1" class="data row13 col1" >0.043500</td>
          <td id="T_78b05_row13_col2" class="data row13 col2" >-0.002300</td>
          <td id="T_78b05_row13_col3" class="data row13 col3" >0.050500</td>
          <td id="T_78b05_row13_col4" class="data row13 col4" >0.010100</td>
          <td id="T_78b05_row13_col5" class="data row13 col5" >-0.015300</td>
          <td id="T_78b05_row13_col6" class="data row13 col6" >-0.050700</td>
          <td id="T_78b05_row13_col7" class="data row13 col7" >-0.053800</td>
          <td id="T_78b05_row13_col8" class="data row13 col8" >0.002000</td>
          <td id="T_78b05_row13_col9" class="data row13 col9" >0.027500</td>
          <td id="T_78b05_row13_col10" class="data row13 col10" >0.019800</td>
          <td id="T_78b05_row13_col11" class="data row13 col11" >0.028000</td>
          <td id="T_78b05_row13_col12" class="data row13 col12" >0.014700</td>
          <td id="T_78b05_row13_col13" class="data row13 col13" >0.046800</td>
          <td id="T_78b05_row13_col14" class="data row13 col14" >0.052100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row14" class="row_heading level0 row14" >15</th>
          <td id="T_78b05_row14_col0" class="data row14 col0" >None</td>
          <td id="T_78b05_row14_col1" class="data row14 col1" >0.047600</td>
          <td id="T_78b05_row14_col2" class="data row14 col2" >-0.070700</td>
          <td id="T_78b05_row14_col3" class="data row14 col3" >0.023900</td>
          <td id="T_78b05_row14_col4" class="data row14 col4" >-0.035300</td>
          <td id="T_78b05_row14_col5" class="data row14 col5" >0.015800</td>
          <td id="T_78b05_row14_col6" class="data row14 col6" >-0.001600</td>
          <td id="T_78b05_row14_col7" class="data row14 col7" >0.036200</td>
          <td id="T_78b05_row14_col8" class="data row14 col8" >0.002100</td>
          <td id="T_78b05_row14_col9" class="data row14 col9" >0.040900</td>
          <td id="T_78b05_row14_col10" class="data row14 col10" >0.006900</td>
          <td id="T_78b05_row14_col11" class="data row14 col11" >0.017400</td>
          <td id="T_78b05_row14_col12" class="data row14 col12" >0.016400</td>
          <td id="T_78b05_row14_col13" class="data row14 col13" >0.002300</td>
          <td id="T_78b05_row14_col14" class="data row14 col14" >0.037900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row15" class="row_heading level0 row15" >16</th>
          <td id="T_78b05_row15_col0" class="data row15 col0" >None</td>
          <td id="T_78b05_row15_col1" class="data row15 col1" >0.046700</td>
          <td id="T_78b05_row15_col2" class="data row15 col2" >-0.072000</td>
          <td id="T_78b05_row15_col3" class="data row15 col3" >0.034800</td>
          <td id="T_78b05_row15_col4" class="data row15 col4" >-0.007500</td>
          <td id="T_78b05_row15_col5" class="data row15 col5" >-0.035600</td>
          <td id="T_78b05_row15_col6" class="data row15 col6" >-0.026600</td>
          <td id="T_78b05_row15_col7" class="data row15 col7" >-0.001000</td>
          <td id="T_78b05_row15_col8" class="data row15 col8" >0.001200</td>
          <td id="T_78b05_row15_col9" class="data row15 col9" >0.042200</td>
          <td id="T_78b05_row15_col10" class="data row15 col10" >0.004000</td>
          <td id="T_78b05_row15_col11" class="data row15 col11" >0.010400</td>
          <td id="T_78b05_row15_col12" class="data row15 col12" >0.035000</td>
          <td id="T_78b05_row15_col13" class="data row15 col13" >0.022800</td>
          <td id="T_78b05_row15_col14" class="data row15 col14" >0.000700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row16" class="row_heading level0 row16" >17</th>
          <td id="T_78b05_row16_col0" class="data row16 col0" >None</td>
          <td id="T_78b05_row16_col1" class="data row16 col1" >0.046800</td>
          <td id="T_78b05_row16_col2" class="data row16 col2" >-0.041200</td>
          <td id="T_78b05_row16_col3" class="data row16 col3" >0.029500</td>
          <td id="T_78b05_row16_col4" class="data row16 col4" >-0.026100</td>
          <td id="T_78b05_row16_col5" class="data row16 col5" >-0.005000</td>
          <td id="T_78b05_row16_col6" class="data row16 col6" >-0.008400</td>
          <td id="T_78b05_row16_col7" class="data row16 col7" >-0.050900</td>
          <td id="T_78b05_row16_col8" class="data row16 col8" >0.001200</td>
          <td id="T_78b05_row16_col9" class="data row16 col9" >0.011300</td>
          <td id="T_78b05_row16_col10" class="data row16 col10" >0.001200</td>
          <td id="T_78b05_row16_col11" class="data row16 col11" >0.008200</td>
          <td id="T_78b05_row16_col12" class="data row16 col12" >0.004400</td>
          <td id="T_78b05_row16_col13" class="data row16 col13" >0.004500</td>
          <td id="T_78b05_row16_col14" class="data row16 col14" >0.049200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row17" class="row_heading level0 row17" >18</th>
          <td id="T_78b05_row17_col0" class="data row17 col0" >None</td>
          <td id="T_78b05_row17_col1" class="data row17 col1" >0.039000</td>
          <td id="T_78b05_row17_col2" class="data row17 col2" >0.004300</td>
          <td id="T_78b05_row17_col3" class="data row17 col3" >0.013200</td>
          <td id="T_78b05_row17_col4" class="data row17 col4" >-0.060300</td>
          <td id="T_78b05_row17_col5" class="data row17 col5" >0.003900</td>
          <td id="T_78b05_row17_col6" class="data row17 col6" >-0.056600</td>
          <td id="T_78b05_row17_col7" class="data row17 col7" >-0.037700</td>
          <td id="T_78b05_row17_col8" class="data row17 col8" >0.006600</td>
          <td id="T_78b05_row17_col9" class="data row17 col9" >0.034100</td>
          <td id="T_78b05_row17_col10" class="data row17 col10" >0.017500</td>
          <td id="T_78b05_row17_col11" class="data row17 col11" >0.042400</td>
          <td id="T_78b05_row17_col12" class="data row17 col12" >0.004500</td>
          <td id="T_78b05_row17_col13" class="data row17 col13" >0.052700</td>
          <td id="T_78b05_row17_col14" class="data row17 col14" >0.036000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row18" class="row_heading level0 row18" >19</th>
          <td id="T_78b05_row18_col0" class="data row18 col0" >None</td>
          <td id="T_78b05_row18_col1" class="data row18 col1" >0.051900</td>
          <td id="T_78b05_row18_col2" class="data row18 col2" >-0.103700</td>
          <td id="T_78b05_row18_col3" class="data row18 col3" >0.063800</td>
          <td id="T_78b05_row18_col4" class="data row18 col4" >-0.001800</td>
          <td id="T_78b05_row18_col5" class="data row18 col5" >-0.004400</td>
          <td id="T_78b05_row18_col6" class="data row18 col6" >-0.006300</td>
          <td id="T_78b05_row18_col7" class="data row18 col7" >0.011400</td>
          <td id="T_78b05_row18_col8" class="data row18 col8" >0.006400</td>
          <td id="T_78b05_row18_col9" class="data row18 col9" >0.073900</td>
          <td id="T_78b05_row18_col10" class="data row18 col10" >0.033100</td>
          <td id="T_78b05_row18_col11" class="data row18 col11" >0.016100</td>
          <td id="T_78b05_row18_col12" class="data row18 col12" >0.003800</td>
          <td id="T_78b05_row18_col13" class="data row18 col13" >0.002400</td>
          <td id="T_78b05_row18_col14" class="data row18 col14" >0.013100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row19" class="row_heading level0 row19" >20</th>
          <td id="T_78b05_row19_col0" class="data row19 col0" >None</td>
          <td id="T_78b05_row19_col1" class="data row19 col1" >0.040000</td>
          <td id="T_78b05_row19_col2" class="data row19 col2" >-0.035600</td>
          <td id="T_78b05_row19_col3" class="data row19 col3" >0.018700</td>
          <td id="T_78b05_row19_col4" class="data row19 col4" >0.015000</td>
          <td id="T_78b05_row19_col5" class="data row19 col5" >0.016400</td>
          <td id="T_78b05_row19_col6" class="data row19 col6" >-0.019300</td>
          <td id="T_78b05_row19_col7" class="data row19 col7" >-0.030300</td>
          <td id="T_78b05_row19_col8" class="data row19 col8" >0.005500</td>
          <td id="T_78b05_row19_col9" class="data row19 col9" >0.005800</td>
          <td id="T_78b05_row19_col10" class="data row19 col10" >0.012100</td>
          <td id="T_78b05_row19_col11" class="data row19 col11" >0.032900</td>
          <td id="T_78b05_row19_col12" class="data row19 col12" >0.017000</td>
          <td id="T_78b05_row19_col13" class="data row19 col13" >0.015400</td>
          <td id="T_78b05_row19_col14" class="data row19 col14" >0.028500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row20" class="row_heading level0 row20" >21</th>
          <td id="T_78b05_row20_col0" class="data row20 col0" >None</td>
          <td id="T_78b05_row20_col1" class="data row20 col1" >0.042900</td>
          <td id="T_78b05_row20_col2" class="data row20 col2" >-0.053500</td>
          <td id="T_78b05_row20_col3" class="data row20 col3" >0.006000</td>
          <td id="T_78b05_row20_col4" class="data row20 col4" >-0.002400</td>
          <td id="T_78b05_row20_col5" class="data row20 col5" >-0.034000</td>
          <td id="T_78b05_row20_col6" class="data row20 col6" >-0.020500</td>
          <td id="T_78b05_row20_col7" class="data row20 col7" >-0.016900</td>
          <td id="T_78b05_row20_col8" class="data row20 col8" >0.002600</td>
          <td id="T_78b05_row20_col9" class="data row20 col9" >0.023700</td>
          <td id="T_78b05_row20_col10" class="data row20 col10" >0.024700</td>
          <td id="T_78b05_row20_col11" class="data row20 col11" >0.015500</td>
          <td id="T_78b05_row20_col12" class="data row20 col12" >0.033300</td>
          <td id="T_78b05_row20_col13" class="data row20 col13" >0.016600</td>
          <td id="T_78b05_row20_col14" class="data row20 col14" >0.015100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row21" class="row_heading level0 row21" >22</th>
          <td id="T_78b05_row21_col0" class="data row21 col0" >None</td>
          <td id="T_78b05_row21_col1" class="data row21 col1" >0.038000</td>
          <td id="T_78b05_row21_col2" class="data row21 col2" >-0.011800</td>
          <td id="T_78b05_row21_col3" class="data row21 col3" >-0.000900</td>
          <td id="T_78b05_row21_col4" class="data row21 col4" >0.018300</td>
          <td id="T_78b05_row21_col5" class="data row21 col5" >0.053600</td>
          <td id="T_78b05_row21_col6" class="data row21 col6" >-0.039500</td>
          <td id="T_78b05_row21_col7" class="data row21 col7" >-0.007400</td>
          <td id="T_78b05_row21_col8" class="data row21 col8" >0.007500</td>
          <td id="T_78b05_row21_col9" class="data row21 col9" >0.018000</td>
          <td id="T_78b05_row21_col10" class="data row21 col10" >0.031700</td>
          <td id="T_78b05_row21_col11" class="data row21 col11" >0.036200</td>
          <td id="T_78b05_row21_col12" class="data row21 col12" >0.054200</td>
          <td id="T_78b05_row21_col13" class="data row21 col13" >0.035600</td>
          <td id="T_78b05_row21_col14" class="data row21 col14" >0.005600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row22" class="row_heading level0 row22" >23</th>
          <td id="T_78b05_row22_col0" class="data row22 col0" >None</td>
          <td id="T_78b05_row22_col1" class="data row22 col1" >0.042200</td>
          <td id="T_78b05_row22_col2" class="data row22 col2" >-0.026900</td>
          <td id="T_78b05_row22_col3" class="data row22 col3" >0.032400</td>
          <td id="T_78b05_row22_col4" class="data row22 col4" >-0.018300</td>
          <td id="T_78b05_row22_col5" class="data row22 col5" >-0.016900</td>
          <td id="T_78b05_row22_col6" class="data row22 col6" >0.014000</td>
          <td id="T_78b05_row22_col7" class="data row22 col7" >-0.083800</td>
          <td id="T_78b05_row22_col8" class="data row22 col8" >0.003400</td>
          <td id="T_78b05_row22_col9" class="data row22 col9" >0.002900</td>
          <td id="T_78b05_row22_col10" class="data row22 col10" >0.001600</td>
          <td id="T_78b05_row22_col11" class="data row22 col11" >0.000400</td>
          <td id="T_78b05_row22_col12" class="data row22 col12" >0.016300</td>
          <td id="T_78b05_row22_col13" class="data row22 col13" >0.017900</td>
          <td id="T_78b05_row22_col14" class="data row22 col14" >0.082100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row23" class="row_heading level0 row23" >24</th>
          <td id="T_78b05_row23_col0" class="data row23 col0" >None</td>
          <td id="T_78b05_row23_col1" class="data row23 col1" >0.049400</td>
          <td id="T_78b05_row23_col2" class="data row23 col2" >-0.038000</td>
          <td id="T_78b05_row23_col3" class="data row23 col3" >0.010200</td>
          <td id="T_78b05_row23_col4" class="data row23 col4" >0.021400</td>
          <td id="T_78b05_row23_col5" class="data row23 col5" >0.027100</td>
          <td id="T_78b05_row23_col6" class="data row23 col6" >0.018100</td>
          <td id="T_78b05_row23_col7" class="data row23 col7" >-0.020300</td>
          <td id="T_78b05_row23_col8" class="data row23 col8" >0.003900</td>
          <td id="T_78b05_row23_col9" class="data row23 col9" >0.008200</td>
          <td id="T_78b05_row23_col10" class="data row23 col10" >0.020500</td>
          <td id="T_78b05_row23_col11" class="data row23 col11" >0.039300</td>
          <td id="T_78b05_row23_col12" class="data row23 col12" >0.027700</td>
          <td id="T_78b05_row23_col13" class="data row23 col13" >0.022000</td>
          <td id="T_78b05_row23_col14" class="data row23 col14" >0.018500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row24" class="row_heading level0 row24" >25</th>
          <td id="T_78b05_row24_col0" class="data row24 col0" >None</td>
          <td id="T_78b05_row24_col1" class="data row24 col1" >0.042100</td>
          <td id="T_78b05_row24_col2" class="data row24 col2" >-0.055000</td>
          <td id="T_78b05_row24_col3" class="data row24 col3" >0.033400</td>
          <td id="T_78b05_row24_col4" class="data row24 col4" >-0.016500</td>
          <td id="T_78b05_row24_col5" class="data row24 col5" >-0.005700</td>
          <td id="T_78b05_row24_col6" class="data row24 col6" >0.000700</td>
          <td id="T_78b05_row24_col7" class="data row24 col7" >-0.081400</td>
          <td id="T_78b05_row24_col8" class="data row24 col8" >0.003400</td>
          <td id="T_78b05_row24_col9" class="data row24 col9" >0.025200</td>
          <td id="T_78b05_row24_col10" class="data row24 col10" >0.002600</td>
          <td id="T_78b05_row24_col11" class="data row24 col11" >0.001400</td>
          <td id="T_78b05_row24_col12" class="data row24 col12" >0.005100</td>
          <td id="T_78b05_row24_col13" class="data row24 col13" >0.004600</td>
          <td id="T_78b05_row24_col14" class="data row24 col14" >0.079600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row25" class="row_heading level0 row25" >26</th>
          <td id="T_78b05_row25_col0" class="data row25 col0" >None</td>
          <td id="T_78b05_row25_col1" class="data row25 col1" >0.043400</td>
          <td id="T_78b05_row25_col2" class="data row25 col2" >-0.061600</td>
          <td id="T_78b05_row25_col3" class="data row25 col3" >-0.003600</td>
          <td id="T_78b05_row25_col4" class="data row25 col4" >-0.051700</td>
          <td id="T_78b05_row25_col5" class="data row25 col5" >-0.025400</td>
          <td id="T_78b05_row25_col6" class="data row25 col6" >-0.001800</td>
          <td id="T_78b05_row25_col7" class="data row25 col7" >-0.050800</td>
          <td id="T_78b05_row25_col8" class="data row25 col8" >0.002100</td>
          <td id="T_78b05_row25_col9" class="data row25 col9" >0.031800</td>
          <td id="T_78b05_row25_col10" class="data row25 col10" >0.034400</td>
          <td id="T_78b05_row25_col11" class="data row25 col11" >0.033800</td>
          <td id="T_78b05_row25_col12" class="data row25 col12" >0.024700</td>
          <td id="T_78b05_row25_col13" class="data row25 col13" >0.002100</td>
          <td id="T_78b05_row25_col14" class="data row25 col14" >0.049100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row26" class="row_heading level0 row26" >27</th>
          <td id="T_78b05_row26_col0" class="data row26 col0" >None</td>
          <td id="T_78b05_row26_col1" class="data row26 col1" >0.045700</td>
          <td id="T_78b05_row26_col2" class="data row26 col2" >-0.021300</td>
          <td id="T_78b05_row26_col3" class="data row26 col3" >0.010200</td>
          <td id="T_78b05_row26_col4" class="data row26 col4" >-0.051100</td>
          <td id="T_78b05_row26_col5" class="data row26 col5" >0.003100</td>
          <td id="T_78b05_row26_col6" class="data row26 col6" >-0.017700</td>
          <td id="T_78b05_row26_col7" class="data row26 col7" >0.015000</td>
          <td id="T_78b05_row26_col8" class="data row26 col8" >0.000200</td>
          <td id="T_78b05_row26_col9" class="data row26 col9" >0.008500</td>
          <td id="T_78b05_row26_col10" class="data row26 col10" >0.020500</td>
          <td id="T_78b05_row26_col11" class="data row26 col11" >0.033200</td>
          <td id="T_78b05_row26_col12" class="data row26 col12" >0.003700</td>
          <td id="T_78b05_row26_col13" class="data row26 col13" >0.013800</td>
          <td id="T_78b05_row26_col14" class="data row26 col14" >0.016800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row27" class="row_heading level0 row27" >28</th>
          <td id="T_78b05_row27_col0" class="data row27 col0" >None</td>
          <td id="T_78b05_row27_col1" class="data row27 col1" >0.036600</td>
          <td id="T_78b05_row27_col2" class="data row27 col2" >0.050700</td>
          <td id="T_78b05_row27_col3" class="data row27 col3" >0.011700</td>
          <td id="T_78b05_row27_col4" class="data row27 col4" >0.015500</td>
          <td id="T_78b05_row27_col5" class="data row27 col5" >0.048400</td>
          <td id="T_78b05_row27_col6" class="data row27 col6" >-0.004900</td>
          <td id="T_78b05_row27_col7" class="data row27 col7" >-0.039900</td>
          <td id="T_78b05_row27_col8" class="data row27 col8" >0.008900</td>
          <td id="T_78b05_row27_col9" class="data row27 col9" >0.080500</td>
          <td id="T_78b05_row27_col10" class="data row27 col10" >0.019000</td>
          <td id="T_78b05_row27_col11" class="data row27 col11" >0.033400</td>
          <td id="T_78b05_row27_col12" class="data row27 col12" >0.049100</td>
          <td id="T_78b05_row27_col13" class="data row27 col13" >0.001000</td>
          <td id="T_78b05_row27_col14" class="data row27 col14" >0.038200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row28" class="row_heading level0 row28" >29</th>
          <td id="T_78b05_row28_col0" class="data row28 col0" >None</td>
          <td id="T_78b05_row28_col1" class="data row28 col1" >0.047500</td>
          <td id="T_78b05_row28_col2" class="data row28 col2" >-0.016200</td>
          <td id="T_78b05_row28_col3" class="data row28 col3" >0.012400</td>
          <td id="T_78b05_row28_col4" class="data row28 col4" >-0.022400</td>
          <td id="T_78b05_row28_col5" class="data row28 col5" >-0.016800</td>
          <td id="T_78b05_row28_col6" class="data row28 col6" >-0.024400</td>
          <td id="T_78b05_row28_col7" class="data row28 col7" >0.037600</td>
          <td id="T_78b05_row28_col8" class="data row28 col8" >0.002000</td>
          <td id="T_78b05_row28_col9" class="data row28 col9" >0.013600</td>
          <td id="T_78b05_row28_col10" class="data row28 col10" >0.018400</td>
          <td id="T_78b05_row28_col11" class="data row28 col11" >0.004500</td>
          <td id="T_78b05_row28_col12" class="data row28 col12" >0.016200</td>
          <td id="T_78b05_row28_col13" class="data row28 col13" >0.020500</td>
          <td id="T_78b05_row28_col14" class="data row28 col14" >0.039400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row29" class="row_heading level0 row29" >30</th>
          <td id="T_78b05_row29_col0" class="data row29 col0" >None</td>
          <td id="T_78b05_row29_col1" class="data row29 col1" >0.040200</td>
          <td id="T_78b05_row29_col2" class="data row29 col2" >-0.012200</td>
          <td id="T_78b05_row29_col3" class="data row29 col3" >0.003000</td>
          <td id="T_78b05_row29_col4" class="data row29 col4" >-0.035900</td>
          <td id="T_78b05_row29_col5" class="data row29 col5" >0.031600</td>
          <td id="T_78b05_row29_col6" class="data row29 col6" >-0.044500</td>
          <td id="T_78b05_row29_col7" class="data row29 col7" >0.028400</td>
          <td id="T_78b05_row29_col8" class="data row29 col8" >0.005300</td>
          <td id="T_78b05_row29_col9" class="data row29 col9" >0.017600</td>
          <td id="T_78b05_row29_col10" class="data row29 col10" >0.027700</td>
          <td id="T_78b05_row29_col11" class="data row29 col11" >0.017900</td>
          <td id="T_78b05_row29_col12" class="data row29 col12" >0.032300</td>
          <td id="T_78b05_row29_col13" class="data row29 col13" >0.040600</td>
          <td id="T_78b05_row29_col14" class="data row29 col14" >0.030200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row30" class="row_heading level0 row30" >31</th>
          <td id="T_78b05_row30_col0" class="data row30 col0" >None</td>
          <td id="T_78b05_row30_col1" class="data row30 col1" >0.055100</td>
          <td id="T_78b05_row30_col2" class="data row30 col2" >-0.066400</td>
          <td id="T_78b05_row30_col3" class="data row30 col3" >0.025600</td>
          <td id="T_78b05_row30_col4" class="data row30 col4" >-0.006900</td>
          <td id="T_78b05_row30_col5" class="data row30 col5" >0.061000</td>
          <td id="T_78b05_row30_col6" class="data row30 col6" >0.015200</td>
          <td id="T_78b05_row30_col7" class="data row30 col7" >0.008600</td>
          <td id="T_78b05_row30_col8" class="data row30 col8" >0.009600</td>
          <td id="T_78b05_row30_col9" class="data row30 col9" >0.036600</td>
          <td id="T_78b05_row30_col10" class="data row30 col10" >0.005200</td>
          <td id="T_78b05_row30_col11" class="data row30 col11" >0.011000</td>
          <td id="T_78b05_row30_col12" class="data row30 col12" >0.061600</td>
          <td id="T_78b05_row30_col13" class="data row30 col13" >0.019100</td>
          <td id="T_78b05_row30_col14" class="data row30 col14" >0.010300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row31" class="row_heading level0 row31" >32</th>
          <td id="T_78b05_row31_col0" class="data row31 col0" >None</td>
          <td id="T_78b05_row31_col1" class="data row31 col1" >0.047100</td>
          <td id="T_78b05_row31_col2" class="data row31 col2" >-0.033200</td>
          <td id="T_78b05_row31_col3" class="data row31 col3" >0.049500</td>
          <td id="T_78b05_row31_col4" class="data row31 col4" >0.006700</td>
          <td id="T_78b05_row31_col5" class="data row31 col5" >-0.004800</td>
          <td id="T_78b05_row31_col6" class="data row31 col6" >-0.022500</td>
          <td id="T_78b05_row31_col7" class="data row31 col7" >-0.031900</td>
          <td id="T_78b05_row31_col8" class="data row31 col8" >0.001600</td>
          <td id="T_78b05_row31_col9" class="data row31 col9" >0.003400</td>
          <td id="T_78b05_row31_col10" class="data row31 col10" >0.018700</td>
          <td id="T_78b05_row31_col11" class="data row31 col11" >0.024600</td>
          <td id="T_78b05_row31_col12" class="data row31 col12" >0.004200</td>
          <td id="T_78b05_row31_col13" class="data row31 col13" >0.018600</td>
          <td id="T_78b05_row31_col14" class="data row31 col14" >0.030100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row32" class="row_heading level0 row32" >33</th>
          <td id="T_78b05_row32_col0" class="data row32 col0" >None</td>
          <td id="T_78b05_row32_col1" class="data row32 col1" >0.050100</td>
          <td id="T_78b05_row32_col2" class="data row32 col2" >-0.042700</td>
          <td id="T_78b05_row32_col3" class="data row32 col3" >0.074800</td>
          <td id="T_78b05_row32_col4" class="data row32 col4" >-0.019800</td>
          <td id="T_78b05_row32_col5" class="data row32 col5" >0.037100</td>
          <td id="T_78b05_row32_col6" class="data row32 col6" >0.013200</td>
          <td id="T_78b05_row32_col7" class="data row32 col7" >0.001100</td>
          <td id="T_78b05_row32_col8" class="data row32 col8" >0.004600</td>
          <td id="T_78b05_row32_col9" class="data row32 col9" >0.012800</td>
          <td id="T_78b05_row32_col10" class="data row32 col10" >0.044000</td>
          <td id="T_78b05_row32_col11" class="data row32 col11" >0.001900</td>
          <td id="T_78b05_row32_col12" class="data row32 col12" >0.037800</td>
          <td id="T_78b05_row32_col13" class="data row32 col13" >0.017100</td>
          <td id="T_78b05_row32_col14" class="data row32 col14" >0.002900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row33" class="row_heading level0 row33" >34</th>
          <td id="T_78b05_row33_col0" class="data row33 col0" >None</td>
          <td id="T_78b05_row33_col1" class="data row33 col1" >0.038300</td>
          <td id="T_78b05_row33_col2" class="data row33 col2" >-0.013400</td>
          <td id="T_78b05_row33_col3" class="data row33 col3" >-0.000300</td>
          <td id="T_78b05_row33_col4" class="data row33 col4" >-0.055000</td>
          <td id="T_78b05_row33_col5" class="data row33 col5" >-0.044200</td>
          <td id="T_78b05_row33_col6" class="data row33 col6" >-0.032900</td>
          <td id="T_78b05_row33_col7" class="data row33 col7" >-0.012200</td>
          <td id="T_78b05_row33_col8" class="data row33 col8" >0.007200</td>
          <td id="T_78b05_row33_col9" class="data row33 col9" >0.016400</td>
          <td id="T_78b05_row33_col10" class="data row33 col10" >0.031100</td>
          <td id="T_78b05_row33_col11" class="data row33 col11" >0.037100</td>
          <td id="T_78b05_row33_col12" class="data row33 col12" >0.043600</td>
          <td id="T_78b05_row33_col13" class="data row33 col13" >0.029000</td>
          <td id="T_78b05_row33_col14" class="data row33 col14" >0.010500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row34" class="row_heading level0 row34" >35</th>
          <td id="T_78b05_row34_col0" class="data row34 col0" >None</td>
          <td id="T_78b05_row34_col1" class="data row34 col1" >0.034600</td>
          <td id="T_78b05_row34_col2" class="data row34 col2" >-0.014800</td>
          <td id="T_78b05_row34_col3" class="data row34 col3" >-0.001400</td>
          <td id="T_78b05_row34_col4" class="data row34 col4" >-0.067500</td>
          <td id="T_78b05_row34_col5" class="data row34 col5" >-0.068200</td>
          <td id="T_78b05_row34_col6" class="data row34 col6" >0.038400</td>
          <td id="T_78b05_row34_col7" class="data row34 col7" >0.029100</td>
          <td id="T_78b05_row34_col8" class="data row34 col8" >0.011000</td>
          <td id="T_78b05_row34_col9" class="data row34 col9" >0.015000</td>
          <td id="T_78b05_row34_col10" class="data row34 col10" >0.032200</td>
          <td id="T_78b05_row34_col11" class="data row34 col11" >0.049600</td>
          <td id="T_78b05_row34_col12" class="data row34 col12" >0.067600</td>
          <td id="T_78b05_row34_col13" class="data row34 col13" >0.042300</td>
          <td id="T_78b05_row34_col14" class="data row34 col14" >0.030800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row35" class="row_heading level0 row35" >36</th>
          <td id="T_78b05_row35_col0" class="data row35 col0" >None</td>
          <td id="T_78b05_row35_col1" class="data row35 col1" >0.045900</td>
          <td id="T_78b05_row35_col2" class="data row35 col2" >-0.036000</td>
          <td id="T_78b05_row35_col3" class="data row35 col3" >0.059100</td>
          <td id="T_78b05_row35_col4" class="data row35 col4" >-0.039700</td>
          <td id="T_78b05_row35_col5" class="data row35 col5" >0.019700</td>
          <td id="T_78b05_row35_col6" class="data row35 col6" >-0.038700</td>
          <td id="T_78b05_row35_col7" class="data row35 col7" >-0.009200</td>
          <td id="T_78b05_row35_col8" class="data row35 col8" >0.000300</td>
          <td id="T_78b05_row35_col9" class="data row35 col9" >0.006100</td>
          <td id="T_78b05_row35_col10" class="data row35 col10" >0.028400</td>
          <td id="T_78b05_row35_col11" class="data row35 col11" >0.021800</td>
          <td id="T_78b05_row35_col12" class="data row35 col12" >0.020400</td>
          <td id="T_78b05_row35_col13" class="data row35 col13" >0.034800</td>
          <td id="T_78b05_row35_col14" class="data row35 col14" >0.007500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row36" class="row_heading level0 row36" >37</th>
          <td id="T_78b05_row36_col0" class="data row36 col0" >None</td>
          <td id="T_78b05_row36_col1" class="data row36 col1" >0.047500</td>
          <td id="T_78b05_row36_col2" class="data row36 col2" >-0.066500</td>
          <td id="T_78b05_row36_col3" class="data row36 col3" >0.013600</td>
          <td id="T_78b05_row36_col4" class="data row36 col4" >-0.045700</td>
          <td id="T_78b05_row36_col5" class="data row36 col5" >-0.014400</td>
          <td id="T_78b05_row36_col6" class="data row36 col6" >-0.032500</td>
          <td id="T_78b05_row36_col7" class="data row36 col7" >0.070000</td>
          <td id="T_78b05_row36_col8" class="data row36 col8" >0.002000</td>
          <td id="T_78b05_row36_col9" class="data row36 col9" >0.036700</td>
          <td id="T_78b05_row36_col10" class="data row36 col10" >0.017200</td>
          <td id="T_78b05_row36_col11" class="data row36 col11" >0.027800</td>
          <td id="T_78b05_row36_col12" class="data row36 col12" >0.013700</td>
          <td id="T_78b05_row36_col13" class="data row36 col13" >0.028600</td>
          <td id="T_78b05_row36_col14" class="data row36 col14" >0.071700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row37" class="row_heading level0 row37" >38</th>
          <td id="T_78b05_row37_col0" class="data row37 col0" >None</td>
          <td id="T_78b05_row37_col1" class="data row37 col1" >0.041000</td>
          <td id="T_78b05_row37_col2" class="data row37 col2" >0.075600</td>
          <td id="T_78b05_row37_col3" class="data row37 col3" >0.063000</td>
          <td id="T_78b05_row37_col4" class="data row37 col4" >-0.027800</td>
          <td id="T_78b05_row37_col5" class="data row37 col5" >0.011100</td>
          <td id="T_78b05_row37_col6" class="data row37 col6" >-0.040300</td>
          <td id="T_78b05_row37_col7" class="data row37 col7" >-0.019100</td>
          <td id="T_78b05_row37_col8" class="data row37 col8" >0.004600</td>
          <td id="T_78b05_row37_col9" class="data row37 col9" >0.105400</td>
          <td id="T_78b05_row37_col10" class="data row37 col10" >0.032300</td>
          <td id="T_78b05_row37_col11" class="data row37 col11" >0.009900</td>
          <td id="T_78b05_row37_col12" class="data row37 col12" >0.011800</td>
          <td id="T_78b05_row37_col13" class="data row37 col13" >0.036500</td>
          <td id="T_78b05_row37_col14" class="data row37 col14" >0.017300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row38" class="row_heading level0 row38" >39</th>
          <td id="T_78b05_row38_col0" class="data row38 col0" >None</td>
          <td id="T_78b05_row38_col1" class="data row38 col1" >0.051200</td>
          <td id="T_78b05_row38_col2" class="data row38 col2" >-0.065300</td>
          <td id="T_78b05_row38_col3" class="data row38 col3" >0.061000</td>
          <td id="T_78b05_row38_col4" class="data row38 col4" >-0.005800</td>
          <td id="T_78b05_row38_col5" class="data row38 col5" >-0.028400</td>
          <td id="T_78b05_row38_col6" class="data row38 col6" >0.088500</td>
          <td id="T_78b05_row38_col7" class="data row38 col7" >0.013200</td>
          <td id="T_78b05_row38_col8" class="data row38 col8" >0.005700</td>
          <td id="T_78b05_row38_col9" class="data row38 col9" >0.035400</td>
          <td id="T_78b05_row38_col10" class="data row38 col10" >0.030300</td>
          <td id="T_78b05_row38_col11" class="data row38 col11" >0.012100</td>
          <td id="T_78b05_row38_col12" class="data row38 col12" >0.027800</td>
          <td id="T_78b05_row38_col13" class="data row38 col13" >0.092400</td>
          <td id="T_78b05_row38_col14" class="data row38 col14" >0.015000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row39" class="row_heading level0 row39" >40</th>
          <td id="T_78b05_row39_col0" class="data row39 col0" >None</td>
          <td id="T_78b05_row39_col1" class="data row39 col1" >0.038500</td>
          <td id="T_78b05_row39_col2" class="data row39 col2" >-0.031600</td>
          <td id="T_78b05_row39_col3" class="data row39 col3" >-0.001800</td>
          <td id="T_78b05_row39_col4" class="data row39 col4" >-0.004700</td>
          <td id="T_78b05_row39_col5" class="data row39 col5" >-0.044800</td>
          <td id="T_78b05_row39_col6" class="data row39 col6" >0.022000</td>
          <td id="T_78b05_row39_col7" class="data row39 col7" >0.070500</td>
          <td id="T_78b05_row39_col8" class="data row39 col8" >0.007000</td>
          <td id="T_78b05_row39_col9" class="data row39 col9" >0.001800</td>
          <td id="T_78b05_row39_col10" class="data row39 col10" >0.032600</td>
          <td id="T_78b05_row39_col11" class="data row39 col11" >0.013200</td>
          <td id="T_78b05_row39_col12" class="data row39 col12" >0.044200</td>
          <td id="T_78b05_row39_col13" class="data row39 col13" >0.025900</td>
          <td id="T_78b05_row39_col14" class="data row39 col14" >0.072200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row40" class="row_heading level0 row40" >41</th>
          <td id="T_78b05_row40_col0" class="data row40 col0" >None</td>
          <td id="T_78b05_row40_col1" class="data row40 col1" >0.052600</td>
          <td id="T_78b05_row40_col2" class="data row40 col2" >-0.059500</td>
          <td id="T_78b05_row40_col3" class="data row40 col3" >0.043100</td>
          <td id="T_78b05_row40_col4" class="data row40 col4" >0.012800</td>
          <td id="T_78b05_row40_col5" class="data row40 col5" >0.060700</td>
          <td id="T_78b05_row40_col6" class="data row40 col6" >0.015300</td>
          <td id="T_78b05_row40_col7" class="data row40 col7" >-0.007400</td>
          <td id="T_78b05_row40_col8" class="data row40 col8" >0.007000</td>
          <td id="T_78b05_row40_col9" class="data row40 col9" >0.029700</td>
          <td id="T_78b05_row40_col10" class="data row40 col10" >0.012400</td>
          <td id="T_78b05_row40_col11" class="data row40 col11" >0.030700</td>
          <td id="T_78b05_row40_col12" class="data row40 col12" >0.061400</td>
          <td id="T_78b05_row40_col13" class="data row40 col13" >0.019200</td>
          <td id="T_78b05_row40_col14" class="data row40 col14" >0.005600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row41" class="row_heading level0 row41" >42</th>
          <td id="T_78b05_row41_col0" class="data row41 col0" >None</td>
          <td id="T_78b05_row41_col1" class="data row41 col1" >0.051900</td>
          <td id="T_78b05_row41_col2" class="data row41 col2" >-0.064600</td>
          <td id="T_78b05_row41_col3" class="data row41 col3" >0.034800</td>
          <td id="T_78b05_row41_col4" class="data row41 col4" >-0.005000</td>
          <td id="T_78b05_row41_col5" class="data row41 col5" >0.072300</td>
          <td id="T_78b05_row41_col6" class="data row41 col6" >0.017100</td>
          <td id="T_78b05_row41_col7" class="data row41 col7" >-0.024400</td>
          <td id="T_78b05_row41_col8" class="data row41 col8" >0.006400</td>
          <td id="T_78b05_row41_col9" class="data row41 col9" >0.034800</td>
          <td id="T_78b05_row41_col10" class="data row41 col10" >0.004000</td>
          <td id="T_78b05_row41_col11" class="data row41 col11" >0.012900</td>
          <td id="T_78b05_row41_col12" class="data row41 col12" >0.072900</td>
          <td id="T_78b05_row41_col13" class="data row41 col13" >0.021000</td>
          <td id="T_78b05_row41_col14" class="data row41 col14" >0.022600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row42" class="row_heading level0 row42" >43</th>
          <td id="T_78b05_row42_col0" class="data row42 col0" >None</td>
          <td id="T_78b05_row42_col1" class="data row42 col1" >0.036400</td>
          <td id="T_78b05_row42_col2" class="data row42 col2" >0.024800</td>
          <td id="T_78b05_row42_col3" class="data row42 col3" >0.022200</td>
          <td id="T_78b05_row42_col4" class="data row42 col4" >-0.065300</td>
          <td id="T_78b05_row42_col5" class="data row42 col5" >-0.033100</td>
          <td id="T_78b05_row42_col6" class="data row42 col6" >-0.025500</td>
          <td id="T_78b05_row42_col7" class="data row42 col7" >-0.009900</td>
          <td id="T_78b05_row42_col8" class="data row42 col8" >0.009200</td>
          <td id="T_78b05_row42_col9" class="data row42 col9" >0.054600</td>
          <td id="T_78b05_row42_col10" class="data row42 col10" >0.008600</td>
          <td id="T_78b05_row42_col11" class="data row42 col11" >0.047400</td>
          <td id="T_78b05_row42_col12" class="data row42 col12" >0.032400</td>
          <td id="T_78b05_row42_col13" class="data row42 col13" >0.021600</td>
          <td id="T_78b05_row42_col14" class="data row42 col14" >0.008200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row43" class="row_heading level0 row43" >44</th>
          <td id="T_78b05_row43_col0" class="data row43 col0" >None</td>
          <td id="T_78b05_row43_col1" class="data row43 col1" >0.036200</td>
          <td id="T_78b05_row43_col2" class="data row43 col2" >0.011300</td>
          <td id="T_78b05_row43_col3" class="data row43 col3" >0.040200</td>
          <td id="T_78b05_row43_col4" class="data row43 col4" >-0.091200</td>
          <td id="T_78b05_row43_col5" class="data row43 col5" >-0.055300</td>
          <td id="T_78b05_row43_col6" class="data row43 col6" >0.000500</td>
          <td id="T_78b05_row43_col7" class="data row43 col7" >-0.035300</td>
          <td id="T_78b05_row43_col8" class="data row43 col8" >0.009400</td>
          <td id="T_78b05_row43_col9" class="data row43 col9" >0.041100</td>
          <td id="T_78b05_row43_col10" class="data row43 col10" >0.009500</td>
          <td id="T_78b05_row43_col11" class="data row43 col11" >0.073300</td>
          <td id="T_78b05_row43_col12" class="data row43 col12" >0.054700</td>
          <td id="T_78b05_row43_col13" class="data row43 col13" >0.004400</td>
          <td id="T_78b05_row43_col14" class="data row43 col14" >0.033500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row44" class="row_heading level0 row44" >45</th>
          <td id="T_78b05_row44_col0" class="data row44 col0" >None</td>
          <td id="T_78b05_row44_col1" class="data row44 col1" >0.042300</td>
          <td id="T_78b05_row44_col2" class="data row44 col2" >-0.043000</td>
          <td id="T_78b05_row44_col3" class="data row44 col3" >0.013000</td>
          <td id="T_78b05_row44_col4" class="data row44 col4" >-0.040800</td>
          <td id="T_78b05_row44_col5" class="data row44 col5" >-0.005500</td>
          <td id="T_78b05_row44_col6" class="data row44 col6" >-0.011300</td>
          <td id="T_78b05_row44_col7" class="data row44 col7" >0.014300</td>
          <td id="T_78b05_row44_col8" class="data row44 col8" >0.003300</td>
          <td id="T_78b05_row44_col9" class="data row44 col9" >0.013200</td>
          <td id="T_78b05_row44_col10" class="data row44 col10" >0.017800</td>
          <td id="T_78b05_row44_col11" class="data row44 col11" >0.022900</td>
          <td id="T_78b05_row44_col12" class="data row44 col12" >0.004800</td>
          <td id="T_78b05_row44_col13" class="data row44 col13" >0.007400</td>
          <td id="T_78b05_row44_col14" class="data row44 col14" >0.016100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row45" class="row_heading level0 row45" >46</th>
          <td id="T_78b05_row45_col0" class="data row45 col0" >None</td>
          <td id="T_78b05_row45_col1" class="data row45 col1" >0.049700</td>
          <td id="T_78b05_row45_col2" class="data row45 col2" >0.019800</td>
          <td id="T_78b05_row45_col3" class="data row45 col3" >0.033900</td>
          <td id="T_78b05_row45_col4" class="data row45 col4" >-0.046900</td>
          <td id="T_78b05_row45_col5" class="data row45 col5" >-0.022400</td>
          <td id="T_78b05_row45_col6" class="data row45 col6" >-0.020300</td>
          <td id="T_78b05_row45_col7" class="data row45 col7" >0.011000</td>
          <td id="T_78b05_row45_col8" class="data row45 col8" >0.004200</td>
          <td id="T_78b05_row45_col9" class="data row45 col9" >0.049600</td>
          <td id="T_78b05_row45_col10" class="data row45 col10" >0.003200</td>
          <td id="T_78b05_row45_col11" class="data row45 col11" >0.029000</td>
          <td id="T_78b05_row45_col12" class="data row45 col12" >0.021700</td>
          <td id="T_78b05_row45_col13" class="data row45 col13" >0.016400</td>
          <td id="T_78b05_row45_col14" class="data row45 col14" >0.012700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row46" class="row_heading level0 row46" >47</th>
          <td id="T_78b05_row46_col0" class="data row46 col0" >None</td>
          <td id="T_78b05_row46_col1" class="data row46 col1" >0.050800</td>
          <td id="T_78b05_row46_col2" class="data row46 col2" >-0.032400</td>
          <td id="T_78b05_row46_col3" class="data row46 col3" >0.073400</td>
          <td id="T_78b05_row46_col4" class="data row46 col4" >-0.039700</td>
          <td id="T_78b05_row46_col5" class="data row46 col5" >-0.019500</td>
          <td id="T_78b05_row46_col6" class="data row46 col6" >-0.041900</td>
          <td id="T_78b05_row46_col7" class="data row46 col7" >-0.008300</td>
          <td id="T_78b05_row46_col8" class="data row46 col8" >0.005300</td>
          <td id="T_78b05_row46_col9" class="data row46 col9" >0.002500</td>
          <td id="T_78b05_row46_col10" class="data row46 col10" >0.042600</td>
          <td id="T_78b05_row46_col11" class="data row46 col11" >0.021800</td>
          <td id="T_78b05_row46_col12" class="data row46 col12" >0.018800</td>
          <td id="T_78b05_row46_col13" class="data row46 col13" >0.038000</td>
          <td id="T_78b05_row46_col14" class="data row46 col14" >0.006600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row47" class="row_heading level0 row47" >48</th>
          <td id="T_78b05_row47_col0" class="data row47 col0" >None</td>
          <td id="T_78b05_row47_col1" class="data row47 col1" >0.042800</td>
          <td id="T_78b05_row47_col2" class="data row47 col2" >-0.051200</td>
          <td id="T_78b05_row47_col3" class="data row47 col3" >-0.003700</td>
          <td id="T_78b05_row47_col4" class="data row47 col4" >0.026300</td>
          <td id="T_78b05_row47_col5" class="data row47 col5" >-0.039000</td>
          <td id="T_78b05_row47_col6" class="data row47 col6" >-0.037400</td>
          <td id="T_78b05_row47_col7" class="data row47 col7" >0.025300</td>
          <td id="T_78b05_row47_col8" class="data row47 col8" >0.002800</td>
          <td id="T_78b05_row47_col9" class="data row47 col9" >0.021400</td>
          <td id="T_78b05_row47_col10" class="data row47 col10" >0.034400</td>
          <td id="T_78b05_row47_col11" class="data row47 col11" >0.044200</td>
          <td id="T_78b05_row47_col12" class="data row47 col12" >0.038400</td>
          <td id="T_78b05_row47_col13" class="data row47 col13" >0.033500</td>
          <td id="T_78b05_row47_col14" class="data row47 col14" >0.027000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row48" class="row_heading level0 row48" >49</th>
          <td id="T_78b05_row48_col0" class="data row48 col0" >None</td>
          <td id="T_78b05_row48_col1" class="data row48 col1" >0.053300</td>
          <td id="T_78b05_row48_col2" class="data row48 col2" >-0.054600</td>
          <td id="T_78b05_row48_col3" class="data row48 col3" >0.064500</td>
          <td id="T_78b05_row48_col4" class="data row48 col4" >-0.019900</td>
          <td id="T_78b05_row48_col5" class="data row48 col5" >0.035400</td>
          <td id="T_78b05_row48_col6" class="data row48 col6" >0.013600</td>
          <td id="T_78b05_row48_col7" class="data row48 col7" >0.009200</td>
          <td id="T_78b05_row48_col8" class="data row48 col8" >0.007800</td>
          <td id="T_78b05_row48_col9" class="data row48 col9" >0.024800</td>
          <td id="T_78b05_row48_col10" class="data row48 col10" >0.033700</td>
          <td id="T_78b05_row48_col11" class="data row48 col11" >0.002000</td>
          <td id="T_78b05_row48_col12" class="data row48 col12" >0.036100</td>
          <td id="T_78b05_row48_col13" class="data row48 col13" >0.017500</td>
          <td id="T_78b05_row48_col14" class="data row48 col14" >0.010900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row49" class="row_heading level0 row49" >50</th>
          <td id="T_78b05_row49_col0" class="data row49 col0" >None</td>
          <td id="T_78b05_row49_col1" class="data row49 col1" >0.054100</td>
          <td id="T_78b05_row49_col2" class="data row49 col2" >-0.057100</td>
          <td id="T_78b05_row49_col3" class="data row49 col3" >0.043300</td>
          <td id="T_78b05_row49_col4" class="data row49 col4" >0.011300</td>
          <td id="T_78b05_row49_col5" class="data row49 col5" >0.006600</td>
          <td id="T_78b05_row49_col6" class="data row49 col6" >-0.022700</td>
          <td id="T_78b05_row49_col7" class="data row49 col7" >-0.004600</td>
          <td id="T_78b05_row49_col8" class="data row49 col8" >0.008600</td>
          <td id="T_78b05_row49_col9" class="data row49 col9" >0.027200</td>
          <td id="T_78b05_row49_col10" class="data row49 col10" >0.012600</td>
          <td id="T_78b05_row49_col11" class="data row49 col11" >0.029200</td>
          <td id="T_78b05_row49_col12" class="data row49 col12" >0.007300</td>
          <td id="T_78b05_row49_col13" class="data row49 col13" >0.018800</td>
          <td id="T_78b05_row49_col14" class="data row49 col14" >0.002800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row50" class="row_heading level0 row50" >51</th>
          <td id="T_78b05_row50_col0" class="data row50 col0" >None</td>
          <td id="T_78b05_row50_col1" class="data row50 col1" >0.046400</td>
          <td id="T_78b05_row50_col2" class="data row50 col2" >0.037600</td>
          <td id="T_78b05_row50_col3" class="data row50 col3" >0.046700</td>
          <td id="T_78b05_row50_col4" class="data row50 col4" >-0.013500</td>
          <td id="T_78b05_row50_col5" class="data row50 col5" >0.060700</td>
          <td id="T_78b05_row50_col6" class="data row50 col6" >0.077600</td>
          <td id="T_78b05_row50_col7" class="data row50 col7" >0.042000</td>
          <td id="T_78b05_row50_col8" class="data row50 col8" >0.000800</td>
          <td id="T_78b05_row50_col9" class="data row50 col9" >0.067400</td>
          <td id="T_78b05_row50_col10" class="data row50 col10" >0.015900</td>
          <td id="T_78b05_row50_col11" class="data row50 col11" >0.004400</td>
          <td id="T_78b05_row50_col12" class="data row50 col12" >0.061300</td>
          <td id="T_78b05_row50_col13" class="data row50 col13" >0.081500</td>
          <td id="T_78b05_row50_col14" class="data row50 col14" >0.043700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row51" class="row_heading level0 row51" >52</th>
          <td id="T_78b05_row51_col0" class="data row51 col0" >None</td>
          <td id="T_78b05_row51_col1" class="data row51 col1" >0.047100</td>
          <td id="T_78b05_row51_col2" class="data row51 col2" >-0.032000</td>
          <td id="T_78b05_row51_col3" class="data row51 col3" >0.040100</td>
          <td id="T_78b05_row51_col4" class="data row51 col4" >0.002500</td>
          <td id="T_78b05_row51_col5" class="data row51 col5" >-0.020500</td>
          <td id="T_78b05_row51_col6" class="data row51 col6" >-0.036600</td>
          <td id="T_78b05_row51_col7" class="data row51 col7" >0.023100</td>
          <td id="T_78b05_row51_col8" class="data row51 col8" >0.001600</td>
          <td id="T_78b05_row51_col9" class="data row51 col9" >0.002200</td>
          <td id="T_78b05_row51_col10" class="data row51 col10" >0.009400</td>
          <td id="T_78b05_row51_col11" class="data row51 col11" >0.020400</td>
          <td id="T_78b05_row51_col12" class="data row51 col12" >0.019900</td>
          <td id="T_78b05_row51_col13" class="data row51 col13" >0.032700</td>
          <td id="T_78b05_row51_col14" class="data row51 col14" >0.024800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row52" class="row_heading level0 row52" >53</th>
          <td id="T_78b05_row52_col0" class="data row52 col0" >None</td>
          <td id="T_78b05_row52_col1" class="data row52 col1" >0.045500</td>
          <td id="T_78b05_row52_col2" class="data row52 col2" >-0.020200</td>
          <td id="T_78b05_row52_col3" class="data row52 col3" >-0.000300</td>
          <td id="T_78b05_row52_col4" class="data row52 col4" >-0.052800</td>
          <td id="T_78b05_row52_col5" class="data row52 col5" >-0.008100</td>
          <td id="T_78b05_row52_col6" class="data row52 col6" >-0.010800</td>
          <td id="T_78b05_row52_col7" class="data row52 col7" >0.044300</td>
          <td id="T_78b05_row52_col8" class="data row52 col8" >0.000100</td>
          <td id="T_78b05_row52_col9" class="data row52 col9" >0.009600</td>
          <td id="T_78b05_row52_col10" class="data row52 col10" >0.031100</td>
          <td id="T_78b05_row52_col11" class="data row52 col11" >0.034900</td>
          <td id="T_78b05_row52_col12" class="data row52 col12" >0.007400</td>
          <td id="T_78b05_row52_col13" class="data row52 col13" >0.006900</td>
          <td id="T_78b05_row52_col14" class="data row52 col14" >0.046100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row53" class="row_heading level0 row53" >54</th>
          <td id="T_78b05_row53_col0" class="data row53 col0" >None</td>
          <td id="T_78b05_row53_col1" class="data row53 col1" >0.039800</td>
          <td id="T_78b05_row53_col2" class="data row53 col2" >-0.012200</td>
          <td id="T_78b05_row53_col3" class="data row53 col3" >0.007900</td>
          <td id="T_78b05_row53_col4" class="data row53 col4" >-0.044500</td>
          <td id="T_78b05_row53_col5" class="data row53 col5" >-0.059200</td>
          <td id="T_78b05_row53_col6" class="data row53 col6" >-0.015000</td>
          <td id="T_78b05_row53_col7" class="data row53 col7" >0.020400</td>
          <td id="T_78b05_row53_col8" class="data row53 col8" >0.005700</td>
          <td id="T_78b05_row53_col9" class="data row53 col9" >0.017700</td>
          <td id="T_78b05_row53_col10" class="data row53 col10" >0.022900</td>
          <td id="T_78b05_row53_col11" class="data row53 col11" >0.026600</td>
          <td id="T_78b05_row53_col12" class="data row53 col12" >0.058500</td>
          <td id="T_78b05_row53_col13" class="data row53 col13" >0.011100</td>
          <td id="T_78b05_row53_col14" class="data row53 col14" >0.022100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row54" class="row_heading level0 row54" >55</th>
          <td id="T_78b05_row54_col0" class="data row54 col0" >None</td>
          <td id="T_78b05_row54_col1" class="data row54 col1" >0.048900</td>
          <td id="T_78b05_row54_col2" class="data row54 col2" >0.012000</td>
          <td id="T_78b05_row54_col3" class="data row54 col3" >0.082100</td>
          <td id="T_78b05_row54_col4" class="data row54 col4" >-0.055500</td>
          <td id="T_78b05_row54_col5" class="data row54 col5" >0.037800</td>
          <td id="T_78b05_row54_col6" class="data row54 col6" >0.020900</td>
          <td id="T_78b05_row54_col7" class="data row54 col7" >0.018800</td>
          <td id="T_78b05_row54_col8" class="data row54 col8" >0.003400</td>
          <td id="T_78b05_row54_col9" class="data row54 col9" >0.041800</td>
          <td id="T_78b05_row54_col10" class="data row54 col10" >0.051300</td>
          <td id="T_78b05_row54_col11" class="data row54 col11" >0.037600</td>
          <td id="T_78b05_row54_col12" class="data row54 col12" >0.038400</td>
          <td id="T_78b05_row54_col13" class="data row54 col13" >0.024800</td>
          <td id="T_78b05_row54_col14" class="data row54 col14" >0.020500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row55" class="row_heading level0 row55" >56</th>
          <td id="T_78b05_row55_col0" class="data row55 col0" >None</td>
          <td id="T_78b05_row55_col1" class="data row55 col1" >0.044300</td>
          <td id="T_78b05_row55_col2" class="data row55 col2" >-0.043500</td>
          <td id="T_78b05_row55_col3" class="data row55 col3" >0.030300</td>
          <td id="T_78b05_row55_col4" class="data row55 col4" >0.046000</td>
          <td id="T_78b05_row55_col5" class="data row55 col5" >-0.008400</td>
          <td id="T_78b05_row55_col6" class="data row55 col6" >0.024900</td>
          <td id="T_78b05_row55_col7" class="data row55 col7" >-0.042100</td>
          <td id="T_78b05_row55_col8" class="data row55 col8" >0.001200</td>
          <td id="T_78b05_row55_col9" class="data row55 col9" >0.013600</td>
          <td id="T_78b05_row55_col10" class="data row55 col10" >0.000500</td>
          <td id="T_78b05_row55_col11" class="data row55 col11" >0.064000</td>
          <td id="T_78b05_row55_col12" class="data row55 col12" >0.007800</td>
          <td id="T_78b05_row55_col13" class="data row55 col13" >0.028800</td>
          <td id="T_78b05_row55_col14" class="data row55 col14" >0.040400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row56" class="row_heading level0 row56" >57</th>
          <td id="T_78b05_row56_col0" class="data row56 col0" >None</td>
          <td id="T_78b05_row56_col1" class="data row56 col1" >0.045700</td>
          <td id="T_78b05_row56_col2" class="data row56 col2" >-0.010000</td>
          <td id="T_78b05_row56_col3" class="data row56 col3" >0.044300</td>
          <td id="T_78b05_row56_col4" class="data row56 col4" >-0.024500</td>
          <td id="T_78b05_row56_col5" class="data row56 col5" >-0.008800</td>
          <td id="T_78b05_row56_col6" class="data row56 col6" >-0.002300</td>
          <td id="T_78b05_row56_col7" class="data row56 col7" >0.014300</td>
          <td id="T_78b05_row56_col8" class="data row56 col8" >0.000200</td>
          <td id="T_78b05_row56_col9" class="data row56 col9" >0.019800</td>
          <td id="T_78b05_row56_col10" class="data row56 col10" >0.013600</td>
          <td id="T_78b05_row56_col11" class="data row56 col11" >0.006500</td>
          <td id="T_78b05_row56_col12" class="data row56 col12" >0.008200</td>
          <td id="T_78b05_row56_col13" class="data row56 col13" >0.001600</td>
          <td id="T_78b05_row56_col14" class="data row56 col14" >0.016000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row57" class="row_heading level0 row57" >58</th>
          <td id="T_78b05_row57_col0" class="data row57 col0" >None</td>
          <td id="T_78b05_row57_col1" class="data row57 col1" >0.037500</td>
          <td id="T_78b05_row57_col2" class="data row57 col2" >-0.024800</td>
          <td id="T_78b05_row57_col3" class="data row57 col3" >0.034500</td>
          <td id="T_78b05_row57_col4" class="data row57 col4" >-0.045400</td>
          <td id="T_78b05_row57_col5" class="data row57 col5" >-0.050800</td>
          <td id="T_78b05_row57_col6" class="data row57 col6" >0.026300</td>
          <td id="T_78b05_row57_col7" class="data row57 col7" >-0.008300</td>
          <td id="T_78b05_row57_col8" class="data row57 col8" >0.008100</td>
          <td id="T_78b05_row57_col9" class="data row57 col9" >0.005100</td>
          <td id="T_78b05_row57_col10" class="data row57 col10" >0.003700</td>
          <td id="T_78b05_row57_col11" class="data row57 col11" >0.027500</td>
          <td id="T_78b05_row57_col12" class="data row57 col12" >0.050200</td>
          <td id="T_78b05_row57_col13" class="data row57 col13" >0.030100</td>
          <td id="T_78b05_row57_col14" class="data row57 col14" >0.006600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row58" class="row_heading level0 row58" >59</th>
          <td id="T_78b05_row58_col0" class="data row58 col0" >None</td>
          <td id="T_78b05_row58_col1" class="data row58 col1" >0.048600</td>
          <td id="T_78b05_row58_col2" class="data row58 col2" >-0.014900</td>
          <td id="T_78b05_row58_col3" class="data row58 col3" >0.016300</td>
          <td id="T_78b05_row58_col4" class="data row58 col4" >0.032000</td>
          <td id="T_78b05_row58_col5" class="data row58 col5" >0.036400</td>
          <td id="T_78b05_row58_col6" class="data row58 col6" >-0.022700</td>
          <td id="T_78b05_row58_col7" class="data row58 col7" >-0.006000</td>
          <td id="T_78b05_row58_col8" class="data row58 col8" >0.003000</td>
          <td id="T_78b05_row58_col9" class="data row58 col9" >0.015000</td>
          <td id="T_78b05_row58_col10" class="data row58 col10" >0.014400</td>
          <td id="T_78b05_row58_col11" class="data row58 col11" >0.049900</td>
          <td id="T_78b05_row58_col12" class="data row58 col12" >0.037000</td>
          <td id="T_78b05_row58_col13" class="data row58 col13" >0.018800</td>
          <td id="T_78b05_row58_col14" class="data row58 col14" >0.004300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row59" class="row_heading level0 row59" >60</th>
          <td id="T_78b05_row59_col0" class="data row59 col0" >None</td>
          <td id="T_78b05_row59_col1" class="data row59 col1" >0.040100</td>
          <td id="T_78b05_row59_col2" class="data row59 col2" >-0.022600</td>
          <td id="T_78b05_row59_col3" class="data row59 col3" >0.023500</td>
          <td id="T_78b05_row59_col4" class="data row59 col4" >-0.067100</td>
          <td id="T_78b05_row59_col5" class="data row59 col5" >-0.004500</td>
          <td id="T_78b05_row59_col6" class="data row59 col6" >0.014400</td>
          <td id="T_78b05_row59_col7" class="data row59 col7" >-0.006200</td>
          <td id="T_78b05_row59_col8" class="data row59 col8" >0.005500</td>
          <td id="T_78b05_row59_col9" class="data row59 col9" >0.007200</td>
          <td id="T_78b05_row59_col10" class="data row59 col10" >0.007300</td>
          <td id="T_78b05_row59_col11" class="data row59 col11" >0.049200</td>
          <td id="T_78b05_row59_col12" class="data row59 col12" >0.003900</td>
          <td id="T_78b05_row59_col13" class="data row59 col13" >0.018300</td>
          <td id="T_78b05_row59_col14" class="data row59 col14" >0.004500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row60" class="row_heading level0 row60" >61</th>
          <td id="T_78b05_row60_col0" class="data row60 col0" >None</td>
          <td id="T_78b05_row60_col1" class="data row60 col1" >0.051200</td>
          <td id="T_78b05_row60_col2" class="data row60 col2" >-0.064700</td>
          <td id="T_78b05_row60_col3" class="data row60 col3" >0.024500</td>
          <td id="T_78b05_row60_col4" class="data row60 col4" >-0.005700</td>
          <td id="T_78b05_row60_col5" class="data row60 col5" >-0.020600</td>
          <td id="T_78b05_row60_col6" class="data row60 col6" >0.035600</td>
          <td id="T_78b05_row60_col7" class="data row60 col7" >0.024600</td>
          <td id="T_78b05_row60_col8" class="data row60 col8" >0.005600</td>
          <td id="T_78b05_row60_col9" class="data row60 col9" >0.034800</td>
          <td id="T_78b05_row60_col10" class="data row60 col10" >0.006200</td>
          <td id="T_78b05_row60_col11" class="data row60 col11" >0.012200</td>
          <td id="T_78b05_row60_col12" class="data row60 col12" >0.019900</td>
          <td id="T_78b05_row60_col13" class="data row60 col13" >0.039500</td>
          <td id="T_78b05_row60_col14" class="data row60 col14" >0.026400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row61" class="row_heading level0 row61" >62</th>
          <td id="T_78b05_row61_col0" class="data row61 col0" >None</td>
          <td id="T_78b05_row61_col1" class="data row61 col1" >0.049300</td>
          <td id="T_78b05_row61_col2" class="data row61 col2" >-0.051600</td>
          <td id="T_78b05_row61_col3" class="data row61 col3" >0.030500</td>
          <td id="T_78b05_row61_col4" class="data row61 col4" >0.026000</td>
          <td id="T_78b05_row61_col5" class="data row61 col5" >-0.033600</td>
          <td id="T_78b05_row61_col6" class="data row61 col6" >0.020100</td>
          <td id="T_78b05_row61_col7" class="data row61 col7" >-0.005400</td>
          <td id="T_78b05_row61_col8" class="data row61 col8" >0.003700</td>
          <td id="T_78b05_row61_col9" class="data row61 col9" >0.021800</td>
          <td id="T_78b05_row61_col10" class="data row61 col10" >0.000300</td>
          <td id="T_78b05_row61_col11" class="data row61 col11" >0.043900</td>
          <td id="T_78b05_row61_col12" class="data row61 col12" >0.032900</td>
          <td id="T_78b05_row61_col13" class="data row61 col13" >0.024000</td>
          <td id="T_78b05_row61_col14" class="data row61 col14" >0.003600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row62" class="row_heading level0 row62" >63</th>
          <td id="T_78b05_row62_col0" class="data row62 col0" >None</td>
          <td id="T_78b05_row62_col1" class="data row62 col1" >0.043000</td>
          <td id="T_78b05_row62_col2" class="data row62 col2" >-0.008600</td>
          <td id="T_78b05_row62_col3" class="data row62 col3" >-0.016100</td>
          <td id="T_78b05_row62_col4" class="data row62 col4" >-0.041400</td>
          <td id="T_78b05_row62_col5" class="data row62 col5" >0.039300</td>
          <td id="T_78b05_row62_col6" class="data row62 col6" >0.064100</td>
          <td id="T_78b05_row62_col7" class="data row62 col7" >-0.027000</td>
          <td id="T_78b05_row62_col8" class="data row62 col8" >0.002500</td>
          <td id="T_78b05_row62_col9" class="data row62 col9" >0.021200</td>
          <td id="T_78b05_row62_col10" class="data row62 col10" >0.046800</td>
          <td id="T_78b05_row62_col11" class="data row62 col11" >0.023500</td>
          <td id="T_78b05_row62_col12" class="data row62 col12" >0.040000</td>
          <td id="T_78b05_row62_col13" class="data row62 col13" >0.068000</td>
          <td id="T_78b05_row62_col14" class="data row62 col14" >0.025300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row63" class="row_heading level0 row63" >64</th>
          <td id="T_78b05_row63_col0" class="data row63 col0" >None</td>
          <td id="T_78b05_row63_col1" class="data row63 col1" >0.037500</td>
          <td id="T_78b05_row63_col2" class="data row63 col2" >-0.004200</td>
          <td id="T_78b05_row63_col3" class="data row63 col3" >-0.011800</td>
          <td id="T_78b05_row63_col4" class="data row63 col4" >-0.012500</td>
          <td id="T_78b05_row63_col5" class="data row63 col5" >0.031400</td>
          <td id="T_78b05_row63_col6" class="data row63 col6" >0.025400</td>
          <td id="T_78b05_row63_col7" class="data row63 col7" >0.010000</td>
          <td id="T_78b05_row63_col8" class="data row63 col8" >0.008000</td>
          <td id="T_78b05_row63_col9" class="data row63 col9" >0.025600</td>
          <td id="T_78b05_row63_col10" class="data row63 col10" >0.042500</td>
          <td id="T_78b05_row63_col11" class="data row63 col11" >0.005500</td>
          <td id="T_78b05_row63_col12" class="data row63 col12" >0.032100</td>
          <td id="T_78b05_row63_col13" class="data row63 col13" >0.029300</td>
          <td id="T_78b05_row63_col14" class="data row63 col14" >0.011800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row64" class="row_heading level0 row64" >65</th>
          <td id="T_78b05_row64_col0" class="data row64 col0" >None</td>
          <td id="T_78b05_row64_col1" class="data row64 col1" >0.033200</td>
          <td id="T_78b05_row64_col2" class="data row64 col2" >0.025000</td>
          <td id="T_78b05_row64_col3" class="data row64 col3" >0.022700</td>
          <td id="T_78b05_row64_col4" class="data row64 col4" >0.043700</td>
          <td id="T_78b05_row64_col5" class="data row64 col5" >-0.048500</td>
          <td id="T_78b05_row64_col6" class="data row64 col6" >0.009900</td>
          <td id="T_78b05_row64_col7" class="data row64 col7" >0.024500</td>
          <td id="T_78b05_row64_col8" class="data row64 col8" >0.012300</td>
          <td id="T_78b05_row64_col9" class="data row64 col9" >0.054900</td>
          <td id="T_78b05_row64_col10" class="data row64 col10" >0.008100</td>
          <td id="T_78b05_row64_col11" class="data row64 col11" >0.061600</td>
          <td id="T_78b05_row64_col12" class="data row64 col12" >0.047900</td>
          <td id="T_78b05_row64_col13" class="data row64 col13" >0.013800</td>
          <td id="T_78b05_row64_col14" class="data row64 col14" >0.026200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row65" class="row_heading level0 row65" >66</th>
          <td id="T_78b05_row65_col0" class="data row65 col0" >None</td>
          <td id="T_78b05_row65_col1" class="data row65 col1" >0.033100</td>
          <td id="T_78b05_row65_col2" class="data row65 col2" >-0.004200</td>
          <td id="T_78b05_row65_col3" class="data row65 col3" >-0.007400</td>
          <td id="T_78b05_row65_col4" class="data row65 col4" >0.025800</td>
          <td id="T_78b05_row65_col5" class="data row65 col5" >0.001800</td>
          <td id="T_78b05_row65_col6" class="data row65 col6" >-0.041300</td>
          <td id="T_78b05_row65_col7" class="data row65 col7" >-0.020300</td>
          <td id="T_78b05_row65_col8" class="data row65 col8" >0.012400</td>
          <td id="T_78b05_row65_col9" class="data row65 col9" >0.025700</td>
          <td id="T_78b05_row65_col10" class="data row65 col10" >0.038100</td>
          <td id="T_78b05_row65_col11" class="data row65 col11" >0.043700</td>
          <td id="T_78b05_row65_col12" class="data row65 col12" >0.002400</td>
          <td id="T_78b05_row65_col13" class="data row65 col13" >0.037400</td>
          <td id="T_78b05_row65_col14" class="data row65 col14" >0.018600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row66" class="row_heading level0 row66" >67</th>
          <td id="T_78b05_row66_col0" class="data row66 col0" >None</td>
          <td id="T_78b05_row66_col1" class="data row66 col1" >0.044500</td>
          <td id="T_78b05_row66_col2" class="data row66 col2" >-0.029500</td>
          <td id="T_78b05_row66_col3" class="data row66 col3" >0.008800</td>
          <td id="T_78b05_row66_col4" class="data row66 col4" >-0.035700</td>
          <td id="T_78b05_row66_col5" class="data row66 col5" >0.005200</td>
          <td id="T_78b05_row66_col6" class="data row66 col6" >-0.010900</td>
          <td id="T_78b05_row66_col7" class="data row66 col7" >-0.060400</td>
          <td id="T_78b05_row66_col8" class="data row66 col8" >0.001000</td>
          <td id="T_78b05_row66_col9" class="data row66 col9" >0.000400</td>
          <td id="T_78b05_row66_col10" class="data row66 col10" >0.022000</td>
          <td id="T_78b05_row66_col11" class="data row66 col11" >0.017700</td>
          <td id="T_78b05_row66_col12" class="data row66 col12" >0.005900</td>
          <td id="T_78b05_row66_col13" class="data row66 col13" >0.007000</td>
          <td id="T_78b05_row66_col14" class="data row66 col14" >0.058700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row67" class="row_heading level0 row67" >68</th>
          <td id="T_78b05_row67_col0" class="data row67 col0" >None</td>
          <td id="T_78b05_row67_col1" class="data row67 col1" >0.045300</td>
          <td id="T_78b05_row67_col2" class="data row67 col2" >-0.041400</td>
          <td id="T_78b05_row67_col3" class="data row67 col3" >0.025800</td>
          <td id="T_78b05_row67_col4" class="data row67 col4" >0.010600</td>
          <td id="T_78b05_row67_col5" class="data row67 col5" >-0.006900</td>
          <td id="T_78b05_row67_col6" class="data row67 col6" >-0.022200</td>
          <td id="T_78b05_row67_col7" class="data row67 col7" >0.014900</td>
          <td id="T_78b05_row67_col8" class="data row67 col8" >0.000300</td>
          <td id="T_78b05_row67_col9" class="data row67 col9" >0.011600</td>
          <td id="T_78b05_row67_col10" class="data row67 col10" >0.005000</td>
          <td id="T_78b05_row67_col11" class="data row67 col11" >0.028500</td>
          <td id="T_78b05_row67_col12" class="data row67 col12" >0.006200</td>
          <td id="T_78b05_row67_col13" class="data row67 col13" >0.018300</td>
          <td id="T_78b05_row67_col14" class="data row67 col14" >0.016700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row68" class="row_heading level0 row68" >69</th>
          <td id="T_78b05_row68_col0" class="data row68 col0" >None</td>
          <td id="T_78b05_row68_col1" class="data row68 col1" >0.035500</td>
          <td id="T_78b05_row68_col2" class="data row68 col2" >-0.033800</td>
          <td id="T_78b05_row68_col3" class="data row68 col3" >-0.036600</td>
          <td id="T_78b05_row68_col4" class="data row68 col4" >-0.018900</td>
          <td id="T_78b05_row68_col5" class="data row68 col5" >-0.025100</td>
          <td id="T_78b05_row68_col6" class="data row68 col6" >-0.020400</td>
          <td id="T_78b05_row68_col7" class="data row68 col7" >-0.038700</td>
          <td id="T_78b05_row68_col8" class="data row68 col8" >0.010000</td>
          <td id="T_78b05_row68_col9" class="data row68 col9" >0.003900</td>
          <td id="T_78b05_row68_col10" class="data row68 col10" >0.067400</td>
          <td id="T_78b05_row68_col11" class="data row68 col11" >0.000900</td>
          <td id="T_78b05_row68_col12" class="data row68 col12" >0.024500</td>
          <td id="T_78b05_row68_col13" class="data row68 col13" >0.016500</td>
          <td id="T_78b05_row68_col14" class="data row68 col14" >0.037000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row69" class="row_heading level0 row69" >70</th>
          <td id="T_78b05_row69_col0" class="data row69 col0" >None</td>
          <td id="T_78b05_row69_col1" class="data row69 col1" >0.033600</td>
          <td id="T_78b05_row69_col2" class="data row69 col2" >-0.015200</td>
          <td id="T_78b05_row69_col3" class="data row69 col3" >-0.028900</td>
          <td id="T_78b05_row69_col4" class="data row69 col4" >0.057100</td>
          <td id="T_78b05_row69_col5" class="data row69 col5" >-0.056800</td>
          <td id="T_78b05_row69_col6" class="data row69 col6" >-0.054700</td>
          <td id="T_78b05_row69_col7" class="data row69 col7" >0.022700</td>
          <td id="T_78b05_row69_col8" class="data row69 col8" >0.012000</td>
          <td id="T_78b05_row69_col9" class="data row69 col9" >0.014600</td>
          <td id="T_78b05_row69_col10" class="data row69 col10" >0.059600</td>
          <td id="T_78b05_row69_col11" class="data row69 col11" >0.075000</td>
          <td id="T_78b05_row69_col12" class="data row69 col12" >0.056200</td>
          <td id="T_78b05_row69_col13" class="data row69 col13" >0.050800</td>
          <td id="T_78b05_row69_col14" class="data row69 col14" >0.024500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row70" class="row_heading level0 row70" >71</th>
          <td id="T_78b05_row70_col0" class="data row70 col0" >None</td>
          <td id="T_78b05_row70_col1" class="data row70 col1" >0.036100</td>
          <td id="T_78b05_row70_col2" class="data row70 col2" >-0.016900</td>
          <td id="T_78b05_row70_col3" class="data row70 col3" >-0.069500</td>
          <td id="T_78b05_row70_col4" class="data row70 col4" >0.011300</td>
          <td id="T_78b05_row70_col5" class="data row70 col5" >0.074900</td>
          <td id="T_78b05_row70_col6" class="data row70 col6" >0.007700</td>
          <td id="T_78b05_row70_col7" class="data row70 col7" >-0.016100</td>
          <td id="T_78b05_row70_col8" class="data row70 col8" >0.009500</td>
          <td id="T_78b05_row70_col9" class="data row70 col9" >0.012900</td>
          <td id="T_78b05_row70_col10" class="data row70 col10" >0.100200</td>
          <td id="T_78b05_row70_col11" class="data row70 col11" >0.029200</td>
          <td id="T_78b05_row70_col12" class="data row70 col12" >0.075500</td>
          <td id="T_78b05_row70_col13" class="data row70 col13" >0.011600</td>
          <td id="T_78b05_row70_col14" class="data row70 col14" >0.014400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row71" class="row_heading level0 row71" >72</th>
          <td id="T_78b05_row71_col0" class="data row71 col0" >None</td>
          <td id="T_78b05_row71_col1" class="data row71 col1" >0.042800</td>
          <td id="T_78b05_row71_col2" class="data row71 col2" >-0.047100</td>
          <td id="T_78b05_row71_col3" class="data row71 col3" >0.002300</td>
          <td id="T_78b05_row71_col4" class="data row71 col4" >0.000800</td>
          <td id="T_78b05_row71_col5" class="data row71 col5" >0.045600</td>
          <td id="T_78b05_row71_col6" class="data row71 col6" >-0.020100</td>
          <td id="T_78b05_row71_col7" class="data row71 col7" >-0.004300</td>
          <td id="T_78b05_row71_col8" class="data row71 col8" >0.002700</td>
          <td id="T_78b05_row71_col9" class="data row71 col9" >0.017200</td>
          <td id="T_78b05_row71_col10" class="data row71 col10" >0.028400</td>
          <td id="T_78b05_row71_col11" class="data row71 col11" >0.018700</td>
          <td id="T_78b05_row71_col12" class="data row71 col12" >0.046300</td>
          <td id="T_78b05_row71_col13" class="data row71 col13" >0.016300</td>
          <td id="T_78b05_row71_col14" class="data row71 col14" >0.002500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row72" class="row_heading level0 row72" >73</th>
          <td id="T_78b05_row72_col0" class="data row72 col0" >None</td>
          <td id="T_78b05_row72_col1" class="data row72 col1" >0.040300</td>
          <td id="T_78b05_row72_col2" class="data row72 col2" >-0.018700</td>
          <td id="T_78b05_row72_col3" class="data row72 col3" >0.058100</td>
          <td id="T_78b05_row72_col4" class="data row72 col4" >0.026400</td>
          <td id="T_78b05_row72_col5" class="data row72 col5" >-0.061100</td>
          <td id="T_78b05_row72_col6" class="data row72 col6" >-0.081400</td>
          <td id="T_78b05_row72_col7" class="data row72 col7" >-0.022500</td>
          <td id="T_78b05_row72_col8" class="data row72 col8" >0.005300</td>
          <td id="T_78b05_row72_col9" class="data row72 col9" >0.011100</td>
          <td id="T_78b05_row72_col10" class="data row72 col10" >0.027300</td>
          <td id="T_78b05_row72_col11" class="data row72 col11" >0.044300</td>
          <td id="T_78b05_row72_col12" class="data row72 col12" >0.060400</td>
          <td id="T_78b05_row72_col13" class="data row72 col13" >0.077600</td>
          <td id="T_78b05_row72_col14" class="data row72 col14" >0.020700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row73" class="row_heading level0 row73" >74</th>
          <td id="T_78b05_row73_col0" class="data row73 col0" >None</td>
          <td id="T_78b05_row73_col1" class="data row73 col1" >0.035600</td>
          <td id="T_78b05_row73_col2" class="data row73 col2" >-0.055300</td>
          <td id="T_78b05_row73_col3" class="data row73 col3" >-0.041600</td>
          <td id="T_78b05_row73_col4" class="data row73 col4" >0.026100</td>
          <td id="T_78b05_row73_col5" class="data row73 col5" >-0.062700</td>
          <td id="T_78b05_row73_col6" class="data row73 col6" >-0.042000</td>
          <td id="T_78b05_row73_col7" class="data row73 col7" >0.053100</td>
          <td id="T_78b05_row73_col8" class="data row73 col8" >0.009900</td>
          <td id="T_78b05_row73_col9" class="data row73 col9" >0.025500</td>
          <td id="T_78b05_row73_col10" class="data row73 col10" >0.072400</td>
          <td id="T_78b05_row73_col11" class="data row73 col11" >0.044000</td>
          <td id="T_78b05_row73_col12" class="data row73 col12" >0.062000</td>
          <td id="T_78b05_row73_col13" class="data row73 col13" >0.038100</td>
          <td id="T_78b05_row73_col14" class="data row73 col14" >0.054800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row74" class="row_heading level0 row74" >75</th>
          <td id="T_78b05_row74_col0" class="data row74 col0" >None</td>
          <td id="T_78b05_row74_col1" class="data row74 col1" >0.039300</td>
          <td id="T_78b05_row74_col2" class="data row74 col2" >0.030500</td>
          <td id="T_78b05_row74_col3" class="data row74 col3" >0.040600</td>
          <td id="T_78b05_row74_col4" class="data row74 col4" >-0.009300</td>
          <td id="T_78b05_row74_col5" class="data row74 col5" >-0.037400</td>
          <td id="T_78b05_row74_col6" class="data row74 col6" >-0.015200</td>
          <td id="T_78b05_row74_col7" class="data row74 col7" >-0.001400</td>
          <td id="T_78b05_row74_col8" class="data row74 col8" >0.006300</td>
          <td id="T_78b05_row74_col9" class="data row74 col9" >0.060300</td>
          <td id="T_78b05_row74_col10" class="data row74 col10" >0.009900</td>
          <td id="T_78b05_row74_col11" class="data row74 col11" >0.008600</td>
          <td id="T_78b05_row74_col12" class="data row74 col12" >0.036700</td>
          <td id="T_78b05_row74_col13" class="data row74 col13" >0.011300</td>
          <td id="T_78b05_row74_col14" class="data row74 col14" >0.000400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row75" class="row_heading level0 row75" >76</th>
          <td id="T_78b05_row75_col0" class="data row75 col0" >None</td>
          <td id="T_78b05_row75_col1" class="data row75 col1" >0.036300</td>
          <td id="T_78b05_row75_col2" class="data row75 col2" >0.042600</td>
          <td id="T_78b05_row75_col3" class="data row75 col3" >0.000300</td>
          <td id="T_78b05_row75_col4" class="data row75 col4" >-0.045100</td>
          <td id="T_78b05_row75_col5" class="data row75 col5" >0.000500</td>
          <td id="T_78b05_row75_col6" class="data row75 col6" >0.074500</td>
          <td id="T_78b05_row75_col7" class="data row75 col7" >0.051500</td>
          <td id="T_78b05_row75_col8" class="data row75 col8" >0.009200</td>
          <td id="T_78b05_row75_col9" class="data row75 col9" >0.072500</td>
          <td id="T_78b05_row75_col10" class="data row75 col10" >0.030400</td>
          <td id="T_78b05_row75_col11" class="data row75 col11" >0.027200</td>
          <td id="T_78b05_row75_col12" class="data row75 col12" >0.001100</td>
          <td id="T_78b05_row75_col13" class="data row75 col13" >0.078400</td>
          <td id="T_78b05_row75_col14" class="data row75 col14" >0.053200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row76" class="row_heading level0 row76" >77</th>
          <td id="T_78b05_row76_col0" class="data row76 col0" >None</td>
          <td id="T_78b05_row76_col1" class="data row76 col1" >0.035900</td>
          <td id="T_78b05_row76_col2" class="data row76 col2" >0.012800</td>
          <td id="T_78b05_row76_col3" class="data row76 col3" >-0.013600</td>
          <td id="T_78b05_row76_col4" class="data row76 col4" >-0.045200</td>
          <td id="T_78b05_row76_col5" class="data row76 col5" >-0.032200</td>
          <td id="T_78b05_row76_col6" class="data row76 col6" >0.032300</td>
          <td id="T_78b05_row76_col7" class="data row76 col7" >-0.049900</td>
          <td id="T_78b05_row76_col8" class="data row76 col8" >0.009600</td>
          <td id="T_78b05_row76_col9" class="data row76 col9" >0.042600</td>
          <td id="T_78b05_row76_col10" class="data row76 col10" >0.044300</td>
          <td id="T_78b05_row76_col11" class="data row76 col11" >0.027200</td>
          <td id="T_78b05_row76_col12" class="data row76 col12" >0.031500</td>
          <td id="T_78b05_row76_col13" class="data row76 col13" >0.036200</td>
          <td id="T_78b05_row76_col14" class="data row76 col14" >0.048200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row77" class="row_heading level0 row77" >78</th>
          <td id="T_78b05_row77_col0" class="data row77 col0" >None</td>
          <td id="T_78b05_row77_col1" class="data row77 col1" >0.034300</td>
          <td id="T_78b05_row77_col2" class="data row77 col2" >-0.007700</td>
          <td id="T_78b05_row77_col3" class="data row77 col3" >-0.011900</td>
          <td id="T_78b05_row77_col4" class="data row77 col4" >-0.033900</td>
          <td id="T_78b05_row77_col5" class="data row77 col5" >-0.009600</td>
          <td id="T_78b05_row77_col6" class="data row77 col6" >-0.065500</td>
          <td id="T_78b05_row77_col7" class="data row77 col7" >-0.069900</td>
          <td id="T_78b05_row77_col8" class="data row77 col8" >0.011300</td>
          <td id="T_78b05_row77_col9" class="data row77 col9" >0.022100</td>
          <td id="T_78b05_row77_col10" class="data row77 col10" >0.042700</td>
          <td id="T_78b05_row77_col11" class="data row77 col11" >0.016000</td>
          <td id="T_78b05_row77_col12" class="data row77 col12" >0.008900</td>
          <td id="T_78b05_row77_col13" class="data row77 col13" >0.061600</td>
          <td id="T_78b05_row77_col14" class="data row77 col14" >0.068100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row78" class="row_heading level0 row78" >79</th>
          <td id="T_78b05_row78_col0" class="data row78 col0" >None</td>
          <td id="T_78b05_row78_col1" class="data row78 col1" >0.039300</td>
          <td id="T_78b05_row78_col2" class="data row78 col2" >0.034400</td>
          <td id="T_78b05_row78_col3" class="data row78 col3" >0.010400</td>
          <td id="T_78b05_row78_col4" class="data row78 col4" >0.000000</td>
          <td id="T_78b05_row78_col5" class="data row78 col5" >0.065500</td>
          <td id="T_78b05_row78_col6" class="data row78 col6" >0.005300</td>
          <td id="T_78b05_row78_col7" class="data row78 col7" >0.031900</td>
          <td id="T_78b05_row78_col8" class="data row78 col8" >0.006200</td>
          <td id="T_78b05_row78_col9" class="data row78 col9" >0.064200</td>
          <td id="T_78b05_row78_col10" class="data row78 col10" >0.020400</td>
          <td id="T_78b05_row78_col11" class="data row78 col11" >0.017900</td>
          <td id="T_78b05_row78_col12" class="data row78 col12" >0.066200</td>
          <td id="T_78b05_row78_col13" class="data row78 col13" >0.009200</td>
          <td id="T_78b05_row78_col14" class="data row78 col14" >0.033600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row79" class="row_heading level0 row79" >80</th>
          <td id="T_78b05_row79_col0" class="data row79 col0" >None</td>
          <td id="T_78b05_row79_col1" class="data row79 col1" >0.040200</td>
          <td id="T_78b05_row79_col2" class="data row79 col2" >-0.001000</td>
          <td id="T_78b05_row79_col3" class="data row79 col3" >0.010500</td>
          <td id="T_78b05_row79_col4" class="data row79 col4" >-0.017600</td>
          <td id="T_78b05_row79_col5" class="data row79 col5" >0.003700</td>
          <td id="T_78b05_row79_col6" class="data row79 col6" >-0.026700</td>
          <td id="T_78b05_row79_col7" class="data row79 col7" >0.004500</td>
          <td id="T_78b05_row79_col8" class="data row79 col8" >0.005400</td>
          <td id="T_78b05_row79_col9" class="data row79 col9" >0.028800</td>
          <td id="T_78b05_row79_col10" class="data row79 col10" >0.020300</td>
          <td id="T_78b05_row79_col11" class="data row79 col11" >0.000300</td>
          <td id="T_78b05_row79_col12" class="data row79 col12" >0.004400</td>
          <td id="T_78b05_row79_col13" class="data row79 col13" >0.022800</td>
          <td id="T_78b05_row79_col14" class="data row79 col14" >0.006300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row80" class="row_heading level0 row80" >81</th>
          <td id="T_78b05_row80_col0" class="data row80 col0" >PC3</td>
          <td id="T_78b05_row80_col1" class="data row80 col1" >0.033600</td>
          <td id="T_78b05_row80_col2" class="data row80 col2" >-0.007300</td>
          <td id="T_78b05_row80_col3" class="data row80 col3" >-0.098200</td>
          <td id="T_78b05_row80_col4" class="data row80 col4" >-0.007800</td>
          <td id="T_78b05_row80_col5" class="data row80 col5" >0.078400</td>
          <td id="T_78b05_row80_col6" class="data row80 col6" >-0.038000</td>
          <td id="T_78b05_row80_col7" class="data row80 col7" >0.054700</td>
          <td id="T_78b05_row80_col8" class="data row80 col8" >0.011900</td>
          <td id="T_78b05_row80_col9" class="data row80 col9" >0.022600</td>
          <td id="T_78b05_row80_col10" class="data row80 col10" >0.128900</td>
          <td id="T_78b05_row80_col11" class="data row80 col11" >0.010100</td>
          <td id="T_78b05_row80_col12" class="data row80 col12" >0.079100</td>
          <td id="T_78b05_row80_col13" class="data row80 col13" >0.034100</td>
          <td id="T_78b05_row80_col14" class="data row80 col14" >0.056400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row81" class="row_heading level0 row81" >82</th>
          <td id="T_78b05_row81_col0" class="data row81 col0" >PC7</td>
          <td id="T_78b05_row81_col1" class="data row81 col1" >0.033400</td>
          <td id="T_78b05_row81_col2" class="data row81 col2" >0.041100</td>
          <td id="T_78b05_row81_col3" class="data row81 col3" >-0.033500</td>
          <td id="T_78b05_row81_col4" class="data row81 col4" >-0.005200</td>
          <td id="T_78b05_row81_col5" class="data row81 col5" >-0.024700</td>
          <td id="T_78b05_row81_col6" class="data row81 col6" >0.005800</td>
          <td id="T_78b05_row81_col7" class="data row81 col7" >0.121400</td>
          <td id="T_78b05_row81_col8" class="data row81 col8" >0.012100</td>
          <td id="T_78b05_row81_col9" class="data row81 col9" >0.070900</td>
          <td id="T_78b05_row81_col10" class="data row81 col10" >0.064200</td>
          <td id="T_78b05_row81_col11" class="data row81 col11" >0.012700</td>
          <td id="T_78b05_row81_col12" class="data row81 col12" >0.024100</td>
          <td id="T_78b05_row81_col13" class="data row81 col13" >0.009700</td>
          <td id="T_78b05_row81_col14" class="data row81 col14" >0.123200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row82" class="row_heading level0 row82" >83</th>
          <td id="T_78b05_row82_col0" class="data row82 col0" >None</td>
          <td id="T_78b05_row82_col1" class="data row82 col1" >0.042100</td>
          <td id="T_78b05_row82_col2" class="data row82 col2" >-0.015700</td>
          <td id="T_78b05_row82_col3" class="data row82 col3" >0.013700</td>
          <td id="T_78b05_row82_col4" class="data row82 col4" >-0.022000</td>
          <td id="T_78b05_row82_col5" class="data row82 col5" >0.082700</td>
          <td id="T_78b05_row82_col6" class="data row82 col6" >0.023500</td>
          <td id="T_78b05_row82_col7" class="data row82 col7" >0.052000</td>
          <td id="T_78b05_row82_col8" class="data row82 col8" >0.003500</td>
          <td id="T_78b05_row82_col9" class="data row82 col9" >0.014100</td>
          <td id="T_78b05_row82_col10" class="data row82 col10" >0.017100</td>
          <td id="T_78b05_row82_col11" class="data row82 col11" >0.004100</td>
          <td id="T_78b05_row82_col12" class="data row82 col12" >0.083400</td>
          <td id="T_78b05_row82_col13" class="data row82 col13" >0.027400</td>
          <td id="T_78b05_row82_col14" class="data row82 col14" >0.053700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row83" class="row_heading level0 row83" >84</th>
          <td id="T_78b05_row83_col0" class="data row83 col0" >PC1</td>
          <td id="T_78b05_row83_col1" class="data row83 col1" >0.021000</td>
          <td id="T_78b05_row83_col2" class="data row83 col2" >0.047800</td>
          <td id="T_78b05_row83_col3" class="data row83 col3" >-0.075200</td>
          <td id="T_78b05_row83_col4" class="data row83 col4" >-0.005400</td>
          <td id="T_78b05_row83_col5" class="data row83 col5" >0.093500</td>
          <td id="T_78b05_row83_col6" class="data row83 col6" >-0.009600</td>
          <td id="T_78b05_row83_col7" class="data row83 col7" >-0.047100</td>
          <td id="T_78b05_row83_col8" class="data row83 col8" >0.024500</td>
          <td id="T_78b05_row83_col9" class="data row83 col9" >0.077600</td>
          <td id="T_78b05_row83_col10" class="data row83 col10" >0.106000</td>
          <td id="T_78b05_row83_col11" class="data row83 col11" >0.012500</td>
          <td id="T_78b05_row83_col12" class="data row83 col12" >0.094100</td>
          <td id="T_78b05_row83_col13" class="data row83 col13" >0.005800</td>
          <td id="T_78b05_row83_col14" class="data row83 col14" >0.045400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row84" class="row_heading level0 row84" >85</th>
          <td id="T_78b05_row84_col0" class="data row84 col0" >None</td>
          <td id="T_78b05_row84_col1" class="data row84 col1" >0.037500</td>
          <td id="T_78b05_row84_col2" class="data row84 col2" >-0.040400</td>
          <td id="T_78b05_row84_col3" class="data row84 col3" >-0.016300</td>
          <td id="T_78b05_row84_col4" class="data row84 col4" >0.044000</td>
          <td id="T_78b05_row84_col5" class="data row84 col5" >0.071500</td>
          <td id="T_78b05_row84_col6" class="data row84 col6" >0.021400</td>
          <td id="T_78b05_row84_col7" class="data row84 col7" >-0.025800</td>
          <td id="T_78b05_row84_col8" class="data row84 col8" >0.008000</td>
          <td id="T_78b05_row84_col9" class="data row84 col9" >0.010600</td>
          <td id="T_78b05_row84_col10" class="data row84 col10" >0.047000</td>
          <td id="T_78b05_row84_col11" class="data row84 col11" >0.061900</td>
          <td id="T_78b05_row84_col12" class="data row84 col12" >0.072100</td>
          <td id="T_78b05_row84_col13" class="data row84 col13" >0.025300</td>
          <td id="T_78b05_row84_col14" class="data row84 col14" >0.024000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row85" class="row_heading level0 row85" >86</th>
          <td id="T_78b05_row85_col0" class="data row85 col0" >None</td>
          <td id="T_78b05_row85_col1" class="data row85 col1" >0.037600</td>
          <td id="T_78b05_row85_col2" class="data row85 col2" >0.009800</td>
          <td id="T_78b05_row85_col3" class="data row85 col3" >-0.031800</td>
          <td id="T_78b05_row85_col4" class="data row85 col4" >0.013900</td>
          <td id="T_78b05_row85_col5" class="data row85 col5" >0.060000</td>
          <td id="T_78b05_row85_col6" class="data row85 col6" >-0.034300</td>
          <td id="T_78b05_row85_col7" class="data row85 col7" >0.030700</td>
          <td id="T_78b05_row85_col8" class="data row85 col8" >0.008000</td>
          <td id="T_78b05_row85_col9" class="data row85 col9" >0.039700</td>
          <td id="T_78b05_row85_col10" class="data row85 col10" >0.062500</td>
          <td id="T_78b05_row85_col11" class="data row85 col11" >0.031800</td>
          <td id="T_78b05_row85_col12" class="data row85 col12" >0.060600</td>
          <td id="T_78b05_row85_col13" class="data row85 col13" >0.030400</td>
          <td id="T_78b05_row85_col14" class="data row85 col14" >0.032400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row86" class="row_heading level0 row86" >87</th>
          <td id="T_78b05_row86_col0" class="data row86 col0" >None</td>
          <td id="T_78b05_row86_col1" class="data row86 col1" >0.047300</td>
          <td id="T_78b05_row86_col2" class="data row86 col2" >-0.037600</td>
          <td id="T_78b05_row86_col3" class="data row86 col3" >0.056400</td>
          <td id="T_78b05_row86_col4" class="data row86 col4" >0.013600</td>
          <td id="T_78b05_row86_col5" class="data row86 col5" >-0.002100</td>
          <td id="T_78b05_row86_col6" class="data row86 col6" >-0.022200</td>
          <td id="T_78b05_row86_col7" class="data row86 col7" >-0.010600</td>
          <td id="T_78b05_row86_col8" class="data row86 col8" >0.001800</td>
          <td id="T_78b05_row86_col9" class="data row86 col9" >0.007700</td>
          <td id="T_78b05_row86_col10" class="data row86 col10" >0.025700</td>
          <td id="T_78b05_row86_col11" class="data row86 col11" >0.031500</td>
          <td id="T_78b05_row86_col12" class="data row86 col12" >0.001500</td>
          <td id="T_78b05_row86_col13" class="data row86 col13" >0.018300</td>
          <td id="T_78b05_row86_col14" class="data row86 col14" >0.008900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row87" class="row_heading level0 row87" >88</th>
          <td id="T_78b05_row87_col0" class="data row87 col0" >None</td>
          <td id="T_78b05_row87_col1" class="data row87 col1" >0.037200</td>
          <td id="T_78b05_row87_col2" class="data row87 col2" >-0.003300</td>
          <td id="T_78b05_row87_col3" class="data row87 col3" >-0.058300</td>
          <td id="T_78b05_row87_col4" class="data row87 col4" >0.020600</td>
          <td id="T_78b05_row87_col5" class="data row87 col5" >0.046500</td>
          <td id="T_78b05_row87_col6" class="data row87 col6" >-0.018800</td>
          <td id="T_78b05_row87_col7" class="data row87 col7" >0.060700</td>
          <td id="T_78b05_row87_col8" class="data row87 col8" >0.008300</td>
          <td id="T_78b05_row87_col9" class="data row87 col9" >0.026500</td>
          <td id="T_78b05_row87_col10" class="data row87 col10" >0.089000</td>
          <td id="T_78b05_row87_col11" class="data row87 col11" >0.038500</td>
          <td id="T_78b05_row87_col12" class="data row87 col12" >0.047100</td>
          <td id="T_78b05_row87_col13" class="data row87 col13" >0.014900</td>
          <td id="T_78b05_row87_col14" class="data row87 col14" >0.062400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row88" class="row_heading level0 row88" >89</th>
          <td id="T_78b05_row88_col0" class="data row88 col0" >None</td>
          <td id="T_78b05_row88_col1" class="data row88 col1" >0.035800</td>
          <td id="T_78b05_row88_col2" class="data row88 col2" >0.039400</td>
          <td id="T_78b05_row88_col3" class="data row88 col3" >0.022800</td>
          <td id="T_78b05_row88_col4" class="data row88 col4" >-0.004200</td>
          <td id="T_78b05_row88_col5" class="data row88 col5" >0.018600</td>
          <td id="T_78b05_row88_col6" class="data row88 col6" >-0.011500</td>
          <td id="T_78b05_row88_col7" class="data row88 col7" >0.014600</td>
          <td id="T_78b05_row88_col8" class="data row88 col8" >0.009700</td>
          <td id="T_78b05_row88_col9" class="data row88 col9" >0.069200</td>
          <td id="T_78b05_row88_col10" class="data row88 col10" >0.007900</td>
          <td id="T_78b05_row88_col11" class="data row88 col11" >0.013700</td>
          <td id="T_78b05_row88_col12" class="data row88 col12" >0.019200</td>
          <td id="T_78b05_row88_col13" class="data row88 col13" >0.007600</td>
          <td id="T_78b05_row88_col14" class="data row88 col14" >0.016300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row89" class="row_heading level0 row89" >90</th>
          <td id="T_78b05_row89_col0" class="data row89 col0" >PC4</td>
          <td id="T_78b05_row89_col1" class="data row89 col1" >0.039000</td>
          <td id="T_78b05_row89_col2" class="data row89 col2" >0.032000</td>
          <td id="T_78b05_row89_col3" class="data row89 col3" >0.001300</td>
          <td id="T_78b05_row89_col4" class="data row89 col4" >0.110900</td>
          <td id="T_78b05_row89_col5" class="data row89 col5" >0.031400</td>
          <td id="T_78b05_row89_col6" class="data row89 col6" >-0.004200</td>
          <td id="T_78b05_row89_col7" class="data row89 col7" >-0.015400</td>
          <td id="T_78b05_row89_col8" class="data row89 col8" >0.006600</td>
          <td id="T_78b05_row89_col9" class="data row89 col9" >0.061800</td>
          <td id="T_78b05_row89_col10" class="data row89 col10" >0.029400</td>
          <td id="T_78b05_row89_col11" class="data row89 col11" >0.128800</td>
          <td id="T_78b05_row89_col12" class="data row89 col12" >0.032000</td>
          <td id="T_78b05_row89_col13" class="data row89 col13" >0.000300</td>
          <td id="T_78b05_row89_col14" class="data row89 col14" >0.013700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row90" class="row_heading level0 row90" >91</th>
          <td id="T_78b05_row90_col0" class="data row90 col0" >None</td>
          <td id="T_78b05_row90_col1" class="data row90 col1" >0.033900</td>
          <td id="T_78b05_row90_col2" class="data row90 col2" >-0.007300</td>
          <td id="T_78b05_row90_col3" class="data row90 col3" >-0.043400</td>
          <td id="T_78b05_row90_col4" class="data row90 col4" >0.015800</td>
          <td id="T_78b05_row90_col5" class="data row90 col5" >-0.024100</td>
          <td id="T_78b05_row90_col6" class="data row90 col6" >0.014400</td>
          <td id="T_78b05_row90_col7" class="data row90 col7" >-0.009100</td>
          <td id="T_78b05_row90_col8" class="data row90 col8" >0.011600</td>
          <td id="T_78b05_row90_col9" class="data row90 col9" >0.022500</td>
          <td id="T_78b05_row90_col10" class="data row90 col10" >0.074100</td>
          <td id="T_78b05_row90_col11" class="data row90 col11" >0.033700</td>
          <td id="T_78b05_row90_col12" class="data row90 col12" >0.023400</td>
          <td id="T_78b05_row90_col13" class="data row90 col13" >0.018300</td>
          <td id="T_78b05_row90_col14" class="data row90 col14" >0.007400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row91" class="row_heading level0 row91" >92</th>
          <td id="T_78b05_row91_col0" class="data row91 col0" >None</td>
          <td id="T_78b05_row91_col1" class="data row91 col1" >0.033300</td>
          <td id="T_78b05_row91_col2" class="data row91 col2" >0.070200</td>
          <td id="T_78b05_row91_col3" class="data row91 col3" >-0.011600</td>
          <td id="T_78b05_row91_col4" class="data row91 col4" >-0.011900</td>
          <td id="T_78b05_row91_col5" class="data row91 col5" >0.015000</td>
          <td id="T_78b05_row91_col6" class="data row91 col6" >0.061200</td>
          <td id="T_78b05_row91_col7" class="data row91 col7" >0.000400</td>
          <td id="T_78b05_row91_col8" class="data row91 col8" >0.012200</td>
          <td id="T_78b05_row91_col9" class="data row91 col9" >0.100000</td>
          <td id="T_78b05_row91_col10" class="data row91 col10" >0.042400</td>
          <td id="T_78b05_row91_col11" class="data row91 col11" >0.006000</td>
          <td id="T_78b05_row91_col12" class="data row91 col12" >0.015600</td>
          <td id="T_78b05_row91_col13" class="data row91 col13" >0.065100</td>
          <td id="T_78b05_row91_col14" class="data row91 col14" >0.002100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row92" class="row_heading level0 row92" >93</th>
          <td id="T_78b05_row92_col0" class="data row92 col0" >None</td>
          <td id="T_78b05_row92_col1" class="data row92 col1" >0.038500</td>
          <td id="T_78b05_row92_col2" class="data row92 col2" >-0.013600</td>
          <td id="T_78b05_row92_col3" class="data row92 col3" >-0.061200</td>
          <td id="T_78b05_row92_col4" class="data row92 col4" >0.002500</td>
          <td id="T_78b05_row92_col5" class="data row92 col5" >0.064600</td>
          <td id="T_78b05_row92_col6" class="data row92 col6" >0.028900</td>
          <td id="T_78b05_row92_col7" class="data row92 col7" >0.017100</td>
          <td id="T_78b05_row92_col8" class="data row92 col8" >0.007100</td>
          <td id="T_78b05_row92_col9" class="data row92 col9" >0.016200</td>
          <td id="T_78b05_row92_col10" class="data row92 col10" >0.092000</td>
          <td id="T_78b05_row92_col11" class="data row92 col11" >0.020400</td>
          <td id="T_78b05_row92_col12" class="data row92 col12" >0.065200</td>
          <td id="T_78b05_row92_col13" class="data row92 col13" >0.032800</td>
          <td id="T_78b05_row92_col14" class="data row92 col14" >0.018800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row93" class="row_heading level0 row93" >94</th>
          <td id="T_78b05_row93_col0" class="data row93 col0" >None</td>
          <td id="T_78b05_row93_col1" class="data row93 col1" >0.036100</td>
          <td id="T_78b05_row93_col2" class="data row93 col2" >-0.021000</td>
          <td id="T_78b05_row93_col3" class="data row93 col3" >-0.033600</td>
          <td id="T_78b05_row93_col4" class="data row93 col4" >-0.046700</td>
          <td id="T_78b05_row93_col5" class="data row93 col5" >-0.005700</td>
          <td id="T_78b05_row93_col6" class="data row93 col6" >0.036000</td>
          <td id="T_78b05_row93_col7" class="data row93 col7" >-0.042100</td>
          <td id="T_78b05_row93_col8" class="data row93 col8" >0.009500</td>
          <td id="T_78b05_row93_col9" class="data row93 col9" >0.008800</td>
          <td id="T_78b05_row93_col10" class="data row93 col10" >0.064300</td>
          <td id="T_78b05_row93_col11" class="data row93 col11" >0.028800</td>
          <td id="T_78b05_row93_col12" class="data row93 col12" >0.005100</td>
          <td id="T_78b05_row93_col13" class="data row93 col13" >0.039900</td>
          <td id="T_78b05_row93_col14" class="data row93 col14" >0.040300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row94" class="row_heading level0 row94" >95</th>
          <td id="T_78b05_row94_col0" class="data row94 col0" >PC2</td>
          <td id="T_78b05_row94_col1" class="data row94 col1" >0.032000</td>
          <td id="T_78b05_row94_col2" class="data row94 col2" >0.082100</td>
          <td id="T_78b05_row94_col3" class="data row94 col3" >-0.025800</td>
          <td id="T_78b05_row94_col4" class="data row94 col4" >-0.037700</td>
          <td id="T_78b05_row94_col5" class="data row94 col5" >-0.019000</td>
          <td id="T_78b05_row94_col6" class="data row94 col6" >0.028300</td>
          <td id="T_78b05_row94_col7" class="data row94 col7" >0.007000</td>
          <td id="T_78b05_row94_col8" class="data row94 col8" >0.013600</td>
          <td id="T_78b05_row94_col9" class="data row94 col9" >0.111900</td>
          <td id="T_78b05_row94_col10" class="data row94 col10" >0.056500</td>
          <td id="T_78b05_row94_col11" class="data row94 col11" >0.019800</td>
          <td id="T_78b05_row94_col12" class="data row94 col12" >0.018400</td>
          <td id="T_78b05_row94_col13" class="data row94 col13" >0.032200</td>
          <td id="T_78b05_row94_col14" class="data row94 col14" >0.008700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row95" class="row_heading level0 row95" >96</th>
          <td id="T_78b05_row95_col0" class="data row95 col0" >None</td>
          <td id="T_78b05_row95_col1" class="data row95 col1" >0.048000</td>
          <td id="T_78b05_row95_col2" class="data row95 col2" >-0.055400</td>
          <td id="T_78b05_row95_col3" class="data row95 col3" >0.018700</td>
          <td id="T_78b05_row95_col4" class="data row95 col4" >-0.021600</td>
          <td id="T_78b05_row95_col5" class="data row95 col5" >-0.012100</td>
          <td id="T_78b05_row95_col6" class="data row95 col6" >0.014700</td>
          <td id="T_78b05_row95_col7" class="data row95 col7" >0.033700</td>
          <td id="T_78b05_row95_col8" class="data row95 col8" >0.002500</td>
          <td id="T_78b05_row95_col9" class="data row95 col9" >0.025600</td>
          <td id="T_78b05_row95_col10" class="data row95 col10" >0.012100</td>
          <td id="T_78b05_row95_col11" class="data row95 col11" >0.003700</td>
          <td id="T_78b05_row95_col12" class="data row95 col12" >0.011500</td>
          <td id="T_78b05_row95_col13" class="data row95 col13" >0.018600</td>
          <td id="T_78b05_row95_col14" class="data row95 col14" >0.035500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row96" class="row_heading level0 row96" >97</th>
          <td id="T_78b05_row96_col0" class="data row96 col0" >None</td>
          <td id="T_78b05_row96_col1" class="data row96 col1" >0.041000</td>
          <td id="T_78b05_row96_col2" class="data row96 col2" >0.003500</td>
          <td id="T_78b05_row96_col3" class="data row96 col3" >0.000100</td>
          <td id="T_78b05_row96_col4" class="data row96 col4" >-0.009900</td>
          <td id="T_78b05_row96_col5" class="data row96 col5" >0.054400</td>
          <td id="T_78b05_row96_col6" class="data row96 col6" >-0.024500</td>
          <td id="T_78b05_row96_col7" class="data row96 col7" >0.030000</td>
          <td id="T_78b05_row96_col8" class="data row96 col8" >0.004500</td>
          <td id="T_78b05_row96_col9" class="data row96 col9" >0.033300</td>
          <td id="T_78b05_row96_col10" class="data row96 col10" >0.030600</td>
          <td id="T_78b05_row96_col11" class="data row96 col11" >0.008000</td>
          <td id="T_78b05_row96_col12" class="data row96 col12" >0.055100</td>
          <td id="T_78b05_row96_col13" class="data row96 col13" >0.020600</td>
          <td id="T_78b05_row96_col14" class="data row96 col14" >0.031800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row97" class="row_heading level0 row97" >98</th>
          <td id="T_78b05_row97_col0" class="data row97 col0" >None</td>
          <td id="T_78b05_row97_col1" class="data row97 col1" >0.035800</td>
          <td id="T_78b05_row97_col2" class="data row97 col2" >0.013600</td>
          <td id="T_78b05_row97_col3" class="data row97 col3" >-0.013700</td>
          <td id="T_78b05_row97_col4" class="data row97 col4" >0.008300</td>
          <td id="T_78b05_row97_col5" class="data row97 col5" >-0.000200</td>
          <td id="T_78b05_row97_col6" class="data row97 col6" >0.018100</td>
          <td id="T_78b05_row97_col7" class="data row97 col7" >0.039200</td>
          <td id="T_78b05_row97_col8" class="data row97 col8" >0.009700</td>
          <td id="T_78b05_row97_col9" class="data row97 col9" >0.043400</td>
          <td id="T_78b05_row97_col10" class="data row97 col10" >0.044500</td>
          <td id="T_78b05_row97_col11" class="data row97 col11" >0.026200</td>
          <td id="T_78b05_row97_col12" class="data row97 col12" >0.000400</td>
          <td id="T_78b05_row97_col13" class="data row97 col13" >0.022000</td>
          <td id="T_78b05_row97_col14" class="data row97 col14" >0.041000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row98" class="row_heading level0 row98" >99</th>
          <td id="T_78b05_row98_col0" class="data row98 col0" >None</td>
          <td id="T_78b05_row98_col1" class="data row98 col1" >0.034700</td>
          <td id="T_78b05_row98_col2" class="data row98 col2" >0.025700</td>
          <td id="T_78b05_row98_col3" class="data row98 col3" >-0.025800</td>
          <td id="T_78b05_row98_col4" class="data row98 col4" >-0.034800</td>
          <td id="T_78b05_row98_col5" class="data row98 col5" >-0.008300</td>
          <td id="T_78b05_row98_col6" class="data row98 col6" >-0.020000</td>
          <td id="T_78b05_row98_col7" class="data row98 col7" >-0.005700</td>
          <td id="T_78b05_row98_col8" class="data row98 col8" >0.010800</td>
          <td id="T_78b05_row98_col9" class="data row98 col9" >0.055500</td>
          <td id="T_78b05_row98_col10" class="data row98 col10" >0.056600</td>
          <td id="T_78b05_row98_col11" class="data row98 col11" >0.016900</td>
          <td id="T_78b05_row98_col12" class="data row98 col12" >0.007700</td>
          <td id="T_78b05_row98_col13" class="data row98 col13" >0.016100</td>
          <td id="T_78b05_row98_col14" class="data row98 col14" >0.003900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row99" class="row_heading level0 row99" >100</th>
          <td id="T_78b05_row99_col0" class="data row99 col0" >None</td>
          <td id="T_78b05_row99_col1" class="data row99 col1" >0.035500</td>
          <td id="T_78b05_row99_col2" class="data row99 col2" >0.009800</td>
          <td id="T_78b05_row99_col3" class="data row99 col3" >-0.018400</td>
          <td id="T_78b05_row99_col4" class="data row99 col4" >0.089300</td>
          <td id="T_78b05_row99_col5" class="data row99 col5" >-0.049700</td>
          <td id="T_78b05_row99_col6" class="data row99 col6" >0.021900</td>
          <td id="T_78b05_row99_col7" class="data row99 col7" >0.059600</td>
          <td id="T_78b05_row99_col8" class="data row99 col8" >0.010000</td>
          <td id="T_78b05_row99_col9" class="data row99 col9" >0.039600</td>
          <td id="T_78b05_row99_col10" class="data row99 col10" >0.049200</td>
          <td id="T_78b05_row99_col11" class="data row99 col11" >0.107200</td>
          <td id="T_78b05_row99_col12" class="data row99 col12" >0.049100</td>
          <td id="T_78b05_row99_col13" class="data row99 col13" >0.025800</td>
          <td id="T_78b05_row99_col14" class="data row99 col14" >0.061400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row100" class="row_heading level0 row100" >101</th>
          <td id="T_78b05_row100_col0" class="data row100 col0" >None</td>
          <td id="T_78b05_row100_col1" class="data row100 col1" >0.030500</td>
          <td id="T_78b05_row100_col2" class="data row100 col2" >0.038000</td>
          <td id="T_78b05_row100_col3" class="data row100 col3" >-0.045600</td>
          <td id="T_78b05_row100_col4" class="data row100 col4" >0.005400</td>
          <td id="T_78b05_row100_col5" class="data row100 col5" >0.025500</td>
          <td id="T_78b05_row100_col6" class="data row100 col6" >-0.016800</td>
          <td id="T_78b05_row100_col7" class="data row100 col7" >-0.078200</td>
          <td id="T_78b05_row100_col8" class="data row100 col8" >0.015000</td>
          <td id="T_78b05_row100_col9" class="data row100 col9" >0.067900</td>
          <td id="T_78b05_row100_col10" class="data row100 col10" >0.076300</td>
          <td id="T_78b05_row100_col11" class="data row100 col11" >0.023400</td>
          <td id="T_78b05_row100_col12" class="data row100 col12" >0.026200</td>
          <td id="T_78b05_row100_col13" class="data row100 col13" >0.012900</td>
          <td id="T_78b05_row100_col14" class="data row100 col14" >0.076400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row101" class="row_heading level0 row101" >102</th>
          <td id="T_78b05_row101_col0" class="data row101 col0" >None</td>
          <td id="T_78b05_row101_col1" class="data row101 col1" >0.029300</td>
          <td id="T_78b05_row101_col2" class="data row101 col2" >0.017400</td>
          <td id="T_78b05_row101_col3" class="data row101 col3" >-0.043100</td>
          <td id="T_78b05_row101_col4" class="data row101 col4" >0.005600</td>
          <td id="T_78b05_row101_col5" class="data row101 col5" >0.000800</td>
          <td id="T_78b05_row101_col6" class="data row101 col6" >0.021700</td>
          <td id="T_78b05_row101_col7" class="data row101 col7" >-0.011000</td>
          <td id="T_78b05_row101_col8" class="data row101 col8" >0.016200</td>
          <td id="T_78b05_row101_col9" class="data row101 col9" >0.047200</td>
          <td id="T_78b05_row101_col10" class="data row101 col10" >0.073900</td>
          <td id="T_78b05_row101_col11" class="data row101 col11" >0.023500</td>
          <td id="T_78b05_row101_col12" class="data row101 col12" >0.001500</td>
          <td id="T_78b05_row101_col13" class="data row101 col13" >0.025600</td>
          <td id="T_78b05_row101_col14" class="data row101 col14" >0.009200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row102" class="row_heading level0 row102" >103</th>
          <td id="T_78b05_row102_col0" class="data row102 col0" >None</td>
          <td id="T_78b05_row102_col1" class="data row102 col1" >0.043400</td>
          <td id="T_78b05_row102_col2" class="data row102 col2" >-0.027300</td>
          <td id="T_78b05_row102_col3" class="data row102 col3" >-0.017700</td>
          <td id="T_78b05_row102_col4" class="data row102 col4" >0.002400</td>
          <td id="T_78b05_row102_col5" class="data row102 col5" >0.029700</td>
          <td id="T_78b05_row102_col6" class="data row102 col6" >-0.010700</td>
          <td id="T_78b05_row102_col7" class="data row102 col7" >0.031500</td>
          <td id="T_78b05_row102_col8" class="data row102 col8" >0.002100</td>
          <td id="T_78b05_row102_col9" class="data row102 col9" >0.002500</td>
          <td id="T_78b05_row102_col10" class="data row102 col10" >0.048400</td>
          <td id="T_78b05_row102_col11" class="data row102 col11" >0.020300</td>
          <td id="T_78b05_row102_col12" class="data row102 col12" >0.030400</td>
          <td id="T_78b05_row102_col13" class="data row102 col13" >0.006900</td>
          <td id="T_78b05_row102_col14" class="data row102 col14" >0.033300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row103" class="row_heading level0 row103" >104</th>
          <td id="T_78b05_row103_col0" class="data row103 col0" >None</td>
          <td id="T_78b05_row103_col1" class="data row103 col1" >0.032200</td>
          <td id="T_78b05_row103_col2" class="data row103 col2" >0.013900</td>
          <td id="T_78b05_row103_col3" class="data row103 col3" >-0.070800</td>
          <td id="T_78b05_row103_col4" class="data row103 col4" >-0.050100</td>
          <td id="T_78b05_row103_col5" class="data row103 col5" >0.011200</td>
          <td id="T_78b05_row103_col6" class="data row103 col6" >-0.015500</td>
          <td id="T_78b05_row103_col7" class="data row103 col7" >0.028300</td>
          <td id="T_78b05_row103_col8" class="data row103 col8" >0.013300</td>
          <td id="T_78b05_row103_col9" class="data row103 col9" >0.043800</td>
          <td id="T_78b05_row103_col10" class="data row103 col10" >0.101600</td>
          <td id="T_78b05_row103_col11" class="data row103 col11" >0.032200</td>
          <td id="T_78b05_row103_col12" class="data row103 col12" >0.011800</td>
          <td id="T_78b05_row103_col13" class="data row103 col13" >0.011600</td>
          <td id="T_78b05_row103_col14" class="data row103 col14" >0.030000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row104" class="row_heading level0 row104" >105</th>
          <td id="T_78b05_row104_col0" class="data row104 col0" >None</td>
          <td id="T_78b05_row104_col1" class="data row104 col1" >0.026800</td>
          <td id="T_78b05_row104_col2" class="data row104 col2" >-0.015500</td>
          <td id="T_78b05_row104_col3" class="data row104 col3" >-0.026200</td>
          <td id="T_78b05_row104_col4" class="data row104 col4" >-0.005200</td>
          <td id="T_78b05_row104_col5" class="data row104 col5" >-0.059500</td>
          <td id="T_78b05_row104_col6" class="data row104 col6" >-0.022100</td>
          <td id="T_78b05_row104_col7" class="data row104 col7" >-0.009300</td>
          <td id="T_78b05_row104_col8" class="data row104 col8" >0.018700</td>
          <td id="T_78b05_row104_col9" class="data row104 col9" >0.014400</td>
          <td id="T_78b05_row104_col10" class="data row104 col10" >0.057000</td>
          <td id="T_78b05_row104_col11" class="data row104 col11" >0.012700</td>
          <td id="T_78b05_row104_col12" class="data row104 col12" >0.058900</td>
          <td id="T_78b05_row104_col13" class="data row104 col13" >0.018200</td>
          <td id="T_78b05_row104_col14" class="data row104 col14" >0.007600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row105" class="row_heading level0 row105" >106</th>
          <td id="T_78b05_row105_col0" class="data row105 col0" >None</td>
          <td id="T_78b05_row105_col1" class="data row105 col1" >0.039100</td>
          <td id="T_78b05_row105_col2" class="data row105 col2" >0.004900</td>
          <td id="T_78b05_row105_col3" class="data row105 col3" >0.015400</td>
          <td id="T_78b05_row105_col4" class="data row105 col4" >0.065500</td>
          <td id="T_78b05_row105_col5" class="data row105 col5" >-0.062900</td>
          <td id="T_78b05_row105_col6" class="data row105 col6" >0.006600</td>
          <td id="T_78b05_row105_col7" class="data row105 col7" >0.025800</td>
          <td id="T_78b05_row105_col8" class="data row105 col8" >0.006400</td>
          <td id="T_78b05_row105_col9" class="data row105 col9" >0.034800</td>
          <td id="T_78b05_row105_col10" class="data row105 col10" >0.015400</td>
          <td id="T_78b05_row105_col11" class="data row105 col11" >0.083400</td>
          <td id="T_78b05_row105_col12" class="data row105 col12" >0.062200</td>
          <td id="T_78b05_row105_col13" class="data row105 col13" >0.010500</td>
          <td id="T_78b05_row105_col14" class="data row105 col14" >0.027500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row106" class="row_heading level0 row106" >107</th>
          <td id="T_78b05_row106_col0" class="data row106 col0" >None</td>
          <td id="T_78b05_row106_col1" class="data row106 col1" >0.036600</td>
          <td id="T_78b05_row106_col2" class="data row106 col2" >-0.002200</td>
          <td id="T_78b05_row106_col3" class="data row106 col3" >-0.026900</td>
          <td id="T_78b05_row106_col4" class="data row106 col4" >-0.005000</td>
          <td id="T_78b05_row106_col5" class="data row106 col5" >0.021300</td>
          <td id="T_78b05_row106_col6" class="data row106 col6" >-0.064700</td>
          <td id="T_78b05_row106_col7" class="data row106 col7" >-0.039100</td>
          <td id="T_78b05_row106_col8" class="data row106 col8" >0.008900</td>
          <td id="T_78b05_row106_col9" class="data row106 col9" >0.027600</td>
          <td id="T_78b05_row106_col10" class="data row106 col10" >0.057600</td>
          <td id="T_78b05_row106_col11" class="data row106 col11" >0.012900</td>
          <td id="T_78b05_row106_col12" class="data row106 col12" >0.021900</td>
          <td id="T_78b05_row106_col13" class="data row106 col13" >0.060800</td>
          <td id="T_78b05_row106_col14" class="data row106 col14" >0.037300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row107" class="row_heading level0 row107" >108</th>
          <td id="T_78b05_row107_col0" class="data row107 col0" >None</td>
          <td id="T_78b05_row107_col1" class="data row107 col1" >0.029500</td>
          <td id="T_78b05_row107_col2" class="data row107 col2" >-0.014800</td>
          <td id="T_78b05_row107_col3" class="data row107 col3" >-0.050900</td>
          <td id="T_78b05_row107_col4" class="data row107 col4" >-0.000800</td>
          <td id="T_78b05_row107_col5" class="data row107 col5" >-0.049900</td>
          <td id="T_78b05_row107_col6" class="data row107 col6" >0.051100</td>
          <td id="T_78b05_row107_col7" class="data row107 col7" >0.023900</td>
          <td id="T_78b05_row107_col8" class="data row107 col8" >0.016100</td>
          <td id="T_78b05_row107_col9" class="data row107 col9" >0.015000</td>
          <td id="T_78b05_row107_col10" class="data row107 col10" >0.081700</td>
          <td id="T_78b05_row107_col11" class="data row107 col11" >0.017100</td>
          <td id="T_78b05_row107_col12" class="data row107 col12" >0.049300</td>
          <td id="T_78b05_row107_col13" class="data row107 col13" >0.054900</td>
          <td id="T_78b05_row107_col14" class="data row107 col14" >0.025700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row108" class="row_heading level0 row108" >109</th>
          <td id="T_78b05_row108_col0" class="data row108 col0" >PC1</td>
          <td id="T_78b05_row108_col1" class="data row108 col1" >0.026100</td>
          <td id="T_78b05_row108_col2" class="data row108 col2" >0.058500</td>
          <td id="T_78b05_row108_col3" class="data row108 col3" >-0.075700</td>
          <td id="T_78b05_row108_col4" class="data row108 col4" >-0.020900</td>
          <td id="T_78b05_row108_col5" class="data row108 col5" >-0.026500</td>
          <td id="T_78b05_row108_col6" class="data row108 col6" >0.009900</td>
          <td id="T_78b05_row108_col7" class="data row108 col7" >-0.002900</td>
          <td id="T_78b05_row108_col8" class="data row108 col8" >0.019500</td>
          <td id="T_78b05_row108_col9" class="data row108 col9" >0.088400</td>
          <td id="T_78b05_row108_col10" class="data row108 col10" >0.106400</td>
          <td id="T_78b05_row108_col11" class="data row108 col11" >0.003000</td>
          <td id="T_78b05_row108_col12" class="data row108 col12" >0.025800</td>
          <td id="T_78b05_row108_col13" class="data row108 col13" >0.013800</td>
          <td id="T_78b05_row108_col14" class="data row108 col14" >0.001100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row109" class="row_heading level0 row109" >110</th>
          <td id="T_78b05_row109_col0" class="data row109 col0" >None</td>
          <td id="T_78b05_row109_col1" class="data row109 col1" >0.030000</td>
          <td id="T_78b05_row109_col2" class="data row109 col2" >-0.010200</td>
          <td id="T_78b05_row109_col3" class="data row109 col3" >-0.024000</td>
          <td id="T_78b05_row109_col4" class="data row109 col4" >0.015200</td>
          <td id="T_78b05_row109_col5" class="data row109 col5" >-0.025200</td>
          <td id="T_78b05_row109_col6" class="data row109 col6" >-0.041200</td>
          <td id="T_78b05_row109_col7" class="data row109 col7" >-0.012100</td>
          <td id="T_78b05_row109_col8" class="data row109 col8" >0.015500</td>
          <td id="T_78b05_row109_col9" class="data row109 col9" >0.019600</td>
          <td id="T_78b05_row109_col10" class="data row109 col10" >0.054700</td>
          <td id="T_78b05_row109_col11" class="data row109 col11" >0.033200</td>
          <td id="T_78b05_row109_col12" class="data row109 col12" >0.024600</td>
          <td id="T_78b05_row109_col13" class="data row109 col13" >0.037300</td>
          <td id="T_78b05_row109_col14" class="data row109 col14" >0.010300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row110" class="row_heading level0 row110" >111</th>
          <td id="T_78b05_row110_col0" class="data row110 col0" >None</td>
          <td id="T_78b05_row110_col1" class="data row110 col1" >0.044700</td>
          <td id="T_78b05_row110_col2" class="data row110 col2" >-0.001700</td>
          <td id="T_78b05_row110_col3" class="data row110 col3" >0.015300</td>
          <td id="T_78b05_row110_col4" class="data row110 col4" >0.038400</td>
          <td id="T_78b05_row110_col5" class="data row110 col5" >0.065100</td>
          <td id="T_78b05_row110_col6" class="data row110 col6" >-0.031600</td>
          <td id="T_78b05_row110_col7" class="data row110 col7" >-0.006800</td>
          <td id="T_78b05_row110_col8" class="data row110 col8" >0.000900</td>
          <td id="T_78b05_row110_col9" class="data row110 col9" >0.028200</td>
          <td id="T_78b05_row110_col10" class="data row110 col10" >0.015500</td>
          <td id="T_78b05_row110_col11" class="data row110 col11" >0.056300</td>
          <td id="T_78b05_row110_col12" class="data row110 col12" >0.065700</td>
          <td id="T_78b05_row110_col13" class="data row110 col13" >0.027700</td>
          <td id="T_78b05_row110_col14" class="data row110 col14" >0.005000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row111" class="row_heading level0 row111" >112</th>
          <td id="T_78b05_row111_col0" class="data row111 col0" >None</td>
          <td id="T_78b05_row111_col1" class="data row111 col1" >0.038000</td>
          <td id="T_78b05_row111_col2" class="data row111 col2" >0.009200</td>
          <td id="T_78b05_row111_col3" class="data row111 col3" >-0.018000</td>
          <td id="T_78b05_row111_col4" class="data row111 col4" >-0.001300</td>
          <td id="T_78b05_row111_col5" class="data row111 col5" >-0.034300</td>
          <td id="T_78b05_row111_col6" class="data row111 col6" >0.022600</td>
          <td id="T_78b05_row111_col7" class="data row111 col7" >0.016700</td>
          <td id="T_78b05_row111_col8" class="data row111 col8" >0.007600</td>
          <td id="T_78b05_row111_col9" class="data row111 col9" >0.039000</td>
          <td id="T_78b05_row111_col10" class="data row111 col10" >0.048800</td>
          <td id="T_78b05_row111_col11" class="data row111 col11" >0.016600</td>
          <td id="T_78b05_row111_col12" class="data row111 col12" >0.033600</td>
          <td id="T_78b05_row111_col13" class="data row111 col13" >0.026500</td>
          <td id="T_78b05_row111_col14" class="data row111 col14" >0.018400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row112" class="row_heading level0 row112" >113</th>
          <td id="T_78b05_row112_col0" class="data row112 col0" >None</td>
          <td id="T_78b05_row112_col1" class="data row112 col1" >0.040500</td>
          <td id="T_78b05_row112_col2" class="data row112 col2" >-0.040800</td>
          <td id="T_78b05_row112_col3" class="data row112 col3" >-0.036800</td>
          <td id="T_78b05_row112_col4" class="data row112 col4" >0.000200</td>
          <td id="T_78b05_row112_col5" class="data row112 col5" >0.032400</td>
          <td id="T_78b05_row112_col6" class="data row112 col6" >0.051300</td>
          <td id="T_78b05_row112_col7" class="data row112 col7" >0.002300</td>
          <td id="T_78b05_row112_col8" class="data row112 col8" >0.005000</td>
          <td id="T_78b05_row112_col9" class="data row112 col9" >0.010900</td>
          <td id="T_78b05_row112_col10" class="data row112 col10" >0.067600</td>
          <td id="T_78b05_row112_col11" class="data row112 col11" >0.018100</td>
          <td id="T_78b05_row112_col12" class="data row112 col12" >0.033100</td>
          <td id="T_78b05_row112_col13" class="data row112 col13" >0.055200</td>
          <td id="T_78b05_row112_col14" class="data row112 col14" >0.004000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row113" class="row_heading level0 row113" >114</th>
          <td id="T_78b05_row113_col0" class="data row113 col0" >None</td>
          <td id="T_78b05_row113_col1" class="data row113 col1" >0.047500</td>
          <td id="T_78b05_row113_col2" class="data row113 col2" >-0.053500</td>
          <td id="T_78b05_row113_col3" class="data row113 col3" >0.010700</td>
          <td id="T_78b05_row113_col4" class="data row113 col4" >-0.056100</td>
          <td id="T_78b05_row113_col5" class="data row113 col5" >-0.040800</td>
          <td id="T_78b05_row113_col6" class="data row113 col6" >-0.041500</td>
          <td id="T_78b05_row113_col7" class="data row113 col7" >0.027500</td>
          <td id="T_78b05_row113_col8" class="data row113 col8" >0.002000</td>
          <td id="T_78b05_row113_col9" class="data row113 col9" >0.023700</td>
          <td id="T_78b05_row113_col10" class="data row113 col10" >0.020100</td>
          <td id="T_78b05_row113_col11" class="data row113 col11" >0.038200</td>
          <td id="T_78b05_row113_col12" class="data row113 col12" >0.040200</td>
          <td id="T_78b05_row113_col13" class="data row113 col13" >0.037600</td>
          <td id="T_78b05_row113_col14" class="data row113 col14" >0.029300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row114" class="row_heading level0 row114" >115</th>
          <td id="T_78b05_row114_col0" class="data row114 col0" >None</td>
          <td id="T_78b05_row114_col1" class="data row114 col1" >0.037700</td>
          <td id="T_78b05_row114_col2" class="data row114 col2" >0.018400</td>
          <td id="T_78b05_row114_col3" class="data row114 col3" >-0.047700</td>
          <td id="T_78b05_row114_col4" class="data row114 col4" >-0.035100</td>
          <td id="T_78b05_row114_col5" class="data row114 col5" >-0.014700</td>
          <td id="T_78b05_row114_col6" class="data row114 col6" >-0.019300</td>
          <td id="T_78b05_row114_col7" class="data row114 col7" >0.064500</td>
          <td id="T_78b05_row114_col8" class="data row114 col8" >0.007900</td>
          <td id="T_78b05_row114_col9" class="data row114 col9" >0.048200</td>
          <td id="T_78b05_row114_col10" class="data row114 col10" >0.078500</td>
          <td id="T_78b05_row114_col11" class="data row114 col11" >0.017200</td>
          <td id="T_78b05_row114_col12" class="data row114 col12" >0.014100</td>
          <td id="T_78b05_row114_col13" class="data row114 col13" >0.015400</td>
          <td id="T_78b05_row114_col14" class="data row114 col14" >0.066200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row115" class="row_heading level0 row115" >116</th>
          <td id="T_78b05_row115_col0" class="data row115 col0" >None</td>
          <td id="T_78b05_row115_col1" class="data row115 col1" >0.050300</td>
          <td id="T_78b05_row115_col2" class="data row115 col2" >-0.010300</td>
          <td id="T_78b05_row115_col3" class="data row115 col3" >0.014500</td>
          <td id="T_78b05_row115_col4" class="data row115 col4" >-0.010300</td>
          <td id="T_78b05_row115_col5" class="data row115 col5" >0.051500</td>
          <td id="T_78b05_row115_col6" class="data row115 col6" >-0.015200</td>
          <td id="T_78b05_row115_col7" class="data row115 col7" >0.011000</td>
          <td id="T_78b05_row115_col8" class="data row115 col8" >0.004800</td>
          <td id="T_78b05_row115_col9" class="data row115 col9" >0.019500</td>
          <td id="T_78b05_row115_col10" class="data row115 col10" >0.016200</td>
          <td id="T_78b05_row115_col11" class="data row115 col11" >0.007600</td>
          <td id="T_78b05_row115_col12" class="data row115 col12" >0.052100</td>
          <td id="T_78b05_row115_col13" class="data row115 col13" >0.011300</td>
          <td id="T_78b05_row115_col14" class="data row115 col14" >0.012800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row116" class="row_heading level0 row116" >117</th>
          <td id="T_78b05_row116_col0" class="data row116 col0" >None</td>
          <td id="T_78b05_row116_col1" class="data row116 col1" >0.047900</td>
          <td id="T_78b05_row116_col2" class="data row116 col2" >-0.018200</td>
          <td id="T_78b05_row116_col3" class="data row116 col3" >-0.004600</td>
          <td id="T_78b05_row116_col4" class="data row116 col4" >-0.006000</td>
          <td id="T_78b05_row116_col5" class="data row116 col5" >0.039000</td>
          <td id="T_78b05_row116_col6" class="data row116 col6" >0.026500</td>
          <td id="T_78b05_row116_col7" class="data row116 col7" >-0.009400</td>
          <td id="T_78b05_row116_col8" class="data row116 col8" >0.002300</td>
          <td id="T_78b05_row116_col9" class="data row116 col9" >0.011700</td>
          <td id="T_78b05_row116_col10" class="data row116 col10" >0.035400</td>
          <td id="T_78b05_row116_col11" class="data row116 col11" >0.011900</td>
          <td id="T_78b05_row116_col12" class="data row116 col12" >0.039600</td>
          <td id="T_78b05_row116_col13" class="data row116 col13" >0.030400</td>
          <td id="T_78b05_row116_col14" class="data row116 col14" >0.007600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row117" class="row_heading level0 row117" >118</th>
          <td id="T_78b05_row117_col0" class="data row117 col0" >None</td>
          <td id="T_78b05_row117_col1" class="data row117 col1" >0.034900</td>
          <td id="T_78b05_row117_col2" class="data row117 col2" >0.022400</td>
          <td id="T_78b05_row117_col3" class="data row117 col3" >-0.001600</td>
          <td id="T_78b05_row117_col4" class="data row117 col4" >-0.007600</td>
          <td id="T_78b05_row117_col5" class="data row117 col5" >-0.045400</td>
          <td id="T_78b05_row117_col6" class="data row117 col6" >-0.019200</td>
          <td id="T_78b05_row117_col7" class="data row117 col7" >-0.016000</td>
          <td id="T_78b05_row117_col8" class="data row117 col8" >0.010600</td>
          <td id="T_78b05_row117_col9" class="data row117 col9" >0.052200</td>
          <td id="T_78b05_row117_col10" class="data row117 col10" >0.032300</td>
          <td id="T_78b05_row117_col11" class="data row117 col11" >0.010300</td>
          <td id="T_78b05_row117_col12" class="data row117 col12" >0.044700</td>
          <td id="T_78b05_row117_col13" class="data row117 col13" >0.015300</td>
          <td id="T_78b05_row117_col14" class="data row117 col14" >0.014300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row118" class="row_heading level0 row118" >119</th>
          <td id="T_78b05_row118_col0" class="data row118 col0" >None</td>
          <td id="T_78b05_row118_col1" class="data row118 col1" >0.043000</td>
          <td id="T_78b05_row118_col2" class="data row118 col2" >0.002700</td>
          <td id="T_78b05_row118_col3" class="data row118 col3" >0.021300</td>
          <td id="T_78b05_row118_col4" class="data row118 col4" >-0.033200</td>
          <td id="T_78b05_row118_col5" class="data row118 col5" >0.021500</td>
          <td id="T_78b05_row118_col6" class="data row118 col6" >0.041100</td>
          <td id="T_78b05_row118_col7" class="data row118 col7" >-0.044800</td>
          <td id="T_78b05_row118_col8" class="data row118 col8" >0.002600</td>
          <td id="T_78b05_row118_col9" class="data row118 col9" >0.032500</td>
          <td id="T_78b05_row118_col10" class="data row118 col10" >0.009400</td>
          <td id="T_78b05_row118_col11" class="data row118 col11" >0.015300</td>
          <td id="T_78b05_row118_col12" class="data row118 col12" >0.022200</td>
          <td id="T_78b05_row118_col13" class="data row118 col13" >0.045000</td>
          <td id="T_78b05_row118_col14" class="data row118 col14" >0.043100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row119" class="row_heading level0 row119" >120</th>
          <td id="T_78b05_row119_col0" class="data row119 col0" >None</td>
          <td id="T_78b05_row119_col1" class="data row119 col1" >0.036300</td>
          <td id="T_78b05_row119_col2" class="data row119 col2" >-0.037300</td>
          <td id="T_78b05_row119_col3" class="data row119 col3" >-0.030600</td>
          <td id="T_78b05_row119_col4" class="data row119 col4" >0.021200</td>
          <td id="T_78b05_row119_col5" class="data row119 col5" >0.033700</td>
          <td id="T_78b05_row119_col6" class="data row119 col6" >0.083500</td>
          <td id="T_78b05_row119_col7" class="data row119 col7" >-0.013300</td>
          <td id="T_78b05_row119_col8" class="data row119 col8" >0.009200</td>
          <td id="T_78b05_row119_col9" class="data row119 col9" >0.007400</td>
          <td id="T_78b05_row119_col10" class="data row119 col10" >0.061400</td>
          <td id="T_78b05_row119_col11" class="data row119 col11" >0.039200</td>
          <td id="T_78b05_row119_col12" class="data row119 col12" >0.034400</td>
          <td id="T_78b05_row119_col13" class="data row119 col13" >0.087400</td>
          <td id="T_78b05_row119_col14" class="data row119 col14" >0.011600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row120" class="row_heading level0 row120" >121</th>
          <td id="T_78b05_row120_col0" class="data row120 col0" >None</td>
          <td id="T_78b05_row120_col1" class="data row120 col1" >0.031600</td>
          <td id="T_78b05_row120_col2" class="data row120 col2" >0.019800</td>
          <td id="T_78b05_row120_col3" class="data row120 col3" >-0.035600</td>
          <td id="T_78b05_row120_col4" class="data row120 col4" >-0.009800</td>
          <td id="T_78b05_row120_col5" class="data row120 col5" >-0.015500</td>
          <td id="T_78b05_row120_col6" class="data row120 col6" >0.012200</td>
          <td id="T_78b05_row120_col7" class="data row120 col7" >-0.039300</td>
          <td id="T_78b05_row120_col8" class="data row120 col8" >0.013900</td>
          <td id="T_78b05_row120_col9" class="data row120 col9" >0.049700</td>
          <td id="T_78b05_row120_col10" class="data row120 col10" >0.066300</td>
          <td id="T_78b05_row120_col11" class="data row120 col11" >0.008200</td>
          <td id="T_78b05_row120_col12" class="data row120 col12" >0.014800</td>
          <td id="T_78b05_row120_col13" class="data row120 col13" >0.016100</td>
          <td id="T_78b05_row120_col14" class="data row120 col14" >0.037600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row121" class="row_heading level0 row121" >122</th>
          <td id="T_78b05_row121_col0" class="data row121 col0" >None</td>
          <td id="T_78b05_row121_col1" class="data row121 col1" >0.039600</td>
          <td id="T_78b05_row121_col2" class="data row121 col2" >0.045100</td>
          <td id="T_78b05_row121_col3" class="data row121 col3" >0.052400</td>
          <td id="T_78b05_row121_col4" class="data row121 col4" >-0.020300</td>
          <td id="T_78b05_row121_col5" class="data row121 col5" >-0.019900</td>
          <td id="T_78b05_row121_col6" class="data row121 col6" >-0.023600</td>
          <td id="T_78b05_row121_col7" class="data row121 col7" >0.020800</td>
          <td id="T_78b05_row121_col8" class="data row121 col8" >0.005900</td>
          <td id="T_78b05_row121_col9" class="data row121 col9" >0.075000</td>
          <td id="T_78b05_row121_col10" class="data row121 col10" >0.021600</td>
          <td id="T_78b05_row121_col11" class="data row121 col11" >0.002400</td>
          <td id="T_78b05_row121_col12" class="data row121 col12" >0.019200</td>
          <td id="T_78b05_row121_col13" class="data row121 col13" >0.019700</td>
          <td id="T_78b05_row121_col14" class="data row121 col14" >0.022600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row122" class="row_heading level0 row122" >123</th>
          <td id="T_78b05_row122_col0" class="data row122 col0" >None</td>
          <td id="T_78b05_row122_col1" class="data row122 col1" >0.035400</td>
          <td id="T_78b05_row122_col2" class="data row122 col2" >0.045200</td>
          <td id="T_78b05_row122_col3" class="data row122 col3" >-0.013200</td>
          <td id="T_78b05_row122_col4" class="data row122 col4" >-0.042300</td>
          <td id="T_78b05_row122_col5" class="data row122 col5" >-0.024800</td>
          <td id="T_78b05_row122_col6" class="data row122 col6" >-0.030500</td>
          <td id="T_78b05_row122_col7" class="data row122 col7" >-0.008300</td>
          <td id="T_78b05_row122_col8" class="data row122 col8" >0.010100</td>
          <td id="T_78b05_row122_col9" class="data row122 col9" >0.075100</td>
          <td id="T_78b05_row122_col10" class="data row122 col10" >0.044000</td>
          <td id="T_78b05_row122_col11" class="data row122 col11" >0.024400</td>
          <td id="T_78b05_row122_col12" class="data row122 col12" >0.024100</td>
          <td id="T_78b05_row122_col13" class="data row122 col13" >0.026600</td>
          <td id="T_78b05_row122_col14" class="data row122 col14" >0.006500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row123" class="row_heading level0 row123" >124</th>
          <td id="T_78b05_row123_col0" class="data row123 col0" >None</td>
          <td id="T_78b05_row123_col1" class="data row123 col1" >0.033200</td>
          <td id="T_78b05_row123_col2" class="data row123 col2" >0.047900</td>
          <td id="T_78b05_row123_col3" class="data row123 col3" >-0.006700</td>
          <td id="T_78b05_row123_col4" class="data row123 col4" >-0.028700</td>
          <td id="T_78b05_row123_col5" class="data row123 col5" >-0.009700</td>
          <td id="T_78b05_row123_col6" class="data row123 col6" >0.052300</td>
          <td id="T_78b05_row123_col7" class="data row123 col7" >-0.092900</td>
          <td id="T_78b05_row123_col8" class="data row123 col8" >0.012400</td>
          <td id="T_78b05_row123_col9" class="data row123 col9" >0.077700</td>
          <td id="T_78b05_row123_col10" class="data row123 col10" >0.037500</td>
          <td id="T_78b05_row123_col11" class="data row123 col11" >0.010800</td>
          <td id="T_78b05_row123_col12" class="data row123 col12" >0.009100</td>
          <td id="T_78b05_row123_col13" class="data row123 col13" >0.056200</td>
          <td id="T_78b05_row123_col14" class="data row123 col14" >0.091100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row124" class="row_heading level0 row124" >125</th>
          <td id="T_78b05_row124_col0" class="data row124 col0" >None</td>
          <td id="T_78b05_row124_col1" class="data row124 col1" >0.046300</td>
          <td id="T_78b05_row124_col2" class="data row124 col2" >-0.038800</td>
          <td id="T_78b05_row124_col3" class="data row124 col3" >0.002100</td>
          <td id="T_78b05_row124_col4" class="data row124 col4" >0.073500</td>
          <td id="T_78b05_row124_col5" class="data row124 col5" >0.055700</td>
          <td id="T_78b05_row124_col6" class="data row124 col6" >-0.008100</td>
          <td id="T_78b05_row124_col7" class="data row124 col7" >-0.020600</td>
          <td id="T_78b05_row124_col8" class="data row124 col8" >0.000800</td>
          <td id="T_78b05_row124_col9" class="data row124 col9" >0.009000</td>
          <td id="T_78b05_row124_col10" class="data row124 col10" >0.028600</td>
          <td id="T_78b05_row124_col11" class="data row124 col11" >0.091500</td>
          <td id="T_78b05_row124_col12" class="data row124 col12" >0.056400</td>
          <td id="T_78b05_row124_col13" class="data row124 col13" >0.004200</td>
          <td id="T_78b05_row124_col14" class="data row124 col14" >0.018900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row125" class="row_heading level0 row125" >126</th>
          <td id="T_78b05_row125_col0" class="data row125 col0" >None</td>
          <td id="T_78b05_row125_col1" class="data row125 col1" >0.041800</td>
          <td id="T_78b05_row125_col2" class="data row125 col2" >-0.004100</td>
          <td id="T_78b05_row125_col3" class="data row125 col3" >-0.019000</td>
          <td id="T_78b05_row125_col4" class="data row125 col4" >0.014800</td>
          <td id="T_78b05_row125_col5" class="data row125 col5" >-0.033700</td>
          <td id="T_78b05_row125_col6" class="data row125 col6" >0.035500</td>
          <td id="T_78b05_row125_col7" class="data row125 col7" >0.035700</td>
          <td id="T_78b05_row125_col8" class="data row125 col8" >0.003700</td>
          <td id="T_78b05_row125_col9" class="data row125 col9" >0.025800</td>
          <td id="T_78b05_row125_col10" class="data row125 col10" >0.049800</td>
          <td id="T_78b05_row125_col11" class="data row125 col11" >0.032700</td>
          <td id="T_78b05_row125_col12" class="data row125 col12" >0.033000</td>
          <td id="T_78b05_row125_col13" class="data row125 col13" >0.039400</td>
          <td id="T_78b05_row125_col14" class="data row125 col14" >0.037400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row126" class="row_heading level0 row126" >127</th>
          <td id="T_78b05_row126_col0" class="data row126 col0" >None</td>
          <td id="T_78b05_row126_col1" class="data row126 col1" >0.036800</td>
          <td id="T_78b05_row126_col2" class="data row126 col2" >-0.037300</td>
          <td id="T_78b05_row126_col3" class="data row126 col3" >-0.035000</td>
          <td id="T_78b05_row126_col4" class="data row126 col4" >-0.005300</td>
          <td id="T_78b05_row126_col5" class="data row126 col5" >-0.056800</td>
          <td id="T_78b05_row126_col6" class="data row126 col6" >-0.011400</td>
          <td id="T_78b05_row126_col7" class="data row126 col7" >-0.008100</td>
          <td id="T_78b05_row126_col8" class="data row126 col8" >0.008700</td>
          <td id="T_78b05_row126_col9" class="data row126 col9" >0.007500</td>
          <td id="T_78b05_row126_col10" class="data row126 col10" >0.065700</td>
          <td id="T_78b05_row126_col11" class="data row126 col11" >0.012600</td>
          <td id="T_78b05_row126_col12" class="data row126 col12" >0.056100</td>
          <td id="T_78b05_row126_col13" class="data row126 col13" >0.007500</td>
          <td id="T_78b05_row126_col14" class="data row126 col14" >0.006300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row127" class="row_heading level0 row127" >128</th>
          <td id="T_78b05_row127_col0" class="data row127 col0" >None</td>
          <td id="T_78b05_row127_col1" class="data row127 col1" >0.033900</td>
          <td id="T_78b05_row127_col2" class="data row127 col2" >-0.005700</td>
          <td id="T_78b05_row127_col3" class="data row127 col3" >-0.055600</td>
          <td id="T_78b05_row127_col4" class="data row127 col4" >-0.003600</td>
          <td id="T_78b05_row127_col5" class="data row127 col5" >0.063300</td>
          <td id="T_78b05_row127_col6" class="data row127 col6" >0.011000</td>
          <td id="T_78b05_row127_col7" class="data row127 col7" >0.002700</td>
          <td id="T_78b05_row127_col8" class="data row127 col8" >0.011700</td>
          <td id="T_78b05_row127_col9" class="data row127 col9" >0.024100</td>
          <td id="T_78b05_row127_col10" class="data row127 col10" >0.086400</td>
          <td id="T_78b05_row127_col11" class="data row127 col11" >0.014300</td>
          <td id="T_78b05_row127_col12" class="data row127 col12" >0.063900</td>
          <td id="T_78b05_row127_col13" class="data row127 col13" >0.014900</td>
          <td id="T_78b05_row127_col14" class="data row127 col14" >0.004400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row128" class="row_heading level0 row128" >129</th>
          <td id="T_78b05_row128_col0" class="data row128 col0" >None</td>
          <td id="T_78b05_row128_col1" class="data row128 col1" >0.032000</td>
          <td id="T_78b05_row128_col2" class="data row128 col2" >0.055500</td>
          <td id="T_78b05_row128_col3" class="data row128 col3" >0.013400</td>
          <td id="T_78b05_row128_col4" class="data row128 col4" >-0.003400</td>
          <td id="T_78b05_row128_col5" class="data row128 col5" >-0.055400</td>
          <td id="T_78b05_row128_col6" class="data row128 col6" >-0.023800</td>
          <td id="T_78b05_row128_col7" class="data row128 col7" >-0.027000</td>
          <td id="T_78b05_row128_col8" class="data row128 col8" >0.013500</td>
          <td id="T_78b05_row128_col9" class="data row128 col9" >0.085300</td>
          <td id="T_78b05_row128_col10" class="data row128 col10" >0.017300</td>
          <td id="T_78b05_row128_col11" class="data row128 col11" >0.014500</td>
          <td id="T_78b05_row128_col12" class="data row128 col12" >0.054700</td>
          <td id="T_78b05_row128_col13" class="data row128 col13" >0.019900</td>
          <td id="T_78b05_row128_col14" class="data row128 col14" >0.025300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row129" class="row_heading level0 row129" >130</th>
          <td id="T_78b05_row129_col0" class="data row129 col0" >None</td>
          <td id="T_78b05_row129_col1" class="data row129 col1" >0.038000</td>
          <td id="T_78b05_row129_col2" class="data row129 col2" >-0.025900</td>
          <td id="T_78b05_row129_col3" class="data row129 col3" >-0.027500</td>
          <td id="T_78b05_row129_col4" class="data row129 col4" >-0.022200</td>
          <td id="T_78b05_row129_col5" class="data row129 col5" >-0.018000</td>
          <td id="T_78b05_row129_col6" class="data row129 col6" >-0.012500</td>
          <td id="T_78b05_row129_col7" class="data row129 col7" >-0.005200</td>
          <td id="T_78b05_row129_col8" class="data row129 col8" >0.007500</td>
          <td id="T_78b05_row129_col9" class="data row129 col9" >0.004000</td>
          <td id="T_78b05_row129_col10" class="data row129 col10" >0.058300</td>
          <td id="T_78b05_row129_col11" class="data row129 col11" >0.004300</td>
          <td id="T_78b05_row129_col12" class="data row129 col12" >0.017300</td>
          <td id="T_78b05_row129_col13" class="data row129 col13" >0.008600</td>
          <td id="T_78b05_row129_col14" class="data row129 col14" >0.003400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row130" class="row_heading level0 row130" >131</th>
          <td id="T_78b05_row130_col0" class="data row130 col0" >None</td>
          <td id="T_78b05_row130_col1" class="data row130 col1" >0.030100</td>
          <td id="T_78b05_row130_col2" class="data row130 col2" >-0.024300</td>
          <td id="T_78b05_row130_col3" class="data row130 col3" >-0.037800</td>
          <td id="T_78b05_row130_col4" class="data row130 col4" >0.013400</td>
          <td id="T_78b05_row130_col5" class="data row130 col5" >-0.038600</td>
          <td id="T_78b05_row130_col6" class="data row130 col6" >-0.015700</td>
          <td id="T_78b05_row130_col7" class="data row130 col7" >-0.051200</td>
          <td id="T_78b05_row130_col8" class="data row130 col8" >0.015500</td>
          <td id="T_78b05_row130_col9" class="data row130 col9" >0.005500</td>
          <td id="T_78b05_row130_col10" class="data row130 col10" >0.068500</td>
          <td id="T_78b05_row130_col11" class="data row130 col11" >0.031300</td>
          <td id="T_78b05_row130_col12" class="data row130 col12" >0.037900</td>
          <td id="T_78b05_row130_col13" class="data row130 col13" >0.011800</td>
          <td id="T_78b05_row130_col14" class="data row130 col14" >0.049500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row131" class="row_heading level0 row131" >132</th>
          <td id="T_78b05_row131_col0" class="data row131 col0" >None</td>
          <td id="T_78b05_row131_col1" class="data row131 col1" >0.042800</td>
          <td id="T_78b05_row131_col2" class="data row131 col2" >0.019800</td>
          <td id="T_78b05_row131_col3" class="data row131 col3" >0.044400</td>
          <td id="T_78b05_row131_col4" class="data row131 col4" >0.092500</td>
          <td id="T_78b05_row131_col5" class="data row131 col5" >0.019800</td>
          <td id="T_78b05_row131_col6" class="data row131 col6" >0.031000</td>
          <td id="T_78b05_row131_col7" class="data row131 col7" >0.040100</td>
          <td id="T_78b05_row131_col8" class="data row131 col8" >0.002700</td>
          <td id="T_78b05_row131_col9" class="data row131 col9" >0.049600</td>
          <td id="T_78b05_row131_col10" class="data row131 col10" >0.013600</td>
          <td id="T_78b05_row131_col11" class="data row131 col11" >0.110400</td>
          <td id="T_78b05_row131_col12" class="data row131 col12" >0.020500</td>
          <td id="T_78b05_row131_col13" class="data row131 col13" >0.034900</td>
          <td id="T_78b05_row131_col14" class="data row131 col14" >0.041800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row132" class="row_heading level0 row132" >133</th>
          <td id="T_78b05_row132_col0" class="data row132 col0" >None</td>
          <td id="T_78b05_row132_col1" class="data row132 col1" >0.036100</td>
          <td id="T_78b05_row132_col2" class="data row132 col2" >-0.062700</td>
          <td id="T_78b05_row132_col3" class="data row132 col3" >-0.053400</td>
          <td id="T_78b05_row132_col4" class="data row132 col4" >0.004700</td>
          <td id="T_78b05_row132_col5" class="data row132 col5" >-0.005500</td>
          <td id="T_78b05_row132_col6" class="data row132 col6" >0.026800</td>
          <td id="T_78b05_row132_col7" class="data row132 col7" >-0.017400</td>
          <td id="T_78b05_row132_col8" class="data row132 col8" >0.009500</td>
          <td id="T_78b05_row132_col9" class="data row132 col9" >0.032900</td>
          <td id="T_78b05_row132_col10" class="data row132 col10" >0.084100</td>
          <td id="T_78b05_row132_col11" class="data row132 col11" >0.022600</td>
          <td id="T_78b05_row132_col12" class="data row132 col12" >0.004900</td>
          <td id="T_78b05_row132_col13" class="data row132 col13" >0.030700</td>
          <td id="T_78b05_row132_col14" class="data row132 col14" >0.015600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row133" class="row_heading level0 row133" >134</th>
          <td id="T_78b05_row133_col0" class="data row133 col0" >None</td>
          <td id="T_78b05_row133_col1" class="data row133 col1" >0.035800</td>
          <td id="T_78b05_row133_col2" class="data row133 col2" >0.005500</td>
          <td id="T_78b05_row133_col3" class="data row133 col3" >0.000900</td>
          <td id="T_78b05_row133_col4" class="data row133 col4" >0.087900</td>
          <td id="T_78b05_row133_col5" class="data row133 col5" >-0.005600</td>
          <td id="T_78b05_row133_col6" class="data row133 col6" >-0.052700</td>
          <td id="T_78b05_row133_col7" class="data row133 col7" >0.015000</td>
          <td id="T_78b05_row133_col8" class="data row133 col8" >0.009700</td>
          <td id="T_78b05_row133_col9" class="data row133 col9" >0.035300</td>
          <td id="T_78b05_row133_col10" class="data row133 col10" >0.029900</td>
          <td id="T_78b05_row133_col11" class="data row133 col11" >0.105900</td>
          <td id="T_78b05_row133_col12" class="data row133 col12" >0.004900</td>
          <td id="T_78b05_row133_col13" class="data row133 col13" >0.048800</td>
          <td id="T_78b05_row133_col14" class="data row133 col14" >0.016700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row134" class="row_heading level0 row134" >135</th>
          <td id="T_78b05_row134_col0" class="data row134 col0" >None</td>
          <td id="T_78b05_row134_col1" class="data row134 col1" >0.035900</td>
          <td id="T_78b05_row134_col2" class="data row134 col2" >-0.003500</td>
          <td id="T_78b05_row134_col3" class="data row134 col3" >0.028300</td>
          <td id="T_78b05_row134_col4" class="data row134 col4" >0.097800</td>
          <td id="T_78b05_row134_col5" class="data row134 col5" >-0.045100</td>
          <td id="T_78b05_row134_col6" class="data row134 col6" >-0.034000</td>
          <td id="T_78b05_row134_col7" class="data row134 col7" >-0.040500</td>
          <td id="T_78b05_row134_col8" class="data row134 col8" >0.009700</td>
          <td id="T_78b05_row134_col9" class="data row134 col9" >0.026300</td>
          <td id="T_78b05_row134_col10" class="data row134 col10" >0.002400</td>
          <td id="T_78b05_row134_col11" class="data row134 col11" >0.115700</td>
          <td id="T_78b05_row134_col12" class="data row134 col12" >0.044500</td>
          <td id="T_78b05_row134_col13" class="data row134 col13" >0.030100</td>
          <td id="T_78b05_row134_col14" class="data row134 col14" >0.038800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row135" class="row_heading level0 row135" >136</th>
          <td id="T_78b05_row135_col0" class="data row135 col0" >None</td>
          <td id="T_78b05_row135_col1" class="data row135 col1" >0.041800</td>
          <td id="T_78b05_row135_col2" class="data row135 col2" >-0.068900</td>
          <td id="T_78b05_row135_col3" class="data row135 col3" >0.002900</td>
          <td id="T_78b05_row135_col4" class="data row135 col4" >0.007400</td>
          <td id="T_78b05_row135_col5" class="data row135 col5" >0.000900</td>
          <td id="T_78b05_row135_col6" class="data row135 col6" >0.043200</td>
          <td id="T_78b05_row135_col7" class="data row135 col7" >-0.036500</td>
          <td id="T_78b05_row135_col8" class="data row135 col8" >0.003700</td>
          <td id="T_78b05_row135_col9" class="data row135 col9" >0.039100</td>
          <td id="T_78b05_row135_col10" class="data row135 col10" >0.027900</td>
          <td id="T_78b05_row135_col11" class="data row135 col11" >0.025400</td>
          <td id="T_78b05_row135_col12" class="data row135 col12" >0.001500</td>
          <td id="T_78b05_row135_col13" class="data row135 col13" >0.047100</td>
          <td id="T_78b05_row135_col14" class="data row135 col14" >0.034700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row136" class="row_heading level0 row136" >137</th>
          <td id="T_78b05_row136_col0" class="data row136 col0" >None</td>
          <td id="T_78b05_row136_col1" class="data row136 col1" >0.043800</td>
          <td id="T_78b05_row136_col2" class="data row136 col2" >-0.007500</td>
          <td id="T_78b05_row136_col3" class="data row136 col3" >0.039000</td>
          <td id="T_78b05_row136_col4" class="data row136 col4" >0.063900</td>
          <td id="T_78b05_row136_col5" class="data row136 col5" >-0.015600</td>
          <td id="T_78b05_row136_col6" class="data row136 col6" >-0.026800</td>
          <td id="T_78b05_row136_col7" class="data row136 col7" >-0.024000</td>
          <td id="T_78b05_row136_col8" class="data row136 col8" >0.001800</td>
          <td id="T_78b05_row136_col9" class="data row136 col9" >0.022300</td>
          <td id="T_78b05_row136_col10" class="data row136 col10" >0.008300</td>
          <td id="T_78b05_row136_col11" class="data row136 col11" >0.081800</td>
          <td id="T_78b05_row136_col12" class="data row136 col12" >0.015000</td>
          <td id="T_78b05_row136_col13" class="data row136 col13" >0.022900</td>
          <td id="T_78b05_row136_col14" class="data row136 col14" >0.022300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row137" class="row_heading level0 row137" >138</th>
          <td id="T_78b05_row137_col0" class="data row137 col0" >None</td>
          <td id="T_78b05_row137_col1" class="data row137 col1" >0.039200</td>
          <td id="T_78b05_row137_col2" class="data row137 col2" >-0.032500</td>
          <td id="T_78b05_row137_col3" class="data row137 col3" >-0.006500</td>
          <td id="T_78b05_row137_col4" class="data row137 col4" >-0.007500</td>
          <td id="T_78b05_row137_col5" class="data row137 col5" >-0.049300</td>
          <td id="T_78b05_row137_col6" class="data row137 col6" >0.008100</td>
          <td id="T_78b05_row137_col7" class="data row137 col7" >0.008900</td>
          <td id="T_78b05_row137_col8" class="data row137 col8" >0.006300</td>
          <td id="T_78b05_row137_col9" class="data row137 col9" >0.002700</td>
          <td id="T_78b05_row137_col10" class="data row137 col10" >0.037200</td>
          <td id="T_78b05_row137_col11" class="data row137 col11" >0.010400</td>
          <td id="T_78b05_row137_col12" class="data row137 col12" >0.048700</td>
          <td id="T_78b05_row137_col13" class="data row137 col13" >0.012000</td>
          <td id="T_78b05_row137_col14" class="data row137 col14" >0.010600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row138" class="row_heading level0 row138" >139</th>
          <td id="T_78b05_row138_col0" class="data row138 col0" >None</td>
          <td id="T_78b05_row138_col1" class="data row138 col1" >0.039700</td>
          <td id="T_78b05_row138_col2" class="data row138 col2" >0.056600</td>
          <td id="T_78b05_row138_col3" class="data row138 col3" >-0.004500</td>
          <td id="T_78b05_row138_col4" class="data row138 col4" >-0.018300</td>
          <td id="T_78b05_row138_col5" class="data row138 col5" >0.033700</td>
          <td id="T_78b05_row138_col6" class="data row138 col6" >0.001100</td>
          <td id="T_78b05_row138_col7" class="data row138 col7" >-0.019300</td>
          <td id="T_78b05_row138_col8" class="data row138 col8" >0.005900</td>
          <td id="T_78b05_row138_col9" class="data row138 col9" >0.086400</td>
          <td id="T_78b05_row138_col10" class="data row138 col10" >0.035300</td>
          <td id="T_78b05_row138_col11" class="data row138 col11" >0.000300</td>
          <td id="T_78b05_row138_col12" class="data row138 col12" >0.034400</td>
          <td id="T_78b05_row138_col13" class="data row138 col13" >0.005000</td>
          <td id="T_78b05_row138_col14" class="data row138 col14" >0.017600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row139" class="row_heading level0 row139" >140</th>
          <td id="T_78b05_row139_col0" class="data row139 col0" >None</td>
          <td id="T_78b05_row139_col1" class="data row139 col1" >0.041400</td>
          <td id="T_78b05_row139_col2" class="data row139 col2" >-0.027200</td>
          <td id="T_78b05_row139_col3" class="data row139 col3" >0.016200</td>
          <td id="T_78b05_row139_col4" class="data row139 col4" >0.034600</td>
          <td id="T_78b05_row139_col5" class="data row139 col5" >-0.064600</td>
          <td id="T_78b05_row139_col6" class="data row139 col6" >-0.027100</td>
          <td id="T_78b05_row139_col7" class="data row139 col7" >0.012700</td>
          <td id="T_78b05_row139_col8" class="data row139 col8" >0.004200</td>
          <td id="T_78b05_row139_col9" class="data row139 col9" >0.002700</td>
          <td id="T_78b05_row139_col10" class="data row139 col10" >0.014500</td>
          <td id="T_78b05_row139_col11" class="data row139 col11" >0.052500</td>
          <td id="T_78b05_row139_col12" class="data row139 col12" >0.064000</td>
          <td id="T_78b05_row139_col13" class="data row139 col13" >0.023200</td>
          <td id="T_78b05_row139_col14" class="data row139 col14" >0.014500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row140" class="row_heading level0 row140" >141</th>
          <td id="T_78b05_row140_col0" class="data row140 col0" >None</td>
          <td id="T_78b05_row140_col1" class="data row140 col1" >0.039800</td>
          <td id="T_78b05_row140_col2" class="data row140 col2" >-0.047600</td>
          <td id="T_78b05_row140_col3" class="data row140 col3" >-0.018000</td>
          <td id="T_78b05_row140_col4" class="data row140 col4" >0.038200</td>
          <td id="T_78b05_row140_col5" class="data row140 col5" >-0.018300</td>
          <td id="T_78b05_row140_col6" class="data row140 col6" >-0.013400</td>
          <td id="T_78b05_row140_col7" class="data row140 col7" >-0.067300</td>
          <td id="T_78b05_row140_col8" class="data row140 col8" >0.005700</td>
          <td id="T_78b05_row140_col9" class="data row140 col9" >0.017800</td>
          <td id="T_78b05_row140_col10" class="data row140 col10" >0.048700</td>
          <td id="T_78b05_row140_col11" class="data row140 col11" >0.056100</td>
          <td id="T_78b05_row140_col12" class="data row140 col12" >0.017700</td>
          <td id="T_78b05_row140_col13" class="data row140 col13" >0.009500</td>
          <td id="T_78b05_row140_col14" class="data row140 col14" >0.065600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row141" class="row_heading level0 row141" >142</th>
          <td id="T_78b05_row141_col0" class="data row141 col0" >None</td>
          <td id="T_78b05_row141_col1" class="data row141 col1" >0.038400</td>
          <td id="T_78b05_row141_col2" class="data row141 col2" >0.045200</td>
          <td id="T_78b05_row141_col3" class="data row141 col3" >0.018900</td>
          <td id="T_78b05_row141_col4" class="data row141 col4" >-0.026400</td>
          <td id="T_78b05_row141_col5" class="data row141 col5" >-0.023200</td>
          <td id="T_78b05_row141_col6" class="data row141 col6" >-0.003600</td>
          <td id="T_78b05_row141_col7" class="data row141 col7" >-0.009900</td>
          <td id="T_78b05_row141_col8" class="data row141 col8" >0.007100</td>
          <td id="T_78b05_row141_col9" class="data row141 col9" >0.075000</td>
          <td id="T_78b05_row141_col10" class="data row141 col10" >0.011800</td>
          <td id="T_78b05_row141_col11" class="data row141 col11" >0.008500</td>
          <td id="T_78b05_row141_col12" class="data row141 col12" >0.022600</td>
          <td id="T_78b05_row141_col13" class="data row141 col13" >0.000300</td>
          <td id="T_78b05_row141_col14" class="data row141 col14" >0.008100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row142" class="row_heading level0 row142" >143</th>
          <td id="T_78b05_row142_col0" class="data row142 col0" >None</td>
          <td id="T_78b05_row142_col1" class="data row142 col1" >0.044400</td>
          <td id="T_78b05_row142_col2" class="data row142 col2" >-0.050800</td>
          <td id="T_78b05_row142_col3" class="data row142 col3" >0.018700</td>
          <td id="T_78b05_row142_col4" class="data row142 col4" >-0.006000</td>
          <td id="T_78b05_row142_col5" class="data row142 col5" >-0.022100</td>
          <td id="T_78b05_row142_col6" class="data row142 col6" >-0.036300</td>
          <td id="T_78b05_row142_col7" class="data row142 col7" >-0.015000</td>
          <td id="T_78b05_row142_col8" class="data row142 col8" >0.001200</td>
          <td id="T_78b05_row142_col9" class="data row142 col9" >0.021000</td>
          <td id="T_78b05_row142_col10" class="data row142 col10" >0.012100</td>
          <td id="T_78b05_row142_col11" class="data row142 col11" >0.011900</td>
          <td id="T_78b05_row142_col12" class="data row142 col12" >0.021400</td>
          <td id="T_78b05_row142_col13" class="data row142 col13" >0.032400</td>
          <td id="T_78b05_row142_col14" class="data row142 col14" >0.013300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row143" class="row_heading level0 row143" >144</th>
          <td id="T_78b05_row143_col0" class="data row143 col0" >None</td>
          <td id="T_78b05_row143_col1" class="data row143 col1" >0.034100</td>
          <td id="T_78b05_row143_col2" class="data row143 col2" >-0.006200</td>
          <td id="T_78b05_row143_col3" class="data row143 col3" >-0.073200</td>
          <td id="T_78b05_row143_col4" class="data row143 col4" >0.000700</td>
          <td id="T_78b05_row143_col5" class="data row143 col5" >0.027700</td>
          <td id="T_78b05_row143_col6" class="data row143 col6" >0.014200</td>
          <td id="T_78b05_row143_col7" class="data row143 col7" >0.012900</td>
          <td id="T_78b05_row143_col8" class="data row143 col8" >0.011500</td>
          <td id="T_78b05_row143_col9" class="data row143 col9" >0.023700</td>
          <td id="T_78b05_row143_col10" class="data row143 col10" >0.103900</td>
          <td id="T_78b05_row143_col11" class="data row143 col11" >0.018600</td>
          <td id="T_78b05_row143_col12" class="data row143 col12" >0.028300</td>
          <td id="T_78b05_row143_col13" class="data row143 col13" >0.018100</td>
          <td id="T_78b05_row143_col14" class="data row143 col14" >0.014600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row144" class="row_heading level0 row144" >145</th>
          <td id="T_78b05_row144_col0" class="data row144 col0" >None</td>
          <td id="T_78b05_row144_col1" class="data row144 col1" >0.041900</td>
          <td id="T_78b05_row144_col2" class="data row144 col2" >-0.009000</td>
          <td id="T_78b05_row144_col3" class="data row144 col3" >0.003400</td>
          <td id="T_78b05_row144_col4" class="data row144 col4" >-0.016800</td>
          <td id="T_78b05_row144_col5" class="data row144 col5" >-0.013100</td>
          <td id="T_78b05_row144_col6" class="data row144 col6" >-0.019400</td>
          <td id="T_78b05_row144_col7" class="data row144 col7" >-0.033200</td>
          <td id="T_78b05_row144_col8" class="data row144 col8" >0.003700</td>
          <td id="T_78b05_row144_col9" class="data row144 col9" >0.020800</td>
          <td id="T_78b05_row144_col10" class="data row144 col10" >0.027400</td>
          <td id="T_78b05_row144_col11" class="data row144 col11" >0.001100</td>
          <td id="T_78b05_row144_col12" class="data row144 col12" >0.012400</td>
          <td id="T_78b05_row144_col13" class="data row144 col13" >0.015500</td>
          <td id="T_78b05_row144_col14" class="data row144 col14" >0.031400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row145" class="row_heading level0 row145" >146</th>
          <td id="T_78b05_row145_col0" class="data row145 col0" >None</td>
          <td id="T_78b05_row145_col1" class="data row145 col1" >0.032300</td>
          <td id="T_78b05_row145_col2" class="data row145 col2" >-0.022500</td>
          <td id="T_78b05_row145_col3" class="data row145 col3" >-0.050200</td>
          <td id="T_78b05_row145_col4" class="data row145 col4" >0.001900</td>
          <td id="T_78b05_row145_col5" class="data row145 col5" >0.024000</td>
          <td id="T_78b05_row145_col6" class="data row145 col6" >0.010300</td>
          <td id="T_78b05_row145_col7" class="data row145 col7" >0.070200</td>
          <td id="T_78b05_row145_col8" class="data row145 col8" >0.013200</td>
          <td id="T_78b05_row145_col9" class="data row145 col9" >0.007300</td>
          <td id="T_78b05_row145_col10" class="data row145 col10" >0.080900</td>
          <td id="T_78b05_row145_col11" class="data row145 col11" >0.019900</td>
          <td id="T_78b05_row145_col12" class="data row145 col12" >0.024600</td>
          <td id="T_78b05_row145_col13" class="data row145 col13" >0.014200</td>
          <td id="T_78b05_row145_col14" class="data row145 col14" >0.072000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row146" class="row_heading level0 row146" >147</th>
          <td id="T_78b05_row146_col0" class="data row146 col0" >None</td>
          <td id="T_78b05_row146_col1" class="data row146 col1" >0.029000</td>
          <td id="T_78b05_row146_col2" class="data row146 col2" >0.021100</td>
          <td id="T_78b05_row146_col3" class="data row146 col3" >-0.035300</td>
          <td id="T_78b05_row146_col4" class="data row146 col4" >-0.039600</td>
          <td id="T_78b05_row146_col5" class="data row146 col5" >-0.009500</td>
          <td id="T_78b05_row146_col6" class="data row146 col6" >-0.024000</td>
          <td id="T_78b05_row146_col7" class="data row146 col7" >-0.000400</td>
          <td id="T_78b05_row146_col8" class="data row146 col8" >0.016600</td>
          <td id="T_78b05_row146_col9" class="data row146 col9" >0.050900</td>
          <td id="T_78b05_row146_col10" class="data row146 col10" >0.066100</td>
          <td id="T_78b05_row146_col11" class="data row146 col11" >0.021700</td>
          <td id="T_78b05_row146_col12" class="data row146 col12" >0.008800</td>
          <td id="T_78b05_row146_col13" class="data row146 col13" >0.020100</td>
          <td id="T_78b05_row146_col14" class="data row146 col14" >0.001300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row147" class="row_heading level0 row147" >148</th>
          <td id="T_78b05_row147_col0" class="data row147 col0" >None</td>
          <td id="T_78b05_row147_col1" class="data row147 col1" >0.037500</td>
          <td id="T_78b05_row147_col2" class="data row147 col2" >-0.035800</td>
          <td id="T_78b05_row147_col3" class="data row147 col3" >-0.031100</td>
          <td id="T_78b05_row147_col4" class="data row147 col4" >0.022800</td>
          <td id="T_78b05_row147_col5" class="data row147 col5" >-0.033400</td>
          <td id="T_78b05_row147_col6" class="data row147 col6" >0.012500</td>
          <td id="T_78b05_row147_col7" class="data row147 col7" >0.007300</td>
          <td id="T_78b05_row147_col8" class="data row147 col8" >0.008100</td>
          <td id="T_78b05_row147_col9" class="data row147 col9" >0.005900</td>
          <td id="T_78b05_row147_col10" class="data row147 col10" >0.061800</td>
          <td id="T_78b05_row147_col11" class="data row147 col11" >0.040700</td>
          <td id="T_78b05_row147_col12" class="data row147 col12" >0.032700</td>
          <td id="T_78b05_row147_col13" class="data row147 col13" >0.016400</td>
          <td id="T_78b05_row147_col14" class="data row147 col14" >0.009100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row148" class="row_heading level0 row148" >149</th>
          <td id="T_78b05_row148_col0" class="data row148 col0" >PC1</td>
          <td id="T_78b05_row148_col1" class="data row148 col1" >0.026500</td>
          <td id="T_78b05_row148_col2" class="data row148 col2" >0.038000</td>
          <td id="T_78b05_row148_col3" class="data row148 col3" >-0.019100</td>
          <td id="T_78b05_row148_col4" class="data row148 col4" >0.045500</td>
          <td id="T_78b05_row148_col5" class="data row148 col5" >0.032200</td>
          <td id="T_78b05_row148_col6" class="data row148 col6" >-0.054000</td>
          <td id="T_78b05_row148_col7" class="data row148 col7" >-0.005300</td>
          <td id="T_78b05_row148_col8" class="data row148 col8" >0.019000</td>
          <td id="T_78b05_row148_col9" class="data row148 col9" >0.067800</td>
          <td id="T_78b05_row148_col10" class="data row148 col10" >0.049800</td>
          <td id="T_78b05_row148_col11" class="data row148 col11" >0.063400</td>
          <td id="T_78b05_row148_col12" class="data row148 col12" >0.032900</td>
          <td id="T_78b05_row148_col13" class="data row148 col13" >0.050100</td>
          <td id="T_78b05_row148_col14" class="data row148 col14" >0.003600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row149" class="row_heading level0 row149" >150</th>
          <td id="T_78b05_row149_col0" class="data row149 col0" >None</td>
          <td id="T_78b05_row149_col1" class="data row149 col1" >0.031500</td>
          <td id="T_78b05_row149_col2" class="data row149 col2" >0.018900</td>
          <td id="T_78b05_row149_col3" class="data row149 col3" >-0.061600</td>
          <td id="T_78b05_row149_col4" class="data row149 col4" >0.033800</td>
          <td id="T_78b05_row149_col5" class="data row149 col5" >-0.020800</td>
          <td id="T_78b05_row149_col6" class="data row149 col6" >-0.010500</td>
          <td id="T_78b05_row149_col7" class="data row149 col7" >0.062400</td>
          <td id="T_78b05_row149_col8" class="data row149 col8" >0.014100</td>
          <td id="T_78b05_row149_col9" class="data row149 col9" >0.048800</td>
          <td id="T_78b05_row149_col10" class="data row149 col10" >0.092300</td>
          <td id="T_78b05_row149_col11" class="data row149 col11" >0.051700</td>
          <td id="T_78b05_row149_col12" class="data row149 col12" >0.020100</td>
          <td id="T_78b05_row149_col13" class="data row149 col13" >0.006600</td>
          <td id="T_78b05_row149_col14" class="data row149 col14" >0.064100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row150" class="row_heading level0 row150" >151</th>
          <td id="T_78b05_row150_col0" class="data row150 col0" >None</td>
          <td id="T_78b05_row150_col1" class="data row150 col1" >0.040100</td>
          <td id="T_78b05_row150_col2" class="data row150 col2" >-0.028600</td>
          <td id="T_78b05_row150_col3" class="data row150 col3" >-0.000800</td>
          <td id="T_78b05_row150_col4" class="data row150 col4" >-0.010000</td>
          <td id="T_78b05_row150_col5" class="data row150 col5" >-0.088300</td>
          <td id="T_78b05_row150_col6" class="data row150 col6" >-0.032900</td>
          <td id="T_78b05_row150_col7" class="data row150 col7" >-0.022300</td>
          <td id="T_78b05_row150_col8" class="data row150 col8" >0.005500</td>
          <td id="T_78b05_row150_col9" class="data row150 col9" >0.001300</td>
          <td id="T_78b05_row150_col10" class="data row150 col10" >0.031600</td>
          <td id="T_78b05_row150_col11" class="data row150 col11" >0.007900</td>
          <td id="T_78b05_row150_col12" class="data row150 col12" >0.087600</td>
          <td id="T_78b05_row150_col13" class="data row150 col13" >0.029000</td>
          <td id="T_78b05_row150_col14" class="data row150 col14" >0.020500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row151" class="row_heading level0 row151" >152</th>
          <td id="T_78b05_row151_col0" class="data row151 col0" >None</td>
          <td id="T_78b05_row151_col1" class="data row151 col1" >0.039000</td>
          <td id="T_78b05_row151_col2" class="data row151 col2" >0.006300</td>
          <td id="T_78b05_row151_col3" class="data row151 col3" >-0.049900</td>
          <td id="T_78b05_row151_col4" class="data row151 col4" >-0.013200</td>
          <td id="T_78b05_row151_col5" class="data row151 col5" >0.032200</td>
          <td id="T_78b05_row151_col6" class="data row151 col6" >-0.017800</td>
          <td id="T_78b05_row151_col7" class="data row151 col7" >0.057800</td>
          <td id="T_78b05_row151_col8" class="data row151 col8" >0.006500</td>
          <td id="T_78b05_row151_col9" class="data row151 col9" >0.036200</td>
          <td id="T_78b05_row151_col10" class="data row151 col10" >0.080700</td>
          <td id="T_78b05_row151_col11" class="data row151 col11" >0.004700</td>
          <td id="T_78b05_row151_col12" class="data row151 col12" >0.032900</td>
          <td id="T_78b05_row151_col13" class="data row151 col13" >0.013900</td>
          <td id="T_78b05_row151_col14" class="data row151 col14" >0.059500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row152" class="row_heading level0 row152" >153</th>
          <td id="T_78b05_row152_col0" class="data row152 col0" >None</td>
          <td id="T_78b05_row152_col1" class="data row152 col1" >0.044500</td>
          <td id="T_78b05_row152_col2" class="data row152 col2" >-0.059000</td>
          <td id="T_78b05_row152_col3" class="data row152 col3" >0.011700</td>
          <td id="T_78b05_row152_col4" class="data row152 col4" >-0.059500</td>
          <td id="T_78b05_row152_col5" class="data row152 col5" >-0.013700</td>
          <td id="T_78b05_row152_col6" class="data row152 col6" >0.015100</td>
          <td id="T_78b05_row152_col7" class="data row152 col7" >0.040100</td>
          <td id="T_78b05_row152_col8" class="data row152 col8" >0.001100</td>
          <td id="T_78b05_row152_col9" class="data row152 col9" >0.029200</td>
          <td id="T_78b05_row152_col10" class="data row152 col10" >0.019000</td>
          <td id="T_78b05_row152_col11" class="data row152 col11" >0.041600</td>
          <td id="T_78b05_row152_col12" class="data row152 col12" >0.013000</td>
          <td id="T_78b05_row152_col13" class="data row152 col13" >0.019000</td>
          <td id="T_78b05_row152_col14" class="data row152 col14" >0.041900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row153" class="row_heading level0 row153" >154</th>
          <td id="T_78b05_row153_col0" class="data row153 col0" >None</td>
          <td id="T_78b05_row153_col1" class="data row153 col1" >0.040500</td>
          <td id="T_78b05_row153_col2" class="data row153 col2" >-0.004300</td>
          <td id="T_78b05_row153_col3" class="data row153 col3" >0.010700</td>
          <td id="T_78b05_row153_col4" class="data row153 col4" >0.032300</td>
          <td id="T_78b05_row153_col5" class="data row153 col5" >0.062900</td>
          <td id="T_78b05_row153_col6" class="data row153 col6" >-0.079700</td>
          <td id="T_78b05_row153_col7" class="data row153 col7" >-0.004800</td>
          <td id="T_78b05_row153_col8" class="data row153 col8" >0.005000</td>
          <td id="T_78b05_row153_col9" class="data row153 col9" >0.025500</td>
          <td id="T_78b05_row153_col10" class="data row153 col10" >0.020000</td>
          <td id="T_78b05_row153_col11" class="data row153 col11" >0.050200</td>
          <td id="T_78b05_row153_col12" class="data row153 col12" >0.063600</td>
          <td id="T_78b05_row153_col13" class="data row153 col13" >0.075800</td>
          <td id="T_78b05_row153_col14" class="data row153 col14" >0.003000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row154" class="row_heading level0 row154" >155</th>
          <td id="T_78b05_row154_col0" class="data row154 col0" >None</td>
          <td id="T_78b05_row154_col1" class="data row154 col1" >0.032600</td>
          <td id="T_78b05_row154_col2" class="data row154 col2" >0.014600</td>
          <td id="T_78b05_row154_col3" class="data row154 col3" >-0.048800</td>
          <td id="T_78b05_row154_col4" class="data row154 col4" >-0.011700</td>
          <td id="T_78b05_row154_col5" class="data row154 col5" >-0.033300</td>
          <td id="T_78b05_row154_col6" class="data row154 col6" >0.025100</td>
          <td id="T_78b05_row154_col7" class="data row154 col7" >0.003500</td>
          <td id="T_78b05_row154_col8" class="data row154 col8" >0.012900</td>
          <td id="T_78b05_row154_col9" class="data row154 col9" >0.044400</td>
          <td id="T_78b05_row154_col10" class="data row154 col10" >0.079600</td>
          <td id="T_78b05_row154_col11" class="data row154 col11" >0.006200</td>
          <td id="T_78b05_row154_col12" class="data row154 col12" >0.032700</td>
          <td id="T_78b05_row154_col13" class="data row154 col13" >0.029000</td>
          <td id="T_78b05_row154_col14" class="data row154 col14" >0.005200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row155" class="row_heading level0 row155" >156</th>
          <td id="T_78b05_row155_col0" class="data row155 col0" >None</td>
          <td id="T_78b05_row155_col1" class="data row155 col1" >0.031900</td>
          <td id="T_78b05_row155_col2" class="data row155 col2" >-0.034000</td>
          <td id="T_78b05_row155_col3" class="data row155 col3" >-0.038700</td>
          <td id="T_78b05_row155_col4" class="data row155 col4" >-0.017100</td>
          <td id="T_78b05_row155_col5" class="data row155 col5" >-0.028300</td>
          <td id="T_78b05_row155_col6" class="data row155 col6" >-0.049400</td>
          <td id="T_78b05_row155_col7" class="data row155 col7" >-0.039400</td>
          <td id="T_78b05_row155_col8" class="data row155 col8" >0.013600</td>
          <td id="T_78b05_row155_col9" class="data row155 col9" >0.004100</td>
          <td id="T_78b05_row155_col10" class="data row155 col10" >0.069500</td>
          <td id="T_78b05_row155_col11" class="data row155 col11" >0.000800</td>
          <td id="T_78b05_row155_col12" class="data row155 col12" >0.027700</td>
          <td id="T_78b05_row155_col13" class="data row155 col13" >0.045500</td>
          <td id="T_78b05_row155_col14" class="data row155 col14" >0.037600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row156" class="row_heading level0 row156" >157</th>
          <td id="T_78b05_row156_col0" class="data row156 col0" >None</td>
          <td id="T_78b05_row156_col1" class="data row156 col1" >0.044600</td>
          <td id="T_78b05_row156_col2" class="data row156 col2" >0.017900</td>
          <td id="T_78b05_row156_col3" class="data row156 col3" >0.058000</td>
          <td id="T_78b05_row156_col4" class="data row156 col4" >-0.031600</td>
          <td id="T_78b05_row156_col5" class="data row156 col5" >0.008500</td>
          <td id="T_78b05_row156_col6" class="data row156 col6" >0.072300</td>
          <td id="T_78b05_row156_col7" class="data row156 col7" >-0.006200</td>
          <td id="T_78b05_row156_col8" class="data row156 col8" >0.000900</td>
          <td id="T_78b05_row156_col9" class="data row156 col9" >0.047700</td>
          <td id="T_78b05_row156_col10" class="data row156 col10" >0.027200</td>
          <td id="T_78b05_row156_col11" class="data row156 col11" >0.013700</td>
          <td id="T_78b05_row156_col12" class="data row156 col12" >0.009100</td>
          <td id="T_78b05_row156_col13" class="data row156 col13" >0.076200</td>
          <td id="T_78b05_row156_col14" class="data row156 col14" >0.004400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row157" class="row_heading level0 row157" >158</th>
          <td id="T_78b05_row157_col0" class="data row157 col0" >PC1</td>
          <td id="T_78b05_row157_col1" class="data row157 col1" >0.023500</td>
          <td id="T_78b05_row157_col2" class="data row157 col2" >0.060700</td>
          <td id="T_78b05_row157_col3" class="data row157 col3" >-0.054000</td>
          <td id="T_78b05_row157_col4" class="data row157 col4" >0.000900</td>
          <td id="T_78b05_row157_col5" class="data row157 col5" >-0.036200</td>
          <td id="T_78b05_row157_col6" class="data row157 col6" >0.019500</td>
          <td id="T_78b05_row157_col7" class="data row157 col7" >0.091900</td>
          <td id="T_78b05_row157_col8" class="data row157 col8" >0.022000</td>
          <td id="T_78b05_row157_col9" class="data row157 col9" >0.090500</td>
          <td id="T_78b05_row157_col10" class="data row157 col10" >0.084800</td>
          <td id="T_78b05_row157_col11" class="data row157 col11" >0.018800</td>
          <td id="T_78b05_row157_col12" class="data row157 col12" >0.035500</td>
          <td id="T_78b05_row157_col13" class="data row157 col13" >0.023400</td>
          <td id="T_78b05_row157_col14" class="data row157 col14" >0.093700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row158" class="row_heading level0 row158" >159</th>
          <td id="T_78b05_row158_col0" class="data row158 col0" >None</td>
          <td id="T_78b05_row158_col1" class="data row158 col1" >0.048200</td>
          <td id="T_78b05_row158_col2" class="data row158 col2" >-0.026500</td>
          <td id="T_78b05_row158_col3" class="data row158 col3" >0.052800</td>
          <td id="T_78b05_row158_col4" class="data row158 col4" >0.016600</td>
          <td id="T_78b05_row158_col5" class="data row158 col5" >0.018000</td>
          <td id="T_78b05_row158_col6" class="data row158 col6" >-0.054300</td>
          <td id="T_78b05_row158_col7" class="data row158 col7" >0.027400</td>
          <td id="T_78b05_row158_col8" class="data row158 col8" >0.002700</td>
          <td id="T_78b05_row158_col9" class="data row158 col9" >0.003300</td>
          <td id="T_78b05_row158_col10" class="data row158 col10" >0.022000</td>
          <td id="T_78b05_row158_col11" class="data row158 col11" >0.034500</td>
          <td id="T_78b05_row158_col12" class="data row158 col12" >0.018600</td>
          <td id="T_78b05_row158_col13" class="data row158 col13" >0.050400</td>
          <td id="T_78b05_row158_col14" class="data row158 col14" >0.029200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row159" class="row_heading level0 row159" >160</th>
          <td id="T_78b05_row159_col0" class="data row159 col0" >None</td>
          <td id="T_78b05_row159_col1" class="data row159 col1" >0.046000</td>
          <td id="T_78b05_row159_col2" class="data row159 col2" >-0.006000</td>
          <td id="T_78b05_row159_col3" class="data row159 col3" >0.043400</td>
          <td id="T_78b05_row159_col4" class="data row159 col4" >0.026800</td>
          <td id="T_78b05_row159_col5" class="data row159 col5" >0.047200</td>
          <td id="T_78b05_row159_col6" class="data row159 col6" >-0.069600</td>
          <td id="T_78b05_row159_col7" class="data row159 col7" >0.033200</td>
          <td id="T_78b05_row159_col8" class="data row159 col8" >0.000500</td>
          <td id="T_78b05_row159_col9" class="data row159 col9" >0.023800</td>
          <td id="T_78b05_row159_col10" class="data row159 col10" >0.012700</td>
          <td id="T_78b05_row159_col11" class="data row159 col11" >0.044700</td>
          <td id="T_78b05_row159_col12" class="data row159 col12" >0.047900</td>
          <td id="T_78b05_row159_col13" class="data row159 col13" >0.065700</td>
          <td id="T_78b05_row159_col14" class="data row159 col14" >0.034900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row160" class="row_heading level0 row160" >161</th>
          <td id="T_78b05_row160_col0" class="data row160 col0" >PC1</td>
          <td id="T_78b05_row160_col1" class="data row160 col1" >0.025900</td>
          <td id="T_78b05_row160_col2" class="data row160 col2" >-0.031400</td>
          <td id="T_78b05_row160_col3" class="data row160 col3" >-0.044900</td>
          <td id="T_78b05_row160_col4" class="data row160 col4" >0.055400</td>
          <td id="T_78b05_row160_col5" class="data row160 col5" >-0.046000</td>
          <td id="T_78b05_row160_col6" class="data row160 col6" >-0.011100</td>
          <td id="T_78b05_row160_col7" class="data row160 col7" >-0.078500</td>
          <td id="T_78b05_row160_col8" class="data row160 col8" >0.019600</td>
          <td id="T_78b05_row160_col9" class="data row160 col9" >0.001600</td>
          <td id="T_78b05_row160_col10" class="data row160 col10" >0.075700</td>
          <td id="T_78b05_row160_col11" class="data row160 col11" >0.073400</td>
          <td id="T_78b05_row160_col12" class="data row160 col12" >0.045400</td>
          <td id="T_78b05_row160_col13" class="data row160 col13" >0.007200</td>
          <td id="T_78b05_row160_col14" class="data row160 col14" >0.076700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row161" class="row_heading level0 row161" >162</th>
          <td id="T_78b05_row161_col0" class="data row161 col0" >None</td>
          <td id="T_78b05_row161_col1" class="data row161 col1" >0.037400</td>
          <td id="T_78b05_row161_col2" class="data row161 col2" >-0.013800</td>
          <td id="T_78b05_row161_col3" class="data row161 col3" >-0.019100</td>
          <td id="T_78b05_row161_col4" class="data row161 col4" >-0.038700</td>
          <td id="T_78b05_row161_col5" class="data row161 col5" >-0.033600</td>
          <td id="T_78b05_row161_col6" class="data row161 col6" >-0.049700</td>
          <td id="T_78b05_row161_col7" class="data row161 col7" >0.084700</td>
          <td id="T_78b05_row161_col8" class="data row161 col8" >0.008100</td>
          <td id="T_78b05_row161_col9" class="data row161 col9" >0.016000</td>
          <td id="T_78b05_row161_col10" class="data row161 col10" >0.049800</td>
          <td id="T_78b05_row161_col11" class="data row161 col11" >0.020800</td>
          <td id="T_78b05_row161_col12" class="data row161 col12" >0.032900</td>
          <td id="T_78b05_row161_col13" class="data row161 col13" >0.045800</td>
          <td id="T_78b05_row161_col14" class="data row161 col14" >0.086400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row162" class="row_heading level0 row162" >163</th>
          <td id="T_78b05_row162_col0" class="data row162 col0" >None</td>
          <td id="T_78b05_row162_col1" class="data row162 col1" >0.032400</td>
          <td id="T_78b05_row162_col2" class="data row162 col2" >0.056800</td>
          <td id="T_78b05_row162_col3" class="data row162 col3" >-0.005200</td>
          <td id="T_78b05_row162_col4" class="data row162 col4" >0.020600</td>
          <td id="T_78b05_row162_col5" class="data row162 col5" >-0.063600</td>
          <td id="T_78b05_row162_col6" class="data row162 col6" >0.011000</td>
          <td id="T_78b05_row162_col7" class="data row162 col7" >-0.016000</td>
          <td id="T_78b05_row162_col8" class="data row162 col8" >0.013200</td>
          <td id="T_78b05_row162_col9" class="data row162 col9" >0.086600</td>
          <td id="T_78b05_row162_col10" class="data row162 col10" >0.035900</td>
          <td id="T_78b05_row162_col11" class="data row162 col11" >0.038500</td>
          <td id="T_78b05_row162_col12" class="data row162 col12" >0.063000</td>
          <td id="T_78b05_row162_col13" class="data row162 col13" >0.014800</td>
          <td id="T_78b05_row162_col14" class="data row162 col14" >0.014300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row163" class="row_heading level0 row163" >164</th>
          <td id="T_78b05_row163_col0" class="data row163 col0" >None</td>
          <td id="T_78b05_row163_col1" class="data row163 col1" >0.031900</td>
          <td id="T_78b05_row163_col2" class="data row163 col2" >0.042800</td>
          <td id="T_78b05_row163_col3" class="data row163 col3" >-0.021700</td>
          <td id="T_78b05_row163_col4" class="data row163 col4" >0.037100</td>
          <td id="T_78b05_row163_col5" class="data row163 col5" >-0.050900</td>
          <td id="T_78b05_row163_col6" class="data row163 col6" >0.027800</td>
          <td id="T_78b05_row163_col7" class="data row163 col7" >0.062800</td>
          <td id="T_78b05_row163_col8" class="data row163 col8" >0.013600</td>
          <td id="T_78b05_row163_col9" class="data row163 col9" >0.072600</td>
          <td id="T_78b05_row163_col10" class="data row163 col10" >0.052400</td>
          <td id="T_78b05_row163_col11" class="data row163 col11" >0.055000</td>
          <td id="T_78b05_row163_col12" class="data row163 col12" >0.050200</td>
          <td id="T_78b05_row163_col13" class="data row163 col13" >0.031700</td>
          <td id="T_78b05_row163_col14" class="data row163 col14" >0.064600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row164" class="row_heading level0 row164" >165</th>
          <td id="T_78b05_row164_col0" class="data row164 col0" >None</td>
          <td id="T_78b05_row164_col1" class="data row164 col1" >0.037600</td>
          <td id="T_78b05_row164_col2" class="data row164 col2" >0.021200</td>
          <td id="T_78b05_row164_col3" class="data row164 col3" >-0.004100</td>
          <td id="T_78b05_row164_col4" class="data row164 col4" >0.054900</td>
          <td id="T_78b05_row164_col5" class="data row164 col5" >-0.003800</td>
          <td id="T_78b05_row164_col6" class="data row164 col6" >-0.003600</td>
          <td id="T_78b05_row164_col7" class="data row164 col7" >0.018200</td>
          <td id="T_78b05_row164_col8" class="data row164 col8" >0.007900</td>
          <td id="T_78b05_row164_col9" class="data row164 col9" >0.051000</td>
          <td id="T_78b05_row164_col10" class="data row164 col10" >0.034800</td>
          <td id="T_78b05_row164_col11" class="data row164 col11" >0.072800</td>
          <td id="T_78b05_row164_col12" class="data row164 col12" >0.003200</td>
          <td id="T_78b05_row164_col13" class="data row164 col13" >0.000300</td>
          <td id="T_78b05_row164_col14" class="data row164 col14" >0.019900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row165" class="row_heading level0 row165" >166</th>
          <td id="T_78b05_row165_col0" class="data row165 col0" >None</td>
          <td id="T_78b05_row165_col1" class="data row165 col1" >0.036000</td>
          <td id="T_78b05_row165_col2" class="data row165 col2" >0.053800</td>
          <td id="T_78b05_row165_col3" class="data row165 col3" >0.050500</td>
          <td id="T_78b05_row165_col4" class="data row165 col4" >0.036900</td>
          <td id="T_78b05_row165_col5" class="data row165 col5" >-0.021100</td>
          <td id="T_78b05_row165_col6" class="data row165 col6" >-0.006300</td>
          <td id="T_78b05_row165_col7" class="data row165 col7" >-0.035700</td>
          <td id="T_78b05_row165_col8" class="data row165 col8" >0.009600</td>
          <td id="T_78b05_row165_col9" class="data row165 col9" >0.083600</td>
          <td id="T_78b05_row165_col10" class="data row165 col10" >0.019800</td>
          <td id="T_78b05_row165_col11" class="data row165 col11" >0.054800</td>
          <td id="T_78b05_row165_col12" class="data row165 col12" >0.020500</td>
          <td id="T_78b05_row165_col13" class="data row165 col13" >0.002400</td>
          <td id="T_78b05_row165_col14" class="data row165 col14" >0.034000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row166" class="row_heading level0 row166" >167</th>
          <td id="T_78b05_row166_col0" class="data row166 col0" >None</td>
          <td id="T_78b05_row166_col1" class="data row166 col1" >0.038700</td>
          <td id="T_78b05_row166_col2" class="data row166 col2" >0.017200</td>
          <td id="T_78b05_row166_col3" class="data row166 col3" >-0.017200</td>
          <td id="T_78b05_row166_col4" class="data row166 col4" >-0.030700</td>
          <td id="T_78b05_row166_col5" class="data row166 col5" >0.006700</td>
          <td id="T_78b05_row166_col6" class="data row166 col6" >-0.052900</td>
          <td id="T_78b05_row166_col7" class="data row166 col7" >0.024800</td>
          <td id="T_78b05_row166_col8" class="data row166 col8" >0.006800</td>
          <td id="T_78b05_row166_col9" class="data row166 col9" >0.047000</td>
          <td id="T_78b05_row166_col10" class="data row166 col10" >0.047900</td>
          <td id="T_78b05_row166_col11" class="data row166 col11" >0.012800</td>
          <td id="T_78b05_row166_col12" class="data row166 col12" >0.007300</td>
          <td id="T_78b05_row166_col13" class="data row166 col13" >0.049000</td>
          <td id="T_78b05_row166_col14" class="data row166 col14" >0.026600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row167" class="row_heading level0 row167" >168</th>
          <td id="T_78b05_row167_col0" class="data row167 col0" >None</td>
          <td id="T_78b05_row167_col1" class="data row167 col1" >0.035900</td>
          <td id="T_78b05_row167_col2" class="data row167 col2" >0.039700</td>
          <td id="T_78b05_row167_col3" class="data row167 col3" >0.003200</td>
          <td id="T_78b05_row167_col4" class="data row167 col4" >-0.014100</td>
          <td id="T_78b05_row167_col5" class="data row167 col5" >-0.040400</td>
          <td id="T_78b05_row167_col6" class="data row167 col6" >0.053900</td>
          <td id="T_78b05_row167_col7" class="data row167 col7" >-0.053000</td>
          <td id="T_78b05_row167_col8" class="data row167 col8" >0.009700</td>
          <td id="T_78b05_row167_col9" class="data row167 col9" >0.069600</td>
          <td id="T_78b05_row167_col10" class="data row167 col10" >0.027600</td>
          <td id="T_78b05_row167_col11" class="data row167 col11" >0.003800</td>
          <td id="T_78b05_row167_col12" class="data row167 col12" >0.039800</td>
          <td id="T_78b05_row167_col13" class="data row167 col13" >0.057800</td>
          <td id="T_78b05_row167_col14" class="data row167 col14" >0.051300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row168" class="row_heading level0 row168" >169</th>
          <td id="T_78b05_row168_col0" class="data row168 col0" >PC1</td>
          <td id="T_78b05_row168_col1" class="data row168 col1" >0.026500</td>
          <td id="T_78b05_row168_col2" class="data row168 col2" >0.009900</td>
          <td id="T_78b05_row168_col3" class="data row168 col3" >-0.012500</td>
          <td id="T_78b05_row168_col4" class="data row168 col4" >-0.016700</td>
          <td id="T_78b05_row168_col5" class="data row168 col5" >-0.020400</td>
          <td id="T_78b05_row168_col6" class="data row168 col6" >-0.056800</td>
          <td id="T_78b05_row168_col7" class="data row168 col7" >0.018700</td>
          <td id="T_78b05_row168_col8" class="data row168 col8" >0.019000</td>
          <td id="T_78b05_row168_col9" class="data row168 col9" >0.039700</td>
          <td id="T_78b05_row168_col10" class="data row168 col10" >0.043200</td>
          <td id="T_78b05_row168_col11" class="data row168 col11" >0.001200</td>
          <td id="T_78b05_row168_col12" class="data row168 col12" >0.019700</td>
          <td id="T_78b05_row168_col13" class="data row168 col13" >0.052900</td>
          <td id="T_78b05_row168_col14" class="data row168 col14" >0.020400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row169" class="row_heading level0 row169" >170</th>
          <td id="T_78b05_row169_col0" class="data row169 col0" >PC1</td>
          <td id="T_78b05_row169_col1" class="data row169 col1" >0.026100</td>
          <td id="T_78b05_row169_col2" class="data row169 col2" >0.035300</td>
          <td id="T_78b05_row169_col3" class="data row169 col3" >-0.058300</td>
          <td id="T_78b05_row169_col4" class="data row169 col4" >0.025800</td>
          <td id="T_78b05_row169_col5" class="data row169 col5" >-0.106600</td>
          <td id="T_78b05_row169_col6" class="data row169 col6" >-0.009100</td>
          <td id="T_78b05_row169_col7" class="data row169 col7" >-0.029400</td>
          <td id="T_78b05_row169_col8" class="data row169 col8" >0.019500</td>
          <td id="T_78b05_row169_col9" class="data row169 col9" >0.065100</td>
          <td id="T_78b05_row169_col10" class="data row169 col10" >0.089100</td>
          <td id="T_78b05_row169_col11" class="data row169 col11" >0.043700</td>
          <td id="T_78b05_row169_col12" class="data row169 col12" >0.106000</td>
          <td id="T_78b05_row169_col13" class="data row169 col13" >0.005200</td>
          <td id="T_78b05_row169_col14" class="data row169 col14" >0.027700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row170" class="row_heading level0 row170" >171</th>
          <td id="T_78b05_row170_col0" class="data row170 col0" >None</td>
          <td id="T_78b05_row170_col1" class="data row170 col1" >0.043000</td>
          <td id="T_78b05_row170_col2" class="data row170 col2" >0.035900</td>
          <td id="T_78b05_row170_col3" class="data row170 col3" >0.007500</td>
          <td id="T_78b05_row170_col4" class="data row170 col4" >0.022100</td>
          <td id="T_78b05_row170_col5" class="data row170 col5" >0.043100</td>
          <td id="T_78b05_row170_col6" class="data row170 col6" >-0.029100</td>
          <td id="T_78b05_row170_col7" class="data row170 col7" >-0.030500</td>
          <td id="T_78b05_row170_col8" class="data row170 col8" >0.002600</td>
          <td id="T_78b05_row170_col9" class="data row170 col9" >0.065800</td>
          <td id="T_78b05_row170_col10" class="data row170 col10" >0.023200</td>
          <td id="T_78b05_row170_col11" class="data row170 col11" >0.040000</td>
          <td id="T_78b05_row170_col12" class="data row170 col12" >0.043800</td>
          <td id="T_78b05_row170_col13" class="data row170 col13" >0.025200</td>
          <td id="T_78b05_row170_col14" class="data row170 col14" >0.028800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row171" class="row_heading level0 row171" >172</th>
          <td id="T_78b05_row171_col0" class="data row171 col0" >None</td>
          <td id="T_78b05_row171_col1" class="data row171 col1" >0.036300</td>
          <td id="T_78b05_row171_col2" class="data row171 col2" >-0.055900</td>
          <td id="T_78b05_row171_col3" class="data row171 col3" >-0.086900</td>
          <td id="T_78b05_row171_col4" class="data row171 col4" >-0.018000</td>
          <td id="T_78b05_row171_col5" class="data row171 col5" >-0.023800</td>
          <td id="T_78b05_row171_col6" class="data row171 col6" >-0.015700</td>
          <td id="T_78b05_row171_col7" class="data row171 col7" >-0.008100</td>
          <td id="T_78b05_row171_col8" class="data row171 col8" >0.009200</td>
          <td id="T_78b05_row171_col9" class="data row171 col9" >0.026000</td>
          <td id="T_78b05_row171_col10" class="data row171 col10" >0.117600</td>
          <td id="T_78b05_row171_col11" class="data row171 col11" >0.000100</td>
          <td id="T_78b05_row171_col12" class="data row171 col12" >0.023200</td>
          <td id="T_78b05_row171_col13" class="data row171 col13" >0.011900</td>
          <td id="T_78b05_row171_col14" class="data row171 col14" >0.006300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row172" class="row_heading level0 row172" >173</th>
          <td id="T_78b05_row172_col0" class="data row172 col0" >None</td>
          <td id="T_78b05_row172_col1" class="data row172 col1" >0.037500</td>
          <td id="T_78b05_row172_col2" class="data row172 col2" >0.062300</td>
          <td id="T_78b05_row172_col3" class="data row172 col3" >-0.002700</td>
          <td id="T_78b05_row172_col4" class="data row172 col4" >-0.050300</td>
          <td id="T_78b05_row172_col5" class="data row172 col5" >-0.027900</td>
          <td id="T_78b05_row172_col6" class="data row172 col6" >0.012300</td>
          <td id="T_78b05_row172_col7" class="data row172 col7" >0.000200</td>
          <td id="T_78b05_row172_col8" class="data row172 col8" >0.008000</td>
          <td id="T_78b05_row172_col9" class="data row172 col9" >0.092100</td>
          <td id="T_78b05_row172_col10" class="data row172 col10" >0.033400</td>
          <td id="T_78b05_row172_col11" class="data row172 col11" >0.032400</td>
          <td id="T_78b05_row172_col12" class="data row172 col12" >0.027300</td>
          <td id="T_78b05_row172_col13" class="data row172 col13" >0.016200</td>
          <td id="T_78b05_row172_col14" class="data row172 col14" >0.002000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row173" class="row_heading level0 row173" >174</th>
          <td id="T_78b05_row173_col0" class="data row173 col0" >None</td>
          <td id="T_78b05_row173_col1" class="data row173 col1" >0.037200</td>
          <td id="T_78b05_row173_col2" class="data row173 col2" >-0.059100</td>
          <td id="T_78b05_row173_col3" class="data row173 col3" >-0.014800</td>
          <td id="T_78b05_row173_col4" class="data row173 col4" >-0.033300</td>
          <td id="T_78b05_row173_col5" class="data row173 col5" >-0.065400</td>
          <td id="T_78b05_row173_col6" class="data row173 col6" >0.010000</td>
          <td id="T_78b05_row173_col7" class="data row173 col7" >-0.037000</td>
          <td id="T_78b05_row173_col8" class="data row173 col8" >0.008300</td>
          <td id="T_78b05_row173_col9" class="data row173 col9" >0.029200</td>
          <td id="T_78b05_row173_col10" class="data row173 col10" >0.045600</td>
          <td id="T_78b05_row173_col11" class="data row173 col11" >0.015400</td>
          <td id="T_78b05_row173_col12" class="data row173 col12" >0.064800</td>
          <td id="T_78b05_row173_col13" class="data row173 col13" >0.013900</td>
          <td id="T_78b05_row173_col14" class="data row173 col14" >0.035200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row174" class="row_heading level0 row174" >175</th>
          <td id="T_78b05_row174_col0" class="data row174 col0" >None</td>
          <td id="T_78b05_row174_col1" class="data row174 col1" >0.038400</td>
          <td id="T_78b05_row174_col2" class="data row174 col2" >0.004900</td>
          <td id="T_78b05_row174_col3" class="data row174 col3" >-0.046700</td>
          <td id="T_78b05_row174_col4" class="data row174 col4" >0.004300</td>
          <td id="T_78b05_row174_col5" class="data row174 col5" >0.053400</td>
          <td id="T_78b05_row174_col6" class="data row174 col6" >0.065500</td>
          <td id="T_78b05_row174_col7" class="data row174 col7" >0.041500</td>
          <td id="T_78b05_row174_col8" class="data row174 col8" >0.007200</td>
          <td id="T_78b05_row174_col9" class="data row174 col9" >0.034700</td>
          <td id="T_78b05_row174_col10" class="data row174 col10" >0.077400</td>
          <td id="T_78b05_row174_col11" class="data row174 col11" >0.022200</td>
          <td id="T_78b05_row174_col12" class="data row174 col12" >0.054000</td>
          <td id="T_78b05_row174_col13" class="data row174 col13" >0.069400</td>
          <td id="T_78b05_row174_col14" class="data row174 col14" >0.043200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row175" class="row_heading level0 row175" >176</th>
          <td id="T_78b05_row175_col0" class="data row175 col0" >None</td>
          <td id="T_78b05_row175_col1" class="data row175 col1" >0.043700</td>
          <td id="T_78b05_row175_col2" class="data row175 col2" >0.028900</td>
          <td id="T_78b05_row175_col3" class="data row175 col3" >-0.006300</td>
          <td id="T_78b05_row175_col4" class="data row175 col4" >0.001400</td>
          <td id="T_78b05_row175_col5" class="data row175 col5" >0.014500</td>
          <td id="T_78b05_row175_col6" class="data row175 col6" >0.033600</td>
          <td id="T_78b05_row175_col7" class="data row175 col7" >0.046100</td>
          <td id="T_78b05_row175_col8" class="data row175 col8" >0.001800</td>
          <td id="T_78b05_row175_col9" class="data row175 col9" >0.058700</td>
          <td id="T_78b05_row175_col10" class="data row175 col10" >0.037000</td>
          <td id="T_78b05_row175_col11" class="data row175 col11" >0.019300</td>
          <td id="T_78b05_row175_col12" class="data row175 col12" >0.015100</td>
          <td id="T_78b05_row175_col13" class="data row175 col13" >0.037500</td>
          <td id="T_78b05_row175_col14" class="data row175 col14" >0.047900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row176" class="row_heading level0 row176" >177</th>
          <td id="T_78b05_row176_col0" class="data row176 col0" >None</td>
          <td id="T_78b05_row176_col1" class="data row176 col1" >0.036300</td>
          <td id="T_78b05_row176_col2" class="data row176 col2" >-0.012000</td>
          <td id="T_78b05_row176_col3" class="data row176 col3" >-0.029600</td>
          <td id="T_78b05_row176_col4" class="data row176 col4" >0.105400</td>
          <td id="T_78b05_row176_col5" class="data row176 col5" >0.059600</td>
          <td id="T_78b05_row176_col6" class="data row176 col6" >-0.034400</td>
          <td id="T_78b05_row176_col7" class="data row176 col7" >0.020600</td>
          <td id="T_78b05_row176_col8" class="data row176 col8" >0.009200</td>
          <td id="T_78b05_row176_col9" class="data row176 col9" >0.017800</td>
          <td id="T_78b05_row176_col10" class="data row176 col10" >0.060400</td>
          <td id="T_78b05_row176_col11" class="data row176 col11" >0.123300</td>
          <td id="T_78b05_row176_col12" class="data row176 col12" >0.060300</td>
          <td id="T_78b05_row176_col13" class="data row176 col13" >0.030600</td>
          <td id="T_78b05_row176_col14" class="data row176 col14" >0.022300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row177" class="row_heading level0 row177" >178</th>
          <td id="T_78b05_row177_col0" class="data row177 col0" >None</td>
          <td id="T_78b05_row177_col1" class="data row177 col1" >0.043300</td>
          <td id="T_78b05_row177_col2" class="data row177 col2" >0.034200</td>
          <td id="T_78b05_row177_col3" class="data row177 col3" >0.034800</td>
          <td id="T_78b05_row177_col4" class="data row177 col4" >0.020400</td>
          <td id="T_78b05_row177_col5" class="data row177 col5" >0.094800</td>
          <td id="T_78b05_row177_col6" class="data row177 col6" >-0.012500</td>
          <td id="T_78b05_row177_col7" class="data row177 col7" >0.020600</td>
          <td id="T_78b05_row177_col8" class="data row177 col8" >0.002300</td>
          <td id="T_78b05_row177_col9" class="data row177 col9" >0.064000</td>
          <td id="T_78b05_row177_col10" class="data row177 col10" >0.004000</td>
          <td id="T_78b05_row177_col11" class="data row177 col11" >0.038300</td>
          <td id="T_78b05_row177_col12" class="data row177 col12" >0.095400</td>
          <td id="T_78b05_row177_col13" class="data row177 col13" >0.008600</td>
          <td id="T_78b05_row177_col14" class="data row177 col14" >0.022400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row178" class="row_heading level0 row178" >179</th>
          <td id="T_78b05_row178_col0" class="data row178 col0" >None</td>
          <td id="T_78b05_row178_col1" class="data row178 col1" >0.045400</td>
          <td id="T_78b05_row178_col2" class="data row178 col2" >-0.060600</td>
          <td id="T_78b05_row178_col3" class="data row178 col3" >-0.006600</td>
          <td id="T_78b05_row178_col4" class="data row178 col4" >0.011300</td>
          <td id="T_78b05_row178_col5" class="data row178 col5" >-0.035400</td>
          <td id="T_78b05_row178_col6" class="data row178 col6" >-0.021000</td>
          <td id="T_78b05_row178_col7" class="data row178 col7" >0.025900</td>
          <td id="T_78b05_row178_col8" class="data row178 col8" >0.000100</td>
          <td id="T_78b05_row178_col9" class="data row178 col9" >0.030800</td>
          <td id="T_78b05_row178_col10" class="data row178 col10" >0.037400</td>
          <td id="T_78b05_row178_col11" class="data row178 col11" >0.029200</td>
          <td id="T_78b05_row178_col12" class="data row178 col12" >0.034800</td>
          <td id="T_78b05_row178_col13" class="data row178 col13" >0.017200</td>
          <td id="T_78b05_row178_col14" class="data row178 col14" >0.027600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row179" class="row_heading level0 row179" >180</th>
          <td id="T_78b05_row179_col0" class="data row179 col0" >None</td>
          <td id="T_78b05_row179_col1" class="data row179 col1" >0.035200</td>
          <td id="T_78b05_row179_col2" class="data row179 col2" >0.012300</td>
          <td id="T_78b05_row179_col3" class="data row179 col3" >0.004300</td>
          <td id="T_78b05_row179_col4" class="data row179 col4" >-0.045200</td>
          <td id="T_78b05_row179_col5" class="data row179 col5" >-0.033800</td>
          <td id="T_78b05_row179_col6" class="data row179 col6" >-0.052700</td>
          <td id="T_78b05_row179_col7" class="data row179 col7" >0.049800</td>
          <td id="T_78b05_row179_col8" class="data row179 col8" >0.010300</td>
          <td id="T_78b05_row179_col9" class="data row179 col9" >0.042100</td>
          <td id="T_78b05_row179_col10" class="data row179 col10" >0.026500</td>
          <td id="T_78b05_row179_col11" class="data row179 col11" >0.027300</td>
          <td id="T_78b05_row179_col12" class="data row179 col12" >0.033100</td>
          <td id="T_78b05_row179_col13" class="data row179 col13" >0.048800</td>
          <td id="T_78b05_row179_col14" class="data row179 col14" >0.051500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row180" class="row_heading level0 row180" >181</th>
          <td id="T_78b05_row180_col0" class="data row180 col0" >None</td>
          <td id="T_78b05_row180_col1" class="data row180 col1" >0.035700</td>
          <td id="T_78b05_row180_col2" class="data row180 col2" >0.032200</td>
          <td id="T_78b05_row180_col3" class="data row180 col3" >-0.022900</td>
          <td id="T_78b05_row180_col4" class="data row180 col4" >-0.020200</td>
          <td id="T_78b05_row180_col5" class="data row180 col5" >0.009300</td>
          <td id="T_78b05_row180_col6" class="data row180 col6" >-0.017500</td>
          <td id="T_78b05_row180_col7" class="data row180 col7" >0.064100</td>
          <td id="T_78b05_row180_col8" class="data row180 col8" >0.009800</td>
          <td id="T_78b05_row180_col9" class="data row180 col9" >0.062100</td>
          <td id="T_78b05_row180_col10" class="data row180 col10" >0.053700</td>
          <td id="T_78b05_row180_col11" class="data row180 col11" >0.002300</td>
          <td id="T_78b05_row180_col12" class="data row180 col12" >0.009900</td>
          <td id="T_78b05_row180_col13" class="data row180 col13" >0.013600</td>
          <td id="T_78b05_row180_col14" class="data row180 col14" >0.065900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row181" class="row_heading level0 row181" >182</th>
          <td id="T_78b05_row181_col0" class="data row181 col0" >None</td>
          <td id="T_78b05_row181_col1" class="data row181 col1" >0.039700</td>
          <td id="T_78b05_row181_col2" class="data row181 col2" >-0.030900</td>
          <td id="T_78b05_row181_col3" class="data row181 col3" >-0.021400</td>
          <td id="T_78b05_row181_col4" class="data row181 col4" >-0.009100</td>
          <td id="T_78b05_row181_col5" class="data row181 col5" >-0.049900</td>
          <td id="T_78b05_row181_col6" class="data row181 col6" >-0.030000</td>
          <td id="T_78b05_row181_col7" class="data row181 col7" >0.074100</td>
          <td id="T_78b05_row181_col8" class="data row181 col8" >0.005800</td>
          <td id="T_78b05_row181_col9" class="data row181 col9" >0.001100</td>
          <td id="T_78b05_row181_col10" class="data row181 col10" >0.052200</td>
          <td id="T_78b05_row181_col11" class="data row181 col11" >0.008800</td>
          <td id="T_78b05_row181_col12" class="data row181 col12" >0.049300</td>
          <td id="T_78b05_row181_col13" class="data row181 col13" >0.026100</td>
          <td id="T_78b05_row181_col14" class="data row181 col14" >0.075800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row182" class="row_heading level0 row182" >183</th>
          <td id="T_78b05_row182_col0" class="data row182 col0" >None</td>
          <td id="T_78b05_row182_col1" class="data row182 col1" >0.042100</td>
          <td id="T_78b05_row182_col2" class="data row182 col2" >0.003600</td>
          <td id="T_78b05_row182_col3" class="data row182 col3" >0.030100</td>
          <td id="T_78b05_row182_col4" class="data row182 col4" >-0.019300</td>
          <td id="T_78b05_row182_col5" class="data row182 col5" >-0.015000</td>
          <td id="T_78b05_row182_col6" class="data row182 col6" >0.021200</td>
          <td id="T_78b05_row182_col7" class="data row182 col7" >-0.011000</td>
          <td id="T_78b05_row182_col8" class="data row182 col8" >0.003400</td>
          <td id="T_78b05_row182_col9" class="data row182 col9" >0.033500</td>
          <td id="T_78b05_row182_col10" class="data row182 col10" >0.000700</td>
          <td id="T_78b05_row182_col11" class="data row182 col11" >0.001400</td>
          <td id="T_78b05_row182_col12" class="data row182 col12" >0.014300</td>
          <td id="T_78b05_row182_col13" class="data row182 col13" >0.025100</td>
          <td id="T_78b05_row182_col14" class="data row182 col14" >0.009200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row183" class="row_heading level0 row183" >184</th>
          <td id="T_78b05_row183_col0" class="data row183 col0" >None</td>
          <td id="T_78b05_row183_col1" class="data row183 col1" >0.048300</td>
          <td id="T_78b05_row183_col2" class="data row183 col2" >-0.041800</td>
          <td id="T_78b05_row183_col3" class="data row183 col3" >0.049500</td>
          <td id="T_78b05_row183_col4" class="data row183 col4" >-0.051500</td>
          <td id="T_78b05_row183_col5" class="data row183 col5" >0.010800</td>
          <td id="T_78b05_row183_col6" class="data row183 col6" >-0.037000</td>
          <td id="T_78b05_row183_col7" class="data row183 col7" >0.051300</td>
          <td id="T_78b05_row183_col8" class="data row183 col8" >0.002800</td>
          <td id="T_78b05_row183_col9" class="data row183 col9" >0.012000</td>
          <td id="T_78b05_row183_col10" class="data row183 col10" >0.018800</td>
          <td id="T_78b05_row183_col11" class="data row183 col11" >0.033500</td>
          <td id="T_78b05_row183_col12" class="data row183 col12" >0.011400</td>
          <td id="T_78b05_row183_col13" class="data row183 col13" >0.033100</td>
          <td id="T_78b05_row183_col14" class="data row183 col14" >0.053000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row184" class="row_heading level0 row184" >185</th>
          <td id="T_78b05_row184_col0" class="data row184 col0" >None</td>
          <td id="T_78b05_row184_col1" class="data row184 col1" >0.037300</td>
          <td id="T_78b05_row184_col2" class="data row184 col2" >0.013700</td>
          <td id="T_78b05_row184_col3" class="data row184 col3" >-0.001000</td>
          <td id="T_78b05_row184_col4" class="data row184 col4" >-0.023900</td>
          <td id="T_78b05_row184_col5" class="data row184 col5" >-0.000500</td>
          <td id="T_78b05_row184_col6" class="data row184 col6" >0.014800</td>
          <td id="T_78b05_row184_col7" class="data row184 col7" >-0.017400</td>
          <td id="T_78b05_row184_col8" class="data row184 col8" >0.008200</td>
          <td id="T_78b05_row184_col9" class="data row184 col9" >0.043500</td>
          <td id="T_78b05_row184_col10" class="data row184 col10" >0.031700</td>
          <td id="T_78b05_row184_col11" class="data row184 col11" >0.006000</td>
          <td id="T_78b05_row184_col12" class="data row184 col12" >0.000100</td>
          <td id="T_78b05_row184_col13" class="data row184 col13" >0.018700</td>
          <td id="T_78b05_row184_col14" class="data row184 col14" >0.015700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row185" class="row_heading level0 row185" >186</th>
          <td id="T_78b05_row185_col0" class="data row185 col0" >None</td>
          <td id="T_78b05_row185_col1" class="data row185 col1" >0.045300</td>
          <td id="T_78b05_row185_col2" class="data row185 col2" >-0.012100</td>
          <td id="T_78b05_row185_col3" class="data row185 col3" >0.020000</td>
          <td id="T_78b05_row185_col4" class="data row185 col4" >0.029300</td>
          <td id="T_78b05_row185_col5" class="data row185 col5" >0.047100</td>
          <td id="T_78b05_row185_col6" class="data row185 col6" >0.012000</td>
          <td id="T_78b05_row185_col7" class="data row185 col7" >0.026700</td>
          <td id="T_78b05_row185_col8" class="data row185 col8" >0.000200</td>
          <td id="T_78b05_row185_col9" class="data row185 col9" >0.017700</td>
          <td id="T_78b05_row185_col10" class="data row185 col10" >0.010800</td>
          <td id="T_78b05_row185_col11" class="data row185 col11" >0.047200</td>
          <td id="T_78b05_row185_col12" class="data row185 col12" >0.047800</td>
          <td id="T_78b05_row185_col13" class="data row185 col13" >0.015900</td>
          <td id="T_78b05_row185_col14" class="data row185 col14" >0.028500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row186" class="row_heading level0 row186" >187</th>
          <td id="T_78b05_row186_col0" class="data row186 col0" >PC1</td>
          <td id="T_78b05_row186_col1" class="data row186 col1" >0.026100</td>
          <td id="T_78b05_row186_col2" class="data row186 col2" >-0.018800</td>
          <td id="T_78b05_row186_col3" class="data row186 col3" >-0.050600</td>
          <td id="T_78b05_row186_col4" class="data row186 col4" >0.038600</td>
          <td id="T_78b05_row186_col5" class="data row186 col5" >-0.009100</td>
          <td id="T_78b05_row186_col6" class="data row186 col6" >-0.023100</td>
          <td id="T_78b05_row186_col7" class="data row186 col7" >-0.059800</td>
          <td id="T_78b05_row186_col8" class="data row186 col8" >0.019400</td>
          <td id="T_78b05_row186_col9" class="data row186 col9" >0.011000</td>
          <td id="T_78b05_row186_col10" class="data row186 col10" >0.081300</td>
          <td id="T_78b05_row186_col11" class="data row186 col11" >0.056600</td>
          <td id="T_78b05_row186_col12" class="data row186 col12" >0.008400</td>
          <td id="T_78b05_row186_col13" class="data row186 col13" >0.019200</td>
          <td id="T_78b05_row186_col14" class="data row186 col14" >0.058100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row187" class="row_heading level0 row187" >188</th>
          <td id="T_78b05_row187_col0" class="data row187 col0" >None</td>
          <td id="T_78b05_row187_col1" class="data row187 col1" >0.032300</td>
          <td id="T_78b05_row187_col2" class="data row187 col2" >-0.053900</td>
          <td id="T_78b05_row187_col3" class="data row187 col3" >0.006300</td>
          <td id="T_78b05_row187_col4" class="data row187 col4" >0.063600</td>
          <td id="T_78b05_row187_col5" class="data row187 col5" >-0.075100</td>
          <td id="T_78b05_row187_col6" class="data row187 col6" >0.020200</td>
          <td id="T_78b05_row187_col7" class="data row187 col7" >-0.087300</td>
          <td id="T_78b05_row187_col8" class="data row187 col8" >0.013200</td>
          <td id="T_78b05_row187_col9" class="data row187 col9" >0.024100</td>
          <td id="T_78b05_row187_col10" class="data row187 col10" >0.024500</td>
          <td id="T_78b05_row187_col11" class="data row187 col11" >0.081500</td>
          <td id="T_78b05_row187_col12" class="data row187 col12" >0.074400</td>
          <td id="T_78b05_row187_col13" class="data row187 col13" >0.024100</td>
          <td id="T_78b05_row187_col14" class="data row187 col14" >0.085500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row188" class="row_heading level0 row188" >189</th>
          <td id="T_78b05_row188_col0" class="data row188 col0" >None</td>
          <td id="T_78b05_row188_col1" class="data row188 col1" >0.043500</td>
          <td id="T_78b05_row188_col2" class="data row188 col2" >-0.033700</td>
          <td id="T_78b05_row188_col3" class="data row188 col3" >0.000600</td>
          <td id="T_78b05_row188_col4" class="data row188 col4" >0.027600</td>
          <td id="T_78b05_row188_col5" class="data row188 col5" >-0.017800</td>
          <td id="T_78b05_row188_col6" class="data row188 col6" >-0.047900</td>
          <td id="T_78b05_row188_col7" class="data row188 col7" >-0.017000</td>
          <td id="T_78b05_row188_col8" class="data row188 col8" >0.002100</td>
          <td id="T_78b05_row188_col9" class="data row188 col9" >0.003900</td>
          <td id="T_78b05_row188_col10" class="data row188 col10" >0.030100</td>
          <td id="T_78b05_row188_col11" class="data row188 col11" >0.045500</td>
          <td id="T_78b05_row188_col12" class="data row188 col12" >0.017100</td>
          <td id="T_78b05_row188_col13" class="data row188 col13" >0.044000</td>
          <td id="T_78b05_row188_col14" class="data row188 col14" >0.015300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row189" class="row_heading level0 row189" >190</th>
          <td id="T_78b05_row189_col0" class="data row189 col0" >None</td>
          <td id="T_78b05_row189_col1" class="data row189 col1" >0.033100</td>
          <td id="T_78b05_row189_col2" class="data row189 col2" >0.022700</td>
          <td id="T_78b05_row189_col3" class="data row189 col3" >-0.032800</td>
          <td id="T_78b05_row189_col4" class="data row189 col4" >-0.000100</td>
          <td id="T_78b05_row189_col5" class="data row189 col5" >-0.028600</td>
          <td id="T_78b05_row189_col6" class="data row189 col6" >-0.008400</td>
          <td id="T_78b05_row189_col7" class="data row189 col7" >0.023100</td>
          <td id="T_78b05_row189_col8" class="data row189 col8" >0.012500</td>
          <td id="T_78b05_row189_col9" class="data row189 col9" >0.052600</td>
          <td id="T_78b05_row189_col10" class="data row189 col10" >0.063600</td>
          <td id="T_78b05_row189_col11" class="data row189 col11" >0.017800</td>
          <td id="T_78b05_row189_col12" class="data row189 col12" >0.028000</td>
          <td id="T_78b05_row189_col13" class="data row189 col13" >0.004500</td>
          <td id="T_78b05_row189_col14" class="data row189 col14" >0.024900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row190" class="row_heading level0 row190" >191</th>
          <td id="T_78b05_row190_col0" class="data row190 col0" >None</td>
          <td id="T_78b05_row190_col1" class="data row190 col1" >0.044500</td>
          <td id="T_78b05_row190_col2" class="data row190 col2" >0.010700</td>
          <td id="T_78b05_row190_col3" class="data row190 col3" >0.026000</td>
          <td id="T_78b05_row190_col4" class="data row190 col4" >0.008400</td>
          <td id="T_78b05_row190_col5" class="data row190 col5" >0.011800</td>
          <td id="T_78b05_row190_col6" class="data row190 col6" >0.006300</td>
          <td id="T_78b05_row190_col7" class="data row190 col7" >0.067600</td>
          <td id="T_78b05_row190_col8" class="data row190 col8" >0.001000</td>
          <td id="T_78b05_row190_col9" class="data row190 col9" >0.040500</td>
          <td id="T_78b05_row190_col10" class="data row190 col10" >0.004800</td>
          <td id="T_78b05_row190_col11" class="data row190 col11" >0.026300</td>
          <td id="T_78b05_row190_col12" class="data row190 col12" >0.012400</td>
          <td id="T_78b05_row190_col13" class="data row190 col13" >0.010200</td>
          <td id="T_78b05_row190_col14" class="data row190 col14" >0.069300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row191" class="row_heading level0 row191" >192</th>
          <td id="T_78b05_row191_col0" class="data row191 col0" >PC6</td>
          <td id="T_78b05_row191_col1" class="data row191 col1" >0.040100</td>
          <td id="T_78b05_row191_col2" class="data row191 col2" >0.002200</td>
          <td id="T_78b05_row191_col3" class="data row191 col3" >-0.004300</td>
          <td id="T_78b05_row191_col4" class="data row191 col4" >-0.053600</td>
          <td id="T_78b05_row191_col5" class="data row191 col5" >0.035900</td>
          <td id="T_78b05_row191_col6" class="data row191 col6" >0.107200</td>
          <td id="T_78b05_row191_col7" class="data row191 col7" >-0.065900</td>
          <td id="T_78b05_row191_col8" class="data row191 col8" >0.005400</td>
          <td id="T_78b05_row191_col9" class="data row191 col9" >0.032000</td>
          <td id="T_78b05_row191_col10" class="data row191 col10" >0.035000</td>
          <td id="T_78b05_row191_col11" class="data row191 col11" >0.035700</td>
          <td id="T_78b05_row191_col12" class="data row191 col12" >0.036600</td>
          <td id="T_78b05_row191_col13" class="data row191 col13" >0.111100</td>
          <td id="T_78b05_row191_col14" class="data row191 col14" >0.064100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row192" class="row_heading level0 row192" >193</th>
          <td id="T_78b05_row192_col0" class="data row192 col0" >PC1</td>
          <td id="T_78b05_row192_col1" class="data row192 col1" >0.024700</td>
          <td id="T_78b05_row192_col2" class="data row192 col2" >0.056900</td>
          <td id="T_78b05_row192_col3" class="data row192 col3" >-0.051300</td>
          <td id="T_78b05_row192_col4" class="data row192 col4" >-0.035600</td>
          <td id="T_78b05_row192_col5" class="data row192 col5" >0.029000</td>
          <td id="T_78b05_row192_col6" class="data row192 col6" >-0.057200</td>
          <td id="T_78b05_row192_col7" class="data row192 col7" >-0.087300</td>
          <td id="T_78b05_row192_col8" class="data row192 col8" >0.020800</td>
          <td id="T_78b05_row192_col9" class="data row192 col9" >0.086700</td>
          <td id="T_78b05_row192_col10" class="data row192 col10" >0.082100</td>
          <td id="T_78b05_row192_col11" class="data row192 col11" >0.017700</td>
          <td id="T_78b05_row192_col12" class="data row192 col12" >0.029700</td>
          <td id="T_78b05_row192_col13" class="data row192 col13" >0.053300</td>
          <td id="T_78b05_row192_col14" class="data row192 col14" >0.085500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row193" class="row_heading level0 row193" >194</th>
          <td id="T_78b05_row193_col0" class="data row193 col0" >None</td>
          <td id="T_78b05_row193_col1" class="data row193 col1" >0.038400</td>
          <td id="T_78b05_row193_col2" class="data row193 col2" >-0.048300</td>
          <td id="T_78b05_row193_col3" class="data row193 col3" >-0.035900</td>
          <td id="T_78b05_row193_col4" class="data row193 col4" >-0.049200</td>
          <td id="T_78b05_row193_col5" class="data row193 col5" >0.009000</td>
          <td id="T_78b05_row193_col6" class="data row193 col6" >-0.023800</td>
          <td id="T_78b05_row193_col7" class="data row193 col7" >-0.029000</td>
          <td id="T_78b05_row193_col8" class="data row193 col8" >0.007100</td>
          <td id="T_78b05_row193_col9" class="data row193 col9" >0.018500</td>
          <td id="T_78b05_row193_col10" class="data row193 col10" >0.066600</td>
          <td id="T_78b05_row193_col11" class="data row193 col11" >0.031300</td>
          <td id="T_78b05_row193_col12" class="data row193 col12" >0.009600</td>
          <td id="T_78b05_row193_col13" class="data row193 col13" >0.019900</td>
          <td id="T_78b05_row193_col14" class="data row193 col14" >0.027200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row194" class="row_heading level0 row194" >195</th>
          <td id="T_78b05_row194_col0" class="data row194 col0" >PC5</td>
          <td id="T_78b05_row194_col1" class="data row194 col1" >0.029900</td>
          <td id="T_78b05_row194_col2" class="data row194 col2" >-0.006500</td>
          <td id="T_78b05_row194_col3" class="data row194 col3" >-0.035800</td>
          <td id="T_78b05_row194_col4" class="data row194 col4" >0.050200</td>
          <td id="T_78b05_row194_col5" class="data row194 col5" >-0.104000</td>
          <td id="T_78b05_row194_col6" class="data row194 col6" >-0.026700</td>
          <td id="T_78b05_row194_col7" class="data row194 col7" >0.072300</td>
          <td id="T_78b05_row194_col8" class="data row194 col8" >0.015600</td>
          <td id="T_78b05_row194_col9" class="data row194 col9" >0.023300</td>
          <td id="T_78b05_row194_col10" class="data row194 col10" >0.066500</td>
          <td id="T_78b05_row194_col11" class="data row194 col11" >0.068100</td>
          <td id="T_78b05_row194_col12" class="data row194 col12" >0.103300</td>
          <td id="T_78b05_row194_col13" class="data row194 col13" >0.022800</td>
          <td id="T_78b05_row194_col14" class="data row194 col14" >0.074100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row195" class="row_heading level0 row195" >196</th>
          <td id="T_78b05_row195_col0" class="data row195 col0" >None</td>
          <td id="T_78b05_row195_col1" class="data row195 col1" >0.034200</td>
          <td id="T_78b05_row195_col2" class="data row195 col2" >0.040300</td>
          <td id="T_78b05_row195_col3" class="data row195 col3" >-0.048900</td>
          <td id="T_78b05_row195_col4" class="data row195 col4" >0.016900</td>
          <td id="T_78b05_row195_col5" class="data row195 col5" >-0.001200</td>
          <td id="T_78b05_row195_col6" class="data row195 col6" >-0.059000</td>
          <td id="T_78b05_row195_col7" class="data row195 col7" >-0.020800</td>
          <td id="T_78b05_row195_col8" class="data row195 col8" >0.011300</td>
          <td id="T_78b05_row195_col9" class="data row195 col9" >0.070200</td>
          <td id="T_78b05_row195_col10" class="data row195 col10" >0.079600</td>
          <td id="T_78b05_row195_col11" class="data row195 col11" >0.034800</td>
          <td id="T_78b05_row195_col12" class="data row195 col12" >0.000500</td>
          <td id="T_78b05_row195_col13" class="data row195 col13" >0.055100</td>
          <td id="T_78b05_row195_col14" class="data row195 col14" >0.019100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row196" class="row_heading level0 row196" >197</th>
          <td id="T_78b05_row196_col0" class="data row196 col0" >None</td>
          <td id="T_78b05_row196_col1" class="data row196 col1" >0.033000</td>
          <td id="T_78b05_row196_col2" class="data row196 col2" >0.014300</td>
          <td id="T_78b05_row196_col3" class="data row196 col3" >-0.036800</td>
          <td id="T_78b05_row196_col4" class="data row196 col4" >0.024400</td>
          <td id="T_78b05_row196_col5" class="data row196 col5" >0.000600</td>
          <td id="T_78b05_row196_col6" class="data row196 col6" >-0.045300</td>
          <td id="T_78b05_row196_col7" class="data row196 col7" >0.022100</td>
          <td id="T_78b05_row196_col8" class="data row196 col8" >0.012500</td>
          <td id="T_78b05_row196_col9" class="data row196 col9" >0.044200</td>
          <td id="T_78b05_row196_col10" class="data row196 col10" >0.067500</td>
          <td id="T_78b05_row196_col11" class="data row196 col11" >0.042300</td>
          <td id="T_78b05_row196_col12" class="data row196 col12" >0.001300</td>
          <td id="T_78b05_row196_col13" class="data row196 col13" >0.041400</td>
          <td id="T_78b05_row196_col14" class="data row196 col14" >0.023800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row197" class="row_heading level0 row197" >198</th>
          <td id="T_78b05_row197_col0" class="data row197 col0" >None</td>
          <td id="T_78b05_row197_col1" class="data row197 col1" >0.036800</td>
          <td id="T_78b05_row197_col2" class="data row197 col2" >-0.036200</td>
          <td id="T_78b05_row197_col3" class="data row197 col3" >-0.062200</td>
          <td id="T_78b05_row197_col4" class="data row197 col4" >-0.044500</td>
          <td id="T_78b05_row197_col5" class="data row197 col5" >0.041100</td>
          <td id="T_78b05_row197_col6" class="data row197 col6" >-0.062400</td>
          <td id="T_78b05_row197_col7" class="data row197 col7" >0.021600</td>
          <td id="T_78b05_row197_col8" class="data row197 col8" >0.008800</td>
          <td id="T_78b05_row197_col9" class="data row197 col9" >0.006300</td>
          <td id="T_78b05_row197_col10" class="data row197 col10" >0.093000</td>
          <td id="T_78b05_row197_col11" class="data row197 col11" >0.026600</td>
          <td id="T_78b05_row197_col12" class="data row197 col12" >0.041700</td>
          <td id="T_78b05_row197_col13" class="data row197 col13" >0.058500</td>
          <td id="T_78b05_row197_col14" class="data row197 col14" >0.023400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row198" class="row_heading level0 row198" >199</th>
          <td id="T_78b05_row198_col0" class="data row198 col0" >None</td>
          <td id="T_78b05_row198_col1" class="data row198 col1" >0.041200</td>
          <td id="T_78b05_row198_col2" class="data row198 col2" >0.024800</td>
          <td id="T_78b05_row198_col3" class="data row198 col3" >0.010300</td>
          <td id="T_78b05_row198_col4" class="data row198 col4" >0.024200</td>
          <td id="T_78b05_row198_col5" class="data row198 col5" >0.014800</td>
          <td id="T_78b05_row198_col6" class="data row198 col6" >0.048200</td>
          <td id="T_78b05_row198_col7" class="data row198 col7" >-0.017800</td>
          <td id="T_78b05_row198_col8" class="data row198 col8" >0.004300</td>
          <td id="T_78b05_row198_col9" class="data row198 col9" >0.054600</td>
          <td id="T_78b05_row198_col10" class="data row198 col10" >0.020400</td>
          <td id="T_78b05_row198_col11" class="data row198 col11" >0.042100</td>
          <td id="T_78b05_row198_col12" class="data row198 col12" >0.015400</td>
          <td id="T_78b05_row198_col13" class="data row198 col13" >0.052100</td>
          <td id="T_78b05_row198_col14" class="data row198 col14" >0.016100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row199" class="row_heading level0 row199" >200</th>
          <td id="T_78b05_row199_col0" class="data row199 col0" >PC1</td>
          <td id="T_78b05_row199_col1" class="data row199 col1" >0.021200</td>
          <td id="T_78b05_row199_col2" class="data row199 col2" >0.056200</td>
          <td id="T_78b05_row199_col3" class="data row199 col3" >-0.005700</td>
          <td id="T_78b05_row199_col4" class="data row199 col4" >0.072600</td>
          <td id="T_78b05_row199_col5" class="data row199 col5" >-0.114900</td>
          <td id="T_78b05_row199_col6" class="data row199 col6" >-0.003800</td>
          <td id="T_78b05_row199_col7" class="data row199 col7" >-0.023600</td>
          <td id="T_78b05_row199_col8" class="data row199 col8" >0.024300</td>
          <td id="T_78b05_row199_col9" class="data row199 col9" >0.086100</td>
          <td id="T_78b05_row199_col10" class="data row199 col10" >0.036500</td>
          <td id="T_78b05_row199_col11" class="data row199 col11" >0.090500</td>
          <td id="T_78b05_row199_col12" class="data row199 col12" >0.114300</td>
          <td id="T_78b05_row199_col13" class="data row199 col13" >0.000000</td>
          <td id="T_78b05_row199_col14" class="data row199 col14" >0.021800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row200" class="row_heading level0 row200" >201</th>
          <td id="T_78b05_row200_col0" class="data row200 col0" >None</td>
          <td id="T_78b05_row200_col1" class="data row200 col1" >0.026600</td>
          <td id="T_78b05_row200_col2" class="data row200 col2" >-0.028300</td>
          <td id="T_78b05_row200_col3" class="data row200 col3" >-0.074200</td>
          <td id="T_78b05_row200_col4" class="data row200 col4" >0.048000</td>
          <td id="T_78b05_row200_col5" class="data row200 col5" >0.025000</td>
          <td id="T_78b05_row200_col6" class="data row200 col6" >-0.025500</td>
          <td id="T_78b05_row200_col7" class="data row200 col7" >-0.076800</td>
          <td id="T_78b05_row200_col8" class="data row200 col8" >0.018900</td>
          <td id="T_78b05_row200_col9" class="data row200 col9" >0.001500</td>
          <td id="T_78b05_row200_col10" class="data row200 col10" >0.105000</td>
          <td id="T_78b05_row200_col11" class="data row200 col11" >0.065900</td>
          <td id="T_78b05_row200_col12" class="data row200 col12" >0.025700</td>
          <td id="T_78b05_row200_col13" class="data row200 col13" >0.021600</td>
          <td id="T_78b05_row200_col14" class="data row200 col14" >0.075100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row201" class="row_heading level0 row201" >202</th>
          <td id="T_78b05_row201_col0" class="data row201 col0" >None</td>
          <td id="T_78b05_row201_col1" class="data row201 col1" >0.027800</td>
          <td id="T_78b05_row201_col2" class="data row201 col2" >0.050000</td>
          <td id="T_78b05_row201_col3" class="data row201 col3" >-0.064200</td>
          <td id="T_78b05_row201_col4" class="data row201 col4" >0.046200</td>
          <td id="T_78b05_row201_col5" class="data row201 col5" >0.007600</td>
          <td id="T_78b05_row201_col6" class="data row201 col6" >-0.055400</td>
          <td id="T_78b05_row201_col7" class="data row201 col7" >-0.004700</td>
          <td id="T_78b05_row201_col8" class="data row201 col8" >0.017700</td>
          <td id="T_78b05_row201_col9" class="data row201 col9" >0.079900</td>
          <td id="T_78b05_row201_col10" class="data row201 col10" >0.094900</td>
          <td id="T_78b05_row201_col11" class="data row201 col11" >0.064200</td>
          <td id="T_78b05_row201_col12" class="data row201 col12" >0.008200</td>
          <td id="T_78b05_row201_col13" class="data row201 col13" >0.051600</td>
          <td id="T_78b05_row201_col14" class="data row201 col14" >0.002900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row202" class="row_heading level0 row202" >203</th>
          <td id="T_78b05_row202_col0" class="data row202 col0" >None</td>
          <td id="T_78b05_row202_col1" class="data row202 col1" >0.036000</td>
          <td id="T_78b05_row202_col2" class="data row202 col2" >0.056000</td>
          <td id="T_78b05_row202_col3" class="data row202 col3" >0.030600</td>
          <td id="T_78b05_row202_col4" class="data row202 col4" >0.027000</td>
          <td id="T_78b05_row202_col5" class="data row202 col5" >-0.074200</td>
          <td id="T_78b05_row202_col6" class="data row202 col6" >-0.003700</td>
          <td id="T_78b05_row202_col7" class="data row202 col7" >-0.061100</td>
          <td id="T_78b05_row202_col8" class="data row202 col8" >0.009500</td>
          <td id="T_78b05_row202_col9" class="data row202 col9" >0.085800</td>
          <td id="T_78b05_row202_col10" class="data row202 col10" >0.000100</td>
          <td id="T_78b05_row202_col11" class="data row202 col11" >0.044900</td>
          <td id="T_78b05_row202_col12" class="data row202 col12" >0.073500</td>
          <td id="T_78b05_row202_col13" class="data row202 col13" >0.000200</td>
          <td id="T_78b05_row202_col14" class="data row202 col14" >0.059400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row203" class="row_heading level0 row203" >204</th>
          <td id="T_78b05_row203_col0" class="data row203 col0" >PC1</td>
          <td id="T_78b05_row203_col1" class="data row203 col1" >0.025500</td>
          <td id="T_78b05_row203_col2" class="data row203 col2" >0.007100</td>
          <td id="T_78b05_row203_col3" class="data row203 col3" >-0.062900</td>
          <td id="T_78b05_row203_col4" class="data row203 col4" >-0.052500</td>
          <td id="T_78b05_row203_col5" class="data row203 col5" >0.021400</td>
          <td id="T_78b05_row203_col6" class="data row203 col6" >-0.028200</td>
          <td id="T_78b05_row203_col7" class="data row203 col7" >0.007400</td>
          <td id="T_78b05_row203_col8" class="data row203 col8" >0.020000</td>
          <td id="T_78b05_row203_col9" class="data row203 col9" >0.036900</td>
          <td id="T_78b05_row203_col10" class="data row203 col10" >0.093700</td>
          <td id="T_78b05_row203_col11" class="data row203 col11" >0.034500</td>
          <td id="T_78b05_row203_col12" class="data row203 col12" >0.022100</td>
          <td id="T_78b05_row203_col13" class="data row203 col13" >0.024300</td>
          <td id="T_78b05_row203_col14" class="data row203 col14" >0.009200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row204" class="row_heading level0 row204" >205</th>
          <td id="T_78b05_row204_col0" class="data row204 col0" >None</td>
          <td id="T_78b05_row204_col1" class="data row204 col1" >0.042800</td>
          <td id="T_78b05_row204_col2" class="data row204 col2" >-0.051500</td>
          <td id="T_78b05_row204_col3" class="data row204 col3" >-0.045200</td>
          <td id="T_78b05_row204_col4" class="data row204 col4" >-0.020800</td>
          <td id="T_78b05_row204_col5" class="data row204 col5" >-0.032900</td>
          <td id="T_78b05_row204_col6" class="data row204 col6" >0.000900</td>
          <td id="T_78b05_row204_col7" class="data row204 col7" >0.031800</td>
          <td id="T_78b05_row204_col8" class="data row204 col8" >0.002700</td>
          <td id="T_78b05_row204_col9" class="data row204 col9" >0.021700</td>
          <td id="T_78b05_row204_col10" class="data row204 col10" >0.075900</td>
          <td id="T_78b05_row204_col11" class="data row204 col11" >0.002900</td>
          <td id="T_78b05_row204_col12" class="data row204 col12" >0.032200</td>
          <td id="T_78b05_row204_col13" class="data row204 col13" >0.004800</td>
          <td id="T_78b05_row204_col14" class="data row204 col14" >0.033600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row205" class="row_heading level0 row205" >206</th>
          <td id="T_78b05_row205_col0" class="data row205 col0" >None</td>
          <td id="T_78b05_row205_col1" class="data row205 col1" >0.035100</td>
          <td id="T_78b05_row205_col2" class="data row205 col2" >0.051200</td>
          <td id="T_78b05_row205_col3" class="data row205 col3" >-0.002800</td>
          <td id="T_78b05_row205_col4" class="data row205 col4" >-0.022400</td>
          <td id="T_78b05_row205_col5" class="data row205 col5" >-0.003600</td>
          <td id="T_78b05_row205_col6" class="data row205 col6" >-0.010100</td>
          <td id="T_78b05_row205_col7" class="data row205 col7" >0.002500</td>
          <td id="T_78b05_row205_col8" class="data row205 col8" >0.010400</td>
          <td id="T_78b05_row205_col9" class="data row205 col9" >0.081000</td>
          <td id="T_78b05_row205_col10" class="data row205 col10" >0.033600</td>
          <td id="T_78b05_row205_col11" class="data row205 col11" >0.004500</td>
          <td id="T_78b05_row205_col12" class="data row205 col12" >0.003000</td>
          <td id="T_78b05_row205_col13" class="data row205 col13" >0.006200</td>
          <td id="T_78b05_row205_col14" class="data row205 col14" >0.004300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row206" class="row_heading level0 row206" >207</th>
          <td id="T_78b05_row206_col0" class="data row206 col0" >None</td>
          <td id="T_78b05_row206_col1" class="data row206 col1" >0.047000</td>
          <td id="T_78b05_row206_col2" class="data row206 col2" >0.033700</td>
          <td id="T_78b05_row206_col3" class="data row206 col3" >0.062200</td>
          <td id="T_78b05_row206_col4" class="data row206 col4" >0.005100</td>
          <td id="T_78b05_row206_col5" class="data row206 col5" >0.064500</td>
          <td id="T_78b05_row206_col6" class="data row206 col6" >0.080700</td>
          <td id="T_78b05_row206_col7" class="data row206 col7" >0.029400</td>
          <td id="T_78b05_row206_col8" class="data row206 col8" >0.001500</td>
          <td id="T_78b05_row206_col9" class="data row206 col9" >0.063500</td>
          <td id="T_78b05_row206_col10" class="data row206 col10" >0.031400</td>
          <td id="T_78b05_row206_col11" class="data row206 col11" >0.023000</td>
          <td id="T_78b05_row206_col12" class="data row206 col12" >0.065200</td>
          <td id="T_78b05_row206_col13" class="data row206 col13" >0.084600</td>
          <td id="T_78b05_row206_col14" class="data row206 col14" >0.031100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row207" class="row_heading level0 row207" >208</th>
          <td id="T_78b05_row207_col0" class="data row207 col0" >None</td>
          <td id="T_78b05_row207_col1" class="data row207 col1" >0.031300</td>
          <td id="T_78b05_row207_col2" class="data row207 col2" >0.038000</td>
          <td id="T_78b05_row207_col3" class="data row207 col3" >0.040500</td>
          <td id="T_78b05_row207_col4" class="data row207 col4" >-0.011300</td>
          <td id="T_78b05_row207_col5" class="data row207 col5" >-0.049700</td>
          <td id="T_78b05_row207_col6" class="data row207 col6" >0.063000</td>
          <td id="T_78b05_row207_col7" class="data row207 col7" >0.015000</td>
          <td id="T_78b05_row207_col8" class="data row207 col8" >0.014200</td>
          <td id="T_78b05_row207_col9" class="data row207 col9" >0.067800</td>
          <td id="T_78b05_row207_col10" class="data row207 col10" >0.009800</td>
          <td id="T_78b05_row207_col11" class="data row207 col11" >0.006600</td>
          <td id="T_78b05_row207_col12" class="data row207 col12" >0.049000</td>
          <td id="T_78b05_row207_col13" class="data row207 col13" >0.066900</td>
          <td id="T_78b05_row207_col14" class="data row207 col14" >0.016800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row208" class="row_heading level0 row208" >209</th>
          <td id="T_78b05_row208_col0" class="data row208 col0" >None</td>
          <td id="T_78b05_row208_col1" class="data row208 col1" >0.042500</td>
          <td id="T_78b05_row208_col2" class="data row208 col2" >0.024400</td>
          <td id="T_78b05_row208_col3" class="data row208 col3" >-0.010400</td>
          <td id="T_78b05_row208_col4" class="data row208 col4" >0.012500</td>
          <td id="T_78b05_row208_col5" class="data row208 col5" >0.045600</td>
          <td id="T_78b05_row208_col6" class="data row208 col6" >0.095900</td>
          <td id="T_78b05_row208_col7" class="data row208 col7" >0.041300</td>
          <td id="T_78b05_row208_col8" class="data row208 col8" >0.003000</td>
          <td id="T_78b05_row208_col9" class="data row208 col9" >0.054300</td>
          <td id="T_78b05_row208_col10" class="data row208 col10" >0.041100</td>
          <td id="T_78b05_row208_col11" class="data row208 col11" >0.030400</td>
          <td id="T_78b05_row208_col12" class="data row208 col12" >0.046300</td>
          <td id="T_78b05_row208_col13" class="data row208 col13" >0.099800</td>
          <td id="T_78b05_row208_col14" class="data row208 col14" >0.043100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row209" class="row_heading level0 row209" >210</th>
          <td id="T_78b05_row209_col0" class="data row209 col0" >None</td>
          <td id="T_78b05_row209_col1" class="data row209 col1" >0.035700</td>
          <td id="T_78b05_row209_col2" class="data row209 col2" >-0.026000</td>
          <td id="T_78b05_row209_col3" class="data row209 col3" >0.018700</td>
          <td id="T_78b05_row209_col4" class="data row209 col4" >0.011800</td>
          <td id="T_78b05_row209_col5" class="data row209 col5" >-0.023200</td>
          <td id="T_78b05_row209_col6" class="data row209 col6" >0.017700</td>
          <td id="T_78b05_row209_col7" class="data row209 col7" >-0.027000</td>
          <td id="T_78b05_row209_col8" class="data row209 col8" >0.009800</td>
          <td id="T_78b05_row209_col9" class="data row209 col9" >0.003800</td>
          <td id="T_78b05_row209_col10" class="data row209 col10" >0.012100</td>
          <td id="T_78b05_row209_col11" class="data row209 col11" >0.029700</td>
          <td id="T_78b05_row209_col12" class="data row209 col12" >0.022600</td>
          <td id="T_78b05_row209_col13" class="data row209 col13" >0.021600</td>
          <td id="T_78b05_row209_col14" class="data row209 col14" >0.025200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row210" class="row_heading level0 row210" >211</th>
          <td id="T_78b05_row210_col0" class="data row210 col0" >None</td>
          <td id="T_78b05_row210_col1" class="data row210 col1" >0.033700</td>
          <td id="T_78b05_row210_col2" class="data row210 col2" >0.006100</td>
          <td id="T_78b05_row210_col3" class="data row210 col3" >-0.037500</td>
          <td id="T_78b05_row210_col4" class="data row210 col4" >0.019300</td>
          <td id="T_78b05_row210_col5" class="data row210 col5" >0.059700</td>
          <td id="T_78b05_row210_col6" class="data row210 col6" >0.044100</td>
          <td id="T_78b05_row210_col7" class="data row210 col7" >-0.012500</td>
          <td id="T_78b05_row210_col8" class="data row210 col8" >0.011900</td>
          <td id="T_78b05_row210_col9" class="data row210 col9" >0.035900</td>
          <td id="T_78b05_row210_col10" class="data row210 col10" >0.068200</td>
          <td id="T_78b05_row210_col11" class="data row210 col11" >0.037200</td>
          <td id="T_78b05_row210_col12" class="data row210 col12" >0.060400</td>
          <td id="T_78b05_row210_col13" class="data row210 col13" >0.048000</td>
          <td id="T_78b05_row210_col14" class="data row210 col14" >0.010700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row211" class="row_heading level0 row211" >212</th>
          <td id="T_78b05_row211_col0" class="data row211 col0" >None</td>
          <td id="T_78b05_row211_col1" class="data row211 col1" >0.045000</td>
          <td id="T_78b05_row211_col2" class="data row211 col2" >0.025700</td>
          <td id="T_78b05_row211_col3" class="data row211 col3" >0.090700</td>
          <td id="T_78b05_row211_col4" class="data row211 col4" >-0.022900</td>
          <td id="T_78b05_row211_col5" class="data row211 col5" >-0.033300</td>
          <td id="T_78b05_row211_col6" class="data row211 col6" >0.038000</td>
          <td id="T_78b05_row211_col7" class="data row211 col7" >0.007100</td>
          <td id="T_78b05_row211_col8" class="data row211 col8" >0.000500</td>
          <td id="T_78b05_row211_col9" class="data row211 col9" >0.055500</td>
          <td id="T_78b05_row211_col10" class="data row211 col10" >0.059900</td>
          <td id="T_78b05_row211_col11" class="data row211 col11" >0.005000</td>
          <td id="T_78b05_row211_col12" class="data row211 col12" >0.032600</td>
          <td id="T_78b05_row211_col13" class="data row211 col13" >0.041900</td>
          <td id="T_78b05_row211_col14" class="data row211 col14" >0.008800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row212" class="row_heading level0 row212" >213</th>
          <td id="T_78b05_row212_col0" class="data row212 col0" >None</td>
          <td id="T_78b05_row212_col1" class="data row212 col1" >0.043400</td>
          <td id="T_78b05_row212_col2" class="data row212 col2" >0.028200</td>
          <td id="T_78b05_row212_col3" class="data row212 col3" >0.079900</td>
          <td id="T_78b05_row212_col4" class="data row212 col4" >-0.020800</td>
          <td id="T_78b05_row212_col5" class="data row212 col5" >-0.034100</td>
          <td id="T_78b05_row212_col6" class="data row212 col6" >0.005800</td>
          <td id="T_78b05_row212_col7" class="data row212 col7" >0.011300</td>
          <td id="T_78b05_row212_col8" class="data row212 col8" >0.002100</td>
          <td id="T_78b05_row212_col9" class="data row212 col9" >0.058000</td>
          <td id="T_78b05_row212_col10" class="data row212 col10" >0.049200</td>
          <td id="T_78b05_row212_col11" class="data row212 col11" >0.002900</td>
          <td id="T_78b05_row212_col12" class="data row212 col12" >0.033400</td>
          <td id="T_78b05_row212_col13" class="data row212 col13" >0.009700</td>
          <td id="T_78b05_row212_col14" class="data row212 col14" >0.013000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row213" class="row_heading level0 row213" >214</th>
          <td id="T_78b05_row213_col0" class="data row213 col0" >None</td>
          <td id="T_78b05_row213_col1" class="data row213 col1" >0.034000</td>
          <td id="T_78b05_row213_col2" class="data row213 col2" >-0.013800</td>
          <td id="T_78b05_row213_col3" class="data row213 col3" >-0.025000</td>
          <td id="T_78b05_row213_col4" class="data row213 col4" >-0.059500</td>
          <td id="T_78b05_row213_col5" class="data row213 col5" >-0.006400</td>
          <td id="T_78b05_row213_col6" class="data row213 col6" >-0.015700</td>
          <td id="T_78b05_row213_col7" class="data row213 col7" >0.069700</td>
          <td id="T_78b05_row213_col8" class="data row213 col8" >0.011500</td>
          <td id="T_78b05_row213_col9" class="data row213 col9" >0.016000</td>
          <td id="T_78b05_row213_col10" class="data row213 col10" >0.055800</td>
          <td id="T_78b05_row213_col11" class="data row213 col11" >0.041600</td>
          <td id="T_78b05_row213_col12" class="data row213 col12" >0.005800</td>
          <td id="T_78b05_row213_col13" class="data row213 col13" >0.011800</td>
          <td id="T_78b05_row213_col14" class="data row213 col14" >0.071400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row214" class="row_heading level0 row214" >215</th>
          <td id="T_78b05_row214_col0" class="data row214 col0" >None</td>
          <td id="T_78b05_row214_col1" class="data row214 col1" >0.037100</td>
          <td id="T_78b05_row214_col2" class="data row214 col2" >0.000100</td>
          <td id="T_78b05_row214_col3" class="data row214 col3" >-0.068400</td>
          <td id="T_78b05_row214_col4" class="data row214 col4" >0.069600</td>
          <td id="T_78b05_row214_col5" class="data row214 col5" >-0.029700</td>
          <td id="T_78b05_row214_col6" class="data row214 col6" >-0.074400</td>
          <td id="T_78b05_row214_col7" class="data row214 col7" >-0.044000</td>
          <td id="T_78b05_row214_col8" class="data row214 col8" >0.008500</td>
          <td id="T_78b05_row214_col9" class="data row214 col9" >0.029900</td>
          <td id="T_78b05_row214_col10" class="data row214 col10" >0.099200</td>
          <td id="T_78b05_row214_col11" class="data row214 col11" >0.087500</td>
          <td id="T_78b05_row214_col12" class="data row214 col12" >0.029100</td>
          <td id="T_78b05_row214_col13" class="data row214 col13" >0.070500</td>
          <td id="T_78b05_row214_col14" class="data row214 col14" >0.042300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row215" class="row_heading level0 row215" >216</th>
          <td id="T_78b05_row215_col0" class="data row215 col0" >None</td>
          <td id="T_78b05_row215_col1" class="data row215 col1" >0.043900</td>
          <td id="T_78b05_row215_col2" class="data row215 col2" >-0.003300</td>
          <td id="T_78b05_row215_col3" class="data row215 col3" >0.048100</td>
          <td id="T_78b05_row215_col4" class="data row215 col4" >0.009600</td>
          <td id="T_78b05_row215_col5" class="data row215 col5" >-0.026400</td>
          <td id="T_78b05_row215_col6" class="data row215 col6" >-0.047500</td>
          <td id="T_78b05_row215_col7" class="data row215 col7" >-0.088600</td>
          <td id="T_78b05_row215_col8" class="data row215 col8" >0.001700</td>
          <td id="T_78b05_row215_col9" class="data row215 col9" >0.026500</td>
          <td id="T_78b05_row215_col10" class="data row215 col10" >0.017400</td>
          <td id="T_78b05_row215_col11" class="data row215 col11" >0.027500</td>
          <td id="T_78b05_row215_col12" class="data row215 col12" >0.025800</td>
          <td id="T_78b05_row215_col13" class="data row215 col13" >0.043600</td>
          <td id="T_78b05_row215_col14" class="data row215 col14" >0.086900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row216" class="row_heading level0 row216" >217</th>
          <td id="T_78b05_row216_col0" class="data row216 col0" >None</td>
          <td id="T_78b05_row216_col1" class="data row216 col1" >0.041700</td>
          <td id="T_78b05_row216_col2" class="data row216 col2" >0.027500</td>
          <td id="T_78b05_row216_col3" class="data row216 col3" >-0.018400</td>
          <td id="T_78b05_row216_col4" class="data row216 col4" >-0.008200</td>
          <td id="T_78b05_row216_col5" class="data row216 col5" >-0.007200</td>
          <td id="T_78b05_row216_col6" class="data row216 col6" >0.009600</td>
          <td id="T_78b05_row216_col7" class="data row216 col7" >0.046000</td>
          <td id="T_78b05_row216_col8" class="data row216 col8" >0.003800</td>
          <td id="T_78b05_row216_col9" class="data row216 col9" >0.057300</td>
          <td id="T_78b05_row216_col10" class="data row216 col10" >0.049100</td>
          <td id="T_78b05_row216_col11" class="data row216 col11" >0.009700</td>
          <td id="T_78b05_row216_col12" class="data row216 col12" >0.006600</td>
          <td id="T_78b05_row216_col13" class="data row216 col13" >0.013500</td>
          <td id="T_78b05_row216_col14" class="data row216 col14" >0.047800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row217" class="row_heading level0 row217" >218</th>
          <td id="T_78b05_row217_col0" class="data row217 col0" >None</td>
          <td id="T_78b05_row217_col1" class="data row217 col1" >0.042200</td>
          <td id="T_78b05_row217_col2" class="data row217 col2" >-0.068100</td>
          <td id="T_78b05_row217_col3" class="data row217 col3" >0.006300</td>
          <td id="T_78b05_row217_col4" class="data row217 col4" >-0.020700</td>
          <td id="T_78b05_row217_col5" class="data row217 col5" >-0.023500</td>
          <td id="T_78b05_row217_col6" class="data row217 col6" >0.022900</td>
          <td id="T_78b05_row217_col7" class="data row217 col7" >-0.038700</td>
          <td id="T_78b05_row217_col8" class="data row217 col8" >0.003400</td>
          <td id="T_78b05_row217_col9" class="data row217 col9" >0.038300</td>
          <td id="T_78b05_row217_col10" class="data row217 col10" >0.024500</td>
          <td id="T_78b05_row217_col11" class="data row217 col11" >0.002800</td>
          <td id="T_78b05_row217_col12" class="data row217 col12" >0.022900</td>
          <td id="T_78b05_row217_col13" class="data row217 col13" >0.026800</td>
          <td id="T_78b05_row217_col14" class="data row217 col14" >0.037000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row218" class="row_heading level0 row218" >219</th>
          <td id="T_78b05_row218_col0" class="data row218 col0" >None</td>
          <td id="T_78b05_row218_col1" class="data row218 col1" >0.036500</td>
          <td id="T_78b05_row218_col2" class="data row218 col2" >-0.017900</td>
          <td id="T_78b05_row218_col3" class="data row218 col3" >-0.001900</td>
          <td id="T_78b05_row218_col4" class="data row218 col4" >-0.069000</td>
          <td id="T_78b05_row218_col5" class="data row218 col5" >-0.006700</td>
          <td id="T_78b05_row218_col6" class="data row218 col6" >-0.022900</td>
          <td id="T_78b05_row218_col7" class="data row218 col7" >0.022800</td>
          <td id="T_78b05_row218_col8" class="data row218 col8" >0.009100</td>
          <td id="T_78b05_row218_col9" class="data row218 col9" >0.011900</td>
          <td id="T_78b05_row218_col10" class="data row218 col10" >0.032600</td>
          <td id="T_78b05_row218_col11" class="data row218 col11" >0.051100</td>
          <td id="T_78b05_row218_col12" class="data row218 col12" >0.006000</td>
          <td id="T_78b05_row218_col13" class="data row218 col13" >0.019000</td>
          <td id="T_78b05_row218_col14" class="data row218 col14" >0.024500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row219" class="row_heading level0 row219" >220</th>
          <td id="T_78b05_row219_col0" class="data row219 col0" >None</td>
          <td id="T_78b05_row219_col1" class="data row219 col1" >0.028200</td>
          <td id="T_78b05_row219_col2" class="data row219 col2" >-0.020600</td>
          <td id="T_78b05_row219_col3" class="data row219 col3" >-0.087500</td>
          <td id="T_78b05_row219_col4" class="data row219 col4" >0.023700</td>
          <td id="T_78b05_row219_col5" class="data row219 col5" >-0.046800</td>
          <td id="T_78b05_row219_col6" class="data row219 col6" >0.005000</td>
          <td id="T_78b05_row219_col7" class="data row219 col7" >0.031500</td>
          <td id="T_78b05_row219_col8" class="data row219 col8" >0.017400</td>
          <td id="T_78b05_row219_col9" class="data row219 col9" >0.009300</td>
          <td id="T_78b05_row219_col10" class="data row219 col10" >0.118300</td>
          <td id="T_78b05_row219_col11" class="data row219 col11" >0.041600</td>
          <td id="T_78b05_row219_col12" class="data row219 col12" >0.046200</td>
          <td id="T_78b05_row219_col13" class="data row219 col13" >0.008900</td>
          <td id="T_78b05_row219_col14" class="data row219 col14" >0.033300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row220" class="row_heading level0 row220" >221</th>
          <td id="T_78b05_row220_col0" class="data row220 col0" >None</td>
          <td id="T_78b05_row220_col1" class="data row220 col1" >0.044600</td>
          <td id="T_78b05_row220_col2" class="data row220 col2" >-0.088800</td>
          <td id="T_78b05_row220_col3" class="data row220 col3" >-0.014400</td>
          <td id="T_78b05_row220_col4" class="data row220 col4" >-0.033100</td>
          <td id="T_78b05_row220_col5" class="data row220 col5" >-0.047900</td>
          <td id="T_78b05_row220_col6" class="data row220 col6" >0.051600</td>
          <td id="T_78b05_row220_col7" class="data row220 col7" >-0.008600</td>
          <td id="T_78b05_row220_col8" class="data row220 col8" >0.000900</td>
          <td id="T_78b05_row220_col9" class="data row220 col9" >0.059000</td>
          <td id="T_78b05_row220_col10" class="data row220 col10" >0.045100</td>
          <td id="T_78b05_row220_col11" class="data row220 col11" >0.015200</td>
          <td id="T_78b05_row220_col12" class="data row220 col12" >0.047200</td>
          <td id="T_78b05_row220_col13" class="data row220 col13" >0.055500</td>
          <td id="T_78b05_row220_col14" class="data row220 col14" >0.006800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row221" class="row_heading level0 row221" >222</th>
          <td id="T_78b05_row221_col0" class="data row221 col0" >None</td>
          <td id="T_78b05_row221_col1" class="data row221 col1" >0.035000</td>
          <td id="T_78b05_row221_col2" class="data row221 col2" >0.012100</td>
          <td id="T_78b05_row221_col3" class="data row221 col3" >0.022000</td>
          <td id="T_78b05_row221_col4" class="data row221 col4" >0.103200</td>
          <td id="T_78b05_row221_col5" class="data row221 col5" >0.013300</td>
          <td id="T_78b05_row221_col6" class="data row221 col6" >-0.028900</td>
          <td id="T_78b05_row221_col7" class="data row221 col7" >0.010000</td>
          <td id="T_78b05_row221_col8" class="data row221 col8" >0.010600</td>
          <td id="T_78b05_row221_col9" class="data row221 col9" >0.041900</td>
          <td id="T_78b05_row221_col10" class="data row221 col10" >0.008700</td>
          <td id="T_78b05_row221_col11" class="data row221 col11" >0.121100</td>
          <td id="T_78b05_row221_col12" class="data row221 col12" >0.014000</td>
          <td id="T_78b05_row221_col13" class="data row221 col13" >0.025000</td>
          <td id="T_78b05_row221_col14" class="data row221 col14" >0.011700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row222" class="row_heading level0 row222" >223</th>
          <td id="T_78b05_row222_col0" class="data row222 col0" >PC1</td>
          <td id="T_78b05_row222_col1" class="data row222 col1" >0.018800</td>
          <td id="T_78b05_row222_col2" class="data row222 col2" >0.043600</td>
          <td id="T_78b05_row222_col3" class="data row222 col3" >-0.048500</td>
          <td id="T_78b05_row222_col4" class="data row222 col4" >-0.072700</td>
          <td id="T_78b05_row222_col5" class="data row222 col5" >0.042500</td>
          <td id="T_78b05_row222_col6" class="data row222 col6" >-0.047400</td>
          <td id="T_78b05_row222_col7" class="data row222 col7" >-0.028800</td>
          <td id="T_78b05_row222_col8" class="data row222 col8" >0.026700</td>
          <td id="T_78b05_row222_col9" class="data row222 col9" >0.073500</td>
          <td id="T_78b05_row222_col10" class="data row222 col10" >0.079300</td>
          <td id="T_78b05_row222_col11" class="data row222 col11" >0.054800</td>
          <td id="T_78b05_row222_col12" class="data row222 col12" >0.043200</td>
          <td id="T_78b05_row222_col13" class="data row222 col13" >0.043500</td>
          <td id="T_78b05_row222_col14" class="data row222 col14" >0.027000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row223" class="row_heading level0 row223" >224</th>
          <td id="T_78b05_row223_col0" class="data row223 col0" >None</td>
          <td id="T_78b05_row223_col1" class="data row223 col1" >0.027800</td>
          <td id="T_78b05_row223_col2" class="data row223 col2" >0.042000</td>
          <td id="T_78b05_row223_col3" class="data row223 col3" >-0.033000</td>
          <td id="T_78b05_row223_col4" class="data row223 col4" >0.027200</td>
          <td id="T_78b05_row223_col5" class="data row223 col5" >-0.014400</td>
          <td id="T_78b05_row223_col6" class="data row223 col6" >-0.051600</td>
          <td id="T_78b05_row223_col7" class="data row223 col7" >0.015100</td>
          <td id="T_78b05_row223_col8" class="data row223 col8" >0.017700</td>
          <td id="T_78b05_row223_col9" class="data row223 col9" >0.071800</td>
          <td id="T_78b05_row223_col10" class="data row223 col10" >0.063700</td>
          <td id="T_78b05_row223_col11" class="data row223 col11" >0.045100</td>
          <td id="T_78b05_row223_col12" class="data row223 col12" >0.013800</td>
          <td id="T_78b05_row223_col13" class="data row223 col13" >0.047800</td>
          <td id="T_78b05_row223_col14" class="data row223 col14" >0.016800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row224" class="row_heading level0 row224" >225</th>
          <td id="T_78b05_row224_col0" class="data row224 col0" >None</td>
          <td id="T_78b05_row224_col1" class="data row224 col1" >0.038500</td>
          <td id="T_78b05_row224_col2" class="data row224 col2" >0.031800</td>
          <td id="T_78b05_row224_col3" class="data row224 col3" >-0.005500</td>
          <td id="T_78b05_row224_col4" class="data row224 col4" >0.037500</td>
          <td id="T_78b05_row224_col5" class="data row224 col5" >-0.046700</td>
          <td id="T_78b05_row224_col6" class="data row224 col6" >-0.057000</td>
          <td id="T_78b05_row224_col7" class="data row224 col7" >0.057900</td>
          <td id="T_78b05_row224_col8" class="data row224 col8" >0.007000</td>
          <td id="T_78b05_row224_col9" class="data row224 col9" >0.061700</td>
          <td id="T_78b05_row224_col10" class="data row224 col10" >0.036200</td>
          <td id="T_78b05_row224_col11" class="data row224 col11" >0.055400</td>
          <td id="T_78b05_row224_col12" class="data row224 col12" >0.046100</td>
          <td id="T_78b05_row224_col13" class="data row224 col13" >0.053100</td>
          <td id="T_78b05_row224_col14" class="data row224 col14" >0.059600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row225" class="row_heading level0 row225" >226</th>
          <td id="T_78b05_row225_col0" class="data row225 col0" >None</td>
          <td id="T_78b05_row225_col1" class="data row225 col1" >0.035900</td>
          <td id="T_78b05_row225_col2" class="data row225 col2" >-0.005600</td>
          <td id="T_78b05_row225_col3" class="data row225 col3" >0.009900</td>
          <td id="T_78b05_row225_col4" class="data row225 col4" >0.063000</td>
          <td id="T_78b05_row225_col5" class="data row225 col5" >-0.014500</td>
          <td id="T_78b05_row225_col6" class="data row225 col6" >-0.028500</td>
          <td id="T_78b05_row225_col7" class="data row225 col7" >0.065500</td>
          <td id="T_78b05_row225_col8" class="data row225 col8" >0.009600</td>
          <td id="T_78b05_row225_col9" class="data row225 col9" >0.024200</td>
          <td id="T_78b05_row225_col10" class="data row225 col10" >0.020900</td>
          <td id="T_78b05_row225_col11" class="data row225 col11" >0.081000</td>
          <td id="T_78b05_row225_col12" class="data row225 col12" >0.013900</td>
          <td id="T_78b05_row225_col13" class="data row225 col13" >0.024600</td>
          <td id="T_78b05_row225_col14" class="data row225 col14" >0.067200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row226" class="row_heading level0 row226" >227</th>
          <td id="T_78b05_row226_col0" class="data row226 col0" >None</td>
          <td id="T_78b05_row226_col1" class="data row226 col1" >0.040700</td>
          <td id="T_78b05_row226_col2" class="data row226 col2" >-0.031000</td>
          <td id="T_78b05_row226_col3" class="data row226 col3" >-0.029600</td>
          <td id="T_78b05_row226_col4" class="data row226 col4" >0.003400</td>
          <td id="T_78b05_row226_col5" class="data row226 col5" >-0.012300</td>
          <td id="T_78b05_row226_col6" class="data row226 col6" >0.040700</td>
          <td id="T_78b05_row226_col7" class="data row226 col7" >0.021300</td>
          <td id="T_78b05_row226_col8" class="data row226 col8" >0.004800</td>
          <td id="T_78b05_row226_col9" class="data row226 col9" >0.001200</td>
          <td id="T_78b05_row226_col10" class="data row226 col10" >0.060300</td>
          <td id="T_78b05_row226_col11" class="data row226 col11" >0.021300</td>
          <td id="T_78b05_row226_col12" class="data row226 col12" >0.011700</td>
          <td id="T_78b05_row226_col13" class="data row226 col13" >0.044600</td>
          <td id="T_78b05_row226_col14" class="data row226 col14" >0.023100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row227" class="row_heading level0 row227" >228</th>
          <td id="T_78b05_row227_col0" class="data row227 col0" >None</td>
          <td id="T_78b05_row227_col1" class="data row227 col1" >0.040300</td>
          <td id="T_78b05_row227_col2" class="data row227 col2" >-0.034200</td>
          <td id="T_78b05_row227_col3" class="data row227 col3" >0.024800</td>
          <td id="T_78b05_row227_col4" class="data row227 col4" >0.038000</td>
          <td id="T_78b05_row227_col5" class="data row227 col5" >-0.028400</td>
          <td id="T_78b05_row227_col6" class="data row227 col6" >-0.007900</td>
          <td id="T_78b05_row227_col7" class="data row227 col7" >-0.004200</td>
          <td id="T_78b05_row227_col8" class="data row227 col8" >0.005300</td>
          <td id="T_78b05_row227_col9" class="data row227 col9" >0.004400</td>
          <td id="T_78b05_row227_col10" class="data row227 col10" >0.005900</td>
          <td id="T_78b05_row227_col11" class="data row227 col11" >0.055900</td>
          <td id="T_78b05_row227_col12" class="data row227 col12" >0.027700</td>
          <td id="T_78b05_row227_col13" class="data row227 col13" >0.004000</td>
          <td id="T_78b05_row227_col14" class="data row227 col14" >0.002400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row228" class="row_heading level0 row228" >229</th>
          <td id="T_78b05_row228_col0" class="data row228 col0" >None</td>
          <td id="T_78b05_row228_col1" class="data row228 col1" >0.038800</td>
          <td id="T_78b05_row228_col2" class="data row228 col2" >0.011800</td>
          <td id="T_78b05_row228_col3" class="data row228 col3" >-0.025300</td>
          <td id="T_78b05_row228_col4" class="data row228 col4" >-0.033500</td>
          <td id="T_78b05_row228_col5" class="data row228 col5" >0.019100</td>
          <td id="T_78b05_row228_col6" class="data row228 col6" >-0.045000</td>
          <td id="T_78b05_row228_col7" class="data row228 col7" >0.020800</td>
          <td id="T_78b05_row228_col8" class="data row228 col8" >0.006700</td>
          <td id="T_78b05_row228_col9" class="data row228 col9" >0.041600</td>
          <td id="T_78b05_row228_col10" class="data row228 col10" >0.056000</td>
          <td id="T_78b05_row228_col11" class="data row228 col11" >0.015600</td>
          <td id="T_78b05_row228_col12" class="data row228 col12" >0.019700</td>
          <td id="T_78b05_row228_col13" class="data row228 col13" >0.041100</td>
          <td id="T_78b05_row228_col14" class="data row228 col14" >0.022500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row229" class="row_heading level0 row229" >230</th>
          <td id="T_78b05_row229_col0" class="data row229 col0" >None</td>
          <td id="T_78b05_row229_col1" class="data row229 col1" >0.046300</td>
          <td id="T_78b05_row229_col2" class="data row229 col2" >-0.035200</td>
          <td id="T_78b05_row229_col3" class="data row229 col3" >0.021000</td>
          <td id="T_78b05_row229_col4" class="data row229 col4" >0.004200</td>
          <td id="T_78b05_row229_col5" class="data row229 col5" >0.005900</td>
          <td id="T_78b05_row229_col6" class="data row229 col6" >0.003600</td>
          <td id="T_78b05_row229_col7" class="data row229 col7" >0.026400</td>
          <td id="T_78b05_row229_col8" class="data row229 col8" >0.000800</td>
          <td id="T_78b05_row229_col9" class="data row229 col9" >0.005300</td>
          <td id="T_78b05_row229_col10" class="data row229 col10" >0.009700</td>
          <td id="T_78b05_row229_col11" class="data row229 col11" >0.022100</td>
          <td id="T_78b05_row229_col12" class="data row229 col12" >0.006500</td>
          <td id="T_78b05_row229_col13" class="data row229 col13" >0.007500</td>
          <td id="T_78b05_row229_col14" class="data row229 col14" >0.028100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row230" class="row_heading level0 row230" >231</th>
          <td id="T_78b05_row230_col0" class="data row230 col0" >None</td>
          <td id="T_78b05_row230_col1" class="data row230 col1" >0.038700</td>
          <td id="T_78b05_row230_col2" class="data row230 col2" >-0.041300</td>
          <td id="T_78b05_row230_col3" class="data row230 col3" >-0.054100</td>
          <td id="T_78b05_row230_col4" class="data row230 col4" >-0.029500</td>
          <td id="T_78b05_row230_col5" class="data row230 col5" >0.048400</td>
          <td id="T_78b05_row230_col6" class="data row230 col6" >-0.019900</td>
          <td id="T_78b05_row230_col7" class="data row230 col7" >0.013100</td>
          <td id="T_78b05_row230_col8" class="data row230 col8" >0.006900</td>
          <td id="T_78b05_row230_col9" class="data row230 col9" >0.011400</td>
          <td id="T_78b05_row230_col10" class="data row230 col10" >0.084800</td>
          <td id="T_78b05_row230_col11" class="data row230 col11" >0.011600</td>
          <td id="T_78b05_row230_col12" class="data row230 col12" >0.049000</td>
          <td id="T_78b05_row230_col13" class="data row230 col13" >0.016000</td>
          <td id="T_78b05_row230_col14" class="data row230 col14" >0.014900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row231" class="row_heading level0 row231" >232</th>
          <td id="T_78b05_row231_col0" class="data row231 col0" >None</td>
          <td id="T_78b05_row231_col1" class="data row231 col1" >0.032500</td>
          <td id="T_78b05_row231_col2" class="data row231 col2" >-0.016200</td>
          <td id="T_78b05_row231_col3" class="data row231 col3" >-0.053700</td>
          <td id="T_78b05_row231_col4" class="data row231 col4" >0.022900</td>
          <td id="T_78b05_row231_col5" class="data row231 col5" >-0.037600</td>
          <td id="T_78b05_row231_col6" class="data row231 col6" >-0.010100</td>
          <td id="T_78b05_row231_col7" class="data row231 col7" >0.022200</td>
          <td id="T_78b05_row231_col8" class="data row231 col8" >0.013000</td>
          <td id="T_78b05_row231_col9" class="data row231 col9" >0.013600</td>
          <td id="T_78b05_row231_col10" class="data row231 col10" >0.084400</td>
          <td id="T_78b05_row231_col11" class="data row231 col11" >0.040800</td>
          <td id="T_78b05_row231_col12" class="data row231 col12" >0.037000</td>
          <td id="T_78b05_row231_col13" class="data row231 col13" >0.006200</td>
          <td id="T_78b05_row231_col14" class="data row231 col14" >0.023900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row232" class="row_heading level0 row232" >233</th>
          <td id="T_78b05_row232_col0" class="data row232 col0" >None</td>
          <td id="T_78b05_row232_col1" class="data row232 col1" >0.042600</td>
          <td id="T_78b05_row232_col2" class="data row232 col2" >-0.061400</td>
          <td id="T_78b05_row232_col3" class="data row232 col3" >0.035800</td>
          <td id="T_78b05_row232_col4" class="data row232 col4" >-0.030400</td>
          <td id="T_78b05_row232_col5" class="data row232 col5" >-0.029500</td>
          <td id="T_78b05_row232_col6" class="data row232 col6" >0.096000</td>
          <td id="T_78b05_row232_col7" class="data row232 col7" >-0.059400</td>
          <td id="T_78b05_row232_col8" class="data row232 col8" >0.002900</td>
          <td id="T_78b05_row232_col9" class="data row232 col9" >0.031600</td>
          <td id="T_78b05_row232_col10" class="data row232 col10" >0.005000</td>
          <td id="T_78b05_row232_col11" class="data row232 col11" >0.012400</td>
          <td id="T_78b05_row232_col12" class="data row232 col12" >0.028800</td>
          <td id="T_78b05_row232_col13" class="data row232 col13" >0.099900</td>
          <td id="T_78b05_row232_col14" class="data row232 col14" >0.057600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row233" class="row_heading level0 row233" >234</th>
          <td id="T_78b05_row233_col0" class="data row233 col0" >None</td>
          <td id="T_78b05_row233_col1" class="data row233 col1" >0.038500</td>
          <td id="T_78b05_row233_col2" class="data row233 col2" >-0.029300</td>
          <td id="T_78b05_row233_col3" class="data row233 col3" >-0.017800</td>
          <td id="T_78b05_row233_col4" class="data row233 col4" >0.009300</td>
          <td id="T_78b05_row233_col5" class="data row233 col5" >-0.006200</td>
          <td id="T_78b05_row233_col6" class="data row233 col6" >-0.012500</td>
          <td id="T_78b05_row233_col7" class="data row233 col7" >0.044800</td>
          <td id="T_78b05_row233_col8" class="data row233 col8" >0.007000</td>
          <td id="T_78b05_row233_col9" class="data row233 col9" >0.000500</td>
          <td id="T_78b05_row233_col10" class="data row233 col10" >0.048500</td>
          <td id="T_78b05_row233_col11" class="data row233 col11" >0.027200</td>
          <td id="T_78b05_row233_col12" class="data row233 col12" >0.005600</td>
          <td id="T_78b05_row233_col13" class="data row233 col13" >0.008600</td>
          <td id="T_78b05_row233_col14" class="data row233 col14" >0.046600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row234" class="row_heading level0 row234" >235</th>
          <td id="T_78b05_row234_col0" class="data row234 col0" >None</td>
          <td id="T_78b05_row234_col1" class="data row234 col1" >0.031700</td>
          <td id="T_78b05_row234_col2" class="data row234 col2" >-0.000800</td>
          <td id="T_78b05_row234_col3" class="data row234 col3" >-0.024000</td>
          <td id="T_78b05_row234_col4" class="data row234 col4" >-0.020800</td>
          <td id="T_78b05_row234_col5" class="data row234 col5" >0.012500</td>
          <td id="T_78b05_row234_col6" class="data row234 col6" >0.032000</td>
          <td id="T_78b05_row234_col7" class="data row234 col7" >-0.060800</td>
          <td id="T_78b05_row234_col8" class="data row234 col8" >0.013800</td>
          <td id="T_78b05_row234_col9" class="data row234 col9" >0.029100</td>
          <td id="T_78b05_row234_col10" class="data row234 col10" >0.054800</td>
          <td id="T_78b05_row234_col11" class="data row234 col11" >0.002900</td>
          <td id="T_78b05_row234_col12" class="data row234 col12" >0.013100</td>
          <td id="T_78b05_row234_col13" class="data row234 col13" >0.035900</td>
          <td id="T_78b05_row234_col14" class="data row234 col14" >0.059100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row235" class="row_heading level0 row235" >236</th>
          <td id="T_78b05_row235_col0" class="data row235 col0" >None</td>
          <td id="T_78b05_row235_col1" class="data row235 col1" >0.044100</td>
          <td id="T_78b05_row235_col2" class="data row235 col2" >-0.049100</td>
          <td id="T_78b05_row235_col3" class="data row235 col3" >0.037600</td>
          <td id="T_78b05_row235_col4" class="data row235 col4" >-0.024200</td>
          <td id="T_78b05_row235_col5" class="data row235 col5" >-0.009800</td>
          <td id="T_78b05_row235_col6" class="data row235 col6" >0.033400</td>
          <td id="T_78b05_row235_col7" class="data row235 col7" >0.029000</td>
          <td id="T_78b05_row235_col8" class="data row235 col8" >0.001400</td>
          <td id="T_78b05_row235_col9" class="data row235 col9" >0.019300</td>
          <td id="T_78b05_row235_col10" class="data row235 col10" >0.006900</td>
          <td id="T_78b05_row235_col11" class="data row235 col11" >0.006300</td>
          <td id="T_78b05_row235_col12" class="data row235 col12" >0.009200</td>
          <td id="T_78b05_row235_col13" class="data row235 col13" >0.037200</td>
          <td id="T_78b05_row235_col14" class="data row235 col14" >0.030800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row236" class="row_heading level0 row236" >237</th>
          <td id="T_78b05_row236_col0" class="data row236 col0" >None</td>
          <td id="T_78b05_row236_col1" class="data row236 col1" >0.035200</td>
          <td id="T_78b05_row236_col2" class="data row236 col2" >0.047900</td>
          <td id="T_78b05_row236_col3" class="data row236 col3" >0.031700</td>
          <td id="T_78b05_row236_col4" class="data row236 col4" >0.004700</td>
          <td id="T_78b05_row236_col5" class="data row236 col5" >0.011500</td>
          <td id="T_78b05_row236_col6" class="data row236 col6" >0.042600</td>
          <td id="T_78b05_row236_col7" class="data row236 col7" >0.007600</td>
          <td id="T_78b05_row236_col8" class="data row236 col8" >0.010300</td>
          <td id="T_78b05_row236_col9" class="data row236 col9" >0.077800</td>
          <td id="T_78b05_row236_col10" class="data row236 col10" >0.000900</td>
          <td id="T_78b05_row236_col11" class="data row236 col11" >0.022600</td>
          <td id="T_78b05_row236_col12" class="data row236 col12" >0.012100</td>
          <td id="T_78b05_row236_col13" class="data row236 col13" >0.046500</td>
          <td id="T_78b05_row236_col14" class="data row236 col14" >0.009400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row237" class="row_heading level0 row237" >238</th>
          <td id="T_78b05_row237_col0" class="data row237 col0" >None</td>
          <td id="T_78b05_row237_col1" class="data row237 col1" >0.031200</td>
          <td id="T_78b05_row237_col2" class="data row237 col2" >0.034600</td>
          <td id="T_78b05_row237_col3" class="data row237 col3" >-0.071700</td>
          <td id="T_78b05_row237_col4" class="data row237 col4" >-0.022900</td>
          <td id="T_78b05_row237_col5" class="data row237 col5" >0.055300</td>
          <td id="T_78b05_row237_col6" class="data row237 col6" >-0.021800</td>
          <td id="T_78b05_row237_col7" class="data row237 col7" >-0.052800</td>
          <td id="T_78b05_row237_col8" class="data row237 col8" >0.014300</td>
          <td id="T_78b05_row237_col9" class="data row237 col9" >0.064400</td>
          <td id="T_78b05_row237_col10" class="data row237 col10" >0.102500</td>
          <td id="T_78b05_row237_col11" class="data row237 col11" >0.005000</td>
          <td id="T_78b05_row237_col12" class="data row237 col12" >0.055900</td>
          <td id="T_78b05_row237_col13" class="data row237 col13" >0.017900</td>
          <td id="T_78b05_row237_col14" class="data row237 col14" >0.051100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row238" class="row_heading level0 row238" >239</th>
          <td id="T_78b05_row238_col0" class="data row238 col0" >None</td>
          <td id="T_78b05_row238_col1" class="data row238 col1" >0.044800</td>
          <td id="T_78b05_row238_col2" class="data row238 col2" >0.048500</td>
          <td id="T_78b05_row238_col3" class="data row238 col3" >0.078500</td>
          <td id="T_78b05_row238_col4" class="data row238 col4" >-0.020300</td>
          <td id="T_78b05_row238_col5" class="data row238 col5" >0.021100</td>
          <td id="T_78b05_row238_col6" class="data row238 col6" >0.072600</td>
          <td id="T_78b05_row238_col7" class="data row238 col7" >0.006300</td>
          <td id="T_78b05_row238_col8" class="data row238 col8" >0.000700</td>
          <td id="T_78b05_row238_col9" class="data row238 col9" >0.078300</td>
          <td id="T_78b05_row238_col10" class="data row238 col10" >0.047800</td>
          <td id="T_78b05_row238_col11" class="data row238 col11" >0.002400</td>
          <td id="T_78b05_row238_col12" class="data row238 col12" >0.021800</td>
          <td id="T_78b05_row238_col13" class="data row238 col13" >0.076500</td>
          <td id="T_78b05_row238_col14" class="data row238 col14" >0.008000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row239" class="row_heading level0 row239" >240</th>
          <td id="T_78b05_row239_col0" class="data row239 col0" >None</td>
          <td id="T_78b05_row239_col1" class="data row239 col1" >0.031200</td>
          <td id="T_78b05_row239_col2" class="data row239 col2" >0.066900</td>
          <td id="T_78b05_row239_col3" class="data row239 col3" >0.014300</td>
          <td id="T_78b05_row239_col4" class="data row239 col4" >0.010000</td>
          <td id="T_78b05_row239_col5" class="data row239 col5" >-0.042900</td>
          <td id="T_78b05_row239_col6" class="data row239 col6" >0.067300</td>
          <td id="T_78b05_row239_col7" class="data row239 col7" >-0.000100</td>
          <td id="T_78b05_row239_col8" class="data row239 col8" >0.014300</td>
          <td id="T_78b05_row239_col9" class="data row239 col9" >0.096700</td>
          <td id="T_78b05_row239_col10" class="data row239 col10" >0.016400</td>
          <td id="T_78b05_row239_col11" class="data row239 col11" >0.027900</td>
          <td id="T_78b05_row239_col12" class="data row239 col12" >0.042200</td>
          <td id="T_78b05_row239_col13" class="data row239 col13" >0.071200</td>
          <td id="T_78b05_row239_col14" class="data row239 col14" >0.001600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row240" class="row_heading level0 row240" >241</th>
          <td id="T_78b05_row240_col0" class="data row240 col0" >None</td>
          <td id="T_78b05_row240_col1" class="data row240 col1" >0.042100</td>
          <td id="T_78b05_row240_col2" class="data row240 col2" >-0.088400</td>
          <td id="T_78b05_row240_col3" class="data row240 col3" >0.003800</td>
          <td id="T_78b05_row240_col4" class="data row240 col4" >-0.009700</td>
          <td id="T_78b05_row240_col5" class="data row240 col5" >-0.033000</td>
          <td id="T_78b05_row240_col6" class="data row240 col6" >0.031500</td>
          <td id="T_78b05_row240_col7" class="data row240 col7" >-0.009100</td>
          <td id="T_78b05_row240_col8" class="data row240 col8" >0.003400</td>
          <td id="T_78b05_row240_col9" class="data row240 col9" >0.058600</td>
          <td id="T_78b05_row240_col10" class="data row240 col10" >0.026900</td>
          <td id="T_78b05_row240_col11" class="data row240 col11" >0.008200</td>
          <td id="T_78b05_row240_col12" class="data row240 col12" >0.032400</td>
          <td id="T_78b05_row240_col13" class="data row240 col13" >0.035400</td>
          <td id="T_78b05_row240_col14" class="data row240 col14" >0.007300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row241" class="row_heading level0 row241" >242</th>
          <td id="T_78b05_row241_col0" class="data row241 col0" >None</td>
          <td id="T_78b05_row241_col1" class="data row241 col1" >0.026700</td>
          <td id="T_78b05_row241_col2" class="data row241 col2" >-0.033300</td>
          <td id="T_78b05_row241_col3" class="data row241 col3" >-0.028900</td>
          <td id="T_78b05_row241_col4" class="data row241 col4" >0.035800</td>
          <td id="T_78b05_row241_col5" class="data row241 col5" >0.003000</td>
          <td id="T_78b05_row241_col6" class="data row241 col6" >-0.093100</td>
          <td id="T_78b05_row241_col7" class="data row241 col7" >-0.030600</td>
          <td id="T_78b05_row241_col8" class="data row241 col8" >0.018800</td>
          <td id="T_78b05_row241_col9" class="data row241 col9" >0.003400</td>
          <td id="T_78b05_row241_col10" class="data row241 col10" >0.059700</td>
          <td id="T_78b05_row241_col11" class="data row241 col11" >0.053700</td>
          <td id="T_78b05_row241_col12" class="data row241 col12" >0.003700</td>
          <td id="T_78b05_row241_col13" class="data row241 col13" >0.089200</td>
          <td id="T_78b05_row241_col14" class="data row241 col14" >0.028900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row242" class="row_heading level0 row242" >243</th>
          <td id="T_78b05_row242_col0" class="data row242 col0" >None</td>
          <td id="T_78b05_row242_col1" class="data row242 col1" >0.037300</td>
          <td id="T_78b05_row242_col2" class="data row242 col2" >-0.030400</td>
          <td id="T_78b05_row242_col3" class="data row242 col3" >-0.050000</td>
          <td id="T_78b05_row242_col4" class="data row242 col4" >-0.014500</td>
          <td id="T_78b05_row242_col5" class="data row242 col5" >0.038000</td>
          <td id="T_78b05_row242_col6" class="data row242 col6" >-0.037100</td>
          <td id="T_78b05_row242_col7" class="data row242 col7" >-0.002300</td>
          <td id="T_78b05_row242_col8" class="data row242 col8" >0.008200</td>
          <td id="T_78b05_row242_col9" class="data row242 col9" >0.000600</td>
          <td id="T_78b05_row242_col10" class="data row242 col10" >0.080800</td>
          <td id="T_78b05_row242_col11" class="data row242 col11" >0.003500</td>
          <td id="T_78b05_row242_col12" class="data row242 col12" >0.038700</td>
          <td id="T_78b05_row242_col13" class="data row242 col13" >0.033200</td>
          <td id="T_78b05_row242_col14" class="data row242 col14" >0.000600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row243" class="row_heading level0 row243" >244</th>
          <td id="T_78b05_row243_col0" class="data row243 col0" >None</td>
          <td id="T_78b05_row243_col1" class="data row243 col1" >0.037600</td>
          <td id="T_78b05_row243_col2" class="data row243 col2" >0.062300</td>
          <td id="T_78b05_row243_col3" class="data row243 col3" >0.004200</td>
          <td id="T_78b05_row243_col4" class="data row243 col4" >-0.053000</td>
          <td id="T_78b05_row243_col5" class="data row243 col5" >-0.007600</td>
          <td id="T_78b05_row243_col6" class="data row243 col6" >-0.018100</td>
          <td id="T_78b05_row243_col7" class="data row243 col7" >0.010800</td>
          <td id="T_78b05_row243_col8" class="data row243 col8" >0.008000</td>
          <td id="T_78b05_row243_col9" class="data row243 col9" >0.092100</td>
          <td id="T_78b05_row243_col10" class="data row243 col10" >0.026600</td>
          <td id="T_78b05_row243_col11" class="data row243 col11" >0.035100</td>
          <td id="T_78b05_row243_col12" class="data row243 col12" >0.006900</td>
          <td id="T_78b05_row243_col13" class="data row243 col13" >0.014200</td>
          <td id="T_78b05_row243_col14" class="data row243 col14" >0.012500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row244" class="row_heading level0 row244" >245</th>
          <td id="T_78b05_row244_col0" class="data row244 col0" >None</td>
          <td id="T_78b05_row244_col1" class="data row244 col1" >0.030100</td>
          <td id="T_78b05_row244_col2" class="data row244 col2" >0.067500</td>
          <td id="T_78b05_row244_col3" class="data row244 col3" >-0.023400</td>
          <td id="T_78b05_row244_col4" class="data row244 col4" >-0.028800</td>
          <td id="T_78b05_row244_col5" class="data row244 col5" >0.019700</td>
          <td id="T_78b05_row244_col6" class="data row244 col6" >0.054000</td>
          <td id="T_78b05_row244_col7" class="data row244 col7" >-0.018700</td>
          <td id="T_78b05_row244_col8" class="data row244 col8" >0.015500</td>
          <td id="T_78b05_row244_col9" class="data row244 col9" >0.097400</td>
          <td id="T_78b05_row244_col10" class="data row244 col10" >0.054100</td>
          <td id="T_78b05_row244_col11" class="data row244 col11" >0.010900</td>
          <td id="T_78b05_row244_col12" class="data row244 col12" >0.020300</td>
          <td id="T_78b05_row244_col13" class="data row244 col13" >0.057900</td>
          <td id="T_78b05_row244_col14" class="data row244 col14" >0.016900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row245" class="row_heading level0 row245" >246</th>
          <td id="T_78b05_row245_col0" class="data row245 col0" >None</td>
          <td id="T_78b05_row245_col1" class="data row245 col1" >0.028400</td>
          <td id="T_78b05_row245_col2" class="data row245 col2" >0.017100</td>
          <td id="T_78b05_row245_col3" class="data row245 col3" >-0.031800</td>
          <td id="T_78b05_row245_col4" class="data row245 col4" >0.052300</td>
          <td id="T_78b05_row245_col5" class="data row245 col5" >-0.011900</td>
          <td id="T_78b05_row245_col6" class="data row245 col6" >-0.020800</td>
          <td id="T_78b05_row245_col7" class="data row245 col7" >-0.017700</td>
          <td id="T_78b05_row245_col8" class="data row245 col8" >0.017100</td>
          <td id="T_78b05_row245_col9" class="data row245 col9" >0.046900</td>
          <td id="T_78b05_row245_col10" class="data row245 col10" >0.062500</td>
          <td id="T_78b05_row245_col11" class="data row245 col11" >0.070200</td>
          <td id="T_78b05_row245_col12" class="data row245 col12" >0.011300</td>
          <td id="T_78b05_row245_col13" class="data row245 col13" >0.016900</td>
          <td id="T_78b05_row245_col14" class="data row245 col14" >0.016000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row246" class="row_heading level0 row246" >247</th>
          <td id="T_78b05_row246_col0" class="data row246 col0" >None</td>
          <td id="T_78b05_row246_col1" class="data row246 col1" >0.033600</td>
          <td id="T_78b05_row246_col2" class="data row246 col2" >0.009300</td>
          <td id="T_78b05_row246_col3" class="data row246 col3" >0.039900</td>
          <td id="T_78b05_row246_col4" class="data row246 col4" >-0.023800</td>
          <td id="T_78b05_row246_col5" class="data row246 col5" >-0.004100</td>
          <td id="T_78b05_row246_col6" class="data row246 col6" >-0.053900</td>
          <td id="T_78b05_row246_col7" class="data row246 col7" >0.030200</td>
          <td id="T_78b05_row246_col8" class="data row246 col8" >0.011900</td>
          <td id="T_78b05_row246_col9" class="data row246 col9" >0.039200</td>
          <td id="T_78b05_row246_col10" class="data row246 col10" >0.009100</td>
          <td id="T_78b05_row246_col11" class="data row246 col11" >0.005900</td>
          <td id="T_78b05_row246_col12" class="data row246 col12" >0.003500</td>
          <td id="T_78b05_row246_col13" class="data row246 col13" >0.050000</td>
          <td id="T_78b05_row246_col14" class="data row246 col14" >0.032000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row247" class="row_heading level0 row247" >248</th>
          <td id="T_78b05_row247_col0" class="data row247 col0" >None</td>
          <td id="T_78b05_row247_col1" class="data row247 col1" >0.044700</td>
          <td id="T_78b05_row247_col2" class="data row247 col2" >-0.018700</td>
          <td id="T_78b05_row247_col3" class="data row247 col3" >0.027300</td>
          <td id="T_78b05_row247_col4" class="data row247 col4" >0.025800</td>
          <td id="T_78b05_row247_col5" class="data row247 col5" >-0.007600</td>
          <td id="T_78b05_row247_col6" class="data row247 col6" >-0.058200</td>
          <td id="T_78b05_row247_col7" class="data row247 col7" >-0.046300</td>
          <td id="T_78b05_row247_col8" class="data row247 col8" >0.000800</td>
          <td id="T_78b05_row247_col9" class="data row247 col9" >0.011100</td>
          <td id="T_78b05_row247_col10" class="data row247 col10" >0.003400</td>
          <td id="T_78b05_row247_col11" class="data row247 col11" >0.043700</td>
          <td id="T_78b05_row247_col12" class="data row247 col12" >0.007000</td>
          <td id="T_78b05_row247_col13" class="data row247 col13" >0.054300</td>
          <td id="T_78b05_row247_col14" class="data row247 col14" >0.044600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row248" class="row_heading level0 row248" >249</th>
          <td id="T_78b05_row248_col0" class="data row248 col0" >None</td>
          <td id="T_78b05_row248_col1" class="data row248 col1" >0.032700</td>
          <td id="T_78b05_row248_col2" class="data row248 col2" >-0.011500</td>
          <td id="T_78b05_row248_col3" class="data row248 col3" >-0.033100</td>
          <td id="T_78b05_row248_col4" class="data row248 col4" >0.022100</td>
          <td id="T_78b05_row248_col5" class="data row248 col5" >0.005700</td>
          <td id="T_78b05_row248_col6" class="data row248 col6" >-0.023500</td>
          <td id="T_78b05_row248_col7" class="data row248 col7" >0.021400</td>
          <td id="T_78b05_row248_col8" class="data row248 col8" >0.012800</td>
          <td id="T_78b05_row248_col9" class="data row248 col9" >0.018400</td>
          <td id="T_78b05_row248_col10" class="data row248 col10" >0.063800</td>
          <td id="T_78b05_row248_col11" class="data row248 col11" >0.040000</td>
          <td id="T_78b05_row248_col12" class="data row248 col12" >0.006400</td>
          <td id="T_78b05_row248_col13" class="data row248 col13" >0.019600</td>
          <td id="T_78b05_row248_col14" class="data row248 col14" >0.023200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row249" class="row_heading level0 row249" >250</th>
          <td id="T_78b05_row249_col0" class="data row249 col0" >None</td>
          <td id="T_78b05_row249_col1" class="data row249 col1" >0.046100</td>
          <td id="T_78b05_row249_col2" class="data row249 col2" >-0.030400</td>
          <td id="T_78b05_row249_col3" class="data row249 col3" >0.037800</td>
          <td id="T_78b05_row249_col4" class="data row249 col4" >0.025400</td>
          <td id="T_78b05_row249_col5" class="data row249 col5" >0.030900</td>
          <td id="T_78b05_row249_col6" class="data row249 col6" >-0.045800</td>
          <td id="T_78b05_row249_col7" class="data row249 col7" >0.047700</td>
          <td id="T_78b05_row249_col8" class="data row249 col8" >0.000600</td>
          <td id="T_78b05_row249_col9" class="data row249 col9" >0.000500</td>
          <td id="T_78b05_row249_col10" class="data row249 col10" >0.007100</td>
          <td id="T_78b05_row249_col11" class="data row249 col11" >0.043300</td>
          <td id="T_78b05_row249_col12" class="data row249 col12" >0.031500</td>
          <td id="T_78b05_row249_col13" class="data row249 col13" >0.041900</td>
          <td id="T_78b05_row249_col14" class="data row249 col14" >0.049400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row250" class="row_heading level0 row250" >251</th>
          <td id="T_78b05_row250_col0" class="data row250 col0" >None</td>
          <td id="T_78b05_row250_col1" class="data row250 col1" >0.048100</td>
          <td id="T_78b05_row250_col2" class="data row250 col2" >-0.044900</td>
          <td id="T_78b05_row250_col3" class="data row250 col3" >0.042400</td>
          <td id="T_78b05_row250_col4" class="data row250 col4" >-0.065600</td>
          <td id="T_78b05_row250_col5" class="data row250 col5" >0.005600</td>
          <td id="T_78b05_row250_col6" class="data row250 col6" >-0.040100</td>
          <td id="T_78b05_row250_col7" class="data row250 col7" >0.042400</td>
          <td id="T_78b05_row250_col8" class="data row250 col8" >0.002600</td>
          <td id="T_78b05_row250_col9" class="data row250 col9" >0.015100</td>
          <td id="T_78b05_row250_col10" class="data row250 col10" >0.011700</td>
          <td id="T_78b05_row250_col11" class="data row250 col11" >0.047700</td>
          <td id="T_78b05_row250_col12" class="data row250 col12" >0.006200</td>
          <td id="T_78b05_row250_col13" class="data row250 col13" >0.036200</td>
          <td id="T_78b05_row250_col14" class="data row250 col14" >0.044200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row251" class="row_heading level0 row251" >252</th>
          <td id="T_78b05_row251_col0" class="data row251 col0" >None</td>
          <td id="T_78b05_row251_col1" class="data row251 col1" >0.030300</td>
          <td id="T_78b05_row251_col2" class="data row251 col2" >0.008400</td>
          <td id="T_78b05_row251_col3" class="data row251 col3" >-0.018200</td>
          <td id="T_78b05_row251_col4" class="data row251 col4" >-0.020400</td>
          <td id="T_78b05_row251_col5" class="data row251 col5" >0.003100</td>
          <td id="T_78b05_row251_col6" class="data row251 col6" >-0.036100</td>
          <td id="T_78b05_row251_col7" class="data row251 col7" >0.051200</td>
          <td id="T_78b05_row251_col8" class="data row251 col8" >0.015200</td>
          <td id="T_78b05_row251_col9" class="data row251 col9" >0.038200</td>
          <td id="T_78b05_row251_col10" class="data row251 col10" >0.048900</td>
          <td id="T_78b05_row251_col11" class="data row251 col11" >0.002500</td>
          <td id="T_78b05_row251_col12" class="data row251 col12" >0.003700</td>
          <td id="T_78b05_row251_col13" class="data row251 col13" >0.032200</td>
          <td id="T_78b05_row251_col14" class="data row251 col14" >0.052900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row252" class="row_heading level0 row252" >253</th>
          <td id="T_78b05_row252_col0" class="data row252 col0" >None</td>
          <td id="T_78b05_row252_col1" class="data row252 col1" >0.043800</td>
          <td id="T_78b05_row252_col2" class="data row252 col2" >-0.044100</td>
          <td id="T_78b05_row252_col3" class="data row252 col3" >-0.030500</td>
          <td id="T_78b05_row252_col4" class="data row252 col4" >-0.037900</td>
          <td id="T_78b05_row252_col5" class="data row252 col5" >-0.000400</td>
          <td id="T_78b05_row252_col6" class="data row252 col6" >-0.002200</td>
          <td id="T_78b05_row252_col7" class="data row252 col7" >-0.024700</td>
          <td id="T_78b05_row252_col8" class="data row252 col8" >0.001700</td>
          <td id="T_78b05_row252_col9" class="data row252 col9" >0.014300</td>
          <td id="T_78b05_row252_col10" class="data row252 col10" >0.061300</td>
          <td id="T_78b05_row252_col11" class="data row252 col11" >0.020000</td>
          <td id="T_78b05_row252_col12" class="data row252 col12" >0.000300</td>
          <td id="T_78b05_row252_col13" class="data row252 col13" >0.001700</td>
          <td id="T_78b05_row252_col14" class="data row252 col14" >0.023000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row253" class="row_heading level0 row253" >254</th>
          <td id="T_78b05_row253_col0" class="data row253 col0" >PC1</td>
          <td id="T_78b05_row253_col1" class="data row253 col1" >0.021500</td>
          <td id="T_78b05_row253_col2" class="data row253 col2" >0.012900</td>
          <td id="T_78b05_row253_col3" class="data row253 col3" >-0.071500</td>
          <td id="T_78b05_row253_col4" class="data row253 col4" >0.038500</td>
          <td id="T_78b05_row253_col5" class="data row253 col5" >-0.044900</td>
          <td id="T_78b05_row253_col6" class="data row253 col6" >-0.010500</td>
          <td id="T_78b05_row253_col7" class="data row253 col7" >0.008200</td>
          <td id="T_78b05_row253_col8" class="data row253 col8" >0.024000</td>
          <td id="T_78b05_row253_col9" class="data row253 col9" >0.042700</td>
          <td id="T_78b05_row253_col10" class="data row253 col10" >0.102200</td>
          <td id="T_78b05_row253_col11" class="data row253 col11" >0.056400</td>
          <td id="T_78b05_row253_col12" class="data row253 col12" >0.044300</td>
          <td id="T_78b05_row253_col13" class="data row253 col13" >0.006700</td>
          <td id="T_78b05_row253_col14" class="data row253 col14" >0.009900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row254" class="row_heading level0 row254" >255</th>
          <td id="T_78b05_row254_col0" class="data row254 col0" >None</td>
          <td id="T_78b05_row254_col1" class="data row254 col1" >0.044500</td>
          <td id="T_78b05_row254_col2" class="data row254 col2" >-0.007800</td>
          <td id="T_78b05_row254_col3" class="data row254 col3" >0.050100</td>
          <td id="T_78b05_row254_col4" class="data row254 col4" >0.025800</td>
          <td id="T_78b05_row254_col5" class="data row254 col5" >0.038900</td>
          <td id="T_78b05_row254_col6" class="data row254 col6" >-0.060100</td>
          <td id="T_78b05_row254_col7" class="data row254 col7" >-0.058800</td>
          <td id="T_78b05_row254_col8" class="data row254 col8" >0.001000</td>
          <td id="T_78b05_row254_col9" class="data row254 col9" >0.022100</td>
          <td id="T_78b05_row254_col10" class="data row254 col10" >0.019300</td>
          <td id="T_78b05_row254_col11" class="data row254 col11" >0.043700</td>
          <td id="T_78b05_row254_col12" class="data row254 col12" >0.039500</td>
          <td id="T_78b05_row254_col13" class="data row254 col13" >0.056200</td>
          <td id="T_78b05_row254_col14" class="data row254 col14" >0.057000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row255" class="row_heading level0 row255" >256</th>
          <td id="T_78b05_row255_col0" class="data row255 col0" >None</td>
          <td id="T_78b05_row255_col1" class="data row255 col1" >0.038200</td>
          <td id="T_78b05_row255_col2" class="data row255 col2" >-0.054400</td>
          <td id="T_78b05_row255_col3" class="data row255 col3" >-0.011800</td>
          <td id="T_78b05_row255_col4" class="data row255 col4" >-0.049200</td>
          <td id="T_78b05_row255_col5" class="data row255 col5" >-0.036200</td>
          <td id="T_78b05_row255_col6" class="data row255 col6" >0.048700</td>
          <td id="T_78b05_row255_col7" class="data row255 col7" >-0.076900</td>
          <td id="T_78b05_row255_col8" class="data row255 col8" >0.007400</td>
          <td id="T_78b05_row255_col9" class="data row255 col9" >0.024600</td>
          <td id="T_78b05_row255_col10" class="data row255 col10" >0.042600</td>
          <td id="T_78b05_row255_col11" class="data row255 col11" >0.031300</td>
          <td id="T_78b05_row255_col12" class="data row255 col12" >0.035600</td>
          <td id="T_78b05_row255_col13" class="data row255 col13" >0.052600</td>
          <td id="T_78b05_row255_col14" class="data row255 col14" >0.075100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row256" class="row_heading level0 row256" >257</th>
          <td id="T_78b05_row256_col0" class="data row256 col0" >None</td>
          <td id="T_78b05_row256_col1" class="data row256 col1" >0.038400</td>
          <td id="T_78b05_row256_col2" class="data row256 col2" >-0.040100</td>
          <td id="T_78b05_row256_col3" class="data row256 col3" >-0.061100</td>
          <td id="T_78b05_row256_col4" class="data row256 col4" >-0.031800</td>
          <td id="T_78b05_row256_col5" class="data row256 col5" >0.027000</td>
          <td id="T_78b05_row256_col6" class="data row256 col6" >-0.032300</td>
          <td id="T_78b05_row256_col7" class="data row256 col7" >-0.002100</td>
          <td id="T_78b05_row256_col8" class="data row256 col8" >0.007100</td>
          <td id="T_78b05_row256_col9" class="data row256 col9" >0.010300</td>
          <td id="T_78b05_row256_col10" class="data row256 col10" >0.091800</td>
          <td id="T_78b05_row256_col11" class="data row256 col11" >0.013900</td>
          <td id="T_78b05_row256_col12" class="data row256 col12" >0.027700</td>
          <td id="T_78b05_row256_col13" class="data row256 col13" >0.028400</td>
          <td id="T_78b05_row256_col14" class="data row256 col14" >0.000400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row257" class="row_heading level0 row257" >258</th>
          <td id="T_78b05_row257_col0" class="data row257 col0" >None</td>
          <td id="T_78b05_row257_col1" class="data row257 col1" >0.032700</td>
          <td id="T_78b05_row257_col2" class="data row257 col2" >0.044100</td>
          <td id="T_78b05_row257_col3" class="data row257 col3" >-0.042100</td>
          <td id="T_78b05_row257_col4" class="data row257 col4" >-0.015300</td>
          <td id="T_78b05_row257_col5" class="data row257 col5" >0.008700</td>
          <td id="T_78b05_row257_col6" class="data row257 col6" >0.045200</td>
          <td id="T_78b05_row257_col7" class="data row257 col7" >0.005300</td>
          <td id="T_78b05_row257_col8" class="data row257 col8" >0.012800</td>
          <td id="T_78b05_row257_col9" class="data row257 col9" >0.073900</td>
          <td id="T_78b05_row257_col10" class="data row257 col10" >0.072800</td>
          <td id="T_78b05_row257_col11" class="data row257 col11" >0.002600</td>
          <td id="T_78b05_row257_col12" class="data row257 col12" >0.009400</td>
          <td id="T_78b05_row257_col13" class="data row257 col13" >0.049100</td>
          <td id="T_78b05_row257_col14" class="data row257 col14" >0.007100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row258" class="row_heading level0 row258" >259</th>
          <td id="T_78b05_row258_col0" class="data row258 col0" >None</td>
          <td id="T_78b05_row258_col1" class="data row258 col1" >0.036800</td>
          <td id="T_78b05_row258_col2" class="data row258 col2" >0.025500</td>
          <td id="T_78b05_row258_col3" class="data row258 col3" >-0.022900</td>
          <td id="T_78b05_row258_col4" class="data row258 col4" >-0.045100</td>
          <td id="T_78b05_row258_col5" class="data row258 col5" >-0.002700</td>
          <td id="T_78b05_row258_col6" class="data row258 col6" >-0.004800</td>
          <td id="T_78b05_row258_col7" class="data row258 col7" >-0.006600</td>
          <td id="T_78b05_row258_col8" class="data row258 col8" >0.008700</td>
          <td id="T_78b05_row258_col9" class="data row258 col9" >0.055300</td>
          <td id="T_78b05_row258_col10" class="data row258 col10" >0.053700</td>
          <td id="T_78b05_row258_col11" class="data row258 col11" >0.027100</td>
          <td id="T_78b05_row258_col12" class="data row258 col12" >0.002100</td>
          <td id="T_78b05_row258_col13" class="data row258 col13" >0.000900</td>
          <td id="T_78b05_row258_col14" class="data row258 col14" >0.004800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row259" class="row_heading level0 row259" >260</th>
          <td id="T_78b05_row259_col0" class="data row259 col0" >None</td>
          <td id="T_78b05_row259_col1" class="data row259 col1" >0.031500</td>
          <td id="T_78b05_row259_col2" class="data row259 col2" >-0.031500</td>
          <td id="T_78b05_row259_col3" class="data row259 col3" >-0.059800</td>
          <td id="T_78b05_row259_col4" class="data row259 col4" >-0.049500</td>
          <td id="T_78b05_row259_col5" class="data row259 col5" >-0.031900</td>
          <td id="T_78b05_row259_col6" class="data row259 col6" >-0.077200</td>
          <td id="T_78b05_row259_col7" class="data row259 col7" >0.002000</td>
          <td id="T_78b05_row259_col8" class="data row259 col8" >0.014000</td>
          <td id="T_78b05_row259_col9" class="data row259 col9" >0.001700</td>
          <td id="T_78b05_row259_col10" class="data row259 col10" >0.090500</td>
          <td id="T_78b05_row259_col11" class="data row259 col11" >0.031600</td>
          <td id="T_78b05_row259_col12" class="data row259 col12" >0.031200</td>
          <td id="T_78b05_row259_col13" class="data row259 col13" >0.073300</td>
          <td id="T_78b05_row259_col14" class="data row259 col14" >0.003700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row260" class="row_heading level0 row260" >261</th>
          <td id="T_78b05_row260_col0" class="data row260 col0" >None</td>
          <td id="T_78b05_row260_col1" class="data row260 col1" >0.035700</td>
          <td id="T_78b05_row260_col2" class="data row260 col2" >0.005900</td>
          <td id="T_78b05_row260_col3" class="data row260 col3" >0.033300</td>
          <td id="T_78b05_row260_col4" class="data row260 col4" >0.021700</td>
          <td id="T_78b05_row260_col5" class="data row260 col5" >-0.061000</td>
          <td id="T_78b05_row260_col6" class="data row260 col6" >0.002700</td>
          <td id="T_78b05_row260_col7" class="data row260 col7" >0.030500</td>
          <td id="T_78b05_row260_col8" class="data row260 col8" >0.009800</td>
          <td id="T_78b05_row260_col9" class="data row260 col9" >0.035700</td>
          <td id="T_78b05_row260_col10" class="data row260 col10" >0.002600</td>
          <td id="T_78b05_row260_col11" class="data row260 col11" >0.039600</td>
          <td id="T_78b05_row260_col12" class="data row260 col12" >0.060400</td>
          <td id="T_78b05_row260_col13" class="data row260 col13" >0.006600</td>
          <td id="T_78b05_row260_col14" class="data row260 col14" >0.032300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row261" class="row_heading level0 row261" >262</th>
          <td id="T_78b05_row261_col0" class="data row261 col0" >None</td>
          <td id="T_78b05_row261_col1" class="data row261 col1" >0.045900</td>
          <td id="T_78b05_row261_col2" class="data row261 col2" >-0.005900</td>
          <td id="T_78b05_row261_col3" class="data row261 col3" >0.059400</td>
          <td id="T_78b05_row261_col4" class="data row261 col4" >0.064300</td>
          <td id="T_78b05_row261_col5" class="data row261 col5" >0.019300</td>
          <td id="T_78b05_row261_col6" class="data row261 col6" >-0.076900</td>
          <td id="T_78b05_row261_col7" class="data row261 col7" >-0.009400</td>
          <td id="T_78b05_row261_col8" class="data row261 col8" >0.000400</td>
          <td id="T_78b05_row261_col9" class="data row261 col9" >0.023900</td>
          <td id="T_78b05_row261_col10" class="data row261 col10" >0.028600</td>
          <td id="T_78b05_row261_col11" class="data row261 col11" >0.082200</td>
          <td id="T_78b05_row261_col12" class="data row261 col12" >0.019900</td>
          <td id="T_78b05_row261_col13" class="data row261 col13" >0.073000</td>
          <td id="T_78b05_row261_col14" class="data row261 col14" >0.007700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row262" class="row_heading level0 row262" >263</th>
          <td id="T_78b05_row262_col0" class="data row262 col0" >None</td>
          <td id="T_78b05_row262_col1" class="data row262 col1" >0.034800</td>
          <td id="T_78b05_row262_col2" class="data row262 col2" >0.013000</td>
          <td id="T_78b05_row262_col3" class="data row262 col3" >-0.033000</td>
          <td id="T_78b05_row262_col4" class="data row262 col4" >-0.019500</td>
          <td id="T_78b05_row262_col5" class="data row262 col5" >-0.019700</td>
          <td id="T_78b05_row262_col6" class="data row262 col6" >0.004600</td>
          <td id="T_78b05_row262_col7" class="data row262 col7" >0.016800</td>
          <td id="T_78b05_row262_col8" class="data row262 col8" >0.010700</td>
          <td id="T_78b05_row262_col9" class="data row262 col9" >0.042800</td>
          <td id="T_78b05_row262_col10" class="data row262 col10" >0.063800</td>
          <td id="T_78b05_row262_col11" class="data row262 col11" >0.001600</td>
          <td id="T_78b05_row262_col12" class="data row262 col12" >0.019100</td>
          <td id="T_78b05_row262_col13" class="data row262 col13" >0.008400</td>
          <td id="T_78b05_row262_col14" class="data row262 col14" >0.018500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row263" class="row_heading level0 row263" >264</th>
          <td id="T_78b05_row263_col0" class="data row263 col0" >PC4</td>
          <td id="T_78b05_row263_col1" class="data row263 col1" >0.040500</td>
          <td id="T_78b05_row263_col2" class="data row263 col2" >-0.023100</td>
          <td id="T_78b05_row263_col3" class="data row263 col3" >0.024700</td>
          <td id="T_78b05_row263_col4" class="data row263 col4" >0.113800</td>
          <td id="T_78b05_row263_col5" class="data row263 col5" >-0.010200</td>
          <td id="T_78b05_row263_col6" class="data row263 col6" >-0.047400</td>
          <td id="T_78b05_row263_col7" class="data row263 col7" >0.030400</td>
          <td id="T_78b05_row263_col8" class="data row263 col8" >0.005100</td>
          <td id="T_78b05_row263_col9" class="data row263 col9" >0.006700</td>
          <td id="T_78b05_row263_col10" class="data row263 col10" >0.006100</td>
          <td id="T_78b05_row263_col11" class="data row263 col11" >0.131700</td>
          <td id="T_78b05_row263_col12" class="data row263 col12" >0.009500</td>
          <td id="T_78b05_row263_col13" class="data row263 col13" >0.043500</td>
          <td id="T_78b05_row263_col14" class="data row263 col14" >0.032100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row264" class="row_heading level0 row264" >265</th>
          <td id="T_78b05_row264_col0" class="data row264 col0" >None</td>
          <td id="T_78b05_row264_col1" class="data row264 col1" >0.033200</td>
          <td id="T_78b05_row264_col2" class="data row264 col2" >0.073700</td>
          <td id="T_78b05_row264_col3" class="data row264 col3" >-0.008600</td>
          <td id="T_78b05_row264_col4" class="data row264 col4" >-0.033800</td>
          <td id="T_78b05_row264_col5" class="data row264 col5" >-0.005200</td>
          <td id="T_78b05_row264_col6" class="data row264 col6" >0.036800</td>
          <td id="T_78b05_row264_col7" class="data row264 col7" >0.010800</td>
          <td id="T_78b05_row264_col8" class="data row264 col8" >0.012400</td>
          <td id="T_78b05_row264_col9" class="data row264 col9" >0.103600</td>
          <td id="T_78b05_row264_col10" class="data row264 col10" >0.039300</td>
          <td id="T_78b05_row264_col11" class="data row264 col11" >0.015900</td>
          <td id="T_78b05_row264_col12" class="data row264 col12" >0.004500</td>
          <td id="T_78b05_row264_col13" class="data row264 col13" >0.040700</td>
          <td id="T_78b05_row264_col14" class="data row264 col14" >0.012500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row265" class="row_heading level0 row265" >266</th>
          <td id="T_78b05_row265_col0" class="data row265 col0" >None</td>
          <td id="T_78b05_row265_col1" class="data row265 col1" >0.038900</td>
          <td id="T_78b05_row265_col2" class="data row265 col2" >0.009300</td>
          <td id="T_78b05_row265_col3" class="data row265 col3" >0.007100</td>
          <td id="T_78b05_row265_col4" class="data row265 col4" >-0.004900</td>
          <td id="T_78b05_row265_col5" class="data row265 col5" >0.043700</td>
          <td id="T_78b05_row265_col6" class="data row265 col6" >0.036400</td>
          <td id="T_78b05_row265_col7" class="data row265 col7" >0.018500</td>
          <td id="T_78b05_row265_col8" class="data row265 col8" >0.006600</td>
          <td id="T_78b05_row265_col9" class="data row265 col9" >0.039200</td>
          <td id="T_78b05_row265_col10" class="data row265 col10" >0.023600</td>
          <td id="T_78b05_row265_col11" class="data row265 col11" >0.013000</td>
          <td id="T_78b05_row265_col12" class="data row265 col12" >0.044400</td>
          <td id="T_78b05_row265_col13" class="data row265 col13" >0.040300</td>
          <td id="T_78b05_row265_col14" class="data row265 col14" >0.020200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row266" class="row_heading level0 row266" >267</th>
          <td id="T_78b05_row266_col0" class="data row266 col0" >None</td>
          <td id="T_78b05_row266_col1" class="data row266 col1" >0.027100</td>
          <td id="T_78b05_row266_col2" class="data row266 col2" >0.042400</td>
          <td id="T_78b05_row266_col3" class="data row266 col3" >-0.015000</td>
          <td id="T_78b05_row266_col4" class="data row266 col4" >0.051800</td>
          <td id="T_78b05_row266_col5" class="data row266 col5" >-0.017700</td>
          <td id="T_78b05_row266_col6" class="data row266 col6" >-0.045400</td>
          <td id="T_78b05_row266_col7" class="data row266 col7" >-0.001800</td>
          <td id="T_78b05_row266_col8" class="data row266 col8" >0.018400</td>
          <td id="T_78b05_row266_col9" class="data row266 col9" >0.072200</td>
          <td id="T_78b05_row266_col10" class="data row266 col10" >0.045800</td>
          <td id="T_78b05_row266_col11" class="data row266 col11" >0.069700</td>
          <td id="T_78b05_row266_col12" class="data row266 col12" >0.017100</td>
          <td id="T_78b05_row266_col13" class="data row266 col13" >0.041500</td>
          <td id="T_78b05_row266_col14" class="data row266 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row267" class="row_heading level0 row267" >268</th>
          <td id="T_78b05_row267_col0" class="data row267 col0" >None</td>
          <td id="T_78b05_row267_col1" class="data row267 col1" >0.043500</td>
          <td id="T_78b05_row267_col2" class="data row267 col2" >0.017300</td>
          <td id="T_78b05_row267_col3" class="data row267 col3" >0.036000</td>
          <td id="T_78b05_row267_col4" class="data row267 col4" >-0.040500</td>
          <td id="T_78b05_row267_col5" class="data row267 col5" >-0.006000</td>
          <td id="T_78b05_row267_col6" class="data row267 col6" >-0.032300</td>
          <td id="T_78b05_row267_col7" class="data row267 col7" >-0.031400</td>
          <td id="T_78b05_row267_col8" class="data row267 col8" >0.002100</td>
          <td id="T_78b05_row267_col9" class="data row267 col9" >0.047100</td>
          <td id="T_78b05_row267_col10" class="data row267 col10" >0.005300</td>
          <td id="T_78b05_row267_col11" class="data row267 col11" >0.022600</td>
          <td id="T_78b05_row267_col12" class="data row267 col12" >0.005300</td>
          <td id="T_78b05_row267_col13" class="data row267 col13" >0.028400</td>
          <td id="T_78b05_row267_col14" class="data row267 col14" >0.029600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row268" class="row_heading level0 row268" >269</th>
          <td id="T_78b05_row268_col0" class="data row268 col0" >None</td>
          <td id="T_78b05_row268_col1" class="data row268 col1" >0.037900</td>
          <td id="T_78b05_row268_col2" class="data row268 col2" >0.028600</td>
          <td id="T_78b05_row268_col3" class="data row268 col3" >0.051300</td>
          <td id="T_78b05_row268_col4" class="data row268 col4" >-0.002300</td>
          <td id="T_78b05_row268_col5" class="data row268 col5" >0.002100</td>
          <td id="T_78b05_row268_col6" class="data row268 col6" >-0.060000</td>
          <td id="T_78b05_row268_col7" class="data row268 col7" >-0.035600</td>
          <td id="T_78b05_row268_col8" class="data row268 col8" >0.007600</td>
          <td id="T_78b05_row268_col9" class="data row268 col9" >0.058400</td>
          <td id="T_78b05_row268_col10" class="data row268 col10" >0.020600</td>
          <td id="T_78b05_row268_col11" class="data row268 col11" >0.015700</td>
          <td id="T_78b05_row268_col12" class="data row268 col12" >0.002800</td>
          <td id="T_78b05_row268_col13" class="data row268 col13" >0.056100</td>
          <td id="T_78b05_row268_col14" class="data row268 col14" >0.033900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row269" class="row_heading level0 row269" >270</th>
          <td id="T_78b05_row269_col0" class="data row269 col0" >None</td>
          <td id="T_78b05_row269_col1" class="data row269 col1" >0.034600</td>
          <td id="T_78b05_row269_col2" class="data row269 col2" >0.041200</td>
          <td id="T_78b05_row269_col3" class="data row269 col3" >-0.029900</td>
          <td id="T_78b05_row269_col4" class="data row269 col4" >0.016300</td>
          <td id="T_78b05_row269_col5" class="data row269 col5" >0.000400</td>
          <td id="T_78b05_row269_col6" class="data row269 col6" >0.051500</td>
          <td id="T_78b05_row269_col7" class="data row269 col7" >0.034100</td>
          <td id="T_78b05_row269_col8" class="data row269 col8" >0.010900</td>
          <td id="T_78b05_row269_col9" class="data row269 col9" >0.071000</td>
          <td id="T_78b05_row269_col10" class="data row269 col10" >0.060700</td>
          <td id="T_78b05_row269_col11" class="data row269 col11" >0.034200</td>
          <td id="T_78b05_row269_col12" class="data row269 col12" >0.001000</td>
          <td id="T_78b05_row269_col13" class="data row269 col13" >0.055400</td>
          <td id="T_78b05_row269_col14" class="data row269 col14" >0.035900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row270" class="row_heading level0 row270" >271</th>
          <td id="T_78b05_row270_col0" class="data row270 col0" >None</td>
          <td id="T_78b05_row270_col1" class="data row270 col1" >0.041900</td>
          <td id="T_78b05_row270_col2" class="data row270 col2" >-0.039000</td>
          <td id="T_78b05_row270_col3" class="data row270 col3" >0.026200</td>
          <td id="T_78b05_row270_col4" class="data row270 col4" >-0.006400</td>
          <td id="T_78b05_row270_col5" class="data row270 col5" >-0.018500</td>
          <td id="T_78b05_row270_col6" class="data row270 col6" >-0.002400</td>
          <td id="T_78b05_row270_col7" class="data row270 col7" >-0.049100</td>
          <td id="T_78b05_row270_col8" class="data row270 col8" >0.003600</td>
          <td id="T_78b05_row270_col9" class="data row270 col9" >0.009200</td>
          <td id="T_78b05_row270_col10" class="data row270 col10" >0.004600</td>
          <td id="T_78b05_row270_col11" class="data row270 col11" >0.011600</td>
          <td id="T_78b05_row270_col12" class="data row270 col12" >0.017800</td>
          <td id="T_78b05_row270_col13" class="data row270 col13" >0.001400</td>
          <td id="T_78b05_row270_col14" class="data row270 col14" >0.047300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row271" class="row_heading level0 row271" >272</th>
          <td id="T_78b05_row271_col0" class="data row271 col0" >None</td>
          <td id="T_78b05_row271_col1" class="data row271 col1" >0.041000</td>
          <td id="T_78b05_row271_col2" class="data row271 col2" >-0.068900</td>
          <td id="T_78b05_row271_col3" class="data row271 col3" >0.005600</td>
          <td id="T_78b05_row271_col4" class="data row271 col4" >0.004600</td>
          <td id="T_78b05_row271_col5" class="data row271 col5" >-0.020500</td>
          <td id="T_78b05_row271_col6" class="data row271 col6" >0.062200</td>
          <td id="T_78b05_row271_col7" class="data row271 col7" >-0.037600</td>
          <td id="T_78b05_row271_col8" class="data row271 col8" >0.004500</td>
          <td id="T_78b05_row271_col9" class="data row271 col9" >0.039100</td>
          <td id="T_78b05_row271_col10" class="data row271 col10" >0.025100</td>
          <td id="T_78b05_row271_col11" class="data row271 col11" >0.022500</td>
          <td id="T_78b05_row271_col12" class="data row271 col12" >0.019900</td>
          <td id="T_78b05_row271_col13" class="data row271 col13" >0.066100</td>
          <td id="T_78b05_row271_col14" class="data row271 col14" >0.035800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row272" class="row_heading level0 row272" >273</th>
          <td id="T_78b05_row272_col0" class="data row272 col0" >None</td>
          <td id="T_78b05_row272_col1" class="data row272 col1" >0.031100</td>
          <td id="T_78b05_row272_col2" class="data row272 col2" >0.059400</td>
          <td id="T_78b05_row272_col3" class="data row272 col3" >0.031900</td>
          <td id="T_78b05_row272_col4" class="data row272 col4" >-0.001600</td>
          <td id="T_78b05_row272_col5" class="data row272 col5" >0.003100</td>
          <td id="T_78b05_row272_col6" class="data row272 col6" >0.008600</td>
          <td id="T_78b05_row272_col7" class="data row272 col7" >0.015600</td>
          <td id="T_78b05_row272_col8" class="data row272 col8" >0.014400</td>
          <td id="T_78b05_row272_col9" class="data row272 col9" >0.089200</td>
          <td id="T_78b05_row272_col10" class="data row272 col10" >0.001100</td>
          <td id="T_78b05_row272_col11" class="data row272 col11" >0.016300</td>
          <td id="T_78b05_row272_col12" class="data row272 col12" >0.003800</td>
          <td id="T_78b05_row272_col13" class="data row272 col13" >0.012500</td>
          <td id="T_78b05_row272_col14" class="data row272 col14" >0.017400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row273" class="row_heading level0 row273" >274</th>
          <td id="T_78b05_row273_col0" class="data row273 col0" >None</td>
          <td id="T_78b05_row273_col1" class="data row273 col1" >0.041100</td>
          <td id="T_78b05_row273_col2" class="data row273 col2" >-0.007400</td>
          <td id="T_78b05_row273_col3" class="data row273 col3" >0.010000</td>
          <td id="T_78b05_row273_col4" class="data row273 col4" >0.022000</td>
          <td id="T_78b05_row273_col5" class="data row273 col5" >0.021900</td>
          <td id="T_78b05_row273_col6" class="data row273 col6" >-0.007100</td>
          <td id="T_78b05_row273_col7" class="data row273 col7" >-0.050800</td>
          <td id="T_78b05_row273_col8" class="data row273 col8" >0.004400</td>
          <td id="T_78b05_row273_col9" class="data row273 col9" >0.022500</td>
          <td id="T_78b05_row273_col10" class="data row273 col10" >0.020700</td>
          <td id="T_78b05_row273_col11" class="data row273 col11" >0.039900</td>
          <td id="T_78b05_row273_col12" class="data row273 col12" >0.022500</td>
          <td id="T_78b05_row273_col13" class="data row273 col13" >0.003200</td>
          <td id="T_78b05_row273_col14" class="data row273 col14" >0.049100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row274" class="row_heading level0 row274" >275</th>
          <td id="T_78b05_row274_col0" class="data row274 col0" >None</td>
          <td id="T_78b05_row274_col1" class="data row274 col1" >0.029800</td>
          <td id="T_78b05_row274_col2" class="data row274 col2" >0.006700</td>
          <td id="T_78b05_row274_col3" class="data row274 col3" >-0.043500</td>
          <td id="T_78b05_row274_col4" class="data row274 col4" >0.026500</td>
          <td id="T_78b05_row274_col5" class="data row274 col5" >-0.053400</td>
          <td id="T_78b05_row274_col6" class="data row274 col6" >0.079500</td>
          <td id="T_78b05_row274_col7" class="data row274 col7" >0.002200</td>
          <td id="T_78b05_row274_col8" class="data row274 col8" >0.015700</td>
          <td id="T_78b05_row274_col9" class="data row274 col9" >0.036500</td>
          <td id="T_78b05_row274_col10" class="data row274 col10" >0.074200</td>
          <td id="T_78b05_row274_col11" class="data row274 col11" >0.044400</td>
          <td id="T_78b05_row274_col12" class="data row274 col12" >0.052800</td>
          <td id="T_78b05_row274_col13" class="data row274 col13" >0.083400</td>
          <td id="T_78b05_row274_col14" class="data row274 col14" >0.003900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row275" class="row_heading level0 row275" >276</th>
          <td id="T_78b05_row275_col0" class="data row275 col0" >None</td>
          <td id="T_78b05_row275_col1" class="data row275 col1" >0.036500</td>
          <td id="T_78b05_row275_col2" class="data row275 col2" >-0.000200</td>
          <td id="T_78b05_row275_col3" class="data row275 col3" >0.016500</td>
          <td id="T_78b05_row275_col4" class="data row275 col4" >0.034100</td>
          <td id="T_78b05_row275_col5" class="data row275 col5" >-0.005100</td>
          <td id="T_78b05_row275_col6" class="data row275 col6" >-0.027200</td>
          <td id="T_78b05_row275_col7" class="data row275 col7" >-0.021400</td>
          <td id="T_78b05_row275_col8" class="data row275 col8" >0.009000</td>
          <td id="T_78b05_row275_col9" class="data row275 col9" >0.029600</td>
          <td id="T_78b05_row275_col10" class="data row275 col10" >0.014300</td>
          <td id="T_78b05_row275_col11" class="data row275 col11" >0.052000</td>
          <td id="T_78b05_row275_col12" class="data row275 col12" >0.004500</td>
          <td id="T_78b05_row275_col13" class="data row275 col13" >0.023300</td>
          <td id="T_78b05_row275_col14" class="data row275 col14" >0.019700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row276" class="row_heading level0 row276" >277</th>
          <td id="T_78b05_row276_col0" class="data row276 col0" >None</td>
          <td id="T_78b05_row276_col1" class="data row276 col1" >0.030800</td>
          <td id="T_78b05_row276_col2" class="data row276 col2" >0.077000</td>
          <td id="T_78b05_row276_col3" class="data row276 col3" >-0.007200</td>
          <td id="T_78b05_row276_col4" class="data row276 col4" >0.089300</td>
          <td id="T_78b05_row276_col5" class="data row276 col5" >0.017000</td>
          <td id="T_78b05_row276_col6" class="data row276 col6" >0.027000</td>
          <td id="T_78b05_row276_col7" class="data row276 col7" >-0.015200</td>
          <td id="T_78b05_row276_col8" class="data row276 col8" >0.014700</td>
          <td id="T_78b05_row276_col9" class="data row276 col9" >0.106900</td>
          <td id="T_78b05_row276_col10" class="data row276 col10" >0.038000</td>
          <td id="T_78b05_row276_col11" class="data row276 col11" >0.107200</td>
          <td id="T_78b05_row276_col12" class="data row276 col12" >0.017600</td>
          <td id="T_78b05_row276_col13" class="data row276 col13" >0.030900</td>
          <td id="T_78b05_row276_col14" class="data row276 col14" >0.013400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row277" class="row_heading level0 row277" >278</th>
          <td id="T_78b05_row277_col0" class="data row277 col0" >None</td>
          <td id="T_78b05_row277_col1" class="data row277 col1" >0.040300</td>
          <td id="T_78b05_row277_col2" class="data row277 col2" >0.075800</td>
          <td id="T_78b05_row277_col3" class="data row277 col3" >0.089800</td>
          <td id="T_78b05_row277_col4" class="data row277 col4" >-0.015900</td>
          <td id="T_78b05_row277_col5" class="data row277 col5" >-0.004900</td>
          <td id="T_78b05_row277_col6" class="data row277 col6" >-0.034300</td>
          <td id="T_78b05_row277_col7" class="data row277 col7" >-0.067600</td>
          <td id="T_78b05_row277_col8" class="data row277 col8" >0.005200</td>
          <td id="T_78b05_row277_col9" class="data row277 col9" >0.105700</td>
          <td id="T_78b05_row277_col10" class="data row277 col10" >0.059100</td>
          <td id="T_78b05_row277_col11" class="data row277 col11" >0.002000</td>
          <td id="T_78b05_row277_col12" class="data row277 col12" >0.004300</td>
          <td id="T_78b05_row277_col13" class="data row277 col13" >0.030400</td>
          <td id="T_78b05_row277_col14" class="data row277 col14" >0.065800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row278" class="row_heading level0 row278" >279</th>
          <td id="T_78b05_row278_col0" class="data row278 col0" >None</td>
          <td id="T_78b05_row278_col1" class="data row278 col1" >0.047000</td>
          <td id="T_78b05_row278_col2" class="data row278 col2" >-0.013100</td>
          <td id="T_78b05_row278_col3" class="data row278 col3" >0.032300</td>
          <td id="T_78b05_row278_col4" class="data row278 col4" >0.009800</td>
          <td id="T_78b05_row278_col5" class="data row278 col5" >0.014900</td>
          <td id="T_78b05_row278_col6" class="data row278 col6" >0.038600</td>
          <td id="T_78b05_row278_col7" class="data row278 col7" >0.004200</td>
          <td id="T_78b05_row278_col8" class="data row278 col8" >0.001500</td>
          <td id="T_78b05_row278_col9" class="data row278 col9" >0.016800</td>
          <td id="T_78b05_row278_col10" class="data row278 col10" >0.001500</td>
          <td id="T_78b05_row278_col11" class="data row278 col11" >0.027700</td>
          <td id="T_78b05_row278_col12" class="data row278 col12" >0.015500</td>
          <td id="T_78b05_row278_col13" class="data row278 col13" >0.042500</td>
          <td id="T_78b05_row278_col14" class="data row278 col14" >0.005900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row279" class="row_heading level0 row279" >280</th>
          <td id="T_78b05_row279_col0" class="data row279 col0" >None</td>
          <td id="T_78b05_row279_col1" class="data row279 col1" >0.038800</td>
          <td id="T_78b05_row279_col2" class="data row279 col2" >0.049000</td>
          <td id="T_78b05_row279_col3" class="data row279 col3" >-0.022800</td>
          <td id="T_78b05_row279_col4" class="data row279 col4" >-0.011800</td>
          <td id="T_78b05_row279_col5" class="data row279 col5" >0.065900</td>
          <td id="T_78b05_row279_col6" class="data row279 col6" >-0.043300</td>
          <td id="T_78b05_row279_col7" class="data row279 col7" >0.009100</td>
          <td id="T_78b05_row279_col8" class="data row279 col8" >0.006700</td>
          <td id="T_78b05_row279_col9" class="data row279 col9" >0.078800</td>
          <td id="T_78b05_row279_col10" class="data row279 col10" >0.053500</td>
          <td id="T_78b05_row279_col11" class="data row279 col11" >0.006100</td>
          <td id="T_78b05_row279_col12" class="data row279 col12" >0.066500</td>
          <td id="T_78b05_row279_col13" class="data row279 col13" >0.039400</td>
          <td id="T_78b05_row279_col14" class="data row279 col14" >0.010900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row280" class="row_heading level0 row280" >281</th>
          <td id="T_78b05_row280_col0" class="data row280 col0" >None</td>
          <td id="T_78b05_row280_col1" class="data row280 col1" >0.048400</td>
          <td id="T_78b05_row280_col2" class="data row280 col2" >-0.018700</td>
          <td id="T_78b05_row280_col3" class="data row280 col3" >0.040600</td>
          <td id="T_78b05_row280_col4" class="data row280 col4" >-0.018900</td>
          <td id="T_78b05_row280_col5" class="data row280 col5" >0.006700</td>
          <td id="T_78b05_row280_col6" class="data row280 col6" >0.039300</td>
          <td id="T_78b05_row280_col7" class="data row280 col7" >0.049400</td>
          <td id="T_78b05_row280_col8" class="data row280 col8" >0.002800</td>
          <td id="T_78b05_row280_col9" class="data row280 col9" >0.011100</td>
          <td id="T_78b05_row280_col10" class="data row280 col10" >0.009800</td>
          <td id="T_78b05_row280_col11" class="data row280 col11" >0.001000</td>
          <td id="T_78b05_row280_col12" class="data row280 col12" >0.007400</td>
          <td id="T_78b05_row280_col13" class="data row280 col13" >0.043200</td>
          <td id="T_78b05_row280_col14" class="data row280 col14" >0.051100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row281" class="row_heading level0 row281" >282</th>
          <td id="T_78b05_row281_col0" class="data row281 col0" >None</td>
          <td id="T_78b05_row281_col1" class="data row281 col1" >0.036800</td>
          <td id="T_78b05_row281_col2" class="data row281 col2" >-0.008800</td>
          <td id="T_78b05_row281_col3" class="data row281 col3" >-0.084500</td>
          <td id="T_78b05_row281_col4" class="data row281 col4" >-0.027300</td>
          <td id="T_78b05_row281_col5" class="data row281 col5" >0.067600</td>
          <td id="T_78b05_row281_col6" class="data row281 col6" >0.065800</td>
          <td id="T_78b05_row281_col7" class="data row281 col7" >-0.039900</td>
          <td id="T_78b05_row281_col8" class="data row281 col8" >0.008800</td>
          <td id="T_78b05_row281_col9" class="data row281 col9" >0.021000</td>
          <td id="T_78b05_row281_col10" class="data row281 col10" >0.115200</td>
          <td id="T_78b05_row281_col11" class="data row281 col11" >0.009400</td>
          <td id="T_78b05_row281_col12" class="data row281 col12" >0.068200</td>
          <td id="T_78b05_row281_col13" class="data row281 col13" >0.069700</td>
          <td id="T_78b05_row281_col14" class="data row281 col14" >0.038100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row282" class="row_heading level0 row282" >283</th>
          <td id="T_78b05_row282_col0" class="data row282 col0" >None</td>
          <td id="T_78b05_row282_col1" class="data row282 col1" >0.034500</td>
          <td id="T_78b05_row282_col2" class="data row282 col2" >0.047200</td>
          <td id="T_78b05_row282_col3" class="data row282 col3" >-0.022100</td>
          <td id="T_78b05_row282_col4" class="data row282 col4" >0.018200</td>
          <td id="T_78b05_row282_col5" class="data row282 col5" >0.065300</td>
          <td id="T_78b05_row282_col6" class="data row282 col6" >-0.006300</td>
          <td id="T_78b05_row282_col7" class="data row282 col7" >0.018400</td>
          <td id="T_78b05_row282_col8" class="data row282 col8" >0.011000</td>
          <td id="T_78b05_row282_col9" class="data row282 col9" >0.077000</td>
          <td id="T_78b05_row282_col10" class="data row282 col10" >0.052900</td>
          <td id="T_78b05_row282_col11" class="data row282 col11" >0.036200</td>
          <td id="T_78b05_row282_col12" class="data row282 col12" >0.066000</td>
          <td id="T_78b05_row282_col13" class="data row282 col13" >0.002400</td>
          <td id="T_78b05_row282_col14" class="data row282 col14" >0.020100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row283" class="row_heading level0 row283" >284</th>
          <td id="T_78b05_row283_col0" class="data row283 col0" >None</td>
          <td id="T_78b05_row283_col1" class="data row283 col1" >0.038900</td>
          <td id="T_78b05_row283_col2" class="data row283 col2" >-0.039400</td>
          <td id="T_78b05_row283_col3" class="data row283 col3" >-0.085700</td>
          <td id="T_78b05_row283_col4" class="data row283 col4" >0.006300</td>
          <td id="T_78b05_row283_col5" class="data row283 col5" >0.055400</td>
          <td id="T_78b05_row283_col6" class="data row283 col6" >0.073200</td>
          <td id="T_78b05_row283_col7" class="data row283 col7" >0.040900</td>
          <td id="T_78b05_row283_col8" class="data row283 col8" >0.006600</td>
          <td id="T_78b05_row283_col9" class="data row283 col9" >0.009600</td>
          <td id="T_78b05_row283_col10" class="data row283 col10" >0.116500</td>
          <td id="T_78b05_row283_col11" class="data row283 col11" >0.024200</td>
          <td id="T_78b05_row283_col12" class="data row283 col12" >0.056000</td>
          <td id="T_78b05_row283_col13" class="data row283 col13" >0.077000</td>
          <td id="T_78b05_row283_col14" class="data row283 col14" >0.042700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row284" class="row_heading level0 row284" >285</th>
          <td id="T_78b05_row284_col0" class="data row284 col0" >None</td>
          <td id="T_78b05_row284_col1" class="data row284 col1" >0.040100</td>
          <td id="T_78b05_row284_col2" class="data row284 col2" >0.049600</td>
          <td id="T_78b05_row284_col3" class="data row284 col3" >0.009100</td>
          <td id="T_78b05_row284_col4" class="data row284 col4" >-0.022300</td>
          <td id="T_78b05_row284_col5" class="data row284 col5" >0.002900</td>
          <td id="T_78b05_row284_col6" class="data row284 col6" >-0.024000</td>
          <td id="T_78b05_row284_col7" class="data row284 col7" >-0.012400</td>
          <td id="T_78b05_row284_col8" class="data row284 col8" >0.005400</td>
          <td id="T_78b05_row284_col9" class="data row284 col9" >0.079500</td>
          <td id="T_78b05_row284_col10" class="data row284 col10" >0.021600</td>
          <td id="T_78b05_row284_col11" class="data row284 col11" >0.004400</td>
          <td id="T_78b05_row284_col12" class="data row284 col12" >0.003500</td>
          <td id="T_78b05_row284_col13" class="data row284 col13" >0.020100</td>
          <td id="T_78b05_row284_col14" class="data row284 col14" >0.010600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row285" class="row_heading level0 row285" >286</th>
          <td id="T_78b05_row285_col0" class="data row285 col0" >None</td>
          <td id="T_78b05_row285_col1" class="data row285 col1" >0.036900</td>
          <td id="T_78b05_row285_col2" class="data row285 col2" >0.011400</td>
          <td id="T_78b05_row285_col3" class="data row285 col3" >0.051900</td>
          <td id="T_78b05_row285_col4" class="data row285 col4" >-0.025100</td>
          <td id="T_78b05_row285_col5" class="data row285 col5" >-0.031900</td>
          <td id="T_78b05_row285_col6" class="data row285 col6" >-0.015400</td>
          <td id="T_78b05_row285_col7" class="data row285 col7" >0.018600</td>
          <td id="T_78b05_row285_col8" class="data row285 col8" >0.008600</td>
          <td id="T_78b05_row285_col9" class="data row285 col9" >0.041300</td>
          <td id="T_78b05_row285_col10" class="data row285 col10" >0.021200</td>
          <td id="T_78b05_row285_col11" class="data row285 col11" >0.007100</td>
          <td id="T_78b05_row285_col12" class="data row285 col12" >0.031300</td>
          <td id="T_78b05_row285_col13" class="data row285 col13" >0.011500</td>
          <td id="T_78b05_row285_col14" class="data row285 col14" >0.020300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row286" class="row_heading level0 row286" >287</th>
          <td id="T_78b05_row286_col0" class="data row286 col0" >None</td>
          <td id="T_78b05_row286_col1" class="data row286 col1" >0.042100</td>
          <td id="T_78b05_row286_col2" class="data row286 col2" >0.000600</td>
          <td id="T_78b05_row286_col3" class="data row286 col3" >0.052900</td>
          <td id="T_78b05_row286_col4" class="data row286 col4" >0.042200</td>
          <td id="T_78b05_row286_col5" class="data row286 col5" >-0.044300</td>
          <td id="T_78b05_row286_col6" class="data row286 col6" >0.065000</td>
          <td id="T_78b05_row286_col7" class="data row286 col7" >-0.001500</td>
          <td id="T_78b05_row286_col8" class="data row286 col8" >0.003400</td>
          <td id="T_78b05_row286_col9" class="data row286 col9" >0.030500</td>
          <td id="T_78b05_row286_col10" class="data row286 col10" >0.022100</td>
          <td id="T_78b05_row286_col11" class="data row286 col11" >0.060100</td>
          <td id="T_78b05_row286_col12" class="data row286 col12" >0.043700</td>
          <td id="T_78b05_row286_col13" class="data row286 col13" >0.068900</td>
          <td id="T_78b05_row286_col14" class="data row286 col14" >0.000300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row287" class="row_heading level0 row287" >288</th>
          <td id="T_78b05_row287_col0" class="data row287 col0" >None</td>
          <td id="T_78b05_row287_col1" class="data row287 col1" >0.039400</td>
          <td id="T_78b05_row287_col2" class="data row287 col2" >0.023400</td>
          <td id="T_78b05_row287_col3" class="data row287 col3" >0.032800</td>
          <td id="T_78b05_row287_col4" class="data row287 col4" >0.023200</td>
          <td id="T_78b05_row287_col5" class="data row287 col5" >0.007400</td>
          <td id="T_78b05_row287_col6" class="data row287 col6" >-0.000500</td>
          <td id="T_78b05_row287_col7" class="data row287 col7" >0.006000</td>
          <td id="T_78b05_row287_col8" class="data row287 col8" >0.006100</td>
          <td id="T_78b05_row287_col9" class="data row287 col9" >0.053300</td>
          <td id="T_78b05_row287_col10" class="data row287 col10" >0.002000</td>
          <td id="T_78b05_row287_col11" class="data row287 col11" >0.041100</td>
          <td id="T_78b05_row287_col12" class="data row287 col12" >0.008000</td>
          <td id="T_78b05_row287_col13" class="data row287 col13" >0.003400</td>
          <td id="T_78b05_row287_col14" class="data row287 col14" >0.007800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row288" class="row_heading level0 row288" >289</th>
          <td id="T_78b05_row288_col0" class="data row288 col0" >None</td>
          <td id="T_78b05_row288_col1" class="data row288 col1" >0.039000</td>
          <td id="T_78b05_row288_col2" class="data row288 col2" >-0.011800</td>
          <td id="T_78b05_row288_col3" class="data row288 col3" >0.007700</td>
          <td id="T_78b05_row288_col4" class="data row288 col4" >-0.046400</td>
          <td id="T_78b05_row288_col5" class="data row288 col5" >0.002600</td>
          <td id="T_78b05_row288_col6" class="data row288 col6" >-0.031200</td>
          <td id="T_78b05_row288_col7" class="data row288 col7" >0.060800</td>
          <td id="T_78b05_row288_col8" class="data row288 col8" >0.006500</td>
          <td id="T_78b05_row288_col9" class="data row288 col9" >0.018000</td>
          <td id="T_78b05_row288_col10" class="data row288 col10" >0.023100</td>
          <td id="T_78b05_row288_col11" class="data row288 col11" >0.028500</td>
          <td id="T_78b05_row288_col12" class="data row288 col12" >0.003300</td>
          <td id="T_78b05_row288_col13" class="data row288 col13" >0.027300</td>
          <td id="T_78b05_row288_col14" class="data row288 col14" >0.062500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row289" class="row_heading level0 row289" >290</th>
          <td id="T_78b05_row289_col0" class="data row289 col0" >None</td>
          <td id="T_78b05_row289_col1" class="data row289 col1" >0.033000</td>
          <td id="T_78b05_row289_col2" class="data row289 col2" >0.013600</td>
          <td id="T_78b05_row289_col3" class="data row289 col3" >-0.029900</td>
          <td id="T_78b05_row289_col4" class="data row289 col4" >0.050400</td>
          <td id="T_78b05_row289_col5" class="data row289 col5" >-0.028300</td>
          <td id="T_78b05_row289_col6" class="data row289 col6" >-0.012900</td>
          <td id="T_78b05_row289_col7" class="data row289 col7" >0.005100</td>
          <td id="T_78b05_row289_col8" class="data row289 col8" >0.012500</td>
          <td id="T_78b05_row289_col9" class="data row289 col9" >0.043400</td>
          <td id="T_78b05_row289_col10" class="data row289 col10" >0.060600</td>
          <td id="T_78b05_row289_col11" class="data row289 col11" >0.068300</td>
          <td id="T_78b05_row289_col12" class="data row289 col12" >0.027600</td>
          <td id="T_78b05_row289_col13" class="data row289 col13" >0.009000</td>
          <td id="T_78b05_row289_col14" class="data row289 col14" >0.006900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row290" class="row_heading level0 row290" >291</th>
          <td id="T_78b05_row290_col0" class="data row290 col0" >None</td>
          <td id="T_78b05_row290_col1" class="data row290 col1" >0.038400</td>
          <td id="T_78b05_row290_col2" class="data row290 col2" >0.059700</td>
          <td id="T_78b05_row290_col3" class="data row290 col3" >0.013500</td>
          <td id="T_78b05_row290_col4" class="data row290 col4" >-0.044200</td>
          <td id="T_78b05_row290_col5" class="data row290 col5" >-0.020900</td>
          <td id="T_78b05_row290_col6" class="data row290 col6" >-0.010000</td>
          <td id="T_78b05_row290_col7" class="data row290 col7" >-0.047700</td>
          <td id="T_78b05_row290_col8" class="data row290 col8" >0.007100</td>
          <td id="T_78b05_row290_col9" class="data row290 col9" >0.089600</td>
          <td id="T_78b05_row290_col10" class="data row290 col10" >0.017300</td>
          <td id="T_78b05_row290_col11" class="data row290 col11" >0.026300</td>
          <td id="T_78b05_row290_col12" class="data row290 col12" >0.020300</td>
          <td id="T_78b05_row290_col13" class="data row290 col13" >0.006100</td>
          <td id="T_78b05_row290_col14" class="data row290 col14" >0.046000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row291" class="row_heading level0 row291" >292</th>
          <td id="T_78b05_row291_col0" class="data row291 col0" >None</td>
          <td id="T_78b05_row291_col1" class="data row291 col1" >0.041300</td>
          <td id="T_78b05_row291_col2" class="data row291 col2" >-0.032600</td>
          <td id="T_78b05_row291_col3" class="data row291 col3" >0.068000</td>
          <td id="T_78b05_row291_col4" class="data row291 col4" >0.023300</td>
          <td id="T_78b05_row291_col5" class="data row291 col5" >-0.025000</td>
          <td id="T_78b05_row291_col6" class="data row291 col6" >0.040700</td>
          <td id="T_78b05_row291_col7" class="data row291 col7" >-0.025600</td>
          <td id="T_78b05_row291_col8" class="data row291 col8" >0.004300</td>
          <td id="T_78b05_row291_col9" class="data row291 col9" >0.002800</td>
          <td id="T_78b05_row291_col10" class="data row291 col10" >0.037200</td>
          <td id="T_78b05_row291_col11" class="data row291 col11" >0.041300</td>
          <td id="T_78b05_row291_col12" class="data row291 col12" >0.024400</td>
          <td id="T_78b05_row291_col13" class="data row291 col13" >0.044600</td>
          <td id="T_78b05_row291_col14" class="data row291 col14" >0.023900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row292" class="row_heading level0 row292" >293</th>
          <td id="T_78b05_row292_col0" class="data row292 col0" >None</td>
          <td id="T_78b05_row292_col1" class="data row292 col1" >0.033700</td>
          <td id="T_78b05_row292_col2" class="data row292 col2" >0.051400</td>
          <td id="T_78b05_row292_col3" class="data row292 col3" >-0.004600</td>
          <td id="T_78b05_row292_col4" class="data row292 col4" >-0.051100</td>
          <td id="T_78b05_row292_col5" class="data row292 col5" >-0.040900</td>
          <td id="T_78b05_row292_col6" class="data row292 col6" >0.043800</td>
          <td id="T_78b05_row292_col7" class="data row292 col7" >-0.024100</td>
          <td id="T_78b05_row292_col8" class="data row292 col8" >0.011800</td>
          <td id="T_78b05_row292_col9" class="data row292 col9" >0.081200</td>
          <td id="T_78b05_row292_col10" class="data row292 col10" >0.035300</td>
          <td id="T_78b05_row292_col11" class="data row292 col11" >0.033200</td>
          <td id="T_78b05_row292_col12" class="data row292 col12" >0.040300</td>
          <td id="T_78b05_row292_col13" class="data row292 col13" >0.047700</td>
          <td id="T_78b05_row292_col14" class="data row292 col14" >0.022400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row293" class="row_heading level0 row293" >294</th>
          <td id="T_78b05_row293_col0" class="data row293 col0" >None</td>
          <td id="T_78b05_row293_col1" class="data row293 col1" >0.036300</td>
          <td id="T_78b05_row293_col2" class="data row293 col2" >0.002700</td>
          <td id="T_78b05_row293_col3" class="data row293 col3" >-0.022900</td>
          <td id="T_78b05_row293_col4" class="data row293 col4" >0.069600</td>
          <td id="T_78b05_row293_col5" class="data row293 col5" >-0.005700</td>
          <td id="T_78b05_row293_col6" class="data row293 col6" >0.040600</td>
          <td id="T_78b05_row293_col7" class="data row293 col7" >0.003300</td>
          <td id="T_78b05_row293_col8" class="data row293 col8" >0.009300</td>
          <td id="T_78b05_row293_col9" class="data row293 col9" >0.032600</td>
          <td id="T_78b05_row293_col10" class="data row293 col10" >0.053600</td>
          <td id="T_78b05_row293_col11" class="data row293 col11" >0.087500</td>
          <td id="T_78b05_row293_col12" class="data row293 col12" >0.005000</td>
          <td id="T_78b05_row293_col13" class="data row293 col13" >0.044400</td>
          <td id="T_78b05_row293_col14" class="data row293 col14" >0.005000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row294" class="row_heading level0 row294" >295</th>
          <td id="T_78b05_row294_col0" class="data row294 col0" >None</td>
          <td id="T_78b05_row294_col1" class="data row294 col1" >0.040800</td>
          <td id="T_78b05_row294_col2" class="data row294 col2" >-0.019900</td>
          <td id="T_78b05_row294_col3" class="data row294 col3" >0.045200</td>
          <td id="T_78b05_row294_col4" class="data row294 col4" >-0.007400</td>
          <td id="T_78b05_row294_col5" class="data row294 col5" >0.056300</td>
          <td id="T_78b05_row294_col6" class="data row294 col6" >-0.026300</td>
          <td id="T_78b05_row294_col7" class="data row294 col7" >-0.007800</td>
          <td id="T_78b05_row294_col8" class="data row294 col8" >0.004700</td>
          <td id="T_78b05_row294_col9" class="data row294 col9" >0.010000</td>
          <td id="T_78b05_row294_col10" class="data row294 col10" >0.014400</td>
          <td id="T_78b05_row294_col11" class="data row294 col11" >0.010500</td>
          <td id="T_78b05_row294_col12" class="data row294 col12" >0.057000</td>
          <td id="T_78b05_row294_col13" class="data row294 col13" >0.022400</td>
          <td id="T_78b05_row294_col14" class="data row294 col14" >0.006100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row295" class="row_heading level0 row295" >296</th>
          <td id="T_78b05_row295_col0" class="data row295 col0" >None</td>
          <td id="T_78b05_row295_col1" class="data row295 col1" >0.031900</td>
          <td id="T_78b05_row295_col2" class="data row295 col2" >0.046000</td>
          <td id="T_78b05_row295_col3" class="data row295 col3" >-0.032700</td>
          <td id="T_78b05_row295_col4" class="data row295 col4" >-0.040900</td>
          <td id="T_78b05_row295_col5" class="data row295 col5" >-0.008500</td>
          <td id="T_78b05_row295_col6" class="data row295 col6" >0.008300</td>
          <td id="T_78b05_row295_col7" class="data row295 col7" >-0.080800</td>
          <td id="T_78b05_row295_col8" class="data row295 col8" >0.013600</td>
          <td id="T_78b05_row295_col9" class="data row295 col9" >0.075800</td>
          <td id="T_78b05_row295_col10" class="data row295 col10" >0.063400</td>
          <td id="T_78b05_row295_col11" class="data row295 col11" >0.023000</td>
          <td id="T_78b05_row295_col12" class="data row295 col12" >0.007800</td>
          <td id="T_78b05_row295_col13" class="data row295 col13" >0.012200</td>
          <td id="T_78b05_row295_col14" class="data row295 col14" >0.079000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row296" class="row_heading level0 row296" >297</th>
          <td id="T_78b05_row296_col0" class="data row296 col0" >None</td>
          <td id="T_78b05_row296_col1" class="data row296 col1" >0.033200</td>
          <td id="T_78b05_row296_col2" class="data row296 col2" >0.038500</td>
          <td id="T_78b05_row296_col3" class="data row296 col3" >-0.011500</td>
          <td id="T_78b05_row296_col4" class="data row296 col4" >0.004000</td>
          <td id="T_78b05_row296_col5" class="data row296 col5" >-0.006200</td>
          <td id="T_78b05_row296_col6" class="data row296 col6" >-0.044900</td>
          <td id="T_78b05_row296_col7" class="data row296 col7" >-0.003800</td>
          <td id="T_78b05_row296_col8" class="data row296 col8" >0.012300</td>
          <td id="T_78b05_row296_col9" class="data row296 col9" >0.068300</td>
          <td id="T_78b05_row296_col10" class="data row296 col10" >0.042200</td>
          <td id="T_78b05_row296_col11" class="data row296 col11" >0.021900</td>
          <td id="T_78b05_row296_col12" class="data row296 col12" >0.005500</td>
          <td id="T_78b05_row296_col13" class="data row296 col13" >0.041000</td>
          <td id="T_78b05_row296_col14" class="data row296 col14" >0.002000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row297" class="row_heading level0 row297" >298</th>
          <td id="T_78b05_row297_col0" class="data row297 col0" >None</td>
          <td id="T_78b05_row297_col1" class="data row297 col1" >0.035800</td>
          <td id="T_78b05_row297_col2" class="data row297 col2" >-0.007600</td>
          <td id="T_78b05_row297_col3" class="data row297 col3" >-0.001900</td>
          <td id="T_78b05_row297_col4" class="data row297 col4" >-0.027000</td>
          <td id="T_78b05_row297_col5" class="data row297 col5" >-0.031800</td>
          <td id="T_78b05_row297_col6" class="data row297 col6" >0.000700</td>
          <td id="T_78b05_row297_col7" class="data row297 col7" >-0.043700</td>
          <td id="T_78b05_row297_col8" class="data row297 col8" >0.009700</td>
          <td id="T_78b05_row297_col9" class="data row297 col9" >0.022200</td>
          <td id="T_78b05_row297_col10" class="data row297 col10" >0.032600</td>
          <td id="T_78b05_row297_col11" class="data row297 col11" >0.009100</td>
          <td id="T_78b05_row297_col12" class="data row297 col12" >0.031200</td>
          <td id="T_78b05_row297_col13" class="data row297 col13" >0.004600</td>
          <td id="T_78b05_row297_col14" class="data row297 col14" >0.041900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row298" class="row_heading level0 row298" >299</th>
          <td id="T_78b05_row298_col0" class="data row298 col0" >None</td>
          <td id="T_78b05_row298_col1" class="data row298 col1" >0.036500</td>
          <td id="T_78b05_row298_col2" class="data row298 col2" >0.037200</td>
          <td id="T_78b05_row298_col3" class="data row298 col3" >0.015000</td>
          <td id="T_78b05_row298_col4" class="data row298 col4" >0.041700</td>
          <td id="T_78b05_row298_col5" class="data row298 col5" >0.004900</td>
          <td id="T_78b05_row298_col6" class="data row298 col6" >-0.042800</td>
          <td id="T_78b05_row298_col7" class="data row298 col7" >-0.053000</td>
          <td id="T_78b05_row298_col8" class="data row298 col8" >0.009000</td>
          <td id="T_78b05_row298_col9" class="data row298 col9" >0.067000</td>
          <td id="T_78b05_row298_col10" class="data row298 col10" >0.015700</td>
          <td id="T_78b05_row298_col11" class="data row298 col11" >0.059600</td>
          <td id="T_78b05_row298_col12" class="data row298 col12" >0.005600</td>
          <td id="T_78b05_row298_col13" class="data row298 col13" >0.038900</td>
          <td id="T_78b05_row298_col14" class="data row298 col14" >0.051200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row299" class="row_heading level0 row299" >300</th>
          <td id="T_78b05_row299_col0" class="data row299 col0" >PC1</td>
          <td id="T_78b05_row299_col1" class="data row299 col1" >0.024900</td>
          <td id="T_78b05_row299_col2" class="data row299 col2" >0.013500</td>
          <td id="T_78b05_row299_col3" class="data row299 col3" >-0.052900</td>
          <td id="T_78b05_row299_col4" class="data row299 col4" >-0.013700</td>
          <td id="T_78b05_row299_col5" class="data row299 col5" >0.005900</td>
          <td id="T_78b05_row299_col6" class="data row299 col6" >0.025900</td>
          <td id="T_78b05_row299_col7" class="data row299 col7" >-0.017200</td>
          <td id="T_78b05_row299_col8" class="data row299 col8" >0.020600</td>
          <td id="T_78b05_row299_col9" class="data row299 col9" >0.043300</td>
          <td id="T_78b05_row299_col10" class="data row299 col10" >0.083700</td>
          <td id="T_78b05_row299_col11" class="data row299 col11" >0.004200</td>
          <td id="T_78b05_row299_col12" class="data row299 col12" >0.006500</td>
          <td id="T_78b05_row299_col13" class="data row299 col13" >0.029800</td>
          <td id="T_78b05_row299_col14" class="data row299 col14" >0.015400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row300" class="row_heading level0 row300" >301</th>
          <td id="T_78b05_row300_col0" class="data row300 col0" >None</td>
          <td id="T_78b05_row300_col1" class="data row300 col1" >0.034300</td>
          <td id="T_78b05_row300_col2" class="data row300 col2" >0.020700</td>
          <td id="T_78b05_row300_col3" class="data row300 col3" >-0.000400</td>
          <td id="T_78b05_row300_col4" class="data row300 col4" >0.011900</td>
          <td id="T_78b05_row300_col5" class="data row300 col5" >0.066800</td>
          <td id="T_78b05_row300_col6" class="data row300 col6" >0.013500</td>
          <td id="T_78b05_row300_col7" class="data row300 col7" >0.062000</td>
          <td id="T_78b05_row300_col8" class="data row300 col8" >0.011300</td>
          <td id="T_78b05_row300_col9" class="data row300 col9" >0.050500</td>
          <td id="T_78b05_row300_col10" class="data row300 col10" >0.031100</td>
          <td id="T_78b05_row300_col11" class="data row300 col11" >0.029800</td>
          <td id="T_78b05_row300_col12" class="data row300 col12" >0.067400</td>
          <td id="T_78b05_row300_col13" class="data row300 col13" >0.017400</td>
          <td id="T_78b05_row300_col14" class="data row300 col14" >0.063800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row301" class="row_heading level0 row301" >302</th>
          <td id="T_78b05_row301_col0" class="data row301 col0" >None</td>
          <td id="T_78b05_row301_col1" class="data row301 col1" >0.043100</td>
          <td id="T_78b05_row301_col2" class="data row301 col2" >0.002200</td>
          <td id="T_78b05_row301_col3" class="data row301 col3" >0.026600</td>
          <td id="T_78b05_row301_col4" class="data row301 col4" >0.016500</td>
          <td id="T_78b05_row301_col5" class="data row301 col5" >0.048000</td>
          <td id="T_78b05_row301_col6" class="data row301 col6" >-0.043100</td>
          <td id="T_78b05_row301_col7" class="data row301 col7" >0.074100</td>
          <td id="T_78b05_row301_col8" class="data row301 col8" >0.002400</td>
          <td id="T_78b05_row301_col9" class="data row301 col9" >0.032100</td>
          <td id="T_78b05_row301_col10" class="data row301 col10" >0.004100</td>
          <td id="T_78b05_row301_col11" class="data row301 col11" >0.034400</td>
          <td id="T_78b05_row301_col12" class="data row301 col12" >0.048700</td>
          <td id="T_78b05_row301_col13" class="data row301 col13" >0.039200</td>
          <td id="T_78b05_row301_col14" class="data row301 col14" >0.075800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row302" class="row_heading level0 row302" >303</th>
          <td id="T_78b05_row302_col0" class="data row302 col0" >None</td>
          <td id="T_78b05_row302_col1" class="data row302 col1" >0.034800</td>
          <td id="T_78b05_row302_col2" class="data row302 col2" >-0.031900</td>
          <td id="T_78b05_row302_col3" class="data row302 col3" >-0.019800</td>
          <td id="T_78b05_row302_col4" class="data row302 col4" >-0.049000</td>
          <td id="T_78b05_row302_col5" class="data row302 col5" >-0.014400</td>
          <td id="T_78b05_row302_col6" class="data row302 col6" >-0.041900</td>
          <td id="T_78b05_row302_col7" class="data row302 col7" >0.014100</td>
          <td id="T_78b05_row302_col8" class="data row302 col8" >0.010700</td>
          <td id="T_78b05_row302_col9" class="data row302 col9" >0.002000</td>
          <td id="T_78b05_row302_col10" class="data row302 col10" >0.050600</td>
          <td id="T_78b05_row302_col11" class="data row302 col11" >0.031100</td>
          <td id="T_78b05_row302_col12" class="data row302 col12" >0.013800</td>
          <td id="T_78b05_row302_col13" class="data row302 col13" >0.038100</td>
          <td id="T_78b05_row302_col14" class="data row302 col14" >0.015800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row303" class="row_heading level0 row303" >304</th>
          <td id="T_78b05_row303_col0" class="data row303 col0" >None</td>
          <td id="T_78b05_row303_col1" class="data row303 col1" >0.029200</td>
          <td id="T_78b05_row303_col2" class="data row303 col2" >0.055700</td>
          <td id="T_78b05_row303_col3" class="data row303 col3" >-0.019200</td>
          <td id="T_78b05_row303_col4" class="data row303 col4" >0.004600</td>
          <td id="T_78b05_row303_col5" class="data row303 col5" >0.002600</td>
          <td id="T_78b05_row303_col6" class="data row303 col6" >-0.003300</td>
          <td id="T_78b05_row303_col7" class="data row303 col7" >-0.017300</td>
          <td id="T_78b05_row303_col8" class="data row303 col8" >0.016400</td>
          <td id="T_78b05_row303_col9" class="data row303 col9" >0.085500</td>
          <td id="T_78b05_row303_col10" class="data row303 col10" >0.049900</td>
          <td id="T_78b05_row303_col11" class="data row303 col11" >0.022600</td>
          <td id="T_78b05_row303_col12" class="data row303 col12" >0.003200</td>
          <td id="T_78b05_row303_col13" class="data row303 col13" >0.000600</td>
          <td id="T_78b05_row303_col14" class="data row303 col14" >0.015600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row304" class="row_heading level0 row304" >305</th>
          <td id="T_78b05_row304_col0" class="data row304 col0" >None</td>
          <td id="T_78b05_row304_col1" class="data row304 col1" >0.037300</td>
          <td id="T_78b05_row304_col2" class="data row304 col2" >0.011100</td>
          <td id="T_78b05_row304_col3" class="data row304 col3" >0.006800</td>
          <td id="T_78b05_row304_col4" class="data row304 col4" >0.096100</td>
          <td id="T_78b05_row304_col5" class="data row304 col5" >0.004600</td>
          <td id="T_78b05_row304_col6" class="data row304 col6" >-0.003400</td>
          <td id="T_78b05_row304_col7" class="data row304 col7" >0.048100</td>
          <td id="T_78b05_row304_col8" class="data row304 col8" >0.008300</td>
          <td id="T_78b05_row304_col9" class="data row304 col9" >0.040900</td>
          <td id="T_78b05_row304_col10" class="data row304 col10" >0.023900</td>
          <td id="T_78b05_row304_col11" class="data row304 col11" >0.114100</td>
          <td id="T_78b05_row304_col12" class="data row304 col12" >0.005300</td>
          <td id="T_78b05_row304_col13" class="data row304 col13" >0.000500</td>
          <td id="T_78b05_row304_col14" class="data row304 col14" >0.049900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row305" class="row_heading level0 row305" >306</th>
          <td id="T_78b05_row305_col0" class="data row305 col0" >None</td>
          <td id="T_78b05_row305_col1" class="data row305 col1" >0.043100</td>
          <td id="T_78b05_row305_col2" class="data row305 col2" >0.020300</td>
          <td id="T_78b05_row305_col3" class="data row305 col3" >0.013000</td>
          <td id="T_78b05_row305_col4" class="data row305 col4" >0.064400</td>
          <td id="T_78b05_row305_col5" class="data row305 col5" >0.055800</td>
          <td id="T_78b05_row305_col6" class="data row305 col6" >0.073600</td>
          <td id="T_78b05_row305_col7" class="data row305 col7" >-0.024800</td>
          <td id="T_78b05_row305_col8" class="data row305 col8" >0.002400</td>
          <td id="T_78b05_row305_col9" class="data row305 col9" >0.050100</td>
          <td id="T_78b05_row305_col10" class="data row305 col10" >0.017700</td>
          <td id="T_78b05_row305_col11" class="data row305 col11" >0.082300</td>
          <td id="T_78b05_row305_col12" class="data row305 col12" >0.056500</td>
          <td id="T_78b05_row305_col13" class="data row305 col13" >0.077500</td>
          <td id="T_78b05_row305_col14" class="data row305 col14" >0.023100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row306" class="row_heading level0 row306" >307</th>
          <td id="T_78b05_row306_col0" class="data row306 col0" >None</td>
          <td id="T_78b05_row306_col1" class="data row306 col1" >0.033400</td>
          <td id="T_78b05_row306_col2" class="data row306 col2" >0.024600</td>
          <td id="T_78b05_row306_col3" class="data row306 col3" >0.011100</td>
          <td id="T_78b05_row306_col4" class="data row306 col4" >-0.038900</td>
          <td id="T_78b05_row306_col5" class="data row306 col5" >-0.005800</td>
          <td id="T_78b05_row306_col6" class="data row306 col6" >0.059800</td>
          <td id="T_78b05_row306_col7" class="data row306 col7" >0.064400</td>
          <td id="T_78b05_row306_col8" class="data row306 col8" >0.012100</td>
          <td id="T_78b05_row306_col9" class="data row306 col9" >0.054400</td>
          <td id="T_78b05_row306_col10" class="data row306 col10" >0.019700</td>
          <td id="T_78b05_row306_col11" class="data row306 col11" >0.021000</td>
          <td id="T_78b05_row306_col12" class="data row306 col12" >0.005200</td>
          <td id="T_78b05_row306_col13" class="data row306 col13" >0.063700</td>
          <td id="T_78b05_row306_col14" class="data row306 col14" >0.066100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row307" class="row_heading level0 row307" >308</th>
          <td id="T_78b05_row307_col0" class="data row307 col0" >PC1</td>
          <td id="T_78b05_row307_col1" class="data row307 col1" >0.025400</td>
          <td id="T_78b05_row307_col2" class="data row307 col2" >0.030600</td>
          <td id="T_78b05_row307_col3" class="data row307 col3" >-0.033100</td>
          <td id="T_78b05_row307_col4" class="data row307 col4" >0.034500</td>
          <td id="T_78b05_row307_col5" class="data row307 col5" >-0.008800</td>
          <td id="T_78b05_row307_col6" class="data row307 col6" >0.019000</td>
          <td id="T_78b05_row307_col7" class="data row307 col7" >0.007100</td>
          <td id="T_78b05_row307_col8" class="data row307 col8" >0.020200</td>
          <td id="T_78b05_row307_col9" class="data row307 col9" >0.060400</td>
          <td id="T_78b05_row307_col10" class="data row307 col10" >0.063900</td>
          <td id="T_78b05_row307_col11" class="data row307 col11" >0.052400</td>
          <td id="T_78b05_row307_col12" class="data row307 col12" >0.008200</td>
          <td id="T_78b05_row307_col13" class="data row307 col13" >0.022900</td>
          <td id="T_78b05_row307_col14" class="data row307 col14" >0.008800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row308" class="row_heading level0 row308" >309</th>
          <td id="T_78b05_row308_col0" class="data row308 col0" >None</td>
          <td id="T_78b05_row308_col1" class="data row308 col1" >0.045300</td>
          <td id="T_78b05_row308_col2" class="data row308 col2" >0.028400</td>
          <td id="T_78b05_row308_col3" class="data row308 col3" >0.019700</td>
          <td id="T_78b05_row308_col4" class="data row308 col4" >-0.022800</td>
          <td id="T_78b05_row308_col5" class="data row308 col5" >0.056200</td>
          <td id="T_78b05_row308_col6" class="data row308 col6" >-0.016300</td>
          <td id="T_78b05_row308_col7" class="data row308 col7" >0.012700</td>
          <td id="T_78b05_row308_col8" class="data row308 col8" >0.000200</td>
          <td id="T_78b05_row308_col9" class="data row308 col9" >0.058200</td>
          <td id="T_78b05_row308_col10" class="data row308 col10" >0.011000</td>
          <td id="T_78b05_row308_col11" class="data row308 col11" >0.004900</td>
          <td id="T_78b05_row308_col12" class="data row308 col12" >0.056800</td>
          <td id="T_78b05_row308_col13" class="data row308 col13" >0.012400</td>
          <td id="T_78b05_row308_col14" class="data row308 col14" >0.014400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row309" class="row_heading level0 row309" >310</th>
          <td id="T_78b05_row309_col0" class="data row309 col0" >None</td>
          <td id="T_78b05_row309_col1" class="data row309 col1" >0.035200</td>
          <td id="T_78b05_row309_col2" class="data row309 col2" >-0.002500</td>
          <td id="T_78b05_row309_col3" class="data row309 col3" >0.031700</td>
          <td id="T_78b05_row309_col4" class="data row309 col4" >0.051300</td>
          <td id="T_78b05_row309_col5" class="data row309 col5" >-0.028500</td>
          <td id="T_78b05_row309_col6" class="data row309 col6" >-0.067000</td>
          <td id="T_78b05_row309_col7" class="data row309 col7" >-0.001500</td>
          <td id="T_78b05_row309_col8" class="data row309 col8" >0.010300</td>
          <td id="T_78b05_row309_col9" class="data row309 col9" >0.027400</td>
          <td id="T_78b05_row309_col10" class="data row309 col10" >0.000900</td>
          <td id="T_78b05_row309_col11" class="data row309 col11" >0.069200</td>
          <td id="T_78b05_row309_col12" class="data row309 col12" >0.027900</td>
          <td id="T_78b05_row309_col13" class="data row309 col13" >0.063100</td>
          <td id="T_78b05_row309_col14" class="data row309 col14" >0.000200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row310" class="row_heading level0 row310" >311</th>
          <td id="T_78b05_row310_col0" class="data row310 col0" >None</td>
          <td id="T_78b05_row310_col1" class="data row310 col1" >0.030000</td>
          <td id="T_78b05_row310_col2" class="data row310 col2" >0.027400</td>
          <td id="T_78b05_row310_col3" class="data row310 col3" >0.025300</td>
          <td id="T_78b05_row310_col4" class="data row310 col4" >-0.007900</td>
          <td id="T_78b05_row310_col5" class="data row310 col5" >-0.016000</td>
          <td id="T_78b05_row310_col6" class="data row310 col6" >-0.008400</td>
          <td id="T_78b05_row310_col7" class="data row310 col7" >-0.040700</td>
          <td id="T_78b05_row310_col8" class="data row310 col8" >0.015600</td>
          <td id="T_78b05_row310_col9" class="data row310 col9" >0.057200</td>
          <td id="T_78b05_row310_col10" class="data row310 col10" >0.005500</td>
          <td id="T_78b05_row310_col11" class="data row310 col11" >0.010000</td>
          <td id="T_78b05_row310_col12" class="data row310 col12" >0.015400</td>
          <td id="T_78b05_row310_col13" class="data row310 col13" >0.004500</td>
          <td id="T_78b05_row310_col14" class="data row310 col14" >0.038900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row311" class="row_heading level0 row311" >312</th>
          <td id="T_78b05_row311_col0" class="data row311 col0" >None</td>
          <td id="T_78b05_row311_col1" class="data row311 col1" >0.029600</td>
          <td id="T_78b05_row311_col2" class="data row311 col2" >-0.051200</td>
          <td id="T_78b05_row311_col3" class="data row311 col3" >-0.057900</td>
          <td id="T_78b05_row311_col4" class="data row311 col4" >0.015700</td>
          <td id="T_78b05_row311_col5" class="data row311 col5" >-0.017800</td>
          <td id="T_78b05_row311_col6" class="data row311 col6" >-0.010200</td>
          <td id="T_78b05_row311_col7" class="data row311 col7" >-0.056100</td>
          <td id="T_78b05_row311_col8" class="data row311 col8" >0.015900</td>
          <td id="T_78b05_row311_col9" class="data row311 col9" >0.021400</td>
          <td id="T_78b05_row311_col10" class="data row311 col10" >0.088700</td>
          <td id="T_78b05_row311_col11" class="data row311 col11" >0.033600</td>
          <td id="T_78b05_row311_col12" class="data row311 col12" >0.017200</td>
          <td id="T_78b05_row311_col13" class="data row311 col13" >0.006300</td>
          <td id="T_78b05_row311_col14" class="data row311 col14" >0.054300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row312" class="row_heading level0 row312" >313</th>
          <td id="T_78b05_row312_col0" class="data row312 col0" >None</td>
          <td id="T_78b05_row312_col1" class="data row312 col1" >0.031200</td>
          <td id="T_78b05_row312_col2" class="data row312 col2" >0.029900</td>
          <td id="T_78b05_row312_col3" class="data row312 col3" >-0.027600</td>
          <td id="T_78b05_row312_col4" class="data row312 col4" >-0.013900</td>
          <td id="T_78b05_row312_col5" class="data row312 col5" >-0.014000</td>
          <td id="T_78b05_row312_col6" class="data row312 col6" >-0.083000</td>
          <td id="T_78b05_row312_col7" class="data row312 col7" >-0.001400</td>
          <td id="T_78b05_row312_col8" class="data row312 col8" >0.014300</td>
          <td id="T_78b05_row312_col9" class="data row312 col9" >0.059800</td>
          <td id="T_78b05_row312_col10" class="data row312 col10" >0.058400</td>
          <td id="T_78b05_row312_col11" class="data row312 col11" >0.004000</td>
          <td id="T_78b05_row312_col12" class="data row312 col12" >0.013400</td>
          <td id="T_78b05_row312_col13" class="data row312 col13" >0.079100</td>
          <td id="T_78b05_row312_col14" class="data row312 col14" >0.000400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row313" class="row_heading level0 row313" >314</th>
          <td id="T_78b05_row313_col0" class="data row313 col0" >None</td>
          <td id="T_78b05_row313_col1" class="data row313 col1" >0.040400</td>
          <td id="T_78b05_row313_col2" class="data row313 col2" >-0.002200</td>
          <td id="T_78b05_row313_col3" class="data row313 col3" >-0.004900</td>
          <td id="T_78b05_row313_col4" class="data row313 col4" >-0.007900</td>
          <td id="T_78b05_row313_col5" class="data row313 col5" >0.059500</td>
          <td id="T_78b05_row313_col6" class="data row313 col6" >0.008000</td>
          <td id="T_78b05_row313_col7" class="data row313 col7" >-0.027700</td>
          <td id="T_78b05_row313_col8" class="data row313 col8" >0.005200</td>
          <td id="T_78b05_row313_col9" class="data row313 col9" >0.027600</td>
          <td id="T_78b05_row313_col10" class="data row313 col10" >0.035600</td>
          <td id="T_78b05_row313_col11" class="data row313 col11" >0.010000</td>
          <td id="T_78b05_row313_col12" class="data row313 col12" >0.060100</td>
          <td id="T_78b05_row313_col13" class="data row313 col13" >0.011900</td>
          <td id="T_78b05_row313_col14" class="data row313 col14" >0.025900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row314" class="row_heading level0 row314" >315</th>
          <td id="T_78b05_row314_col0" class="data row314 col0" >None</td>
          <td id="T_78b05_row314_col1" class="data row314 col1" >0.041400</td>
          <td id="T_78b05_row314_col2" class="data row314 col2" >-0.023600</td>
          <td id="T_78b05_row314_col3" class="data row314 col3" >0.007800</td>
          <td id="T_78b05_row314_col4" class="data row314 col4" >-0.034300</td>
          <td id="T_78b05_row314_col5" class="data row314 col5" >0.017900</td>
          <td id="T_78b05_row314_col6" class="data row314 col6" >-0.019500</td>
          <td id="T_78b05_row314_col7" class="data row314 col7" >0.062300</td>
          <td id="T_78b05_row314_col8" class="data row314 col8" >0.004200</td>
          <td id="T_78b05_row314_col9" class="data row314 col9" >0.006200</td>
          <td id="T_78b05_row314_col10" class="data row314 col10" >0.023000</td>
          <td id="T_78b05_row314_col11" class="data row314 col11" >0.016400</td>
          <td id="T_78b05_row314_col12" class="data row314 col12" >0.018600</td>
          <td id="T_78b05_row314_col13" class="data row314 col13" >0.015600</td>
          <td id="T_78b05_row314_col14" class="data row314 col14" >0.064100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row315" class="row_heading level0 row315" >316</th>
          <td id="T_78b05_row315_col0" class="data row315 col0" >None</td>
          <td id="T_78b05_row315_col1" class="data row315 col1" >0.035200</td>
          <td id="T_78b05_row315_col2" class="data row315 col2" >-0.018400</td>
          <td id="T_78b05_row315_col3" class="data row315 col3" >-0.048700</td>
          <td id="T_78b05_row315_col4" class="data row315 col4" >0.056700</td>
          <td id="T_78b05_row315_col5" class="data row315 col5" >-0.011900</td>
          <td id="T_78b05_row315_col6" class="data row315 col6" >-0.018100</td>
          <td id="T_78b05_row315_col7" class="data row315 col7" >-0.017400</td>
          <td id="T_78b05_row315_col8" class="data row315 col8" >0.010300</td>
          <td id="T_78b05_row315_col9" class="data row315 col9" >0.011400</td>
          <td id="T_78b05_row315_col10" class="data row315 col10" >0.079400</td>
          <td id="T_78b05_row315_col11" class="data row315 col11" >0.074600</td>
          <td id="T_78b05_row315_col12" class="data row315 col12" >0.011300</td>
          <td id="T_78b05_row315_col13" class="data row315 col13" >0.014200</td>
          <td id="T_78b05_row315_col14" class="data row315 col14" >0.015600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row316" class="row_heading level0 row316" >317</th>
          <td id="T_78b05_row316_col0" class="data row316 col0" >None</td>
          <td id="T_78b05_row316_col1" class="data row316 col1" >0.036700</td>
          <td id="T_78b05_row316_col2" class="data row316 col2" >-0.007200</td>
          <td id="T_78b05_row316_col3" class="data row316 col3" >-0.005600</td>
          <td id="T_78b05_row316_col4" class="data row316 col4" >0.017200</td>
          <td id="T_78b05_row316_col5" class="data row316 col5" >-0.084000</td>
          <td id="T_78b05_row316_col6" class="data row316 col6" >0.021000</td>
          <td id="T_78b05_row316_col7" class="data row316 col7" >0.042400</td>
          <td id="T_78b05_row316_col8" class="data row316 col8" >0.008800</td>
          <td id="T_78b05_row316_col9" class="data row316 col9" >0.022600</td>
          <td id="T_78b05_row316_col10" class="data row316 col10" >0.036300</td>
          <td id="T_78b05_row316_col11" class="data row316 col11" >0.035100</td>
          <td id="T_78b05_row316_col12" class="data row316 col12" >0.083400</td>
          <td id="T_78b05_row316_col13" class="data row316 col13" >0.024900</td>
          <td id="T_78b05_row316_col14" class="data row316 col14" >0.044100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row317" class="row_heading level0 row317" >318</th>
          <td id="T_78b05_row317_col0" class="data row317 col0" >None</td>
          <td id="T_78b05_row317_col1" class="data row317 col1" >0.034300</td>
          <td id="T_78b05_row317_col2" class="data row317 col2" >-0.014100</td>
          <td id="T_78b05_row317_col3" class="data row317 col3" >-0.025500</td>
          <td id="T_78b05_row317_col4" class="data row317 col4" >-0.075700</td>
          <td id="T_78b05_row317_col5" class="data row317 col5" >-0.028300</td>
          <td id="T_78b05_row317_col6" class="data row317 col6" >-0.025300</td>
          <td id="T_78b05_row317_col7" class="data row317 col7" >0.040100</td>
          <td id="T_78b05_row317_col8" class="data row317 col8" >0.011200</td>
          <td id="T_78b05_row317_col9" class="data row317 col9" >0.015700</td>
          <td id="T_78b05_row317_col10" class="data row317 col10" >0.056200</td>
          <td id="T_78b05_row317_col11" class="data row317 col11" >0.057800</td>
          <td id="T_78b05_row317_col12" class="data row317 col12" >0.027700</td>
          <td id="T_78b05_row317_col13" class="data row317 col13" >0.021400</td>
          <td id="T_78b05_row317_col14" class="data row317 col14" >0.041800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row318" class="row_heading level0 row318" >319</th>
          <td id="T_78b05_row318_col0" class="data row318 col0" >None</td>
          <td id="T_78b05_row318_col1" class="data row318 col1" >0.034000</td>
          <td id="T_78b05_row318_col2" class="data row318 col2" >0.053400</td>
          <td id="T_78b05_row318_col3" class="data row318 col3" >-0.039000</td>
          <td id="T_78b05_row318_col4" class="data row318 col4" >0.023000</td>
          <td id="T_78b05_row318_col5" class="data row318 col5" >0.007700</td>
          <td id="T_78b05_row318_col6" class="data row318 col6" >-0.016700</td>
          <td id="T_78b05_row318_col7" class="data row318 col7" >0.053400</td>
          <td id="T_78b05_row318_col8" class="data row318 col8" >0.011500</td>
          <td id="T_78b05_row318_col9" class="data row318 col9" >0.083200</td>
          <td id="T_78b05_row318_col10" class="data row318 col10" >0.069800</td>
          <td id="T_78b05_row318_col11" class="data row318 col11" >0.040900</td>
          <td id="T_78b05_row318_col12" class="data row318 col12" >0.008400</td>
          <td id="T_78b05_row318_col13" class="data row318 col13" >0.012800</td>
          <td id="T_78b05_row318_col14" class="data row318 col14" >0.055100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row319" class="row_heading level0 row319" >320</th>
          <td id="T_78b05_row319_col0" class="data row319 col0" >None</td>
          <td id="T_78b05_row319_col1" class="data row319 col1" >0.039400</td>
          <td id="T_78b05_row319_col2" class="data row319 col2" >-0.023000</td>
          <td id="T_78b05_row319_col3" class="data row319 col3" >-0.035800</td>
          <td id="T_78b05_row319_col4" class="data row319 col4" >0.030300</td>
          <td id="T_78b05_row319_col5" class="data row319 col5" >-0.021600</td>
          <td id="T_78b05_row319_col6" class="data row319 col6" >0.002700</td>
          <td id="T_78b05_row319_col7" class="data row319 col7" >0.065800</td>
          <td id="T_78b05_row319_col8" class="data row319 col8" >0.006200</td>
          <td id="T_78b05_row319_col9" class="data row319 col9" >0.006800</td>
          <td id="T_78b05_row319_col10" class="data row319 col10" >0.066500</td>
          <td id="T_78b05_row319_col11" class="data row319 col11" >0.048200</td>
          <td id="T_78b05_row319_col12" class="data row319 col12" >0.021000</td>
          <td id="T_78b05_row319_col13" class="data row319 col13" >0.006600</td>
          <td id="T_78b05_row319_col14" class="data row319 col14" >0.067500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row320" class="row_heading level0 row320" >321</th>
          <td id="T_78b05_row320_col0" class="data row320 col0" >None</td>
          <td id="T_78b05_row320_col1" class="data row320 col1" >0.045100</td>
          <td id="T_78b05_row320_col2" class="data row320 col2" >-0.000100</td>
          <td id="T_78b05_row320_col3" class="data row320 col3" >0.029000</td>
          <td id="T_78b05_row320_col4" class="data row320 col4" >0.090400</td>
          <td id="T_78b05_row320_col5" class="data row320 col5" >0.052700</td>
          <td id="T_78b05_row320_col6" class="data row320 col6" >-0.019500</td>
          <td id="T_78b05_row320_col7" class="data row320 col7" >0.001300</td>
          <td id="T_78b05_row320_col8" class="data row320 col8" >0.000500</td>
          <td id="T_78b05_row320_col9" class="data row320 col9" >0.029700</td>
          <td id="T_78b05_row320_col10" class="data row320 col10" >0.001700</td>
          <td id="T_78b05_row320_col11" class="data row320 col11" >0.108300</td>
          <td id="T_78b05_row320_col12" class="data row320 col12" >0.053300</td>
          <td id="T_78b05_row320_col13" class="data row320 col13" >0.015600</td>
          <td id="T_78b05_row320_col14" class="data row320 col14" >0.003100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row321" class="row_heading level0 row321" >322</th>
          <td id="T_78b05_row321_col0" class="data row321 col0" >PC2</td>
          <td id="T_78b05_row321_col1" class="data row321 col1" >0.041300</td>
          <td id="T_78b05_row321_col2" class="data row321 col2" >0.096300</td>
          <td id="T_78b05_row321_col3" class="data row321 col3" >0.075700</td>
          <td id="T_78b05_row321_col4" class="data row321 col4" >-0.004700</td>
          <td id="T_78b05_row321_col5" class="data row321 col5" >-0.014300</td>
          <td id="T_78b05_row321_col6" class="data row321 col6" >-0.045600</td>
          <td id="T_78b05_row321_col7" class="data row321 col7" >-0.031700</td>
          <td id="T_78b05_row321_col8" class="data row321 col8" >0.004200</td>
          <td id="T_78b05_row321_col9" class="data row321 col9" >0.126100</td>
          <td id="T_78b05_row321_col10" class="data row321 col10" >0.044900</td>
          <td id="T_78b05_row321_col11" class="data row321 col11" >0.013300</td>
          <td id="T_78b05_row321_col12" class="data row321 col12" >0.013600</td>
          <td id="T_78b05_row321_col13" class="data row321 col13" >0.041700</td>
          <td id="T_78b05_row321_col14" class="data row321 col14" >0.030000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row322" class="row_heading level0 row322" >323</th>
          <td id="T_78b05_row322_col0" class="data row322 col0" >None</td>
          <td id="T_78b05_row322_col1" class="data row322 col1" >0.035000</td>
          <td id="T_78b05_row322_col2" class="data row322 col2" >0.011900</td>
          <td id="T_78b05_row322_col3" class="data row322 col3" >-0.039700</td>
          <td id="T_78b05_row322_col4" class="data row322 col4" >-0.025800</td>
          <td id="T_78b05_row322_col5" class="data row322 col5" >0.008500</td>
          <td id="T_78b05_row322_col6" class="data row322 col6" >-0.006300</td>
          <td id="T_78b05_row322_col7" class="data row322 col7" >0.045300</td>
          <td id="T_78b05_row322_col8" class="data row322 col8" >0.010600</td>
          <td id="T_78b05_row322_col9" class="data row322 col9" >0.041700</td>
          <td id="T_78b05_row322_col10" class="data row322 col10" >0.070400</td>
          <td id="T_78b05_row322_col11" class="data row322 col11" >0.007900</td>
          <td id="T_78b05_row322_col12" class="data row322 col12" >0.009100</td>
          <td id="T_78b05_row322_col13" class="data row322 col13" >0.002400</td>
          <td id="T_78b05_row322_col14" class="data row322 col14" >0.047000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row323" class="row_heading level0 row323" >324</th>
          <td id="T_78b05_row323_col0" class="data row323 col0" >None</td>
          <td id="T_78b05_row323_col1" class="data row323 col1" >0.033100</td>
          <td id="T_78b05_row323_col2" class="data row323 col2" >0.030300</td>
          <td id="T_78b05_row323_col3" class="data row323 col3" >0.030200</td>
          <td id="T_78b05_row323_col4" class="data row323 col4" >-0.042200</td>
          <td id="T_78b05_row323_col5" class="data row323 col5" >-0.049000</td>
          <td id="T_78b05_row323_col6" class="data row323 col6" >-0.014200</td>
          <td id="T_78b05_row323_col7" class="data row323 col7" >0.035400</td>
          <td id="T_78b05_row323_col8" class="data row323 col8" >0.012400</td>
          <td id="T_78b05_row323_col9" class="data row323 col9" >0.060100</td>
          <td id="T_78b05_row323_col10" class="data row323 col10" >0.000600</td>
          <td id="T_78b05_row323_col11" class="data row323 col11" >0.024300</td>
          <td id="T_78b05_row323_col12" class="data row323 col12" >0.048400</td>
          <td id="T_78b05_row323_col13" class="data row323 col13" >0.010300</td>
          <td id="T_78b05_row323_col14" class="data row323 col14" >0.037200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row324" class="row_heading level0 row324" >325</th>
          <td id="T_78b05_row324_col0" class="data row324 col0" >None</td>
          <td id="T_78b05_row324_col1" class="data row324 col1" >0.034100</td>
          <td id="T_78b05_row324_col2" class="data row324 col2" >0.052500</td>
          <td id="T_78b05_row324_col3" class="data row324 col3" >0.010900</td>
          <td id="T_78b05_row324_col4" class="data row324 col4" >-0.001500</td>
          <td id="T_78b05_row324_col5" class="data row324 col5" >-0.019200</td>
          <td id="T_78b05_row324_col6" class="data row324 col6" >0.019200</td>
          <td id="T_78b05_row324_col7" class="data row324 col7" >0.013500</td>
          <td id="T_78b05_row324_col8" class="data row324 col8" >0.011500</td>
          <td id="T_78b05_row324_col9" class="data row324 col9" >0.082300</td>
          <td id="T_78b05_row324_col10" class="data row324 col10" >0.019900</td>
          <td id="T_78b05_row324_col11" class="data row324 col11" >0.016400</td>
          <td id="T_78b05_row324_col12" class="data row324 col12" >0.018600</td>
          <td id="T_78b05_row324_col13" class="data row324 col13" >0.023100</td>
          <td id="T_78b05_row324_col14" class="data row324 col14" >0.015300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row325" class="row_heading level0 row325" >326</th>
          <td id="T_78b05_row325_col0" class="data row325 col0" >None</td>
          <td id="T_78b05_row325_col1" class="data row325 col1" >0.035600</td>
          <td id="T_78b05_row325_col2" class="data row325 col2" >0.019200</td>
          <td id="T_78b05_row325_col3" class="data row325 col3" >0.006400</td>
          <td id="T_78b05_row325_col4" class="data row325 col4" >-0.066900</td>
          <td id="T_78b05_row325_col5" class="data row325 col5" >0.032700</td>
          <td id="T_78b05_row325_col6" class="data row325 col6" >-0.039900</td>
          <td id="T_78b05_row325_col7" class="data row325 col7" >0.030400</td>
          <td id="T_78b05_row325_col8" class="data row325 col8" >0.010000</td>
          <td id="T_78b05_row325_col9" class="data row325 col9" >0.049000</td>
          <td id="T_78b05_row325_col10" class="data row325 col10" >0.024400</td>
          <td id="T_78b05_row325_col11" class="data row325 col11" >0.049000</td>
          <td id="T_78b05_row325_col12" class="data row325 col12" >0.033400</td>
          <td id="T_78b05_row325_col13" class="data row325 col13" >0.036000</td>
          <td id="T_78b05_row325_col14" class="data row325 col14" >0.032100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row326" class="row_heading level0 row326" >327</th>
          <td id="T_78b05_row326_col0" class="data row326 col0" >None</td>
          <td id="T_78b05_row326_col1" class="data row326 col1" >0.043700</td>
          <td id="T_78b05_row326_col2" class="data row326 col2" >0.011400</td>
          <td id="T_78b05_row326_col3" class="data row326 col3" >-0.003900</td>
          <td id="T_78b05_row326_col4" class="data row326 col4" >-0.006300</td>
          <td id="T_78b05_row326_col5" class="data row326 col5" >0.086100</td>
          <td id="T_78b05_row326_col6" class="data row326 col6" >0.024700</td>
          <td id="T_78b05_row326_col7" class="data row326 col7" >-0.030900</td>
          <td id="T_78b05_row326_col8" class="data row326 col8" >0.001900</td>
          <td id="T_78b05_row326_col9" class="data row326 col9" >0.041300</td>
          <td id="T_78b05_row326_col10" class="data row326 col10" >0.034600</td>
          <td id="T_78b05_row326_col11" class="data row326 col11" >0.011600</td>
          <td id="T_78b05_row326_col12" class="data row326 col12" >0.086800</td>
          <td id="T_78b05_row326_col13" class="data row326 col13" >0.028600</td>
          <td id="T_78b05_row326_col14" class="data row326 col14" >0.029100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row327" class="row_heading level0 row327" >328</th>
          <td id="T_78b05_row327_col0" class="data row327 col0" >None</td>
          <td id="T_78b05_row327_col1" class="data row327 col1" >0.036600</td>
          <td id="T_78b05_row327_col2" class="data row327 col2" >0.007700</td>
          <td id="T_78b05_row327_col3" class="data row327 col3" >0.003600</td>
          <td id="T_78b05_row327_col4" class="data row327 col4" >-0.019900</td>
          <td id="T_78b05_row327_col5" class="data row327 col5" >0.004400</td>
          <td id="T_78b05_row327_col6" class="data row327 col6" >-0.030200</td>
          <td id="T_78b05_row327_col7" class="data row327 col7" >-0.020600</td>
          <td id="T_78b05_row327_col8" class="data row327 col8" >0.009000</td>
          <td id="T_78b05_row327_col9" class="data row327 col9" >0.037600</td>
          <td id="T_78b05_row327_col10" class="data row327 col10" >0.027100</td>
          <td id="T_78b05_row327_col11" class="data row327 col11" >0.002000</td>
          <td id="T_78b05_row327_col12" class="data row327 col12" >0.005000</td>
          <td id="T_78b05_row327_col13" class="data row327 col13" >0.026300</td>
          <td id="T_78b05_row327_col14" class="data row327 col14" >0.018800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row328" class="row_heading level0 row328" >329</th>
          <td id="T_78b05_row328_col0" class="data row328 col0" >PC1</td>
          <td id="T_78b05_row328_col1" class="data row328 col1" >0.025800</td>
          <td id="T_78b05_row328_col2" class="data row328 col2" >0.014900</td>
          <td id="T_78b05_row328_col3" class="data row328 col3" >-0.043100</td>
          <td id="T_78b05_row328_col4" class="data row328 col4" >0.049200</td>
          <td id="T_78b05_row328_col5" class="data row328 col5" >-0.099300</td>
          <td id="T_78b05_row328_col6" class="data row328 col6" >-0.014800</td>
          <td id="T_78b05_row328_col7" class="data row328 col7" >0.102700</td>
          <td id="T_78b05_row328_col8" class="data row328 col8" >0.019700</td>
          <td id="T_78b05_row328_col9" class="data row328 col9" >0.044700</td>
          <td id="T_78b05_row328_col10" class="data row328 col10" >0.073900</td>
          <td id="T_78b05_row328_col11" class="data row328 col11" >0.067100</td>
          <td id="T_78b05_row328_col12" class="data row328 col12" >0.098700</td>
          <td id="T_78b05_row328_col13" class="data row328 col13" >0.010900</td>
          <td id="T_78b05_row328_col14" class="data row328 col14" >0.104500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row329" class="row_heading level0 row329" >330</th>
          <td id="T_78b05_row329_col0" class="data row329 col0" >None</td>
          <td id="T_78b05_row329_col1" class="data row329 col1" >0.035300</td>
          <td id="T_78b05_row329_col2" class="data row329 col2" >0.009700</td>
          <td id="T_78b05_row329_col3" class="data row329 col3" >-0.026300</td>
          <td id="T_78b05_row329_col4" class="data row329 col4" >0.051300</td>
          <td id="T_78b05_row329_col5" class="data row329 col5" >-0.007900</td>
          <td id="T_78b05_row329_col6" class="data row329 col6" >0.012400</td>
          <td id="T_78b05_row329_col7" class="data row329 col7" >0.036900</td>
          <td id="T_78b05_row329_col8" class="data row329 col8" >0.010200</td>
          <td id="T_78b05_row329_col9" class="data row329 col9" >0.039500</td>
          <td id="T_78b05_row329_col10" class="data row329 col10" >0.057100</td>
          <td id="T_78b05_row329_col11" class="data row329 col11" >0.069200</td>
          <td id="T_78b05_row329_col12" class="data row329 col12" >0.007200</td>
          <td id="T_78b05_row329_col13" class="data row329 col13" >0.016300</td>
          <td id="T_78b05_row329_col14" class="data row329 col14" >0.038700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row330" class="row_heading level0 row330" >331</th>
          <td id="T_78b05_row330_col0" class="data row330 col0" >None</td>
          <td id="T_78b05_row330_col1" class="data row330 col1" >0.038900</td>
          <td id="T_78b05_row330_col2" class="data row330 col2" >0.001700</td>
          <td id="T_78b05_row330_col3" class="data row330 col3" >0.006800</td>
          <td id="T_78b05_row330_col4" class="data row330 col4" >0.052400</td>
          <td id="T_78b05_row330_col5" class="data row330 col5" >-0.042100</td>
          <td id="T_78b05_row330_col6" class="data row330 col6" >-0.008500</td>
          <td id="T_78b05_row330_col7" class="data row330 col7" >0.034500</td>
          <td id="T_78b05_row330_col8" class="data row330 col8" >0.006700</td>
          <td id="T_78b05_row330_col9" class="data row330 col9" >0.031500</td>
          <td id="T_78b05_row330_col10" class="data row330 col10" >0.023900</td>
          <td id="T_78b05_row330_col11" class="data row330 col11" >0.070300</td>
          <td id="T_78b05_row330_col12" class="data row330 col12" >0.041400</td>
          <td id="T_78b05_row330_col13" class="data row330 col13" >0.004600</td>
          <td id="T_78b05_row330_col14" class="data row330 col14" >0.036200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row331" class="row_heading level0 row331" >332</th>
          <td id="T_78b05_row331_col0" class="data row331 col0" >None</td>
          <td id="T_78b05_row331_col1" class="data row331 col1" >0.031300</td>
          <td id="T_78b05_row331_col2" class="data row331 col2" >0.027400</td>
          <td id="T_78b05_row331_col3" class="data row331 col3" >-0.053400</td>
          <td id="T_78b05_row331_col4" class="data row331 col4" >-0.011100</td>
          <td id="T_78b05_row331_col5" class="data row331 col5" >0.008300</td>
          <td id="T_78b05_row331_col6" class="data row331 col6" >-0.073800</td>
          <td id="T_78b05_row331_col7" class="data row331 col7" >0.031400</td>
          <td id="T_78b05_row331_col8" class="data row331 col8" >0.014200</td>
          <td id="T_78b05_row331_col9" class="data row331 col9" >0.057300</td>
          <td id="T_78b05_row331_col10" class="data row331 col10" >0.084100</td>
          <td id="T_78b05_row331_col11" class="data row331 col11" >0.006800</td>
          <td id="T_78b05_row331_col12" class="data row331 col12" >0.009000</td>
          <td id="T_78b05_row331_col13" class="data row331 col13" >0.069900</td>
          <td id="T_78b05_row331_col14" class="data row331 col14" >0.033100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row332" class="row_heading level0 row332" >333</th>
          <td id="T_78b05_row332_col0" class="data row332 col0" >None</td>
          <td id="T_78b05_row332_col1" class="data row332 col1" >0.033300</td>
          <td id="T_78b05_row332_col2" class="data row332 col2" >0.071300</td>
          <td id="T_78b05_row332_col3" class="data row332 col3" >-0.003800</td>
          <td id="T_78b05_row332_col4" class="data row332 col4" >-0.022500</td>
          <td id="T_78b05_row332_col5" class="data row332 col5" >0.018800</td>
          <td id="T_78b05_row332_col6" class="data row332 col6" >0.054200</td>
          <td id="T_78b05_row332_col7" class="data row332 col7" >0.003900</td>
          <td id="T_78b05_row332_col8" class="data row332 col8" >0.012300</td>
          <td id="T_78b05_row332_col9" class="data row332 col9" >0.101100</td>
          <td id="T_78b05_row332_col10" class="data row332 col10" >0.034600</td>
          <td id="T_78b05_row332_col11" class="data row332 col11" >0.004600</td>
          <td id="T_78b05_row332_col12" class="data row332 col12" >0.019400</td>
          <td id="T_78b05_row332_col13" class="data row332 col13" >0.058100</td>
          <td id="T_78b05_row332_col14" class="data row332 col14" >0.005700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row333" class="row_heading level0 row333" >334</th>
          <td id="T_78b05_row333_col0" class="data row333 col0" >None</td>
          <td id="T_78b05_row333_col1" class="data row333 col1" >0.031800</td>
          <td id="T_78b05_row333_col2" class="data row333 col2" >-0.032300</td>
          <td id="T_78b05_row333_col3" class="data row333 col3" >-0.017400</td>
          <td id="T_78b05_row333_col4" class="data row333 col4" >-0.028300</td>
          <td id="T_78b05_row333_col5" class="data row333 col5" >-0.014700</td>
          <td id="T_78b05_row333_col6" class="data row333 col6" >-0.057300</td>
          <td id="T_78b05_row333_col7" class="data row333 col7" >0.045500</td>
          <td id="T_78b05_row333_col8" class="data row333 col8" >0.013700</td>
          <td id="T_78b05_row333_col9" class="data row333 col9" >0.002400</td>
          <td id="T_78b05_row333_col10" class="data row333 col10" >0.048100</td>
          <td id="T_78b05_row333_col11" class="data row333 col11" >0.010400</td>
          <td id="T_78b05_row333_col12" class="data row333 col12" >0.014100</td>
          <td id="T_78b05_row333_col13" class="data row333 col13" >0.053400</td>
          <td id="T_78b05_row333_col14" class="data row333 col14" >0.047200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row334" class="row_heading level0 row334" >335</th>
          <td id="T_78b05_row334_col0" class="data row334 col0" >None</td>
          <td id="T_78b05_row334_col1" class="data row334 col1" >0.042800</td>
          <td id="T_78b05_row334_col2" class="data row334 col2" >-0.025900</td>
          <td id="T_78b05_row334_col3" class="data row334 col3" >-0.001500</td>
          <td id="T_78b05_row334_col4" class="data row334 col4" >-0.043600</td>
          <td id="T_78b05_row334_col5" class="data row334 col5" >0.032400</td>
          <td id="T_78b05_row334_col6" class="data row334 col6" >-0.011600</td>
          <td id="T_78b05_row334_col7" class="data row334 col7" >0.001700</td>
          <td id="T_78b05_row334_col8" class="data row334 col8" >0.002700</td>
          <td id="T_78b05_row334_col9" class="data row334 col9" >0.004000</td>
          <td id="T_78b05_row334_col10" class="data row334 col10" >0.032300</td>
          <td id="T_78b05_row334_col11" class="data row334 col11" >0.025700</td>
          <td id="T_78b05_row334_col12" class="data row334 col12" >0.033100</td>
          <td id="T_78b05_row334_col13" class="data row334 col13" >0.007700</td>
          <td id="T_78b05_row334_col14" class="data row334 col14" >0.003500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row335" class="row_heading level0 row335" >336</th>
          <td id="T_78b05_row335_col0" class="data row335 col0" >PC1</td>
          <td id="T_78b05_row335_col1" class="data row335 col1" >0.026200</td>
          <td id="T_78b05_row335_col2" class="data row335 col2" >0.020700</td>
          <td id="T_78b05_row335_col3" class="data row335 col3" >-0.032200</td>
          <td id="T_78b05_row335_col4" class="data row335 col4" >0.064600</td>
          <td id="T_78b05_row335_col5" class="data row335 col5" >-0.058400</td>
          <td id="T_78b05_row335_col6" class="data row335 col6" >-0.025000</td>
          <td id="T_78b05_row335_col7" class="data row335 col7" >-0.033900</td>
          <td id="T_78b05_row335_col8" class="data row335 col8" >0.019300</td>
          <td id="T_78b05_row335_col9" class="data row335 col9" >0.050600</td>
          <td id="T_78b05_row335_col10" class="data row335 col10" >0.063000</td>
          <td id="T_78b05_row335_col11" class="data row335 col11" >0.082500</td>
          <td id="T_78b05_row335_col12" class="data row335 col12" >0.057700</td>
          <td id="T_78b05_row335_col13" class="data row335 col13" >0.021100</td>
          <td id="T_78b05_row335_col14" class="data row335 col14" >0.032200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row336" class="row_heading level0 row336" >337</th>
          <td id="T_78b05_row336_col0" class="data row336 col0" >PC2</td>
          <td id="T_78b05_row336_col1" class="data row336 col1" >0.035100</td>
          <td id="T_78b05_row336_col2" class="data row336 col2" >0.102700</td>
          <td id="T_78b05_row336_col3" class="data row336 col3" >0.021700</td>
          <td id="T_78b05_row336_col4" class="data row336 col4" >-0.030100</td>
          <td id="T_78b05_row336_col5" class="data row336 col5" >0.091900</td>
          <td id="T_78b05_row336_col6" class="data row336 col6" >0.030100</td>
          <td id="T_78b05_row336_col7" class="data row336 col7" >-0.017000</td>
          <td id="T_78b05_row336_col8" class="data row336 col8" >0.010400</td>
          <td id="T_78b05_row336_col9" class="data row336 col9" >0.132500</td>
          <td id="T_78b05_row336_col10" class="data row336 col10" >0.009100</td>
          <td id="T_78b05_row336_col11" class="data row336 col11" >0.012200</td>
          <td id="T_78b05_row336_col12" class="data row336 col12" >0.092500</td>
          <td id="T_78b05_row336_col13" class="data row336 col13" >0.034000</td>
          <td id="T_78b05_row336_col14" class="data row336 col14" >0.015200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row337" class="row_heading level0 row337" >338</th>
          <td id="T_78b05_row337_col0" class="data row337 col0" >None</td>
          <td id="T_78b05_row337_col1" class="data row337 col1" >0.038900</td>
          <td id="T_78b05_row337_col2" class="data row337 col2" >0.055400</td>
          <td id="T_78b05_row337_col3" class="data row337 col3" >0.037400</td>
          <td id="T_78b05_row337_col4" class="data row337 col4" >-0.020800</td>
          <td id="T_78b05_row337_col5" class="data row337 col5" >-0.035100</td>
          <td id="T_78b05_row337_col6" class="data row337 col6" >0.032500</td>
          <td id="T_78b05_row337_col7" class="data row337 col7" >0.073800</td>
          <td id="T_78b05_row337_col8" class="data row337 col8" >0.006700</td>
          <td id="T_78b05_row337_col9" class="data row337 col9" >0.085300</td>
          <td id="T_78b05_row337_col10" class="data row337 col10" >0.006600</td>
          <td id="T_78b05_row337_col11" class="data row337 col11" >0.002900</td>
          <td id="T_78b05_row337_col12" class="data row337 col12" >0.034500</td>
          <td id="T_78b05_row337_col13" class="data row337 col13" >0.036400</td>
          <td id="T_78b05_row337_col14" class="data row337 col14" >0.075500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row338" class="row_heading level0 row338" >339</th>
          <td id="T_78b05_row338_col0" class="data row338 col0" >None</td>
          <td id="T_78b05_row338_col1" class="data row338 col1" >0.032600</td>
          <td id="T_78b05_row338_col2" class="data row338 col2" >0.032600</td>
          <td id="T_78b05_row338_col3" class="data row338 col3" >-0.029300</td>
          <td id="T_78b05_row338_col4" class="data row338 col4" >-0.011900</td>
          <td id="T_78b05_row338_col5" class="data row338 col5" >-0.070200</td>
          <td id="T_78b05_row338_col6" class="data row338 col6" >0.047400</td>
          <td id="T_78b05_row338_col7" class="data row338 col7" >0.009100</td>
          <td id="T_78b05_row338_col8" class="data row338 col8" >0.012900</td>
          <td id="T_78b05_row338_col9" class="data row338 col9" >0.062400</td>
          <td id="T_78b05_row338_col10" class="data row338 col10" >0.060100</td>
          <td id="T_78b05_row338_col11" class="data row338 col11" >0.006000</td>
          <td id="T_78b05_row338_col12" class="data row338 col12" >0.069600</td>
          <td id="T_78b05_row338_col13" class="data row338 col13" >0.051300</td>
          <td id="T_78b05_row338_col14" class="data row338 col14" >0.010900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row339" class="row_heading level0 row339" >340</th>
          <td id="T_78b05_row339_col0" class="data row339 col0" >None</td>
          <td id="T_78b05_row339_col1" class="data row339 col1" >0.040800</td>
          <td id="T_78b05_row339_col2" class="data row339 col2" >0.016900</td>
          <td id="T_78b05_row339_col3" class="data row339 col3" >-0.006500</td>
          <td id="T_78b05_row339_col4" class="data row339 col4" >-0.047200</td>
          <td id="T_78b05_row339_col5" class="data row339 col5" >0.040900</td>
          <td id="T_78b05_row339_col6" class="data row339 col6" >-0.029300</td>
          <td id="T_78b05_row339_col7" class="data row339 col7" >-0.022500</td>
          <td id="T_78b05_row339_col8" class="data row339 col8" >0.004700</td>
          <td id="T_78b05_row339_col9" class="data row339 col9" >0.046800</td>
          <td id="T_78b05_row339_col10" class="data row339 col10" >0.037200</td>
          <td id="T_78b05_row339_col11" class="data row339 col11" >0.029300</td>
          <td id="T_78b05_row339_col12" class="data row339 col12" >0.041500</td>
          <td id="T_78b05_row339_col13" class="data row339 col13" >0.025400</td>
          <td id="T_78b05_row339_col14" class="data row339 col14" >0.020800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row340" class="row_heading level0 row340" >341</th>
          <td id="T_78b05_row340_col0" class="data row340 col0" >None</td>
          <td id="T_78b05_row340_col1" class="data row340 col1" >0.038200</td>
          <td id="T_78b05_row340_col2" class="data row340 col2" >0.050000</td>
          <td id="T_78b05_row340_col3" class="data row340 col3" >0.005900</td>
          <td id="T_78b05_row340_col4" class="data row340 col4" >0.011200</td>
          <td id="T_78b05_row340_col5" class="data row340 col5" >0.028500</td>
          <td id="T_78b05_row340_col6" class="data row340 col6" >0.069800</td>
          <td id="T_78b05_row340_col7" class="data row340 col7" >-0.010200</td>
          <td id="T_78b05_row340_col8" class="data row340 col8" >0.007300</td>
          <td id="T_78b05_row340_col9" class="data row340 col9" >0.079800</td>
          <td id="T_78b05_row340_col10" class="data row340 col10" >0.024800</td>
          <td id="T_78b05_row340_col11" class="data row340 col11" >0.029100</td>
          <td id="T_78b05_row340_col12" class="data row340 col12" >0.029100</td>
          <td id="T_78b05_row340_col13" class="data row340 col13" >0.073700</td>
          <td id="T_78b05_row340_col14" class="data row340 col14" >0.008500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row341" class="row_heading level0 row341" >342</th>
          <td id="T_78b05_row341_col0" class="data row341 col0" >None</td>
          <td id="T_78b05_row341_col1" class="data row341 col1" >0.038500</td>
          <td id="T_78b05_row341_col2" class="data row341 col2" >0.023100</td>
          <td id="T_78b05_row341_col3" class="data row341 col3" >-0.021600</td>
          <td id="T_78b05_row341_col4" class="data row341 col4" >0.019500</td>
          <td id="T_78b05_row341_col5" class="data row341 col5" >-0.033000</td>
          <td id="T_78b05_row341_col6" class="data row341 col6" >0.009100</td>
          <td id="T_78b05_row341_col7" class="data row341 col7" >0.026500</td>
          <td id="T_78b05_row341_col8" class="data row341 col8" >0.007000</td>
          <td id="T_78b05_row341_col9" class="data row341 col9" >0.052900</td>
          <td id="T_78b05_row341_col10" class="data row341 col10" >0.052400</td>
          <td id="T_78b05_row341_col11" class="data row341 col11" >0.037400</td>
          <td id="T_78b05_row341_col12" class="data row341 col12" >0.032400</td>
          <td id="T_78b05_row341_col13" class="data row341 col13" >0.013000</td>
          <td id="T_78b05_row341_col14" class="data row341 col14" >0.028300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row342" class="row_heading level0 row342" >343</th>
          <td id="T_78b05_row342_col0" class="data row342 col0" >None</td>
          <td id="T_78b05_row342_col1" class="data row342 col1" >0.032700</td>
          <td id="T_78b05_row342_col2" class="data row342 col2" >0.016800</td>
          <td id="T_78b05_row342_col3" class="data row342 col3" >-0.010300</td>
          <td id="T_78b05_row342_col4" class="data row342 col4" >-0.032900</td>
          <td id="T_78b05_row342_col5" class="data row342 col5" >-0.027000</td>
          <td id="T_78b05_row342_col6" class="data row342 col6" >0.042800</td>
          <td id="T_78b05_row342_col7" class="data row342 col7" >0.053800</td>
          <td id="T_78b05_row342_col8" class="data row342 col8" >0.012800</td>
          <td id="T_78b05_row342_col9" class="data row342 col9" >0.046700</td>
          <td id="T_78b05_row342_col10" class="data row342 col10" >0.041000</td>
          <td id="T_78b05_row342_col11" class="data row342 col11" >0.015000</td>
          <td id="T_78b05_row342_col12" class="data row342 col12" >0.026400</td>
          <td id="T_78b05_row342_col13" class="data row342 col13" >0.046700</td>
          <td id="T_78b05_row342_col14" class="data row342 col14" >0.055600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row343" class="row_heading level0 row343" >344</th>
          <td id="T_78b05_row343_col0" class="data row343 col0" >None</td>
          <td id="T_78b05_row343_col1" class="data row343 col1" >0.029800</td>
          <td id="T_78b05_row343_col2" class="data row343 col2" >0.047500</td>
          <td id="T_78b05_row343_col3" class="data row343 col3" >-0.012800</td>
          <td id="T_78b05_row343_col4" class="data row343 col4" >0.035100</td>
          <td id="T_78b05_row343_col5" class="data row343 col5" >-0.010200</td>
          <td id="T_78b05_row343_col6" class="data row343 col6" >0.039500</td>
          <td id="T_78b05_row343_col7" class="data row343 col7" >-0.009800</td>
          <td id="T_78b05_row343_col8" class="data row343 col8" >0.015700</td>
          <td id="T_78b05_row343_col9" class="data row343 col9" >0.077400</td>
          <td id="T_78b05_row343_col10" class="data row343 col10" >0.043500</td>
          <td id="T_78b05_row343_col11" class="data row343 col11" >0.053000</td>
          <td id="T_78b05_row343_col12" class="data row343 col12" >0.009500</td>
          <td id="T_78b05_row343_col13" class="data row343 col13" >0.043400</td>
          <td id="T_78b05_row343_col14" class="data row343 col14" >0.008100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row344" class="row_heading level0 row344" >345</th>
          <td id="T_78b05_row344_col0" class="data row344 col0" >None</td>
          <td id="T_78b05_row344_col1" class="data row344 col1" >0.037000</td>
          <td id="T_78b05_row344_col2" class="data row344 col2" >-0.052000</td>
          <td id="T_78b05_row344_col3" class="data row344 col3" >-0.021600</td>
          <td id="T_78b05_row344_col4" class="data row344 col4" >-0.031300</td>
          <td id="T_78b05_row344_col5" class="data row344 col5" >-0.036000</td>
          <td id="T_78b05_row344_col6" class="data row344 col6" >0.005500</td>
          <td id="T_78b05_row344_col7" class="data row344 col7" >-0.031700</td>
          <td id="T_78b05_row344_col8" class="data row344 col8" >0.008500</td>
          <td id="T_78b05_row344_col9" class="data row344 col9" >0.022200</td>
          <td id="T_78b05_row344_col10" class="data row344 col10" >0.052400</td>
          <td id="T_78b05_row344_col11" class="data row344 col11" >0.013400</td>
          <td id="T_78b05_row344_col12" class="data row344 col12" >0.035400</td>
          <td id="T_78b05_row344_col13" class="data row344 col13" >0.009400</td>
          <td id="T_78b05_row344_col14" class="data row344 col14" >0.029900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row345" class="row_heading level0 row345" >346</th>
          <td id="T_78b05_row345_col0" class="data row345 col0" >None</td>
          <td id="T_78b05_row345_col1" class="data row345 col1" >0.038900</td>
          <td id="T_78b05_row345_col2" class="data row345 col2" >-0.016000</td>
          <td id="T_78b05_row345_col3" class="data row345 col3" >0.018900</td>
          <td id="T_78b05_row345_col4" class="data row345 col4" >-0.033900</td>
          <td id="T_78b05_row345_col5" class="data row345 col5" >-0.011800</td>
          <td id="T_78b05_row345_col6" class="data row345 col6" >-0.049800</td>
          <td id="T_78b05_row345_col7" class="data row345 col7" >0.037500</td>
          <td id="T_78b05_row345_col8" class="data row345 col8" >0.006700</td>
          <td id="T_78b05_row345_col9" class="data row345 col9" >0.013900</td>
          <td id="T_78b05_row345_col10" class="data row345 col10" >0.011900</td>
          <td id="T_78b05_row345_col11" class="data row345 col11" >0.016000</td>
          <td id="T_78b05_row345_col12" class="data row345 col12" >0.011200</td>
          <td id="T_78b05_row345_col13" class="data row345 col13" >0.045900</td>
          <td id="T_78b05_row345_col14" class="data row345 col14" >0.039200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row346" class="row_heading level0 row346" >347</th>
          <td id="T_78b05_row346_col0" class="data row346 col0" >None</td>
          <td id="T_78b05_row346_col1" class="data row346 col1" >0.036400</td>
          <td id="T_78b05_row346_col2" class="data row346 col2" >0.046800</td>
          <td id="T_78b05_row346_col3" class="data row346 col3" >0.063200</td>
          <td id="T_78b05_row346_col4" class="data row346 col4" >0.006000</td>
          <td id="T_78b05_row346_col5" class="data row346 col5" >-0.013000</td>
          <td id="T_78b05_row346_col6" class="data row346 col6" >0.028900</td>
          <td id="T_78b05_row346_col7" class="data row346 col7" >0.042800</td>
          <td id="T_78b05_row346_col8" class="data row346 col8" >0.009100</td>
          <td id="T_78b05_row346_col9" class="data row346 col9" >0.076600</td>
          <td id="T_78b05_row346_col10" class="data row346 col10" >0.032500</td>
          <td id="T_78b05_row346_col11" class="data row346 col11" >0.024000</td>
          <td id="T_78b05_row346_col12" class="data row346 col12" >0.012400</td>
          <td id="T_78b05_row346_col13" class="data row346 col13" >0.032800</td>
          <td id="T_78b05_row346_col14" class="data row346 col14" >0.044600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row347" class="row_heading level0 row347" >348</th>
          <td id="T_78b05_row347_col0" class="data row347 col0" >None</td>
          <td id="T_78b05_row347_col1" class="data row347 col1" >0.037600</td>
          <td id="T_78b05_row347_col2" class="data row347 col2" >0.058800</td>
          <td id="T_78b05_row347_col3" class="data row347 col3" >0.015100</td>
          <td id="T_78b05_row347_col4" class="data row347 col4" >0.014800</td>
          <td id="T_78b05_row347_col5" class="data row347 col5" >0.021400</td>
          <td id="T_78b05_row347_col6" class="data row347 col6" >-0.052000</td>
          <td id="T_78b05_row347_col7" class="data row347 col7" >-0.012900</td>
          <td id="T_78b05_row347_col8" class="data row347 col8" >0.008000</td>
          <td id="T_78b05_row347_col9" class="data row347 col9" >0.088600</td>
          <td id="T_78b05_row347_col10" class="data row347 col10" >0.015600</td>
          <td id="T_78b05_row347_col11" class="data row347 col11" >0.032700</td>
          <td id="T_78b05_row347_col12" class="data row347 col12" >0.022000</td>
          <td id="T_78b05_row347_col13" class="data row347 col13" >0.048100</td>
          <td id="T_78b05_row347_col14" class="data row347 col14" >0.011100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row348" class="row_heading level0 row348" >349</th>
          <td id="T_78b05_row348_col0" class="data row348 col0" >None</td>
          <td id="T_78b05_row348_col1" class="data row348 col1" >0.034100</td>
          <td id="T_78b05_row348_col2" class="data row348 col2" >0.037300</td>
          <td id="T_78b05_row348_col3" class="data row348 col3" >-0.004200</td>
          <td id="T_78b05_row348_col4" class="data row348 col4" >0.010500</td>
          <td id="T_78b05_row348_col5" class="data row348 col5" >0.025100</td>
          <td id="T_78b05_row348_col6" class="data row348 col6" >-0.070700</td>
          <td id="T_78b05_row348_col7" class="data row348 col7" >0.019500</td>
          <td id="T_78b05_row348_col8" class="data row348 col8" >0.011500</td>
          <td id="T_78b05_row348_col9" class="data row348 col9" >0.067100</td>
          <td id="T_78b05_row348_col10" class="data row348 col10" >0.035000</td>
          <td id="T_78b05_row348_col11" class="data row348 col11" >0.028400</td>
          <td id="T_78b05_row348_col12" class="data row348 col12" >0.025800</td>
          <td id="T_78b05_row348_col13" class="data row348 col13" >0.066800</td>
          <td id="T_78b05_row348_col14" class="data row348 col14" >0.021200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row349" class="row_heading level0 row349" >350</th>
          <td id="T_78b05_row349_col0" class="data row349 col0" >None</td>
          <td id="T_78b05_row349_col1" class="data row349 col1" >0.037500</td>
          <td id="T_78b05_row349_col2" class="data row349 col2" >0.051300</td>
          <td id="T_78b05_row349_col3" class="data row349 col3" >0.049800</td>
          <td id="T_78b05_row349_col4" class="data row349 col4" >0.078900</td>
          <td id="T_78b05_row349_col5" class="data row349 col5" >0.021200</td>
          <td id="T_78b05_row349_col6" class="data row349 col6" >-0.075200</td>
          <td id="T_78b05_row349_col7" class="data row349 col7" >0.031500</td>
          <td id="T_78b05_row349_col8" class="data row349 col8" >0.008000</td>
          <td id="T_78b05_row349_col9" class="data row349 col9" >0.081100</td>
          <td id="T_78b05_row349_col10" class="data row349 col10" >0.019100</td>
          <td id="T_78b05_row349_col11" class="data row349 col11" >0.096800</td>
          <td id="T_78b05_row349_col12" class="data row349 col12" >0.021900</td>
          <td id="T_78b05_row349_col13" class="data row349 col13" >0.071300</td>
          <td id="T_78b05_row349_col14" class="data row349 col14" >0.033200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row350" class="row_heading level0 row350" >351</th>
          <td id="T_78b05_row350_col0" class="data row350 col0" >None</td>
          <td id="T_78b05_row350_col1" class="data row350 col1" >0.032200</td>
          <td id="T_78b05_row350_col2" class="data row350 col2" >0.030900</td>
          <td id="T_78b05_row350_col3" class="data row350 col3" >-0.023000</td>
          <td id="T_78b05_row350_col4" class="data row350 col4" >0.007400</td>
          <td id="T_78b05_row350_col5" class="data row350 col5" >-0.044700</td>
          <td id="T_78b05_row350_col6" class="data row350 col6" >-0.018400</td>
          <td id="T_78b05_row350_col7" class="data row350 col7" >0.027700</td>
          <td id="T_78b05_row350_col8" class="data row350 col8" >0.013300</td>
          <td id="T_78b05_row350_col9" class="data row350 col9" >0.060700</td>
          <td id="T_78b05_row350_col10" class="data row350 col10" >0.053700</td>
          <td id="T_78b05_row350_col11" class="data row350 col11" >0.025300</td>
          <td id="T_78b05_row350_col12" class="data row350 col12" >0.044000</td>
          <td id="T_78b05_row350_col13" class="data row350 col13" >0.014500</td>
          <td id="T_78b05_row350_col14" class="data row350 col14" >0.029500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row351" class="row_heading level0 row351" >352</th>
          <td id="T_78b05_row351_col0" class="data row351 col0" >None</td>
          <td id="T_78b05_row351_col1" class="data row351 col1" >0.041700</td>
          <td id="T_78b05_row351_col2" class="data row351 col2" >0.026000</td>
          <td id="T_78b05_row351_col3" class="data row351 col3" >0.041200</td>
          <td id="T_78b05_row351_col4" class="data row351 col4" >-0.033200</td>
          <td id="T_78b05_row351_col5" class="data row351 col5" >0.019500</td>
          <td id="T_78b05_row351_col6" class="data row351 col6" >-0.028100</td>
          <td id="T_78b05_row351_col7" class="data row351 col7" >0.040200</td>
          <td id="T_78b05_row351_col8" class="data row351 col8" >0.003800</td>
          <td id="T_78b05_row351_col9" class="data row351 col9" >0.055800</td>
          <td id="T_78b05_row351_col10" class="data row351 col10" >0.010500</td>
          <td id="T_78b05_row351_col11" class="data row351 col11" >0.015300</td>
          <td id="T_78b05_row351_col12" class="data row351 col12" >0.020100</td>
          <td id="T_78b05_row351_col13" class="data row351 col13" >0.024200</td>
          <td id="T_78b05_row351_col14" class="data row351 col14" >0.041900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row352" class="row_heading level0 row352" >353</th>
          <td id="T_78b05_row352_col0" class="data row352 col0" >None</td>
          <td id="T_78b05_row352_col1" class="data row352 col1" >0.040400</td>
          <td id="T_78b05_row352_col2" class="data row352 col2" >0.022600</td>
          <td id="T_78b05_row352_col3" class="data row352 col3" >0.059300</td>
          <td id="T_78b05_row352_col4" class="data row352 col4" >-0.051000</td>
          <td id="T_78b05_row352_col5" class="data row352 col5" >0.004900</td>
          <td id="T_78b05_row352_col6" class="data row352 col6" >-0.000100</td>
          <td id="T_78b05_row352_col7" class="data row352 col7" >-0.022500</td>
          <td id="T_78b05_row352_col8" class="data row352 col8" >0.005200</td>
          <td id="T_78b05_row352_col9" class="data row352 col9" >0.052500</td>
          <td id="T_78b05_row352_col10" class="data row352 col10" >0.028600</td>
          <td id="T_78b05_row352_col11" class="data row352 col11" >0.033100</td>
          <td id="T_78b05_row352_col12" class="data row352 col12" >0.005500</td>
          <td id="T_78b05_row352_col13" class="data row352 col13" >0.003800</td>
          <td id="T_78b05_row352_col14" class="data row352 col14" >0.020800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row353" class="row_heading level0 row353" >354</th>
          <td id="T_78b05_row353_col0" class="data row353 col0" >None</td>
          <td id="T_78b05_row353_col1" class="data row353 col1" >0.035900</td>
          <td id="T_78b05_row353_col2" class="data row353 col2" >0.043100</td>
          <td id="T_78b05_row353_col3" class="data row353 col3" >0.029200</td>
          <td id="T_78b05_row353_col4" class="data row353 col4" >0.050800</td>
          <td id="T_78b05_row353_col5" class="data row353 col5" >-0.005400</td>
          <td id="T_78b05_row353_col6" class="data row353 col6" >0.053900</td>
          <td id="T_78b05_row353_col7" class="data row353 col7" >0.016100</td>
          <td id="T_78b05_row353_col8" class="data row353 col8" >0.009700</td>
          <td id="T_78b05_row353_col9" class="data row353 col9" >0.072900</td>
          <td id="T_78b05_row353_col10" class="data row353 col10" >0.001500</td>
          <td id="T_78b05_row353_col11" class="data row353 col11" >0.068700</td>
          <td id="T_78b05_row353_col12" class="data row353 col12" >0.004800</td>
          <td id="T_78b05_row353_col13" class="data row353 col13" >0.057800</td>
          <td id="T_78b05_row353_col14" class="data row353 col14" >0.017900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row354" class="row_heading level0 row354" >355</th>
          <td id="T_78b05_row354_col0" class="data row354 col0" >None</td>
          <td id="T_78b05_row354_col1" class="data row354 col1" >0.038600</td>
          <td id="T_78b05_row354_col2" class="data row354 col2" >-0.065700</td>
          <td id="T_78b05_row354_col3" class="data row354 col3" >-0.049000</td>
          <td id="T_78b05_row354_col4" class="data row354 col4" >-0.018300</td>
          <td id="T_78b05_row354_col5" class="data row354 col5" >0.054200</td>
          <td id="T_78b05_row354_col6" class="data row354 col6" >0.045200</td>
          <td id="T_78b05_row354_col7" class="data row354 col7" >0.010400</td>
          <td id="T_78b05_row354_col8" class="data row354 col8" >0.006900</td>
          <td id="T_78b05_row354_col9" class="data row354 col9" >0.035900</td>
          <td id="T_78b05_row354_col10" class="data row354 col10" >0.079800</td>
          <td id="T_78b05_row354_col11" class="data row354 col11" >0.000400</td>
          <td id="T_78b05_row354_col12" class="data row354 col12" >0.054900</td>
          <td id="T_78b05_row354_col13" class="data row354 col13" >0.049100</td>
          <td id="T_78b05_row354_col14" class="data row354 col14" >0.012100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row355" class="row_heading level0 row355" >356</th>
          <td id="T_78b05_row355_col0" class="data row355 col0" >None</td>
          <td id="T_78b05_row355_col1" class="data row355 col1" >0.045000</td>
          <td id="T_78b05_row355_col2" class="data row355 col2" >0.039100</td>
          <td id="T_78b05_row355_col3" class="data row355 col3" >0.021600</td>
          <td id="T_78b05_row355_col4" class="data row355 col4" >0.034400</td>
          <td id="T_78b05_row355_col5" class="data row355 col5" >0.095800</td>
          <td id="T_78b05_row355_col6" class="data row355 col6" >0.088200</td>
          <td id="T_78b05_row355_col7" class="data row355 col7" >0.018900</td>
          <td id="T_78b05_row355_col8" class="data row355 col8" >0.000600</td>
          <td id="T_78b05_row355_col9" class="data row355 col9" >0.068900</td>
          <td id="T_78b05_row355_col10" class="data row355 col10" >0.009100</td>
          <td id="T_78b05_row355_col11" class="data row355 col11" >0.052300</td>
          <td id="T_78b05_row355_col12" class="data row355 col12" >0.096400</td>
          <td id="T_78b05_row355_col13" class="data row355 col13" >0.092100</td>
          <td id="T_78b05_row355_col14" class="data row355 col14" >0.020600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row356" class="row_heading level0 row356" >357</th>
          <td id="T_78b05_row356_col0" class="data row356 col0" >None</td>
          <td id="T_78b05_row356_col1" class="data row356 col1" >0.039200</td>
          <td id="T_78b05_row356_col2" class="data row356 col2" >0.029600</td>
          <td id="T_78b05_row356_col3" class="data row356 col3" >0.086700</td>
          <td id="T_78b05_row356_col4" class="data row356 col4" >0.027400</td>
          <td id="T_78b05_row356_col5" class="data row356 col5" >-0.022500</td>
          <td id="T_78b05_row356_col6" class="data row356 col6" >-0.020600</td>
          <td id="T_78b05_row356_col7" class="data row356 col7" >-0.052800</td>
          <td id="T_78b05_row356_col8" class="data row356 col8" >0.006300</td>
          <td id="T_78b05_row356_col9" class="data row356 col9" >0.059400</td>
          <td id="T_78b05_row356_col10" class="data row356 col10" >0.055900</td>
          <td id="T_78b05_row356_col11" class="data row356 col11" >0.045300</td>
          <td id="T_78b05_row356_col12" class="data row356 col12" >0.021800</td>
          <td id="T_78b05_row356_col13" class="data row356 col13" >0.016700</td>
          <td id="T_78b05_row356_col14" class="data row356 col14" >0.051100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row357" class="row_heading level0 row357" >358</th>
          <td id="T_78b05_row357_col0" class="data row357 col0" >None</td>
          <td id="T_78b05_row357_col1" class="data row357 col1" >0.039700</td>
          <td id="T_78b05_row357_col2" class="data row357 col2" >-0.043200</td>
          <td id="T_78b05_row357_col3" class="data row357 col3" >-0.057000</td>
          <td id="T_78b05_row357_col4" class="data row357 col4" >0.000400</td>
          <td id="T_78b05_row357_col5" class="data row357 col5" >0.010800</td>
          <td id="T_78b05_row357_col6" class="data row357 col6" >0.021400</td>
          <td id="T_78b05_row357_col7" class="data row357 col7" >0.009800</td>
          <td id="T_78b05_row357_col8" class="data row357 col8" >0.005800</td>
          <td id="T_78b05_row357_col9" class="data row357 col9" >0.013400</td>
          <td id="T_78b05_row357_col10" class="data row357 col10" >0.087700</td>
          <td id="T_78b05_row357_col11" class="data row357 col11" >0.018300</td>
          <td id="T_78b05_row357_col12" class="data row357 col12" >0.011400</td>
          <td id="T_78b05_row357_col13" class="data row357 col13" >0.025300</td>
          <td id="T_78b05_row357_col14" class="data row357 col14" >0.011600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row358" class="row_heading level0 row358" >359</th>
          <td id="T_78b05_row358_col0" class="data row358 col0" >None</td>
          <td id="T_78b05_row358_col1" class="data row358 col1" >0.039100</td>
          <td id="T_78b05_row358_col2" class="data row358 col2" >-0.040800</td>
          <td id="T_78b05_row358_col3" class="data row358 col3" >0.029100</td>
          <td id="T_78b05_row358_col4" class="data row358 col4" >-0.015000</td>
          <td id="T_78b05_row358_col5" class="data row358 col5" >-0.009000</td>
          <td id="T_78b05_row358_col6" class="data row358 col6" >-0.013400</td>
          <td id="T_78b05_row358_col7" class="data row358 col7" >-0.045800</td>
          <td id="T_78b05_row358_col8" class="data row358 col8" >0.006400</td>
          <td id="T_78b05_row358_col9" class="data row358 col9" >0.011000</td>
          <td id="T_78b05_row358_col10" class="data row358 col10" >0.001700</td>
          <td id="T_78b05_row358_col11" class="data row358 col11" >0.002900</td>
          <td id="T_78b05_row358_col12" class="data row358 col12" >0.008400</td>
          <td id="T_78b05_row358_col13" class="data row358 col13" >0.009500</td>
          <td id="T_78b05_row358_col14" class="data row358 col14" >0.044100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row359" class="row_heading level0 row359" >360</th>
          <td id="T_78b05_row359_col0" class="data row359 col0" >None</td>
          <td id="T_78b05_row359_col1" class="data row359 col1" >0.042400</td>
          <td id="T_78b05_row359_col2" class="data row359 col2" >-0.002800</td>
          <td id="T_78b05_row359_col3" class="data row359 col3" >0.029100</td>
          <td id="T_78b05_row359_col4" class="data row359 col4" >0.029900</td>
          <td id="T_78b05_row359_col5" class="data row359 col5" >-0.038500</td>
          <td id="T_78b05_row359_col6" class="data row359 col6" >-0.011500</td>
          <td id="T_78b05_row359_col7" class="data row359 col7" >0.054900</td>
          <td id="T_78b05_row359_col8" class="data row359 col8" >0.003100</td>
          <td id="T_78b05_row359_col9" class="data row359 col9" >0.027000</td>
          <td id="T_78b05_row359_col10" class="data row359 col10" >0.001600</td>
          <td id="T_78b05_row359_col11" class="data row359 col11" >0.047800</td>
          <td id="T_78b05_row359_col12" class="data row359 col12" >0.037800</td>
          <td id="T_78b05_row359_col13" class="data row359 col13" >0.007600</td>
          <td id="T_78b05_row359_col14" class="data row359 col14" >0.056700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row360" class="row_heading level0 row360" >361</th>
          <td id="T_78b05_row360_col0" class="data row360 col0" >None</td>
          <td id="T_78b05_row360_col1" class="data row360 col1" >0.042500</td>
          <td id="T_78b05_row360_col2" class="data row360 col2" >-0.034600</td>
          <td id="T_78b05_row360_col3" class="data row360 col3" >-0.012000</td>
          <td id="T_78b05_row360_col4" class="data row360 col4" >0.029800</td>
          <td id="T_78b05_row360_col5" class="data row360 col5" >0.046600</td>
          <td id="T_78b05_row360_col6" class="data row360 col6" >-0.027400</td>
          <td id="T_78b05_row360_col7" class="data row360 col7" >0.056700</td>
          <td id="T_78b05_row360_col8" class="data row360 col8" >0.003000</td>
          <td id="T_78b05_row360_col9" class="data row360 col9" >0.004800</td>
          <td id="T_78b05_row360_col10" class="data row360 col10" >0.042800</td>
          <td id="T_78b05_row360_col11" class="data row360 col11" >0.047700</td>
          <td id="T_78b05_row360_col12" class="data row360 col12" >0.047200</td>
          <td id="T_78b05_row360_col13" class="data row360 col13" >0.023500</td>
          <td id="T_78b05_row360_col14" class="data row360 col14" >0.058400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row361" class="row_heading level0 row361" >362</th>
          <td id="T_78b05_row361_col0" class="data row361 col0" >None</td>
          <td id="T_78b05_row361_col1" class="data row361 col1" >0.029500</td>
          <td id="T_78b05_row361_col2" class="data row361 col2" >0.062600</td>
          <td id="T_78b05_row361_col3" class="data row361 col3" >-0.024400</td>
          <td id="T_78b05_row361_col4" class="data row361 col4" >-0.034900</td>
          <td id="T_78b05_row361_col5" class="data row361 col5" >-0.015100</td>
          <td id="T_78b05_row361_col6" class="data row361 col6" >-0.012600</td>
          <td id="T_78b05_row361_col7" class="data row361 col7" >-0.078900</td>
          <td id="T_78b05_row361_col8" class="data row361 col8" >0.016000</td>
          <td id="T_78b05_row361_col9" class="data row361 col9" >0.092500</td>
          <td id="T_78b05_row361_col10" class="data row361 col10" >0.055100</td>
          <td id="T_78b05_row361_col11" class="data row361 col11" >0.017000</td>
          <td id="T_78b05_row361_col12" class="data row361 col12" >0.014500</td>
          <td id="T_78b05_row361_col13" class="data row361 col13" >0.008700</td>
          <td id="T_78b05_row361_col14" class="data row361 col14" >0.077200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row362" class="row_heading level0 row362" >363</th>
          <td id="T_78b05_row362_col0" class="data row362 col0" >None</td>
          <td id="T_78b05_row362_col1" class="data row362 col1" >0.027400</td>
          <td id="T_78b05_row362_col2" class="data row362 col2" >0.033200</td>
          <td id="T_78b05_row362_col3" class="data row362 col3" >-0.047900</td>
          <td id="T_78b05_row362_col4" class="data row362 col4" >-0.036200</td>
          <td id="T_78b05_row362_col5" class="data row362 col5" >0.020800</td>
          <td id="T_78b05_row362_col6" class="data row362 col6" >-0.034900</td>
          <td id="T_78b05_row362_col7" class="data row362 col7" >0.071300</td>
          <td id="T_78b05_row362_col8" class="data row362 col8" >0.018100</td>
          <td id="T_78b05_row362_col9" class="data row362 col9" >0.063000</td>
          <td id="T_78b05_row362_col10" class="data row362 col10" >0.078700</td>
          <td id="T_78b05_row362_col11" class="data row362 col11" >0.018300</td>
          <td id="T_78b05_row362_col12" class="data row362 col12" >0.021400</td>
          <td id="T_78b05_row362_col13" class="data row362 col13" >0.031000</td>
          <td id="T_78b05_row362_col14" class="data row362 col14" >0.073100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row363" class="row_heading level0 row363" >364</th>
          <td id="T_78b05_row363_col0" class="data row363 col0" >None</td>
          <td id="T_78b05_row363_col1" class="data row363 col1" >0.042500</td>
          <td id="T_78b05_row363_col2" class="data row363 col2" >-0.011500</td>
          <td id="T_78b05_row363_col3" class="data row363 col3" >0.024300</td>
          <td id="T_78b05_row363_col4" class="data row363 col4" >-0.021900</td>
          <td id="T_78b05_row363_col5" class="data row363 col5" >0.046800</td>
          <td id="T_78b05_row363_col6" class="data row363 col6" >0.025100</td>
          <td id="T_78b05_row363_col7" class="data row363 col7" >-0.022600</td>
          <td id="T_78b05_row363_col8" class="data row363 col8" >0.003100</td>
          <td id="T_78b05_row363_col9" class="data row363 col9" >0.018400</td>
          <td id="T_78b05_row363_col10" class="data row363 col10" >0.006400</td>
          <td id="T_78b05_row363_col11" class="data row363 col11" >0.004000</td>
          <td id="T_78b05_row363_col12" class="data row363 col12" >0.047400</td>
          <td id="T_78b05_row363_col13" class="data row363 col13" >0.029000</td>
          <td id="T_78b05_row363_col14" class="data row363 col14" >0.020900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row364" class="row_heading level0 row364" >365</th>
          <td id="T_78b05_row364_col0" class="data row364 col0" >None</td>
          <td id="T_78b05_row364_col1" class="data row364 col1" >0.042300</td>
          <td id="T_78b05_row364_col2" class="data row364 col2" >-0.003800</td>
          <td id="T_78b05_row364_col3" class="data row364 col3" >0.017500</td>
          <td id="T_78b05_row364_col4" class="data row364 col4" >-0.011300</td>
          <td id="T_78b05_row364_col5" class="data row364 col5" >0.060900</td>
          <td id="T_78b05_row364_col6" class="data row364 col6" >0.062300</td>
          <td id="T_78b05_row364_col7" class="data row364 col7" >-0.015200</td>
          <td id="T_78b05_row364_col8" class="data row364 col8" >0.003200</td>
          <td id="T_78b05_row364_col9" class="data row364 col9" >0.026000</td>
          <td id="T_78b05_row364_col10" class="data row364 col10" >0.013300</td>
          <td id="T_78b05_row364_col11" class="data row364 col11" >0.006700</td>
          <td id="T_78b05_row364_col12" class="data row364 col12" >0.061500</td>
          <td id="T_78b05_row364_col13" class="data row364 col13" >0.066200</td>
          <td id="T_78b05_row364_col14" class="data row364 col14" >0.013400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row365" class="row_heading level0 row365" >366</th>
          <td id="T_78b05_row365_col0" class="data row365 col0" >None</td>
          <td id="T_78b05_row365_col1" class="data row365 col1" >0.033100</td>
          <td id="T_78b05_row365_col2" class="data row365 col2" >-0.003500</td>
          <td id="T_78b05_row365_col3" class="data row365 col3" >-0.051000</td>
          <td id="T_78b05_row365_col4" class="data row365 col4" >0.009700</td>
          <td id="T_78b05_row365_col5" class="data row365 col5" >-0.039400</td>
          <td id="T_78b05_row365_col6" class="data row365 col6" >-0.025500</td>
          <td id="T_78b05_row365_col7" class="data row365 col7" >-0.015000</td>
          <td id="T_78b05_row365_col8" class="data row365 col8" >0.012500</td>
          <td id="T_78b05_row365_col9" class="data row365 col9" >0.026400</td>
          <td id="T_78b05_row365_col10" class="data row365 col10" >0.081800</td>
          <td id="T_78b05_row365_col11" class="data row365 col11" >0.027600</td>
          <td id="T_78b05_row365_col12" class="data row365 col12" >0.038700</td>
          <td id="T_78b05_row365_col13" class="data row365 col13" >0.021600</td>
          <td id="T_78b05_row365_col14" class="data row365 col14" >0.013300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row366" class="row_heading level0 row366" >367</th>
          <td id="T_78b05_row366_col0" class="data row366 col0" >None</td>
          <td id="T_78b05_row366_col1" class="data row366 col1" >0.031800</td>
          <td id="T_78b05_row366_col2" class="data row366 col2" >-0.005700</td>
          <td id="T_78b05_row366_col3" class="data row366 col3" >-0.061100</td>
          <td id="T_78b05_row366_col4" class="data row366 col4" >-0.011000</td>
          <td id="T_78b05_row366_col5" class="data row366 col5" >0.029100</td>
          <td id="T_78b05_row366_col6" class="data row366 col6" >0.000500</td>
          <td id="T_78b05_row366_col7" class="data row366 col7" >-0.033700</td>
          <td id="T_78b05_row366_col8" class="data row366 col8" >0.013800</td>
          <td id="T_78b05_row366_col9" class="data row366 col9" >0.024100</td>
          <td id="T_78b05_row366_col10" class="data row366 col10" >0.091800</td>
          <td id="T_78b05_row366_col11" class="data row366 col11" >0.007000</td>
          <td id="T_78b05_row366_col12" class="data row366 col12" >0.029700</td>
          <td id="T_78b05_row366_col13" class="data row366 col13" >0.004400</td>
          <td id="T_78b05_row366_col14" class="data row366 col14" >0.031900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row367" class="row_heading level0 row367" >368</th>
          <td id="T_78b05_row367_col0" class="data row367 col0" >None</td>
          <td id="T_78b05_row367_col1" class="data row367 col1" >0.042100</td>
          <td id="T_78b05_row367_col2" class="data row367 col2" >-0.025000</td>
          <td id="T_78b05_row367_col3" class="data row367 col3" >0.017900</td>
          <td id="T_78b05_row367_col4" class="data row367 col4" >0.028700</td>
          <td id="T_78b05_row367_col5" class="data row367 col5" >-0.058800</td>
          <td id="T_78b05_row367_col6" class="data row367 col6" >-0.017900</td>
          <td id="T_78b05_row367_col7" class="data row367 col7" >0.028600</td>
          <td id="T_78b05_row367_col8" class="data row367 col8" >0.003400</td>
          <td id="T_78b05_row367_col9" class="data row367 col9" >0.004800</td>
          <td id="T_78b05_row367_col10" class="data row367 col10" >0.012900</td>
          <td id="T_78b05_row367_col11" class="data row367 col11" >0.046600</td>
          <td id="T_78b05_row367_col12" class="data row367 col12" >0.058200</td>
          <td id="T_78b05_row367_col13" class="data row367 col13" >0.014000</td>
          <td id="T_78b05_row367_col14" class="data row367 col14" >0.030300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row368" class="row_heading level0 row368" >369</th>
          <td id="T_78b05_row368_col0" class="data row368 col0" >None</td>
          <td id="T_78b05_row368_col1" class="data row368 col1" >0.031700</td>
          <td id="T_78b05_row368_col2" class="data row368 col2" >0.005700</td>
          <td id="T_78b05_row368_col3" class="data row368 col3" >-0.031000</td>
          <td id="T_78b05_row368_col4" class="data row368 col4" >0.038700</td>
          <td id="T_78b05_row368_col5" class="data row368 col5" >-0.078000</td>
          <td id="T_78b05_row368_col6" class="data row368 col6" >0.035200</td>
          <td id="T_78b05_row368_col7" class="data row368 col7" >0.043400</td>
          <td id="T_78b05_row368_col8" class="data row368 col8" >0.013800</td>
          <td id="T_78b05_row368_col9" class="data row368 col9" >0.035500</td>
          <td id="T_78b05_row368_col10" class="data row368 col10" >0.061800</td>
          <td id="T_78b05_row368_col11" class="data row368 col11" >0.056600</td>
          <td id="T_78b05_row368_col12" class="data row368 col12" >0.077400</td>
          <td id="T_78b05_row368_col13" class="data row368 col13" >0.039100</td>
          <td id="T_78b05_row368_col14" class="data row368 col14" >0.045100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row369" class="row_heading level0 row369" >370</th>
          <td id="T_78b05_row369_col0" class="data row369 col0" >None</td>
          <td id="T_78b05_row369_col1" class="data row369 col1" >0.038600</td>
          <td id="T_78b05_row369_col2" class="data row369 col2" >0.048100</td>
          <td id="T_78b05_row369_col3" class="data row369 col3" >0.052100</td>
          <td id="T_78b05_row369_col4" class="data row369 col4" >-0.037700</td>
          <td id="T_78b05_row369_col5" class="data row369 col5" >-0.063200</td>
          <td id="T_78b05_row369_col6" class="data row369 col6" >-0.069400</td>
          <td id="T_78b05_row369_col7" class="data row369 col7" >-0.008500</td>
          <td id="T_78b05_row369_col8" class="data row369 col8" >0.006900</td>
          <td id="T_78b05_row369_col9" class="data row369 col9" >0.077900</td>
          <td id="T_78b05_row369_col10" class="data row369 col10" >0.021300</td>
          <td id="T_78b05_row369_col11" class="data row369 col11" >0.019800</td>
          <td id="T_78b05_row369_col12" class="data row369 col12" >0.062500</td>
          <td id="T_78b05_row369_col13" class="data row369 col13" >0.065500</td>
          <td id="T_78b05_row369_col14" class="data row369 col14" >0.006800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row370" class="row_heading level0 row370" >371</th>
          <td id="T_78b05_row370_col0" class="data row370 col0" >None</td>
          <td id="T_78b05_row370_col1" class="data row370 col1" >0.036600</td>
          <td id="T_78b05_row370_col2" class="data row370 col2" >0.003800</td>
          <td id="T_78b05_row370_col3" class="data row370 col3" >0.003900</td>
          <td id="T_78b05_row370_col4" class="data row370 col4" >0.031700</td>
          <td id="T_78b05_row370_col5" class="data row370 col5" >-0.035700</td>
          <td id="T_78b05_row370_col6" class="data row370 col6" >-0.100900</td>
          <td id="T_78b05_row370_col7" class="data row370 col7" >0.015300</td>
          <td id="T_78b05_row370_col8" class="data row370 col8" >0.008900</td>
          <td id="T_78b05_row370_col9" class="data row370 col9" >0.033600</td>
          <td id="T_78b05_row370_col10" class="data row370 col10" >0.026800</td>
          <td id="T_78b05_row370_col11" class="data row370 col11" >0.049700</td>
          <td id="T_78b05_row370_col12" class="data row370 col12" >0.035000</td>
          <td id="T_78b05_row370_col13" class="data row370 col13" >0.097000</td>
          <td id="T_78b05_row370_col14" class="data row370 col14" >0.017100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row371" class="row_heading level0 row371" >372</th>
          <td id="T_78b05_row371_col0" class="data row371 col0" >None</td>
          <td id="T_78b05_row371_col1" class="data row371 col1" >0.037600</td>
          <td id="T_78b05_row371_col2" class="data row371 col2" >0.001600</td>
          <td id="T_78b05_row371_col3" class="data row371 col3" >0.041400</td>
          <td id="T_78b05_row371_col4" class="data row371 col4" >0.021500</td>
          <td id="T_78b05_row371_col5" class="data row371 col5" >-0.007300</td>
          <td id="T_78b05_row371_col6" class="data row371 col6" >-0.080000</td>
          <td id="T_78b05_row371_col7" class="data row371 col7" >-0.027500</td>
          <td id="T_78b05_row371_col8" class="data row371 col8" >0.007900</td>
          <td id="T_78b05_row371_col9" class="data row371 col9" >0.031500</td>
          <td id="T_78b05_row371_col10" class="data row371 col10" >0.010600</td>
          <td id="T_78b05_row371_col11" class="data row371 col11" >0.039400</td>
          <td id="T_78b05_row371_col12" class="data row371 col12" >0.006700</td>
          <td id="T_78b05_row371_col13" class="data row371 col13" >0.076100</td>
          <td id="T_78b05_row371_col14" class="data row371 col14" >0.025700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row372" class="row_heading level0 row372" >373</th>
          <td id="T_78b05_row372_col0" class="data row372 col0" >None</td>
          <td id="T_78b05_row372_col1" class="data row372 col1" >0.031100</td>
          <td id="T_78b05_row372_col2" class="data row372 col2" >0.050700</td>
          <td id="T_78b05_row372_col3" class="data row372 col3" >-0.006400</td>
          <td id="T_78b05_row372_col4" class="data row372 col4" >-0.005000</td>
          <td id="T_78b05_row372_col5" class="data row372 col5" >0.022100</td>
          <td id="T_78b05_row372_col6" class="data row372 col6" >0.022800</td>
          <td id="T_78b05_row372_col7" class="data row372 col7" >-0.043400</td>
          <td id="T_78b05_row372_col8" class="data row372 col8" >0.014400</td>
          <td id="T_78b05_row372_col9" class="data row372 col9" >0.080600</td>
          <td id="T_78b05_row372_col10" class="data row372 col10" >0.037200</td>
          <td id="T_78b05_row372_col11" class="data row372 col11" >0.012900</td>
          <td id="T_78b05_row372_col12" class="data row372 col12" >0.022800</td>
          <td id="T_78b05_row372_col13" class="data row372 col13" >0.026700</td>
          <td id="T_78b05_row372_col14" class="data row372 col14" >0.041700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row373" class="row_heading level0 row373" >374</th>
          <td id="T_78b05_row373_col0" class="data row373 col0" >None</td>
          <td id="T_78b05_row373_col1" class="data row373 col1" >0.043200</td>
          <td id="T_78b05_row373_col2" class="data row373 col2" >0.054100</td>
          <td id="T_78b05_row373_col3" class="data row373 col3" >0.040400</td>
          <td id="T_78b05_row373_col4" class="data row373 col4" >-0.022400</td>
          <td id="T_78b05_row373_col5" class="data row373 col5" >-0.015100</td>
          <td id="T_78b05_row373_col6" class="data row373 col6" >0.030500</td>
          <td id="T_78b05_row373_col7" class="data row373 col7" >-0.017100</td>
          <td id="T_78b05_row373_col8" class="data row373 col8" >0.002300</td>
          <td id="T_78b05_row373_col9" class="data row373 col9" >0.083900</td>
          <td id="T_78b05_row373_col10" class="data row373 col10" >0.009700</td>
          <td id="T_78b05_row373_col11" class="data row373 col11" >0.004500</td>
          <td id="T_78b05_row373_col12" class="data row373 col12" >0.014400</td>
          <td id="T_78b05_row373_col13" class="data row373 col13" >0.034400</td>
          <td id="T_78b05_row373_col14" class="data row373 col14" >0.015400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row374" class="row_heading level0 row374" >375</th>
          <td id="T_78b05_row374_col0" class="data row374 col0" >None</td>
          <td id="T_78b05_row374_col1" class="data row374 col1" >0.033400</td>
          <td id="T_78b05_row374_col2" class="data row374 col2" >-0.032500</td>
          <td id="T_78b05_row374_col3" class="data row374 col3" >-0.056700</td>
          <td id="T_78b05_row374_col4" class="data row374 col4" >0.038600</td>
          <td id="T_78b05_row374_col5" class="data row374 col5" >-0.046500</td>
          <td id="T_78b05_row374_col6" class="data row374 col6" >-0.005300</td>
          <td id="T_78b05_row374_col7" class="data row374 col7" >0.012400</td>
          <td id="T_78b05_row374_col8" class="data row374 col8" >0.012200</td>
          <td id="T_78b05_row374_col9" class="data row374 col9" >0.002700</td>
          <td id="T_78b05_row374_col10" class="data row374 col10" >0.087400</td>
          <td id="T_78b05_row374_col11" class="data row374 col11" >0.056500</td>
          <td id="T_78b05_row374_col12" class="data row374 col12" >0.045800</td>
          <td id="T_78b05_row374_col13" class="data row374 col13" >0.001400</td>
          <td id="T_78b05_row374_col14" class="data row374 col14" >0.014200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row375" class="row_heading level0 row375" >376</th>
          <td id="T_78b05_row375_col0" class="data row375 col0" >None</td>
          <td id="T_78b05_row375_col1" class="data row375 col1" >0.038900</td>
          <td id="T_78b05_row375_col2" class="data row375 col2" >0.026900</td>
          <td id="T_78b05_row375_col3" class="data row375 col3" >0.010100</td>
          <td id="T_78b05_row375_col4" class="data row375 col4" >-0.035100</td>
          <td id="T_78b05_row375_col5" class="data row375 col5" >-0.005300</td>
          <td id="T_78b05_row375_col6" class="data row375 col6" >0.004600</td>
          <td id="T_78b05_row375_col7" class="data row375 col7" >0.077200</td>
          <td id="T_78b05_row375_col8" class="data row375 col8" >0.006600</td>
          <td id="T_78b05_row375_col9" class="data row375 col9" >0.056700</td>
          <td id="T_78b05_row375_col10" class="data row375 col10" >0.020700</td>
          <td id="T_78b05_row375_col11" class="data row375 col11" >0.017200</td>
          <td id="T_78b05_row375_col12" class="data row375 col12" >0.004600</td>
          <td id="T_78b05_row375_col13" class="data row375 col13" >0.008500</td>
          <td id="T_78b05_row375_col14" class="data row375 col14" >0.078900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row376" class="row_heading level0 row376" >377</th>
          <td id="T_78b05_row376_col0" class="data row376 col0" >None</td>
          <td id="T_78b05_row376_col1" class="data row376 col1" >0.037900</td>
          <td id="T_78b05_row376_col2" class="data row376 col2" >-0.024000</td>
          <td id="T_78b05_row376_col3" class="data row376 col3" >-0.002300</td>
          <td id="T_78b05_row376_col4" class="data row376 col4" >-0.031300</td>
          <td id="T_78b05_row376_col5" class="data row376 col5" >-0.018400</td>
          <td id="T_78b05_row376_col6" class="data row376 col6" >-0.024900</td>
          <td id="T_78b05_row376_col7" class="data row376 col7" >-0.028900</td>
          <td id="T_78b05_row376_col8" class="data row376 col8" >0.007600</td>
          <td id="T_78b05_row376_col9" class="data row376 col9" >0.005800</td>
          <td id="T_78b05_row376_col10" class="data row376 col10" >0.033000</td>
          <td id="T_78b05_row376_col11" class="data row376 col11" >0.013300</td>
          <td id="T_78b05_row376_col12" class="data row376 col12" >0.017800</td>
          <td id="T_78b05_row376_col13" class="data row376 col13" >0.021000</td>
          <td id="T_78b05_row376_col14" class="data row376 col14" >0.027200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row377" class="row_heading level0 row377" >378</th>
          <td id="T_78b05_row377_col0" class="data row377 col0" >None</td>
          <td id="T_78b05_row377_col1" class="data row377 col1" >0.039700</td>
          <td id="T_78b05_row377_col2" class="data row377 col2" >-0.002600</td>
          <td id="T_78b05_row377_col3" class="data row377 col3" >0.000400</td>
          <td id="T_78b05_row377_col4" class="data row377 col4" >-0.052500</td>
          <td id="T_78b05_row377_col5" class="data row377 col5" >0.012000</td>
          <td id="T_78b05_row377_col6" class="data row377 col6" >-0.046100</td>
          <td id="T_78b05_row377_col7" class="data row377 col7" >-0.008700</td>
          <td id="T_78b05_row377_col8" class="data row377 col8" >0.005800</td>
          <td id="T_78b05_row377_col9" class="data row377 col9" >0.027200</td>
          <td id="T_78b05_row377_col10" class="data row377 col10" >0.030300</td>
          <td id="T_78b05_row377_col11" class="data row377 col11" >0.034600</td>
          <td id="T_78b05_row377_col12" class="data row377 col12" >0.012600</td>
          <td id="T_78b05_row377_col13" class="data row377 col13" >0.042200</td>
          <td id="T_78b05_row377_col14" class="data row377 col14" >0.007000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row378" class="row_heading level0 row378" >379</th>
          <td id="T_78b05_row378_col0" class="data row378 col0" >None</td>
          <td id="T_78b05_row378_col1" class="data row378 col1" >0.037000</td>
          <td id="T_78b05_row378_col2" class="data row378 col2" >-0.043700</td>
          <td id="T_78b05_row378_col3" class="data row378 col3" >0.014400</td>
          <td id="T_78b05_row378_col4" class="data row378 col4" >0.030000</td>
          <td id="T_78b05_row378_col5" class="data row378 col5" >-0.061600</td>
          <td id="T_78b05_row378_col6" class="data row378 col6" >-0.000700</td>
          <td id="T_78b05_row378_col7" class="data row378 col7" >-0.041700</td>
          <td id="T_78b05_row378_col8" class="data row378 col8" >0.008500</td>
          <td id="T_78b05_row378_col9" class="data row378 col9" >0.013900</td>
          <td id="T_78b05_row378_col10" class="data row378 col10" >0.016400</td>
          <td id="T_78b05_row378_col11" class="data row378 col11" >0.047900</td>
          <td id="T_78b05_row378_col12" class="data row378 col12" >0.061000</td>
          <td id="T_78b05_row378_col13" class="data row378 col13" >0.003200</td>
          <td id="T_78b05_row378_col14" class="data row378 col14" >0.040000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row379" class="row_heading level0 row379" >380</th>
          <td id="T_78b05_row379_col0" class="data row379 col0" >None</td>
          <td id="T_78b05_row379_col1" class="data row379 col1" >0.037200</td>
          <td id="T_78b05_row379_col2" class="data row379 col2" >0.016800</td>
          <td id="T_78b05_row379_col3" class="data row379 col3" >-0.031200</td>
          <td id="T_78b05_row379_col4" class="data row379 col4" >0.004800</td>
          <td id="T_78b05_row379_col5" class="data row379 col5" >-0.005200</td>
          <td id="T_78b05_row379_col6" class="data row379 col6" >-0.009900</td>
          <td id="T_78b05_row379_col7" class="data row379 col7" >0.073600</td>
          <td id="T_78b05_row379_col8" class="data row379 col8" >0.008300</td>
          <td id="T_78b05_row379_col9" class="data row379 col9" >0.046600</td>
          <td id="T_78b05_row379_col10" class="data row379 col10" >0.061900</td>
          <td id="T_78b05_row379_col11" class="data row379 col11" >0.022700</td>
          <td id="T_78b05_row379_col12" class="data row379 col12" >0.004600</td>
          <td id="T_78b05_row379_col13" class="data row379 col13" >0.006000</td>
          <td id="T_78b05_row379_col14" class="data row379 col14" >0.075400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row380" class="row_heading level0 row380" >381</th>
          <td id="T_78b05_row380_col0" class="data row380 col0" >None</td>
          <td id="T_78b05_row380_col1" class="data row380 col1" >0.031500</td>
          <td id="T_78b05_row380_col2" class="data row380 col2" >-0.005100</td>
          <td id="T_78b05_row380_col3" class="data row380 col3" >-0.018300</td>
          <td id="T_78b05_row380_col4" class="data row380 col4" >-0.036700</td>
          <td id="T_78b05_row380_col5" class="data row380 col5" >-0.003400</td>
          <td id="T_78b05_row380_col6" class="data row380 col6" >-0.035300</td>
          <td id="T_78b05_row380_col7" class="data row380 col7" >-0.062000</td>
          <td id="T_78b05_row380_col8" class="data row380 col8" >0.014100</td>
          <td id="T_78b05_row380_col9" class="data row380 col9" >0.024700</td>
          <td id="T_78b05_row380_col10" class="data row380 col10" >0.049000</td>
          <td id="T_78b05_row380_col11" class="data row380 col11" >0.018800</td>
          <td id="T_78b05_row380_col12" class="data row380 col12" >0.002700</td>
          <td id="T_78b05_row380_col13" class="data row380 col13" >0.031400</td>
          <td id="T_78b05_row380_col14" class="data row380 col14" >0.060200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row381" class="row_heading level0 row381" >382</th>
          <td id="T_78b05_row381_col0" class="data row381 col0" >None</td>
          <td id="T_78b05_row381_col1" class="data row381 col1" >0.027400</td>
          <td id="T_78b05_row381_col2" class="data row381 col2" >0.009400</td>
          <td id="T_78b05_row381_col3" class="data row381 col3" >-0.042600</td>
          <td id="T_78b05_row381_col4" class="data row381 col4" >-0.039400</td>
          <td id="T_78b05_row381_col5" class="data row381 col5" >-0.027700</td>
          <td id="T_78b05_row381_col6" class="data row381 col6" >0.011600</td>
          <td id="T_78b05_row381_col7" class="data row381 col7" >-0.053000</td>
          <td id="T_78b05_row381_col8" class="data row381 col8" >0.018100</td>
          <td id="T_78b05_row381_col9" class="data row381 col9" >0.039200</td>
          <td id="T_78b05_row381_col10" class="data row381 col10" >0.073300</td>
          <td id="T_78b05_row381_col11" class="data row381 col11" >0.021500</td>
          <td id="T_78b05_row381_col12" class="data row381 col12" >0.027100</td>
          <td id="T_78b05_row381_col13" class="data row381 col13" >0.015500</td>
          <td id="T_78b05_row381_col14" class="data row381 col14" >0.051200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row382" class="row_heading level0 row382" >383</th>
          <td id="T_78b05_row382_col0" class="data row382 col0" >None</td>
          <td id="T_78b05_row382_col1" class="data row382 col1" >0.029400</td>
          <td id="T_78b05_row382_col2" class="data row382 col2" >-0.018000</td>
          <td id="T_78b05_row382_col3" class="data row382 col3" >-0.041400</td>
          <td id="T_78b05_row382_col4" class="data row382 col4" >-0.037100</td>
          <td id="T_78b05_row382_col5" class="data row382 col5" >-0.021400</td>
          <td id="T_78b05_row382_col6" class="data row382 col6" >0.043700</td>
          <td id="T_78b05_row382_col7" class="data row382 col7" >-0.056300</td>
          <td id="T_78b05_row382_col8" class="data row382 col8" >0.016200</td>
          <td id="T_78b05_row382_col9" class="data row382 col9" >0.011800</td>
          <td id="T_78b05_row382_col10" class="data row382 col10" >0.072200</td>
          <td id="T_78b05_row382_col11" class="data row382 col11" >0.019200</td>
          <td id="T_78b05_row382_col12" class="data row382 col12" >0.020700</td>
          <td id="T_78b05_row382_col13" class="data row382 col13" >0.047600</td>
          <td id="T_78b05_row382_col14" class="data row382 col14" >0.054500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row383" class="row_heading level0 row383" >384</th>
          <td id="T_78b05_row383_col0" class="data row383 col0" >None</td>
          <td id="T_78b05_row383_col1" class="data row383 col1" >0.034500</td>
          <td id="T_78b05_row383_col2" class="data row383 col2" >0.021800</td>
          <td id="T_78b05_row383_col3" class="data row383 col3" >-0.013600</td>
          <td id="T_78b05_row383_col4" class="data row383 col4" >0.056400</td>
          <td id="T_78b05_row383_col5" class="data row383 col5" >0.029400</td>
          <td id="T_78b05_row383_col6" class="data row383 col6" >0.002400</td>
          <td id="T_78b05_row383_col7" class="data row383 col7" >0.003400</td>
          <td id="T_78b05_row383_col8" class="data row383 col8" >0.011100</td>
          <td id="T_78b05_row383_col9" class="data row383 col9" >0.051600</td>
          <td id="T_78b05_row383_col10" class="data row383 col10" >0.044300</td>
          <td id="T_78b05_row383_col11" class="data row383 col11" >0.074300</td>
          <td id="T_78b05_row383_col12" class="data row383 col12" >0.030000</td>
          <td id="T_78b05_row383_col13" class="data row383 col13" >0.006300</td>
          <td id="T_78b05_row383_col14" class="data row383 col14" >0.005200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row384" class="row_heading level0 row384" >385</th>
          <td id="T_78b05_row384_col0" class="data row384 col0" >None</td>
          <td id="T_78b05_row384_col1" class="data row384 col1" >0.033700</td>
          <td id="T_78b05_row384_col2" class="data row384 col2" >0.033100</td>
          <td id="T_78b05_row384_col3" class="data row384 col3" >-0.024400</td>
          <td id="T_78b05_row384_col4" class="data row384 col4" >-0.001400</td>
          <td id="T_78b05_row384_col5" class="data row384 col5" >0.042300</td>
          <td id="T_78b05_row384_col6" class="data row384 col6" >0.015200</td>
          <td id="T_78b05_row384_col7" class="data row384 col7" >0.032300</td>
          <td id="T_78b05_row384_col8" class="data row384 col8" >0.011800</td>
          <td id="T_78b05_row384_col9" class="data row384 col9" >0.062900</td>
          <td id="T_78b05_row384_col10" class="data row384 col10" >0.055100</td>
          <td id="T_78b05_row384_col11" class="data row384 col11" >0.016500</td>
          <td id="T_78b05_row384_col12" class="data row384 col12" >0.042900</td>
          <td id="T_78b05_row384_col13" class="data row384 col13" >0.019100</td>
          <td id="T_78b05_row384_col14" class="data row384 col14" >0.034000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row385" class="row_heading level0 row385" >386</th>
          <td id="T_78b05_row385_col0" class="data row385 col0" >None</td>
          <td id="T_78b05_row385_col1" class="data row385 col1" >0.047900</td>
          <td id="T_78b05_row385_col2" class="data row385 col2" >-0.046500</td>
          <td id="T_78b05_row385_col3" class="data row385 col3" >-0.026800</td>
          <td id="T_78b05_row385_col4" class="data row385 col4" >-0.002700</td>
          <td id="T_78b05_row385_col5" class="data row385 col5" >0.031900</td>
          <td id="T_78b05_row385_col6" class="data row385 col6" >-0.003400</td>
          <td id="T_78b05_row385_col7" class="data row385 col7" >0.041600</td>
          <td id="T_78b05_row385_col8" class="data row385 col8" >0.002400</td>
          <td id="T_78b05_row385_col9" class="data row385 col9" >0.016600</td>
          <td id="T_78b05_row385_col10" class="data row385 col10" >0.057600</td>
          <td id="T_78b05_row385_col11" class="data row385 col11" >0.015200</td>
          <td id="T_78b05_row385_col12" class="data row385 col12" >0.032600</td>
          <td id="T_78b05_row385_col13" class="data row385 col13" >0.000400</td>
          <td id="T_78b05_row385_col14" class="data row385 col14" >0.043300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row386" class="row_heading level0 row386" >387</th>
          <td id="T_78b05_row386_col0" class="data row386 col0" >None</td>
          <td id="T_78b05_row386_col1" class="data row386 col1" >0.030800</td>
          <td id="T_78b05_row386_col2" class="data row386 col2" >-0.015200</td>
          <td id="T_78b05_row386_col3" class="data row386 col3" >-0.073000</td>
          <td id="T_78b05_row386_col4" class="data row386 col4" >-0.015400</td>
          <td id="T_78b05_row386_col5" class="data row386 col5" >0.031800</td>
          <td id="T_78b05_row386_col6" class="data row386 col6" >-0.028100</td>
          <td id="T_78b05_row386_col7" class="data row386 col7" >-0.005000</td>
          <td id="T_78b05_row386_col8" class="data row386 col8" >0.014800</td>
          <td id="T_78b05_row386_col9" class="data row386 col9" >0.014600</td>
          <td id="T_78b05_row386_col10" class="data row386 col10" >0.103800</td>
          <td id="T_78b05_row386_col11" class="data row386 col11" >0.002500</td>
          <td id="T_78b05_row386_col12" class="data row386 col12" >0.032500</td>
          <td id="T_78b05_row386_col13" class="data row386 col13" >0.024200</td>
          <td id="T_78b05_row386_col14" class="data row386 col14" >0.003200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row387" class="row_heading level0 row387" >388</th>
          <td id="T_78b05_row387_col0" class="data row387 col0" >None</td>
          <td id="T_78b05_row387_col1" class="data row387 col1" >0.042200</td>
          <td id="T_78b05_row387_col2" class="data row387 col2" >0.010100</td>
          <td id="T_78b05_row387_col3" class="data row387 col3" >0.066200</td>
          <td id="T_78b05_row387_col4" class="data row387 col4" >0.052600</td>
          <td id="T_78b05_row387_col5" class="data row387 col5" >0.043100</td>
          <td id="T_78b05_row387_col6" class="data row387 col6" >0.046500</td>
          <td id="T_78b05_row387_col7" class="data row387 col7" >-0.010400</td>
          <td id="T_78b05_row387_col8" class="data row387 col8" >0.003300</td>
          <td id="T_78b05_row387_col9" class="data row387 col9" >0.040000</td>
          <td id="T_78b05_row387_col10" class="data row387 col10" >0.035500</td>
          <td id="T_78b05_row387_col11" class="data row387 col11" >0.070500</td>
          <td id="T_78b05_row387_col12" class="data row387 col12" >0.043700</td>
          <td id="T_78b05_row387_col13" class="data row387 col13" >0.050400</td>
          <td id="T_78b05_row387_col14" class="data row387 col14" >0.008600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row388" class="row_heading level0 row388" >389</th>
          <td id="T_78b05_row388_col0" class="data row388 col0" >None</td>
          <td id="T_78b05_row388_col1" class="data row388 col1" >0.041800</td>
          <td id="T_78b05_row388_col2" class="data row388 col2" >-0.042400</td>
          <td id="T_78b05_row388_col3" class="data row388 col3" >0.014700</td>
          <td id="T_78b05_row388_col4" class="data row388 col4" >0.057000</td>
          <td id="T_78b05_row388_col5" class="data row388 col5" >0.006000</td>
          <td id="T_78b05_row388_col6" class="data row388 col6" >0.026400</td>
          <td id="T_78b05_row388_col7" class="data row388 col7" >-0.019400</td>
          <td id="T_78b05_row388_col8" class="data row388 col8" >0.003800</td>
          <td id="T_78b05_row388_col9" class="data row388 col9" >0.012600</td>
          <td id="T_78b05_row388_col10" class="data row388 col10" >0.016100</td>
          <td id="T_78b05_row388_col11" class="data row388 col11" >0.075000</td>
          <td id="T_78b05_row388_col12" class="data row388 col12" >0.006700</td>
          <td id="T_78b05_row388_col13" class="data row388 col13" >0.030300</td>
          <td id="T_78b05_row388_col14" class="data row388 col14" >0.017600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row389" class="row_heading level0 row389" >390</th>
          <td id="T_78b05_row389_col0" class="data row389 col0" >None</td>
          <td id="T_78b05_row389_col1" class="data row389 col1" >0.029900</td>
          <td id="T_78b05_row389_col2" class="data row389 col2" >-0.037800</td>
          <td id="T_78b05_row389_col3" class="data row389 col3" >-0.050700</td>
          <td id="T_78b05_row389_col4" class="data row389 col4" >-0.011000</td>
          <td id="T_78b05_row389_col5" class="data row389 col5" >0.067200</td>
          <td id="T_78b05_row389_col6" class="data row389 col6" >0.036200</td>
          <td id="T_78b05_row389_col7" class="data row389 col7" >-0.070800</td>
          <td id="T_78b05_row389_col8" class="data row389 col8" >0.015600</td>
          <td id="T_78b05_row389_col9" class="data row389 col9" >0.008000</td>
          <td id="T_78b05_row389_col10" class="data row389 col10" >0.081500</td>
          <td id="T_78b05_row389_col11" class="data row389 col11" >0.007000</td>
          <td id="T_78b05_row389_col12" class="data row389 col12" >0.067800</td>
          <td id="T_78b05_row389_col13" class="data row389 col13" >0.040100</td>
          <td id="T_78b05_row389_col14" class="data row389 col14" >0.069000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row390" class="row_heading level0 row390" >391</th>
          <td id="T_78b05_row390_col0" class="data row390 col0" >None</td>
          <td id="T_78b05_row390_col1" class="data row390 col1" >0.033200</td>
          <td id="T_78b05_row390_col2" class="data row390 col2" >0.003800</td>
          <td id="T_78b05_row390_col3" class="data row390 col3" >-0.022500</td>
          <td id="T_78b05_row390_col4" class="data row390 col4" >0.057100</td>
          <td id="T_78b05_row390_col5" class="data row390 col5" >-0.061900</td>
          <td id="T_78b05_row390_col6" class="data row390 col6" >-0.002300</td>
          <td id="T_78b05_row390_col7" class="data row390 col7" >-0.002100</td>
          <td id="T_78b05_row390_col8" class="data row390 col8" >0.012300</td>
          <td id="T_78b05_row390_col9" class="data row390 col9" >0.033600</td>
          <td id="T_78b05_row390_col10" class="data row390 col10" >0.053300</td>
          <td id="T_78b05_row390_col11" class="data row390 col11" >0.075000</td>
          <td id="T_78b05_row390_col12" class="data row390 col12" >0.061200</td>
          <td id="T_78b05_row390_col13" class="data row390 col13" >0.001600</td>
          <td id="T_78b05_row390_col14" class="data row390 col14" >0.000300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row391" class="row_heading level0 row391" >392</th>
          <td id="T_78b05_row391_col0" class="data row391 col0" >None</td>
          <td id="T_78b05_row391_col1" class="data row391 col1" >0.030800</td>
          <td id="T_78b05_row391_col2" class="data row391 col2" >0.077900</td>
          <td id="T_78b05_row391_col3" class="data row391 col3" >-0.021600</td>
          <td id="T_78b05_row391_col4" class="data row391 col4" >-0.058200</td>
          <td id="T_78b05_row391_col5" class="data row391 col5" >-0.039900</td>
          <td id="T_78b05_row391_col6" class="data row391 col6" >0.011100</td>
          <td id="T_78b05_row391_col7" class="data row391 col7" >-0.079400</td>
          <td id="T_78b05_row391_col8" class="data row391 col8" >0.014800</td>
          <td id="T_78b05_row391_col9" class="data row391 col9" >0.107800</td>
          <td id="T_78b05_row391_col10" class="data row391 col10" >0.052400</td>
          <td id="T_78b05_row391_col11" class="data row391 col11" >0.040300</td>
          <td id="T_78b05_row391_col12" class="data row391 col12" >0.039300</td>
          <td id="T_78b05_row391_col13" class="data row391 col13" >0.015000</td>
          <td id="T_78b05_row391_col14" class="data row391 col14" >0.077700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row392" class="row_heading level0 row392" >393</th>
          <td id="T_78b05_row392_col0" class="data row392 col0" >None</td>
          <td id="T_78b05_row392_col1" class="data row392 col1" >0.032200</td>
          <td id="T_78b05_row392_col2" class="data row392 col2" >0.053800</td>
          <td id="T_78b05_row392_col3" class="data row392 col3" >0.012500</td>
          <td id="T_78b05_row392_col4" class="data row392 col4" >-0.041600</td>
          <td id="T_78b05_row392_col5" class="data row392 col5" >0.005600</td>
          <td id="T_78b05_row392_col6" class="data row392 col6" >-0.029300</td>
          <td id="T_78b05_row392_col7" class="data row392 col7" >0.034600</td>
          <td id="T_78b05_row392_col8" class="data row392 col8" >0.013400</td>
          <td id="T_78b05_row392_col9" class="data row392 col9" >0.083700</td>
          <td id="T_78b05_row392_col10" class="data row392 col10" >0.018200</td>
          <td id="T_78b05_row392_col11" class="data row392 col11" >0.023700</td>
          <td id="T_78b05_row392_col12" class="data row392 col12" >0.006300</td>
          <td id="T_78b05_row392_col13" class="data row392 col13" >0.025400</td>
          <td id="T_78b05_row392_col14" class="data row392 col14" >0.036300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row393" class="row_heading level0 row393" >394</th>
          <td id="T_78b05_row393_col0" class="data row393 col0" >None</td>
          <td id="T_78b05_row393_col1" class="data row393 col1" >0.038000</td>
          <td id="T_78b05_row393_col2" class="data row393 col2" >0.026300</td>
          <td id="T_78b05_row393_col3" class="data row393 col3" >0.033300</td>
          <td id="T_78b05_row393_col4" class="data row393 col4" >0.008900</td>
          <td id="T_78b05_row393_col5" class="data row393 col5" >-0.041400</td>
          <td id="T_78b05_row393_col6" class="data row393 col6" >0.086500</td>
          <td id="T_78b05_row393_col7" class="data row393 col7" >0.033800</td>
          <td id="T_78b05_row393_col8" class="data row393 col8" >0.007500</td>
          <td id="T_78b05_row393_col9" class="data row393 col9" >0.056100</td>
          <td id="T_78b05_row393_col10" class="data row393 col10" >0.002500</td>
          <td id="T_78b05_row393_col11" class="data row393 col11" >0.026800</td>
          <td id="T_78b05_row393_col12" class="data row393 col12" >0.040700</td>
          <td id="T_78b05_row393_col13" class="data row393 col13" >0.090400</td>
          <td id="T_78b05_row393_col14" class="data row393 col14" >0.035500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row394" class="row_heading level0 row394" >395</th>
          <td id="T_78b05_row394_col0" class="data row394 col0" >None</td>
          <td id="T_78b05_row394_col1" class="data row394 col1" >0.027000</td>
          <td id="T_78b05_row394_col2" class="data row394 col2" >0.058200</td>
          <td id="T_78b05_row394_col3" class="data row394 col3" >-0.049800</td>
          <td id="T_78b05_row394_col4" class="data row394 col4" >0.012000</td>
          <td id="T_78b05_row394_col5" class="data row394 col5" >0.004700</td>
          <td id="T_78b05_row394_col6" class="data row394 col6" >0.033500</td>
          <td id="T_78b05_row394_col7" class="data row394 col7" >-0.116400</td>
          <td id="T_78b05_row394_col8" class="data row394 col8" >0.018500</td>
          <td id="T_78b05_row394_col9" class="data row394 col9" >0.088000</td>
          <td id="T_78b05_row394_col10" class="data row394 col10" >0.080600</td>
          <td id="T_78b05_row394_col11" class="data row394 col11" >0.029900</td>
          <td id="T_78b05_row394_col12" class="data row394 col12" >0.005400</td>
          <td id="T_78b05_row394_col13" class="data row394 col13" >0.037400</td>
          <td id="T_78b05_row394_col14" class="data row394 col14" >0.114700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row395" class="row_heading level0 row395" >396</th>
          <td id="T_78b05_row395_col0" class="data row395 col0" >None</td>
          <td id="T_78b05_row395_col1" class="data row395 col1" >0.036500</td>
          <td id="T_78b05_row395_col2" class="data row395 col2" >-0.031400</td>
          <td id="T_78b05_row395_col3" class="data row395 col3" >-0.074000</td>
          <td id="T_78b05_row395_col4" class="data row395 col4" >0.013300</td>
          <td id="T_78b05_row395_col5" class="data row395 col5" >0.015300</td>
          <td id="T_78b05_row395_col6" class="data row395 col6" >0.026200</td>
          <td id="T_78b05_row395_col7" class="data row395 col7" >-0.032100</td>
          <td id="T_78b05_row395_col8" class="data row395 col8" >0.009100</td>
          <td id="T_78b05_row395_col9" class="data row395 col9" >0.001500</td>
          <td id="T_78b05_row395_col10" class="data row395 col10" >0.104700</td>
          <td id="T_78b05_row395_col11" class="data row395 col11" >0.031200</td>
          <td id="T_78b05_row395_col12" class="data row395 col12" >0.016000</td>
          <td id="T_78b05_row395_col13" class="data row395 col13" >0.030100</td>
          <td id="T_78b05_row395_col14" class="data row395 col14" >0.030400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row396" class="row_heading level0 row396" >397</th>
          <td id="T_78b05_row396_col0" class="data row396 col0" >None</td>
          <td id="T_78b05_row396_col1" class="data row396 col1" >0.031800</td>
          <td id="T_78b05_row396_col2" class="data row396 col2" >0.026700</td>
          <td id="T_78b05_row396_col3" class="data row396 col3" >-0.006700</td>
          <td id="T_78b05_row396_col4" class="data row396 col4" >0.026300</td>
          <td id="T_78b05_row396_col5" class="data row396 col5" >0.016500</td>
          <td id="T_78b05_row396_col6" class="data row396 col6" >0.028900</td>
          <td id="T_78b05_row396_col7" class="data row396 col7" >0.065300</td>
          <td id="T_78b05_row396_col8" class="data row396 col8" >0.013800</td>
          <td id="T_78b05_row396_col9" class="data row396 col9" >0.056500</td>
          <td id="T_78b05_row396_col10" class="data row396 col10" >0.037400</td>
          <td id="T_78b05_row396_col11" class="data row396 col11" >0.044200</td>
          <td id="T_78b05_row396_col12" class="data row396 col12" >0.017100</td>
          <td id="T_78b05_row396_col13" class="data row396 col13" >0.032800</td>
          <td id="T_78b05_row396_col14" class="data row396 col14" >0.067000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row397" class="row_heading level0 row397" >398</th>
          <td id="T_78b05_row397_col0" class="data row397 col0" >None</td>
          <td id="T_78b05_row397_col1" class="data row397 col1" >0.027100</td>
          <td id="T_78b05_row397_col2" class="data row397 col2" >0.054100</td>
          <td id="T_78b05_row397_col3" class="data row397 col3" >-0.057300</td>
          <td id="T_78b05_row397_col4" class="data row397 col4" >-0.009500</td>
          <td id="T_78b05_row397_col5" class="data row397 col5" >-0.046300</td>
          <td id="T_78b05_row397_col6" class="data row397 col6" >0.034200</td>
          <td id="T_78b05_row397_col7" class="data row397 col7" >-0.027100</td>
          <td id="T_78b05_row397_col8" class="data row397 col8" >0.018500</td>
          <td id="T_78b05_row397_col9" class="data row397 col9" >0.083900</td>
          <td id="T_78b05_row397_col10" class="data row397 col10" >0.088000</td>
          <td id="T_78b05_row397_col11" class="data row397 col11" >0.008400</td>
          <td id="T_78b05_row397_col12" class="data row397 col12" >0.045700</td>
          <td id="T_78b05_row397_col13" class="data row397 col13" >0.038100</td>
          <td id="T_78b05_row397_col14" class="data row397 col14" >0.025300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row398" class="row_heading level0 row398" >399</th>
          <td id="T_78b05_row398_col0" class="data row398 col0" >None</td>
          <td id="T_78b05_row398_col1" class="data row398 col1" >0.041600</td>
          <td id="T_78b05_row398_col2" class="data row398 col2" >0.015000</td>
          <td id="T_78b05_row398_col3" class="data row398 col3" >0.010700</td>
          <td id="T_78b05_row398_col4" class="data row398 col4" >0.015600</td>
          <td id="T_78b05_row398_col5" class="data row398 col5" >0.008900</td>
          <td id="T_78b05_row398_col6" class="data row398 col6" >0.025000</td>
          <td id="T_78b05_row398_col7" class="data row398 col7" >0.058200</td>
          <td id="T_78b05_row398_col8" class="data row398 col8" >0.003900</td>
          <td id="T_78b05_row398_col9" class="data row398 col9" >0.044800</td>
          <td id="T_78b05_row398_col10" class="data row398 col10" >0.020100</td>
          <td id="T_78b05_row398_col11" class="data row398 col11" >0.033500</td>
          <td id="T_78b05_row398_col12" class="data row398 col12" >0.009600</td>
          <td id="T_78b05_row398_col13" class="data row398 col13" >0.028800</td>
          <td id="T_78b05_row398_col14" class="data row398 col14" >0.060000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row399" class="row_heading level0 row399" >400</th>
          <td id="T_78b05_row399_col0" class="data row399 col0" >None</td>
          <td id="T_78b05_row399_col1" class="data row399 col1" >0.040100</td>
          <td id="T_78b05_row399_col2" class="data row399 col2" >0.013200</td>
          <td id="T_78b05_row399_col3" class="data row399 col3" >0.011700</td>
          <td id="T_78b05_row399_col4" class="data row399 col4" >-0.010500</td>
          <td id="T_78b05_row399_col5" class="data row399 col5" >0.015900</td>
          <td id="T_78b05_row399_col6" class="data row399 col6" >0.020200</td>
          <td id="T_78b05_row399_col7" class="data row399 col7" >-0.046400</td>
          <td id="T_78b05_row399_col8" class="data row399 col8" >0.005400</td>
          <td id="T_78b05_row399_col9" class="data row399 col9" >0.043000</td>
          <td id="T_78b05_row399_col10" class="data row399 col10" >0.019100</td>
          <td id="T_78b05_row399_col11" class="data row399 col11" >0.007400</td>
          <td id="T_78b05_row399_col12" class="data row399 col12" >0.016600</td>
          <td id="T_78b05_row399_col13" class="data row399 col13" >0.024100</td>
          <td id="T_78b05_row399_col14" class="data row399 col14" >0.044700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row400" class="row_heading level0 row400" >401</th>
          <td id="T_78b05_row400_col0" class="data row400 col0" >None</td>
          <td id="T_78b05_row400_col1" class="data row400 col1" >0.037500</td>
          <td id="T_78b05_row400_col2" class="data row400 col2" >-0.032400</td>
          <td id="T_78b05_row400_col3" class="data row400 col3" >-0.055500</td>
          <td id="T_78b05_row400_col4" class="data row400 col4" >-0.028800</td>
          <td id="T_78b05_row400_col5" class="data row400 col5" >-0.003100</td>
          <td id="T_78b05_row400_col6" class="data row400 col6" >0.031500</td>
          <td id="T_78b05_row400_col7" class="data row400 col7" >-0.017900</td>
          <td id="T_78b05_row400_col8" class="data row400 col8" >0.008000</td>
          <td id="T_78b05_row400_col9" class="data row400 col9" >0.002500</td>
          <td id="T_78b05_row400_col10" class="data row400 col10" >0.086200</td>
          <td id="T_78b05_row400_col11" class="data row400 col11" >0.010900</td>
          <td id="T_78b05_row400_col12" class="data row400 col12" >0.002500</td>
          <td id="T_78b05_row400_col13" class="data row400 col13" >0.035400</td>
          <td id="T_78b05_row400_col14" class="data row400 col14" >0.016200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row401" class="row_heading level0 row401" >402</th>
          <td id="T_78b05_row401_col0" class="data row401 col0" >None</td>
          <td id="T_78b05_row401_col1" class="data row401 col1" >0.028300</td>
          <td id="T_78b05_row401_col2" class="data row401 col2" >0.052200</td>
          <td id="T_78b05_row401_col3" class="data row401 col3" >-0.020000</td>
          <td id="T_78b05_row401_col4" class="data row401 col4" >-0.011300</td>
          <td id="T_78b05_row401_col5" class="data row401 col5" >0.028800</td>
          <td id="T_78b05_row401_col6" class="data row401 col6" >-0.021500</td>
          <td id="T_78b05_row401_col7" class="data row401 col7" >-0.020200</td>
          <td id="T_78b05_row401_col8" class="data row401 col8" >0.017200</td>
          <td id="T_78b05_row401_col9" class="data row401 col9" >0.082100</td>
          <td id="T_78b05_row401_col10" class="data row401 col10" >0.050800</td>
          <td id="T_78b05_row401_col11" class="data row401 col11" >0.006600</td>
          <td id="T_78b05_row401_col12" class="data row401 col12" >0.029400</td>
          <td id="T_78b05_row401_col13" class="data row401 col13" >0.017600</td>
          <td id="T_78b05_row401_col14" class="data row401 col14" >0.018400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row402" class="row_heading level0 row402" >403</th>
          <td id="T_78b05_row402_col0" class="data row402 col0" >None</td>
          <td id="T_78b05_row402_col1" class="data row402 col1" >0.032700</td>
          <td id="T_78b05_row402_col2" class="data row402 col2" >0.034200</td>
          <td id="T_78b05_row402_col3" class="data row402 col3" >-0.072900</td>
          <td id="T_78b05_row402_col4" class="data row402 col4" >-0.056700</td>
          <td id="T_78b05_row402_col5" class="data row402 col5" >0.025400</td>
          <td id="T_78b05_row402_col6" class="data row402 col6" >0.011600</td>
          <td id="T_78b05_row402_col7" class="data row402 col7" >0.040100</td>
          <td id="T_78b05_row402_col8" class="data row402 col8" >0.012800</td>
          <td id="T_78b05_row402_col9" class="data row402 col9" >0.064000</td>
          <td id="T_78b05_row402_col10" class="data row402 col10" >0.103600</td>
          <td id="T_78b05_row402_col11" class="data row402 col11" >0.038700</td>
          <td id="T_78b05_row402_col12" class="data row402 col12" >0.026000</td>
          <td id="T_78b05_row402_col13" class="data row402 col13" >0.015500</td>
          <td id="T_78b05_row402_col14" class="data row402 col14" >0.041900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row403" class="row_heading level0 row403" >404</th>
          <td id="T_78b05_row403_col0" class="data row403 col0" >None</td>
          <td id="T_78b05_row403_col1" class="data row403 col1" >0.032000</td>
          <td id="T_78b05_row403_col2" class="data row403 col2" >-0.030200</td>
          <td id="T_78b05_row403_col3" class="data row403 col3" >-0.026300</td>
          <td id="T_78b05_row403_col4" class="data row403 col4" >-0.018500</td>
          <td id="T_78b05_row403_col5" class="data row403 col5" >-0.053300</td>
          <td id="T_78b05_row403_col6" class="data row403 col6" >0.013000</td>
          <td id="T_78b05_row403_col7" class="data row403 col7" >0.007100</td>
          <td id="T_78b05_row403_col8" class="data row403 col8" >0.013600</td>
          <td id="T_78b05_row403_col9" class="data row403 col9" >0.000400</td>
          <td id="T_78b05_row403_col10" class="data row403 col10" >0.057000</td>
          <td id="T_78b05_row403_col11" class="data row403 col11" >0.000500</td>
          <td id="T_78b05_row403_col12" class="data row403 col12" >0.052700</td>
          <td id="T_78b05_row403_col13" class="data row403 col13" >0.016900</td>
          <td id="T_78b05_row403_col14" class="data row403 col14" >0.008800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row404" class="row_heading level0 row404" >405</th>
          <td id="T_78b05_row404_col0" class="data row404 col0" >None</td>
          <td id="T_78b05_row404_col1" class="data row404 col1" >0.041200</td>
          <td id="T_78b05_row404_col2" class="data row404 col2" >0.036600</td>
          <td id="T_78b05_row404_col3" class="data row404 col3" >-0.022000</td>
          <td id="T_78b05_row404_col4" class="data row404 col4" >-0.033800</td>
          <td id="T_78b05_row404_col5" class="data row404 col5" >-0.025000</td>
          <td id="T_78b05_row404_col6" class="data row404 col6" >0.073600</td>
          <td id="T_78b05_row404_col7" class="data row404 col7" >0.022000</td>
          <td id="T_78b05_row404_col8" class="data row404 col8" >0.004300</td>
          <td id="T_78b05_row404_col9" class="data row404 col9" >0.066400</td>
          <td id="T_78b05_row404_col10" class="data row404 col10" >0.052700</td>
          <td id="T_78b05_row404_col11" class="data row404 col11" >0.015900</td>
          <td id="T_78b05_row404_col12" class="data row404 col12" >0.024300</td>
          <td id="T_78b05_row404_col13" class="data row404 col13" >0.077400</td>
          <td id="T_78b05_row404_col14" class="data row404 col14" >0.023800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row405" class="row_heading level0 row405" >406</th>
          <td id="T_78b05_row405_col0" class="data row405 col0" >PC1</td>
          <td id="T_78b05_row405_col1" class="data row405 col1" >0.025400</td>
          <td id="T_78b05_row405_col2" class="data row405 col2" >0.030800</td>
          <td id="T_78b05_row405_col3" class="data row405 col3" >-0.027200</td>
          <td id="T_78b05_row405_col4" class="data row405 col4" >-0.010500</td>
          <td id="T_78b05_row405_col5" class="data row405 col5" >0.051800</td>
          <td id="T_78b05_row405_col6" class="data row405 col6" >-0.001500</td>
          <td id="T_78b05_row405_col7" class="data row405 col7" >-0.008900</td>
          <td id="T_78b05_row405_col8" class="data row405 col8" >0.020100</td>
          <td id="T_78b05_row405_col9" class="data row405 col9" >0.060600</td>
          <td id="T_78b05_row405_col10" class="data row405 col10" >0.057900</td>
          <td id="T_78b05_row405_col11" class="data row405 col11" >0.007400</td>
          <td id="T_78b05_row405_col12" class="data row405 col12" >0.052400</td>
          <td id="T_78b05_row405_col13" class="data row405 col13" >0.002400</td>
          <td id="T_78b05_row405_col14" class="data row405 col14" >0.007100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row406" class="row_heading level0 row406" >407</th>
          <td id="T_78b05_row406_col0" class="data row406 col0" >None</td>
          <td id="T_78b05_row406_col1" class="data row406 col1" >0.044500</td>
          <td id="T_78b05_row406_col2" class="data row406 col2" >-0.021600</td>
          <td id="T_78b05_row406_col3" class="data row406 col3" >0.024400</td>
          <td id="T_78b05_row406_col4" class="data row406 col4" >-0.036200</td>
          <td id="T_78b05_row406_col5" class="data row406 col5" >-0.005300</td>
          <td id="T_78b05_row406_col6" class="data row406 col6" >0.012700</td>
          <td id="T_78b05_row406_col7" class="data row406 col7" >-0.002600</td>
          <td id="T_78b05_row406_col8" class="data row406 col8" >0.001000</td>
          <td id="T_78b05_row406_col9" class="data row406 col9" >0.008200</td>
          <td id="T_78b05_row406_col10" class="data row406 col10" >0.006300</td>
          <td id="T_78b05_row406_col11" class="data row406 col11" >0.018300</td>
          <td id="T_78b05_row406_col12" class="data row406 col12" >0.004700</td>
          <td id="T_78b05_row406_col13" class="data row406 col13" >0.016600</td>
          <td id="T_78b05_row406_col14" class="data row406 col14" >0.000800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row407" class="row_heading level0 row407" >408</th>
          <td id="T_78b05_row407_col0" class="data row407 col0" >None</td>
          <td id="T_78b05_row407_col1" class="data row407 col1" >0.031100</td>
          <td id="T_78b05_row407_col2" class="data row407 col2" >-0.023000</td>
          <td id="T_78b05_row407_col3" class="data row407 col3" >-0.041800</td>
          <td id="T_78b05_row407_col4" class="data row407 col4" >-0.024500</td>
          <td id="T_78b05_row407_col5" class="data row407 col5" >-0.009600</td>
          <td id="T_78b05_row407_col6" class="data row407 col6" >0.045600</td>
          <td id="T_78b05_row407_col7" class="data row407 col7" >-0.074900</td>
          <td id="T_78b05_row407_col8" class="data row407 col8" >0.014400</td>
          <td id="T_78b05_row407_col9" class="data row407 col9" >0.006800</td>
          <td id="T_78b05_row407_col10" class="data row407 col10" >0.072500</td>
          <td id="T_78b05_row407_col11" class="data row407 col11" >0.006600</td>
          <td id="T_78b05_row407_col12" class="data row407 col12" >0.008900</td>
          <td id="T_78b05_row407_col13" class="data row407 col13" >0.049500</td>
          <td id="T_78b05_row407_col14" class="data row407 col14" >0.073200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row408" class="row_heading level0 row408" >409</th>
          <td id="T_78b05_row408_col0" class="data row408 col0" >None</td>
          <td id="T_78b05_row408_col1" class="data row408 col1" >0.031400</td>
          <td id="T_78b05_row408_col2" class="data row408 col2" >0.069800</td>
          <td id="T_78b05_row408_col3" class="data row408 col3" >0.016800</td>
          <td id="T_78b05_row408_col4" class="data row408 col4" >0.002100</td>
          <td id="T_78b05_row408_col5" class="data row408 col5" >-0.052600</td>
          <td id="T_78b05_row408_col6" class="data row408 col6" >0.051200</td>
          <td id="T_78b05_row408_col7" class="data row408 col7" >0.048800</td>
          <td id="T_78b05_row408_col8" class="data row408 col8" >0.014200</td>
          <td id="T_78b05_row408_col9" class="data row408 col9" >0.099600</td>
          <td id="T_78b05_row408_col10" class="data row408 col10" >0.014000</td>
          <td id="T_78b05_row408_col11" class="data row408 col11" >0.020000</td>
          <td id="T_78b05_row408_col12" class="data row408 col12" >0.052000</td>
          <td id="T_78b05_row408_col13" class="data row408 col13" >0.055100</td>
          <td id="T_78b05_row408_col14" class="data row408 col14" >0.050500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row409" class="row_heading level0 row409" >410</th>
          <td id="T_78b05_row409_col0" class="data row409 col0" >None</td>
          <td id="T_78b05_row409_col1" class="data row409 col1" >0.029900</td>
          <td id="T_78b05_row409_col2" class="data row409 col2" >0.079200</td>
          <td id="T_78b05_row409_col3" class="data row409 col3" >-0.022100</td>
          <td id="T_78b05_row409_col4" class="data row409 col4" >-0.023900</td>
          <td id="T_78b05_row409_col5" class="data row409 col5" >-0.036700</td>
          <td id="T_78b05_row409_col6" class="data row409 col6" >-0.008700</td>
          <td id="T_78b05_row409_col7" class="data row409 col7" >0.036500</td>
          <td id="T_78b05_row409_col8" class="data row409 col8" >0.015600</td>
          <td id="T_78b05_row409_col9" class="data row409 col9" >0.109000</td>
          <td id="T_78b05_row409_col10" class="data row409 col10" >0.052800</td>
          <td id="T_78b05_row409_col11" class="data row409 col11" >0.006000</td>
          <td id="T_78b05_row409_col12" class="data row409 col12" >0.036100</td>
          <td id="T_78b05_row409_col13" class="data row409 col13" >0.004800</td>
          <td id="T_78b05_row409_col14" class="data row409 col14" >0.038200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row410" class="row_heading level0 row410" >411</th>
          <td id="T_78b05_row410_col0" class="data row410 col0" >None</td>
          <td id="T_78b05_row410_col1" class="data row410 col1" >0.044200</td>
          <td id="T_78b05_row410_col2" class="data row410 col2" >-0.032100</td>
          <td id="T_78b05_row410_col3" class="data row410 col3" >0.014000</td>
          <td id="T_78b05_row410_col4" class="data row410 col4" >0.030700</td>
          <td id="T_78b05_row410_col5" class="data row410 col5" >0.085100</td>
          <td id="T_78b05_row410_col6" class="data row410 col6" >0.018300</td>
          <td id="T_78b05_row410_col7" class="data row410 col7" >0.020100</td>
          <td id="T_78b05_row410_col8" class="data row410 col8" >0.001400</td>
          <td id="T_78b05_row410_col9" class="data row410 col9" >0.002200</td>
          <td id="T_78b05_row410_col10" class="data row410 col10" >0.016700</td>
          <td id="T_78b05_row410_col11" class="data row410 col11" >0.048600</td>
          <td id="T_78b05_row410_col12" class="data row410 col12" >0.085800</td>
          <td id="T_78b05_row410_col13" class="data row410 col13" >0.022200</td>
          <td id="T_78b05_row410_col14" class="data row410 col14" >0.021900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row411" class="row_heading level0 row411" >412</th>
          <td id="T_78b05_row411_col0" class="data row411 col0" >None</td>
          <td id="T_78b05_row411_col1" class="data row411 col1" >0.036300</td>
          <td id="T_78b05_row411_col2" class="data row411 col2" >-0.005000</td>
          <td id="T_78b05_row411_col3" class="data row411 col3" >-0.037200</td>
          <td id="T_78b05_row411_col4" class="data row411 col4" >-0.013200</td>
          <td id="T_78b05_row411_col5" class="data row411 col5" >-0.010900</td>
          <td id="T_78b05_row411_col6" class="data row411 col6" >0.101300</td>
          <td id="T_78b05_row411_col7" class="data row411 col7" >-0.001700</td>
          <td id="T_78b05_row411_col8" class="data row411 col8" >0.009200</td>
          <td id="T_78b05_row411_col9" class="data row411 col9" >0.024800</td>
          <td id="T_78b05_row411_col10" class="data row411 col10" >0.067900</td>
          <td id="T_78b05_row411_col11" class="data row411 col11" >0.004800</td>
          <td id="T_78b05_row411_col12" class="data row411 col12" >0.010200</td>
          <td id="T_78b05_row411_col13" class="data row411 col13" >0.105200</td>
          <td id="T_78b05_row411_col14" class="data row411 col14" >0.000100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row412" class="row_heading level0 row412" >413</th>
          <td id="T_78b05_row412_col0" class="data row412 col0" >None</td>
          <td id="T_78b05_row412_col1" class="data row412 col1" >0.039700</td>
          <td id="T_78b05_row412_col2" class="data row412 col2" >-0.048300</td>
          <td id="T_78b05_row412_col3" class="data row412 col3" >-0.039700</td>
          <td id="T_78b05_row412_col4" class="data row412 col4" >-0.028400</td>
          <td id="T_78b05_row412_col5" class="data row412 col5" >0.046400</td>
          <td id="T_78b05_row412_col6" class="data row412 col6" >0.010600</td>
          <td id="T_78b05_row412_col7" class="data row412 col7" >-0.007900</td>
          <td id="T_78b05_row412_col8" class="data row412 col8" >0.005800</td>
          <td id="T_78b05_row412_col9" class="data row412 col9" >0.018500</td>
          <td id="T_78b05_row412_col10" class="data row412 col10" >0.070500</td>
          <td id="T_78b05_row412_col11" class="data row412 col11" >0.010500</td>
          <td id="T_78b05_row412_col12" class="data row412 col12" >0.047000</td>
          <td id="T_78b05_row412_col13" class="data row412 col13" >0.014500</td>
          <td id="T_78b05_row412_col14" class="data row412 col14" >0.006200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row413" class="row_heading level0 row413" >414</th>
          <td id="T_78b05_row413_col0" class="data row413 col0" >None</td>
          <td id="T_78b05_row413_col1" class="data row413 col1" >0.043500</td>
          <td id="T_78b05_row413_col2" class="data row413 col2" >-0.021900</td>
          <td id="T_78b05_row413_col3" class="data row413 col3" >0.022700</td>
          <td id="T_78b05_row413_col4" class="data row413 col4" >-0.022900</td>
          <td id="T_78b05_row413_col5" class="data row413 col5" >0.008600</td>
          <td id="T_78b05_row413_col6" class="data row413 col6" >0.019400</td>
          <td id="T_78b05_row413_col7" class="data row413 col7" >0.031300</td>
          <td id="T_78b05_row413_col8" class="data row413 col8" >0.002000</td>
          <td id="T_78b05_row413_col9" class="data row413 col9" >0.007900</td>
          <td id="T_78b05_row413_col10" class="data row413 col10" >0.008100</td>
          <td id="T_78b05_row413_col11" class="data row413 col11" >0.005000</td>
          <td id="T_78b05_row413_col12" class="data row413 col12" >0.009200</td>
          <td id="T_78b05_row413_col13" class="data row413 col13" >0.023300</td>
          <td id="T_78b05_row413_col14" class="data row413 col14" >0.033100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row414" class="row_heading level0 row414" >415</th>
          <td id="T_78b05_row414_col0" class="data row414 col0" >None</td>
          <td id="T_78b05_row414_col1" class="data row414 col1" >0.046300</td>
          <td id="T_78b05_row414_col2" class="data row414 col2" >0.021000</td>
          <td id="T_78b05_row414_col3" class="data row414 col3" >0.029000</td>
          <td id="T_78b05_row414_col4" class="data row414 col4" >-0.042800</td>
          <td id="T_78b05_row414_col5" class="data row414 col5" >-0.058400</td>
          <td id="T_78b05_row414_col6" class="data row414 col6" >0.052200</td>
          <td id="T_78b05_row414_col7" class="data row414 col7" >0.080600</td>
          <td id="T_78b05_row414_col8" class="data row414 col8" >0.000700</td>
          <td id="T_78b05_row414_col9" class="data row414 col9" >0.050900</td>
          <td id="T_78b05_row414_col10" class="data row414 col10" >0.001800</td>
          <td id="T_78b05_row414_col11" class="data row414 col11" >0.024900</td>
          <td id="T_78b05_row414_col12" class="data row414 col12" >0.057800</td>
          <td id="T_78b05_row414_col13" class="data row414 col13" >0.056100</td>
          <td id="T_78b05_row414_col14" class="data row414 col14" >0.082300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row415" class="row_heading level0 row415" >416</th>
          <td id="T_78b05_row415_col0" class="data row415 col0" >None</td>
          <td id="T_78b05_row415_col1" class="data row415 col1" >0.041100</td>
          <td id="T_78b05_row415_col2" class="data row415 col2" >-0.062100</td>
          <td id="T_78b05_row415_col3" class="data row415 col3" >0.024500</td>
          <td id="T_78b05_row415_col4" class="data row415 col4" >0.027900</td>
          <td id="T_78b05_row415_col5" class="data row415 col5" >-0.080400</td>
          <td id="T_78b05_row415_col6" class="data row415 col6" >0.005300</td>
          <td id="T_78b05_row415_col7" class="data row415 col7" >-0.010900</td>
          <td id="T_78b05_row415_col8" class="data row415 col8" >0.004400</td>
          <td id="T_78b05_row415_col9" class="data row415 col9" >0.032300</td>
          <td id="T_78b05_row415_col10" class="data row415 col10" >0.006200</td>
          <td id="T_78b05_row415_col11" class="data row415 col11" >0.045800</td>
          <td id="T_78b05_row415_col12" class="data row415 col12" >0.079800</td>
          <td id="T_78b05_row415_col13" class="data row415 col13" >0.009200</td>
          <td id="T_78b05_row415_col14" class="data row415 col14" >0.009100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row416" class="row_heading level0 row416" >417</th>
          <td id="T_78b05_row416_col0" class="data row416 col0" >None</td>
          <td id="T_78b05_row416_col1" class="data row416 col1" >0.033400</td>
          <td id="T_78b05_row416_col2" class="data row416 col2" >-0.018000</td>
          <td id="T_78b05_row416_col3" class="data row416 col3" >-0.031900</td>
          <td id="T_78b05_row416_col4" class="data row416 col4" >-0.018700</td>
          <td id="T_78b05_row416_col5" class="data row416 col5" >-0.004000</td>
          <td id="T_78b05_row416_col6" class="data row416 col6" >0.025400</td>
          <td id="T_78b05_row416_col7" class="data row416 col7" >-0.034700</td>
          <td id="T_78b05_row416_col8" class="data row416 col8" >0.012200</td>
          <td id="T_78b05_row416_col9" class="data row416 col9" >0.011900</td>
          <td id="T_78b05_row416_col10" class="data row416 col10" >0.062700</td>
          <td id="T_78b05_row416_col11" class="data row416 col11" >0.000800</td>
          <td id="T_78b05_row416_col12" class="data row416 col12" >0.003300</td>
          <td id="T_78b05_row416_col13" class="data row416 col13" >0.029300</td>
          <td id="T_78b05_row416_col14" class="data row416 col14" >0.032900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row417" class="row_heading level0 row417" >418</th>
          <td id="T_78b05_row417_col0" class="data row417 col0" >None</td>
          <td id="T_78b05_row417_col1" class="data row417 col1" >0.041400</td>
          <td id="T_78b05_row417_col2" class="data row417 col2" >0.010300</td>
          <td id="T_78b05_row417_col3" class="data row417 col3" >0.019900</td>
          <td id="T_78b05_row417_col4" class="data row417 col4" >0.029300</td>
          <td id="T_78b05_row417_col5" class="data row417 col5" >0.014300</td>
          <td id="T_78b05_row417_col6" class="data row417 col6" >-0.022500</td>
          <td id="T_78b05_row417_col7" class="data row417 col7" >-0.051600</td>
          <td id="T_78b05_row417_col8" class="data row417 col8" >0.004100</td>
          <td id="T_78b05_row417_col9" class="data row417 col9" >0.040100</td>
          <td id="T_78b05_row417_col10" class="data row417 col10" >0.010800</td>
          <td id="T_78b05_row417_col11" class="data row417 col11" >0.047200</td>
          <td id="T_78b05_row417_col12" class="data row417 col12" >0.014900</td>
          <td id="T_78b05_row417_col13" class="data row417 col13" >0.018600</td>
          <td id="T_78b05_row417_col14" class="data row417 col14" >0.049800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row418" class="row_heading level0 row418" >419</th>
          <td id="T_78b05_row418_col0" class="data row418 col0" >None</td>
          <td id="T_78b05_row418_col1" class="data row418 col1" >0.038600</td>
          <td id="T_78b05_row418_col2" class="data row418 col2" >0.001400</td>
          <td id="T_78b05_row418_col3" class="data row418 col3" >-0.003900</td>
          <td id="T_78b05_row418_col4" class="data row418 col4" >0.020700</td>
          <td id="T_78b05_row418_col5" class="data row418 col5" >0.024100</td>
          <td id="T_78b05_row418_col6" class="data row418 col6" >0.032600</td>
          <td id="T_78b05_row418_col7" class="data row418 col7" >-0.032300</td>
          <td id="T_78b05_row418_col8" class="data row418 col8" >0.006900</td>
          <td id="T_78b05_row418_col9" class="data row418 col9" >0.031200</td>
          <td id="T_78b05_row418_col10" class="data row418 col10" >0.034700</td>
          <td id="T_78b05_row418_col11" class="data row418 col11" >0.038600</td>
          <td id="T_78b05_row418_col12" class="data row418 col12" >0.024800</td>
          <td id="T_78b05_row418_col13" class="data row418 col13" >0.036500</td>
          <td id="T_78b05_row418_col14" class="data row418 col14" >0.030600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row419" class="row_heading level0 row419" >420</th>
          <td id="T_78b05_row419_col0" class="data row419 col0" >None</td>
          <td id="T_78b05_row419_col1" class="data row419 col1" >0.028800</td>
          <td id="T_78b05_row419_col2" class="data row419 col2" >0.021000</td>
          <td id="T_78b05_row419_col3" class="data row419 col3" >-0.011900</td>
          <td id="T_78b05_row419_col4" class="data row419 col4" >0.062900</td>
          <td id="T_78b05_row419_col5" class="data row419 col5" >-0.001100</td>
          <td id="T_78b05_row419_col6" class="data row419 col6" >-0.003300</td>
          <td id="T_78b05_row419_col7" class="data row419 col7" >-0.099500</td>
          <td id="T_78b05_row419_col8" class="data row419 col8" >0.016700</td>
          <td id="T_78b05_row419_col9" class="data row419 col9" >0.050800</td>
          <td id="T_78b05_row419_col10" class="data row419 col10" >0.042600</td>
          <td id="T_78b05_row419_col11" class="data row419 col11" >0.080800</td>
          <td id="T_78b05_row419_col12" class="data row419 col12" >0.000500</td>
          <td id="T_78b05_row419_col13" class="data row419 col13" >0.000600</td>
          <td id="T_78b05_row419_col14" class="data row419 col14" >0.097800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row420" class="row_heading level0 row420" >421</th>
          <td id="T_78b05_row420_col0" class="data row420 col0" >None</td>
          <td id="T_78b05_row420_col1" class="data row420 col1" >0.034600</td>
          <td id="T_78b05_row420_col2" class="data row420 col2" >0.065700</td>
          <td id="T_78b05_row420_col3" class="data row420 col3" >0.023400</td>
          <td id="T_78b05_row420_col4" class="data row420 col4" >0.009500</td>
          <td id="T_78b05_row420_col5" class="data row420 col5" >-0.010900</td>
          <td id="T_78b05_row420_col6" class="data row420 col6" >-0.061400</td>
          <td id="T_78b05_row420_col7" class="data row420 col7" >-0.056700</td>
          <td id="T_78b05_row420_col8" class="data row420 col8" >0.010900</td>
          <td id="T_78b05_row420_col9" class="data row420 col9" >0.095500</td>
          <td id="T_78b05_row420_col10" class="data row420 col10" >0.007300</td>
          <td id="T_78b05_row420_col11" class="data row420 col11" >0.027400</td>
          <td id="T_78b05_row420_col12" class="data row420 col12" >0.010300</td>
          <td id="T_78b05_row420_col13" class="data row420 col13" >0.057500</td>
          <td id="T_78b05_row420_col14" class="data row420 col14" >0.055000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row421" class="row_heading level0 row421" >422</th>
          <td id="T_78b05_row421_col0" class="data row421 col0" >None</td>
          <td id="T_78b05_row421_col1" class="data row421 col1" >0.038600</td>
          <td id="T_78b05_row421_col2" class="data row421 col2" >-0.015700</td>
          <td id="T_78b05_row421_col3" class="data row421 col3" >0.020000</td>
          <td id="T_78b05_row421_col4" class="data row421 col4" >-0.019300</td>
          <td id="T_78b05_row421_col5" class="data row421 col5" >-0.002900</td>
          <td id="T_78b05_row421_col6" class="data row421 col6" >-0.099700</td>
          <td id="T_78b05_row421_col7" class="data row421 col7" >-0.010000</td>
          <td id="T_78b05_row421_col8" class="data row421 col8" >0.006900</td>
          <td id="T_78b05_row421_col9" class="data row421 col9" >0.014200</td>
          <td id="T_78b05_row421_col10" class="data row421 col10" >0.010700</td>
          <td id="T_78b05_row421_col11" class="data row421 col11" >0.001400</td>
          <td id="T_78b05_row421_col12" class="data row421 col12" >0.002300</td>
          <td id="T_78b05_row421_col13" class="data row421 col13" >0.095800</td>
          <td id="T_78b05_row421_col14" class="data row421 col14" >0.008200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row422" class="row_heading level0 row422" >423</th>
          <td id="T_78b05_row422_col0" class="data row422 col0" >None</td>
          <td id="T_78b05_row422_col1" class="data row422 col1" >0.038000</td>
          <td id="T_78b05_row422_col2" class="data row422 col2" >0.017700</td>
          <td id="T_78b05_row422_col3" class="data row422 col3" >0.018400</td>
          <td id="T_78b05_row422_col4" class="data row422 col4" >0.023700</td>
          <td id="T_78b05_row422_col5" class="data row422 col5" >0.014100</td>
          <td id="T_78b05_row422_col6" class="data row422 col6" >0.007400</td>
          <td id="T_78b05_row422_col7" class="data row422 col7" >0.047100</td>
          <td id="T_78b05_row422_col8" class="data row422 col8" >0.007500</td>
          <td id="T_78b05_row422_col9" class="data row422 col9" >0.047500</td>
          <td id="T_78b05_row422_col10" class="data row422 col10" >0.012400</td>
          <td id="T_78b05_row422_col11" class="data row422 col11" >0.041600</td>
          <td id="T_78b05_row422_col12" class="data row422 col12" >0.014700</td>
          <td id="T_78b05_row422_col13" class="data row422 col13" >0.011300</td>
          <td id="T_78b05_row422_col14" class="data row422 col14" >0.048900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row423" class="row_heading level0 row423" >424</th>
          <td id="T_78b05_row423_col0" class="data row423 col0" >None</td>
          <td id="T_78b05_row423_col1" class="data row423 col1" >0.042400</td>
          <td id="T_78b05_row423_col2" class="data row423 col2" >0.018200</td>
          <td id="T_78b05_row423_col3" class="data row423 col3" >0.070600</td>
          <td id="T_78b05_row423_col4" class="data row423 col4" >0.059900</td>
          <td id="T_78b05_row423_col5" class="data row423 col5" >-0.021700</td>
          <td id="T_78b05_row423_col6" class="data row423 col6" >0.025300</td>
          <td id="T_78b05_row423_col7" class="data row423 col7" >-0.016100</td>
          <td id="T_78b05_row423_col8" class="data row423 col8" >0.003100</td>
          <td id="T_78b05_row423_col9" class="data row423 col9" >0.048000</td>
          <td id="T_78b05_row423_col10" class="data row423 col10" >0.039800</td>
          <td id="T_78b05_row423_col11" class="data row423 col11" >0.077800</td>
          <td id="T_78b05_row423_col12" class="data row423 col12" >0.021100</td>
          <td id="T_78b05_row423_col13" class="data row423 col13" >0.029100</td>
          <td id="T_78b05_row423_col14" class="data row423 col14" >0.014300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row424" class="row_heading level0 row424" >425</th>
          <td id="T_78b05_row424_col0" class="data row424 col0" >None</td>
          <td id="T_78b05_row424_col1" class="data row424 col1" >0.046300</td>
          <td id="T_78b05_row424_col2" class="data row424 col2" >-0.000900</td>
          <td id="T_78b05_row424_col3" class="data row424 col3" >0.011500</td>
          <td id="T_78b05_row424_col4" class="data row424 col4" >0.005900</td>
          <td id="T_78b05_row424_col5" class="data row424 col5" >0.011900</td>
          <td id="T_78b05_row424_col6" class="data row424 col6" >0.046800</td>
          <td id="T_78b05_row424_col7" class="data row424 col7" >0.023500</td>
          <td id="T_78b05_row424_col8" class="data row424 col8" >0.000700</td>
          <td id="T_78b05_row424_col9" class="data row424 col9" >0.028900</td>
          <td id="T_78b05_row424_col10" class="data row424 col10" >0.019200</td>
          <td id="T_78b05_row424_col11" class="data row424 col11" >0.023800</td>
          <td id="T_78b05_row424_col12" class="data row424 col12" >0.012500</td>
          <td id="T_78b05_row424_col13" class="data row424 col13" >0.050700</td>
          <td id="T_78b05_row424_col14" class="data row424 col14" >0.025300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row425" class="row_heading level0 row425" >426</th>
          <td id="T_78b05_row425_col0" class="data row425 col0" >None</td>
          <td id="T_78b05_row425_col1" class="data row425 col1" >0.046200</td>
          <td id="T_78b05_row425_col2" class="data row425 col2" >-0.066400</td>
          <td id="T_78b05_row425_col3" class="data row425 col3" >0.015100</td>
          <td id="T_78b05_row425_col4" class="data row425 col4" >-0.003900</td>
          <td id="T_78b05_row425_col5" class="data row425 col5" >0.011400</td>
          <td id="T_78b05_row425_col6" class="data row425 col6" >0.019000</td>
          <td id="T_78b05_row425_col7" class="data row425 col7" >-0.019900</td>
          <td id="T_78b05_row425_col8" class="data row425 col8" >0.000700</td>
          <td id="T_78b05_row425_col9" class="data row425 col9" >0.036600</td>
          <td id="T_78b05_row425_col10" class="data row425 col10" >0.015600</td>
          <td id="T_78b05_row425_col11" class="data row425 col11" >0.014000</td>
          <td id="T_78b05_row425_col12" class="data row425 col12" >0.012100</td>
          <td id="T_78b05_row425_col13" class="data row425 col13" >0.022900</td>
          <td id="T_78b05_row425_col14" class="data row425 col14" >0.018100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row426" class="row_heading level0 row426" >427</th>
          <td id="T_78b05_row426_col0" class="data row426 col0" >None</td>
          <td id="T_78b05_row426_col1" class="data row426 col1" >0.039200</td>
          <td id="T_78b05_row426_col2" class="data row426 col2" >-0.009500</td>
          <td id="T_78b05_row426_col3" class="data row426 col3" >-0.014200</td>
          <td id="T_78b05_row426_col4" class="data row426 col4" >-0.001200</td>
          <td id="T_78b05_row426_col5" class="data row426 col5" >0.046300</td>
          <td id="T_78b05_row426_col6" class="data row426 col6" >-0.066200</td>
          <td id="T_78b05_row426_col7" class="data row426 col7" >0.050400</td>
          <td id="T_78b05_row426_col8" class="data row426 col8" >0.006300</td>
          <td id="T_78b05_row426_col9" class="data row426 col9" >0.020300</td>
          <td id="T_78b05_row426_col10" class="data row426 col10" >0.044900</td>
          <td id="T_78b05_row426_col11" class="data row426 col11" >0.016700</td>
          <td id="T_78b05_row426_col12" class="data row426 col12" >0.046900</td>
          <td id="T_78b05_row426_col13" class="data row426 col13" >0.062400</td>
          <td id="T_78b05_row426_col14" class="data row426 col14" >0.052200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row427" class="row_heading level0 row427" >428</th>
          <td id="T_78b05_row427_col0" class="data row427 col0" >None</td>
          <td id="T_78b05_row427_col1" class="data row427 col1" >0.031100</td>
          <td id="T_78b05_row427_col2" class="data row427 col2" >-0.018500</td>
          <td id="T_78b05_row427_col3" class="data row427 col3" >-0.034300</td>
          <td id="T_78b05_row427_col4" class="data row427 col4" >0.065700</td>
          <td id="T_78b05_row427_col5" class="data row427 col5" >-0.011700</td>
          <td id="T_78b05_row427_col6" class="data row427 col6" >-0.001600</td>
          <td id="T_78b05_row427_col7" class="data row427 col7" >0.000300</td>
          <td id="T_78b05_row427_col8" class="data row427 col8" >0.014400</td>
          <td id="T_78b05_row427_col9" class="data row427 col9" >0.011400</td>
          <td id="T_78b05_row427_col10" class="data row427 col10" >0.065000</td>
          <td id="T_78b05_row427_col11" class="data row427 col11" >0.083600</td>
          <td id="T_78b05_row427_col12" class="data row427 col12" >0.011100</td>
          <td id="T_78b05_row427_col13" class="data row427 col13" >0.002300</td>
          <td id="T_78b05_row427_col14" class="data row427 col14" >0.002000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row428" class="row_heading level0 row428" >429</th>
          <td id="T_78b05_row428_col0" class="data row428 col0" >None</td>
          <td id="T_78b05_row428_col1" class="data row428 col1" >0.044700</td>
          <td id="T_78b05_row428_col2" class="data row428 col2" >-0.018500</td>
          <td id="T_78b05_row428_col3" class="data row428 col3" >0.037500</td>
          <td id="T_78b05_row428_col4" class="data row428 col4" >0.000700</td>
          <td id="T_78b05_row428_col5" class="data row428 col5" >-0.024700</td>
          <td id="T_78b05_row428_col6" class="data row428 col6" >-0.043000</td>
          <td id="T_78b05_row428_col7" class="data row428 col7" >-0.056500</td>
          <td id="T_78b05_row428_col8" class="data row428 col8" >0.000900</td>
          <td id="T_78b05_row428_col9" class="data row428 col9" >0.011300</td>
          <td id="T_78b05_row428_col10" class="data row428 col10" >0.006700</td>
          <td id="T_78b05_row428_col11" class="data row428 col11" >0.018600</td>
          <td id="T_78b05_row428_col12" class="data row428 col12" >0.024000</td>
          <td id="T_78b05_row428_col13" class="data row428 col13" >0.039100</td>
          <td id="T_78b05_row428_col14" class="data row428 col14" >0.054700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row429" class="row_heading level0 row429" >430</th>
          <td id="T_78b05_row429_col0" class="data row429 col0" >None</td>
          <td id="T_78b05_row429_col1" class="data row429 col1" >0.042700</td>
          <td id="T_78b05_row429_col2" class="data row429 col2" >-0.014600</td>
          <td id="T_78b05_row429_col3" class="data row429 col3" >0.000700</td>
          <td id="T_78b05_row429_col4" class="data row429 col4" >0.037200</td>
          <td id="T_78b05_row429_col5" class="data row429 col5" >0.021800</td>
          <td id="T_78b05_row429_col6" class="data row429 col6" >0.060800</td>
          <td id="T_78b05_row429_col7" class="data row429 col7" >0.014800</td>
          <td id="T_78b05_row429_col8" class="data row429 col8" >0.002800</td>
          <td id="T_78b05_row429_col9" class="data row429 col9" >0.015200</td>
          <td id="T_78b05_row429_col10" class="data row429 col10" >0.030100</td>
          <td id="T_78b05_row429_col11" class="data row429 col11" >0.055100</td>
          <td id="T_78b05_row429_col12" class="data row429 col12" >0.022500</td>
          <td id="T_78b05_row429_col13" class="data row429 col13" >0.064700</td>
          <td id="T_78b05_row429_col14" class="data row429 col14" >0.016600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row430" class="row_heading level0 row430" >431</th>
          <td id="T_78b05_row430_col0" class="data row430 col0" >None</td>
          <td id="T_78b05_row430_col1" class="data row430 col1" >0.036400</td>
          <td id="T_78b05_row430_col2" class="data row430 col2" >0.062200</td>
          <td id="T_78b05_row430_col3" class="data row430 col3" >0.052300</td>
          <td id="T_78b05_row430_col4" class="data row430 col4" >-0.002300</td>
          <td id="T_78b05_row430_col5" class="data row430 col5" >-0.059300</td>
          <td id="T_78b05_row430_col6" class="data row430 col6" >0.028000</td>
          <td id="T_78b05_row430_col7" class="data row430 col7" >-0.011300</td>
          <td id="T_78b05_row430_col8" class="data row430 col8" >0.009200</td>
          <td id="T_78b05_row430_col9" class="data row430 col9" >0.092000</td>
          <td id="T_78b05_row430_col10" class="data row430 col10" >0.021500</td>
          <td id="T_78b05_row430_col11" class="data row430 col11" >0.015600</td>
          <td id="T_78b05_row430_col12" class="data row430 col12" >0.058600</td>
          <td id="T_78b05_row430_col13" class="data row430 col13" >0.031900</td>
          <td id="T_78b05_row430_col14" class="data row430 col14" >0.009500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row431" class="row_heading level0 row431" >432</th>
          <td id="T_78b05_row431_col0" class="data row431 col0" >None</td>
          <td id="T_78b05_row431_col1" class="data row431 col1" >0.038300</td>
          <td id="T_78b05_row431_col2" class="data row431 col2" >0.058800</td>
          <td id="T_78b05_row431_col3" class="data row431 col3" >0.064900</td>
          <td id="T_78b05_row431_col4" class="data row431 col4" >0.058400</td>
          <td id="T_78b05_row431_col5" class="data row431 col5" >-0.065600</td>
          <td id="T_78b05_row431_col6" class="data row431 col6" >0.035900</td>
          <td id="T_78b05_row431_col7" class="data row431 col7" >-0.000200</td>
          <td id="T_78b05_row431_col8" class="data row431 col8" >0.007300</td>
          <td id="T_78b05_row431_col9" class="data row431 col9" >0.088700</td>
          <td id="T_78b05_row431_col10" class="data row431 col10" >0.034200</td>
          <td id="T_78b05_row431_col11" class="data row431 col11" >0.076300</td>
          <td id="T_78b05_row431_col12" class="data row431 col12" >0.065000</td>
          <td id="T_78b05_row431_col13" class="data row431 col13" >0.039800</td>
          <td id="T_78b05_row431_col14" class="data row431 col14" >0.001600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row432" class="row_heading level0 row432" >433</th>
          <td id="T_78b05_row432_col0" class="data row432 col0" >None</td>
          <td id="T_78b05_row432_col1" class="data row432 col1" >0.051300</td>
          <td id="T_78b05_row432_col2" class="data row432 col2" >-0.037200</td>
          <td id="T_78b05_row432_col3" class="data row432 col3" >0.035300</td>
          <td id="T_78b05_row432_col4" class="data row432 col4" >0.017400</td>
          <td id="T_78b05_row432_col5" class="data row432 col5" >0.006600</td>
          <td id="T_78b05_row432_col6" class="data row432 col6" >-0.041700</td>
          <td id="T_78b05_row432_col7" class="data row432 col7" >0.012500</td>
          <td id="T_78b05_row432_col8" class="data row432 col8" >0.005800</td>
          <td id="T_78b05_row432_col9" class="data row432 col9" >0.007300</td>
          <td id="T_78b05_row432_col10" class="data row432 col10" >0.004600</td>
          <td id="T_78b05_row432_col11" class="data row432 col11" >0.035300</td>
          <td id="T_78b05_row432_col12" class="data row432 col12" >0.007300</td>
          <td id="T_78b05_row432_col13" class="data row432 col13" >0.037800</td>
          <td id="T_78b05_row432_col14" class="data row432 col14" >0.014200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row433" class="row_heading level0 row433" >434</th>
          <td id="T_78b05_row433_col0" class="data row433 col0" >None</td>
          <td id="T_78b05_row433_col1" class="data row433 col1" >0.033100</td>
          <td id="T_78b05_row433_col2" class="data row433 col2" >0.042900</td>
          <td id="T_78b05_row433_col3" class="data row433 col3" >-0.031400</td>
          <td id="T_78b05_row433_col4" class="data row433 col4" >-0.022900</td>
          <td id="T_78b05_row433_col5" class="data row433 col5" >0.028500</td>
          <td id="T_78b05_row433_col6" class="data row433 col6" >-0.032200</td>
          <td id="T_78b05_row433_col7" class="data row433 col7" >-0.035800</td>
          <td id="T_78b05_row433_col8" class="data row433 col8" >0.012500</td>
          <td id="T_78b05_row433_col9" class="data row433 col9" >0.072700</td>
          <td id="T_78b05_row433_col10" class="data row433 col10" >0.062100</td>
          <td id="T_78b05_row433_col11" class="data row433 col11" >0.005000</td>
          <td id="T_78b05_row433_col12" class="data row433 col12" >0.029200</td>
          <td id="T_78b05_row433_col13" class="data row433 col13" >0.028300</td>
          <td id="T_78b05_row433_col14" class="data row433 col14" >0.034000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row434" class="row_heading level0 row434" >435</th>
          <td id="T_78b05_row434_col0" class="data row434 col0" >None</td>
          <td id="T_78b05_row434_col1" class="data row434 col1" >0.035300</td>
          <td id="T_78b05_row434_col2" class="data row434 col2" >0.015400</td>
          <td id="T_78b05_row434_col3" class="data row434 col3" >-0.060000</td>
          <td id="T_78b05_row434_col4" class="data row434 col4" >-0.015500</td>
          <td id="T_78b05_row434_col5" class="data row434 col5" >0.066300</td>
          <td id="T_78b05_row434_col6" class="data row434 col6" >0.031000</td>
          <td id="T_78b05_row434_col7" class="data row434 col7" >0.037500</td>
          <td id="T_78b05_row434_col8" class="data row434 col8" >0.010200</td>
          <td id="T_78b05_row434_col9" class="data row434 col9" >0.045300</td>
          <td id="T_78b05_row434_col10" class="data row434 col10" >0.090700</td>
          <td id="T_78b05_row434_col11" class="data row434 col11" >0.002400</td>
          <td id="T_78b05_row434_col12" class="data row434 col12" >0.067000</td>
          <td id="T_78b05_row434_col13" class="data row434 col13" >0.034900</td>
          <td id="T_78b05_row434_col14" class="data row434 col14" >0.039300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row435" class="row_heading level0 row435" >436</th>
          <td id="T_78b05_row435_col0" class="data row435 col0" >None</td>
          <td id="T_78b05_row435_col1" class="data row435 col1" >0.045200</td>
          <td id="T_78b05_row435_col2" class="data row435 col2" >-0.044800</td>
          <td id="T_78b05_row435_col3" class="data row435 col3" >0.036000</td>
          <td id="T_78b05_row435_col4" class="data row435 col4" >-0.048500</td>
          <td id="T_78b05_row435_col5" class="data row435 col5" >-0.005800</td>
          <td id="T_78b05_row435_col6" class="data row435 col6" >-0.066100</td>
          <td id="T_78b05_row435_col7" class="data row435 col7" >-0.030600</td>
          <td id="T_78b05_row435_col8" class="data row435 col8" >0.000300</td>
          <td id="T_78b05_row435_col9" class="data row435 col9" >0.015000</td>
          <td id="T_78b05_row435_col10" class="data row435 col10" >0.005200</td>
          <td id="T_78b05_row435_col11" class="data row435 col11" >0.030600</td>
          <td id="T_78b05_row435_col12" class="data row435 col12" >0.005200</td>
          <td id="T_78b05_row435_col13" class="data row435 col13" >0.062200</td>
          <td id="T_78b05_row435_col14" class="data row435 col14" >0.028900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row436" class="row_heading level0 row436" >437</th>
          <td id="T_78b05_row436_col0" class="data row436 col0" >None</td>
          <td id="T_78b05_row436_col1" class="data row436 col1" >0.031400</td>
          <td id="T_78b05_row436_col2" class="data row436 col2" >0.021200</td>
          <td id="T_78b05_row436_col3" class="data row436 col3" >0.006600</td>
          <td id="T_78b05_row436_col4" class="data row436 col4" >0.023400</td>
          <td id="T_78b05_row436_col5" class="data row436 col5" >-0.025300</td>
          <td id="T_78b05_row436_col6" class="data row436 col6" >0.084400</td>
          <td id="T_78b05_row436_col7" class="data row436 col7" >-0.052900</td>
          <td id="T_78b05_row436_col8" class="data row436 col8" >0.014100</td>
          <td id="T_78b05_row436_col9" class="data row436 col9" >0.051000</td>
          <td id="T_78b05_row436_col10" class="data row436 col10" >0.024100</td>
          <td id="T_78b05_row436_col11" class="data row436 col11" >0.041300</td>
          <td id="T_78b05_row436_col12" class="data row436 col12" >0.024600</td>
          <td id="T_78b05_row436_col13" class="data row436 col13" >0.088300</td>
          <td id="T_78b05_row436_col14" class="data row436 col14" >0.051100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row437" class="row_heading level0 row437" >438</th>
          <td id="T_78b05_row437_col0" class="data row437 col0" >None</td>
          <td id="T_78b05_row437_col1" class="data row437 col1" >0.047200</td>
          <td id="T_78b05_row437_col2" class="data row437 col2" >-0.035400</td>
          <td id="T_78b05_row437_col3" class="data row437 col3" >0.023400</td>
          <td id="T_78b05_row437_col4" class="data row437 col4" >-0.014000</td>
          <td id="T_78b05_row437_col5" class="data row437 col5" >0.026100</td>
          <td id="T_78b05_row437_col6" class="data row437 col6" >0.033400</td>
          <td id="T_78b05_row437_col7" class="data row437 col7" >0.023100</td>
          <td id="T_78b05_row437_col8" class="data row437 col8" >0.001700</td>
          <td id="T_78b05_row437_col9" class="data row437 col9" >0.005600</td>
          <td id="T_78b05_row437_col10" class="data row437 col10" >0.007300</td>
          <td id="T_78b05_row437_col11" class="data row437 col11" >0.003900</td>
          <td id="T_78b05_row437_col12" class="data row437 col12" >0.026800</td>
          <td id="T_78b05_row437_col13" class="data row437 col13" >0.037300</td>
          <td id="T_78b05_row437_col14" class="data row437 col14" >0.024900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row438" class="row_heading level0 row438" >439</th>
          <td id="T_78b05_row438_col0" class="data row438 col0" >None</td>
          <td id="T_78b05_row438_col1" class="data row438 col1" >0.039200</td>
          <td id="T_78b05_row438_col2" class="data row438 col2" >-0.030400</td>
          <td id="T_78b05_row438_col3" class="data row438 col3" >0.027400</td>
          <td id="T_78b05_row438_col4" class="data row438 col4" >0.034400</td>
          <td id="T_78b05_row438_col5" class="data row438 col5" >-0.042500</td>
          <td id="T_78b05_row438_col6" class="data row438 col6" >-0.023600</td>
          <td id="T_78b05_row438_col7" class="data row438 col7" >-0.061400</td>
          <td id="T_78b05_row438_col8" class="data row438 col8" >0.006300</td>
          <td id="T_78b05_row438_col9" class="data row438 col9" >0.000600</td>
          <td id="T_78b05_row438_col10" class="data row438 col10" >0.003400</td>
          <td id="T_78b05_row438_col11" class="data row438 col11" >0.052300</td>
          <td id="T_78b05_row438_col12" class="data row438 col12" >0.041800</td>
          <td id="T_78b05_row438_col13" class="data row438 col13" >0.019700</td>
          <td id="T_78b05_row438_col14" class="data row438 col14" >0.059700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row439" class="row_heading level0 row439" >440</th>
          <td id="T_78b05_row439_col0" class="data row439 col0" >None</td>
          <td id="T_78b05_row439_col1" class="data row439 col1" >0.040100</td>
          <td id="T_78b05_row439_col2" class="data row439 col2" >0.015700</td>
          <td id="T_78b05_row439_col3" class="data row439 col3" >0.063300</td>
          <td id="T_78b05_row439_col4" class="data row439 col4" >0.088000</td>
          <td id="T_78b05_row439_col5" class="data row439 col5" >0.002500</td>
          <td id="T_78b05_row439_col6" class="data row439 col6" >0.008500</td>
          <td id="T_78b05_row439_col7" class="data row439 col7" >-0.011100</td>
          <td id="T_78b05_row439_col8" class="data row439 col8" >0.005400</td>
          <td id="T_78b05_row439_col9" class="data row439 col9" >0.045500</td>
          <td id="T_78b05_row439_col10" class="data row439 col10" >0.032600</td>
          <td id="T_78b05_row439_col11" class="data row439 col11" >0.105900</td>
          <td id="T_78b05_row439_col12" class="data row439 col12" >0.003100</td>
          <td id="T_78b05_row439_col13" class="data row439 col13" >0.012400</td>
          <td id="T_78b05_row439_col14" class="data row439 col14" >0.009400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row440" class="row_heading level0 row440" >441</th>
          <td id="T_78b05_row440_col0" class="data row440 col0" >None</td>
          <td id="T_78b05_row440_col1" class="data row440 col1" >0.042000</td>
          <td id="T_78b05_row440_col2" class="data row440 col2" >-0.004400</td>
          <td id="T_78b05_row440_col3" class="data row440 col3" >-0.024800</td>
          <td id="T_78b05_row440_col4" class="data row440 col4" >0.046100</td>
          <td id="T_78b05_row440_col5" class="data row440 col5" >0.063900</td>
          <td id="T_78b05_row440_col6" class="data row440 col6" >-0.003800</td>
          <td id="T_78b05_row440_col7" class="data row440 col7" >0.009500</td>
          <td id="T_78b05_row440_col8" class="data row440 col8" >0.003600</td>
          <td id="T_78b05_row440_col9" class="data row440 col9" >0.025500</td>
          <td id="T_78b05_row440_col10" class="data row440 col10" >0.055600</td>
          <td id="T_78b05_row440_col11" class="data row440 col11" >0.064000</td>
          <td id="T_78b05_row440_col12" class="data row440 col12" >0.064500</td>
          <td id="T_78b05_row440_col13" class="data row440 col13" >0.000100</td>
          <td id="T_78b05_row440_col14" class="data row440 col14" >0.011200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row441" class="row_heading level0 row441" >442</th>
          <td id="T_78b05_row441_col0" class="data row441 col0" >None</td>
          <td id="T_78b05_row441_col1" class="data row441 col1" >0.043300</td>
          <td id="T_78b05_row441_col2" class="data row441 col2" >-0.053100</td>
          <td id="T_78b05_row441_col3" class="data row441 col3" >-0.000500</td>
          <td id="T_78b05_row441_col4" class="data row441 col4" >-0.052000</td>
          <td id="T_78b05_row441_col5" class="data row441 col5" >0.000900</td>
          <td id="T_78b05_row441_col6" class="data row441 col6" >0.026800</td>
          <td id="T_78b05_row441_col7" class="data row441 col7" >-0.012200</td>
          <td id="T_78b05_row441_col8" class="data row441 col8" >0.002300</td>
          <td id="T_78b05_row441_col9" class="data row441 col9" >0.023300</td>
          <td id="T_78b05_row441_col10" class="data row441 col10" >0.031300</td>
          <td id="T_78b05_row441_col11" class="data row441 col11" >0.034100</td>
          <td id="T_78b05_row441_col12" class="data row441 col12" >0.001500</td>
          <td id="T_78b05_row441_col13" class="data row441 col13" >0.030700</td>
          <td id="T_78b05_row441_col14" class="data row441 col14" >0.010500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row442" class="row_heading level0 row442" >443</th>
          <td id="T_78b05_row442_col0" class="data row442 col0" >None</td>
          <td id="T_78b05_row442_col1" class="data row442 col1" >0.039600</td>
          <td id="T_78b05_row442_col2" class="data row442 col2" >0.011000</td>
          <td id="T_78b05_row442_col3" class="data row442 col3" >0.056500</td>
          <td id="T_78b05_row442_col4" class="data row442 col4" >0.024300</td>
          <td id="T_78b05_row442_col5" class="data row442 col5" >-0.000000</td>
          <td id="T_78b05_row442_col6" class="data row442 col6" >-0.004200</td>
          <td id="T_78b05_row442_col7" class="data row442 col7" >-0.011500</td>
          <td id="T_78b05_row442_col8" class="data row442 col8" >0.005900</td>
          <td id="T_78b05_row442_col9" class="data row442 col9" >0.040900</td>
          <td id="T_78b05_row442_col10" class="data row442 col10" >0.025800</td>
          <td id="T_78b05_row442_col11" class="data row442 col11" >0.042200</td>
          <td id="T_78b05_row442_col12" class="data row442 col12" >0.000600</td>
          <td id="T_78b05_row442_col13" class="data row442 col13" >0.000300</td>
          <td id="T_78b05_row442_col14" class="data row442 col14" >0.009700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row443" class="row_heading level0 row443" >444</th>
          <td id="T_78b05_row443_col0" class="data row443 col0" >None</td>
          <td id="T_78b05_row443_col1" class="data row443 col1" >0.043900</td>
          <td id="T_78b05_row443_col2" class="data row443 col2" >-0.039200</td>
          <td id="T_78b05_row443_col3" class="data row443 col3" >0.021100</td>
          <td id="T_78b05_row443_col4" class="data row443 col4" >0.005900</td>
          <td id="T_78b05_row443_col5" class="data row443 col5" >-0.013600</td>
          <td id="T_78b05_row443_col6" class="data row443 col6" >-0.026600</td>
          <td id="T_78b05_row443_col7" class="data row443 col7" >-0.022700</td>
          <td id="T_78b05_row443_col8" class="data row443 col8" >0.001600</td>
          <td id="T_78b05_row443_col9" class="data row443 col9" >0.009400</td>
          <td id="T_78b05_row443_col10" class="data row443 col10" >0.009700</td>
          <td id="T_78b05_row443_col11" class="data row443 col11" >0.023800</td>
          <td id="T_78b05_row443_col12" class="data row443 col12" >0.013000</td>
          <td id="T_78b05_row443_col13" class="data row443 col13" >0.022700</td>
          <td id="T_78b05_row443_col14" class="data row443 col14" >0.021000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row444" class="row_heading level0 row444" >445</th>
          <td id="T_78b05_row444_col0" class="data row444 col0" >None</td>
          <td id="T_78b05_row444_col1" class="data row444 col1" >0.033100</td>
          <td id="T_78b05_row444_col2" class="data row444 col2" >-0.011800</td>
          <td id="T_78b05_row444_col3" class="data row444 col3" >-0.019400</td>
          <td id="T_78b05_row444_col4" class="data row444 col4" >-0.009400</td>
          <td id="T_78b05_row444_col5" class="data row444 col5" >-0.022100</td>
          <td id="T_78b05_row444_col6" class="data row444 col6" >-0.061600</td>
          <td id="T_78b05_row444_col7" class="data row444 col7" >-0.049500</td>
          <td id="T_78b05_row444_col8" class="data row444 col8" >0.012400</td>
          <td id="T_78b05_row444_col9" class="data row444 col9" >0.018100</td>
          <td id="T_78b05_row444_col10" class="data row444 col10" >0.050200</td>
          <td id="T_78b05_row444_col11" class="data row444 col11" >0.008500</td>
          <td id="T_78b05_row444_col12" class="data row444 col12" >0.021500</td>
          <td id="T_78b05_row444_col13" class="data row444 col13" >0.057700</td>
          <td id="T_78b05_row444_col14" class="data row444 col14" >0.047700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row445" class="row_heading level0 row445" >446</th>
          <td id="T_78b05_row445_col0" class="data row445 col0" >PC1</td>
          <td id="T_78b05_row445_col1" class="data row445 col1" >0.026200</td>
          <td id="T_78b05_row445_col2" class="data row445 col2" >0.013700</td>
          <td id="T_78b05_row445_col3" class="data row445 col3" >-0.054500</td>
          <td id="T_78b05_row445_col4" class="data row445 col4" >-0.062400</td>
          <td id="T_78b05_row445_col5" class="data row445 col5" >0.023600</td>
          <td id="T_78b05_row445_col6" class="data row445 col6" >-0.019300</td>
          <td id="T_78b05_row445_col7" class="data row445 col7" >-0.053700</td>
          <td id="T_78b05_row445_col8" class="data row445 col8" >0.019400</td>
          <td id="T_78b05_row445_col9" class="data row445 col9" >0.043500</td>
          <td id="T_78b05_row445_col10" class="data row445 col10" >0.085200</td>
          <td id="T_78b05_row445_col11" class="data row445 col11" >0.044500</td>
          <td id="T_78b05_row445_col12" class="data row445 col12" >0.024200</td>
          <td id="T_78b05_row445_col13" class="data row445 col13" >0.015400</td>
          <td id="T_78b05_row445_col14" class="data row445 col14" >0.052000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row446" class="row_heading level0 row446" >447</th>
          <td id="T_78b05_row446_col0" class="data row446 col0" >None</td>
          <td id="T_78b05_row446_col1" class="data row446 col1" >0.026700</td>
          <td id="T_78b05_row446_col2" class="data row446 col2" >0.017900</td>
          <td id="T_78b05_row446_col3" class="data row446 col3" >-0.086000</td>
          <td id="T_78b05_row446_col4" class="data row446 col4" >-0.054000</td>
          <td id="T_78b05_row446_col5" class="data row446 col5" >-0.031000</td>
          <td id="T_78b05_row446_col6" class="data row446 col6" >-0.021000</td>
          <td id="T_78b05_row446_col7" class="data row446 col7" >-0.081600</td>
          <td id="T_78b05_row446_col8" class="data row446 col8" >0.018900</td>
          <td id="T_78b05_row446_col9" class="data row446 col9" >0.047800</td>
          <td id="T_78b05_row446_col10" class="data row446 col10" >0.116800</td>
          <td id="T_78b05_row446_col11" class="data row446 col11" >0.036100</td>
          <td id="T_78b05_row446_col12" class="data row446 col12" >0.030300</td>
          <td id="T_78b05_row446_col13" class="data row446 col13" >0.017100</td>
          <td id="T_78b05_row446_col14" class="data row446 col14" >0.079800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row447" class="row_heading level0 row447" >448</th>
          <td id="T_78b05_row447_col0" class="data row447 col0" >None</td>
          <td id="T_78b05_row447_col1" class="data row447 col1" >0.031600</td>
          <td id="T_78b05_row447_col2" class="data row447 col2" >0.020600</td>
          <td id="T_78b05_row447_col3" class="data row447 col3" >-0.037000</td>
          <td id="T_78b05_row447_col4" class="data row447 col4" >0.046100</td>
          <td id="T_78b05_row447_col5" class="data row447 col5" >-0.044500</td>
          <td id="T_78b05_row447_col6" class="data row447 col6" >-0.048100</td>
          <td id="T_78b05_row447_col7" class="data row447 col7" >0.049800</td>
          <td id="T_78b05_row447_col8" class="data row447 col8" >0.013900</td>
          <td id="T_78b05_row447_col9" class="data row447 col9" >0.050400</td>
          <td id="T_78b05_row447_col10" class="data row447 col10" >0.067700</td>
          <td id="T_78b05_row447_col11" class="data row447 col11" >0.064000</td>
          <td id="T_78b05_row447_col12" class="data row447 col12" >0.043800</td>
          <td id="T_78b05_row447_col13" class="data row447 col13" >0.044200</td>
          <td id="T_78b05_row447_col14" class="data row447 col14" >0.051500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row448" class="row_heading level0 row448" >449</th>
          <td id="T_78b05_row448_col0" class="data row448 col0" >None</td>
          <td id="T_78b05_row448_col1" class="data row448 col1" >0.042400</td>
          <td id="T_78b05_row448_col2" class="data row448 col2" >0.013500</td>
          <td id="T_78b05_row448_col3" class="data row448 col3" >0.007400</td>
          <td id="T_78b05_row448_col4" class="data row448 col4" >-0.000500</td>
          <td id="T_78b05_row448_col5" class="data row448 col5" >0.027600</td>
          <td id="T_78b05_row448_col6" class="data row448 col6" >0.000600</td>
          <td id="T_78b05_row448_col7" class="data row448 col7" >-0.019400</td>
          <td id="T_78b05_row448_col8" class="data row448 col8" >0.003100</td>
          <td id="T_78b05_row448_col9" class="data row448 col9" >0.043400</td>
          <td id="T_78b05_row448_col10" class="data row448 col10" >0.023300</td>
          <td id="T_78b05_row448_col11" class="data row448 col11" >0.017400</td>
          <td id="T_78b05_row448_col12" class="data row448 col12" >0.028300</td>
          <td id="T_78b05_row448_col13" class="data row448 col13" >0.004500</td>
          <td id="T_78b05_row448_col14" class="data row448 col14" >0.017700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row449" class="row_heading level0 row449" >450</th>
          <td id="T_78b05_row449_col0" class="data row449 col0" >None</td>
          <td id="T_78b05_row449_col1" class="data row449 col1" >0.037400</td>
          <td id="T_78b05_row449_col2" class="data row449 col2" >0.041000</td>
          <td id="T_78b05_row449_col3" class="data row449 col3" >0.001800</td>
          <td id="T_78b05_row449_col4" class="data row449 col4" >-0.046600</td>
          <td id="T_78b05_row449_col5" class="data row449 col5" >-0.029500</td>
          <td id="T_78b05_row449_col6" class="data row449 col6" >0.022900</td>
          <td id="T_78b05_row449_col7" class="data row449 col7" >0.001300</td>
          <td id="T_78b05_row449_col8" class="data row449 col8" >0.008100</td>
          <td id="T_78b05_row449_col9" class="data row449 col9" >0.070900</td>
          <td id="T_78b05_row449_col10" class="data row449 col10" >0.028900</td>
          <td id="T_78b05_row449_col11" class="data row449 col11" >0.028700</td>
          <td id="T_78b05_row449_col12" class="data row449 col12" >0.028800</td>
          <td id="T_78b05_row449_col13" class="data row449 col13" >0.026800</td>
          <td id="T_78b05_row449_col14" class="data row449 col14" >0.003100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row450" class="row_heading level0 row450" >451</th>
          <td id="T_78b05_row450_col0" class="data row450 col0" >None</td>
          <td id="T_78b05_row450_col1" class="data row450 col1" >0.039500</td>
          <td id="T_78b05_row450_col2" class="data row450 col2" >0.050400</td>
          <td id="T_78b05_row450_col3" class="data row450 col3" >0.022400</td>
          <td id="T_78b05_row450_col4" class="data row450 col4" >-0.039700</td>
          <td id="T_78b05_row450_col5" class="data row450 col5" >0.008400</td>
          <td id="T_78b05_row450_col6" class="data row450 col6" >-0.042000</td>
          <td id="T_78b05_row450_col7" class="data row450 col7" >0.027900</td>
          <td id="T_78b05_row450_col8" class="data row450 col8" >0.006000</td>
          <td id="T_78b05_row450_col9" class="data row450 col9" >0.080200</td>
          <td id="T_78b05_row450_col10" class="data row450 col10" >0.008300</td>
          <td id="T_78b05_row450_col11" class="data row450 col11" >0.021700</td>
          <td id="T_78b05_row450_col12" class="data row450 col12" >0.009000</td>
          <td id="T_78b05_row450_col13" class="data row450 col13" >0.038100</td>
          <td id="T_78b05_row450_col14" class="data row450 col14" >0.029700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row451" class="row_heading level0 row451" >452</th>
          <td id="T_78b05_row451_col0" class="data row451 col0" >None</td>
          <td id="T_78b05_row451_col1" class="data row451 col1" >0.049300</td>
          <td id="T_78b05_row451_col2" class="data row451 col2" >-0.058300</td>
          <td id="T_78b05_row451_col3" class="data row451 col3" >0.028800</td>
          <td id="T_78b05_row451_col4" class="data row451 col4" >0.038200</td>
          <td id="T_78b05_row451_col5" class="data row451 col5" >0.046000</td>
          <td id="T_78b05_row451_col6" class="data row451 col6" >0.043500</td>
          <td id="T_78b05_row451_col7" class="data row451 col7" >0.022200</td>
          <td id="T_78b05_row451_col8" class="data row451 col8" >0.003800</td>
          <td id="T_78b05_row451_col9" class="data row451 col9" >0.028400</td>
          <td id="T_78b05_row451_col10" class="data row451 col10" >0.001900</td>
          <td id="T_78b05_row451_col11" class="data row451 col11" >0.056100</td>
          <td id="T_78b05_row451_col12" class="data row451 col12" >0.046700</td>
          <td id="T_78b05_row451_col13" class="data row451 col13" >0.047400</td>
          <td id="T_78b05_row451_col14" class="data row451 col14" >0.023900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row452" class="row_heading level0 row452" >453</th>
          <td id="T_78b05_row452_col0" class="data row452 col0" >None</td>
          <td id="T_78b05_row452_col1" class="data row452 col1" >0.037600</td>
          <td id="T_78b05_row452_col2" class="data row452 col2" >0.043600</td>
          <td id="T_78b05_row452_col3" class="data row452 col3" >0.002600</td>
          <td id="T_78b05_row452_col4" class="data row452 col4" >-0.008100</td>
          <td id="T_78b05_row452_col5" class="data row452 col5" >-0.056900</td>
          <td id="T_78b05_row452_col6" class="data row452 col6" >-0.005100</td>
          <td id="T_78b05_row452_col7" class="data row452 col7" >0.051800</td>
          <td id="T_78b05_row452_col8" class="data row452 col8" >0.007900</td>
          <td id="T_78b05_row452_col9" class="data row452 col9" >0.073500</td>
          <td id="T_78b05_row452_col10" class="data row452 col10" >0.028100</td>
          <td id="T_78b05_row452_col11" class="data row452 col11" >0.009800</td>
          <td id="T_78b05_row452_col12" class="data row452 col12" >0.056300</td>
          <td id="T_78b05_row452_col13" class="data row452 col13" >0.001200</td>
          <td id="T_78b05_row452_col14" class="data row452 col14" >0.053600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row453" class="row_heading level0 row453" >454</th>
          <td id="T_78b05_row453_col0" class="data row453 col0" >None</td>
          <td id="T_78b05_row453_col1" class="data row453 col1" >0.033000</td>
          <td id="T_78b05_row453_col2" class="data row453 col2" >0.006600</td>
          <td id="T_78b05_row453_col3" class="data row453 col3" >-0.044500</td>
          <td id="T_78b05_row453_col4" class="data row453 col4" >0.096800</td>
          <td id="T_78b05_row453_col5" class="data row453 col5" >-0.007600</td>
          <td id="T_78b05_row453_col6" class="data row453 col6" >-0.021300</td>
          <td id="T_78b05_row453_col7" class="data row453 col7" >-0.027500</td>
          <td id="T_78b05_row453_col8" class="data row453 col8" >0.012500</td>
          <td id="T_78b05_row453_col9" class="data row453 col9" >0.036500</td>
          <td id="T_78b05_row453_col10" class="data row453 col10" >0.075200</td>
          <td id="T_78b05_row453_col11" class="data row453 col11" >0.114700</td>
          <td id="T_78b05_row453_col12" class="data row453 col12" >0.007000</td>
          <td id="T_78b05_row453_col13" class="data row453 col13" >0.017400</td>
          <td id="T_78b05_row453_col14" class="data row453 col14" >0.025800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row454" class="row_heading level0 row454" >455</th>
          <td id="T_78b05_row454_col0" class="data row454 col0" >PC1</td>
          <td id="T_78b05_row454_col1" class="data row454 col1" >0.026600</td>
          <td id="T_78b05_row454_col2" class="data row454 col2" >0.052100</td>
          <td id="T_78b05_row454_col3" class="data row454 col3" >-0.089500</td>
          <td id="T_78b05_row454_col4" class="data row454 col4" >-0.044600</td>
          <td id="T_78b05_row454_col5" class="data row454 col5" >-0.072800</td>
          <td id="T_78b05_row454_col6" class="data row454 col6" >0.029300</td>
          <td id="T_78b05_row454_col7" class="data row454 col7" >0.061000</td>
          <td id="T_78b05_row454_col8" class="data row454 col8" >0.019000</td>
          <td id="T_78b05_row454_col9" class="data row454 col9" >0.082000</td>
          <td id="T_78b05_row454_col10" class="data row454 col10" >0.120200</td>
          <td id="T_78b05_row454_col11" class="data row454 col11" >0.026700</td>
          <td id="T_78b05_row454_col12" class="data row454 col12" >0.072200</td>
          <td id="T_78b05_row454_col13" class="data row454 col13" >0.033200</td>
          <td id="T_78b05_row454_col14" class="data row454 col14" >0.062800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row455" class="row_heading level0 row455" >456</th>
          <td id="T_78b05_row455_col0" class="data row455 col0" >None</td>
          <td id="T_78b05_row455_col1" class="data row455 col1" >0.031300</td>
          <td id="T_78b05_row455_col2" class="data row455 col2" >0.075100</td>
          <td id="T_78b05_row455_col3" class="data row455 col3" >0.035600</td>
          <td id="T_78b05_row455_col4" class="data row455 col4" >0.073600</td>
          <td id="T_78b05_row455_col5" class="data row455 col5" >-0.010100</td>
          <td id="T_78b05_row455_col6" class="data row455 col6" >0.044900</td>
          <td id="T_78b05_row455_col7" class="data row455 col7" >-0.005500</td>
          <td id="T_78b05_row455_col8" class="data row455 col8" >0.014200</td>
          <td id="T_78b05_row455_col9" class="data row455 col9" >0.104900</td>
          <td id="T_78b05_row455_col10" class="data row455 col10" >0.004800</td>
          <td id="T_78b05_row455_col11" class="data row455 col11" >0.091500</td>
          <td id="T_78b05_row455_col12" class="data row455 col12" >0.009400</td>
          <td id="T_78b05_row455_col13" class="data row455 col13" >0.048800</td>
          <td id="T_78b05_row455_col14" class="data row455 col14" >0.003700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row456" class="row_heading level0 row456" >457</th>
          <td id="T_78b05_row456_col0" class="data row456 col0" >None</td>
          <td id="T_78b05_row456_col1" class="data row456 col1" >0.038900</td>
          <td id="T_78b05_row456_col2" class="data row456 col2" >0.024800</td>
          <td id="T_78b05_row456_col3" class="data row456 col3" >0.023300</td>
          <td id="T_78b05_row456_col4" class="data row456 col4" >0.030800</td>
          <td id="T_78b05_row456_col5" class="data row456 col5" >-0.046700</td>
          <td id="T_78b05_row456_col6" class="data row456 col6" >0.042600</td>
          <td id="T_78b05_row456_col7" class="data row456 col7" >0.062800</td>
          <td id="T_78b05_row456_col8" class="data row456 col8" >0.006600</td>
          <td id="T_78b05_row456_col9" class="data row456 col9" >0.054700</td>
          <td id="T_78b05_row456_col10" class="data row456 col10" >0.007500</td>
          <td id="T_78b05_row456_col11" class="data row456 col11" >0.048700</td>
          <td id="T_78b05_row456_col12" class="data row456 col12" >0.046000</td>
          <td id="T_78b05_row456_col13" class="data row456 col13" >0.046500</td>
          <td id="T_78b05_row456_col14" class="data row456 col14" >0.064600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row457" class="row_heading level0 row457" >458</th>
          <td id="T_78b05_row457_col0" class="data row457 col0" >None</td>
          <td id="T_78b05_row457_col1" class="data row457 col1" >0.034500</td>
          <td id="T_78b05_row457_col2" class="data row457 col2" >0.027500</td>
          <td id="T_78b05_row457_col3" class="data row457 col3" >-0.001100</td>
          <td id="T_78b05_row457_col4" class="data row457 col4" >0.003000</td>
          <td id="T_78b05_row457_col5" class="data row457 col5" >0.015200</td>
          <td id="T_78b05_row457_col6" class="data row457 col6" >-0.033500</td>
          <td id="T_78b05_row457_col7" class="data row457 col7" >0.057200</td>
          <td id="T_78b05_row457_col8" class="data row457 col8" >0.011100</td>
          <td id="T_78b05_row457_col9" class="data row457 col9" >0.057300</td>
          <td id="T_78b05_row457_col10" class="data row457 col10" >0.031800</td>
          <td id="T_78b05_row457_col11" class="data row457 col11" >0.020900</td>
          <td id="T_78b05_row457_col12" class="data row457 col12" >0.015900</td>
          <td id="T_78b05_row457_col13" class="data row457 col13" >0.029600</td>
          <td id="T_78b05_row457_col14" class="data row457 col14" >0.058900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row458" class="row_heading level0 row458" >459</th>
          <td id="T_78b05_row458_col0" class="data row458 col0" >None</td>
          <td id="T_78b05_row458_col1" class="data row458 col1" >0.029500</td>
          <td id="T_78b05_row458_col2" class="data row458 col2" >0.059200</td>
          <td id="T_78b05_row458_col3" class="data row458 col3" >0.014300</td>
          <td id="T_78b05_row458_col4" class="data row458 col4" >-0.027800</td>
          <td id="T_78b05_row458_col5" class="data row458 col5" >-0.031500</td>
          <td id="T_78b05_row458_col6" class="data row458 col6" >-0.019300</td>
          <td id="T_78b05_row458_col7" class="data row458 col7" >0.049600</td>
          <td id="T_78b05_row458_col8" class="data row458 col8" >0.016000</td>
          <td id="T_78b05_row458_col9" class="data row458 col9" >0.089000</td>
          <td id="T_78b05_row458_col10" class="data row458 col10" >0.016400</td>
          <td id="T_78b05_row458_col11" class="data row458 col11" >0.009900</td>
          <td id="T_78b05_row458_col12" class="data row458 col12" >0.030800</td>
          <td id="T_78b05_row458_col13" class="data row458 col13" >0.015400</td>
          <td id="T_78b05_row458_col14" class="data row458 col14" >0.051400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row459" class="row_heading level0 row459" >460</th>
          <td id="T_78b05_row459_col0" class="data row459 col0" >None</td>
          <td id="T_78b05_row459_col1" class="data row459 col1" >0.041300</td>
          <td id="T_78b05_row459_col2" class="data row459 col2" >-0.002400</td>
          <td id="T_78b05_row459_col3" class="data row459 col3" >0.059200</td>
          <td id="T_78b05_row459_col4" class="data row459 col4" >0.054100</td>
          <td id="T_78b05_row459_col5" class="data row459 col5" >-0.045800</td>
          <td id="T_78b05_row459_col6" class="data row459 col6" >0.063700</td>
          <td id="T_78b05_row459_col7" class="data row459 col7" >-0.009800</td>
          <td id="T_78b05_row459_col8" class="data row459 col8" >0.004200</td>
          <td id="T_78b05_row459_col9" class="data row459 col9" >0.027400</td>
          <td id="T_78b05_row459_col10" class="data row459 col10" >0.028500</td>
          <td id="T_78b05_row459_col11" class="data row459 col11" >0.072000</td>
          <td id="T_78b05_row459_col12" class="data row459 col12" >0.045100</td>
          <td id="T_78b05_row459_col13" class="data row459 col13" >0.067600</td>
          <td id="T_78b05_row459_col14" class="data row459 col14" >0.008000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row460" class="row_heading level0 row460" >461</th>
          <td id="T_78b05_row460_col0" class="data row460 col0" >None</td>
          <td id="T_78b05_row460_col1" class="data row460 col1" >0.040900</td>
          <td id="T_78b05_row460_col2" class="data row460 col2" >0.001400</td>
          <td id="T_78b05_row460_col3" class="data row460 col3" >0.083700</td>
          <td id="T_78b05_row460_col4" class="data row460 col4" >0.032600</td>
          <td id="T_78b05_row460_col5" class="data row460 col5" >-0.002700</td>
          <td id="T_78b05_row460_col6" class="data row460 col6" >0.012700</td>
          <td id="T_78b05_row460_col7" class="data row460 col7" >0.037700</td>
          <td id="T_78b05_row460_col8" class="data row460 col8" >0.004700</td>
          <td id="T_78b05_row460_col9" class="data row460 col9" >0.031200</td>
          <td id="T_78b05_row460_col10" class="data row460 col10" >0.052900</td>
          <td id="T_78b05_row460_col11" class="data row460 col11" >0.050500</td>
          <td id="T_78b05_row460_col12" class="data row460 col12" >0.002100</td>
          <td id="T_78b05_row460_col13" class="data row460 col13" >0.016600</td>
          <td id="T_78b05_row460_col14" class="data row460 col14" >0.039400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row461" class="row_heading level0 row461" >462</th>
          <td id="T_78b05_row461_col0" class="data row461 col0" >None</td>
          <td id="T_78b05_row461_col1" class="data row461 col1" >0.045800</td>
          <td id="T_78b05_row461_col2" class="data row461 col2" >-0.034000</td>
          <td id="T_78b05_row461_col3" class="data row461 col3" >0.001800</td>
          <td id="T_78b05_row461_col4" class="data row461 col4" >-0.011000</td>
          <td id="T_78b05_row461_col5" class="data row461 col5" >0.050700</td>
          <td id="T_78b05_row461_col6" class="data row461 col6" >-0.010400</td>
          <td id="T_78b05_row461_col7" class="data row461 col7" >0.000400</td>
          <td id="T_78b05_row461_col8" class="data row461 col8" >0.000300</td>
          <td id="T_78b05_row461_col9" class="data row461 col9" >0.004200</td>
          <td id="T_78b05_row461_col10" class="data row461 col10" >0.028900</td>
          <td id="T_78b05_row461_col11" class="data row461 col11" >0.006900</td>
          <td id="T_78b05_row461_col12" class="data row461 col12" >0.051300</td>
          <td id="T_78b05_row461_col13" class="data row461 col13" >0.006500</td>
          <td id="T_78b05_row461_col14" class="data row461 col14" >0.002200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row462" class="row_heading level0 row462" >463</th>
          <td id="T_78b05_row462_col0" class="data row462 col0" >None</td>
          <td id="T_78b05_row462_col1" class="data row462 col1" >0.029900</td>
          <td id="T_78b05_row462_col2" class="data row462 col2" >0.036200</td>
          <td id="T_78b05_row462_col3" class="data row462 col3" >-0.013500</td>
          <td id="T_78b05_row462_col4" class="data row462 col4" >-0.002900</td>
          <td id="T_78b05_row462_col5" class="data row462 col5" >0.042200</td>
          <td id="T_78b05_row462_col6" class="data row462 col6" >-0.013400</td>
          <td id="T_78b05_row462_col7" class="data row462 col7" >0.034700</td>
          <td id="T_78b05_row462_col8" class="data row462 col8" >0.015600</td>
          <td id="T_78b05_row462_col9" class="data row462 col9" >0.066000</td>
          <td id="T_78b05_row462_col10" class="data row462 col10" >0.044300</td>
          <td id="T_78b05_row462_col11" class="data row462 col11" >0.015000</td>
          <td id="T_78b05_row462_col12" class="data row462 col12" >0.042900</td>
          <td id="T_78b05_row462_col13" class="data row462 col13" >0.009600</td>
          <td id="T_78b05_row462_col14" class="data row462 col14" >0.036400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row463" class="row_heading level0 row463" >464</th>
          <td id="T_78b05_row463_col0" class="data row463 col0" >None</td>
          <td id="T_78b05_row463_col1" class="data row463 col1" >0.031400</td>
          <td id="T_78b05_row463_col2" class="data row463 col2" >-0.058900</td>
          <td id="T_78b05_row463_col3" class="data row463 col3" >-0.033800</td>
          <td id="T_78b05_row463_col4" class="data row463 col4" >-0.042800</td>
          <td id="T_78b05_row463_col5" class="data row463 col5" >-0.032900</td>
          <td id="T_78b05_row463_col6" class="data row463 col6" >0.092300</td>
          <td id="T_78b05_row463_col7" class="data row463 col7" >-0.079600</td>
          <td id="T_78b05_row463_col8" class="data row463 col8" >0.014100</td>
          <td id="T_78b05_row463_col9" class="data row463 col9" >0.029000</td>
          <td id="T_78b05_row463_col10" class="data row463 col10" >0.064500</td>
          <td id="T_78b05_row463_col11" class="data row463 col11" >0.024900</td>
          <td id="T_78b05_row463_col12" class="data row463 col12" >0.032200</td>
          <td id="T_78b05_row463_col13" class="data row463 col13" >0.096200</td>
          <td id="T_78b05_row463_col14" class="data row463 col14" >0.077800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row464" class="row_heading level0 row464" >465</th>
          <td id="T_78b05_row464_col0" class="data row464 col0" >None</td>
          <td id="T_78b05_row464_col1" class="data row464 col1" >0.044000</td>
          <td id="T_78b05_row464_col2" class="data row464 col2" >0.033000</td>
          <td id="T_78b05_row464_col3" class="data row464 col3" >0.034400</td>
          <td id="T_78b05_row464_col4" class="data row464 col4" >-0.018100</td>
          <td id="T_78b05_row464_col5" class="data row464 col5" >0.030800</td>
          <td id="T_78b05_row464_col6" class="data row464 col6" >0.045300</td>
          <td id="T_78b05_row464_col7" class="data row464 col7" >0.057300</td>
          <td id="T_78b05_row464_col8" class="data row464 col8" >0.001500</td>
          <td id="T_78b05_row464_col9" class="data row464 col9" >0.062800</td>
          <td id="T_78b05_row464_col10" class="data row464 col10" >0.003700</td>
          <td id="T_78b05_row464_col11" class="data row464 col11" >0.000200</td>
          <td id="T_78b05_row464_col12" class="data row464 col12" >0.031500</td>
          <td id="T_78b05_row464_col13" class="data row464 col13" >0.049200</td>
          <td id="T_78b05_row464_col14" class="data row464 col14" >0.059000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row465" class="row_heading level0 row465" >466</th>
          <td id="T_78b05_row465_col0" class="data row465 col0" >None</td>
          <td id="T_78b05_row465_col1" class="data row465 col1" >0.035100</td>
          <td id="T_78b05_row465_col2" class="data row465 col2" >-0.029600</td>
          <td id="T_78b05_row465_col3" class="data row465 col3" >-0.043200</td>
          <td id="T_78b05_row465_col4" class="data row465 col4" >0.026900</td>
          <td id="T_78b05_row465_col5" class="data row465 col5" >-0.024200</td>
          <td id="T_78b05_row465_col6" class="data row465 col6" >-0.077200</td>
          <td id="T_78b05_row465_col7" class="data row465 col7" >0.022700</td>
          <td id="T_78b05_row465_col8" class="data row465 col8" >0.010400</td>
          <td id="T_78b05_row465_col9" class="data row465 col9" >0.000300</td>
          <td id="T_78b05_row465_col10" class="data row465 col10" >0.073900</td>
          <td id="T_78b05_row465_col11" class="data row465 col11" >0.044800</td>
          <td id="T_78b05_row465_col12" class="data row465 col12" >0.023600</td>
          <td id="T_78b05_row465_col13" class="data row465 col13" >0.073300</td>
          <td id="T_78b05_row465_col14" class="data row465 col14" >0.024500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row466" class="row_heading level0 row466" >467</th>
          <td id="T_78b05_row466_col0" class="data row466 col0" >None</td>
          <td id="T_78b05_row466_col1" class="data row466 col1" >0.035500</td>
          <td id="T_78b05_row466_col2" class="data row466 col2" >0.007400</td>
          <td id="T_78b05_row466_col3" class="data row466 col3" >-0.051100</td>
          <td id="T_78b05_row466_col4" class="data row466 col4" >0.004500</td>
          <td id="T_78b05_row466_col5" class="data row466 col5" >0.018500</td>
          <td id="T_78b05_row466_col6" class="data row466 col6" >-0.021200</td>
          <td id="T_78b05_row466_col7" class="data row466 col7" >0.015200</td>
          <td id="T_78b05_row466_col8" class="data row466 col8" >0.010100</td>
          <td id="T_78b05_row466_col9" class="data row466 col9" >0.037300</td>
          <td id="T_78b05_row466_col10" class="data row466 col10" >0.081800</td>
          <td id="T_78b05_row466_col11" class="data row466 col11" >0.022400</td>
          <td id="T_78b05_row466_col12" class="data row466 col12" >0.019200</td>
          <td id="T_78b05_row466_col13" class="data row466 col13" >0.017300</td>
          <td id="T_78b05_row466_col14" class="data row466 col14" >0.016900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row467" class="row_heading level0 row467" >468</th>
          <td id="T_78b05_row467_col0" class="data row467 col0" >PC1</td>
          <td id="T_78b05_row467_col1" class="data row467 col1" >0.025600</td>
          <td id="T_78b05_row467_col2" class="data row467 col2" >0.068800</td>
          <td id="T_78b05_row467_col3" class="data row467 col3" >-0.011800</td>
          <td id="T_78b05_row467_col4" class="data row467 col4" >-0.003100</td>
          <td id="T_78b05_row467_col5" class="data row467 col5" >-0.027400</td>
          <td id="T_78b05_row467_col6" class="data row467 col6" >0.006500</td>
          <td id="T_78b05_row467_col7" class="data row467 col7" >-0.078000</td>
          <td id="T_78b05_row467_col8" class="data row467 col8" >0.019900</td>
          <td id="T_78b05_row467_col9" class="data row467 col9" >0.098600</td>
          <td id="T_78b05_row467_col10" class="data row467 col10" >0.042500</td>
          <td id="T_78b05_row467_col11" class="data row467 col11" >0.014800</td>
          <td id="T_78b05_row467_col12" class="data row467 col12" >0.026700</td>
          <td id="T_78b05_row467_col13" class="data row467 col13" >0.010400</td>
          <td id="T_78b05_row467_col14" class="data row467 col14" >0.076300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row468" class="row_heading level0 row468" >469</th>
          <td id="T_78b05_row468_col0" class="data row468 col0" >None</td>
          <td id="T_78b05_row468_col1" class="data row468 col1" >0.039500</td>
          <td id="T_78b05_row468_col2" class="data row468 col2" >-0.054600</td>
          <td id="T_78b05_row468_col3" class="data row468 col3" >-0.024800</td>
          <td id="T_78b05_row468_col4" class="data row468 col4" >-0.049500</td>
          <td id="T_78b05_row468_col5" class="data row468 col5" >-0.085000</td>
          <td id="T_78b05_row468_col6" class="data row468 col6" >0.062000</td>
          <td id="T_78b05_row468_col7" class="data row468 col7" >0.034400</td>
          <td id="T_78b05_row468_col8" class="data row468 col8" >0.006000</td>
          <td id="T_78b05_row468_col9" class="data row468 col9" >0.024800</td>
          <td id="T_78b05_row468_col10" class="data row468 col10" >0.055600</td>
          <td id="T_78b05_row468_col11" class="data row468 col11" >0.031600</td>
          <td id="T_78b05_row468_col12" class="data row468 col12" >0.084400</td>
          <td id="T_78b05_row468_col13" class="data row468 col13" >0.065900</td>
          <td id="T_78b05_row468_col14" class="data row468 col14" >0.036200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row469" class="row_heading level0 row469" >470</th>
          <td id="T_78b05_row469_col0" class="data row469 col0" >None</td>
          <td id="T_78b05_row469_col1" class="data row469 col1" >0.036300</td>
          <td id="T_78b05_row469_col2" class="data row469 col2" >0.012500</td>
          <td id="T_78b05_row469_col3" class="data row469 col3" >-0.013000</td>
          <td id="T_78b05_row469_col4" class="data row469 col4" >-0.025100</td>
          <td id="T_78b05_row469_col5" class="data row469 col5" >0.041200</td>
          <td id="T_78b05_row469_col6" class="data row469 col6" >-0.043700</td>
          <td id="T_78b05_row469_col7" class="data row469 col7" >0.038600</td>
          <td id="T_78b05_row469_col8" class="data row469 col8" >0.009200</td>
          <td id="T_78b05_row469_col9" class="data row469 col9" >0.042400</td>
          <td id="T_78b05_row469_col10" class="data row469 col10" >0.043800</td>
          <td id="T_78b05_row469_col11" class="data row469 col11" >0.007200</td>
          <td id="T_78b05_row469_col12" class="data row469 col12" >0.041800</td>
          <td id="T_78b05_row469_col13" class="data row469 col13" >0.039800</td>
          <td id="T_78b05_row469_col14" class="data row469 col14" >0.040300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row470" class="row_heading level0 row470" >471</th>
          <td id="T_78b05_row470_col0" class="data row470 col0" >PC1</td>
          <td id="T_78b05_row470_col1" class="data row470 col1" >0.025000</td>
          <td id="T_78b05_row470_col2" class="data row470 col2" >0.005500</td>
          <td id="T_78b05_row470_col3" class="data row470 col3" >-0.083500</td>
          <td id="T_78b05_row470_col4" class="data row470 col4" >-0.016500</td>
          <td id="T_78b05_row470_col5" class="data row470 col5" >-0.063500</td>
          <td id="T_78b05_row470_col6" class="data row470 col6" >-0.055700</td>
          <td id="T_78b05_row470_col7" class="data row470 col7" >-0.007100</td>
          <td id="T_78b05_row470_col8" class="data row470 col8" >0.020500</td>
          <td id="T_78b05_row470_col9" class="data row470 col9" >0.035300</td>
          <td id="T_78b05_row470_col10" class="data row470 col10" >0.114300</td>
          <td id="T_78b05_row470_col11" class="data row470 col11" >0.001500</td>
          <td id="T_78b05_row470_col12" class="data row470 col12" >0.062800</td>
          <td id="T_78b05_row470_col13" class="data row470 col13" >0.051800</td>
          <td id="T_78b05_row470_col14" class="data row470 col14" >0.005300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row471" class="row_heading level0 row471" >472</th>
          <td id="T_78b05_row471_col0" class="data row471 col0" >None</td>
          <td id="T_78b05_row471_col1" class="data row471 col1" >0.038900</td>
          <td id="T_78b05_row471_col2" class="data row471 col2" >0.006000</td>
          <td id="T_78b05_row471_col3" class="data row471 col3" >0.060200</td>
          <td id="T_78b05_row471_col4" class="data row471 col4" >-0.002600</td>
          <td id="T_78b05_row471_col5" class="data row471 col5" >0.069200</td>
          <td id="T_78b05_row471_col6" class="data row471 col6" >-0.039300</td>
          <td id="T_78b05_row471_col7" class="data row471 col7" >-0.038700</td>
          <td id="T_78b05_row471_col8" class="data row471 col8" >0.006600</td>
          <td id="T_78b05_row471_col9" class="data row471 col9" >0.035800</td>
          <td id="T_78b05_row471_col10" class="data row471 col10" >0.029500</td>
          <td id="T_78b05_row471_col11" class="data row471 col11" >0.015300</td>
          <td id="T_78b05_row471_col12" class="data row471 col12" >0.069800</td>
          <td id="T_78b05_row471_col13" class="data row471 col13" >0.035400</td>
          <td id="T_78b05_row471_col14" class="data row471 col14" >0.036900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row472" class="row_heading level0 row472" >473</th>
          <td id="T_78b05_row472_col0" class="data row472 col0" >None</td>
          <td id="T_78b05_row472_col1" class="data row472 col1" >0.040400</td>
          <td id="T_78b05_row472_col2" class="data row472 col2" >-0.016300</td>
          <td id="T_78b05_row472_col3" class="data row472 col3" >-0.006700</td>
          <td id="T_78b05_row472_col4" class="data row472 col4" >-0.016600</td>
          <td id="T_78b05_row472_col5" class="data row472 col5" >-0.005800</td>
          <td id="T_78b05_row472_col6" class="data row472 col6" >-0.016900</td>
          <td id="T_78b05_row472_col7" class="data row472 col7" >-0.035400</td>
          <td id="T_78b05_row472_col8" class="data row472 col8" >0.005200</td>
          <td id="T_78b05_row472_col9" class="data row472 col9" >0.013500</td>
          <td id="T_78b05_row472_col10" class="data row472 col10" >0.037400</td>
          <td id="T_78b05_row472_col11" class="data row472 col11" >0.001300</td>
          <td id="T_78b05_row472_col12" class="data row472 col12" >0.005100</td>
          <td id="T_78b05_row472_col13" class="data row472 col13" >0.013000</td>
          <td id="T_78b05_row472_col14" class="data row472 col14" >0.033700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row473" class="row_heading level0 row473" >474</th>
          <td id="T_78b05_row473_col0" class="data row473 col0" >None</td>
          <td id="T_78b05_row473_col1" class="data row473 col1" >0.037900</td>
          <td id="T_78b05_row473_col2" class="data row473 col2" >-0.012100</td>
          <td id="T_78b05_row473_col3" class="data row473 col3" >-0.022500</td>
          <td id="T_78b05_row473_col4" class="data row473 col4" >-0.042100</td>
          <td id="T_78b05_row473_col5" class="data row473 col5" >-0.045700</td>
          <td id="T_78b05_row473_col6" class="data row473 col6" >-0.007100</td>
          <td id="T_78b05_row473_col7" class="data row473 col7" >0.021900</td>
          <td id="T_78b05_row473_col8" class="data row473 col8" >0.007600</td>
          <td id="T_78b05_row473_col9" class="data row473 col9" >0.017800</td>
          <td id="T_78b05_row473_col10" class="data row473 col10" >0.053200</td>
          <td id="T_78b05_row473_col11" class="data row473 col11" >0.024200</td>
          <td id="T_78b05_row473_col12" class="data row473 col12" >0.045100</td>
          <td id="T_78b05_row473_col13" class="data row473 col13" >0.003200</td>
          <td id="T_78b05_row473_col14" class="data row473 col14" >0.023600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row474" class="row_heading level0 row474" >475</th>
          <td id="T_78b05_row474_col0" class="data row474 col0" >None</td>
          <td id="T_78b05_row474_col1" class="data row474 col1" >0.037000</td>
          <td id="T_78b05_row474_col2" class="data row474 col2" >-0.017500</td>
          <td id="T_78b05_row474_col3" class="data row474 col3" >-0.014100</td>
          <td id="T_78b05_row474_col4" class="data row474 col4" >-0.055400</td>
          <td id="T_78b05_row474_col5" class="data row474 col5" >-0.039200</td>
          <td id="T_78b05_row474_col6" class="data row474 col6" >-0.005000</td>
          <td id="T_78b05_row474_col7" class="data row474 col7" >0.008600</td>
          <td id="T_78b05_row474_col8" class="data row474 col8" >0.008500</td>
          <td id="T_78b05_row474_col9" class="data row474 col9" >0.012400</td>
          <td id="T_78b05_row474_col10" class="data row474 col10" >0.044800</td>
          <td id="T_78b05_row474_col11" class="data row474 col11" >0.037500</td>
          <td id="T_78b05_row474_col12" class="data row474 col12" >0.038600</td>
          <td id="T_78b05_row474_col13" class="data row474 col13" >0.001100</td>
          <td id="T_78b05_row474_col14" class="data row474 col14" >0.010400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row475" class="row_heading level0 row475" >476</th>
          <td id="T_78b05_row475_col0" class="data row475 col0" >None</td>
          <td id="T_78b05_row475_col1" class="data row475 col1" >0.045500</td>
          <td id="T_78b05_row475_col2" class="data row475 col2" >0.015400</td>
          <td id="T_78b05_row475_col3" class="data row475 col3" >-0.003500</td>
          <td id="T_78b05_row475_col4" class="data row475 col4" >0.024400</td>
          <td id="T_78b05_row475_col5" class="data row475 col5" >0.055300</td>
          <td id="T_78b05_row475_col6" class="data row475 col6" >-0.034000</td>
          <td id="T_78b05_row475_col7" class="data row475 col7" >0.000700</td>
          <td id="T_78b05_row475_col8" class="data row475 col8" >0.000000</td>
          <td id="T_78b05_row475_col9" class="data row475 col9" >0.045200</td>
          <td id="T_78b05_row475_col10" class="data row475 col10" >0.034300</td>
          <td id="T_78b05_row475_col11" class="data row475 col11" >0.042400</td>
          <td id="T_78b05_row475_col12" class="data row475 col12" >0.056000</td>
          <td id="T_78b05_row475_col13" class="data row475 col13" >0.030100</td>
          <td id="T_78b05_row475_col14" class="data row475 col14" >0.002400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row476" class="row_heading level0 row476" >477</th>
          <td id="T_78b05_row476_col0" class="data row476 col0" >None</td>
          <td id="T_78b05_row476_col1" class="data row476 col1" >0.036600</td>
          <td id="T_78b05_row476_col2" class="data row476 col2" >0.003400</td>
          <td id="T_78b05_row476_col3" class="data row476 col3" >0.016300</td>
          <td id="T_78b05_row476_col4" class="data row476 col4" >0.009400</td>
          <td id="T_78b05_row476_col5" class="data row476 col5" >0.014800</td>
          <td id="T_78b05_row476_col6" class="data row476 col6" >0.041900</td>
          <td id="T_78b05_row476_col7" class="data row476 col7" >-0.028900</td>
          <td id="T_78b05_row476_col8" class="data row476 col8" >0.008900</td>
          <td id="T_78b05_row476_col9" class="data row476 col9" >0.033300</td>
          <td id="T_78b05_row476_col10" class="data row476 col10" >0.014400</td>
          <td id="T_78b05_row476_col11" class="data row476 col11" >0.027300</td>
          <td id="T_78b05_row476_col12" class="data row476 col12" >0.015400</td>
          <td id="T_78b05_row476_col13" class="data row476 col13" >0.045800</td>
          <td id="T_78b05_row476_col14" class="data row476 col14" >0.027200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row477" class="row_heading level0 row477" >478</th>
          <td id="T_78b05_row477_col0" class="data row477 col0" >None</td>
          <td id="T_78b05_row477_col1" class="data row477 col1" >0.040300</td>
          <td id="T_78b05_row477_col2" class="data row477 col2" >0.002600</td>
          <td id="T_78b05_row477_col3" class="data row477 col3" >0.006500</td>
          <td id="T_78b05_row477_col4" class="data row477 col4" >-0.019800</td>
          <td id="T_78b05_row477_col5" class="data row477 col5" >0.004500</td>
          <td id="T_78b05_row477_col6" class="data row477 col6" >-0.058400</td>
          <td id="T_78b05_row477_col7" class="data row477 col7" >0.013700</td>
          <td id="T_78b05_row477_col8" class="data row477 col8" >0.005300</td>
          <td id="T_78b05_row477_col9" class="data row477 col9" >0.032400</td>
          <td id="T_78b05_row477_col10" class="data row477 col10" >0.024300</td>
          <td id="T_78b05_row477_col11" class="data row477 col11" >0.001900</td>
          <td id="T_78b05_row477_col12" class="data row477 col12" >0.005100</td>
          <td id="T_78b05_row477_col13" class="data row477 col13" >0.054500</td>
          <td id="T_78b05_row477_col14" class="data row477 col14" >0.015400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row478" class="row_heading level0 row478" >479</th>
          <td id="T_78b05_row478_col0" class="data row478 col0" >None</td>
          <td id="T_78b05_row478_col1" class="data row478 col1" >0.038700</td>
          <td id="T_78b05_row478_col2" class="data row478 col2" >-0.011700</td>
          <td id="T_78b05_row478_col3" class="data row478 col3" >-0.055500</td>
          <td id="T_78b05_row478_col4" class="data row478 col4" >0.001500</td>
          <td id="T_78b05_row478_col5" class="data row478 col5" >0.033600</td>
          <td id="T_78b05_row478_col6" class="data row478 col6" >-0.065100</td>
          <td id="T_78b05_row478_col7" class="data row478 col7" >0.017700</td>
          <td id="T_78b05_row478_col8" class="data row478 col8" >0.006800</td>
          <td id="T_78b05_row478_col9" class="data row478 col9" >0.018200</td>
          <td id="T_78b05_row478_col10" class="data row478 col10" >0.086300</td>
          <td id="T_78b05_row478_col11" class="data row478 col11" >0.019400</td>
          <td id="T_78b05_row478_col12" class="data row478 col12" >0.034200</td>
          <td id="T_78b05_row478_col13" class="data row478 col13" >0.061200</td>
          <td id="T_78b05_row478_col14" class="data row478 col14" >0.019400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row479" class="row_heading level0 row479" >480</th>
          <td id="T_78b05_row479_col0" class="data row479 col0" >None</td>
          <td id="T_78b05_row479_col1" class="data row479 col1" >0.029900</td>
          <td id="T_78b05_row479_col2" class="data row479 col2" >-0.011900</td>
          <td id="T_78b05_row479_col3" class="data row479 col3" >-0.011600</td>
          <td id="T_78b05_row479_col4" class="data row479 col4" >-0.018500</td>
          <td id="T_78b05_row479_col5" class="data row479 col5" >-0.066100</td>
          <td id="T_78b05_row479_col6" class="data row479 col6" >-0.053000</td>
          <td id="T_78b05_row479_col7" class="data row479 col7" >-0.065600</td>
          <td id="T_78b05_row479_col8" class="data row479 col8" >0.015700</td>
          <td id="T_78b05_row479_col9" class="data row479 col9" >0.017900</td>
          <td id="T_78b05_row479_col10" class="data row479 col10" >0.042300</td>
          <td id="T_78b05_row479_col11" class="data row479 col11" >0.000600</td>
          <td id="T_78b05_row479_col12" class="data row479 col12" >0.065500</td>
          <td id="T_78b05_row479_col13" class="data row479 col13" >0.049100</td>
          <td id="T_78b05_row479_col14" class="data row479 col14" >0.063800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row480" class="row_heading level0 row480" >481</th>
          <td id="T_78b05_row480_col0" class="data row480 col0" >None</td>
          <td id="T_78b05_row480_col1" class="data row480 col1" >0.040500</td>
          <td id="T_78b05_row480_col2" class="data row480 col2" >0.035300</td>
          <td id="T_78b05_row480_col3" class="data row480 col3" >0.050700</td>
          <td id="T_78b05_row480_col4" class="data row480 col4" >0.008000</td>
          <td id="T_78b05_row480_col5" class="data row480 col5" >-0.010600</td>
          <td id="T_78b05_row480_col6" class="data row480 col6" >-0.042000</td>
          <td id="T_78b05_row480_col7" class="data row480 col7" >-0.053900</td>
          <td id="T_78b05_row480_col8" class="data row480 col8" >0.005000</td>
          <td id="T_78b05_row480_col9" class="data row480 col9" >0.065200</td>
          <td id="T_78b05_row480_col10" class="data row480 col10" >0.020000</td>
          <td id="T_78b05_row480_col11" class="data row480 col11" >0.025900</td>
          <td id="T_78b05_row480_col12" class="data row480 col12" >0.010000</td>
          <td id="T_78b05_row480_col13" class="data row480 col13" >0.038100</td>
          <td id="T_78b05_row480_col14" class="data row480 col14" >0.052200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row481" class="row_heading level0 row481" >482</th>
          <td id="T_78b05_row481_col0" class="data row481 col0" >None</td>
          <td id="T_78b05_row481_col1" class="data row481 col1" >0.031800</td>
          <td id="T_78b05_row481_col2" class="data row481 col2" >0.038900</td>
          <td id="T_78b05_row481_col3" class="data row481 col3" >-0.084500</td>
          <td id="T_78b05_row481_col4" class="data row481 col4" >-0.006900</td>
          <td id="T_78b05_row481_col5" class="data row481 col5" >-0.000700</td>
          <td id="T_78b05_row481_col6" class="data row481 col6" >0.035900</td>
          <td id="T_78b05_row481_col7" class="data row481 col7" >0.048300</td>
          <td id="T_78b05_row481_col8" class="data row481 col8" >0.013700</td>
          <td id="T_78b05_row481_col9" class="data row481 col9" >0.068700</td>
          <td id="T_78b05_row481_col10" class="data row481 col10" >0.115200</td>
          <td id="T_78b05_row481_col11" class="data row481 col11" >0.011000</td>
          <td id="T_78b05_row481_col12" class="data row481 col12" >0.000100</td>
          <td id="T_78b05_row481_col13" class="data row481 col13" >0.039800</td>
          <td id="T_78b05_row481_col14" class="data row481 col14" >0.050100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row482" class="row_heading level0 row482" >483</th>
          <td id="T_78b05_row482_col0" class="data row482 col0" >None</td>
          <td id="T_78b05_row482_col1" class="data row482 col1" >0.040700</td>
          <td id="T_78b05_row482_col2" class="data row482 col2" >0.043500</td>
          <td id="T_78b05_row482_col3" class="data row482 col3" >0.025300</td>
          <td id="T_78b05_row482_col4" class="data row482 col4" >-0.072200</td>
          <td id="T_78b05_row482_col5" class="data row482 col5" >0.016600</td>
          <td id="T_78b05_row482_col6" class="data row482 col6" >0.051500</td>
          <td id="T_78b05_row482_col7" class="data row482 col7" >-0.011100</td>
          <td id="T_78b05_row482_col8" class="data row482 col8" >0.004800</td>
          <td id="T_78b05_row482_col9" class="data row482 col9" >0.073300</td>
          <td id="T_78b05_row482_col10" class="data row482 col10" >0.005500</td>
          <td id="T_78b05_row482_col11" class="data row482 col11" >0.054300</td>
          <td id="T_78b05_row482_col12" class="data row482 col12" >0.017200</td>
          <td id="T_78b05_row482_col13" class="data row482 col13" >0.055400</td>
          <td id="T_78b05_row482_col14" class="data row482 col14" >0.009300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row483" class="row_heading level0 row483" >484</th>
          <td id="T_78b05_row483_col0" class="data row483 col0" >None</td>
          <td id="T_78b05_row483_col1" class="data row483 col1" >0.043300</td>
          <td id="T_78b05_row483_col2" class="data row483 col2" >0.008200</td>
          <td id="T_78b05_row483_col3" class="data row483 col3" >0.072800</td>
          <td id="T_78b05_row483_col4" class="data row483 col4" >0.050300</td>
          <td id="T_78b05_row483_col5" class="data row483 col5" >-0.011000</td>
          <td id="T_78b05_row483_col6" class="data row483 col6" >-0.020000</td>
          <td id="T_78b05_row483_col7" class="data row483 col7" >-0.008000</td>
          <td id="T_78b05_row483_col8" class="data row483 col8" >0.002200</td>
          <td id="T_78b05_row483_col9" class="data row483 col9" >0.038000</td>
          <td id="T_78b05_row483_col10" class="data row483 col10" >0.042000</td>
          <td id="T_78b05_row483_col11" class="data row483 col11" >0.068200</td>
          <td id="T_78b05_row483_col12" class="data row483 col12" >0.010300</td>
          <td id="T_78b05_row483_col13" class="data row483 col13" >0.016100</td>
          <td id="T_78b05_row483_col14" class="data row483 col14" >0.006300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row484" class="row_heading level0 row484" >485</th>
          <td id="T_78b05_row484_col0" class="data row484 col0" >None</td>
          <td id="T_78b05_row484_col1" class="data row484 col1" >0.034500</td>
          <td id="T_78b05_row484_col2" class="data row484 col2" >0.017800</td>
          <td id="T_78b05_row484_col3" class="data row484 col3" >-0.021400</td>
          <td id="T_78b05_row484_col4" class="data row484 col4" >0.010800</td>
          <td id="T_78b05_row484_col5" class="data row484 col5" >0.024400</td>
          <td id="T_78b05_row484_col6" class="data row484 col6" >-0.055000</td>
          <td id="T_78b05_row484_col7" class="data row484 col7" >-0.060900</td>
          <td id="T_78b05_row484_col8" class="data row484 col8" >0.011100</td>
          <td id="T_78b05_row484_col9" class="data row484 col9" >0.047600</td>
          <td id="T_78b05_row484_col10" class="data row484 col10" >0.052200</td>
          <td id="T_78b05_row484_col11" class="data row484 col11" >0.028700</td>
          <td id="T_78b05_row484_col12" class="data row484 col12" >0.025000</td>
          <td id="T_78b05_row484_col13" class="data row484 col13" >0.051100</td>
          <td id="T_78b05_row484_col14" class="data row484 col14" >0.059200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row485" class="row_heading level0 row485" >486</th>
          <td id="T_78b05_row485_col0" class="data row485 col0" >None</td>
          <td id="T_78b05_row485_col1" class="data row485 col1" >0.039400</td>
          <td id="T_78b05_row485_col2" class="data row485 col2" >-0.015900</td>
          <td id="T_78b05_row485_col3" class="data row485 col3" >0.002900</td>
          <td id="T_78b05_row485_col4" class="data row485 col4" >-0.037800</td>
          <td id="T_78b05_row485_col5" class="data row485 col5" >0.002600</td>
          <td id="T_78b05_row485_col6" class="data row485 col6" >-0.008100</td>
          <td id="T_78b05_row485_col7" class="data row485 col7" >-0.046300</td>
          <td id="T_78b05_row485_col8" class="data row485 col8" >0.006100</td>
          <td id="T_78b05_row485_col9" class="data row485 col9" >0.013900</td>
          <td id="T_78b05_row485_col10" class="data row485 col10" >0.027900</td>
          <td id="T_78b05_row485_col11" class="data row485 col11" >0.019900</td>
          <td id="T_78b05_row485_col12" class="data row485 col12" >0.003200</td>
          <td id="T_78b05_row485_col13" class="data row485 col13" >0.004200</td>
          <td id="T_78b05_row485_col14" class="data row485 col14" >0.044500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row486" class="row_heading level0 row486" >487</th>
          <td id="T_78b05_row486_col0" class="data row486 col0" >None</td>
          <td id="T_78b05_row486_col1" class="data row486 col1" >0.041200</td>
          <td id="T_78b05_row486_col2" class="data row486 col2" >-0.005000</td>
          <td id="T_78b05_row486_col3" class="data row486 col3" >0.023000</td>
          <td id="T_78b05_row486_col4" class="data row486 col4" >0.004800</td>
          <td id="T_78b05_row486_col5" class="data row486 col5" >0.033700</td>
          <td id="T_78b05_row486_col6" class="data row486 col6" >0.061800</td>
          <td id="T_78b05_row486_col7" class="data row486 col7" >0.004600</td>
          <td id="T_78b05_row486_col8" class="data row486 col8" >0.004400</td>
          <td id="T_78b05_row486_col9" class="data row486 col9" >0.024800</td>
          <td id="T_78b05_row486_col10" class="data row486 col10" >0.007700</td>
          <td id="T_78b05_row486_col11" class="data row486 col11" >0.022700</td>
          <td id="T_78b05_row486_col12" class="data row486 col12" >0.034300</td>
          <td id="T_78b05_row486_col13" class="data row486 col13" >0.065700</td>
          <td id="T_78b05_row486_col14" class="data row486 col14" >0.006300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row487" class="row_heading level0 row487" >488</th>
          <td id="T_78b05_row487_col0" class="data row487 col0" >None</td>
          <td id="T_78b05_row487_col1" class="data row487 col1" >0.028400</td>
          <td id="T_78b05_row487_col2" class="data row487 col2" >0.078200</td>
          <td id="T_78b05_row487_col3" class="data row487 col3" >0.016700</td>
          <td id="T_78b05_row487_col4" class="data row487 col4" >0.001300</td>
          <td id="T_78b05_row487_col5" class="data row487 col5" >-0.059600</td>
          <td id="T_78b05_row487_col6" class="data row487 col6" >-0.035700</td>
          <td id="T_78b05_row487_col7" class="data row487 col7" >0.039100</td>
          <td id="T_78b05_row487_col8" class="data row487 col8" >0.017200</td>
          <td id="T_78b05_row487_col9" class="data row487 col9" >0.108000</td>
          <td id="T_78b05_row487_col10" class="data row487 col10" >0.014100</td>
          <td id="T_78b05_row487_col11" class="data row487 col11" >0.019200</td>
          <td id="T_78b05_row487_col12" class="data row487 col12" >0.059000</td>
          <td id="T_78b05_row487_col13" class="data row487 col13" >0.031800</td>
          <td id="T_78b05_row487_col14" class="data row487 col14" >0.040800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row488" class="row_heading level0 row488" >489</th>
          <td id="T_78b05_row488_col0" class="data row488 col0" >None</td>
          <td id="T_78b05_row488_col1" class="data row488 col1" >0.040900</td>
          <td id="T_78b05_row488_col2" class="data row488 col2" >-0.020800</td>
          <td id="T_78b05_row488_col3" class="data row488 col3" >0.014300</td>
          <td id="T_78b05_row488_col4" class="data row488 col4" >-0.028800</td>
          <td id="T_78b05_row488_col5" class="data row488 col5" >-0.033300</td>
          <td id="T_78b05_row488_col6" class="data row488 col6" >0.019500</td>
          <td id="T_78b05_row488_col7" class="data row488 col7" >0.084900</td>
          <td id="T_78b05_row488_col8" class="data row488 col8" >0.004600</td>
          <td id="T_78b05_row488_col9" class="data row488 col9" >0.009000</td>
          <td id="T_78b05_row488_col10" class="data row488 col10" >0.016500</td>
          <td id="T_78b05_row488_col11" class="data row488 col11" >0.010900</td>
          <td id="T_78b05_row488_col12" class="data row488 col12" >0.032700</td>
          <td id="T_78b05_row488_col13" class="data row488 col13" >0.023400</td>
          <td id="T_78b05_row488_col14" class="data row488 col14" >0.086700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row489" class="row_heading level0 row489" >490</th>
          <td id="T_78b05_row489_col0" class="data row489 col0" >None</td>
          <td id="T_78b05_row489_col1" class="data row489 col1" >0.047200</td>
          <td id="T_78b05_row489_col2" class="data row489 col2" >-0.006100</td>
          <td id="T_78b05_row489_col3" class="data row489 col3" >0.036400</td>
          <td id="T_78b05_row489_col4" class="data row489 col4" >0.046300</td>
          <td id="T_78b05_row489_col5" class="data row489 col5" >0.068100</td>
          <td id="T_78b05_row489_col6" class="data row489 col6" >-0.029400</td>
          <td id="T_78b05_row489_col7" class="data row489 col7" >0.011800</td>
          <td id="T_78b05_row489_col8" class="data row489 col8" >0.001600</td>
          <td id="T_78b05_row489_col9" class="data row489 col9" >0.023700</td>
          <td id="T_78b05_row489_col10" class="data row489 col10" >0.005700</td>
          <td id="T_78b05_row489_col11" class="data row489 col11" >0.064300</td>
          <td id="T_78b05_row489_col12" class="data row489 col12" >0.068700</td>
          <td id="T_78b05_row489_col13" class="data row489 col13" >0.025500</td>
          <td id="T_78b05_row489_col14" class="data row489 col14" >0.013500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row490" class="row_heading level0 row490" >491</th>
          <td id="T_78b05_row490_col0" class="data row490 col0" >None</td>
          <td id="T_78b05_row490_col1" class="data row490 col1" >0.037500</td>
          <td id="T_78b05_row490_col2" class="data row490 col2" >0.046400</td>
          <td id="T_78b05_row490_col3" class="data row490 col3" >0.065300</td>
          <td id="T_78b05_row490_col4" class="data row490 col4" >-0.035800</td>
          <td id="T_78b05_row490_col5" class="data row490 col5" >-0.021400</td>
          <td id="T_78b05_row490_col6" class="data row490 col6" >-0.054400</td>
          <td id="T_78b05_row490_col7" class="data row490 col7" >-0.033700</td>
          <td id="T_78b05_row490_col8" class="data row490 col8" >0.008000</td>
          <td id="T_78b05_row490_col9" class="data row490 col9" >0.076200</td>
          <td id="T_78b05_row490_col10" class="data row490 col10" >0.034600</td>
          <td id="T_78b05_row490_col11" class="data row490 col11" >0.017900</td>
          <td id="T_78b05_row490_col12" class="data row490 col12" >0.020800</td>
          <td id="T_78b05_row490_col13" class="data row490 col13" >0.050500</td>
          <td id="T_78b05_row490_col14" class="data row490 col14" >0.032000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row491" class="row_heading level0 row491" >492</th>
          <td id="T_78b05_row491_col0" class="data row491 col0" >None</td>
          <td id="T_78b05_row491_col1" class="data row491 col1" >0.030400</td>
          <td id="T_78b05_row491_col2" class="data row491 col2" >0.043600</td>
          <td id="T_78b05_row491_col3" class="data row491 col3" >0.038900</td>
          <td id="T_78b05_row491_col4" class="data row491 col4" >-0.036300</td>
          <td id="T_78b05_row491_col5" class="data row491 col5" >-0.061600</td>
          <td id="T_78b05_row491_col6" class="data row491 col6" >0.070100</td>
          <td id="T_78b05_row491_col7" class="data row491 col7" >-0.058400</td>
          <td id="T_78b05_row491_col8" class="data row491 col8" >0.015100</td>
          <td id="T_78b05_row491_col9" class="data row491 col9" >0.073400</td>
          <td id="T_78b05_row491_col10" class="data row491 col10" >0.008100</td>
          <td id="T_78b05_row491_col11" class="data row491 col11" >0.018400</td>
          <td id="T_78b05_row491_col12" class="data row491 col12" >0.061000</td>
          <td id="T_78b05_row491_col13" class="data row491 col13" >0.074000</td>
          <td id="T_78b05_row491_col14" class="data row491 col14" >0.056600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row492" class="row_heading level0 row492" >493</th>
          <td id="T_78b05_row492_col0" class="data row492 col0" >None</td>
          <td id="T_78b05_row492_col1" class="data row492 col1" >0.037000</td>
          <td id="T_78b05_row492_col2" class="data row492 col2" >-0.003200</td>
          <td id="T_78b05_row492_col3" class="data row492 col3" >-0.068400</td>
          <td id="T_78b05_row492_col4" class="data row492 col4" >0.007200</td>
          <td id="T_78b05_row492_col5" class="data row492 col5" >-0.020100</td>
          <td id="T_78b05_row492_col6" class="data row492 col6" >0.006100</td>
          <td id="T_78b05_row492_col7" class="data row492 col7" >0.033600</td>
          <td id="T_78b05_row492_col8" class="data row492 col8" >0.008600</td>
          <td id="T_78b05_row492_col9" class="data row492 col9" >0.026600</td>
          <td id="T_78b05_row492_col10" class="data row492 col10" >0.099100</td>
          <td id="T_78b05_row492_col11" class="data row492 col11" >0.025100</td>
          <td id="T_78b05_row492_col12" class="data row492 col12" >0.019400</td>
          <td id="T_78b05_row492_col13" class="data row492 col13" >0.009900</td>
          <td id="T_78b05_row492_col14" class="data row492 col14" >0.035300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row493" class="row_heading level0 row493" >494</th>
          <td id="T_78b05_row493_col0" class="data row493 col0" >None</td>
          <td id="T_78b05_row493_col1" class="data row493 col1" >0.030200</td>
          <td id="T_78b05_row493_col2" class="data row493 col2" >0.013500</td>
          <td id="T_78b05_row493_col3" class="data row493 col3" >-0.047500</td>
          <td id="T_78b05_row493_col4" class="data row493 col4" >0.042200</td>
          <td id="T_78b05_row493_col5" class="data row493 col5" >-0.002500</td>
          <td id="T_78b05_row493_col6" class="data row493 col6" >0.072000</td>
          <td id="T_78b05_row493_col7" class="data row493 col7" >-0.063100</td>
          <td id="T_78b05_row493_col8" class="data row493 col8" >0.015400</td>
          <td id="T_78b05_row493_col9" class="data row493 col9" >0.043400</td>
          <td id="T_78b05_row493_col10" class="data row493 col10" >0.078300</td>
          <td id="T_78b05_row493_col11" class="data row493 col11" >0.060200</td>
          <td id="T_78b05_row493_col12" class="data row493 col12" >0.001900</td>
          <td id="T_78b05_row493_col13" class="data row493 col13" >0.075900</td>
          <td id="T_78b05_row493_col14" class="data row493 col14" >0.061400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row494" class="row_heading level0 row494" >495</th>
          <td id="T_78b05_row494_col0" class="data row494 col0" >None</td>
          <td id="T_78b05_row494_col1" class="data row494 col1" >0.031900</td>
          <td id="T_78b05_row494_col2" class="data row494 col2" >-0.014600</td>
          <td id="T_78b05_row494_col3" class="data row494 col3" >-0.023100</td>
          <td id="T_78b05_row494_col4" class="data row494 col4" >0.018200</td>
          <td id="T_78b05_row494_col5" class="data row494 col5" >-0.015500</td>
          <td id="T_78b05_row494_col6" class="data row494 col6" >0.024000</td>
          <td id="T_78b05_row494_col7" class="data row494 col7" >-0.056900</td>
          <td id="T_78b05_row494_col8" class="data row494 col8" >0.013700</td>
          <td id="T_78b05_row494_col9" class="data row494 col9" >0.015200</td>
          <td id="T_78b05_row494_col10" class="data row494 col10" >0.053800</td>
          <td id="T_78b05_row494_col11" class="data row494 col11" >0.036100</td>
          <td id="T_78b05_row494_col12" class="data row494 col12" >0.014900</td>
          <td id="T_78b05_row494_col13" class="data row494 col13" >0.027900</td>
          <td id="T_78b05_row494_col14" class="data row494 col14" >0.055200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row495" class="row_heading level0 row495" >496</th>
          <td id="T_78b05_row495_col0" class="data row495 col0" >None</td>
          <td id="T_78b05_row495_col1" class="data row495 col1" >0.042200</td>
          <td id="T_78b05_row495_col2" class="data row495 col2" >-0.045500</td>
          <td id="T_78b05_row495_col3" class="data row495 col3" >0.025200</td>
          <td id="T_78b05_row495_col4" class="data row495 col4" >-0.082700</td>
          <td id="T_78b05_row495_col5" class="data row495 col5" >0.005500</td>
          <td id="T_78b05_row495_col6" class="data row495 col6" >-0.006600</td>
          <td id="T_78b05_row495_col7" class="data row495 col7" >-0.053200</td>
          <td id="T_78b05_row495_col8" class="data row495 col8" >0.003300</td>
          <td id="T_78b05_row495_col9" class="data row495 col9" >0.015700</td>
          <td id="T_78b05_row495_col10" class="data row495 col10" >0.005500</td>
          <td id="T_78b05_row495_col11" class="data row495 col11" >0.064800</td>
          <td id="T_78b05_row495_col12" class="data row495 col12" >0.006200</td>
          <td id="T_78b05_row495_col13" class="data row495 col13" >0.002700</td>
          <td id="T_78b05_row495_col14" class="data row495 col14" >0.051400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row496" class="row_heading level0 row496" >497</th>
          <td id="T_78b05_row496_col0" class="data row496 col0" >PC1</td>
          <td id="T_78b05_row496_col1" class="data row496 col1" >0.022500</td>
          <td id="T_78b05_row496_col2" class="data row496 col2" >0.051200</td>
          <td id="T_78b05_row496_col3" class="data row496 col3" >-0.013400</td>
          <td id="T_78b05_row496_col4" class="data row496 col4" >0.016500</td>
          <td id="T_78b05_row496_col5" class="data row496 col5" >-0.060300</td>
          <td id="T_78b05_row496_col6" class="data row496 col6" >0.063500</td>
          <td id="T_78b05_row496_col7" class="data row496 col7" >-0.070300</td>
          <td id="T_78b05_row496_col8" class="data row496 col8" >0.023000</td>
          <td id="T_78b05_row496_col9" class="data row496 col9" >0.081100</td>
          <td id="T_78b05_row496_col10" class="data row496 col10" >0.044200</td>
          <td id="T_78b05_row496_col11" class="data row496 col11" >0.034500</td>
          <td id="T_78b05_row496_col12" class="data row496 col12" >0.059700</td>
          <td id="T_78b05_row496_col13" class="data row496 col13" >0.067400</td>
          <td id="T_78b05_row496_col14" class="data row496 col14" >0.068500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row497" class="row_heading level0 row497" >498</th>
          <td id="T_78b05_row497_col0" class="data row497 col0" >None</td>
          <td id="T_78b05_row497_col1" class="data row497 col1" >0.045000</td>
          <td id="T_78b05_row497_col2" class="data row497 col2" >-0.049400</td>
          <td id="T_78b05_row497_col3" class="data row497 col3" >0.053000</td>
          <td id="T_78b05_row497_col4" class="data row497 col4" >0.018400</td>
          <td id="T_78b05_row497_col5" class="data row497 col5" >-0.096000</td>
          <td id="T_78b05_row497_col6" class="data row497 col6" >0.027400</td>
          <td id="T_78b05_row497_col7" class="data row497 col7" >-0.022000</td>
          <td id="T_78b05_row497_col8" class="data row497 col8" >0.000600</td>
          <td id="T_78b05_row497_col9" class="data row497 col9" >0.019600</td>
          <td id="T_78b05_row497_col10" class="data row497 col10" >0.022200</td>
          <td id="T_78b05_row497_col11" class="data row497 col11" >0.036300</td>
          <td id="T_78b05_row497_col12" class="data row497 col12" >0.095300</td>
          <td id="T_78b05_row497_col13" class="data row497 col13" >0.031300</td>
          <td id="T_78b05_row497_col14" class="data row497 col14" >0.020300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row498" class="row_heading level0 row498" >499</th>
          <td id="T_78b05_row498_col0" class="data row498 col0" >None</td>
          <td id="T_78b05_row498_col1" class="data row498 col1" >0.029000</td>
          <td id="T_78b05_row498_col2" class="data row498 col2" >0.020800</td>
          <td id="T_78b05_row498_col3" class="data row498 col3" >-0.075600</td>
          <td id="T_78b05_row498_col4" class="data row498 col4" >-0.057900</td>
          <td id="T_78b05_row498_col5" class="data row498 col5" >-0.003800</td>
          <td id="T_78b05_row498_col6" class="data row498 col6" >0.037700</td>
          <td id="T_78b05_row498_col7" class="data row498 col7" >0.038200</td>
          <td id="T_78b05_row498_col8" class="data row498 col8" >0.016500</td>
          <td id="T_78b05_row498_col9" class="data row498 col9" >0.050600</td>
          <td id="T_78b05_row498_col10" class="data row498 col10" >0.106400</td>
          <td id="T_78b05_row498_col11" class="data row498 col11" >0.040000</td>
          <td id="T_78b05_row498_col12" class="data row498 col12" >0.003100</td>
          <td id="T_78b05_row498_col13" class="data row498 col13" >0.041500</td>
          <td id="T_78b05_row498_col14" class="data row498 col14" >0.039900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row499" class="row_heading level0 row499" >500</th>
          <td id="T_78b05_row499_col0" class="data row499 col0" >None</td>
          <td id="T_78b05_row499_col1" class="data row499 col1" >0.034400</td>
          <td id="T_78b05_row499_col2" class="data row499 col2" >0.009000</td>
          <td id="T_78b05_row499_col3" class="data row499 col3" >0.014700</td>
          <td id="T_78b05_row499_col4" class="data row499 col4" >-0.063300</td>
          <td id="T_78b05_row499_col5" class="data row499 col5" >-0.014100</td>
          <td id="T_78b05_row499_col6" class="data row499 col6" >0.082900</td>
          <td id="T_78b05_row499_col7" class="data row499 col7" >-0.062400</td>
          <td id="T_78b05_row499_col8" class="data row499 col8" >0.011100</td>
          <td id="T_78b05_row499_col9" class="data row499 col9" >0.038800</td>
          <td id="T_78b05_row499_col10" class="data row499 col10" >0.016000</td>
          <td id="T_78b05_row499_col11" class="data row499 col11" >0.045400</td>
          <td id="T_78b05_row499_col12" class="data row499 col12" >0.013500</td>
          <td id="T_78b05_row499_col13" class="data row499 col13" >0.086800</td>
          <td id="T_78b05_row499_col14" class="data row499 col14" >0.060600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row500" class="row_heading level0 row500" >501</th>
          <td id="T_78b05_row500_col0" class="data row500 col0" >PC6</td>
          <td id="T_78b05_row500_col1" class="data row500 col1" >0.042100</td>
          <td id="T_78b05_row500_col2" class="data row500 col2" >0.018500</td>
          <td id="T_78b05_row500_col3" class="data row500 col3" >0.050200</td>
          <td id="T_78b05_row500_col4" class="data row500 col4" >0.055700</td>
          <td id="T_78b05_row500_col5" class="data row500 col5" >0.010000</td>
          <td id="T_78b05_row500_col6" class="data row500 col6" >0.101500</td>
          <td id="T_78b05_row500_col7" class="data row500 col7" >0.042700</td>
          <td id="T_78b05_row500_col8" class="data row500 col8" >0.003500</td>
          <td id="T_78b05_row500_col9" class="data row500 col9" >0.048300</td>
          <td id="T_78b05_row500_col10" class="data row500 col10" >0.019400</td>
          <td id="T_78b05_row500_col11" class="data row500 col11" >0.073700</td>
          <td id="T_78b05_row500_col12" class="data row500 col12" >0.010700</td>
          <td id="T_78b05_row500_col13" class="data row500 col13" >0.105400</td>
          <td id="T_78b05_row500_col14" class="data row500 col14" >0.044500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row501" class="row_heading level0 row501" >502</th>
          <td id="T_78b05_row501_col0" class="data row501 col0" >None</td>
          <td id="T_78b05_row501_col1" class="data row501 col1" >0.051200</td>
          <td id="T_78b05_row501_col2" class="data row501 col2" >-0.051900</td>
          <td id="T_78b05_row501_col3" class="data row501 col3" >0.050100</td>
          <td id="T_78b05_row501_col4" class="data row501 col4" >0.048500</td>
          <td id="T_78b05_row501_col5" class="data row501 col5" >-0.024300</td>
          <td id="T_78b05_row501_col6" class="data row501 col6" >0.006600</td>
          <td id="T_78b05_row501_col7" class="data row501 col7" >0.013000</td>
          <td id="T_78b05_row501_col8" class="data row501 col8" >0.005700</td>
          <td id="T_78b05_row501_col9" class="data row501 col9" >0.022000</td>
          <td id="T_78b05_row501_col10" class="data row501 col10" >0.019400</td>
          <td id="T_78b05_row501_col11" class="data row501 col11" >0.066400</td>
          <td id="T_78b05_row501_col12" class="data row501 col12" >0.023600</td>
          <td id="T_78b05_row501_col13" class="data row501 col13" >0.010500</td>
          <td id="T_78b05_row501_col14" class="data row501 col14" >0.014700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row502" class="row_heading level0 row502" >503</th>
          <td id="T_78b05_row502_col0" class="data row502 col0" >None</td>
          <td id="T_78b05_row502_col1" class="data row502 col1" >0.036400</td>
          <td id="T_78b05_row502_col2" class="data row502 col2" >-0.023700</td>
          <td id="T_78b05_row502_col3" class="data row502 col3" >-0.027900</td>
          <td id="T_78b05_row502_col4" class="data row502 col4" >0.010600</td>
          <td id="T_78b05_row502_col5" class="data row502 col5" >-0.035300</td>
          <td id="T_78b05_row502_col6" class="data row502 col6" >0.012200</td>
          <td id="T_78b05_row502_col7" class="data row502 col7" >-0.001600</td>
          <td id="T_78b05_row502_col8" class="data row502 col8" >0.009200</td>
          <td id="T_78b05_row502_col9" class="data row502 col9" >0.006100</td>
          <td id="T_78b05_row502_col10" class="data row502 col10" >0.058600</td>
          <td id="T_78b05_row502_col11" class="data row502 col11" >0.028500</td>
          <td id="T_78b05_row502_col12" class="data row502 col12" >0.034700</td>
          <td id="T_78b05_row502_col13" class="data row502 col13" >0.016100</td>
          <td id="T_78b05_row502_col14" class="data row502 col14" >0.000100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row503" class="row_heading level0 row503" >504</th>
          <td id="T_78b05_row503_col0" class="data row503 col0" >None</td>
          <td id="T_78b05_row503_col1" class="data row503 col1" >0.048100</td>
          <td id="T_78b05_row503_col2" class="data row503 col2" >-0.016200</td>
          <td id="T_78b05_row503_col3" class="data row503 col3" >0.067700</td>
          <td id="T_78b05_row503_col4" class="data row503 col4" >-0.024200</td>
          <td id="T_78b05_row503_col5" class="data row503 col5" >0.059500</td>
          <td id="T_78b05_row503_col6" class="data row503 col6" >-0.006400</td>
          <td id="T_78b05_row503_col7" class="data row503 col7" >0.006200</td>
          <td id="T_78b05_row503_col8" class="data row503 col8" >0.002600</td>
          <td id="T_78b05_row503_col9" class="data row503 col9" >0.013700</td>
          <td id="T_78b05_row503_col10" class="data row503 col10" >0.036900</td>
          <td id="T_78b05_row503_col11" class="data row503 col11" >0.006300</td>
          <td id="T_78b05_row503_col12" class="data row503 col12" >0.060200</td>
          <td id="T_78b05_row503_col13" class="data row503 col13" >0.002500</td>
          <td id="T_78b05_row503_col14" class="data row503 col14" >0.008000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row504" class="row_heading level0 row504" >505</th>
          <td id="T_78b05_row504_col0" class="data row504 col0" >PC1</td>
          <td id="T_78b05_row504_col1" class="data row504 col1" >0.023100</td>
          <td id="T_78b05_row504_col2" class="data row504 col2" >0.048400</td>
          <td id="T_78b05_row504_col3" class="data row504 col3" >-0.033900</td>
          <td id="T_78b05_row504_col4" class="data row504 col4" >0.047200</td>
          <td id="T_78b05_row504_col5" class="data row504 col5" >-0.001700</td>
          <td id="T_78b05_row504_col6" class="data row504 col6" >0.015500</td>
          <td id="T_78b05_row504_col7" class="data row504 col7" >-0.031100</td>
          <td id="T_78b05_row504_col8" class="data row504 col8" >0.022500</td>
          <td id="T_78b05_row504_col9" class="data row504 col9" >0.078200</td>
          <td id="T_78b05_row504_col10" class="data row504 col10" >0.064600</td>
          <td id="T_78b05_row504_col11" class="data row504 col11" >0.065100</td>
          <td id="T_78b05_row504_col12" class="data row504 col12" >0.001000</td>
          <td id="T_78b05_row504_col13" class="data row504 col13" >0.019300</td>
          <td id="T_78b05_row504_col14" class="data row504 col14" >0.029300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row505" class="row_heading level0 row505" >506</th>
          <td id="T_78b05_row505_col0" class="data row505 col0" >None</td>
          <td id="T_78b05_row505_col1" class="data row505 col1" >0.033700</td>
          <td id="T_78b05_row505_col2" class="data row505 col2" >-0.025500</td>
          <td id="T_78b05_row505_col3" class="data row505 col3" >-0.047400</td>
          <td id="T_78b05_row505_col4" class="data row505 col4" >-0.038300</td>
          <td id="T_78b05_row505_col5" class="data row505 col5" >-0.019300</td>
          <td id="T_78b05_row505_col6" class="data row505 col6" >-0.051600</td>
          <td id="T_78b05_row505_col7" class="data row505 col7" >-0.016900</td>
          <td id="T_78b05_row505_col8" class="data row505 col8" >0.011900</td>
          <td id="T_78b05_row505_col9" class="data row505 col9" >0.004400</td>
          <td id="T_78b05_row505_col10" class="data row505 col10" >0.078100</td>
          <td id="T_78b05_row505_col11" class="data row505 col11" >0.020300</td>
          <td id="T_78b05_row505_col12" class="data row505 col12" >0.018700</td>
          <td id="T_78b05_row505_col13" class="data row505 col13" >0.047700</td>
          <td id="T_78b05_row505_col14" class="data row505 col14" >0.015100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row506" class="row_heading level0 row506" >507</th>
          <td id="T_78b05_row506_col0" class="data row506 col0" >None</td>
          <td id="T_78b05_row506_col1" class="data row506 col1" >0.029500</td>
          <td id="T_78b05_row506_col2" class="data row506 col2" >-0.032000</td>
          <td id="T_78b05_row506_col3" class="data row506 col3" >-0.023000</td>
          <td id="T_78b05_row506_col4" class="data row506 col4" >0.031000</td>
          <td id="T_78b05_row506_col5" class="data row506 col5" >-0.007200</td>
          <td id="T_78b05_row506_col6" class="data row506 col6" >0.033500</td>
          <td id="T_78b05_row506_col7" class="data row506 col7" >-0.038700</td>
          <td id="T_78b05_row506_col8" class="data row506 col8" >0.016000</td>
          <td id="T_78b05_row506_col9" class="data row506 col9" >0.002200</td>
          <td id="T_78b05_row506_col10" class="data row506 col10" >0.053800</td>
          <td id="T_78b05_row506_col11" class="data row506 col11" >0.048900</td>
          <td id="T_78b05_row506_col12" class="data row506 col12" >0.006500</td>
          <td id="T_78b05_row506_col13" class="data row506 col13" >0.037400</td>
          <td id="T_78b05_row506_col14" class="data row506 col14" >0.037000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row507" class="row_heading level0 row507" >508</th>
          <td id="T_78b05_row507_col0" class="data row507 col0" >None</td>
          <td id="T_78b05_row507_col1" class="data row507 col1" >0.036500</td>
          <td id="T_78b05_row507_col2" class="data row507 col2" >0.045600</td>
          <td id="T_78b05_row507_col3" class="data row507 col3" >0.028400</td>
          <td id="T_78b05_row507_col4" class="data row507 col4" >0.031200</td>
          <td id="T_78b05_row507_col5" class="data row507 col5" >-0.005400</td>
          <td id="T_78b05_row507_col6" class="data row507 col6" >-0.010900</td>
          <td id="T_78b05_row507_col7" class="data row507 col7" >0.033200</td>
          <td id="T_78b05_row507_col8" class="data row507 col8" >0.009000</td>
          <td id="T_78b05_row507_col9" class="data row507 col9" >0.075400</td>
          <td id="T_78b05_row507_col10" class="data row507 col10" >0.002400</td>
          <td id="T_78b05_row507_col11" class="data row507 col11" >0.049100</td>
          <td id="T_78b05_row507_col12" class="data row507 col12" >0.004800</td>
          <td id="T_78b05_row507_col13" class="data row507 col13" >0.007000</td>
          <td id="T_78b05_row507_col14" class="data row507 col14" >0.035000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row508" class="row_heading level0 row508" >509</th>
          <td id="T_78b05_row508_col0" class="data row508 col0" >PC1</td>
          <td id="T_78b05_row508_col1" class="data row508 col1" >0.022800</td>
          <td id="T_78b05_row508_col2" class="data row508 col2" >0.056300</td>
          <td id="T_78b05_row508_col3" class="data row508 col3" >-0.086300</td>
          <td id="T_78b05_row508_col4" class="data row508 col4" >0.015200</td>
          <td id="T_78b05_row508_col5" class="data row508 col5" >0.002300</td>
          <td id="T_78b05_row508_col6" class="data row508 col6" >-0.005900</td>
          <td id="T_78b05_row508_col7" class="data row508 col7" >0.011400</td>
          <td id="T_78b05_row508_col8" class="data row508 col8" >0.022800</td>
          <td id="T_78b05_row508_col9" class="data row508 col9" >0.086100</td>
          <td id="T_78b05_row508_col10" class="data row508 col10" >0.117100</td>
          <td id="T_78b05_row508_col11" class="data row508 col11" >0.033100</td>
          <td id="T_78b05_row508_col12" class="data row508 col12" >0.002900</td>
          <td id="T_78b05_row508_col13" class="data row508 col13" >0.002000</td>
          <td id="T_78b05_row508_col14" class="data row508 col14" >0.013100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row509" class="row_heading level0 row509" >510</th>
          <td id="T_78b05_row509_col0" class="data row509 col0" >None</td>
          <td id="T_78b05_row509_col1" class="data row509 col1" >0.049400</td>
          <td id="T_78b05_row509_col2" class="data row509 col2" >0.014700</td>
          <td id="T_78b05_row509_col3" class="data row509 col3" >0.054000</td>
          <td id="T_78b05_row509_col4" class="data row509 col4" >0.001200</td>
          <td id="T_78b05_row509_col5" class="data row509 col5" >0.017400</td>
          <td id="T_78b05_row509_col6" class="data row509 col6" >0.029900</td>
          <td id="T_78b05_row509_col7" class="data row509 col7" >-0.016100</td>
          <td id="T_78b05_row509_col8" class="data row509 col8" >0.003900</td>
          <td id="T_78b05_row509_col9" class="data row509 col9" >0.044500</td>
          <td id="T_78b05_row509_col10" class="data row509 col10" >0.023200</td>
          <td id="T_78b05_row509_col11" class="data row509 col11" >0.019100</td>
          <td id="T_78b05_row509_col12" class="data row509 col12" >0.018000</td>
          <td id="T_78b05_row509_col13" class="data row509 col13" >0.033800</td>
          <td id="T_78b05_row509_col14" class="data row509 col14" >0.014400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row510" class="row_heading level0 row510" >511</th>
          <td id="T_78b05_row510_col0" class="data row510 col0" >None</td>
          <td id="T_78b05_row510_col1" class="data row510 col1" >0.036800</td>
          <td id="T_78b05_row510_col2" class="data row510 col2" >-0.011500</td>
          <td id="T_78b05_row510_col3" class="data row510 col3" >-0.030700</td>
          <td id="T_78b05_row510_col4" class="data row510 col4" >-0.028800</td>
          <td id="T_78b05_row510_col5" class="data row510 col5" >-0.024700</td>
          <td id="T_78b05_row510_col6" class="data row510 col6" >0.016200</td>
          <td id="T_78b05_row510_col7" class="data row510 col7" >-0.008200</td>
          <td id="T_78b05_row510_col8" class="data row510 col8" >0.008700</td>
          <td id="T_78b05_row510_col9" class="data row510 col9" >0.018300</td>
          <td id="T_78b05_row510_col10" class="data row510 col10" >0.061400</td>
          <td id="T_78b05_row510_col11" class="data row510 col11" >0.010900</td>
          <td id="T_78b05_row510_col12" class="data row510 col12" >0.024100</td>
          <td id="T_78b05_row510_col13" class="data row510 col13" >0.020100</td>
          <td id="T_78b05_row510_col14" class="data row510 col14" >0.006500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row511" class="row_heading level0 row511" >512</th>
          <td id="T_78b05_row511_col0" class="data row511 col0" >None</td>
          <td id="T_78b05_row511_col1" class="data row511 col1" >0.036600</td>
          <td id="T_78b05_row511_col2" class="data row511 col2" >0.034300</td>
          <td id="T_78b05_row511_col3" class="data row511 col3" >-0.035700</td>
          <td id="T_78b05_row511_col4" class="data row511 col4" >0.010000</td>
          <td id="T_78b05_row511_col5" class="data row511 col5" >0.066900</td>
          <td id="T_78b05_row511_col6" class="data row511 col6" >0.023600</td>
          <td id="T_78b05_row511_col7" class="data row511 col7" >-0.016500</td>
          <td id="T_78b05_row511_col8" class="data row511 col8" >0.008900</td>
          <td id="T_78b05_row511_col9" class="data row511 col9" >0.064200</td>
          <td id="T_78b05_row511_col10" class="data row511 col10" >0.066500</td>
          <td id="T_78b05_row511_col11" class="data row511 col11" >0.027900</td>
          <td id="T_78b05_row511_col12" class="data row511 col12" >0.067600</td>
          <td id="T_78b05_row511_col13" class="data row511 col13" >0.027500</td>
          <td id="T_78b05_row511_col14" class="data row511 col14" >0.014700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row512" class="row_heading level0 row512" >513</th>
          <td id="T_78b05_row512_col0" class="data row512 col0" >None</td>
          <td id="T_78b05_row512_col1" class="data row512 col1" >0.046900</td>
          <td id="T_78b05_row512_col2" class="data row512 col2" >-0.029800</td>
          <td id="T_78b05_row512_col3" class="data row512 col3" >-0.037600</td>
          <td id="T_78b05_row512_col4" class="data row512 col4" >-0.026800</td>
          <td id="T_78b05_row512_col5" class="data row512 col5" >0.078400</td>
          <td id="T_78b05_row512_col6" class="data row512 col6" >-0.019100</td>
          <td id="T_78b05_row512_col7" class="data row512 col7" >-0.015200</td>
          <td id="T_78b05_row512_col8" class="data row512 col8" >0.001400</td>
          <td id="T_78b05_row512_col9" class="data row512 col9" >0.000000</td>
          <td id="T_78b05_row512_col10" class="data row512 col10" >0.068400</td>
          <td id="T_78b05_row512_col11" class="data row512 col11" >0.008900</td>
          <td id="T_78b05_row512_col12" class="data row512 col12" >0.079000</td>
          <td id="T_78b05_row512_col13" class="data row512 col13" >0.015200</td>
          <td id="T_78b05_row512_col14" class="data row512 col14" >0.013500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row513" class="row_heading level0 row513" >514</th>
          <td id="T_78b05_row513_col0" class="data row513 col0" >None</td>
          <td id="T_78b05_row513_col1" class="data row513 col1" >0.038700</td>
          <td id="T_78b05_row513_col2" class="data row513 col2" >0.040200</td>
          <td id="T_78b05_row513_col3" class="data row513 col3" >0.084200</td>
          <td id="T_78b05_row513_col4" class="data row513 col4" >-0.019900</td>
          <td id="T_78b05_row513_col5" class="data row513 col5" >-0.009500</td>
          <td id="T_78b05_row513_col6" class="data row513 col6" >-0.010100</td>
          <td id="T_78b05_row513_col7" class="data row513 col7" >0.009700</td>
          <td id="T_78b05_row513_col8" class="data row513 col8" >0.006800</td>
          <td id="T_78b05_row513_col9" class="data row513 col9" >0.070100</td>
          <td id="T_78b05_row513_col10" class="data row513 col10" >0.053400</td>
          <td id="T_78b05_row513_col11" class="data row513 col11" >0.002000</td>
          <td id="T_78b05_row513_col12" class="data row513 col12" >0.008900</td>
          <td id="T_78b05_row513_col13" class="data row513 col13" >0.006200</td>
          <td id="T_78b05_row513_col14" class="data row513 col14" >0.011400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row514" class="row_heading level0 row514" >515</th>
          <td id="T_78b05_row514_col0" class="data row514 col0" >None</td>
          <td id="T_78b05_row514_col1" class="data row514 col1" >0.045000</td>
          <td id="T_78b05_row514_col2" class="data row514 col2" >-0.062800</td>
          <td id="T_78b05_row514_col3" class="data row514 col3" >0.026000</td>
          <td id="T_78b05_row514_col4" class="data row514 col4" >-0.011300</td>
          <td id="T_78b05_row514_col5" class="data row514 col5" >-0.021700</td>
          <td id="T_78b05_row514_col6" class="data row514 col6" >0.024100</td>
          <td id="T_78b05_row514_col7" class="data row514 col7" >0.012300</td>
          <td id="T_78b05_row514_col8" class="data row514 col8" >0.000500</td>
          <td id="T_78b05_row514_col9" class="data row514 col9" >0.032900</td>
          <td id="T_78b05_row514_col10" class="data row514 col10" >0.004700</td>
          <td id="T_78b05_row514_col11" class="data row514 col11" >0.006600</td>
          <td id="T_78b05_row514_col12" class="data row514 col12" >0.021000</td>
          <td id="T_78b05_row514_col13" class="data row514 col13" >0.028000</td>
          <td id="T_78b05_row514_col14" class="data row514 col14" >0.014000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row515" class="row_heading level0 row515" >516</th>
          <td id="T_78b05_row515_col0" class="data row515 col0" >None</td>
          <td id="T_78b05_row515_col1" class="data row515 col1" >0.040700</td>
          <td id="T_78b05_row515_col2" class="data row515 col2" >-0.008000</td>
          <td id="T_78b05_row515_col3" class="data row515 col3" >0.013900</td>
          <td id="T_78b05_row515_col4" class="data row515 col4" >-0.010500</td>
          <td id="T_78b05_row515_col5" class="data row515 col5" >-0.007900</td>
          <td id="T_78b05_row515_col6" class="data row515 col6" >-0.016100</td>
          <td id="T_78b05_row515_col7" class="data row515 col7" >-0.032600</td>
          <td id="T_78b05_row515_col8" class="data row515 col8" >0.004800</td>
          <td id="T_78b05_row515_col9" class="data row515 col9" >0.021800</td>
          <td id="T_78b05_row515_col10" class="data row515 col10" >0.016900</td>
          <td id="T_78b05_row515_col11" class="data row515 col11" >0.007400</td>
          <td id="T_78b05_row515_col12" class="data row515 col12" >0.007300</td>
          <td id="T_78b05_row515_col13" class="data row515 col13" >0.012200</td>
          <td id="T_78b05_row515_col14" class="data row515 col14" >0.030800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row516" class="row_heading level0 row516" >517</th>
          <td id="T_78b05_row516_col0" class="data row516 col0" >None</td>
          <td id="T_78b05_row516_col1" class="data row516 col1" >0.041100</td>
          <td id="T_78b05_row516_col2" class="data row516 col2" >-0.039600</td>
          <td id="T_78b05_row516_col3" class="data row516 col3" >-0.010900</td>
          <td id="T_78b05_row516_col4" class="data row516 col4" >-0.002800</td>
          <td id="T_78b05_row516_col5" class="data row516 col5" >-0.048700</td>
          <td id="T_78b05_row516_col6" class="data row516 col6" >-0.009900</td>
          <td id="T_78b05_row516_col7" class="data row516 col7" >-0.011600</td>
          <td id="T_78b05_row516_col8" class="data row516 col8" >0.004400</td>
          <td id="T_78b05_row516_col9" class="data row516 col9" >0.009800</td>
          <td id="T_78b05_row516_col10" class="data row516 col10" >0.041600</td>
          <td id="T_78b05_row516_col11" class="data row516 col11" >0.015200</td>
          <td id="T_78b05_row516_col12" class="data row516 col12" >0.048100</td>
          <td id="T_78b05_row516_col13" class="data row516 col13" >0.006000</td>
          <td id="T_78b05_row516_col14" class="data row516 col14" >0.009900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row517" class="row_heading level0 row517" >518</th>
          <td id="T_78b05_row517_col0" class="data row517 col0" >None</td>
          <td id="T_78b05_row517_col1" class="data row517 col1" >0.036500</td>
          <td id="T_78b05_row517_col2" class="data row517 col2" >-0.031900</td>
          <td id="T_78b05_row517_col3" class="data row517 col3" >-0.066600</td>
          <td id="T_78b05_row517_col4" class="data row517 col4" >-0.028700</td>
          <td id="T_78b05_row517_col5" class="data row517 col5" >0.008300</td>
          <td id="T_78b05_row517_col6" class="data row517 col6" >-0.026900</td>
          <td id="T_78b05_row517_col7" class="data row517 col7" >-0.001500</td>
          <td id="T_78b05_row517_col8" class="data row517 col8" >0.009000</td>
          <td id="T_78b05_row517_col9" class="data row517 col9" >0.002100</td>
          <td id="T_78b05_row517_col10" class="data row517 col10" >0.097300</td>
          <td id="T_78b05_row517_col11" class="data row517 col11" >0.010800</td>
          <td id="T_78b05_row517_col12" class="data row517 col12" >0.008900</td>
          <td id="T_78b05_row517_col13" class="data row517 col13" >0.023000</td>
          <td id="T_78b05_row517_col14" class="data row517 col14" >0.000200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row518" class="row_heading level0 row518" >519</th>
          <td id="T_78b05_row518_col0" class="data row518 col0" >None</td>
          <td id="T_78b05_row518_col1" class="data row518 col1" >0.035800</td>
          <td id="T_78b05_row518_col2" class="data row518 col2" >-0.025600</td>
          <td id="T_78b05_row518_col3" class="data row518 col3" >-0.012700</td>
          <td id="T_78b05_row518_col4" class="data row518 col4" >0.028300</td>
          <td id="T_78b05_row518_col5" class="data row518 col5" >-0.066100</td>
          <td id="T_78b05_row518_col6" class="data row518 col6" >-0.042800</td>
          <td id="T_78b05_row518_col7" class="data row518 col7" >0.061300</td>
          <td id="T_78b05_row518_col8" class="data row518 col8" >0.009700</td>
          <td id="T_78b05_row518_col9" class="data row518 col9" >0.004200</td>
          <td id="T_78b05_row518_col10" class="data row518 col10" >0.043400</td>
          <td id="T_78b05_row518_col11" class="data row518 col11" >0.046300</td>
          <td id="T_78b05_row518_col12" class="data row518 col12" >0.065500</td>
          <td id="T_78b05_row518_col13" class="data row518 col13" >0.039000</td>
          <td id="T_78b05_row518_col14" class="data row518 col14" >0.063100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row519" class="row_heading level0 row519" >520</th>
          <td id="T_78b05_row519_col0" class="data row519 col0" >None</td>
          <td id="T_78b05_row519_col1" class="data row519 col1" >0.047100</td>
          <td id="T_78b05_row519_col2" class="data row519 col2" >-0.060100</td>
          <td id="T_78b05_row519_col3" class="data row519 col3" >-0.006600</td>
          <td id="T_78b05_row519_col4" class="data row519 col4" >0.015400</td>
          <td id="T_78b05_row519_col5" class="data row519 col5" >0.001800</td>
          <td id="T_78b05_row519_col6" class="data row519 col6" >-0.021400</td>
          <td id="T_78b05_row519_col7" class="data row519 col7" >0.033400</td>
          <td id="T_78b05_row519_col8" class="data row519 col8" >0.001600</td>
          <td id="T_78b05_row519_col9" class="data row519 col9" >0.030200</td>
          <td id="T_78b05_row519_col10" class="data row519 col10" >0.037400</td>
          <td id="T_78b05_row519_col11" class="data row519 col11" >0.033300</td>
          <td id="T_78b05_row519_col12" class="data row519 col12" >0.002400</td>
          <td id="T_78b05_row519_col13" class="data row519 col13" >0.017500</td>
          <td id="T_78b05_row519_col14" class="data row519 col14" >0.035100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row520" class="row_heading level0 row520" >521</th>
          <td id="T_78b05_row520_col0" class="data row520 col0" >None</td>
          <td id="T_78b05_row520_col1" class="data row520 col1" >0.050100</td>
          <td id="T_78b05_row520_col2" class="data row520 col2" >-0.018100</td>
          <td id="T_78b05_row520_col3" class="data row520 col3" >0.058000</td>
          <td id="T_78b05_row520_col4" class="data row520 col4" >-0.016100</td>
          <td id="T_78b05_row520_col5" class="data row520 col5" >0.048900</td>
          <td id="T_78b05_row520_col6" class="data row520 col6" >-0.011300</td>
          <td id="T_78b05_row520_col7" class="data row520 col7" >0.011400</td>
          <td id="T_78b05_row520_col8" class="data row520 col8" >0.004500</td>
          <td id="T_78b05_row520_col9" class="data row520 col9" >0.011700</td>
          <td id="T_78b05_row520_col10" class="data row520 col10" >0.027200</td>
          <td id="T_78b05_row520_col11" class="data row520 col11" >0.001800</td>
          <td id="T_78b05_row520_col12" class="data row520 col12" >0.049600</td>
          <td id="T_78b05_row520_col13" class="data row520 col13" >0.007400</td>
          <td id="T_78b05_row520_col14" class="data row520 col14" >0.013100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row521" class="row_heading level0 row521" >522</th>
          <td id="T_78b05_row521_col0" class="data row521 col0" >None</td>
          <td id="T_78b05_row521_col1" class="data row521 col1" >0.031100</td>
          <td id="T_78b05_row521_col2" class="data row521 col2" >0.004100</td>
          <td id="T_78b05_row521_col3" class="data row521 col3" >-0.056600</td>
          <td id="T_78b05_row521_col4" class="data row521 col4" >0.047800</td>
          <td id="T_78b05_row521_col5" class="data row521 col5" >-0.044800</td>
          <td id="T_78b05_row521_col6" class="data row521 col6" >0.021200</td>
          <td id="T_78b05_row521_col7" class="data row521 col7" >0.012400</td>
          <td id="T_78b05_row521_col8" class="data row521 col8" >0.014500</td>
          <td id="T_78b05_row521_col9" class="data row521 col9" >0.033900</td>
          <td id="T_78b05_row521_col10" class="data row521 col10" >0.087400</td>
          <td id="T_78b05_row521_col11" class="data row521 col11" >0.065800</td>
          <td id="T_78b05_row521_col12" class="data row521 col12" >0.044100</td>
          <td id="T_78b05_row521_col13" class="data row521 col13" >0.025100</td>
          <td id="T_78b05_row521_col14" class="data row521 col14" >0.014200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row522" class="row_heading level0 row522" >523</th>
          <td id="T_78b05_row522_col0" class="data row522 col0" >None</td>
          <td id="T_78b05_row522_col1" class="data row522 col1" >0.032500</td>
          <td id="T_78b05_row522_col2" class="data row522 col2" >0.042500</td>
          <td id="T_78b05_row522_col3" class="data row522 col3" >-0.016000</td>
          <td id="T_78b05_row522_col4" class="data row522 col4" >0.012200</td>
          <td id="T_78b05_row522_col5" class="data row522 col5" >-0.032700</td>
          <td id="T_78b05_row522_col6" class="data row522 col6" >0.085000</td>
          <td id="T_78b05_row522_col7" class="data row522 col7" >-0.037700</td>
          <td id="T_78b05_row522_col8" class="data row522 col8" >0.013000</td>
          <td id="T_78b05_row522_col9" class="data row522 col9" >0.072400</td>
          <td id="T_78b05_row522_col10" class="data row522 col10" >0.046800</td>
          <td id="T_78b05_row522_col11" class="data row522 col11" >0.030100</td>
          <td id="T_78b05_row522_col12" class="data row522 col12" >0.032100</td>
          <td id="T_78b05_row522_col13" class="data row522 col13" >0.088800</td>
          <td id="T_78b05_row522_col14" class="data row522 col14" >0.036000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row523" class="row_heading level0 row523" >524</th>
          <td id="T_78b05_row523_col0" class="data row523 col0" >PC3</td>
          <td id="T_78b05_row523_col1" class="data row523 col1" >0.031600</td>
          <td id="T_78b05_row523_col2" class="data row523 col2" >-0.028400</td>
          <td id="T_78b05_row523_col3" class="data row523 col3" >-0.106200</td>
          <td id="T_78b05_row523_col4" class="data row523 col4" >0.015700</td>
          <td id="T_78b05_row523_col5" class="data row523 col5" >0.040500</td>
          <td id="T_78b05_row523_col6" class="data row523 col6" >0.059700</td>
          <td id="T_78b05_row523_col7" class="data row523 col7" >-0.001400</td>
          <td id="T_78b05_row523_col8" class="data row523 col8" >0.013900</td>
          <td id="T_78b05_row523_col9" class="data row523 col9" >0.001500</td>
          <td id="T_78b05_row523_col10" class="data row523 col10" >0.136900</td>
          <td id="T_78b05_row523_col11" class="data row523 col11" >0.033600</td>
          <td id="T_78b05_row523_col12" class="data row523 col12" >0.041100</td>
          <td id="T_78b05_row523_col13" class="data row523 col13" >0.063600</td>
          <td id="T_78b05_row523_col14" class="data row523 col14" >0.000400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row524" class="row_heading level0 row524" >525</th>
          <td id="T_78b05_row524_col0" class="data row524 col0" >None</td>
          <td id="T_78b05_row524_col1" class="data row524 col1" >0.032800</td>
          <td id="T_78b05_row524_col2" class="data row524 col2" >0.037500</td>
          <td id="T_78b05_row524_col3" class="data row524 col3" >-0.002400</td>
          <td id="T_78b05_row524_col4" class="data row524 col4" >-0.040400</td>
          <td id="T_78b05_row524_col5" class="data row524 col5" >0.061800</td>
          <td id="T_78b05_row524_col6" class="data row524 col6" >-0.012200</td>
          <td id="T_78b05_row524_col7" class="data row524 col7" >0.004000</td>
          <td id="T_78b05_row524_col8" class="data row524 col8" >0.012700</td>
          <td id="T_78b05_row524_col9" class="data row524 col9" >0.067300</td>
          <td id="T_78b05_row524_col10" class="data row524 col10" >0.033200</td>
          <td id="T_78b05_row524_col11" class="data row524 col11" >0.022500</td>
          <td id="T_78b05_row524_col12" class="data row524 col12" >0.062500</td>
          <td id="T_78b05_row524_col13" class="data row524 col13" >0.008300</td>
          <td id="T_78b05_row524_col14" class="data row524 col14" >0.005700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row525" class="row_heading level0 row525" >526</th>
          <td id="T_78b05_row525_col0" class="data row525 col0" >PC1</td>
          <td id="T_78b05_row525_col1" class="data row525 col1" >0.022500</td>
          <td id="T_78b05_row525_col2" class="data row525 col2" >0.055700</td>
          <td id="T_78b05_row525_col3" class="data row525 col3" >-0.038200</td>
          <td id="T_78b05_row525_col4" class="data row525 col4" >0.037200</td>
          <td id="T_78b05_row525_col5" class="data row525 col5" >0.031800</td>
          <td id="T_78b05_row525_col6" class="data row525 col6" >-0.030200</td>
          <td id="T_78b05_row525_col7" class="data row525 col7" >0.002900</td>
          <td id="T_78b05_row525_col8" class="data row525 col8" >0.023000</td>
          <td id="T_78b05_row525_col9" class="data row525 col9" >0.085500</td>
          <td id="T_78b05_row525_col10" class="data row525 col10" >0.069000</td>
          <td id="T_78b05_row525_col11" class="data row525 col11" >0.055100</td>
          <td id="T_78b05_row525_col12" class="data row525 col12" >0.032400</td>
          <td id="T_78b05_row525_col13" class="data row525 col13" >0.026300</td>
          <td id="T_78b05_row525_col14" class="data row525 col14" >0.004600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row526" class="row_heading level0 row526" >527</th>
          <td id="T_78b05_row526_col0" class="data row526 col0" >None</td>
          <td id="T_78b05_row526_col1" class="data row526 col1" >0.034000</td>
          <td id="T_78b05_row526_col2" class="data row526 col2" >0.024600</td>
          <td id="T_78b05_row526_col3" class="data row526 col3" >-0.054100</td>
          <td id="T_78b05_row526_col4" class="data row526 col4" >-0.005300</td>
          <td id="T_78b05_row526_col5" class="data row526 col5" >0.089600</td>
          <td id="T_78b05_row526_col6" class="data row526 col6" >-0.019200</td>
          <td id="T_78b05_row526_col7" class="data row526 col7" >-0.027500</td>
          <td id="T_78b05_row526_col8" class="data row526 col8" >0.011600</td>
          <td id="T_78b05_row526_col9" class="data row526 col9" >0.054500</td>
          <td id="T_78b05_row526_col10" class="data row526 col10" >0.084800</td>
          <td id="T_78b05_row526_col11" class="data row526 col11" >0.012600</td>
          <td id="T_78b05_row526_col12" class="data row526 col12" >0.090200</td>
          <td id="T_78b05_row526_col13" class="data row526 col13" >0.015300</td>
          <td id="T_78b05_row526_col14" class="data row526 col14" >0.025700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row527" class="row_heading level0 row527" >528</th>
          <td id="T_78b05_row527_col0" class="data row527 col0" >None</td>
          <td id="T_78b05_row527_col1" class="data row527 col1" >0.032400</td>
          <td id="T_78b05_row527_col2" class="data row527 col2" >-0.015100</td>
          <td id="T_78b05_row527_col3" class="data row527 col3" >0.002300</td>
          <td id="T_78b05_row527_col4" class="data row527 col4" >0.050600</td>
          <td id="T_78b05_row527_col5" class="data row527 col5" >-0.010500</td>
          <td id="T_78b05_row527_col6" class="data row527 col6" >-0.009500</td>
          <td id="T_78b05_row527_col7" class="data row527 col7" >-0.051200</td>
          <td id="T_78b05_row527_col8" class="data row527 col8" >0.013100</td>
          <td id="T_78b05_row527_col9" class="data row527 col9" >0.014700</td>
          <td id="T_78b05_row527_col10" class="data row527 col10" >0.028500</td>
          <td id="T_78b05_row527_col11" class="data row527 col11" >0.068500</td>
          <td id="T_78b05_row527_col12" class="data row527 col12" >0.009900</td>
          <td id="T_78b05_row527_col13" class="data row527 col13" >0.005600</td>
          <td id="T_78b05_row527_col14" class="data row527 col14" >0.049400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row528" class="row_heading level0 row528" >529</th>
          <td id="T_78b05_row528_col0" class="data row528 col0" >None</td>
          <td id="T_78b05_row528_col1" class="data row528 col1" >0.038700</td>
          <td id="T_78b05_row528_col2" class="data row528 col2" >-0.046200</td>
          <td id="T_78b05_row528_col3" class="data row528 col3" >-0.064200</td>
          <td id="T_78b05_row528_col4" class="data row528 col4" >-0.026400</td>
          <td id="T_78b05_row528_col5" class="data row528 col5" >0.023600</td>
          <td id="T_78b05_row528_col6" class="data row528 col6" >0.064800</td>
          <td id="T_78b05_row528_col7" class="data row528 col7" >0.002400</td>
          <td id="T_78b05_row528_col8" class="data row528 col8" >0.006900</td>
          <td id="T_78b05_row528_col9" class="data row528 col9" >0.016400</td>
          <td id="T_78b05_row528_col10" class="data row528 col10" >0.095000</td>
          <td id="T_78b05_row528_col11" class="data row528 col11" >0.008500</td>
          <td id="T_78b05_row528_col12" class="data row528 col12" >0.024300</td>
          <td id="T_78b05_row528_col13" class="data row528 col13" >0.068700</td>
          <td id="T_78b05_row528_col14" class="data row528 col14" >0.004200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row529" class="row_heading level0 row529" >530</th>
          <td id="T_78b05_row529_col0" class="data row529 col0" >None</td>
          <td id="T_78b05_row529_col1" class="data row529 col1" >0.035300</td>
          <td id="T_78b05_row529_col2" class="data row529 col2" >0.055000</td>
          <td id="T_78b05_row529_col3" class="data row529 col3" >-0.019900</td>
          <td id="T_78b05_row529_col4" class="data row529 col4" >-0.088100</td>
          <td id="T_78b05_row529_col5" class="data row529 col5" >-0.066700</td>
          <td id="T_78b05_row529_col6" class="data row529 col6" >-0.008300</td>
          <td id="T_78b05_row529_col7" class="data row529 col7" >0.014000</td>
          <td id="T_78b05_row529_col8" class="data row529 col8" >0.010200</td>
          <td id="T_78b05_row529_col9" class="data row529 col9" >0.084800</td>
          <td id="T_78b05_row529_col10" class="data row529 col10" >0.050600</td>
          <td id="T_78b05_row529_col11" class="data row529 col11" >0.070200</td>
          <td id="T_78b05_row529_col12" class="data row529 col12" >0.066100</td>
          <td id="T_78b05_row529_col13" class="data row529 col13" >0.004400</td>
          <td id="T_78b05_row529_col14" class="data row529 col14" >0.015800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row530" class="row_heading level0 row530" >531</th>
          <td id="T_78b05_row530_col0" class="data row530 col0" >None</td>
          <td id="T_78b05_row530_col1" class="data row530 col1" >0.032500</td>
          <td id="T_78b05_row530_col2" class="data row530 col2" >-0.044100</td>
          <td id="T_78b05_row530_col3" class="data row530 col3" >-0.037300</td>
          <td id="T_78b05_row530_col4" class="data row530 col4" >0.052500</td>
          <td id="T_78b05_row530_col5" class="data row530 col5" >-0.056000</td>
          <td id="T_78b05_row530_col6" class="data row530 col6" >0.032500</td>
          <td id="T_78b05_row530_col7" class="data row530 col7" >-0.052900</td>
          <td id="T_78b05_row530_col8" class="data row530 col8" >0.013000</td>
          <td id="T_78b05_row530_col9" class="data row530 col9" >0.014300</td>
          <td id="T_78b05_row530_col10" class="data row530 col10" >0.068000</td>
          <td id="T_78b05_row530_col11" class="data row530 col11" >0.070400</td>
          <td id="T_78b05_row530_col12" class="data row530 col12" >0.055400</td>
          <td id="T_78b05_row530_col13" class="data row530 col13" >0.036300</td>
          <td id="T_78b05_row530_col14" class="data row530 col14" >0.051100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row531" class="row_heading level0 row531" >532</th>
          <td id="T_78b05_row531_col0" class="data row531 col0" >None</td>
          <td id="T_78b05_row531_col1" class="data row531 col1" >0.031400</td>
          <td id="T_78b05_row531_col2" class="data row531 col2" >0.056700</td>
          <td id="T_78b05_row531_col3" class="data row531 col3" >-0.007400</td>
          <td id="T_78b05_row531_col4" class="data row531 col4" >0.072000</td>
          <td id="T_78b05_row531_col5" class="data row531 col5" >0.009700</td>
          <td id="T_78b05_row531_col6" class="data row531 col6" >-0.029700</td>
          <td id="T_78b05_row531_col7" class="data row531 col7" >-0.006700</td>
          <td id="T_78b05_row531_col8" class="data row531 col8" >0.014200</td>
          <td id="T_78b05_row531_col9" class="data row531 col9" >0.086500</td>
          <td id="T_78b05_row531_col10" class="data row531 col10" >0.038200</td>
          <td id="T_78b05_row531_col11" class="data row531 col11" >0.090000</td>
          <td id="T_78b05_row531_col12" class="data row531 col12" >0.010300</td>
          <td id="T_78b05_row531_col13" class="data row531 col13" >0.025800</td>
          <td id="T_78b05_row531_col14" class="data row531 col14" >0.005000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row532" class="row_heading level0 row532" >533</th>
          <td id="T_78b05_row532_col0" class="data row532 col0" >PC2</td>
          <td id="T_78b05_row532_col1" class="data row532 col1" >0.039300</td>
          <td id="T_78b05_row532_col2" class="data row532 col2" >0.094200</td>
          <td id="T_78b05_row532_col3" class="data row532 col3" >0.045800</td>
          <td id="T_78b05_row532_col4" class="data row532 col4" >0.021700</td>
          <td id="T_78b05_row532_col5" class="data row532 col5" >0.012800</td>
          <td id="T_78b05_row532_col6" class="data row532 col6" >0.032800</td>
          <td id="T_78b05_row532_col7" class="data row532 col7" >0.024200</td>
          <td id="T_78b05_row532_col8" class="data row532 col8" >0.006200</td>
          <td id="T_78b05_row532_col9" class="data row532 col9" >0.124000</td>
          <td id="T_78b05_row532_col10" class="data row532 col10" >0.015100</td>
          <td id="T_78b05_row532_col11" class="data row532 col11" >0.039600</td>
          <td id="T_78b05_row532_col12" class="data row532 col12" >0.013400</td>
          <td id="T_78b05_row532_col13" class="data row532 col13" >0.036700</td>
          <td id="T_78b05_row532_col14" class="data row532 col14" >0.026000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row533" class="row_heading level0 row533" >534</th>
          <td id="T_78b05_row533_col0" class="data row533 col0" >PC1</td>
          <td id="T_78b05_row533_col1" class="data row533 col1" >0.026100</td>
          <td id="T_78b05_row533_col2" class="data row533 col2" >0.032300</td>
          <td id="T_78b05_row533_col3" class="data row533 col3" >0.019400</td>
          <td id="T_78b05_row533_col4" class="data row533 col4" >-0.043500</td>
          <td id="T_78b05_row533_col5" class="data row533 col5" >-0.006900</td>
          <td id="T_78b05_row533_col6" class="data row533 col6" >-0.025500</td>
          <td id="T_78b05_row533_col7" class="data row533 col7" >-0.013200</td>
          <td id="T_78b05_row533_col8" class="data row533 col8" >0.019500</td>
          <td id="T_78b05_row533_col9" class="data row533 col9" >0.062100</td>
          <td id="T_78b05_row533_col10" class="data row533 col10" >0.011300</td>
          <td id="T_78b05_row533_col11" class="data row533 col11" >0.025600</td>
          <td id="T_78b05_row533_col12" class="data row533 col12" >0.006300</td>
          <td id="T_78b05_row533_col13" class="data row533 col13" >0.021600</td>
          <td id="T_78b05_row533_col14" class="data row533 col14" >0.011400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row534" class="row_heading level0 row534" >535</th>
          <td id="T_78b05_row534_col0" class="data row534 col0" >None</td>
          <td id="T_78b05_row534_col1" class="data row534 col1" >0.038700</td>
          <td id="T_78b05_row534_col2" class="data row534 col2" >-0.047200</td>
          <td id="T_78b05_row534_col3" class="data row534 col3" >-0.006000</td>
          <td id="T_78b05_row534_col4" class="data row534 col4" >-0.036900</td>
          <td id="T_78b05_row534_col5" class="data row534 col5" >0.022200</td>
          <td id="T_78b05_row534_col6" class="data row534 col6" >0.041800</td>
          <td id="T_78b05_row534_col7" class="data row534 col7" >0.032200</td>
          <td id="T_78b05_row534_col8" class="data row534 col8" >0.006800</td>
          <td id="T_78b05_row534_col9" class="data row534 col9" >0.017400</td>
          <td id="T_78b05_row534_col10" class="data row534 col10" >0.036700</td>
          <td id="T_78b05_row534_col11" class="data row534 col11" >0.018900</td>
          <td id="T_78b05_row534_col12" class="data row534 col12" >0.022900</td>
          <td id="T_78b05_row534_col13" class="data row534 col13" >0.045700</td>
          <td id="T_78b05_row534_col14" class="data row534 col14" >0.034000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row535" class="row_heading level0 row535" >536</th>
          <td id="T_78b05_row535_col0" class="data row535 col0" >None</td>
          <td id="T_78b05_row535_col1" class="data row535 col1" >0.038700</td>
          <td id="T_78b05_row535_col2" class="data row535 col2" >0.036900</td>
          <td id="T_78b05_row535_col3" class="data row535 col3" >0.008700</td>
          <td id="T_78b05_row535_col4" class="data row535 col4" >-0.000600</td>
          <td id="T_78b05_row535_col5" class="data row535 col5" >0.012800</td>
          <td id="T_78b05_row535_col6" class="data row535 col6" >0.035000</td>
          <td id="T_78b05_row535_col7" class="data row535 col7" >0.040300</td>
          <td id="T_78b05_row535_col8" class="data row535 col8" >0.006800</td>
          <td id="T_78b05_row535_col9" class="data row535 col9" >0.066700</td>
          <td id="T_78b05_row535_col10" class="data row535 col10" >0.022000</td>
          <td id="T_78b05_row535_col11" class="data row535 col11" >0.017300</td>
          <td id="T_78b05_row535_col12" class="data row535 col12" >0.013500</td>
          <td id="T_78b05_row535_col13" class="data row535 col13" >0.038900</td>
          <td id="T_78b05_row535_col14" class="data row535 col14" >0.042000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row536" class="row_heading level0 row536" >537</th>
          <td id="T_78b05_row536_col0" class="data row536 col0" >None</td>
          <td id="T_78b05_row536_col1" class="data row536 col1" >0.036500</td>
          <td id="T_78b05_row536_col2" class="data row536 col2" >0.011300</td>
          <td id="T_78b05_row536_col3" class="data row536 col3" >-0.011200</td>
          <td id="T_78b05_row536_col4" class="data row536 col4" >-0.030900</td>
          <td id="T_78b05_row536_col5" class="data row536 col5" >-0.037400</td>
          <td id="T_78b05_row536_col6" class="data row536 col6" >0.015100</td>
          <td id="T_78b05_row536_col7" class="data row536 col7" >0.042000</td>
          <td id="T_78b05_row536_col8" class="data row536 col8" >0.009000</td>
          <td id="T_78b05_row536_col9" class="data row536 col9" >0.041100</td>
          <td id="T_78b05_row536_col10" class="data row536 col10" >0.041900</td>
          <td id="T_78b05_row536_col11" class="data row536 col11" >0.013000</td>
          <td id="T_78b05_row536_col12" class="data row536 col12" >0.036800</td>
          <td id="T_78b05_row536_col13" class="data row536 col13" >0.019000</td>
          <td id="T_78b05_row536_col14" class="data row536 col14" >0.043700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row537" class="row_heading level0 row537" >538</th>
          <td id="T_78b05_row537_col0" class="data row537 col0" >None</td>
          <td id="T_78b05_row537_col1" class="data row537 col1" >0.036400</td>
          <td id="T_78b05_row537_col2" class="data row537 col2" >0.050900</td>
          <td id="T_78b05_row537_col3" class="data row537 col3" >-0.043300</td>
          <td id="T_78b05_row537_col4" class="data row537 col4" >0.037900</td>
          <td id="T_78b05_row537_col5" class="data row537 col5" >0.049100</td>
          <td id="T_78b05_row537_col6" class="data row537 col6" >-0.081100</td>
          <td id="T_78b05_row537_col7" class="data row537 col7" >0.002900</td>
          <td id="T_78b05_row537_col8" class="data row537 col8" >0.009100</td>
          <td id="T_78b05_row537_col9" class="data row537 col9" >0.080700</td>
          <td id="T_78b05_row537_col10" class="data row537 col10" >0.074100</td>
          <td id="T_78b05_row537_col11" class="data row537 col11" >0.055800</td>
          <td id="T_78b05_row537_col12" class="data row537 col12" >0.049800</td>
          <td id="T_78b05_row537_col13" class="data row537 col13" >0.077200</td>
          <td id="T_78b05_row537_col14" class="data row537 col14" >0.004600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row538" class="row_heading level0 row538" >539</th>
          <td id="T_78b05_row538_col0" class="data row538 col0" >None</td>
          <td id="T_78b05_row538_col1" class="data row538 col1" >0.045800</td>
          <td id="T_78b05_row538_col2" class="data row538 col2" >-0.034000</td>
          <td id="T_78b05_row538_col3" class="data row538 col3" >0.020800</td>
          <td id="T_78b05_row538_col4" class="data row538 col4" >0.058700</td>
          <td id="T_78b05_row538_col5" class="data row538 col5" >0.091900</td>
          <td id="T_78b05_row538_col6" class="data row538 col6" >0.009900</td>
          <td id="T_78b05_row538_col7" class="data row538 col7" >0.013700</td>
          <td id="T_78b05_row538_col8" class="data row538 col8" >0.000200</td>
          <td id="T_78b05_row538_col9" class="data row538 col9" >0.004100</td>
          <td id="T_78b05_row538_col10" class="data row538 col10" >0.010000</td>
          <td id="T_78b05_row538_col11" class="data row538 col11" >0.076600</td>
          <td id="T_78b05_row538_col12" class="data row538 col12" >0.092500</td>
          <td id="T_78b05_row538_col13" class="data row538 col13" >0.013800</td>
          <td id="T_78b05_row538_col14" class="data row538 col14" >0.015500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row539" class="row_heading level0 row539" >540</th>
          <td id="T_78b05_row539_col0" class="data row539 col0" >None</td>
          <td id="T_78b05_row539_col1" class="data row539 col1" >0.041800</td>
          <td id="T_78b05_row539_col2" class="data row539 col2" >0.025700</td>
          <td id="T_78b05_row539_col3" class="data row539 col3" >-0.027100</td>
          <td id="T_78b05_row539_col4" class="data row539 col4" >0.047600</td>
          <td id="T_78b05_row539_col5" class="data row539 col5" >0.029200</td>
          <td id="T_78b05_row539_col6" class="data row539 col6" >0.027000</td>
          <td id="T_78b05_row539_col7" class="data row539 col7" >0.004400</td>
          <td id="T_78b05_row539_col8" class="data row539 col8" >0.003700</td>
          <td id="T_78b05_row539_col9" class="data row539 col9" >0.055500</td>
          <td id="T_78b05_row539_col10" class="data row539 col10" >0.057900</td>
          <td id="T_78b05_row539_col11" class="data row539 col11" >0.065500</td>
          <td id="T_78b05_row539_col12" class="data row539 col12" >0.029900</td>
          <td id="T_78b05_row539_col13" class="data row539 col13" >0.030900</td>
          <td id="T_78b05_row539_col14" class="data row539 col14" >0.006100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row540" class="row_heading level0 row540" >541</th>
          <td id="T_78b05_row540_col0" class="data row540 col0" >None</td>
          <td id="T_78b05_row540_col1" class="data row540 col1" >0.033400</td>
          <td id="T_78b05_row540_col2" class="data row540 col2" >0.054600</td>
          <td id="T_78b05_row540_col3" class="data row540 col3" >0.030800</td>
          <td id="T_78b05_row540_col4" class="data row540 col4" >0.021300</td>
          <td id="T_78b05_row540_col5" class="data row540 col5" >0.051900</td>
          <td id="T_78b05_row540_col6" class="data row540 col6" >0.007100</td>
          <td id="T_78b05_row540_col7" class="data row540 col7" >-0.027300</td>
          <td id="T_78b05_row540_col8" class="data row540 col8" >0.012100</td>
          <td id="T_78b05_row540_col9" class="data row540 col9" >0.084500</td>
          <td id="T_78b05_row540_col10" class="data row540 col10" >0.000000</td>
          <td id="T_78b05_row540_col11" class="data row540 col11" >0.039200</td>
          <td id="T_78b05_row540_col12" class="data row540 col12" >0.052600</td>
          <td id="T_78b05_row540_col13" class="data row540 col13" >0.011000</td>
          <td id="T_78b05_row540_col14" class="data row540 col14" >0.025500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row541" class="row_heading level0 row541" >542</th>
          <td id="T_78b05_row541_col0" class="data row541 col0" >PC1</td>
          <td id="T_78b05_row541_col1" class="data row541 col1" >0.026400</td>
          <td id="T_78b05_row541_col2" class="data row541 col2" >0.039100</td>
          <td id="T_78b05_row541_col3" class="data row541 col3" >-0.049200</td>
          <td id="T_78b05_row541_col4" class="data row541 col4" >-0.007200</td>
          <td id="T_78b05_row541_col5" class="data row541 col5" >0.041200</td>
          <td id="T_78b05_row541_col6" class="data row541 col6" >-0.043500</td>
          <td id="T_78b05_row541_col7" class="data row541 col7" >0.018300</td>
          <td id="T_78b05_row541_col8" class="data row541 col8" >0.019200</td>
          <td id="T_78b05_row541_col9" class="data row541 col9" >0.068900</td>
          <td id="T_78b05_row541_col10" class="data row541 col10" >0.080000</td>
          <td id="T_78b05_row541_col11" class="data row541 col11" >0.010700</td>
          <td id="T_78b05_row541_col12" class="data row541 col12" >0.041900</td>
          <td id="T_78b05_row541_col13" class="data row541 col13" >0.039600</td>
          <td id="T_78b05_row541_col14" class="data row541 col14" >0.020100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row542" class="row_heading level0 row542" >543</th>
          <td id="T_78b05_row542_col0" class="data row542 col0" >None</td>
          <td id="T_78b05_row542_col1" class="data row542 col1" >0.039600</td>
          <td id="T_78b05_row542_col2" class="data row542 col2" >-0.050600</td>
          <td id="T_78b05_row542_col3" class="data row542 col3" >0.003100</td>
          <td id="T_78b05_row542_col4" class="data row542 col4" >-0.024300</td>
          <td id="T_78b05_row542_col5" class="data row542 col5" >-0.015800</td>
          <td id="T_78b05_row542_col6" class="data row542 col6" >-0.033400</td>
          <td id="T_78b05_row542_col7" class="data row542 col7" >-0.013000</td>
          <td id="T_78b05_row542_col8" class="data row542 col8" >0.005900</td>
          <td id="T_78b05_row542_col9" class="data row542 col9" >0.020800</td>
          <td id="T_78b05_row542_col10" class="data row542 col10" >0.027600</td>
          <td id="T_78b05_row542_col11" class="data row542 col11" >0.006400</td>
          <td id="T_78b05_row542_col12" class="data row542 col12" >0.015200</td>
          <td id="T_78b05_row542_col13" class="data row542 col13" >0.029500</td>
          <td id="T_78b05_row542_col14" class="data row542 col14" >0.011200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row543" class="row_heading level0 row543" >544</th>
          <td id="T_78b05_row543_col0" class="data row543 col0" >None</td>
          <td id="T_78b05_row543_col1" class="data row543 col1" >0.037200</td>
          <td id="T_78b05_row543_col2" class="data row543 col2" >-0.005000</td>
          <td id="T_78b05_row543_col3" class="data row543 col3" >0.010300</td>
          <td id="T_78b05_row543_col4" class="data row543 col4" >0.060500</td>
          <td id="T_78b05_row543_col5" class="data row543 col5" >-0.014300</td>
          <td id="T_78b05_row543_col6" class="data row543 col6" >-0.007500</td>
          <td id="T_78b05_row543_col7" class="data row543 col7" >-0.044900</td>
          <td id="T_78b05_row543_col8" class="data row543 col8" >0.008300</td>
          <td id="T_78b05_row543_col9" class="data row543 col9" >0.024800</td>
          <td id="T_78b05_row543_col10" class="data row543 col10" >0.020500</td>
          <td id="T_78b05_row543_col11" class="data row543 col11" >0.078400</td>
          <td id="T_78b05_row543_col12" class="data row543 col12" >0.013700</td>
          <td id="T_78b05_row543_col13" class="data row543 col13" >0.003600</td>
          <td id="T_78b05_row543_col14" class="data row543 col14" >0.043200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row544" class="row_heading level0 row544" >545</th>
          <td id="T_78b05_row544_col0" class="data row544 col0" >PC1</td>
          <td id="T_78b05_row544_col1" class="data row544 col1" >0.026200</td>
          <td id="T_78b05_row544_col2" class="data row544 col2" >-0.007200</td>
          <td id="T_78b05_row544_col3" class="data row544 col3" >-0.039200</td>
          <td id="T_78b05_row544_col4" class="data row544 col4" >0.035900</td>
          <td id="T_78b05_row544_col5" class="data row544 col5" >-0.038300</td>
          <td id="T_78b05_row544_col6" class="data row544 col6" >0.057900</td>
          <td id="T_78b05_row544_col7" class="data row544 col7" >-0.019800</td>
          <td id="T_78b05_row544_col8" class="data row544 col8" >0.019300</td>
          <td id="T_78b05_row544_col9" class="data row544 col9" >0.022600</td>
          <td id="T_78b05_row544_col10" class="data row544 col10" >0.070000</td>
          <td id="T_78b05_row544_col11" class="data row544 col11" >0.053800</td>
          <td id="T_78b05_row544_col12" class="data row544 col12" >0.037700</td>
          <td id="T_78b05_row544_col13" class="data row544 col13" >0.061800</td>
          <td id="T_78b05_row544_col14" class="data row544 col14" >0.018100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row545" class="row_heading level0 row545" >546</th>
          <td id="T_78b05_row545_col0" class="data row545 col0" >None</td>
          <td id="T_78b05_row545_col1" class="data row545 col1" >0.030900</td>
          <td id="T_78b05_row545_col2" class="data row545 col2" >0.070900</td>
          <td id="T_78b05_row545_col3" class="data row545 col3" >0.016500</td>
          <td id="T_78b05_row545_col4" class="data row545 col4" >-0.030500</td>
          <td id="T_78b05_row545_col5" class="data row545 col5" >0.057300</td>
          <td id="T_78b05_row545_col6" class="data row545 col6" >0.013100</td>
          <td id="T_78b05_row545_col7" class="data row545 col7" >-0.101200</td>
          <td id="T_78b05_row545_col8" class="data row545 col8" >0.014600</td>
          <td id="T_78b05_row545_col9" class="data row545 col9" >0.100700</td>
          <td id="T_78b05_row545_col10" class="data row545 col10" >0.014300</td>
          <td id="T_78b05_row545_col11" class="data row545 col11" >0.012600</td>
          <td id="T_78b05_row545_col12" class="data row545 col12" >0.057900</td>
          <td id="T_78b05_row545_col13" class="data row545 col13" >0.017000</td>
          <td id="T_78b05_row545_col14" class="data row545 col14" >0.099500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row546" class="row_heading level0 row546" >547</th>
          <td id="T_78b05_row546_col0" class="data row546 col0" >None</td>
          <td id="T_78b05_row546_col1" class="data row546 col1" >0.039500</td>
          <td id="T_78b05_row546_col2" class="data row546 col2" >-0.051400</td>
          <td id="T_78b05_row546_col3" class="data row546 col3" >0.001800</td>
          <td id="T_78b05_row546_col4" class="data row546 col4" >0.010200</td>
          <td id="T_78b05_row546_col5" class="data row546 col5" >-0.034000</td>
          <td id="T_78b05_row546_col6" class="data row546 col6" >0.031600</td>
          <td id="T_78b05_row546_col7" class="data row546 col7" >-0.008800</td>
          <td id="T_78b05_row546_col8" class="data row546 col8" >0.006000</td>
          <td id="T_78b05_row546_col9" class="data row546 col9" >0.021500</td>
          <td id="T_78b05_row546_col10" class="data row546 col10" >0.028900</td>
          <td id="T_78b05_row546_col11" class="data row546 col11" >0.028100</td>
          <td id="T_78b05_row546_col12" class="data row546 col12" >0.033400</td>
          <td id="T_78b05_row546_col13" class="data row546 col13" >0.035500</td>
          <td id="T_78b05_row546_col14" class="data row546 col14" >0.007100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row547" class="row_heading level0 row547" >548</th>
          <td id="T_78b05_row547_col0" class="data row547 col0" >PC1</td>
          <td id="T_78b05_row547_col1" class="data row547 col1" >0.025200</td>
          <td id="T_78b05_row547_col2" class="data row547 col2" >0.056900</td>
          <td id="T_78b05_row547_col3" class="data row547 col3" >-0.039300</td>
          <td id="T_78b05_row547_col4" class="data row547 col4" >-0.054100</td>
          <td id="T_78b05_row547_col5" class="data row547 col5" >-0.016300</td>
          <td id="T_78b05_row547_col6" class="data row547 col6" >0.012800</td>
          <td id="T_78b05_row547_col7" class="data row547 col7" >-0.084900</td>
          <td id="T_78b05_row547_col8" class="data row547 col8" >0.020300</td>
          <td id="T_78b05_row547_col9" class="data row547 col9" >0.086700</td>
          <td id="T_78b05_row547_col10" class="data row547 col10" >0.070100</td>
          <td id="T_78b05_row547_col11" class="data row547 col11" >0.036200</td>
          <td id="T_78b05_row547_col12" class="data row547 col12" >0.015700</td>
          <td id="T_78b05_row547_col13" class="data row547 col13" >0.016700</td>
          <td id="T_78b05_row547_col14" class="data row547 col14" >0.083200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row548" class="row_heading level0 row548" >549</th>
          <td id="T_78b05_row548_col0" class="data row548 col0" >None</td>
          <td id="T_78b05_row548_col1" class="data row548 col1" >0.034600</td>
          <td id="T_78b05_row548_col2" class="data row548 col2" >0.026700</td>
          <td id="T_78b05_row548_col3" class="data row548 col3" >0.011800</td>
          <td id="T_78b05_row548_col4" class="data row548 col4" >0.068300</td>
          <td id="T_78b05_row548_col5" class="data row548 col5" >-0.024100</td>
          <td id="T_78b05_row548_col6" class="data row548 col6" >-0.004500</td>
          <td id="T_78b05_row548_col7" class="data row548 col7" >-0.022300</td>
          <td id="T_78b05_row548_col8" class="data row548 col8" >0.010900</td>
          <td id="T_78b05_row548_col9" class="data row548 col9" >0.056500</td>
          <td id="T_78b05_row548_col10" class="data row548 col10" >0.018900</td>
          <td id="T_78b05_row548_col11" class="data row548 col11" >0.086200</td>
          <td id="T_78b05_row548_col12" class="data row548 col12" >0.023400</td>
          <td id="T_78b05_row548_col13" class="data row548 col13" >0.000600</td>
          <td id="T_78b05_row548_col14" class="data row548 col14" >0.020600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row549" class="row_heading level0 row549" >550</th>
          <td id="T_78b05_row549_col0" class="data row549 col0" >None</td>
          <td id="T_78b05_row549_col1" class="data row549 col1" >0.034600</td>
          <td id="T_78b05_row549_col2" class="data row549 col2" >0.028700</td>
          <td id="T_78b05_row549_col3" class="data row549 col3" >-0.052000</td>
          <td id="T_78b05_row549_col4" class="data row549 col4" >0.026700</td>
          <td id="T_78b05_row549_col5" class="data row549 col5" >-0.007400</td>
          <td id="T_78b05_row549_col6" class="data row549 col6" >0.058700</td>
          <td id="T_78b05_row549_col7" class="data row549 col7" >-0.012900</td>
          <td id="T_78b05_row549_col8" class="data row549 col8" >0.010900</td>
          <td id="T_78b05_row549_col9" class="data row549 col9" >0.058500</td>
          <td id="T_78b05_row549_col10" class="data row549 col10" >0.082700</td>
          <td id="T_78b05_row549_col11" class="data row549 col11" >0.044600</td>
          <td id="T_78b05_row549_col12" class="data row549 col12" >0.006800</td>
          <td id="T_78b05_row549_col13" class="data row549 col13" >0.062600</td>
          <td id="T_78b05_row549_col14" class="data row549 col14" >0.011200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row550" class="row_heading level0 row550" >551</th>
          <td id="T_78b05_row550_col0" class="data row550 col0" >None</td>
          <td id="T_78b05_row550_col1" class="data row550 col1" >0.035200</td>
          <td id="T_78b05_row550_col2" class="data row550 col2" >-0.009100</td>
          <td id="T_78b05_row550_col3" class="data row550 col3" >-0.011000</td>
          <td id="T_78b05_row550_col4" class="data row550 col4" >-0.021100</td>
          <td id="T_78b05_row550_col5" class="data row550 col5" >-0.049500</td>
          <td id="T_78b05_row550_col6" class="data row550 col6" >-0.044000</td>
          <td id="T_78b05_row550_col7" class="data row550 col7" >-0.089500</td>
          <td id="T_78b05_row550_col8" class="data row550 col8" >0.010400</td>
          <td id="T_78b05_row550_col9" class="data row550 col9" >0.020700</td>
          <td id="T_78b05_row550_col10" class="data row550 col10" >0.041800</td>
          <td id="T_78b05_row550_col11" class="data row550 col11" >0.003200</td>
          <td id="T_78b05_row550_col12" class="data row550 col12" >0.048900</td>
          <td id="T_78b05_row550_col13" class="data row550 col13" >0.040100</td>
          <td id="T_78b05_row550_col14" class="data row550 col14" >0.087700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row551" class="row_heading level0 row551" >552</th>
          <td id="T_78b05_row551_col0" class="data row551 col0" >PC1</td>
          <td id="T_78b05_row551_col1" class="data row551 col1" >0.026000</td>
          <td id="T_78b05_row551_col2" class="data row551 col2" >0.072800</td>
          <td id="T_78b05_row551_col3" class="data row551 col3" >0.031000</td>
          <td id="T_78b05_row551_col4" class="data row551 col4" >-0.093000</td>
          <td id="T_78b05_row551_col5" class="data row551 col5" >-0.027800</td>
          <td id="T_78b05_row551_col6" class="data row551 col6" >0.013600</td>
          <td id="T_78b05_row551_col7" class="data row551 col7" >-0.066300</td>
          <td id="T_78b05_row551_col8" class="data row551 col8" >0.019600</td>
          <td id="T_78b05_row551_col9" class="data row551 col9" >0.102600</td>
          <td id="T_78b05_row551_col10" class="data row551 col10" >0.000200</td>
          <td id="T_78b05_row551_col11" class="data row551 col11" >0.075100</td>
          <td id="T_78b05_row551_col12" class="data row551 col12" >0.027200</td>
          <td id="T_78b05_row551_col13" class="data row551 col13" >0.017500</td>
          <td id="T_78b05_row551_col14" class="data row551 col14" >0.064600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row552" class="row_heading level0 row552" >553</th>
          <td id="T_78b05_row552_col0" class="data row552 col0" >PC1</td>
          <td id="T_78b05_row552_col1" class="data row552 col1" >0.020500</td>
          <td id="T_78b05_row552_col2" class="data row552 col2" >0.077500</td>
          <td id="T_78b05_row552_col3" class="data row552 col3" >-0.079700</td>
          <td id="T_78b05_row552_col4" class="data row552 col4" >0.007800</td>
          <td id="T_78b05_row552_col5" class="data row552 col5" >-0.086600</td>
          <td id="T_78b05_row552_col6" class="data row552 col6" >-0.004200</td>
          <td id="T_78b05_row552_col7" class="data row552 col7" >-0.011700</td>
          <td id="T_78b05_row552_col8" class="data row552 col8" >0.025000</td>
          <td id="T_78b05_row552_col9" class="data row552 col9" >0.107300</td>
          <td id="T_78b05_row552_col10" class="data row552 col10" >0.110400</td>
          <td id="T_78b05_row552_col11" class="data row552 col11" >0.025700</td>
          <td id="T_78b05_row552_col12" class="data row552 col12" >0.086000</td>
          <td id="T_78b05_row552_col13" class="data row552 col13" >0.000300</td>
          <td id="T_78b05_row552_col14" class="data row552 col14" >0.010000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row553" class="row_heading level0 row553" >554</th>
          <td id="T_78b05_row553_col0" class="data row553 col0" >None</td>
          <td id="T_78b05_row553_col1" class="data row553 col1" >0.037400</td>
          <td id="T_78b05_row553_col2" class="data row553 col2" >0.038100</td>
          <td id="T_78b05_row553_col3" class="data row553 col3" >0.047700</td>
          <td id="T_78b05_row553_col4" class="data row553 col4" >-0.027100</td>
          <td id="T_78b05_row553_col5" class="data row553 col5" >0.037900</td>
          <td id="T_78b05_row553_col6" class="data row553 col6" >-0.004400</td>
          <td id="T_78b05_row553_col7" class="data row553 col7" >-0.030800</td>
          <td id="T_78b05_row553_col8" class="data row553 col8" >0.008100</td>
          <td id="T_78b05_row553_col9" class="data row553 col9" >0.067900</td>
          <td id="T_78b05_row553_col10" class="data row553 col10" >0.016900</td>
          <td id="T_78b05_row553_col11" class="data row553 col11" >0.009200</td>
          <td id="T_78b05_row553_col12" class="data row553 col12" >0.038500</td>
          <td id="T_78b05_row553_col13" class="data row553 col13" >0.000500</td>
          <td id="T_78b05_row553_col14" class="data row553 col14" >0.029100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row554" class="row_heading level0 row554" >555</th>
          <td id="T_78b05_row554_col0" class="data row554 col0" >None</td>
          <td id="T_78b05_row554_col1" class="data row554 col1" >0.033600</td>
          <td id="T_78b05_row554_col2" class="data row554 col2" >-0.011700</td>
          <td id="T_78b05_row554_col3" class="data row554 col3" >-0.001400</td>
          <td id="T_78b05_row554_col4" class="data row554 col4" >-0.054900</td>
          <td id="T_78b05_row554_col5" class="data row554 col5" >-0.013500</td>
          <td id="T_78b05_row554_col6" class="data row554 col6" >0.015200</td>
          <td id="T_78b05_row554_col7" class="data row554 col7" >-0.038400</td>
          <td id="T_78b05_row554_col8" class="data row554 col8" >0.011900</td>
          <td id="T_78b05_row554_col9" class="data row554 col9" >0.018100</td>
          <td id="T_78b05_row554_col10" class="data row554 col10" >0.032100</td>
          <td id="T_78b05_row554_col11" class="data row554 col11" >0.037000</td>
          <td id="T_78b05_row554_col12" class="data row554 col12" >0.012800</td>
          <td id="T_78b05_row554_col13" class="data row554 col13" >0.019100</td>
          <td id="T_78b05_row554_col14" class="data row554 col14" >0.036700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row555" class="row_heading level0 row555" >556</th>
          <td id="T_78b05_row555_col0" class="data row555 col0" >None</td>
          <td id="T_78b05_row555_col1" class="data row555 col1" >0.035800</td>
          <td id="T_78b05_row555_col2" class="data row555 col2" >0.045100</td>
          <td id="T_78b05_row555_col3" class="data row555 col3" >-0.003900</td>
          <td id="T_78b05_row555_col4" class="data row555 col4" >-0.001000</td>
          <td id="T_78b05_row555_col5" class="data row555 col5" >0.040900</td>
          <td id="T_78b05_row555_col6" class="data row555 col6" >-0.030200</td>
          <td id="T_78b05_row555_col7" class="data row555 col7" >0.013400</td>
          <td id="T_78b05_row555_col8" class="data row555 col8" >0.009700</td>
          <td id="T_78b05_row555_col9" class="data row555 col9" >0.075000</td>
          <td id="T_78b05_row555_col10" class="data row555 col10" >0.034600</td>
          <td id="T_78b05_row555_col11" class="data row555 col11" >0.016900</td>
          <td id="T_78b05_row555_col12" class="data row555 col12" >0.041500</td>
          <td id="T_78b05_row555_col13" class="data row555 col13" >0.026300</td>
          <td id="T_78b05_row555_col14" class="data row555 col14" >0.015200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row556" class="row_heading level0 row556" >557</th>
          <td id="T_78b05_row556_col0" class="data row556 col0" >None</td>
          <td id="T_78b05_row556_col1" class="data row556 col1" >0.036500</td>
          <td id="T_78b05_row556_col2" class="data row556 col2" >0.048200</td>
          <td id="T_78b05_row556_col3" class="data row556 col3" >-0.007200</td>
          <td id="T_78b05_row556_col4" class="data row556 col4" >0.035700</td>
          <td id="T_78b05_row556_col5" class="data row556 col5" >-0.026500</td>
          <td id="T_78b05_row556_col6" class="data row556 col6" >0.001300</td>
          <td id="T_78b05_row556_col7" class="data row556 col7" >0.024300</td>
          <td id="T_78b05_row556_col8" class="data row556 col8" >0.009000</td>
          <td id="T_78b05_row556_col9" class="data row556 col9" >0.078000</td>
          <td id="T_78b05_row556_col10" class="data row556 col10" >0.038000</td>
          <td id="T_78b05_row556_col11" class="data row556 col11" >0.053600</td>
          <td id="T_78b05_row556_col12" class="data row556 col12" >0.025800</td>
          <td id="T_78b05_row556_col13" class="data row556 col13" >0.005200</td>
          <td id="T_78b05_row556_col14" class="data row556 col14" >0.026000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row557" class="row_heading level0 row557" >558</th>
          <td id="T_78b05_row557_col0" class="data row557 col0" >None</td>
          <td id="T_78b05_row557_col1" class="data row557 col1" >0.028900</td>
          <td id="T_78b05_row557_col2" class="data row557 col2" >0.037300</td>
          <td id="T_78b05_row557_col3" class="data row557 col3" >-0.017100</td>
          <td id="T_78b05_row557_col4" class="data row557 col4" >0.037300</td>
          <td id="T_78b05_row557_col5" class="data row557 col5" >-0.011000</td>
          <td id="T_78b05_row557_col6" class="data row557 col6" >0.025400</td>
          <td id="T_78b05_row557_col7" class="data row557 col7" >0.005100</td>
          <td id="T_78b05_row557_col8" class="data row557 col8" >0.016600</td>
          <td id="T_78b05_row557_col9" class="data row557 col9" >0.067100</td>
          <td id="T_78b05_row557_col10" class="data row557 col10" >0.047800</td>
          <td id="T_78b05_row557_col11" class="data row557 col11" >0.055300</td>
          <td id="T_78b05_row557_col12" class="data row557 col12" >0.010400</td>
          <td id="T_78b05_row557_col13" class="data row557 col13" >0.029300</td>
          <td id="T_78b05_row557_col14" class="data row557 col14" >0.006800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row558" class="row_heading level0 row558" >559</th>
          <td id="T_78b05_row558_col0" class="data row558 col0" >None</td>
          <td id="T_78b05_row558_col1" class="data row558 col1" >0.037700</td>
          <td id="T_78b05_row558_col2" class="data row558 col2" >-0.027300</td>
          <td id="T_78b05_row558_col3" class="data row558 col3" >-0.022300</td>
          <td id="T_78b05_row558_col4" class="data row558 col4" >-0.069800</td>
          <td id="T_78b05_row558_col5" class="data row558 col5" >-0.029400</td>
          <td id="T_78b05_row558_col6" class="data row558 col6" >-0.051800</td>
          <td id="T_78b05_row558_col7" class="data row558 col7" >-0.016400</td>
          <td id="T_78b05_row558_col8" class="data row558 col8" >0.007800</td>
          <td id="T_78b05_row558_col9" class="data row558 col9" >0.002500</td>
          <td id="T_78b05_row558_col10" class="data row558 col10" >0.053100</td>
          <td id="T_78b05_row558_col11" class="data row558 col11" >0.051900</td>
          <td id="T_78b05_row558_col12" class="data row558 col12" >0.028800</td>
          <td id="T_78b05_row558_col13" class="data row558 col13" >0.047900</td>
          <td id="T_78b05_row558_col14" class="data row558 col14" >0.014700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row559" class="row_heading level0 row559" >560</th>
          <td id="T_78b05_row559_col0" class="data row559 col0" >None</td>
          <td id="T_78b05_row559_col1" class="data row559 col1" >0.035800</td>
          <td id="T_78b05_row559_col2" class="data row559 col2" >0.021300</td>
          <td id="T_78b05_row559_col3" class="data row559 col3" >-0.002100</td>
          <td id="T_78b05_row559_col4" class="data row559 col4" >-0.055600</td>
          <td id="T_78b05_row559_col5" class="data row559 col5" >-0.094400</td>
          <td id="T_78b05_row559_col6" class="data row559 col6" >0.015000</td>
          <td id="T_78b05_row559_col7" class="data row559 col7" >0.070800</td>
          <td id="T_78b05_row559_col8" class="data row559 col8" >0.009800</td>
          <td id="T_78b05_row559_col9" class="data row559 col9" >0.051100</td>
          <td id="T_78b05_row559_col10" class="data row559 col10" >0.032900</td>
          <td id="T_78b05_row559_col11" class="data row559 col11" >0.037700</td>
          <td id="T_78b05_row559_col12" class="data row559 col12" >0.093700</td>
          <td id="T_78b05_row559_col13" class="data row559 col13" >0.018800</td>
          <td id="T_78b05_row559_col14" class="data row559 col14" >0.072600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row560" class="row_heading level0 row560" >561</th>
          <td id="T_78b05_row560_col0" class="data row560 col0" >None</td>
          <td id="T_78b05_row560_col1" class="data row560 col1" >0.041000</td>
          <td id="T_78b05_row560_col2" class="data row560 col2" >0.033500</td>
          <td id="T_78b05_row560_col3" class="data row560 col3" >-0.001800</td>
          <td id="T_78b05_row560_col4" class="data row560 col4" >0.010100</td>
          <td id="T_78b05_row560_col5" class="data row560 col5" >0.004200</td>
          <td id="T_78b05_row560_col6" class="data row560 col6" >-0.020700</td>
          <td id="T_78b05_row560_col7" class="data row560 col7" >0.007700</td>
          <td id="T_78b05_row560_col8" class="data row560 col8" >0.004500</td>
          <td id="T_78b05_row560_col9" class="data row560 col9" >0.063300</td>
          <td id="T_78b05_row560_col10" class="data row560 col10" >0.032600</td>
          <td id="T_78b05_row560_col11" class="data row560 col11" >0.028000</td>
          <td id="T_78b05_row560_col12" class="data row560 col12" >0.004800</td>
          <td id="T_78b05_row560_col13" class="data row560 col13" >0.016800</td>
          <td id="T_78b05_row560_col14" class="data row560 col14" >0.009500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row561" class="row_heading level0 row561" >562</th>
          <td id="T_78b05_row561_col0" class="data row561 col0" >None</td>
          <td id="T_78b05_row561_col1" class="data row561 col1" >0.035800</td>
          <td id="T_78b05_row561_col2" class="data row561 col2" >0.029100</td>
          <td id="T_78b05_row561_col3" class="data row561 col3" >-0.013900</td>
          <td id="T_78b05_row561_col4" class="data row561 col4" >0.014400</td>
          <td id="T_78b05_row561_col5" class="data row561 col5" >0.041500</td>
          <td id="T_78b05_row561_col6" class="data row561 col6" >0.007400</td>
          <td id="T_78b05_row561_col7" class="data row561 col7" >-0.021700</td>
          <td id="T_78b05_row561_col8" class="data row561 col8" >0.009700</td>
          <td id="T_78b05_row561_col9" class="data row561 col9" >0.058900</td>
          <td id="T_78b05_row561_col10" class="data row561 col10" >0.044600</td>
          <td id="T_78b05_row561_col11" class="data row561 col11" >0.032300</td>
          <td id="T_78b05_row561_col12" class="data row561 col12" >0.042100</td>
          <td id="T_78b05_row561_col13" class="data row561 col13" >0.011300</td>
          <td id="T_78b05_row561_col14" class="data row561 col14" >0.020000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row562" class="row_heading level0 row562" >563</th>
          <td id="T_78b05_row562_col0" class="data row562 col0" >None</td>
          <td id="T_78b05_row562_col1" class="data row562 col1" >0.049400</td>
          <td id="T_78b05_row562_col2" class="data row562 col2" >-0.051200</td>
          <td id="T_78b05_row562_col3" class="data row562 col3" >0.000900</td>
          <td id="T_78b05_row562_col4" class="data row562 col4" >-0.004600</td>
          <td id="T_78b05_row562_col5" class="data row562 col5" >0.071400</td>
          <td id="T_78b05_row562_col6" class="data row562 col6" >0.010300</td>
          <td id="T_78b05_row562_col7" class="data row562 col7" >0.031900</td>
          <td id="T_78b05_row562_col8" class="data row562 col8" >0.003900</td>
          <td id="T_78b05_row562_col9" class="data row562 col9" >0.021300</td>
          <td id="T_78b05_row562_col10" class="data row562 col10" >0.029900</td>
          <td id="T_78b05_row562_col11" class="data row562 col11" >0.013300</td>
          <td id="T_78b05_row562_col12" class="data row562 col12" >0.072000</td>
          <td id="T_78b05_row562_col13" class="data row562 col13" >0.014200</td>
          <td id="T_78b05_row562_col14" class="data row562 col14" >0.033700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row563" class="row_heading level0 row563" >564</th>
          <td id="T_78b05_row563_col0" class="data row563 col0" >None</td>
          <td id="T_78b05_row563_col1" class="data row563 col1" >0.038700</td>
          <td id="T_78b05_row563_col2" class="data row563 col2" >-0.015800</td>
          <td id="T_78b05_row563_col3" class="data row563 col3" >-0.030200</td>
          <td id="T_78b05_row563_col4" class="data row563 col4" >-0.058900</td>
          <td id="T_78b05_row563_col5" class="data row563 col5" >0.008500</td>
          <td id="T_78b05_row563_col6" class="data row563 col6" >-0.060200</td>
          <td id="T_78b05_row563_col7" class="data row563 col7" >-0.043800</td>
          <td id="T_78b05_row563_col8" class="data row563 col8" >0.006800</td>
          <td id="T_78b05_row563_col9" class="data row563 col9" >0.014100</td>
          <td id="T_78b05_row563_col10" class="data row563 col10" >0.060900</td>
          <td id="T_78b05_row563_col11" class="data row563 col11" >0.041000</td>
          <td id="T_78b05_row563_col12" class="data row563 col12" >0.009200</td>
          <td id="T_78b05_row563_col13" class="data row563 col13" >0.056300</td>
          <td id="T_78b05_row563_col14" class="data row563 col14" >0.042100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row564" class="row_heading level0 row564" >565</th>
          <td id="T_78b05_row564_col0" class="data row564 col0" >None</td>
          <td id="T_78b05_row564_col1" class="data row564 col1" >0.042500</td>
          <td id="T_78b05_row564_col2" class="data row564 col2" >0.034300</td>
          <td id="T_78b05_row564_col3" class="data row564 col3" >0.013000</td>
          <td id="T_78b05_row564_col4" class="data row564 col4" >0.051600</td>
          <td id="T_78b05_row564_col5" class="data row564 col5" >0.000700</td>
          <td id="T_78b05_row564_col6" class="data row564 col6" >0.059400</td>
          <td id="T_78b05_row564_col7" class="data row564 col7" >0.017700</td>
          <td id="T_78b05_row564_col8" class="data row564 col8" >0.003000</td>
          <td id="T_78b05_row564_col9" class="data row564 col9" >0.064200</td>
          <td id="T_78b05_row564_col10" class="data row564 col10" >0.017700</td>
          <td id="T_78b05_row564_col11" class="data row564 col11" >0.069500</td>
          <td id="T_78b05_row564_col12" class="data row564 col12" >0.001300</td>
          <td id="T_78b05_row564_col13" class="data row564 col13" >0.063300</td>
          <td id="T_78b05_row564_col14" class="data row564 col14" >0.019500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row565" class="row_heading level0 row565" >566</th>
          <td id="T_78b05_row565_col0" class="data row565 col0" >None</td>
          <td id="T_78b05_row565_col1" class="data row565 col1" >0.050700</td>
          <td id="T_78b05_row565_col2" class="data row565 col2" >-0.044200</td>
          <td id="T_78b05_row565_col3" class="data row565 col3" >0.024800</td>
          <td id="T_78b05_row565_col4" class="data row565 col4" >0.070100</td>
          <td id="T_78b05_row565_col5" class="data row565 col5" >0.007000</td>
          <td id="T_78b05_row565_col6" class="data row565 col6" >0.064700</td>
          <td id="T_78b05_row565_col7" class="data row565 col7" >0.039800</td>
          <td id="T_78b05_row565_col8" class="data row565 col8" >0.005100</td>
          <td id="T_78b05_row565_col9" class="data row565 col9" >0.014300</td>
          <td id="T_78b05_row565_col10" class="data row565 col10" >0.005900</td>
          <td id="T_78b05_row565_col11" class="data row565 col11" >0.088000</td>
          <td id="T_78b05_row565_col12" class="data row565 col12" >0.007600</td>
          <td id="T_78b05_row565_col13" class="data row565 col13" >0.068600</td>
          <td id="T_78b05_row565_col14" class="data row565 col14" >0.041600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row566" class="row_heading level0 row566" >567</th>
          <td id="T_78b05_row566_col0" class="data row566 col0" >None</td>
          <td id="T_78b05_row566_col1" class="data row566 col1" >0.033200</td>
          <td id="T_78b05_row566_col2" class="data row566 col2" >0.037500</td>
          <td id="T_78b05_row566_col3" class="data row566 col3" >-0.020300</td>
          <td id="T_78b05_row566_col4" class="data row566 col4" >-0.026300</td>
          <td id="T_78b05_row566_col5" class="data row566 col5" >-0.050600</td>
          <td id="T_78b05_row566_col6" class="data row566 col6" >0.026300</td>
          <td id="T_78b05_row566_col7" class="data row566 col7" >-0.014100</td>
          <td id="T_78b05_row566_col8" class="data row566 col8" >0.012300</td>
          <td id="T_78b05_row566_col9" class="data row566 col9" >0.067300</td>
          <td id="T_78b05_row566_col10" class="data row566 col10" >0.051000</td>
          <td id="T_78b05_row566_col11" class="data row566 col11" >0.008400</td>
          <td id="T_78b05_row566_col12" class="data row566 col12" >0.050000</td>
          <td id="T_78b05_row566_col13" class="data row566 col13" >0.030200</td>
          <td id="T_78b05_row566_col14" class="data row566 col14" >0.012300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row567" class="row_heading level0 row567" >568</th>
          <td id="T_78b05_row567_col0" class="data row567 col0" >None</td>
          <td id="T_78b05_row567_col1" class="data row567 col1" >0.029400</td>
          <td id="T_78b05_row567_col2" class="data row567 col2" >0.037400</td>
          <td id="T_78b05_row567_col3" class="data row567 col3" >0.019300</td>
          <td id="T_78b05_row567_col4" class="data row567 col4" >-0.009300</td>
          <td id="T_78b05_row567_col5" class="data row567 col5" >-0.061500</td>
          <td id="T_78b05_row567_col6" class="data row567 col6" >0.029300</td>
          <td id="T_78b05_row567_col7" class="data row567 col7" >-0.010600</td>
          <td id="T_78b05_row567_col8" class="data row567 col8" >0.016200</td>
          <td id="T_78b05_row567_col9" class="data row567 col9" >0.067300</td>
          <td id="T_78b05_row567_col10" class="data row567 col10" >0.011500</td>
          <td id="T_78b05_row567_col11" class="data row567 col11" >0.008600</td>
          <td id="T_78b05_row567_col12" class="data row567 col12" >0.060900</td>
          <td id="T_78b05_row567_col13" class="data row567 col13" >0.033200</td>
          <td id="T_78b05_row567_col14" class="data row567 col14" >0.008900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row568" class="row_heading level0 row568" >569</th>
          <td id="T_78b05_row568_col0" class="data row568 col0" >PC1</td>
          <td id="T_78b05_row568_col1" class="data row568 col1" >0.022100</td>
          <td id="T_78b05_row568_col2" class="data row568 col2" >0.043600</td>
          <td id="T_78b05_row568_col3" class="data row568 col3" >-0.065400</td>
          <td id="T_78b05_row568_col4" class="data row568 col4" >0.005000</td>
          <td id="T_78b05_row568_col5" class="data row568 col5" >-0.038200</td>
          <td id="T_78b05_row568_col6" class="data row568 col6" >-0.040100</td>
          <td id="T_78b05_row568_col7" class="data row568 col7" >-0.035200</td>
          <td id="T_78b05_row568_col8" class="data row568 col8" >0.023400</td>
          <td id="T_78b05_row568_col9" class="data row568 col9" >0.073400</td>
          <td id="T_78b05_row568_col10" class="data row568 col10" >0.096200</td>
          <td id="T_78b05_row568_col11" class="data row568 col11" >0.022900</td>
          <td id="T_78b05_row568_col12" class="data row568 col12" >0.037600</td>
          <td id="T_78b05_row568_col13" class="data row568 col13" >0.036200</td>
          <td id="T_78b05_row568_col14" class="data row568 col14" >0.033400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row569" class="row_heading level0 row569" >570</th>
          <td id="T_78b05_row569_col0" class="data row569 col0" >None</td>
          <td id="T_78b05_row569_col1" class="data row569 col1" >0.040900</td>
          <td id="T_78b05_row569_col2" class="data row569 col2" >-0.037900</td>
          <td id="T_78b05_row569_col3" class="data row569 col3" >0.005400</td>
          <td id="T_78b05_row569_col4" class="data row569 col4" >0.088100</td>
          <td id="T_78b05_row569_col5" class="data row569 col5" >0.001000</td>
          <td id="T_78b05_row569_col6" class="data row569 col6" >0.043500</td>
          <td id="T_78b05_row569_col7" class="data row569 col7" >-0.005800</td>
          <td id="T_78b05_row569_col8" class="data row569 col8" >0.004600</td>
          <td id="T_78b05_row569_col9" class="data row569 col9" >0.008100</td>
          <td id="T_78b05_row569_col10" class="data row569 col10" >0.025300</td>
          <td id="T_78b05_row569_col11" class="data row569 col11" >0.106000</td>
          <td id="T_78b05_row569_col12" class="data row569 col12" >0.001700</td>
          <td id="T_78b05_row569_col13" class="data row569 col13" >0.047400</td>
          <td id="T_78b05_row569_col14" class="data row569 col14" >0.004000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row570" class="row_heading level0 row570" >571</th>
          <td id="T_78b05_row570_col0" class="data row570 col0" >None</td>
          <td id="T_78b05_row570_col1" class="data row570 col1" >0.046400</td>
          <td id="T_78b05_row570_col2" class="data row570 col2" >-0.051500</td>
          <td id="T_78b05_row570_col3" class="data row570 col3" >0.009500</td>
          <td id="T_78b05_row570_col4" class="data row570 col4" >0.053200</td>
          <td id="T_78b05_row570_col5" class="data row570 col5" >-0.016200</td>
          <td id="T_78b05_row570_col6" class="data row570 col6" >-0.008400</td>
          <td id="T_78b05_row570_col7" class="data row570 col7" >-0.030200</td>
          <td id="T_78b05_row570_col8" class="data row570 col8" >0.000900</td>
          <td id="T_78b05_row570_col9" class="data row570 col9" >0.021700</td>
          <td id="T_78b05_row570_col10" class="data row570 col10" >0.021300</td>
          <td id="T_78b05_row570_col11" class="data row570 col11" >0.071100</td>
          <td id="T_78b05_row570_col12" class="data row570 col12" >0.015500</td>
          <td id="T_78b05_row570_col13" class="data row570 col13" >0.004500</td>
          <td id="T_78b05_row570_col14" class="data row570 col14" >0.028500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row571" class="row_heading level0 row571" >572</th>
          <td id="T_78b05_row571_col0" class="data row571 col0" >None</td>
          <td id="T_78b05_row571_col1" class="data row571 col1" >0.042800</td>
          <td id="T_78b05_row571_col2" class="data row571 col2" >-0.048600</td>
          <td id="T_78b05_row571_col3" class="data row571 col3" >0.000100</td>
          <td id="T_78b05_row571_col4" class="data row571 col4" >0.052300</td>
          <td id="T_78b05_row571_col5" class="data row571 col5" >0.013400</td>
          <td id="T_78b05_row571_col6" class="data row571 col6" >0.047100</td>
          <td id="T_78b05_row571_col7" class="data row571 col7" >-0.008900</td>
          <td id="T_78b05_row571_col8" class="data row571 col8" >0.002700</td>
          <td id="T_78b05_row571_col9" class="data row571 col9" >0.018700</td>
          <td id="T_78b05_row571_col10" class="data row571 col10" >0.030700</td>
          <td id="T_78b05_row571_col11" class="data row571 col11" >0.070300</td>
          <td id="T_78b05_row571_col12" class="data row571 col12" >0.014000</td>
          <td id="T_78b05_row571_col13" class="data row571 col13" >0.051000</td>
          <td id="T_78b05_row571_col14" class="data row571 col14" >0.007200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row572" class="row_heading level0 row572" >573</th>
          <td id="T_78b05_row572_col0" class="data row572 col0" >None</td>
          <td id="T_78b05_row572_col1" class="data row572 col1" >0.035700</td>
          <td id="T_78b05_row572_col2" class="data row572 col2" >-0.039200</td>
          <td id="T_78b05_row572_col3" class="data row572 col3" >0.007400</td>
          <td id="T_78b05_row572_col4" class="data row572 col4" >-0.010500</td>
          <td id="T_78b05_row572_col5" class="data row572 col5" >-0.004200</td>
          <td id="T_78b05_row572_col6" class="data row572 col6" >-0.008400</td>
          <td id="T_78b05_row572_col7" class="data row572 col7" >-0.015100</td>
          <td id="T_78b05_row572_col8" class="data row572 col8" >0.009800</td>
          <td id="T_78b05_row572_col9" class="data row572 col9" >0.009300</td>
          <td id="T_78b05_row572_col10" class="data row572 col10" >0.023400</td>
          <td id="T_78b05_row572_col11" class="data row572 col11" >0.007400</td>
          <td id="T_78b05_row572_col12" class="data row572 col12" >0.003600</td>
          <td id="T_78b05_row572_col13" class="data row572 col13" >0.004500</td>
          <td id="T_78b05_row572_col14" class="data row572 col14" >0.013400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row573" class="row_heading level0 row573" >574</th>
          <td id="T_78b05_row573_col0" class="data row573 col0" >None</td>
          <td id="T_78b05_row573_col1" class="data row573 col1" >0.035700</td>
          <td id="T_78b05_row573_col2" class="data row573 col2" >-0.036000</td>
          <td id="T_78b05_row573_col3" class="data row573 col3" >-0.088700</td>
          <td id="T_78b05_row573_col4" class="data row573 col4" >0.088100</td>
          <td id="T_78b05_row573_col5" class="data row573 col5" >0.042300</td>
          <td id="T_78b05_row573_col6" class="data row573 col6" >-0.017600</td>
          <td id="T_78b05_row573_col7" class="data row573 col7" >-0.032400</td>
          <td id="T_78b05_row573_col8" class="data row573 col8" >0.009800</td>
          <td id="T_78b05_row573_col9" class="data row573 col9" >0.006200</td>
          <td id="T_78b05_row573_col10" class="data row573 col10" >0.119500</td>
          <td id="T_78b05_row573_col11" class="data row573 col11" >0.106000</td>
          <td id="T_78b05_row573_col12" class="data row573 col12" >0.043000</td>
          <td id="T_78b05_row573_col13" class="data row573 col13" >0.013700</td>
          <td id="T_78b05_row573_col14" class="data row573 col14" >0.030700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row574" class="row_heading level0 row574" >575</th>
          <td id="T_78b05_row574_col0" class="data row574 col0" >None</td>
          <td id="T_78b05_row574_col1" class="data row574 col1" >0.040800</td>
          <td id="T_78b05_row574_col2" class="data row574 col2" >-0.030400</td>
          <td id="T_78b05_row574_col3" class="data row574 col3" >-0.001900</td>
          <td id="T_78b05_row574_col4" class="data row574 col4" >-0.051000</td>
          <td id="T_78b05_row574_col5" class="data row574 col5" >-0.017800</td>
          <td id="T_78b05_row574_col6" class="data row574 col6" >-0.018900</td>
          <td id="T_78b05_row574_col7" class="data row574 col7" >0.037100</td>
          <td id="T_78b05_row574_col8" class="data row574 col8" >0.004700</td>
          <td id="T_78b05_row574_col9" class="data row574 col9" >0.000500</td>
          <td id="T_78b05_row574_col10" class="data row574 col10" >0.032700</td>
          <td id="T_78b05_row574_col11" class="data row574 col11" >0.033100</td>
          <td id="T_78b05_row574_col12" class="data row574 col12" >0.017200</td>
          <td id="T_78b05_row574_col13" class="data row574 col13" >0.015000</td>
          <td id="T_78b05_row574_col14" class="data row574 col14" >0.038900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row575" class="row_heading level0 row575" >576</th>
          <td id="T_78b05_row575_col0" class="data row575 col0" >None</td>
          <td id="T_78b05_row575_col1" class="data row575 col1" >0.037300</td>
          <td id="T_78b05_row575_col2" class="data row575 col2" >0.027400</td>
          <td id="T_78b05_row575_col3" class="data row575 col3" >-0.004000</td>
          <td id="T_78b05_row575_col4" class="data row575 col4" >0.037200</td>
          <td id="T_78b05_row575_col5" class="data row575 col5" >0.018700</td>
          <td id="T_78b05_row575_col6" class="data row575 col6" >-0.002100</td>
          <td id="T_78b05_row575_col7" class="data row575 col7" >0.006200</td>
          <td id="T_78b05_row575_col8" class="data row575 col8" >0.008200</td>
          <td id="T_78b05_row575_col9" class="data row575 col9" >0.057200</td>
          <td id="T_78b05_row575_col10" class="data row575 col10" >0.034700</td>
          <td id="T_78b05_row575_col11" class="data row575 col11" >0.055200</td>
          <td id="T_78b05_row575_col12" class="data row575 col12" >0.019400</td>
          <td id="T_78b05_row575_col13" class="data row575 col13" >0.001800</td>
          <td id="T_78b05_row575_col14" class="data row575 col14" >0.008000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row576" class="row_heading level0 row576" >577</th>
          <td id="T_78b05_row576_col0" class="data row576 col0" >None</td>
          <td id="T_78b05_row576_col1" class="data row576 col1" >0.038600</td>
          <td id="T_78b05_row576_col2" class="data row576 col2" >-0.038500</td>
          <td id="T_78b05_row576_col3" class="data row576 col3" >-0.031200</td>
          <td id="T_78b05_row576_col4" class="data row576 col4" >-0.039000</td>
          <td id="T_78b05_row576_col5" class="data row576 col5" >0.035500</td>
          <td id="T_78b05_row576_col6" class="data row576 col6" >-0.016600</td>
          <td id="T_78b05_row576_col7" class="data row576 col7" >-0.050900</td>
          <td id="T_78b05_row576_col8" class="data row576 col8" >0.006900</td>
          <td id="T_78b05_row576_col9" class="data row576 col9" >0.008700</td>
          <td id="T_78b05_row576_col10" class="data row576 col10" >0.062000</td>
          <td id="T_78b05_row576_col11" class="data row576 col11" >0.021100</td>
          <td id="T_78b05_row576_col12" class="data row576 col12" >0.036200</td>
          <td id="T_78b05_row576_col13" class="data row576 col13" >0.012700</td>
          <td id="T_78b05_row576_col14" class="data row576 col14" >0.049200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row577" class="row_heading level0 row577" >578</th>
          <td id="T_78b05_row577_col0" class="data row577 col0" >None</td>
          <td id="T_78b05_row577_col1" class="data row577 col1" >0.039500</td>
          <td id="T_78b05_row577_col2" class="data row577 col2" >-0.011000</td>
          <td id="T_78b05_row577_col3" class="data row577 col3" >-0.002700</td>
          <td id="T_78b05_row577_col4" class="data row577 col4" >0.060200</td>
          <td id="T_78b05_row577_col5" class="data row577 col5" >0.049300</td>
          <td id="T_78b05_row577_col6" class="data row577 col6" >-0.030300</td>
          <td id="T_78b05_row577_col7" class="data row577 col7" >-0.071000</td>
          <td id="T_78b05_row577_col8" class="data row577 col8" >0.006000</td>
          <td id="T_78b05_row577_col9" class="data row577 col9" >0.018800</td>
          <td id="T_78b05_row577_col10" class="data row577 col10" >0.033500</td>
          <td id="T_78b05_row577_col11" class="data row577 col11" >0.078100</td>
          <td id="T_78b05_row577_col12" class="data row577 col12" >0.049900</td>
          <td id="T_78b05_row577_col13" class="data row577 col13" >0.026400</td>
          <td id="T_78b05_row577_col14" class="data row577 col14" >0.069300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row578" class="row_heading level0 row578" >579</th>
          <td id="T_78b05_row578_col0" class="data row578 col0" >None</td>
          <td id="T_78b05_row578_col1" class="data row578 col1" >0.042300</td>
          <td id="T_78b05_row578_col2" class="data row578 col2" >-0.020300</td>
          <td id="T_78b05_row578_col3" class="data row578 col3" >0.032800</td>
          <td id="T_78b05_row578_col4" class="data row578 col4" >-0.037000</td>
          <td id="T_78b05_row578_col5" class="data row578 col5" >-0.008200</td>
          <td id="T_78b05_row578_col6" class="data row578 col6" >0.021700</td>
          <td id="T_78b05_row578_col7" class="data row578 col7" >-0.000100</td>
          <td id="T_78b05_row578_col8" class="data row578 col8" >0.003200</td>
          <td id="T_78b05_row578_col9" class="data row578 col9" >0.009500</td>
          <td id="T_78b05_row578_col10" class="data row578 col10" >0.002100</td>
          <td id="T_78b05_row578_col11" class="data row578 col11" >0.019100</td>
          <td id="T_78b05_row578_col12" class="data row578 col12" >0.007600</td>
          <td id="T_78b05_row578_col13" class="data row578 col13" >0.025600</td>
          <td id="T_78b05_row578_col14" class="data row578 col14" >0.001600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row579" class="row_heading level0 row579" >580</th>
          <td id="T_78b05_row579_col0" class="data row579 col0" >None</td>
          <td id="T_78b05_row579_col1" class="data row579 col1" >0.034000</td>
          <td id="T_78b05_row579_col2" class="data row579 col2" >0.004100</td>
          <td id="T_78b05_row579_col3" class="data row579 col3" >-0.026200</td>
          <td id="T_78b05_row579_col4" class="data row579 col4" >-0.012600</td>
          <td id="T_78b05_row579_col5" class="data row579 col5" >-0.052200</td>
          <td id="T_78b05_row579_col6" class="data row579 col6" >0.071300</td>
          <td id="T_78b05_row579_col7" class="data row579 col7" >-0.025200</td>
          <td id="T_78b05_row579_col8" class="data row579 col8" >0.011500</td>
          <td id="T_78b05_row579_col9" class="data row579 col9" >0.033900</td>
          <td id="T_78b05_row579_col10" class="data row579 col10" >0.057000</td>
          <td id="T_78b05_row579_col11" class="data row579 col11" >0.005300</td>
          <td id="T_78b05_row579_col12" class="data row579 col12" >0.051500</td>
          <td id="T_78b05_row579_col13" class="data row579 col13" >0.075200</td>
          <td id="T_78b05_row579_col14" class="data row579 col14" >0.023500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row580" class="row_heading level0 row580" >581</th>
          <td id="T_78b05_row580_col0" class="data row580 col0" >None</td>
          <td id="T_78b05_row580_col1" class="data row580 col1" >0.030700</td>
          <td id="T_78b05_row580_col2" class="data row580 col2" >-0.026400</td>
          <td id="T_78b05_row580_col3" class="data row580 col3" >-0.034400</td>
          <td id="T_78b05_row580_col4" class="data row580 col4" >-0.020100</td>
          <td id="T_78b05_row580_col5" class="data row580 col5" >0.015500</td>
          <td id="T_78b05_row580_col6" class="data row580 col6" >-0.024500</td>
          <td id="T_78b05_row580_col7" class="data row580 col7" >-0.011300</td>
          <td id="T_78b05_row580_col8" class="data row580 col8" >0.014800</td>
          <td id="T_78b05_row580_col9" class="data row580 col9" >0.003400</td>
          <td id="T_78b05_row580_col10" class="data row580 col10" >0.065100</td>
          <td id="T_78b05_row580_col11" class="data row580 col11" >0.002100</td>
          <td id="T_78b05_row580_col12" class="data row580 col12" >0.016200</td>
          <td id="T_78b05_row580_col13" class="data row580 col13" >0.020600</td>
          <td id="T_78b05_row580_col14" class="data row580 col14" >0.009600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row581" class="row_heading level0 row581" >582</th>
          <td id="T_78b05_row581_col0" class="data row581 col0" >None</td>
          <td id="T_78b05_row581_col1" class="data row581 col1" >0.044300</td>
          <td id="T_78b05_row581_col2" class="data row581 col2" >-0.066300</td>
          <td id="T_78b05_row581_col3" class="data row581 col3" >-0.013700</td>
          <td id="T_78b05_row581_col4" class="data row581 col4" >-0.040300</td>
          <td id="T_78b05_row581_col5" class="data row581 col5" >0.002700</td>
          <td id="T_78b05_row581_col6" class="data row581 col6" >0.044000</td>
          <td id="T_78b05_row581_col7" class="data row581 col7" >0.026100</td>
          <td id="T_78b05_row581_col8" class="data row581 col8" >0.001200</td>
          <td id="T_78b05_row581_col9" class="data row581 col9" >0.036400</td>
          <td id="T_78b05_row581_col10" class="data row581 col10" >0.044400</td>
          <td id="T_78b05_row581_col11" class="data row581 col11" >0.022400</td>
          <td id="T_78b05_row581_col12" class="data row581 col12" >0.003400</td>
          <td id="T_78b05_row581_col13" class="data row581 col13" >0.047900</td>
          <td id="T_78b05_row581_col14" class="data row581 col14" >0.027900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row582" class="row_heading level0 row582" >583</th>
          <td id="T_78b05_row582_col0" class="data row582 col0" >None</td>
          <td id="T_78b05_row582_col1" class="data row582 col1" >0.045900</td>
          <td id="T_78b05_row582_col2" class="data row582 col2" >0.029700</td>
          <td id="T_78b05_row582_col3" class="data row582 col3" >0.032100</td>
          <td id="T_78b05_row582_col4" class="data row582 col4" >0.021800</td>
          <td id="T_78b05_row582_col5" class="data row582 col5" >0.015500</td>
          <td id="T_78b05_row582_col6" class="data row582 col6" >-0.001100</td>
          <td id="T_78b05_row582_col7" class="data row582 col7" >0.004400</td>
          <td id="T_78b05_row582_col8" class="data row582 col8" >0.000400</td>
          <td id="T_78b05_row582_col9" class="data row582 col9" >0.059600</td>
          <td id="T_78b05_row582_col10" class="data row582 col10" >0.001400</td>
          <td id="T_78b05_row582_col11" class="data row582 col11" >0.039700</td>
          <td id="T_78b05_row582_col12" class="data row582 col12" >0.016200</td>
          <td id="T_78b05_row582_col13" class="data row582 col13" >0.002800</td>
          <td id="T_78b05_row582_col14" class="data row582 col14" >0.006100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row583" class="row_heading level0 row583" >584</th>
          <td id="T_78b05_row583_col0" class="data row583 col0" >None</td>
          <td id="T_78b05_row583_col1" class="data row583 col1" >0.042400</td>
          <td id="T_78b05_row583_col2" class="data row583 col2" >0.020600</td>
          <td id="T_78b05_row583_col3" class="data row583 col3" >0.025900</td>
          <td id="T_78b05_row583_col4" class="data row583 col4" >0.003600</td>
          <td id="T_78b05_row583_col5" class="data row583 col5" >0.003200</td>
          <td id="T_78b05_row583_col6" class="data row583 col6" >0.045900</td>
          <td id="T_78b05_row583_col7" class="data row583 col7" >-0.019900</td>
          <td id="T_78b05_row583_col8" class="data row583 col8" >0.003100</td>
          <td id="T_78b05_row583_col9" class="data row583 col9" >0.050400</td>
          <td id="T_78b05_row583_col10" class="data row583 col10" >0.004900</td>
          <td id="T_78b05_row583_col11" class="data row583 col11" >0.021500</td>
          <td id="T_78b05_row583_col12" class="data row583 col12" >0.003900</td>
          <td id="T_78b05_row583_col13" class="data row583 col13" >0.049800</td>
          <td id="T_78b05_row583_col14" class="data row583 col14" >0.018200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row584" class="row_heading level0 row584" >585</th>
          <td id="T_78b05_row584_col0" class="data row584 col0" >PC1</td>
          <td id="T_78b05_row584_col1" class="data row584 col1" >0.022400</td>
          <td id="T_78b05_row584_col2" class="data row584 col2" >0.022200</td>
          <td id="T_78b05_row584_col3" class="data row584 col3" >-0.087800</td>
          <td id="T_78b05_row584_col4" class="data row584 col4" >0.046900</td>
          <td id="T_78b05_row584_col5" class="data row584 col5" >0.021700</td>
          <td id="T_78b05_row584_col6" class="data row584 col6" >0.025500</td>
          <td id="T_78b05_row584_col7" class="data row584 col7" >-0.098600</td>
          <td id="T_78b05_row584_col8" class="data row584 col8" >0.023100</td>
          <td id="T_78b05_row584_col9" class="data row584 col9" >0.052000</td>
          <td id="T_78b05_row584_col10" class="data row584 col10" >0.118500</td>
          <td id="T_78b05_row584_col11" class="data row584 col11" >0.064800</td>
          <td id="T_78b05_row584_col12" class="data row584 col12" >0.022300</td>
          <td id="T_78b05_row584_col13" class="data row584 col13" >0.029400</td>
          <td id="T_78b05_row584_col14" class="data row584 col14" >0.096800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row585" class="row_heading level0 row585" >586</th>
          <td id="T_78b05_row585_col0" class="data row585 col0" >None</td>
          <td id="T_78b05_row585_col1" class="data row585 col1" >0.038000</td>
          <td id="T_78b05_row585_col2" class="data row585 col2" >-0.002400</td>
          <td id="T_78b05_row585_col3" class="data row585 col3" >-0.025400</td>
          <td id="T_78b05_row585_col4" class="data row585 col4" >0.050200</td>
          <td id="T_78b05_row585_col5" class="data row585 col5" >0.054400</td>
          <td id="T_78b05_row585_col6" class="data row585 col6" >0.044300</td>
          <td id="T_78b05_row585_col7" class="data row585 col7" >0.020900</td>
          <td id="T_78b05_row585_col8" class="data row585 col8" >0.007500</td>
          <td id="T_78b05_row585_col9" class="data row585 col9" >0.027500</td>
          <td id="T_78b05_row585_col10" class="data row585 col10" >0.056200</td>
          <td id="T_78b05_row585_col11" class="data row585 col11" >0.068100</td>
          <td id="T_78b05_row585_col12" class="data row585 col12" >0.055000</td>
          <td id="T_78b05_row585_col13" class="data row585 col13" >0.048200</td>
          <td id="T_78b05_row585_col14" class="data row585 col14" >0.022600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row586" class="row_heading level0 row586" >587</th>
          <td id="T_78b05_row586_col0" class="data row586 col0" >None</td>
          <td id="T_78b05_row586_col1" class="data row586 col1" >0.035500</td>
          <td id="T_78b05_row586_col2" class="data row586 col2" >0.004600</td>
          <td id="T_78b05_row586_col3" class="data row586 col3" >0.023000</td>
          <td id="T_78b05_row586_col4" class="data row586 col4" >-0.011500</td>
          <td id="T_78b05_row586_col5" class="data row586 col5" >-0.006700</td>
          <td id="T_78b05_row586_col6" class="data row586 col6" >0.052700</td>
          <td id="T_78b05_row586_col7" class="data row586 col7" >-0.001400</td>
          <td id="T_78b05_row586_col8" class="data row586 col8" >0.010000</td>
          <td id="T_78b05_row586_col9" class="data row586 col9" >0.034400</td>
          <td id="T_78b05_row586_col10" class="data row586 col10" >0.007700</td>
          <td id="T_78b05_row586_col11" class="data row586 col11" >0.006400</td>
          <td id="T_78b05_row586_col12" class="data row586 col12" >0.006000</td>
          <td id="T_78b05_row586_col13" class="data row586 col13" >0.056600</td>
          <td id="T_78b05_row586_col14" class="data row586 col14" >0.000300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row587" class="row_heading level0 row587" >588</th>
          <td id="T_78b05_row587_col0" class="data row587 col0" >None</td>
          <td id="T_78b05_row587_col1" class="data row587 col1" >0.028000</td>
          <td id="T_78b05_row587_col2" class="data row587 col2" >0.003000</td>
          <td id="T_78b05_row587_col3" class="data row587 col3" >-0.076800</td>
          <td id="T_78b05_row587_col4" class="data row587 col4" >0.016800</td>
          <td id="T_78b05_row587_col5" class="data row587 col5" >0.035200</td>
          <td id="T_78b05_row587_col6" class="data row587 col6" >0.023300</td>
          <td id="T_78b05_row587_col7" class="data row587 col7" >-0.024300</td>
          <td id="T_78b05_row587_col8" class="data row587 col8" >0.017600</td>
          <td id="T_78b05_row587_col9" class="data row587 col9" >0.032900</td>
          <td id="T_78b05_row587_col10" class="data row587 col10" >0.107500</td>
          <td id="T_78b05_row587_col11" class="data row587 col11" >0.034700</td>
          <td id="T_78b05_row587_col12" class="data row587 col12" >0.035800</td>
          <td id="T_78b05_row587_col13" class="data row587 col13" >0.027200</td>
          <td id="T_78b05_row587_col14" class="data row587 col14" >0.022500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row588" class="row_heading level0 row588" >589</th>
          <td id="T_78b05_row588_col0" class="data row588 col0" >None</td>
          <td id="T_78b05_row588_col1" class="data row588 col1" >0.041400</td>
          <td id="T_78b05_row588_col2" class="data row588 col2" >-0.010800</td>
          <td id="T_78b05_row588_col3" class="data row588 col3" >-0.006200</td>
          <td id="T_78b05_row588_col4" class="data row588 col4" >0.107200</td>
          <td id="T_78b05_row588_col5" class="data row588 col5" >0.035500</td>
          <td id="T_78b05_row588_col6" class="data row588 col6" >-0.004100</td>
          <td id="T_78b05_row588_col7" class="data row588 col7" >-0.010400</td>
          <td id="T_78b05_row588_col8" class="data row588 col8" >0.004100</td>
          <td id="T_78b05_row588_col9" class="data row588 col9" >0.019000</td>
          <td id="T_78b05_row588_col10" class="data row588 col10" >0.036900</td>
          <td id="T_78b05_row588_col11" class="data row588 col11" >0.125100</td>
          <td id="T_78b05_row588_col12" class="data row588 col12" >0.036100</td>
          <td id="T_78b05_row588_col13" class="data row588 col13" >0.000300</td>
          <td id="T_78b05_row588_col14" class="data row588 col14" >0.008700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row589" class="row_heading level0 row589" >590</th>
          <td id="T_78b05_row589_col0" class="data row589 col0" >None</td>
          <td id="T_78b05_row589_col1" class="data row589 col1" >0.041600</td>
          <td id="T_78b05_row589_col2" class="data row589 col2" >0.029100</td>
          <td id="T_78b05_row589_col3" class="data row589 col3" >0.014500</td>
          <td id="T_78b05_row589_col4" class="data row589 col4" >-0.023300</td>
          <td id="T_78b05_row589_col5" class="data row589 col5" >0.040000</td>
          <td id="T_78b05_row589_col6" class="data row589 col6" >0.059900</td>
          <td id="T_78b05_row589_col7" class="data row589 col7" >-0.022500</td>
          <td id="T_78b05_row589_col8" class="data row589 col8" >0.004000</td>
          <td id="T_78b05_row589_col9" class="data row589 col9" >0.058900</td>
          <td id="T_78b05_row589_col10" class="data row589 col10" >0.016300</td>
          <td id="T_78b05_row589_col11" class="data row589 col11" >0.005400</td>
          <td id="T_78b05_row589_col12" class="data row589 col12" >0.040600</td>
          <td id="T_78b05_row589_col13" class="data row589 col13" >0.063800</td>
          <td id="T_78b05_row589_col14" class="data row589 col14" >0.020800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row590" class="row_heading level0 row590" >591</th>
          <td id="T_78b05_row590_col0" class="data row590 col0" >PC4</td>
          <td id="T_78b05_row590_col1" class="data row590 col1" >0.031300</td>
          <td id="T_78b05_row590_col2" class="data row590 col2" >0.004000</td>
          <td id="T_78b05_row590_col3" class="data row590 col3" >-0.032100</td>
          <td id="T_78b05_row590_col4" class="data row590 col4" >0.111300</td>
          <td id="T_78b05_row590_col5" class="data row590 col5" >-0.011800</td>
          <td id="T_78b05_row590_col6" class="data row590 col6" >-0.020400</td>
          <td id="T_78b05_row590_col7" class="data row590 col7" >0.002900</td>
          <td id="T_78b05_row590_col8" class="data row590 col8" >0.014200</td>
          <td id="T_78b05_row590_col9" class="data row590 col9" >0.033800</td>
          <td id="T_78b05_row590_col10" class="data row590 col10" >0.062900</td>
          <td id="T_78b05_row590_col11" class="data row590 col11" >0.129200</td>
          <td id="T_78b05_row590_col12" class="data row590 col12" >0.011200</td>
          <td id="T_78b05_row590_col13" class="data row590 col13" >0.016500</td>
          <td id="T_78b05_row590_col14" class="data row590 col14" >0.004600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row591" class="row_heading level0 row591" >592</th>
          <td id="T_78b05_row591_col0" class="data row591 col0" >None</td>
          <td id="T_78b05_row591_col1" class="data row591 col1" >0.045100</td>
          <td id="T_78b05_row591_col2" class="data row591 col2" >-0.001900</td>
          <td id="T_78b05_row591_col3" class="data row591 col3" >0.015800</td>
          <td id="T_78b05_row591_col4" class="data row591 col4" >-0.032300</td>
          <td id="T_78b05_row591_col5" class="data row591 col5" >0.030700</td>
          <td id="T_78b05_row591_col6" class="data row591 col6" >0.035600</td>
          <td id="T_78b05_row591_col7" class="data row591 col7" >0.008700</td>
          <td id="T_78b05_row591_col8" class="data row591 col8" >0.000500</td>
          <td id="T_78b05_row591_col9" class="data row591 col9" >0.028000</td>
          <td id="T_78b05_row591_col10" class="data row591 col10" >0.014900</td>
          <td id="T_78b05_row591_col11" class="data row591 col11" >0.014400</td>
          <td id="T_78b05_row591_col12" class="data row591 col12" >0.031400</td>
          <td id="T_78b05_row591_col13" class="data row591 col13" >0.039500</td>
          <td id="T_78b05_row591_col14" class="data row591 col14" >0.010400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row592" class="row_heading level0 row592" >593</th>
          <td id="T_78b05_row592_col0" class="data row592 col0" >None</td>
          <td id="T_78b05_row592_col1" class="data row592 col1" >0.045400</td>
          <td id="T_78b05_row592_col2" class="data row592 col2" >-0.009600</td>
          <td id="T_78b05_row592_col3" class="data row592 col3" >0.026800</td>
          <td id="T_78b05_row592_col4" class="data row592 col4" >-0.002800</td>
          <td id="T_78b05_row592_col5" class="data row592 col5" >0.048300</td>
          <td id="T_78b05_row592_col6" class="data row592 col6" >-0.051600</td>
          <td id="T_78b05_row592_col7" class="data row592 col7" >-0.002800</td>
          <td id="T_78b05_row592_col8" class="data row592 col8" >0.000100</td>
          <td id="T_78b05_row592_col9" class="data row592 col9" >0.020300</td>
          <td id="T_78b05_row592_col10" class="data row592 col10" >0.004000</td>
          <td id="T_78b05_row592_col11" class="data row592 col11" >0.015100</td>
          <td id="T_78b05_row592_col12" class="data row592 col12" >0.049000</td>
          <td id="T_78b05_row592_col13" class="data row592 col13" >0.047700</td>
          <td id="T_78b05_row592_col14" class="data row592 col14" >0.001100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row593" class="row_heading level0 row593" >594</th>
          <td id="T_78b05_row593_col0" class="data row593 col0" >None</td>
          <td id="T_78b05_row593_col1" class="data row593 col1" >0.037200</td>
          <td id="T_78b05_row593_col2" class="data row593 col2" >0.004500</td>
          <td id="T_78b05_row593_col3" class="data row593 col3" >0.000000</td>
          <td id="T_78b05_row593_col4" class="data row593 col4" >0.007600</td>
          <td id="T_78b05_row593_col5" class="data row593 col5" >-0.031000</td>
          <td id="T_78b05_row593_col6" class="data row593 col6" >0.086700</td>
          <td id="T_78b05_row593_col7" class="data row593 col7" >0.050600</td>
          <td id="T_78b05_row593_col8" class="data row593 col8" >0.008300</td>
          <td id="T_78b05_row593_col9" class="data row593 col9" >0.034300</td>
          <td id="T_78b05_row593_col10" class="data row593 col10" >0.030700</td>
          <td id="T_78b05_row593_col11" class="data row593 col11" >0.025500</td>
          <td id="T_78b05_row593_col12" class="data row593 col12" >0.030400</td>
          <td id="T_78b05_row593_col13" class="data row593 col13" >0.090600</td>
          <td id="T_78b05_row593_col14" class="data row593 col14" >0.052300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row594" class="row_heading level0 row594" >595</th>
          <td id="T_78b05_row594_col0" class="data row594 col0" >None</td>
          <td id="T_78b05_row594_col1" class="data row594 col1" >0.049800</td>
          <td id="T_78b05_row594_col2" class="data row594 col2" >0.001500</td>
          <td id="T_78b05_row594_col3" class="data row594 col3" >0.023200</td>
          <td id="T_78b05_row594_col4" class="data row594 col4" >-0.018400</td>
          <td id="T_78b05_row594_col5" class="data row594 col5" >0.046600</td>
          <td id="T_78b05_row594_col6" class="data row594 col6" >-0.009000</td>
          <td id="T_78b05_row594_col7" class="data row594 col7" >0.033700</td>
          <td id="T_78b05_row594_col8" class="data row594 col8" >0.004300</td>
          <td id="T_78b05_row594_col9" class="data row594 col9" >0.031300</td>
          <td id="T_78b05_row594_col10" class="data row594 col10" >0.007500</td>
          <td id="T_78b05_row594_col11" class="data row594 col11" >0.000500</td>
          <td id="T_78b05_row594_col12" class="data row594 col12" >0.047200</td>
          <td id="T_78b05_row594_col13" class="data row594 col13" >0.005100</td>
          <td id="T_78b05_row594_col14" class="data row594 col14" >0.035500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row595" class="row_heading level0 row595" >596</th>
          <td id="T_78b05_row595_col0" class="data row595 col0" >None</td>
          <td id="T_78b05_row595_col1" class="data row595 col1" >0.046900</td>
          <td id="T_78b05_row595_col2" class="data row595 col2" >-0.047100</td>
          <td id="T_78b05_row595_col3" class="data row595 col3" >0.014400</td>
          <td id="T_78b05_row595_col4" class="data row595 col4" >0.057900</td>
          <td id="T_78b05_row595_col5" class="data row595 col5" >0.036000</td>
          <td id="T_78b05_row595_col6" class="data row595 col6" >-0.014400</td>
          <td id="T_78b05_row595_col7" class="data row595 col7" >-0.024800</td>
          <td id="T_78b05_row595_col8" class="data row595 col8" >0.001400</td>
          <td id="T_78b05_row595_col9" class="data row595 col9" >0.017300</td>
          <td id="T_78b05_row595_col10" class="data row595 col10" >0.016400</td>
          <td id="T_78b05_row595_col11" class="data row595 col11" >0.075800</td>
          <td id="T_78b05_row595_col12" class="data row595 col12" >0.036600</td>
          <td id="T_78b05_row595_col13" class="data row595 col13" >0.010600</td>
          <td id="T_78b05_row595_col14" class="data row595 col14" >0.023000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row596" class="row_heading level0 row596" >597</th>
          <td id="T_78b05_row596_col0" class="data row596 col0" >None</td>
          <td id="T_78b05_row596_col1" class="data row596 col1" >0.040700</td>
          <td id="T_78b05_row596_col2" class="data row596 col2" >0.004500</td>
          <td id="T_78b05_row596_col3" class="data row596 col3" >-0.034100</td>
          <td id="T_78b05_row596_col4" class="data row596 col4" >-0.006200</td>
          <td id="T_78b05_row596_col5" class="data row596 col5" >0.042800</td>
          <td id="T_78b05_row596_col6" class="data row596 col6" >-0.030600</td>
          <td id="T_78b05_row596_col7" class="data row596 col7" >0.010100</td>
          <td id="T_78b05_row596_col8" class="data row596 col8" >0.004800</td>
          <td id="T_78b05_row596_col9" class="data row596 col9" >0.034300</td>
          <td id="T_78b05_row596_col10" class="data row596 col10" >0.064800</td>
          <td id="T_78b05_row596_col11" class="data row596 col11" >0.011700</td>
          <td id="T_78b05_row596_col12" class="data row596 col12" >0.043400</td>
          <td id="T_78b05_row596_col13" class="data row596 col13" >0.026700</td>
          <td id="T_78b05_row596_col14" class="data row596 col14" >0.011800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row597" class="row_heading level0 row597" >598</th>
          <td id="T_78b05_row597_col0" class="data row597 col0" >None</td>
          <td id="T_78b05_row597_col1" class="data row597 col1" >0.036300</td>
          <td id="T_78b05_row597_col2" class="data row597 col2" >0.027600</td>
          <td id="T_78b05_row597_col3" class="data row597 col3" >0.005200</td>
          <td id="T_78b05_row597_col4" class="data row597 col4" >0.079300</td>
          <td id="T_78b05_row597_col5" class="data row597 col5" >0.000700</td>
          <td id="T_78b05_row597_col6" class="data row597 col6" >0.011300</td>
          <td id="T_78b05_row597_col7" class="data row597 col7" >-0.017400</td>
          <td id="T_78b05_row597_col8" class="data row597 col8" >0.009200</td>
          <td id="T_78b05_row597_col9" class="data row597 col9" >0.057400</td>
          <td id="T_78b05_row597_col10" class="data row597 col10" >0.025500</td>
          <td id="T_78b05_row597_col11" class="data row597 col11" >0.097200</td>
          <td id="T_78b05_row597_col12" class="data row597 col12" >0.001300</td>
          <td id="T_78b05_row597_col13" class="data row597 col13" >0.015200</td>
          <td id="T_78b05_row597_col14" class="data row597 col14" >0.015700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row598" class="row_heading level0 row598" >599</th>
          <td id="T_78b05_row598_col0" class="data row598 col0" >None</td>
          <td id="T_78b05_row598_col1" class="data row598 col1" >0.032800</td>
          <td id="T_78b05_row598_col2" class="data row598 col2" >0.064700</td>
          <td id="T_78b05_row598_col3" class="data row598 col3" >0.017700</td>
          <td id="T_78b05_row598_col4" class="data row598 col4" >-0.003200</td>
          <td id="T_78b05_row598_col5" class="data row598 col5" >0.000000</td>
          <td id="T_78b05_row598_col6" class="data row598 col6" >-0.009700</td>
          <td id="T_78b05_row598_col7" class="data row598 col7" >0.024300</td>
          <td id="T_78b05_row598_col8" class="data row598 col8" >0.012800</td>
          <td id="T_78b05_row598_col9" class="data row598 col9" >0.094500</td>
          <td id="T_78b05_row598_col10" class="data row598 col10" >0.013100</td>
          <td id="T_78b05_row598_col11" class="data row598 col11" >0.014700</td>
          <td id="T_78b05_row598_col12" class="data row598 col12" >0.000600</td>
          <td id="T_78b05_row598_col13" class="data row598 col13" >0.005800</td>
          <td id="T_78b05_row598_col14" class="data row598 col14" >0.026100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row599" class="row_heading level0 row599" >600</th>
          <td id="T_78b05_row599_col0" class="data row599 col0" >PC1</td>
          <td id="T_78b05_row599_col1" class="data row599 col1" >0.019600</td>
          <td id="T_78b05_row599_col2" class="data row599 col2" >0.044200</td>
          <td id="T_78b05_row599_col3" class="data row599 col3" >-0.092700</td>
          <td id="T_78b05_row599_col4" class="data row599 col4" >-0.007700</td>
          <td id="T_78b05_row599_col5" class="data row599 col5" >-0.033500</td>
          <td id="T_78b05_row599_col6" class="data row599 col6" >-0.047200</td>
          <td id="T_78b05_row599_col7" class="data row599 col7" >0.005600</td>
          <td id="T_78b05_row599_col8" class="data row599 col8" >0.026000</td>
          <td id="T_78b05_row599_col9" class="data row599 col9" >0.074000</td>
          <td id="T_78b05_row599_col10" class="data row599 col10" >0.123500</td>
          <td id="T_78b05_row599_col11" class="data row599 col11" >0.010200</td>
          <td id="T_78b05_row599_col12" class="data row599 col12" >0.032900</td>
          <td id="T_78b05_row599_col13" class="data row599 col13" >0.043300</td>
          <td id="T_78b05_row599_col14" class="data row599 col14" >0.007300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row600" class="row_heading level0 row600" >601</th>
          <td id="T_78b05_row600_col0" class="data row600 col0" >None</td>
          <td id="T_78b05_row600_col1" class="data row600 col1" >0.029800</td>
          <td id="T_78b05_row600_col2" class="data row600 col2" >0.043400</td>
          <td id="T_78b05_row600_col3" class="data row600 col3" >0.022700</td>
          <td id="T_78b05_row600_col4" class="data row600 col4" >0.011700</td>
          <td id="T_78b05_row600_col5" class="data row600 col5" >-0.054400</td>
          <td id="T_78b05_row600_col6" class="data row600 col6" >-0.018500</td>
          <td id="T_78b05_row600_col7" class="data row600 col7" >0.016300</td>
          <td id="T_78b05_row600_col8" class="data row600 col8" >0.015800</td>
          <td id="T_78b05_row600_col9" class="data row600 col9" >0.073200</td>
          <td id="T_78b05_row600_col10" class="data row600 col10" >0.008100</td>
          <td id="T_78b05_row600_col11" class="data row600 col11" >0.029600</td>
          <td id="T_78b05_row600_col12" class="data row600 col12" >0.053800</td>
          <td id="T_78b05_row600_col13" class="data row600 col13" >0.014600</td>
          <td id="T_78b05_row600_col14" class="data row600 col14" >0.018100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row601" class="row_heading level0 row601" >602</th>
          <td id="T_78b05_row601_col0" class="data row601 col0" >None</td>
          <td id="T_78b05_row601_col1" class="data row601 col1" >0.039100</td>
          <td id="T_78b05_row601_col2" class="data row601 col2" >0.051800</td>
          <td id="T_78b05_row601_col3" class="data row601 col3" >0.061400</td>
          <td id="T_78b05_row601_col4" class="data row601 col4" >-0.018900</td>
          <td id="T_78b05_row601_col5" class="data row601 col5" >0.054800</td>
          <td id="T_78b05_row601_col6" class="data row601 col6" >0.055900</td>
          <td id="T_78b05_row601_col7" class="data row601 col7" >0.044900</td>
          <td id="T_78b05_row601_col8" class="data row601 col8" >0.006500</td>
          <td id="T_78b05_row601_col9" class="data row601 col9" >0.081600</td>
          <td id="T_78b05_row601_col10" class="data row601 col10" >0.030700</td>
          <td id="T_78b05_row601_col11" class="data row601 col11" >0.001000</td>
          <td id="T_78b05_row601_col12" class="data row601 col12" >0.055400</td>
          <td id="T_78b05_row601_col13" class="data row601 col13" >0.059800</td>
          <td id="T_78b05_row601_col14" class="data row601 col14" >0.046600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row602" class="row_heading level0 row602" >603</th>
          <td id="T_78b05_row602_col0" class="data row602 col0" >None</td>
          <td id="T_78b05_row602_col1" class="data row602 col1" >0.029600</td>
          <td id="T_78b05_row602_col2" class="data row602 col2" >0.056600</td>
          <td id="T_78b05_row602_col3" class="data row602 col3" >-0.014200</td>
          <td id="T_78b05_row602_col4" class="data row602 col4" >-0.013200</td>
          <td id="T_78b05_row602_col5" class="data row602 col5" >-0.055300</td>
          <td id="T_78b05_row602_col6" class="data row602 col6" >0.020200</td>
          <td id="T_78b05_row602_col7" class="data row602 col7" >-0.013800</td>
          <td id="T_78b05_row602_col8" class="data row602 col8" >0.015900</td>
          <td id="T_78b05_row602_col9" class="data row602 col9" >0.086400</td>
          <td id="T_78b05_row602_col10" class="data row602 col10" >0.045000</td>
          <td id="T_78b05_row602_col11" class="data row602 col11" >0.004800</td>
          <td id="T_78b05_row602_col12" class="data row602 col12" >0.054700</td>
          <td id="T_78b05_row602_col13" class="data row602 col13" >0.024100</td>
          <td id="T_78b05_row602_col14" class="data row602 col14" >0.012100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row603" class="row_heading level0 row603" >604</th>
          <td id="T_78b05_row603_col0" class="data row603 col0" >PC1</td>
          <td id="T_78b05_row603_col1" class="data row603 col1" >0.024800</td>
          <td id="T_78b05_row603_col2" class="data row603 col2" >0.078800</td>
          <td id="T_78b05_row603_col3" class="data row603 col3" >-0.027600</td>
          <td id="T_78b05_row603_col4" class="data row603 col4" >0.011900</td>
          <td id="T_78b05_row603_col5" class="data row603 col5" >-0.019300</td>
          <td id="T_78b05_row603_col6" class="data row603 col6" >-0.001700</td>
          <td id="T_78b05_row603_col7" class="data row603 col7" >-0.034800</td>
          <td id="T_78b05_row603_col8" class="data row603 col8" >0.020800</td>
          <td id="T_78b05_row603_col9" class="data row603 col9" >0.108600</td>
          <td id="T_78b05_row603_col10" class="data row603 col10" >0.058300</td>
          <td id="T_78b05_row603_col11" class="data row603 col11" >0.029800</td>
          <td id="T_78b05_row603_col12" class="data row603 col12" >0.018700</td>
          <td id="T_78b05_row603_col13" class="data row603 col13" >0.002200</td>
          <td id="T_78b05_row603_col14" class="data row603 col14" >0.033100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row604" class="row_heading level0 row604" >605</th>
          <td id="T_78b05_row604_col0" class="data row604 col0" >PC1</td>
          <td id="T_78b05_row604_col1" class="data row604 col1" >0.025800</td>
          <td id="T_78b05_row604_col2" class="data row604 col2" >0.054500</td>
          <td id="T_78b05_row604_col3" class="data row604 col3" >-0.006700</td>
          <td id="T_78b05_row604_col4" class="data row604 col4" >0.019100</td>
          <td id="T_78b05_row604_col5" class="data row604 col5" >-0.015200</td>
          <td id="T_78b05_row604_col6" class="data row604 col6" >-0.001900</td>
          <td id="T_78b05_row604_col7" class="data row604 col7" >0.021300</td>
          <td id="T_78b05_row604_col8" class="data row604 col8" >0.019800</td>
          <td id="T_78b05_row604_col9" class="data row604 col9" >0.084300</td>
          <td id="T_78b05_row604_col10" class="data row604 col10" >0.037400</td>
          <td id="T_78b05_row604_col11" class="data row604 col11" >0.037000</td>
          <td id="T_78b05_row604_col12" class="data row604 col12" >0.014600</td>
          <td id="T_78b05_row604_col13" class="data row604 col13" >0.002000</td>
          <td id="T_78b05_row604_col14" class="data row604 col14" >0.023100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row605" class="row_heading level0 row605" >606</th>
          <td id="T_78b05_row605_col0" class="data row605 col0" >None</td>
          <td id="T_78b05_row605_col1" class="data row605 col1" >0.035800</td>
          <td id="T_78b05_row605_col2" class="data row605 col2" >0.059400</td>
          <td id="T_78b05_row605_col3" class="data row605 col3" >-0.007500</td>
          <td id="T_78b05_row605_col4" class="data row605 col4" >0.018200</td>
          <td id="T_78b05_row605_col5" class="data row605 col5" >-0.028800</td>
          <td id="T_78b05_row605_col6" class="data row605 col6" >0.004400</td>
          <td id="T_78b05_row605_col7" class="data row605 col7" >-0.009100</td>
          <td id="T_78b05_row605_col8" class="data row605 col8" >0.009700</td>
          <td id="T_78b05_row605_col9" class="data row605 col9" >0.089200</td>
          <td id="T_78b05_row605_col10" class="data row605 col10" >0.038300</td>
          <td id="T_78b05_row605_col11" class="data row605 col11" >0.036100</td>
          <td id="T_78b05_row605_col12" class="data row605 col12" >0.028100</td>
          <td id="T_78b05_row605_col13" class="data row605 col13" >0.008300</td>
          <td id="T_78b05_row605_col14" class="data row605 col14" >0.007400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row606" class="row_heading level0 row606" >607</th>
          <td id="T_78b05_row606_col0" class="data row606 col0" >None</td>
          <td id="T_78b05_row606_col1" class="data row606 col1" >0.035000</td>
          <td id="T_78b05_row606_col2" class="data row606 col2" >-0.008900</td>
          <td id="T_78b05_row606_col3" class="data row606 col3" >-0.011400</td>
          <td id="T_78b05_row606_col4" class="data row606 col4" >0.090900</td>
          <td id="T_78b05_row606_col5" class="data row606 col5" >-0.041200</td>
          <td id="T_78b05_row606_col6" class="data row606 col6" >0.022100</td>
          <td id="T_78b05_row606_col7" class="data row606 col7" >0.010400</td>
          <td id="T_78b05_row606_col8" class="data row606 col8" >0.010600</td>
          <td id="T_78b05_row606_col9" class="data row606 col9" >0.021000</td>
          <td id="T_78b05_row606_col10" class="data row606 col10" >0.042100</td>
          <td id="T_78b05_row606_col11" class="data row606 col11" >0.108800</td>
          <td id="T_78b05_row606_col12" class="data row606 col12" >0.040500</td>
          <td id="T_78b05_row606_col13" class="data row606 col13" >0.026000</td>
          <td id="T_78b05_row606_col14" class="data row606 col14" >0.012100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row607" class="row_heading level0 row607" >608</th>
          <td id="T_78b05_row607_col0" class="data row607 col0" >None</td>
          <td id="T_78b05_row607_col1" class="data row607 col1" >0.033100</td>
          <td id="T_78b05_row607_col2" class="data row607 col2" >0.013900</td>
          <td id="T_78b05_row607_col3" class="data row607 col3" >-0.057400</td>
          <td id="T_78b05_row607_col4" class="data row607 col4" >-0.028300</td>
          <td id="T_78b05_row607_col5" class="data row607 col5" >-0.024400</td>
          <td id="T_78b05_row607_col6" class="data row607 col6" >-0.036900</td>
          <td id="T_78b05_row607_col7" class="data row607 col7" >0.032100</td>
          <td id="T_78b05_row607_col8" class="data row607 col8" >0.012500</td>
          <td id="T_78b05_row607_col9" class="data row607 col9" >0.043700</td>
          <td id="T_78b05_row607_col10" class="data row607 col10" >0.088200</td>
          <td id="T_78b05_row607_col11" class="data row607 col11" >0.010400</td>
          <td id="T_78b05_row607_col12" class="data row607 col12" >0.023700</td>
          <td id="T_78b05_row607_col13" class="data row607 col13" >0.033000</td>
          <td id="T_78b05_row607_col14" class="data row607 col14" >0.033800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row608" class="row_heading level0 row608" >609</th>
          <td id="T_78b05_row608_col0" class="data row608 col0" >None</td>
          <td id="T_78b05_row608_col1" class="data row608 col1" >0.041600</td>
          <td id="T_78b05_row608_col2" class="data row608 col2" >0.021400</td>
          <td id="T_78b05_row608_col3" class="data row608 col3" >0.031600</td>
          <td id="T_78b05_row608_col4" class="data row608 col4" >-0.007600</td>
          <td id="T_78b05_row608_col5" class="data row608 col5" >0.062800</td>
          <td id="T_78b05_row608_col6" class="data row608 col6" >-0.025600</td>
          <td id="T_78b05_row608_col7" class="data row608 col7" >-0.024900</td>
          <td id="T_78b05_row608_col8" class="data row608 col8" >0.003900</td>
          <td id="T_78b05_row608_col9" class="data row608 col9" >0.051200</td>
          <td id="T_78b05_row608_col10" class="data row608 col10" >0.000800</td>
          <td id="T_78b05_row608_col11" class="data row608 col11" >0.010300</td>
          <td id="T_78b05_row608_col12" class="data row608 col12" >0.063400</td>
          <td id="T_78b05_row608_col13" class="data row608 col13" >0.021700</td>
          <td id="T_78b05_row608_col14" class="data row608 col14" >0.023100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row609" class="row_heading level0 row609" >610</th>
          <td id="T_78b05_row609_col0" class="data row609 col0" >None</td>
          <td id="T_78b05_row609_col1" class="data row609 col1" >0.037200</td>
          <td id="T_78b05_row609_col2" class="data row609 col2" >-0.045500</td>
          <td id="T_78b05_row609_col3" class="data row609 col3" >-0.028000</td>
          <td id="T_78b05_row609_col4" class="data row609 col4" >0.038500</td>
          <td id="T_78b05_row609_col5" class="data row609 col5" >-0.013700</td>
          <td id="T_78b05_row609_col6" class="data row609 col6" >0.025700</td>
          <td id="T_78b05_row609_col7" class="data row609 col7" >0.022500</td>
          <td id="T_78b05_row609_col8" class="data row609 col8" >0.008300</td>
          <td id="T_78b05_row609_col9" class="data row609 col9" >0.015700</td>
          <td id="T_78b05_row609_col10" class="data row609 col10" >0.058800</td>
          <td id="T_78b05_row609_col11" class="data row609 col11" >0.056500</td>
          <td id="T_78b05_row609_col12" class="data row609 col12" >0.013000</td>
          <td id="T_78b05_row609_col13" class="data row609 col13" >0.029600</td>
          <td id="T_78b05_row609_col14" class="data row609 col14" >0.024300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row610" class="row_heading level0 row610" >611</th>
          <td id="T_78b05_row610_col0" class="data row610 col0" >None</td>
          <td id="T_78b05_row610_col1" class="data row610 col1" >0.035600</td>
          <td id="T_78b05_row610_col2" class="data row610 col2" >0.032500</td>
          <td id="T_78b05_row610_col3" class="data row610 col3" >0.019900</td>
          <td id="T_78b05_row610_col4" class="data row610 col4" >-0.045500</td>
          <td id="T_78b05_row610_col5" class="data row610 col5" >0.047300</td>
          <td id="T_78b05_row610_col6" class="data row610 col6" >-0.066800</td>
          <td id="T_78b05_row610_col7" class="data row610 col7" >-0.035100</td>
          <td id="T_78b05_row610_col8" class="data row610 col8" >0.009900</td>
          <td id="T_78b05_row610_col9" class="data row610 col9" >0.062300</td>
          <td id="T_78b05_row610_col10" class="data row610 col10" >0.010800</td>
          <td id="T_78b05_row610_col11" class="data row610 col11" >0.027500</td>
          <td id="T_78b05_row610_col12" class="data row610 col12" >0.048000</td>
          <td id="T_78b05_row610_col13" class="data row610 col13" >0.062900</td>
          <td id="T_78b05_row610_col14" class="data row610 col14" >0.033400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row611" class="row_heading level0 row611" >612</th>
          <td id="T_78b05_row611_col0" class="data row611 col0" >None</td>
          <td id="T_78b05_row611_col1" class="data row611 col1" >0.031700</td>
          <td id="T_78b05_row611_col2" class="data row611 col2" >0.075700</td>
          <td id="T_78b05_row611_col3" class="data row611 col3" >0.006400</td>
          <td id="T_78b05_row611_col4" class="data row611 col4" >-0.044500</td>
          <td id="T_78b05_row611_col5" class="data row611 col5" >-0.014800</td>
          <td id="T_78b05_row611_col6" class="data row611 col6" >-0.063000</td>
          <td id="T_78b05_row611_col7" class="data row611 col7" >0.014300</td>
          <td id="T_78b05_row611_col8" class="data row611 col8" >0.013800</td>
          <td id="T_78b05_row611_col9" class="data row611 col9" >0.105600</td>
          <td id="T_78b05_row611_col10" class="data row611 col10" >0.024300</td>
          <td id="T_78b05_row611_col11" class="data row611 col11" >0.026600</td>
          <td id="T_78b05_row611_col12" class="data row611 col12" >0.014100</td>
          <td id="T_78b05_row611_col13" class="data row611 col13" >0.059100</td>
          <td id="T_78b05_row611_col14" class="data row611 col14" >0.016100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row612" class="row_heading level0 row612" >613</th>
          <td id="T_78b05_row612_col0" class="data row612 col0" >None</td>
          <td id="T_78b05_row612_col1" class="data row612 col1" >0.037200</td>
          <td id="T_78b05_row612_col2" class="data row612 col2" >0.028200</td>
          <td id="T_78b05_row612_col3" class="data row612 col3" >0.015400</td>
          <td id="T_78b05_row612_col4" class="data row612 col4" >0.015800</td>
          <td id="T_78b05_row612_col5" class="data row612 col5" >-0.003500</td>
          <td id="T_78b05_row612_col6" class="data row612 col6" >-0.087400</td>
          <td id="T_78b05_row612_col7" class="data row612 col7" >0.020900</td>
          <td id="T_78b05_row612_col8" class="data row612 col8" >0.008300</td>
          <td id="T_78b05_row612_col9" class="data row612 col9" >0.058000</td>
          <td id="T_78b05_row612_col10" class="data row612 col10" >0.015400</td>
          <td id="T_78b05_row612_col11" class="data row612 col11" >0.033700</td>
          <td id="T_78b05_row612_col12" class="data row612 col12" >0.002900</td>
          <td id="T_78b05_row612_col13" class="data row612 col13" >0.083500</td>
          <td id="T_78b05_row612_col14" class="data row612 col14" >0.022700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row613" class="row_heading level0 row613" >614</th>
          <td id="T_78b05_row613_col0" class="data row613 col0" >None</td>
          <td id="T_78b05_row613_col1" class="data row613 col1" >0.039100</td>
          <td id="T_78b05_row613_col2" class="data row613 col2" >-0.009600</td>
          <td id="T_78b05_row613_col3" class="data row613 col3" >0.025600</td>
          <td id="T_78b05_row613_col4" class="data row613 col4" >0.020000</td>
          <td id="T_78b05_row613_col5" class="data row613 col5" >-0.056200</td>
          <td id="T_78b05_row613_col6" class="data row613 col6" >0.020900</td>
          <td id="T_78b05_row613_col7" class="data row613 col7" >0.049900</td>
          <td id="T_78b05_row613_col8" class="data row613 col8" >0.006400</td>
          <td id="T_78b05_row613_col9" class="data row613 col9" >0.020200</td>
          <td id="T_78b05_row613_col10" class="data row613 col10" >0.005100</td>
          <td id="T_78b05_row613_col11" class="data row613 col11" >0.037900</td>
          <td id="T_78b05_row613_col12" class="data row613 col12" >0.055600</td>
          <td id="T_78b05_row613_col13" class="data row613 col13" >0.024800</td>
          <td id="T_78b05_row613_col14" class="data row613 col14" >0.051600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row614" class="row_heading level0 row614" >615</th>
          <td id="T_78b05_row614_col0" class="data row614 col0" >PC1</td>
          <td id="T_78b05_row614_col1" class="data row614 col1" >0.026100</td>
          <td id="T_78b05_row614_col2" class="data row614 col2" >0.053300</td>
          <td id="T_78b05_row614_col3" class="data row614 col3" >-0.099300</td>
          <td id="T_78b05_row614_col4" class="data row614 col4" >0.001500</td>
          <td id="T_78b05_row614_col5" class="data row614 col5" >0.000600</td>
          <td id="T_78b05_row614_col6" class="data row614 col6" >0.027300</td>
          <td id="T_78b05_row614_col7" class="data row614 col7" >0.005200</td>
          <td id="T_78b05_row614_col8" class="data row614 col8" >0.019500</td>
          <td id="T_78b05_row614_col9" class="data row614 col9" >0.083100</td>
          <td id="T_78b05_row614_col10" class="data row614 col10" >0.130000</td>
          <td id="T_78b05_row614_col11" class="data row614 col11" >0.019400</td>
          <td id="T_78b05_row614_col12" class="data row614 col12" >0.001200</td>
          <td id="T_78b05_row614_col13" class="data row614 col13" >0.031200</td>
          <td id="T_78b05_row614_col14" class="data row614 col14" >0.007000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row615" class="row_heading level0 row615" >616</th>
          <td id="T_78b05_row615_col0" class="data row615 col0" >None</td>
          <td id="T_78b05_row615_col1" class="data row615 col1" >0.035000</td>
          <td id="T_78b05_row615_col2" class="data row615 col2" >0.031000</td>
          <td id="T_78b05_row615_col3" class="data row615 col3" >-0.035100</td>
          <td id="T_78b05_row615_col4" class="data row615 col4" >-0.057400</td>
          <td id="T_78b05_row615_col5" class="data row615 col5" >0.033800</td>
          <td id="T_78b05_row615_col6" class="data row615 col6" >0.061300</td>
          <td id="T_78b05_row615_col7" class="data row615 col7" >-0.000500</td>
          <td id="T_78b05_row615_col8" class="data row615 col8" >0.010500</td>
          <td id="T_78b05_row615_col9" class="data row615 col9" >0.060900</td>
          <td id="T_78b05_row615_col10" class="data row615 col10" >0.065800</td>
          <td id="T_78b05_row615_col11" class="data row615 col11" >0.039500</td>
          <td id="T_78b05_row615_col12" class="data row615 col12" >0.034400</td>
          <td id="T_78b05_row615_col13" class="data row615 col13" >0.065200</td>
          <td id="T_78b05_row615_col14" class="data row615 col14" >0.001200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row616" class="row_heading level0 row616" >617</th>
          <td id="T_78b05_row616_col0" class="data row616 col0" >None</td>
          <td id="T_78b05_row616_col1" class="data row616 col1" >0.034600</td>
          <td id="T_78b05_row616_col2" class="data row616 col2" >0.068700</td>
          <td id="T_78b05_row616_col3" class="data row616 col3" >0.033100</td>
          <td id="T_78b05_row616_col4" class="data row616 col4" >0.023100</td>
          <td id="T_78b05_row616_col5" class="data row616 col5" >-0.038100</td>
          <td id="T_78b05_row616_col6" class="data row616 col6" >-0.004500</td>
          <td id="T_78b05_row616_col7" class="data row616 col7" >-0.007100</td>
          <td id="T_78b05_row616_col8" class="data row616 col8" >0.010900</td>
          <td id="T_78b05_row616_col9" class="data row616 col9" >0.098600</td>
          <td id="T_78b05_row616_col10" class="data row616 col10" >0.002400</td>
          <td id="T_78b05_row616_col11" class="data row616 col11" >0.041000</td>
          <td id="T_78b05_row616_col12" class="data row616 col12" >0.037400</td>
          <td id="T_78b05_row616_col13" class="data row616 col13" >0.000700</td>
          <td id="T_78b05_row616_col14" class="data row616 col14" >0.005300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row617" class="row_heading level0 row617" >618</th>
          <td id="T_78b05_row617_col0" class="data row617 col0" >None</td>
          <td id="T_78b05_row617_col1" class="data row617 col1" >0.043800</td>
          <td id="T_78b05_row617_col2" class="data row617 col2" >0.006100</td>
          <td id="T_78b05_row617_col3" class="data row617 col3" >0.030100</td>
          <td id="T_78b05_row617_col4" class="data row617 col4" >-0.028500</td>
          <td id="T_78b05_row617_col5" class="data row617 col5" >0.058700</td>
          <td id="T_78b05_row617_col6" class="data row617 col6" >0.022600</td>
          <td id="T_78b05_row617_col7" class="data row617 col7" >-0.007400</td>
          <td id="T_78b05_row617_col8" class="data row617 col8" >0.001800</td>
          <td id="T_78b05_row617_col9" class="data row617 col9" >0.035900</td>
          <td id="T_78b05_row617_col10" class="data row617 col10" >0.000600</td>
          <td id="T_78b05_row617_col11" class="data row617 col11" >0.010600</td>
          <td id="T_78b05_row617_col12" class="data row617 col12" >0.059300</td>
          <td id="T_78b05_row617_col13" class="data row617 col13" >0.026500</td>
          <td id="T_78b05_row617_col14" class="data row617 col14" >0.005600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row618" class="row_heading level0 row618" >619</th>
          <td id="T_78b05_row618_col0" class="data row618 col0" >None</td>
          <td id="T_78b05_row618_col1" class="data row618 col1" >0.042600</td>
          <td id="T_78b05_row618_col2" class="data row618 col2" >0.008500</td>
          <td id="T_78b05_row618_col3" class="data row618 col3" >0.009400</td>
          <td id="T_78b05_row618_col4" class="data row618 col4" >-0.001100</td>
          <td id="T_78b05_row618_col5" class="data row618 col5" >0.055700</td>
          <td id="T_78b05_row618_col6" class="data row618 col6" >0.050000</td>
          <td id="T_78b05_row618_col7" class="data row618 col7" >-0.006500</td>
          <td id="T_78b05_row618_col8" class="data row618 col8" >0.002900</td>
          <td id="T_78b05_row618_col9" class="data row618 col9" >0.038400</td>
          <td id="T_78b05_row618_col10" class="data row618 col10" >0.021400</td>
          <td id="T_78b05_row618_col11" class="data row618 col11" >0.016800</td>
          <td id="T_78b05_row618_col12" class="data row618 col12" >0.056400</td>
          <td id="T_78b05_row618_col13" class="data row618 col13" >0.053900</td>
          <td id="T_78b05_row618_col14" class="data row618 col14" >0.004700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row619" class="row_heading level0 row619" >620</th>
          <td id="T_78b05_row619_col0" class="data row619 col0" >None</td>
          <td id="T_78b05_row619_col1" class="data row619 col1" >0.033000</td>
          <td id="T_78b05_row619_col2" class="data row619 col2" >0.035100</td>
          <td id="T_78b05_row619_col3" class="data row619 col3" >0.003700</td>
          <td id="T_78b05_row619_col4" class="data row619 col4" >-0.027800</td>
          <td id="T_78b05_row619_col5" class="data row619 col5" >0.005000</td>
          <td id="T_78b05_row619_col6" class="data row619 col6" >-0.031200</td>
          <td id="T_78b05_row619_col7" class="data row619 col7" >0.027900</td>
          <td id="T_78b05_row619_col8" class="data row619 col8" >0.012500</td>
          <td id="T_78b05_row619_col9" class="data row619 col9" >0.064900</td>
          <td id="T_78b05_row619_col10" class="data row619 col10" >0.027000</td>
          <td id="T_78b05_row619_col11" class="data row619 col11" >0.009900</td>
          <td id="T_78b05_row619_col12" class="data row619 col12" >0.005600</td>
          <td id="T_78b05_row619_col13" class="data row619 col13" >0.027300</td>
          <td id="T_78b05_row619_col14" class="data row619 col14" >0.029700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row620" class="row_heading level0 row620" >621</th>
          <td id="T_78b05_row620_col0" class="data row620 col0" >None</td>
          <td id="T_78b05_row620_col1" class="data row620 col1" >0.040500</td>
          <td id="T_78b05_row620_col2" class="data row620 col2" >0.016200</td>
          <td id="T_78b05_row620_col3" class="data row620 col3" >0.005300</td>
          <td id="T_78b05_row620_col4" class="data row620 col4" >0.006600</td>
          <td id="T_78b05_row620_col5" class="data row620 col5" >0.051000</td>
          <td id="T_78b05_row620_col6" class="data row620 col6" >0.068500</td>
          <td id="T_78b05_row620_col7" class="data row620 col7" >0.008900</td>
          <td id="T_78b05_row620_col8" class="data row620 col8" >0.005100</td>
          <td id="T_78b05_row620_col9" class="data row620 col9" >0.046100</td>
          <td id="T_78b05_row620_col10" class="data row620 col10" >0.025400</td>
          <td id="T_78b05_row620_col11" class="data row620 col11" >0.024500</td>
          <td id="T_78b05_row620_col12" class="data row620 col12" >0.051600</td>
          <td id="T_78b05_row620_col13" class="data row620 col13" >0.072400</td>
          <td id="T_78b05_row620_col14" class="data row620 col14" >0.010700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row621" class="row_heading level0 row621" >622</th>
          <td id="T_78b05_row621_col0" class="data row621 col0" >None</td>
          <td id="T_78b05_row621_col1" class="data row621 col1" >0.040400</td>
          <td id="T_78b05_row621_col2" class="data row621 col2" >-0.002100</td>
          <td id="T_78b05_row621_col3" class="data row621 col3" >-0.019300</td>
          <td id="T_78b05_row621_col4" class="data row621 col4" >-0.022400</td>
          <td id="T_78b05_row621_col5" class="data row621 col5" >0.043800</td>
          <td id="T_78b05_row621_col6" class="data row621 col6" >0.031200</td>
          <td id="T_78b05_row621_col7" class="data row621 col7" >0.036800</td>
          <td id="T_78b05_row621_col8" class="data row621 col8" >0.005200</td>
          <td id="T_78b05_row621_col9" class="data row621 col9" >0.027700</td>
          <td id="T_78b05_row621_col10" class="data row621 col10" >0.050000</td>
          <td id="T_78b05_row621_col11" class="data row621 col11" >0.004500</td>
          <td id="T_78b05_row621_col12" class="data row621 col12" >0.044500</td>
          <td id="T_78b05_row621_col13" class="data row621 col13" >0.035100</td>
          <td id="T_78b05_row621_col14" class="data row621 col14" >0.038600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row622" class="row_heading level0 row622" >623</th>
          <td id="T_78b05_row622_col0" class="data row622 col0" >None</td>
          <td id="T_78b05_row622_col1" class="data row622 col1" >0.035800</td>
          <td id="T_78b05_row622_col2" class="data row622 col2" >0.056500</td>
          <td id="T_78b05_row622_col3" class="data row622 col3" >-0.016500</td>
          <td id="T_78b05_row622_col4" class="data row622 col4" >-0.009500</td>
          <td id="T_78b05_row622_col5" class="data row622 col5" >-0.007700</td>
          <td id="T_78b05_row622_col6" class="data row622 col6" >-0.014800</td>
          <td id="T_78b05_row622_col7" class="data row622 col7" >0.035900</td>
          <td id="T_78b05_row622_col8" class="data row622 col8" >0.009800</td>
          <td id="T_78b05_row622_col9" class="data row622 col9" >0.086300</td>
          <td id="T_78b05_row622_col10" class="data row622 col10" >0.047300</td>
          <td id="T_78b05_row622_col11" class="data row622 col11" >0.008400</td>
          <td id="T_78b05_row622_col12" class="data row622 col12" >0.007000</td>
          <td id="T_78b05_row622_col13" class="data row622 col13" >0.010900</td>
          <td id="T_78b05_row622_col14" class="data row622 col14" >0.037700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row623" class="row_heading level0 row623" >624</th>
          <td id="T_78b05_row623_col0" class="data row623 col0" >PC1</td>
          <td id="T_78b05_row623_col1" class="data row623 col1" >0.026500</td>
          <td id="T_78b05_row623_col2" class="data row623 col2" >-0.034500</td>
          <td id="T_78b05_row623_col3" class="data row623 col3" >-0.046800</td>
          <td id="T_78b05_row623_col4" class="data row623 col4" >0.057700</td>
          <td id="T_78b05_row623_col5" class="data row623 col5" >-0.054800</td>
          <td id="T_78b05_row623_col6" class="data row623 col6" >-0.027000</td>
          <td id="T_78b05_row623_col7" class="data row623 col7" >0.027900</td>
          <td id="T_78b05_row623_col8" class="data row623 col8" >0.019000</td>
          <td id="T_78b05_row623_col9" class="data row623 col9" >0.004700</td>
          <td id="T_78b05_row623_col10" class="data row623 col10" >0.077500</td>
          <td id="T_78b05_row623_col11" class="data row623 col11" >0.075600</td>
          <td id="T_78b05_row623_col12" class="data row623 col12" >0.054100</td>
          <td id="T_78b05_row623_col13" class="data row623 col13" >0.023100</td>
          <td id="T_78b05_row623_col14" class="data row623 col14" >0.029700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row624" class="row_heading level0 row624" >625</th>
          <td id="T_78b05_row624_col0" class="data row624 col0" >None</td>
          <td id="T_78b05_row624_col1" class="data row624 col1" >0.037700</td>
          <td id="T_78b05_row624_col2" class="data row624 col2" >-0.032700</td>
          <td id="T_78b05_row624_col3" class="data row624 col3" >-0.009100</td>
          <td id="T_78b05_row624_col4" class="data row624 col4" >0.003200</td>
          <td id="T_78b05_row624_col5" class="data row624 col5" >-0.065800</td>
          <td id="T_78b05_row624_col6" class="data row624 col6" >0.024000</td>
          <td id="T_78b05_row624_col7" class="data row624 col7" >-0.003600</td>
          <td id="T_78b05_row624_col8" class="data row624 col8" >0.007900</td>
          <td id="T_78b05_row624_col9" class="data row624 col9" >0.002900</td>
          <td id="T_78b05_row624_col10" class="data row624 col10" >0.039800</td>
          <td id="T_78b05_row624_col11" class="data row624 col11" >0.021100</td>
          <td id="T_78b05_row624_col12" class="data row624 col12" >0.065100</td>
          <td id="T_78b05_row624_col13" class="data row624 col13" >0.027900</td>
          <td id="T_78b05_row624_col14" class="data row624 col14" >0.001900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row625" class="row_heading level0 row625" >626</th>
          <td id="T_78b05_row625_col0" class="data row625 col0" >None</td>
          <td id="T_78b05_row625_col1" class="data row625 col1" >0.042200</td>
          <td id="T_78b05_row625_col2" class="data row625 col2" >0.018900</td>
          <td id="T_78b05_row625_col3" class="data row625 col3" >0.048000</td>
          <td id="T_78b05_row625_col4" class="data row625 col4" >-0.077600</td>
          <td id="T_78b05_row625_col5" class="data row625 col5" >-0.029200</td>
          <td id="T_78b05_row625_col6" class="data row625 col6" >-0.021600</td>
          <td id="T_78b05_row625_col7" class="data row625 col7" >0.049700</td>
          <td id="T_78b05_row625_col8" class="data row625 col8" >0.003300</td>
          <td id="T_78b05_row625_col9" class="data row625 col9" >0.048700</td>
          <td id="T_78b05_row625_col10" class="data row625 col10" >0.017200</td>
          <td id="T_78b05_row625_col11" class="data row625 col11" >0.059700</td>
          <td id="T_78b05_row625_col12" class="data row625 col12" >0.028600</td>
          <td id="T_78b05_row625_col13" class="data row625 col13" >0.017700</td>
          <td id="T_78b05_row625_col14" class="data row625 col14" >0.051400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row626" class="row_heading level0 row626" >627</th>
          <td id="T_78b05_row626_col0" class="data row626 col0" >None</td>
          <td id="T_78b05_row626_col1" class="data row626 col1" >0.034000</td>
          <td id="T_78b05_row626_col2" class="data row626 col2" >0.042100</td>
          <td id="T_78b05_row626_col3" class="data row626 col3" >-0.026800</td>
          <td id="T_78b05_row626_col4" class="data row626 col4" >0.034000</td>
          <td id="T_78b05_row626_col5" class="data row626 col5" >0.019500</td>
          <td id="T_78b05_row626_col6" class="data row626 col6" >0.038300</td>
          <td id="T_78b05_row626_col7" class="data row626 col7" >-0.005100</td>
          <td id="T_78b05_row626_col8" class="data row626 col8" >0.011500</td>
          <td id="T_78b05_row626_col9" class="data row626 col9" >0.072000</td>
          <td id="T_78b05_row626_col10" class="data row626 col10" >0.057600</td>
          <td id="T_78b05_row626_col11" class="data row626 col11" >0.051900</td>
          <td id="T_78b05_row626_col12" class="data row626 col12" >0.020100</td>
          <td id="T_78b05_row626_col13" class="data row626 col13" >0.042200</td>
          <td id="T_78b05_row626_col14" class="data row626 col14" >0.003400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row627" class="row_heading level0 row627" >628</th>
          <td id="T_78b05_row627_col0" class="data row627 col0" >PC1</td>
          <td id="T_78b05_row627_col1" class="data row627 col1" >0.025600</td>
          <td id="T_78b05_row627_col2" class="data row627 col2" >0.027200</td>
          <td id="T_78b05_row627_col3" class="data row627 col3" >-0.051300</td>
          <td id="T_78b05_row627_col4" class="data row627 col4" >0.097700</td>
          <td id="T_78b05_row627_col5" class="data row627 col5" >-0.058300</td>
          <td id="T_78b05_row627_col6" class="data row627 col6" >-0.033800</td>
          <td id="T_78b05_row627_col7" class="data row627 col7" >-0.027700</td>
          <td id="T_78b05_row627_col8" class="data row627 col8" >0.019900</td>
          <td id="T_78b05_row627_col9" class="data row627 col9" >0.057000</td>
          <td id="T_78b05_row627_col10" class="data row627 col10" >0.082000</td>
          <td id="T_78b05_row627_col11" class="data row627 col11" >0.115600</td>
          <td id="T_78b05_row627_col12" class="data row627 col12" >0.057600</td>
          <td id="T_78b05_row627_col13" class="data row627 col13" >0.029900</td>
          <td id="T_78b05_row627_col14" class="data row627 col14" >0.026000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row628" class="row_heading level0 row628" >629</th>
          <td id="T_78b05_row628_col0" class="data row628 col0" >None</td>
          <td id="T_78b05_row628_col1" class="data row628 col1" >0.033700</td>
          <td id="T_78b05_row628_col2" class="data row628 col2" >-0.049800</td>
          <td id="T_78b05_row628_col3" class="data row628 col3" >-0.055300</td>
          <td id="T_78b05_row628_col4" class="data row628 col4" >0.004800</td>
          <td id="T_78b05_row628_col5" class="data row628 col5" >-0.036700</td>
          <td id="T_78b05_row628_col6" class="data row628 col6" >0.045600</td>
          <td id="T_78b05_row628_col7" class="data row628 col7" >0.044200</td>
          <td id="T_78b05_row628_col8" class="data row628 col8" >0.011900</td>
          <td id="T_78b05_row628_col9" class="data row628 col9" >0.020000</td>
          <td id="T_78b05_row628_col10" class="data row628 col10" >0.086000</td>
          <td id="T_78b05_row628_col11" class="data row628 col11" >0.022700</td>
          <td id="T_78b05_row628_col12" class="data row628 col12" >0.036100</td>
          <td id="T_78b05_row628_col13" class="data row628 col13" >0.049500</td>
          <td id="T_78b05_row628_col14" class="data row628 col14" >0.045900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row629" class="row_heading level0 row629" >630</th>
          <td id="T_78b05_row629_col0" class="data row629 col0" >None</td>
          <td id="T_78b05_row629_col1" class="data row629 col1" >0.031000</td>
          <td id="T_78b05_row629_col2" class="data row629 col2" >-0.014800</td>
          <td id="T_78b05_row629_col3" class="data row629 col3" >-0.010500</td>
          <td id="T_78b05_row629_col4" class="data row629 col4" >0.040400</td>
          <td id="T_78b05_row629_col5" class="data row629 col5" >-0.008900</td>
          <td id="T_78b05_row629_col6" class="data row629 col6" >0.019000</td>
          <td id="T_78b05_row629_col7" class="data row629 col7" >-0.100700</td>
          <td id="T_78b05_row629_col8" class="data row629 col8" >0.014500</td>
          <td id="T_78b05_row629_col9" class="data row629 col9" >0.015000</td>
          <td id="T_78b05_row629_col10" class="data row629 col10" >0.041200</td>
          <td id="T_78b05_row629_col11" class="data row629 col11" >0.058300</td>
          <td id="T_78b05_row629_col12" class="data row629 col12" >0.008300</td>
          <td id="T_78b05_row629_col13" class="data row629 col13" >0.022900</td>
          <td id="T_78b05_row629_col14" class="data row629 col14" >0.099000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row630" class="row_heading level0 row630" >631</th>
          <td id="T_78b05_row630_col0" class="data row630 col0" >None</td>
          <td id="T_78b05_row630_col1" class="data row630 col1" >0.029400</td>
          <td id="T_78b05_row630_col2" class="data row630 col2" >-0.046100</td>
          <td id="T_78b05_row630_col3" class="data row630 col3" >-0.085300</td>
          <td id="T_78b05_row630_col4" class="data row630 col4" >-0.033800</td>
          <td id="T_78b05_row630_col5" class="data row630 col5" >0.007700</td>
          <td id="T_78b05_row630_col6" class="data row630 col6" >-0.042300</td>
          <td id="T_78b05_row630_col7" class="data row630 col7" >-0.013400</td>
          <td id="T_78b05_row630_col8" class="data row630 col8" >0.016100</td>
          <td id="T_78b05_row630_col9" class="data row630 col9" >0.016300</td>
          <td id="T_78b05_row630_col10" class="data row630 col10" >0.116100</td>
          <td id="T_78b05_row630_col11" class="data row630 col11" >0.015900</td>
          <td id="T_78b05_row630_col12" class="data row630 col12" >0.008300</td>
          <td id="T_78b05_row630_col13" class="data row630 col13" >0.038400</td>
          <td id="T_78b05_row630_col14" class="data row630 col14" >0.011600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row631" class="row_heading level0 row631" >632</th>
          <td id="T_78b05_row631_col0" class="data row631 col0" >PC3</td>
          <td id="T_78b05_row631_col1" class="data row631 col1" >0.030100</td>
          <td id="T_78b05_row631_col2" class="data row631 col2" >-0.022500</td>
          <td id="T_78b05_row631_col3" class="data row631 col3" >-0.090800</td>
          <td id="T_78b05_row631_col4" class="data row631 col4" >0.055800</td>
          <td id="T_78b05_row631_col5" class="data row631 col5" >-0.019400</td>
          <td id="T_78b05_row631_col6" class="data row631 col6" >0.006300</td>
          <td id="T_78b05_row631_col7" class="data row631 col7" >-0.044500</td>
          <td id="T_78b05_row631_col8" class="data row631 col8" >0.015400</td>
          <td id="T_78b05_row631_col9" class="data row631 col9" >0.007300</td>
          <td id="T_78b05_row631_col10" class="data row631 col10" >0.121600</td>
          <td id="T_78b05_row631_col11" class="data row631 col11" >0.073700</td>
          <td id="T_78b05_row631_col12" class="data row631 col12" >0.018700</td>
          <td id="T_78b05_row631_col13" class="data row631 col13" >0.010200</td>
          <td id="T_78b05_row631_col14" class="data row631 col14" >0.042800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row632" class="row_heading level0 row632" >633</th>
          <td id="T_78b05_row632_col0" class="data row632 col0" >None</td>
          <td id="T_78b05_row632_col1" class="data row632 col1" >0.031200</td>
          <td id="T_78b05_row632_col2" class="data row632 col2" >0.070900</td>
          <td id="T_78b05_row632_col3" class="data row632 col3" >-0.025300</td>
          <td id="T_78b05_row632_col4" class="data row632 col4" >0.012500</td>
          <td id="T_78b05_row632_col5" class="data row632 col5" >0.022900</td>
          <td id="T_78b05_row632_col6" class="data row632 col6" >0.029600</td>
          <td id="T_78b05_row632_col7" class="data row632 col7" >-0.028300</td>
          <td id="T_78b05_row632_col8" class="data row632 col8" >0.014300</td>
          <td id="T_78b05_row632_col9" class="data row632 col9" >0.100700</td>
          <td id="T_78b05_row632_col10" class="data row632 col10" >0.056100</td>
          <td id="T_78b05_row632_col11" class="data row632 col11" >0.030400</td>
          <td id="T_78b05_row632_col12" class="data row632 col12" >0.023600</td>
          <td id="T_78b05_row632_col13" class="data row632 col13" >0.033500</td>
          <td id="T_78b05_row632_col14" class="data row632 col14" >0.026600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row633" class="row_heading level0 row633" >634</th>
          <td id="T_78b05_row633_col0" class="data row633 col0" >PC1</td>
          <td id="T_78b05_row633_col1" class="data row633 col1" >0.026000</td>
          <td id="T_78b05_row633_col2" class="data row633 col2" >0.042200</td>
          <td id="T_78b05_row633_col3" class="data row633 col3" >-0.007900</td>
          <td id="T_78b05_row633_col4" class="data row633 col4" >0.043000</td>
          <td id="T_78b05_row633_col5" class="data row633 col5" >-0.026800</td>
          <td id="T_78b05_row633_col6" class="data row633 col6" >0.027700</td>
          <td id="T_78b05_row633_col7" class="data row633 col7" >-0.060700</td>
          <td id="T_78b05_row633_col8" class="data row633 col8" >0.019500</td>
          <td id="T_78b05_row633_col9" class="data row633 col9" >0.072100</td>
          <td id="T_78b05_row633_col10" class="data row633 col10" >0.038600</td>
          <td id="T_78b05_row633_col11" class="data row633 col11" >0.060900</td>
          <td id="T_78b05_row633_col12" class="data row633 col12" >0.026200</td>
          <td id="T_78b05_row633_col13" class="data row633 col13" >0.031600</td>
          <td id="T_78b05_row633_col14" class="data row633 col14" >0.059000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row634" class="row_heading level0 row634" >635</th>
          <td id="T_78b05_row634_col0" class="data row634 col0" >PC1</td>
          <td id="T_78b05_row634_col1" class="data row634 col1" >0.025400</td>
          <td id="T_78b05_row634_col2" class="data row634 col2" >-0.040600</td>
          <td id="T_78b05_row634_col3" class="data row634 col3" >-0.054600</td>
          <td id="T_78b05_row634_col4" class="data row634 col4" >0.016100</td>
          <td id="T_78b05_row634_col5" class="data row634 col5" >-0.045800</td>
          <td id="T_78b05_row634_col6" class="data row634 col6" >-0.018200</td>
          <td id="T_78b05_row634_col7" class="data row634 col7" >0.017200</td>
          <td id="T_78b05_row634_col8" class="data row634 col8" >0.020100</td>
          <td id="T_78b05_row634_col9" class="data row634 col9" >0.010800</td>
          <td id="T_78b05_row634_col10" class="data row634 col10" >0.085300</td>
          <td id="T_78b05_row634_col11" class="data row634 col11" >0.034000</td>
          <td id="T_78b05_row634_col12" class="data row634 col12" >0.045200</td>
          <td id="T_78b05_row634_col13" class="data row634 col13" >0.014300</td>
          <td id="T_78b05_row634_col14" class="data row634 col14" >0.019000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row635" class="row_heading level0 row635" >636</th>
          <td id="T_78b05_row635_col0" class="data row635 col0" >None</td>
          <td id="T_78b05_row635_col1" class="data row635 col1" >0.027500</td>
          <td id="T_78b05_row635_col2" class="data row635 col2" >0.052700</td>
          <td id="T_78b05_row635_col3" class="data row635 col3" >-0.039600</td>
          <td id="T_78b05_row635_col4" class="data row635 col4" >-0.105400</td>
          <td id="T_78b05_row635_col5" class="data row635 col5" >-0.049600</td>
          <td id="T_78b05_row635_col6" class="data row635 col6" >0.037200</td>
          <td id="T_78b05_row635_col7" class="data row635 col7" >0.017500</td>
          <td id="T_78b05_row635_col8" class="data row635 col8" >0.018000</td>
          <td id="T_78b05_row635_col9" class="data row635 col9" >0.082600</td>
          <td id="T_78b05_row635_col10" class="data row635 col10" >0.070400</td>
          <td id="T_78b05_row635_col11" class="data row635 col11" >0.087500</td>
          <td id="T_78b05_row635_col12" class="data row635 col12" >0.048900</td>
          <td id="T_78b05_row635_col13" class="data row635 col13" >0.041100</td>
          <td id="T_78b05_row635_col14" class="data row635 col14" >0.019200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row636" class="row_heading level0 row636" >637</th>
          <td id="T_78b05_row636_col0" class="data row636 col0" >PC1</td>
          <td id="T_78b05_row636_col1" class="data row636 col1" >0.022600</td>
          <td id="T_78b05_row636_col2" class="data row636 col2" >0.057800</td>
          <td id="T_78b05_row636_col3" class="data row636 col3" >-0.044500</td>
          <td id="T_78b05_row636_col4" class="data row636 col4" >-0.033500</td>
          <td id="T_78b05_row636_col5" class="data row636 col5" >-0.014900</td>
          <td id="T_78b05_row636_col6" class="data row636 col6" >-0.008700</td>
          <td id="T_78b05_row636_col7" class="data row636 col7" >-0.028900</td>
          <td id="T_78b05_row636_col8" class="data row636 col8" >0.022900</td>
          <td id="T_78b05_row636_col9" class="data row636 col9" >0.087600</td>
          <td id="T_78b05_row636_col10" class="data row636 col10" >0.075300</td>
          <td id="T_78b05_row636_col11" class="data row636 col11" >0.015600</td>
          <td id="T_78b05_row636_col12" class="data row636 col12" >0.014200</td>
          <td id="T_78b05_row636_col13" class="data row636 col13" >0.004800</td>
          <td id="T_78b05_row636_col14" class="data row636 col14" >0.027200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row637" class="row_heading level0 row637" >638</th>
          <td id="T_78b05_row637_col0" class="data row637 col0" >None</td>
          <td id="T_78b05_row637_col1" class="data row637 col1" >0.031500</td>
          <td id="T_78b05_row637_col2" class="data row637 col2" >0.046800</td>
          <td id="T_78b05_row637_col3" class="data row637 col3" >-0.019300</td>
          <td id="T_78b05_row637_col4" class="data row637 col4" >0.013700</td>
          <td id="T_78b05_row637_col5" class="data row637 col5" >0.023400</td>
          <td id="T_78b05_row637_col6" class="data row637 col6" >0.025800</td>
          <td id="T_78b05_row637_col7" class="data row637 col7" >-0.005000</td>
          <td id="T_78b05_row637_col8" class="data row637 col8" >0.014100</td>
          <td id="T_78b05_row637_col9" class="data row637 col9" >0.076700</td>
          <td id="T_78b05_row637_col10" class="data row637 col10" >0.050100</td>
          <td id="T_78b05_row637_col11" class="data row637 col11" >0.031600</td>
          <td id="T_78b05_row637_col12" class="data row637 col12" >0.024000</td>
          <td id="T_78b05_row637_col13" class="data row637 col13" >0.029700</td>
          <td id="T_78b05_row637_col14" class="data row637 col14" >0.003200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row638" class="row_heading level0 row638" >639</th>
          <td id="T_78b05_row638_col0" class="data row638 col0" >None</td>
          <td id="T_78b05_row638_col1" class="data row638 col1" >0.031400</td>
          <td id="T_78b05_row638_col2" class="data row638 col2" >0.075600</td>
          <td id="T_78b05_row638_col3" class="data row638 col3" >0.012400</td>
          <td id="T_78b05_row638_col4" class="data row638 col4" >0.040900</td>
          <td id="T_78b05_row638_col5" class="data row638 col5" >0.017000</td>
          <td id="T_78b05_row638_col6" class="data row638 col6" >-0.054200</td>
          <td id="T_78b05_row638_col7" class="data row638 col7" >-0.048300</td>
          <td id="T_78b05_row638_col8" class="data row638 col8" >0.014100</td>
          <td id="T_78b05_row638_col9" class="data row638 col9" >0.105400</td>
          <td id="T_78b05_row638_col10" class="data row638 col10" >0.018400</td>
          <td id="T_78b05_row638_col11" class="data row638 col11" >0.058800</td>
          <td id="T_78b05_row638_col12" class="data row638 col12" >0.017700</td>
          <td id="T_78b05_row638_col13" class="data row638 col13" >0.050300</td>
          <td id="T_78b05_row638_col14" class="data row638 col14" >0.046600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row639" class="row_heading level0 row639" >640</th>
          <td id="T_78b05_row639_col0" class="data row639 col0" >None</td>
          <td id="T_78b05_row639_col1" class="data row639 col1" >0.035100</td>
          <td id="T_78b05_row639_col2" class="data row639 col2" >0.036900</td>
          <td id="T_78b05_row639_col3" class="data row639 col3" >0.011400</td>
          <td id="T_78b05_row639_col4" class="data row639 col4" >0.033200</td>
          <td id="T_78b05_row639_col5" class="data row639 col5" >0.016900</td>
          <td id="T_78b05_row639_col6" class="data row639 col6" >-0.052700</td>
          <td id="T_78b05_row639_col7" class="data row639 col7" >-0.030700</td>
          <td id="T_78b05_row639_col8" class="data row639 col8" >0.010400</td>
          <td id="T_78b05_row639_col9" class="data row639 col9" >0.066700</td>
          <td id="T_78b05_row639_col10" class="data row639 col10" >0.019300</td>
          <td id="T_78b05_row639_col11" class="data row639 col11" >0.051100</td>
          <td id="T_78b05_row639_col12" class="data row639 col12" >0.017500</td>
          <td id="T_78b05_row639_col13" class="data row639 col13" >0.048800</td>
          <td id="T_78b05_row639_col14" class="data row639 col14" >0.028900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row640" class="row_heading level0 row640" >641</th>
          <td id="T_78b05_row640_col0" class="data row640 col0" >PC5</td>
          <td id="T_78b05_row640_col1" class="data row640 col1" >0.043500</td>
          <td id="T_78b05_row640_col2" class="data row640 col2" >-0.006500</td>
          <td id="T_78b05_row640_col3" class="data row640 col3" >-0.015200</td>
          <td id="T_78b05_row640_col4" class="data row640 col4" >0.021300</td>
          <td id="T_78b05_row640_col5" class="data row640 col5" >0.103100</td>
          <td id="T_78b05_row640_col6" class="data row640 col6" >-0.037000</td>
          <td id="T_78b05_row640_col7" class="data row640 col7" >-0.015500</td>
          <td id="T_78b05_row640_col8" class="data row640 col8" >0.002100</td>
          <td id="T_78b05_row640_col9" class="data row640 col9" >0.023400</td>
          <td id="T_78b05_row640_col10" class="data row640 col10" >0.045900</td>
          <td id="T_78b05_row640_col11" class="data row640 col11" >0.039200</td>
          <td id="T_78b05_row640_col12" class="data row640 col12" >0.103700</td>
          <td id="T_78b05_row640_col13" class="data row640 col13" >0.033100</td>
          <td id="T_78b05_row640_col14" class="data row640 col14" >0.013800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row641" class="row_heading level0 row641" >642</th>
          <td id="T_78b05_row641_col0" class="data row641 col0" >None</td>
          <td id="T_78b05_row641_col1" class="data row641 col1" >0.041500</td>
          <td id="T_78b05_row641_col2" class="data row641 col2" >-0.005000</td>
          <td id="T_78b05_row641_col3" class="data row641 col3" >-0.015000</td>
          <td id="T_78b05_row641_col4" class="data row641 col4" >0.054600</td>
          <td id="T_78b05_row641_col5" class="data row641 col5" >0.004000</td>
          <td id="T_78b05_row641_col6" class="data row641 col6" >0.076500</td>
          <td id="T_78b05_row641_col7" class="data row641 col7" >0.026900</td>
          <td id="T_78b05_row641_col8" class="data row641 col8" >0.004000</td>
          <td id="T_78b05_row641_col9" class="data row641 col9" >0.024900</td>
          <td id="T_78b05_row641_col10" class="data row641 col10" >0.045700</td>
          <td id="T_78b05_row641_col11" class="data row641 col11" >0.072500</td>
          <td id="T_78b05_row641_col12" class="data row641 col12" >0.004700</td>
          <td id="T_78b05_row641_col13" class="data row641 col13" >0.080400</td>
          <td id="T_78b05_row641_col14" class="data row641 col14" >0.028700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row642" class="row_heading level0 row642" >643</th>
          <td id="T_78b05_row642_col0" class="data row642 col0" >None</td>
          <td id="T_78b05_row642_col1" class="data row642 col1" >0.031900</td>
          <td id="T_78b05_row642_col2" class="data row642 col2" >-0.002000</td>
          <td id="T_78b05_row642_col3" class="data row642 col3" >-0.049200</td>
          <td id="T_78b05_row642_col4" class="data row642 col4" >-0.031200</td>
          <td id="T_78b05_row642_col5" class="data row642 col5" >-0.011700</td>
          <td id="T_78b05_row642_col6" class="data row642 col6" >0.002600</td>
          <td id="T_78b05_row642_col7" class="data row642 col7" >-0.082700</td>
          <td id="T_78b05_row642_col8" class="data row642 col8" >0.013600</td>
          <td id="T_78b05_row642_col9" class="data row642 col9" >0.027900</td>
          <td id="T_78b05_row642_col10" class="data row642 col10" >0.080000</td>
          <td id="T_78b05_row642_col11" class="data row642 col11" >0.013300</td>
          <td id="T_78b05_row642_col12" class="data row642 col12" >0.011100</td>
          <td id="T_78b05_row642_col13" class="data row642 col13" >0.006500</td>
          <td id="T_78b05_row642_col14" class="data row642 col14" >0.081000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row643" class="row_heading level0 row643" >644</th>
          <td id="T_78b05_row643_col0" class="data row643 col0" >None</td>
          <td id="T_78b05_row643_col1" class="data row643 col1" >0.033000</td>
          <td id="T_78b05_row643_col2" class="data row643 col2" >-0.015500</td>
          <td id="T_78b05_row643_col3" class="data row643 col3" >-0.005500</td>
          <td id="T_78b05_row643_col4" class="data row643 col4" >-0.013100</td>
          <td id="T_78b05_row643_col5" class="data row643 col5" >-0.050400</td>
          <td id="T_78b05_row643_col6" class="data row643 col6" >0.037800</td>
          <td id="T_78b05_row643_col7" class="data row643 col7" >-0.113100</td>
          <td id="T_78b05_row643_col8" class="data row643 col8" >0.012600</td>
          <td id="T_78b05_row643_col9" class="data row643 col9" >0.014300</td>
          <td id="T_78b05_row643_col10" class="data row643 col10" >0.036200</td>
          <td id="T_78b05_row643_col11" class="data row643 col11" >0.004800</td>
          <td id="T_78b05_row643_col12" class="data row643 col12" >0.049700</td>
          <td id="T_78b05_row643_col13" class="data row643 col13" >0.041700</td>
          <td id="T_78b05_row643_col14" class="data row643 col14" >0.111400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row644" class="row_heading level0 row644" >645</th>
          <td id="T_78b05_row644_col0" class="data row644 col0" >None</td>
          <td id="T_78b05_row644_col1" class="data row644 col1" >0.027100</td>
          <td id="T_78b05_row644_col2" class="data row644 col2" >0.051600</td>
          <td id="T_78b05_row644_col3" class="data row644 col3" >-0.063100</td>
          <td id="T_78b05_row644_col4" class="data row644 col4" >-0.038700</td>
          <td id="T_78b05_row644_col5" class="data row644 col5" >-0.006300</td>
          <td id="T_78b05_row644_col6" class="data row644 col6" >0.021000</td>
          <td id="T_78b05_row644_col7" class="data row644 col7" >0.015400</td>
          <td id="T_78b05_row644_col8" class="data row644 col8" >0.018400</td>
          <td id="T_78b05_row644_col9" class="data row644 col9" >0.081500</td>
          <td id="T_78b05_row644_col10" class="data row644 col10" >0.093800</td>
          <td id="T_78b05_row644_col11" class="data row644 col11" >0.020800</td>
          <td id="T_78b05_row644_col12" class="data row644 col12" >0.005600</td>
          <td id="T_78b05_row644_col13" class="data row644 col13" >0.024900</td>
          <td id="T_78b05_row644_col14" class="data row644 col14" >0.017100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row645" class="row_heading level0 row645" >646</th>
          <td id="T_78b05_row645_col0" class="data row645 col0" >None</td>
          <td id="T_78b05_row645_col1" class="data row645 col1" >0.032300</td>
          <td id="T_78b05_row645_col2" class="data row645 col2" >0.071200</td>
          <td id="T_78b05_row645_col3" class="data row645 col3" >0.021800</td>
          <td id="T_78b05_row645_col4" class="data row645 col4" >-0.082400</td>
          <td id="T_78b05_row645_col5" class="data row645 col5" >0.026600</td>
          <td id="T_78b05_row645_col6" class="data row645 col6" >0.032400</td>
          <td id="T_78b05_row645_col7" class="data row645 col7" >-0.069800</td>
          <td id="T_78b05_row645_col8" class="data row645 col8" >0.013200</td>
          <td id="T_78b05_row645_col9" class="data row645 col9" >0.101000</td>
          <td id="T_78b05_row645_col10" class="data row645 col10" >0.009000</td>
          <td id="T_78b05_row645_col11" class="data row645 col11" >0.064500</td>
          <td id="T_78b05_row645_col12" class="data row645 col12" >0.027300</td>
          <td id="T_78b05_row645_col13" class="data row645 col13" >0.036300</td>
          <td id="T_78b05_row645_col14" class="data row645 col14" >0.068100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row646" class="row_heading level0 row646" >647</th>
          <td id="T_78b05_row646_col0" class="data row646 col0" >None</td>
          <td id="T_78b05_row646_col1" class="data row646 col1" >0.028500</td>
          <td id="T_78b05_row646_col2" class="data row646 col2" >-0.000500</td>
          <td id="T_78b05_row646_col3" class="data row646 col3" >-0.035000</td>
          <td id="T_78b05_row646_col4" class="data row646 col4" >0.044800</td>
          <td id="T_78b05_row646_col5" class="data row646 col5" >0.012900</td>
          <td id="T_78b05_row646_col6" class="data row646 col6" >0.000200</td>
          <td id="T_78b05_row646_col7" class="data row646 col7" >-0.017200</td>
          <td id="T_78b05_row646_col8" class="data row646 col8" >0.017000</td>
          <td id="T_78b05_row646_col9" class="data row646 col9" >0.029300</td>
          <td id="T_78b05_row646_col10" class="data row646 col10" >0.065700</td>
          <td id="T_78b05_row646_col11" class="data row646 col11" >0.062700</td>
          <td id="T_78b05_row646_col12" class="data row646 col12" >0.013600</td>
          <td id="T_78b05_row646_col13" class="data row646 col13" >0.004100</td>
          <td id="T_78b05_row646_col14" class="data row646 col14" >0.015400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row647" class="row_heading level0 row647" >648</th>
          <td id="T_78b05_row647_col0" class="data row647 col0" >None</td>
          <td id="T_78b05_row647_col1" class="data row647 col1" >0.036400</td>
          <td id="T_78b05_row647_col2" class="data row647 col2" >0.062600</td>
          <td id="T_78b05_row647_col3" class="data row647 col3" >0.029900</td>
          <td id="T_78b05_row647_col4" class="data row647 col4" >0.063300</td>
          <td id="T_78b05_row647_col5" class="data row647 col5" >-0.055500</td>
          <td id="T_78b05_row647_col6" class="data row647 col6" >-0.007000</td>
          <td id="T_78b05_row647_col7" class="data row647 col7" >-0.005300</td>
          <td id="T_78b05_row647_col8" class="data row647 col8" >0.009100</td>
          <td id="T_78b05_row647_col9" class="data row647 col9" >0.092400</td>
          <td id="T_78b05_row647_col10" class="data row647 col10" >0.000900</td>
          <td id="T_78b05_row647_col11" class="data row647 col11" >0.081200</td>
          <td id="T_78b05_row647_col12" class="data row647 col12" >0.054800</td>
          <td id="T_78b05_row647_col13" class="data row647 col13" >0.003100</td>
          <td id="T_78b05_row647_col14" class="data row647 col14" >0.003500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row648" class="row_heading level0 row648" >649</th>
          <td id="T_78b05_row648_col0" class="data row648 col0" >PC1</td>
          <td id="T_78b05_row648_col1" class="data row648 col1" >0.022800</td>
          <td id="T_78b05_row648_col2" class="data row648 col2" >0.032400</td>
          <td id="T_78b05_row648_col3" class="data row648 col3" >-0.108000</td>
          <td id="T_78b05_row648_col4" class="data row648 col4" >-0.012300</td>
          <td id="T_78b05_row648_col5" class="data row648 col5" >0.008800</td>
          <td id="T_78b05_row648_col6" class="data row648 col6" >0.049900</td>
          <td id="T_78b05_row648_col7" class="data row648 col7" >0.033600</td>
          <td id="T_78b05_row648_col8" class="data row648 col8" >0.022800</td>
          <td id="T_78b05_row648_col9" class="data row648 col9" >0.062300</td>
          <td id="T_78b05_row648_col10" class="data row648 col10" >0.138800</td>
          <td id="T_78b05_row648_col11" class="data row648 col11" >0.005600</td>
          <td id="T_78b05_row648_col12" class="data row648 col12" >0.009400</td>
          <td id="T_78b05_row648_col13" class="data row648 col13" >0.053800</td>
          <td id="T_78b05_row648_col14" class="data row648 col14" >0.035300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row649" class="row_heading level0 row649" >650</th>
          <td id="T_78b05_row649_col0" class="data row649 col0" >None</td>
          <td id="T_78b05_row649_col1" class="data row649 col1" >0.046900</td>
          <td id="T_78b05_row649_col2" class="data row649 col2" >0.020800</td>
          <td id="T_78b05_row649_col3" class="data row649 col3" >0.055600</td>
          <td id="T_78b05_row649_col4" class="data row649 col4" >-0.029600</td>
          <td id="T_78b05_row649_col5" class="data row649 col5" >-0.006400</td>
          <td id="T_78b05_row649_col6" class="data row649 col6" >0.019100</td>
          <td id="T_78b05_row649_col7" class="data row649 col7" >0.055700</td>
          <td id="T_78b05_row649_col8" class="data row649 col8" >0.001400</td>
          <td id="T_78b05_row649_col9" class="data row649 col9" >0.050600</td>
          <td id="T_78b05_row649_col10" class="data row649 col10" >0.024900</td>
          <td id="T_78b05_row649_col11" class="data row649 col11" >0.011700</td>
          <td id="T_78b05_row649_col12" class="data row649 col12" >0.005700</td>
          <td id="T_78b05_row649_col13" class="data row649 col13" >0.023000</td>
          <td id="T_78b05_row649_col14" class="data row649 col14" >0.057500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row650" class="row_heading level0 row650" >651</th>
          <td id="T_78b05_row650_col0" class="data row650 col0" >None</td>
          <td id="T_78b05_row650_col1" class="data row650 col1" >0.041700</td>
          <td id="T_78b05_row650_col2" class="data row650 col2" >-0.058200</td>
          <td id="T_78b05_row650_col3" class="data row650 col3" >-0.047600</td>
          <td id="T_78b05_row650_col4" class="data row650 col4" >-0.011500</td>
          <td id="T_78b05_row650_col5" class="data row650 col5" >0.100300</td>
          <td id="T_78b05_row650_col6" class="data row650 col6" >0.040900</td>
          <td id="T_78b05_row650_col7" class="data row650 col7" >0.021700</td>
          <td id="T_78b05_row650_col8" class="data row650 col8" >0.003800</td>
          <td id="T_78b05_row650_col9" class="data row650 col9" >0.028400</td>
          <td id="T_78b05_row650_col10" class="data row650 col10" >0.078300</td>
          <td id="T_78b05_row650_col11" class="data row650 col11" >0.006400</td>
          <td id="T_78b05_row650_col12" class="data row650 col12" >0.100900</td>
          <td id="T_78b05_row650_col13" class="data row650 col13" >0.044800</td>
          <td id="T_78b05_row650_col14" class="data row650 col14" >0.023400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row651" class="row_heading level0 row651" >652</th>
          <td id="T_78b05_row651_col0" class="data row651 col0" >None</td>
          <td id="T_78b05_row651_col1" class="data row651 col1" >0.035000</td>
          <td id="T_78b05_row651_col2" class="data row651 col2" >-0.019900</td>
          <td id="T_78b05_row651_col3" class="data row651 col3" >-0.018800</td>
          <td id="T_78b05_row651_col4" class="data row651 col4" >0.024200</td>
          <td id="T_78b05_row651_col5" class="data row651 col5" >-0.025300</td>
          <td id="T_78b05_row651_col6" class="data row651 col6" >0.023900</td>
          <td id="T_78b05_row651_col7" class="data row651 col7" >-0.002400</td>
          <td id="T_78b05_row651_col8" class="data row651 col8" >0.010600</td>
          <td id="T_78b05_row651_col9" class="data row651 col9" >0.010000</td>
          <td id="T_78b05_row651_col10" class="data row651 col10" >0.049600</td>
          <td id="T_78b05_row651_col11" class="data row651 col11" >0.042200</td>
          <td id="T_78b05_row651_col12" class="data row651 col12" >0.024700</td>
          <td id="T_78b05_row651_col13" class="data row651 col13" >0.027800</td>
          <td id="T_78b05_row651_col14" class="data row651 col14" >0.000600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row652" class="row_heading level0 row652" >653</th>
          <td id="T_78b05_row652_col0" class="data row652 col0" >None</td>
          <td id="T_78b05_row652_col1" class="data row652 col1" >0.033600</td>
          <td id="T_78b05_row652_col2" class="data row652 col2" >-0.009300</td>
          <td id="T_78b05_row652_col3" class="data row652 col3" >-0.083600</td>
          <td id="T_78b05_row652_col4" class="data row652 col4" >0.017600</td>
          <td id="T_78b05_row652_col5" class="data row652 col5" >0.009900</td>
          <td id="T_78b05_row652_col6" class="data row652 col6" >0.027900</td>
          <td id="T_78b05_row652_col7" class="data row652 col7" >0.077700</td>
          <td id="T_78b05_row652_col8" class="data row652 col8" >0.011900</td>
          <td id="T_78b05_row652_col9" class="data row652 col9" >0.020500</td>
          <td id="T_78b05_row652_col10" class="data row652 col10" >0.114300</td>
          <td id="T_78b05_row652_col11" class="data row652 col11" >0.035500</td>
          <td id="T_78b05_row652_col12" class="data row652 col12" >0.010500</td>
          <td id="T_78b05_row652_col13" class="data row652 col13" >0.031800</td>
          <td id="T_78b05_row652_col14" class="data row652 col14" >0.079500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row653" class="row_heading level0 row653" >654</th>
          <td id="T_78b05_row653_col0" class="data row653 col0" >None</td>
          <td id="T_78b05_row653_col1" class="data row653 col1" >0.038900</td>
          <td id="T_78b05_row653_col2" class="data row653 col2" >-0.031100</td>
          <td id="T_78b05_row653_col3" class="data row653 col3" >0.027700</td>
          <td id="T_78b05_row653_col4" class="data row653 col4" >0.009500</td>
          <td id="T_78b05_row653_col5" class="data row653 col5" >-0.042000</td>
          <td id="T_78b05_row653_col6" class="data row653 col6" >-0.077800</td>
          <td id="T_78b05_row653_col7" class="data row653 col7" >0.025000</td>
          <td id="T_78b05_row653_col8" class="data row653 col8" >0.006600</td>
          <td id="T_78b05_row653_col9" class="data row653 col9" >0.001200</td>
          <td id="T_78b05_row653_col10" class="data row653 col10" >0.003100</td>
          <td id="T_78b05_row653_col11" class="data row653 col11" >0.027400</td>
          <td id="T_78b05_row653_col12" class="data row653 col12" >0.041300</td>
          <td id="T_78b05_row653_col13" class="data row653 col13" >0.073900</td>
          <td id="T_78b05_row653_col14" class="data row653 col14" >0.026700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row654" class="row_heading level0 row654" >655</th>
          <td id="T_78b05_row654_col0" class="data row654 col0" >None</td>
          <td id="T_78b05_row654_col1" class="data row654 col1" >0.043000</td>
          <td id="T_78b05_row654_col2" class="data row654 col2" >0.017700</td>
          <td id="T_78b05_row654_col3" class="data row654 col3" >0.021900</td>
          <td id="T_78b05_row654_col4" class="data row654 col4" >0.029700</td>
          <td id="T_78b05_row654_col5" class="data row654 col5" >0.057900</td>
          <td id="T_78b05_row654_col6" class="data row654 col6" >0.020000</td>
          <td id="T_78b05_row654_col7" class="data row654 col7" >-0.006400</td>
          <td id="T_78b05_row654_col8" class="data row654 col8" >0.002500</td>
          <td id="T_78b05_row654_col9" class="data row654 col9" >0.047600</td>
          <td id="T_78b05_row654_col10" class="data row654 col10" >0.008900</td>
          <td id="T_78b05_row654_col11" class="data row654 col11" >0.047700</td>
          <td id="T_78b05_row654_col12" class="data row654 col12" >0.058600</td>
          <td id="T_78b05_row654_col13" class="data row654 col13" >0.023900</td>
          <td id="T_78b05_row654_col14" class="data row654 col14" >0.004600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row655" class="row_heading level0 row655" >656</th>
          <td id="T_78b05_row655_col0" class="data row655 col0" >None</td>
          <td id="T_78b05_row655_col1" class="data row655 col1" >0.039700</td>
          <td id="T_78b05_row655_col2" class="data row655 col2" >-0.033500</td>
          <td id="T_78b05_row655_col3" class="data row655 col3" >-0.033900</td>
          <td id="T_78b05_row655_col4" class="data row655 col4" >-0.001500</td>
          <td id="T_78b05_row655_col5" class="data row655 col5" >0.016100</td>
          <td id="T_78b05_row655_col6" class="data row655 col6" >-0.001200</td>
          <td id="T_78b05_row655_col7" class="data row655 col7" >0.067900</td>
          <td id="T_78b05_row655_col8" class="data row655 col8" >0.005800</td>
          <td id="T_78b05_row655_col9" class="data row655 col9" >0.003700</td>
          <td id="T_78b05_row655_col10" class="data row655 col10" >0.064700</td>
          <td id="T_78b05_row655_col11" class="data row655 col11" >0.016400</td>
          <td id="T_78b05_row655_col12" class="data row655 col12" >0.016700</td>
          <td id="T_78b05_row655_col13" class="data row655 col13" >0.002700</td>
          <td id="T_78b05_row655_col14" class="data row655 col14" >0.069600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row656" class="row_heading level0 row656" >657</th>
          <td id="T_78b05_row656_col0" class="data row656 col0" >None</td>
          <td id="T_78b05_row656_col1" class="data row656 col1" >0.036500</td>
          <td id="T_78b05_row656_col2" class="data row656 col2" >0.044000</td>
          <td id="T_78b05_row656_col3" class="data row656 col3" >-0.028000</td>
          <td id="T_78b05_row656_col4" class="data row656 col4" >-0.018500</td>
          <td id="T_78b05_row656_col5" class="data row656 col5" >0.031800</td>
          <td id="T_78b05_row656_col6" class="data row656 col6" >-0.034200</td>
          <td id="T_78b05_row656_col7" class="data row656 col7" >-0.020400</td>
          <td id="T_78b05_row656_col8" class="data row656 col8" >0.009100</td>
          <td id="T_78b05_row656_col9" class="data row656 col9" >0.073800</td>
          <td id="T_78b05_row656_col10" class="data row656 col10" >0.058700</td>
          <td id="T_78b05_row656_col11" class="data row656 col11" >0.000600</td>
          <td id="T_78b05_row656_col12" class="data row656 col12" >0.032500</td>
          <td id="T_78b05_row656_col13" class="data row656 col13" >0.030300</td>
          <td id="T_78b05_row656_col14" class="data row656 col14" >0.018600</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row657" class="row_heading level0 row657" >658</th>
          <td id="T_78b05_row657_col0" class="data row657 col0" >PC2</td>
          <td id="T_78b05_row657_col1" class="data row657 col1" >0.035200</td>
          <td id="T_78b05_row657_col2" class="data row657 col2" >0.081100</td>
          <td id="T_78b05_row657_col3" class="data row657 col3" >0.040700</td>
          <td id="T_78b05_row657_col4" class="data row657 col4" >-0.055900</td>
          <td id="T_78b05_row657_col5" class="data row657 col5" >0.030100</td>
          <td id="T_78b05_row657_col6" class="data row657 col6" >-0.071100</td>
          <td id="T_78b05_row657_col7" class="data row657 col7" >0.000400</td>
          <td id="T_78b05_row657_col8" class="data row657 col8" >0.010300</td>
          <td id="T_78b05_row657_col9" class="data row657 col9" >0.111000</td>
          <td id="T_78b05_row657_col10" class="data row657 col10" >0.010000</td>
          <td id="T_78b05_row657_col11" class="data row657 col11" >0.038000</td>
          <td id="T_78b05_row657_col12" class="data row657 col12" >0.030800</td>
          <td id="T_78b05_row657_col13" class="data row657 col13" >0.067200</td>
          <td id="T_78b05_row657_col14" class="data row657 col14" >0.002200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row658" class="row_heading level0 row658" >659</th>
          <td id="T_78b05_row658_col0" class="data row658 col0" >None</td>
          <td id="T_78b05_row658_col1" class="data row658 col1" >0.040200</td>
          <td id="T_78b05_row658_col2" class="data row658 col2" >0.024500</td>
          <td id="T_78b05_row658_col3" class="data row658 col3" >0.008800</td>
          <td id="T_78b05_row658_col4" class="data row658 col4" >-0.018100</td>
          <td id="T_78b05_row658_col5" class="data row658 col5" >-0.020900</td>
          <td id="T_78b05_row658_col6" class="data row658 col6" >-0.053300</td>
          <td id="T_78b05_row658_col7" class="data row658 col7" >-0.019600</td>
          <td id="T_78b05_row658_col8" class="data row658 col8" >0.005300</td>
          <td id="T_78b05_row658_col9" class="data row658 col9" >0.054300</td>
          <td id="T_78b05_row658_col10" class="data row658 col10" >0.021900</td>
          <td id="T_78b05_row658_col11" class="data row658 col11" >0.000200</td>
          <td id="T_78b05_row658_col12" class="data row658 col12" >0.020200</td>
          <td id="T_78b05_row658_col13" class="data row658 col13" >0.049400</td>
          <td id="T_78b05_row658_col14" class="data row658 col14" >0.017900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row659" class="row_heading level0 row659" >660</th>
          <td id="T_78b05_row659_col0" class="data row659 col0" >None</td>
          <td id="T_78b05_row659_col1" class="data row659 col1" >0.039100</td>
          <td id="T_78b05_row659_col2" class="data row659 col2" >-0.019400</td>
          <td id="T_78b05_row659_col3" class="data row659 col3" >-0.018200</td>
          <td id="T_78b05_row659_col4" class="data row659 col4" >0.055400</td>
          <td id="T_78b05_row659_col5" class="data row659 col5" >0.023700</td>
          <td id="T_78b05_row659_col6" class="data row659 col6" >0.029300</td>
          <td id="T_78b05_row659_col7" class="data row659 col7" >-0.053900</td>
          <td id="T_78b05_row659_col8" class="data row659 col8" >0.006500</td>
          <td id="T_78b05_row659_col9" class="data row659 col9" >0.010400</td>
          <td id="T_78b05_row659_col10" class="data row659 col10" >0.048900</td>
          <td id="T_78b05_row659_col11" class="data row659 col11" >0.073300</td>
          <td id="T_78b05_row659_col12" class="data row659 col12" >0.024300</td>
          <td id="T_78b05_row659_col13" class="data row659 col13" >0.033100</td>
          <td id="T_78b05_row659_col14" class="data row659 col14" >0.052200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row660" class="row_heading level0 row660" >661</th>
          <td id="T_78b05_row660_col0" class="data row660 col0" >None</td>
          <td id="T_78b05_row660_col1" class="data row660 col1" >0.029400</td>
          <td id="T_78b05_row660_col2" class="data row660 col2" >0.045000</td>
          <td id="T_78b05_row660_col3" class="data row660 col3" >-0.037600</td>
          <td id="T_78b05_row660_col4" class="data row660 col4" >-0.050800</td>
          <td id="T_78b05_row660_col5" class="data row660 col5" >-0.001800</td>
          <td id="T_78b05_row660_col6" class="data row660 col6" >0.007100</td>
          <td id="T_78b05_row660_col7" class="data row660 col7" >0.068400</td>
          <td id="T_78b05_row660_col8" class="data row660 col8" >0.016200</td>
          <td id="T_78b05_row660_col9" class="data row660 col9" >0.074800</td>
          <td id="T_78b05_row660_col10" class="data row660 col10" >0.068300</td>
          <td id="T_78b05_row660_col11" class="data row660 col11" >0.032900</td>
          <td id="T_78b05_row660_col12" class="data row660 col12" >0.001100</td>
          <td id="T_78b05_row660_col13" class="data row660 col13" >0.011000</td>
          <td id="T_78b05_row660_col14" class="data row660 col14" >0.070200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row661" class="row_heading level0 row661" >662</th>
          <td id="T_78b05_row661_col0" class="data row661 col0" >None</td>
          <td id="T_78b05_row661_col1" class="data row661 col1" >0.045900</td>
          <td id="T_78b05_row661_col2" class="data row661 col2" >-0.005400</td>
          <td id="T_78b05_row661_col3" class="data row661 col3" >-0.001900</td>
          <td id="T_78b05_row661_col4" class="data row661 col4" >0.031400</td>
          <td id="T_78b05_row661_col5" class="data row661 col5" >0.022100</td>
          <td id="T_78b05_row661_col6" class="data row661 col6" >-0.013000</td>
          <td id="T_78b05_row661_col7" class="data row661 col7" >-0.008600</td>
          <td id="T_78b05_row661_col8" class="data row661 col8" >0.000300</td>
          <td id="T_78b05_row661_col9" class="data row661 col9" >0.024500</td>
          <td id="T_78b05_row661_col10" class="data row661 col10" >0.032700</td>
          <td id="T_78b05_row661_col11" class="data row661 col11" >0.049300</td>
          <td id="T_78b05_row661_col12" class="data row661 col12" >0.022700</td>
          <td id="T_78b05_row661_col13" class="data row661 col13" >0.009100</td>
          <td id="T_78b05_row661_col14" class="data row661 col14" >0.006800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row662" class="row_heading level0 row662" >663</th>
          <td id="T_78b05_row662_col0" class="data row662 col0" >None</td>
          <td id="T_78b05_row662_col1" class="data row662 col1" >0.043000</td>
          <td id="T_78b05_row662_col2" class="data row662 col2" >-0.041200</td>
          <td id="T_78b05_row662_col3" class="data row662 col3" >-0.016000</td>
          <td id="T_78b05_row662_col4" class="data row662 col4" >0.035200</td>
          <td id="T_78b05_row662_col5" class="data row662 col5" >-0.026200</td>
          <td id="T_78b05_row662_col6" class="data row662 col6" >0.013000</td>
          <td id="T_78b05_row662_col7" class="data row662 col7" >0.017700</td>
          <td id="T_78b05_row662_col8" class="data row662 col8" >0.002600</td>
          <td id="T_78b05_row662_col9" class="data row662 col9" >0.011400</td>
          <td id="T_78b05_row662_col10" class="data row662 col10" >0.046800</td>
          <td id="T_78b05_row662_col11" class="data row662 col11" >0.053100</td>
          <td id="T_78b05_row662_col12" class="data row662 col12" >0.025600</td>
          <td id="T_78b05_row662_col13" class="data row662 col13" >0.016900</td>
          <td id="T_78b05_row662_col14" class="data row662 col14" >0.019400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row663" class="row_heading level0 row663" >664</th>
          <td id="T_78b05_row663_col0" class="data row663 col0" >None</td>
          <td id="T_78b05_row663_col1" class="data row663 col1" >0.037300</td>
          <td id="T_78b05_row663_col2" class="data row663 col2" >0.032700</td>
          <td id="T_78b05_row663_col3" class="data row663 col3" >-0.022600</td>
          <td id="T_78b05_row663_col4" class="data row663 col4" >-0.024000</td>
          <td id="T_78b05_row663_col5" class="data row663 col5" >0.051100</td>
          <td id="T_78b05_row663_col6" class="data row663 col6" >-0.037400</td>
          <td id="T_78b05_row663_col7" class="data row663 col7" >0.057100</td>
          <td id="T_78b05_row663_col8" class="data row663 col8" >0.008300</td>
          <td id="T_78b05_row663_col9" class="data row663 col9" >0.062500</td>
          <td id="T_78b05_row663_col10" class="data row663 col10" >0.053400</td>
          <td id="T_78b05_row663_col11" class="data row663 col11" >0.006100</td>
          <td id="T_78b05_row663_col12" class="data row663 col12" >0.051700</td>
          <td id="T_78b05_row663_col13" class="data row663 col13" >0.033600</td>
          <td id="T_78b05_row663_col14" class="data row663 col14" >0.058900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row664" class="row_heading level0 row664" >665</th>
          <td id="T_78b05_row664_col0" class="data row664 col0" >None</td>
          <td id="T_78b05_row664_col1" class="data row664 col1" >0.033500</td>
          <td id="T_78b05_row664_col2" class="data row664 col2" >0.063000</td>
          <td id="T_78b05_row664_col3" class="data row664 col3" >0.020100</td>
          <td id="T_78b05_row664_col4" class="data row664 col4" >0.053800</td>
          <td id="T_78b05_row664_col5" class="data row664 col5" >0.014300</td>
          <td id="T_78b05_row664_col6" class="data row664 col6" >-0.047200</td>
          <td id="T_78b05_row664_col7" class="data row664 col7" >-0.030200</td>
          <td id="T_78b05_row664_col8" class="data row664 col8" >0.012000</td>
          <td id="T_78b05_row664_col9" class="data row664 col9" >0.092800</td>
          <td id="T_78b05_row664_col10" class="data row664 col10" >0.010600</td>
          <td id="T_78b05_row664_col11" class="data row664 col11" >0.071700</td>
          <td id="T_78b05_row664_col12" class="data row664 col12" >0.015000</td>
          <td id="T_78b05_row664_col13" class="data row664 col13" >0.043300</td>
          <td id="T_78b05_row664_col14" class="data row664 col14" >0.028500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row665" class="row_heading level0 row665" >666</th>
          <td id="T_78b05_row665_col0" class="data row665 col0" >PC7</td>
          <td id="T_78b05_row665_col1" class="data row665 col1" >0.035200</td>
          <td id="T_78b05_row665_col2" class="data row665 col2" >-0.075600</td>
          <td id="T_78b05_row665_col3" class="data row665 col3" >0.011600</td>
          <td id="T_78b05_row665_col4" class="data row665 col4" >-0.027000</td>
          <td id="T_78b05_row665_col5" class="data row665 col5" >-0.086300</td>
          <td id="T_78b05_row665_col6" class="data row665 col6" >0.024900</td>
          <td id="T_78b05_row665_col7" class="data row665 col7" >-0.137600</td>
          <td id="T_78b05_row665_col8" class="data row665 col8" >0.010400</td>
          <td id="T_78b05_row665_col9" class="data row665 col9" >0.045700</td>
          <td id="T_78b05_row665_col10" class="data row665 col10" >0.019200</td>
          <td id="T_78b05_row665_col11" class="data row665 col11" >0.009100</td>
          <td id="T_78b05_row665_col12" class="data row665 col12" >0.085600</td>
          <td id="T_78b05_row665_col13" class="data row665 col13" >0.028800</td>
          <td id="T_78b05_row665_col14" class="data row665 col14" >0.135800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row666" class="row_heading level0 row666" >667</th>
          <td id="T_78b05_row666_col0" class="data row666 col0" >None</td>
          <td id="T_78b05_row666_col1" class="data row666 col1" >0.036900</td>
          <td id="T_78b05_row666_col2" class="data row666 col2" >-0.058500</td>
          <td id="T_78b05_row666_col3" class="data row666 col3" >-0.057900</td>
          <td id="T_78b05_row666_col4" class="data row666 col4" >0.018600</td>
          <td id="T_78b05_row666_col5" class="data row666 col5" >-0.008600</td>
          <td id="T_78b05_row666_col6" class="data row666 col6" >0.028000</td>
          <td id="T_78b05_row666_col7" class="data row666 col7" >-0.026300</td>
          <td id="T_78b05_row666_col8" class="data row666 col8" >0.008600</td>
          <td id="T_78b05_row666_col9" class="data row666 col9" >0.028600</td>
          <td id="T_78b05_row666_col10" class="data row666 col10" >0.088600</td>
          <td id="T_78b05_row666_col11" class="data row666 col11" >0.036500</td>
          <td id="T_78b05_row666_col12" class="data row666 col12" >0.008000</td>
          <td id="T_78b05_row666_col13" class="data row666 col13" >0.031900</td>
          <td id="T_78b05_row666_col14" class="data row666 col14" >0.024500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row667" class="row_heading level0 row667" >668</th>
          <td id="T_78b05_row667_col0" class="data row667 col0" >PC1</td>
          <td id="T_78b05_row667_col1" class="data row667 col1" >0.023200</td>
          <td id="T_78b05_row667_col2" class="data row667 col2" >0.016900</td>
          <td id="T_78b05_row667_col3" class="data row667 col3" >-0.076500</td>
          <td id="T_78b05_row667_col4" class="data row667 col4" >0.057600</td>
          <td id="T_78b05_row667_col5" class="data row667 col5" >-0.023400</td>
          <td id="T_78b05_row667_col6" class="data row667 col6" >0.011100</td>
          <td id="T_78b05_row667_col7" class="data row667 col7" >0.041900</td>
          <td id="T_78b05_row667_col8" class="data row667 col8" >0.022400</td>
          <td id="T_78b05_row667_col9" class="data row667 col9" >0.046700</td>
          <td id="T_78b05_row667_col10" class="data row667 col10" >0.107200</td>
          <td id="T_78b05_row667_col11" class="data row667 col11" >0.075500</td>
          <td id="T_78b05_row667_col12" class="data row667 col12" >0.022700</td>
          <td id="T_78b05_row667_col13" class="data row667 col13" >0.015000</td>
          <td id="T_78b05_row667_col14" class="data row667 col14" >0.043700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row668" class="row_heading level0 row668" >669</th>
          <td id="T_78b05_row668_col0" class="data row668 col0" >None</td>
          <td id="T_78b05_row668_col1" class="data row668 col1" >0.031400</td>
          <td id="T_78b05_row668_col2" class="data row668 col2" >0.012000</td>
          <td id="T_78b05_row668_col3" class="data row668 col3" >-0.009600</td>
          <td id="T_78b05_row668_col4" class="data row668 col4" >0.034800</td>
          <td id="T_78b05_row668_col5" class="data row668 col5" >-0.040100</td>
          <td id="T_78b05_row668_col6" class="data row668 col6" >-0.036800</td>
          <td id="T_78b05_row668_col7" class="data row668 col7" >-0.010900</td>
          <td id="T_78b05_row668_col8" class="data row668 col8" >0.014100</td>
          <td id="T_78b05_row668_col9" class="data row668 col9" >0.041800</td>
          <td id="T_78b05_row668_col10" class="data row668 col10" >0.040300</td>
          <td id="T_78b05_row668_col11" class="data row668 col11" >0.052700</td>
          <td id="T_78b05_row668_col12" class="data row668 col12" >0.039400</td>
          <td id="T_78b05_row668_col13" class="data row668 col13" >0.032900</td>
          <td id="T_78b05_row668_col14" class="data row668 col14" >0.009200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row669" class="row_heading level0 row669" >670</th>
          <td id="T_78b05_row669_col0" class="data row669 col0" >None</td>
          <td id="T_78b05_row669_col1" class="data row669 col1" >0.039000</td>
          <td id="T_78b05_row669_col2" class="data row669 col2" >-0.044100</td>
          <td id="T_78b05_row669_col3" class="data row669 col3" >-0.010300</td>
          <td id="T_78b05_row669_col4" class="data row669 col4" >-0.031700</td>
          <td id="T_78b05_row669_col5" class="data row669 col5" >0.007000</td>
          <td id="T_78b05_row669_col6" class="data row669 col6" >0.013200</td>
          <td id="T_78b05_row669_col7" class="data row669 col7" >-0.014500</td>
          <td id="T_78b05_row669_col8" class="data row669 col8" >0.006500</td>
          <td id="T_78b05_row669_col9" class="data row669 col9" >0.014300</td>
          <td id="T_78b05_row669_col10" class="data row669 col10" >0.041000</td>
          <td id="T_78b05_row669_col11" class="data row669 col11" >0.013800</td>
          <td id="T_78b05_row669_col12" class="data row669 col12" >0.007700</td>
          <td id="T_78b05_row669_col13" class="data row669 col13" >0.017100</td>
          <td id="T_78b05_row669_col14" class="data row669 col14" >0.012700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row670" class="row_heading level0 row670" >671</th>
          <td id="T_78b05_row670_col0" class="data row670 col0" >None</td>
          <td id="T_78b05_row670_col1" class="data row670 col1" >0.035300</td>
          <td id="T_78b05_row670_col2" class="data row670 col2" >0.030000</td>
          <td id="T_78b05_row670_col3" class="data row670 col3" >-0.001100</td>
          <td id="T_78b05_row670_col4" class="data row670 col4" >-0.041500</td>
          <td id="T_78b05_row670_col5" class="data row670 col5" >-0.015200</td>
          <td id="T_78b05_row670_col6" class="data row670 col6" >-0.048500</td>
          <td id="T_78b05_row670_col7" class="data row670 col7" >0.017700</td>
          <td id="T_78b05_row670_col8" class="data row670 col8" >0.010200</td>
          <td id="T_78b05_row670_col9" class="data row670 col9" >0.059800</td>
          <td id="T_78b05_row670_col10" class="data row670 col10" >0.031800</td>
          <td id="T_78b05_row670_col11" class="data row670 col11" >0.023600</td>
          <td id="T_78b05_row670_col12" class="data row670 col12" >0.014600</td>
          <td id="T_78b05_row670_col13" class="data row670 col13" >0.044700</td>
          <td id="T_78b05_row670_col14" class="data row670 col14" >0.019500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row671" class="row_heading level0 row671" >672</th>
          <td id="T_78b05_row671_col0" class="data row671 col0" >None</td>
          <td id="T_78b05_row671_col1" class="data row671 col1" >0.038200</td>
          <td id="T_78b05_row671_col2" class="data row671 col2" >0.076100</td>
          <td id="T_78b05_row671_col3" class="data row671 col3" >0.019200</td>
          <td id="T_78b05_row671_col4" class="data row671 col4" >-0.048600</td>
          <td id="T_78b05_row671_col5" class="data row671 col5" >0.029600</td>
          <td id="T_78b05_row671_col6" class="data row671 col6" >0.059300</td>
          <td id="T_78b05_row671_col7" class="data row671 col7" >-0.031500</td>
          <td id="T_78b05_row671_col8" class="data row671 col8" >0.007300</td>
          <td id="T_78b05_row671_col9" class="data row671 col9" >0.106000</td>
          <td id="T_78b05_row671_col10" class="data row671 col10" >0.011600</td>
          <td id="T_78b05_row671_col11" class="data row671 col11" >0.030700</td>
          <td id="T_78b05_row671_col12" class="data row671 col12" >0.030200</td>
          <td id="T_78b05_row671_col13" class="data row671 col13" >0.063200</td>
          <td id="T_78b05_row671_col14" class="data row671 col14" >0.029700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row672" class="row_heading level0 row672" >673</th>
          <td id="T_78b05_row672_col0" class="data row672 col0" >None</td>
          <td id="T_78b05_row672_col1" class="data row672 col1" >0.039800</td>
          <td id="T_78b05_row672_col2" class="data row672 col2" >-0.015500</td>
          <td id="T_78b05_row672_col3" class="data row672 col3" >0.008400</td>
          <td id="T_78b05_row672_col4" class="data row672 col4" >-0.050100</td>
          <td id="T_78b05_row672_col5" class="data row672 col5" >-0.030700</td>
          <td id="T_78b05_row672_col6" class="data row672 col6" >0.013000</td>
          <td id="T_78b05_row672_col7" class="data row672 col7" >0.037600</td>
          <td id="T_78b05_row672_col8" class="data row672 col8" >0.005700</td>
          <td id="T_78b05_row672_col9" class="data row672 col9" >0.014300</td>
          <td id="T_78b05_row672_col10" class="data row672 col10" >0.022300</td>
          <td id="T_78b05_row672_col11" class="data row672 col11" >0.032200</td>
          <td id="T_78b05_row672_col12" class="data row672 col12" >0.030100</td>
          <td id="T_78b05_row672_col13" class="data row672 col13" >0.016900</td>
          <td id="T_78b05_row672_col14" class="data row672 col14" >0.039400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row673" class="row_heading level0 row673" >674</th>
          <td id="T_78b05_row673_col0" class="data row673 col0" >None</td>
          <td id="T_78b05_row673_col1" class="data row673 col1" >0.036200</td>
          <td id="T_78b05_row673_col2" class="data row673 col2" >-0.023900</td>
          <td id="T_78b05_row673_col3" class="data row673 col3" >-0.053100</td>
          <td id="T_78b05_row673_col4" class="data row673 col4" >0.027600</td>
          <td id="T_78b05_row673_col5" class="data row673 col5" >0.060300</td>
          <td id="T_78b05_row673_col6" class="data row673 col6" >0.001100</td>
          <td id="T_78b05_row673_col7" class="data row673 col7" >-0.026800</td>
          <td id="T_78b05_row673_col8" class="data row673 col8" >0.009300</td>
          <td id="T_78b05_row673_col9" class="data row673 col9" >0.005900</td>
          <td id="T_78b05_row673_col10" class="data row673 col10" >0.083800</td>
          <td id="T_78b05_row673_col11" class="data row673 col11" >0.045500</td>
          <td id="T_78b05_row673_col12" class="data row673 col12" >0.060900</td>
          <td id="T_78b05_row673_col13" class="data row673 col13" >0.005000</td>
          <td id="T_78b05_row673_col14" class="data row673 col14" >0.025000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row674" class="row_heading level0 row674" >675</th>
          <td id="T_78b05_row674_col0" class="data row674 col0" >None</td>
          <td id="T_78b05_row674_col1" class="data row674 col1" >0.029100</td>
          <td id="T_78b05_row674_col2" class="data row674 col2" >0.001300</td>
          <td id="T_78b05_row674_col3" class="data row674 col3" >-0.088000</td>
          <td id="T_78b05_row674_col4" class="data row674 col4" >0.043800</td>
          <td id="T_78b05_row674_col5" class="data row674 col5" >0.019000</td>
          <td id="T_78b05_row674_col6" class="data row674 col6" >-0.005800</td>
          <td id="T_78b05_row674_col7" class="data row674 col7" >0.000400</td>
          <td id="T_78b05_row674_col8" class="data row674 col8" >0.016400</td>
          <td id="T_78b05_row674_col9" class="data row674 col9" >0.031200</td>
          <td id="T_78b05_row674_col10" class="data row674 col10" >0.118700</td>
          <td id="T_78b05_row674_col11" class="data row674 col11" >0.061700</td>
          <td id="T_78b05_row674_col12" class="data row674 col12" >0.019600</td>
          <td id="T_78b05_row674_col13" class="data row674 col13" >0.001900</td>
          <td id="T_78b05_row674_col14" class="data row674 col14" >0.002100</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row675" class="row_heading level0 row675" >676</th>
          <td id="T_78b05_row675_col0" class="data row675 col0" >None</td>
          <td id="T_78b05_row675_col1" class="data row675 col1" >0.048200</td>
          <td id="T_78b05_row675_col2" class="data row675 col2" >-0.038200</td>
          <td id="T_78b05_row675_col3" class="data row675 col3" >0.010400</td>
          <td id="T_78b05_row675_col4" class="data row675 col4" >-0.014100</td>
          <td id="T_78b05_row675_col5" class="data row675 col5" >0.041300</td>
          <td id="T_78b05_row675_col6" class="data row675 col6" >0.002600</td>
          <td id="T_78b05_row675_col7" class="data row675 col7" >0.005600</td>
          <td id="T_78b05_row675_col8" class="data row675 col8" >0.002700</td>
          <td id="T_78b05_row675_col9" class="data row675 col9" >0.008400</td>
          <td id="T_78b05_row675_col10" class="data row675 col10" >0.020400</td>
          <td id="T_78b05_row675_col11" class="data row675 col11" >0.003800</td>
          <td id="T_78b05_row675_col12" class="data row675 col12" >0.042000</td>
          <td id="T_78b05_row675_col13" class="data row675 col13" >0.006500</td>
          <td id="T_78b05_row675_col14" class="data row675 col14" >0.007400</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row676" class="row_heading level0 row676" >677</th>
          <td id="T_78b05_row676_col0" class="data row676 col0" >None</td>
          <td id="T_78b05_row676_col1" class="data row676 col1" >0.034100</td>
          <td id="T_78b05_row676_col2" class="data row676 col2" >-0.050800</td>
          <td id="T_78b05_row676_col3" class="data row676 col3" >-0.065100</td>
          <td id="T_78b05_row676_col4" class="data row676 col4" >0.012600</td>
          <td id="T_78b05_row676_col5" class="data row676 col5" >0.003300</td>
          <td id="T_78b05_row676_col6" class="data row676 col6" >0.067800</td>
          <td id="T_78b05_row676_col7" class="data row676 col7" >-0.031500</td>
          <td id="T_78b05_row676_col8" class="data row676 col8" >0.011500</td>
          <td id="T_78b05_row676_col9" class="data row676 col9" >0.021000</td>
          <td id="T_78b05_row676_col10" class="data row676 col10" >0.095800</td>
          <td id="T_78b05_row676_col11" class="data row676 col11" >0.030500</td>
          <td id="T_78b05_row676_col12" class="data row676 col12" >0.004000</td>
          <td id="T_78b05_row676_col13" class="data row676 col13" >0.071700</td>
          <td id="T_78b05_row676_col14" class="data row676 col14" >0.029700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row677" class="row_heading level0 row677" >678</th>
          <td id="T_78b05_row677_col0" class="data row677 col0" >None</td>
          <td id="T_78b05_row677_col1" class="data row677 col1" >0.029500</td>
          <td id="T_78b05_row677_col2" class="data row677 col2" >0.047800</td>
          <td id="T_78b05_row677_col3" class="data row677 col3" >0.013500</td>
          <td id="T_78b05_row677_col4" class="data row677 col4" >-0.015400</td>
          <td id="T_78b05_row677_col5" class="data row677 col5" >0.001700</td>
          <td id="T_78b05_row677_col6" class="data row677 col6" >0.026500</td>
          <td id="T_78b05_row677_col7" class="data row677 col7" >-0.032000</td>
          <td id="T_78b05_row677_col8" class="data row677 col8" >0.016100</td>
          <td id="T_78b05_row677_col9" class="data row677 col9" >0.077700</td>
          <td id="T_78b05_row677_col10" class="data row677 col10" >0.017300</td>
          <td id="T_78b05_row677_col11" class="data row677 col11" >0.002500</td>
          <td id="T_78b05_row677_col12" class="data row677 col12" >0.002400</td>
          <td id="T_78b05_row677_col13" class="data row677 col13" >0.030400</td>
          <td id="T_78b05_row677_col14" class="data row677 col14" >0.030300</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row678" class="row_heading level0 row678" >679</th>
          <td id="T_78b05_row678_col0" class="data row678 col0" >None</td>
          <td id="T_78b05_row678_col1" class="data row678 col1" >0.046900</td>
          <td id="T_78b05_row678_col2" class="data row678 col2" >-0.002700</td>
          <td id="T_78b05_row678_col3" class="data row678 col3" >0.064900</td>
          <td id="T_78b05_row678_col4" class="data row678 col4" >-0.018300</td>
          <td id="T_78b05_row678_col5" class="data row678 col5" >0.021800</td>
          <td id="T_78b05_row678_col6" class="data row678 col6" >0.032800</td>
          <td id="T_78b05_row678_col7" class="data row678 col7" >-0.026400</td>
          <td id="T_78b05_row678_col8" class="data row678 col8" >0.001400</td>
          <td id="T_78b05_row678_col9" class="data row678 col9" >0.027100</td>
          <td id="T_78b05_row678_col10" class="data row678 col10" >0.034100</td>
          <td id="T_78b05_row678_col11" class="data row678 col11" >0.000400</td>
          <td id="T_78b05_row678_col12" class="data row678 col12" >0.022400</td>
          <td id="T_78b05_row678_col13" class="data row678 col13" >0.036700</td>
          <td id="T_78b05_row678_col14" class="data row678 col14" >0.024700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row679" class="row_heading level0 row679" >680</th>
          <td id="T_78b05_row679_col0" class="data row679 col0" >None</td>
          <td id="T_78b05_row679_col1" class="data row679 col1" >0.039300</td>
          <td id="T_78b05_row679_col2" class="data row679 col2" >-0.003200</td>
          <td id="T_78b05_row679_col3" class="data row679 col3" >-0.009000</td>
          <td id="T_78b05_row679_col4" class="data row679 col4" >-0.052900</td>
          <td id="T_78b05_row679_col5" class="data row679 col5" >-0.015800</td>
          <td id="T_78b05_row679_col6" class="data row679 col6" >-0.006400</td>
          <td id="T_78b05_row679_col7" class="data row679 col7" >0.000900</td>
          <td id="T_78b05_row679_col8" class="data row679 col8" >0.006200</td>
          <td id="T_78b05_row679_col9" class="data row679 col9" >0.026700</td>
          <td id="T_78b05_row679_col10" class="data row679 col10" >0.039800</td>
          <td id="T_78b05_row679_col11" class="data row679 col11" >0.035000</td>
          <td id="T_78b05_row679_col12" class="data row679 col12" >0.015200</td>
          <td id="T_78b05_row679_col13" class="data row679 col13" >0.002500</td>
          <td id="T_78b05_row679_col14" class="data row679 col14" >0.002700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row680" class="row_heading level0 row680" >681</th>
          <td id="T_78b05_row680_col0" class="data row680 col0" >None</td>
          <td id="T_78b05_row680_col1" class="data row680 col1" >0.039900</td>
          <td id="T_78b05_row680_col2" class="data row680 col2" >-0.013700</td>
          <td id="T_78b05_row680_col3" class="data row680 col3" >0.000200</td>
          <td id="T_78b05_row680_col4" class="data row680 col4" >-0.064300</td>
          <td id="T_78b05_row680_col5" class="data row680 col5" >-0.026300</td>
          <td id="T_78b05_row680_col6" class="data row680 col6" >-0.042100</td>
          <td id="T_78b05_row680_col7" class="data row680 col7" >-0.007700</td>
          <td id="T_78b05_row680_col8" class="data row680 col8" >0.005700</td>
          <td id="T_78b05_row680_col9" class="data row680 col9" >0.016100</td>
          <td id="T_78b05_row680_col10" class="data row680 col10" >0.030500</td>
          <td id="T_78b05_row680_col11" class="data row680 col11" >0.046300</td>
          <td id="T_78b05_row680_col12" class="data row680 col12" >0.025700</td>
          <td id="T_78b05_row680_col13" class="data row680 col13" >0.038200</td>
          <td id="T_78b05_row680_col14" class="data row680 col14" >0.005900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row681" class="row_heading level0 row681" >682</th>
          <td id="T_78b05_row681_col0" class="data row681 col0" >None</td>
          <td id="T_78b05_row681_col1" class="data row681 col1" >0.037000</td>
          <td id="T_78b05_row681_col2" class="data row681 col2" >-0.026200</td>
          <td id="T_78b05_row681_col3" class="data row681 col3" >-0.042500</td>
          <td id="T_78b05_row681_col4" class="data row681 col4" >-0.062300</td>
          <td id="T_78b05_row681_col5" class="data row681 col5" >0.005300</td>
          <td id="T_78b05_row681_col6" class="data row681 col6" >-0.041300</td>
          <td id="T_78b05_row681_col7" class="data row681 col7" >0.021100</td>
          <td id="T_78b05_row681_col8" class="data row681 col8" >0.008500</td>
          <td id="T_78b05_row681_col9" class="data row681 col9" >0.003600</td>
          <td id="T_78b05_row681_col10" class="data row681 col10" >0.073300</td>
          <td id="T_78b05_row681_col11" class="data row681 col11" >0.044400</td>
          <td id="T_78b05_row681_col12" class="data row681 col12" >0.005900</td>
          <td id="T_78b05_row681_col13" class="data row681 col13" >0.037400</td>
          <td id="T_78b05_row681_col14" class="data row681 col14" >0.022800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row682" class="row_heading level0 row682" >683</th>
          <td id="T_78b05_row682_col0" class="data row682 col0" >PC2</td>
          <td id="T_78b05_row682_col1" class="data row682 col1" >0.028700</td>
          <td id="T_78b05_row682_col2" class="data row682 col2" >0.103200</td>
          <td id="T_78b05_row682_col3" class="data row682 col3" >-0.011900</td>
          <td id="T_78b05_row682_col4" class="data row682 col4" >-0.029600</td>
          <td id="T_78b05_row682_col5" class="data row682 col5" >-0.047000</td>
          <td id="T_78b05_row682_col6" class="data row682 col6" >-0.005900</td>
          <td id="T_78b05_row682_col7" class="data row682 col7" >-0.054900</td>
          <td id="T_78b05_row682_col8" class="data row682 col8" >0.016800</td>
          <td id="T_78b05_row682_col9" class="data row682 col9" >0.133000</td>
          <td id="T_78b05_row682_col10" class="data row682 col10" >0.042600</td>
          <td id="T_78b05_row682_col11" class="data row682 col11" >0.011700</td>
          <td id="T_78b05_row682_col12" class="data row682 col12" >0.046300</td>
          <td id="T_78b05_row682_col13" class="data row682 col13" >0.002000</td>
          <td id="T_78b05_row682_col14" class="data row682 col14" >0.053200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row683" class="row_heading level0 row683" >684</th>
          <td id="T_78b05_row683_col0" class="data row683 col0" >None</td>
          <td id="T_78b05_row683_col1" class="data row683 col1" >0.029800</td>
          <td id="T_78b05_row683_col2" class="data row683 col2" >0.047400</td>
          <td id="T_78b05_row683_col3" class="data row683 col3" >0.028800</td>
          <td id="T_78b05_row683_col4" class="data row683 col4" >0.066900</td>
          <td id="T_78b05_row683_col5" class="data row683 col5" >-0.075700</td>
          <td id="T_78b05_row683_col6" class="data row683 col6" >-0.039000</td>
          <td id="T_78b05_row683_col7" class="data row683 col7" >0.044000</td>
          <td id="T_78b05_row683_col8" class="data row683 col8" >0.015700</td>
          <td id="T_78b05_row683_col9" class="data row683 col9" >0.077300</td>
          <td id="T_78b05_row683_col10" class="data row683 col10" >0.002000</td>
          <td id="T_78b05_row683_col11" class="data row683 col11" >0.084800</td>
          <td id="T_78b05_row683_col12" class="data row683 col12" >0.075100</td>
          <td id="T_78b05_row683_col13" class="data row683 col13" >0.035100</td>
          <td id="T_78b05_row683_col14" class="data row683 col14" >0.045800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row684" class="row_heading level0 row684" >685</th>
          <td id="T_78b05_row684_col0" class="data row684 col0" >None</td>
          <td id="T_78b05_row684_col1" class="data row684 col1" >0.033000</td>
          <td id="T_78b05_row684_col2" class="data row684 col2" >0.056700</td>
          <td id="T_78b05_row684_col3" class="data row684 col3" >-0.010200</td>
          <td id="T_78b05_row684_col4" class="data row684 col4" >0.054300</td>
          <td id="T_78b05_row684_col5" class="data row684 col5" >-0.038200</td>
          <td id="T_78b05_row684_col6" class="data row684 col6" >0.034500</td>
          <td id="T_78b05_row684_col7" class="data row684 col7" >0.043800</td>
          <td id="T_78b05_row684_col8" class="data row684 col8" >0.012500</td>
          <td id="T_78b05_row684_col9" class="data row684 col9" >0.086500</td>
          <td id="T_78b05_row684_col10" class="data row684 col10" >0.041000</td>
          <td id="T_78b05_row684_col11" class="data row684 col11" >0.072200</td>
          <td id="T_78b05_row684_col12" class="data row684 col12" >0.037600</td>
          <td id="T_78b05_row684_col13" class="data row684 col13" >0.038400</td>
          <td id="T_78b05_row684_col14" class="data row684 col14" >0.045500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row685" class="row_heading level0 row685" >686</th>
          <td id="T_78b05_row685_col0" class="data row685 col0" >None</td>
          <td id="T_78b05_row685_col1" class="data row685 col1" >0.035900</td>
          <td id="T_78b05_row685_col2" class="data row685 col2" >-0.037900</td>
          <td id="T_78b05_row685_col3" class="data row685 col3" >-0.029600</td>
          <td id="T_78b05_row685_col4" class="data row685 col4" >-0.047300</td>
          <td id="T_78b05_row685_col5" class="data row685 col5" >-0.050400</td>
          <td id="T_78b05_row685_col6" class="data row685 col6" >0.028100</td>
          <td id="T_78b05_row685_col7" class="data row685 col7" >-0.008000</td>
          <td id="T_78b05_row685_col8" class="data row685 col8" >0.009600</td>
          <td id="T_78b05_row685_col9" class="data row685 col9" >0.008000</td>
          <td id="T_78b05_row685_col10" class="data row685 col10" >0.060400</td>
          <td id="T_78b05_row685_col11" class="data row685 col11" >0.029400</td>
          <td id="T_78b05_row685_col12" class="data row685 col12" >0.049700</td>
          <td id="T_78b05_row685_col13" class="data row685 col13" >0.032000</td>
          <td id="T_78b05_row685_col14" class="data row685 col14" >0.006200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row686" class="row_heading level0 row686" >687</th>
          <td id="T_78b05_row686_col0" class="data row686 col0" >None</td>
          <td id="T_78b05_row686_col1" class="data row686 col1" >0.042000</td>
          <td id="T_78b05_row686_col2" class="data row686 col2" >0.039400</td>
          <td id="T_78b05_row686_col3" class="data row686 col3" >0.039400</td>
          <td id="T_78b05_row686_col4" class="data row686 col4" >-0.012600</td>
          <td id="T_78b05_row686_col5" class="data row686 col5" >0.017200</td>
          <td id="T_78b05_row686_col6" class="data row686 col6" >0.010000</td>
          <td id="T_78b05_row686_col7" class="data row686 col7" >0.010000</td>
          <td id="T_78b05_row686_col8" class="data row686 col8" >0.003500</td>
          <td id="T_78b05_row686_col9" class="data row686 col9" >0.069200</td>
          <td id="T_78b05_row686_col10" class="data row686 col10" >0.008700</td>
          <td id="T_78b05_row686_col11" class="data row686 col11" >0.005300</td>
          <td id="T_78b05_row686_col12" class="data row686 col12" >0.017800</td>
          <td id="T_78b05_row686_col13" class="data row686 col13" >0.013900</td>
          <td id="T_78b05_row686_col14" class="data row686 col14" >0.011800</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row687" class="row_heading level0 row687" >688</th>
          <td id="T_78b05_row687_col0" class="data row687 col0" >None</td>
          <td id="T_78b05_row687_col1" class="data row687 col1" >0.049100</td>
          <td id="T_78b05_row687_col2" class="data row687 col2" >0.004400</td>
          <td id="T_78b05_row687_col3" class="data row687 col3" >0.046600</td>
          <td id="T_78b05_row687_col4" class="data row687 col4" >0.016700</td>
          <td id="T_78b05_row687_col5" class="data row687 col5" >0.066300</td>
          <td id="T_78b05_row687_col6" class="data row687 col6" >-0.065200</td>
          <td id="T_78b05_row687_col7" class="data row687 col7" >-0.010700</td>
          <td id="T_78b05_row687_col8" class="data row687 col8" >0.003600</td>
          <td id="T_78b05_row687_col9" class="data row687 col9" >0.034200</td>
          <td id="T_78b05_row687_col10" class="data row687 col10" >0.015800</td>
          <td id="T_78b05_row687_col11" class="data row687 col11" >0.034600</td>
          <td id="T_78b05_row687_col12" class="data row687 col12" >0.066900</td>
          <td id="T_78b05_row687_col13" class="data row687 col13" >0.061300</td>
          <td id="T_78b05_row687_col14" class="data row687 col14" >0.009000</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row688" class="row_heading level0 row688" >689</th>
          <td id="T_78b05_row688_col0" class="data row688 col0" >None</td>
          <td id="T_78b05_row688_col1" class="data row688 col1" >0.029100</td>
          <td id="T_78b05_row688_col2" class="data row688 col2" >0.026600</td>
          <td id="T_78b05_row688_col3" class="data row688 col3" >-0.021200</td>
          <td id="T_78b05_row688_col4" class="data row688 col4" >0.006000</td>
          <td id="T_78b05_row688_col5" class="data row688 col5" >-0.005000</td>
          <td id="T_78b05_row688_col6" class="data row688 col6" >-0.049900</td>
          <td id="T_78b05_row688_col7" class="data row688 col7" >-0.017000</td>
          <td id="T_78b05_row688_col8" class="data row688 col8" >0.016400</td>
          <td id="T_78b05_row688_col9" class="data row688 col9" >0.056400</td>
          <td id="T_78b05_row688_col10" class="data row688 col10" >0.051900</td>
          <td id="T_78b05_row688_col11" class="data row688 col11" >0.023900</td>
          <td id="T_78b05_row688_col12" class="data row688 col12" >0.004400</td>
          <td id="T_78b05_row688_col13" class="data row688 col13" >0.046100</td>
          <td id="T_78b05_row688_col14" class="data row688 col14" >0.015200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row689" class="row_heading level0 row689" >690</th>
          <td id="T_78b05_row689_col0" class="data row689 col0" >None</td>
          <td id="T_78b05_row689_col1" class="data row689 col1" >0.033700</td>
          <td id="T_78b05_row689_col2" class="data row689 col2" >0.031800</td>
          <td id="T_78b05_row689_col3" class="data row689 col3" >-0.066700</td>
          <td id="T_78b05_row689_col4" class="data row689 col4" >-0.041100</td>
          <td id="T_78b05_row689_col5" class="data row689 col5" >0.065900</td>
          <td id="T_78b05_row689_col6" class="data row689 col6" >-0.061200</td>
          <td id="T_78b05_row689_col7" class="data row689 col7" >-0.001500</td>
          <td id="T_78b05_row689_col8" class="data row689 col8" >0.011800</td>
          <td id="T_78b05_row689_col9" class="data row689 col9" >0.061600</td>
          <td id="T_78b05_row689_col10" class="data row689 col10" >0.097500</td>
          <td id="T_78b05_row689_col11" class="data row689 col11" >0.023200</td>
          <td id="T_78b05_row689_col12" class="data row689 col12" >0.066500</td>
          <td id="T_78b05_row689_col13" class="data row689 col13" >0.057300</td>
          <td id="T_78b05_row689_col14" class="data row689 col14" >0.000200</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row690" class="row_heading level0 row690" >691</th>
          <td id="T_78b05_row690_col0" class="data row690 col0" >None</td>
          <td id="T_78b05_row690_col1" class="data row690 col1" >0.041400</td>
          <td id="T_78b05_row690_col2" class="data row690 col2" >-0.034900</td>
          <td id="T_78b05_row690_col3" class="data row690 col3" >-0.025800</td>
          <td id="T_78b05_row690_col4" class="data row690 col4" >0.031200</td>
          <td id="T_78b05_row690_col5" class="data row690 col5" >-0.005400</td>
          <td id="T_78b05_row690_col6" class="data row690 col6" >-0.018700</td>
          <td id="T_78b05_row690_col7" class="data row690 col7" >-0.022200</td>
          <td id="T_78b05_row690_col8" class="data row690 col8" >0.004200</td>
          <td id="T_78b05_row690_col9" class="data row690 col9" >0.005100</td>
          <td id="T_78b05_row690_col10" class="data row690 col10" >0.056600</td>
          <td id="T_78b05_row690_col11" class="data row690 col11" >0.049100</td>
          <td id="T_78b05_row690_col12" class="data row690 col12" >0.004800</td>
          <td id="T_78b05_row690_col13" class="data row690 col13" >0.014800</td>
          <td id="T_78b05_row690_col14" class="data row690 col14" >0.020500</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row691" class="row_heading level0 row691" >692</th>
          <td id="T_78b05_row691_col0" class="data row691 col0" >None</td>
          <td id="T_78b05_row691_col1" class="data row691 col1" >0.041200</td>
          <td id="T_78b05_row691_col2" class="data row691 col2" >-0.035600</td>
          <td id="T_78b05_row691_col3" class="data row691 col3" >0.005700</td>
          <td id="T_78b05_row691_col4" class="data row691 col4" >-0.003400</td>
          <td id="T_78b05_row691_col5" class="data row691 col5" >-0.008600</td>
          <td id="T_78b05_row691_col6" class="data row691 col6" >0.028200</td>
          <td id="T_78b05_row691_col7" class="data row691 col7" >0.007000</td>
          <td id="T_78b05_row691_col8" class="data row691 col8" >0.004300</td>
          <td id="T_78b05_row691_col9" class="data row691 col9" >0.005800</td>
          <td id="T_78b05_row691_col10" class="data row691 col10" >0.025100</td>
          <td id="T_78b05_row691_col11" class="data row691 col11" >0.014500</td>
          <td id="T_78b05_row691_col12" class="data row691 col12" >0.008000</td>
          <td id="T_78b05_row691_col13" class="data row691 col13" >0.032100</td>
          <td id="T_78b05_row691_col14" class="data row691 col14" >0.008700</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row692" class="row_heading level0 row692" >693</th>
          <td id="T_78b05_row692_col0" class="data row692 col0" >None</td>
          <td id="T_78b05_row692_col1" class="data row692 col1" >0.041900</td>
          <td id="T_78b05_row692_col2" class="data row692 col2" >-0.016000</td>
          <td id="T_78b05_row692_col3" class="data row692 col3" >-0.023200</td>
          <td id="T_78b05_row692_col4" class="data row692 col4" >0.040500</td>
          <td id="T_78b05_row692_col5" class="data row692 col5" >0.003700</td>
          <td id="T_78b05_row692_col6" class="data row692 col6" >0.012900</td>
          <td id="T_78b05_row692_col7" class="data row692 col7" >0.052200</td>
          <td id="T_78b05_row692_col8" class="data row692 col8" >0.003600</td>
          <td id="T_78b05_row692_col9" class="data row692 col9" >0.013900</td>
          <td id="T_78b05_row692_col10" class="data row692 col10" >0.054000</td>
          <td id="T_78b05_row692_col11" class="data row692 col11" >0.058400</td>
          <td id="T_78b05_row692_col12" class="data row692 col12" >0.004400</td>
          <td id="T_78b05_row692_col13" class="data row692 col13" >0.016800</td>
          <td id="T_78b05_row692_col14" class="data row692 col14" >0.053900</td>
        </tr>
        <tr>
          <th id="T_78b05_level0_row693" class="row_heading level0 row693" >694</th>
          <td id="T_78b05_row693_col0" class="data row693 col0" >None</td>
          <td id="T_78b05_row693_col1" class="data row693 col1" >0.043500</td>
          <td id="T_78b05_row693_col2" class="data row693 col2" >-0.017400</td>
          <td id="T_78b05_row693_col3" class="data row693 col3" >-0.001200</td>
          <td id="T_78b05_row693_col4" class="data row693 col4" >0.024100</td>
          <td id="T_78b05_row693_col5" class="data row693 col5" >-0.007700</td>
          <td id="T_78b05_row693_col6" class="data row693 col6" >-0.036300</td>
          <td id="T_78b05_row693_col7" class="data row693 col7" >0.022400</td>
          <td id="T_78b05_row693_col8" class="data row693 col8" >0.002000</td>
          <td id="T_78b05_row693_col9" class="data row693 col9" >0.012400</td>
          <td id="T_78b05_row693_col10" class="data row693 col10" >0.032000</td>
          <td id="T_78b05_row693_col11" class="data row693 col11" >0.042000</td>
          <td id="T_78b05_row693_col12" class="data row693 col12" >0.007100</td>
          <td id="T_78b05_row693_col13" class="data row693 col13" >0.032400</td>
          <td id="T_78b05_row693_col14" class="data row693 col14" >0.024100</td>
        </tr>
      </tbody>
    </table>



To visualize all identified negatives within the compressed feature
space represented by the first two Principal Components (PCs), you can
use the ``dPULearnPlot.pca`` method:

.. code:: ipython2

    dpul_plot = aa.dPULearnPlot()
    dpul_plot.pca(df_pu=df_pu, labels=labels)
    plt.tight_layout()
    plt.show()



.. image:: examples/dpul_plot_pca_1_output_3_0.png


Which can be easily adjusted by our ``aa.plot_settings()`` function:

.. code:: ipython2

    aa.plot_settings(font_scale=0.8)
    dpul_plot.pca(df_pu=df_pu, labels=labels)
    plt.tight_layout()
    plt.show()



.. image:: examples/dpul_plot_pca_2_output_5_0.png


The dashed lines indicate the mean values across the positive samples
for the PC1 and PC2, based on which the samples from the unlabeled group
with the greatest distance were identified as reliable negatives by
dPULearn. This becomes more clear using boolean masks and the
``show_pos_mean_x`` and ``show_pos_mean_y`` parameters:

.. code:: ipython2

    # Filter only positives and negatives selected based on PC1
    mask1 = [x in ["PC1", None] for x in df_pu["selection_via"]]
    mask2 = [x in [0, 1] for x in labels]
    mask = [m1 and m2 for m1, m2 in zip(mask1, mask2)]
    dpul_plot.pca(df_pu=df_pu[mask], labels=labels[mask], show_pos_mean_y=False)
    plt.tight_layout()
    plt.show()



.. image:: examples/dpul_plot_pca_3_output_7_0.png


.. code:: ipython2

    # Filter only positives and negatives selected based on PC1
    mask1 = [x in ["PC2", None] for x in df_pu["selection_via"]]
    mask2 = [x in [0, 1] for x in labels]
    mask = [m1 and m2 for m1, m2 in zip(mask1, mask2)]
    dpul_plot.pca(df_pu=df_pu[mask], labels=labels[mask], show_pos_mean_x=False)
    plt.tight_layout()
    plt.show()



.. image:: examples/dpul_plot_pca_4_output_8_0.png


You can change the PCs to be shown on the x- and y-axis by providing
integers numbers to the ``pc_x`` and ``pc_y`` paramters:

.. code:: ipython2

    mask1 = [x in ["PC3", "PC4", None] for x in df_pu["selection_via"]]
    mask2 = [x in [0, 1, 2] for x in labels]
    mask = [m1 and m2 for m1, m2 in zip(mask1, mask2)]
    dpul_plot.pca(df_pu=df_pu[mask], labels=labels[mask], pc_x=3, pc_y=4)
    plt.tight_layout()
    plt.show()



.. image:: examples/dpul_plot_pca_5_output_10_0.png


Adjustment of ``colors`` and ``names`` must be aligned:

.. code:: ipython2

    colors = ["r", "black", "b"]
    names = ["Red group", "Black group", "Blue group"]
    dpul_plot.pca(df_pu=df_pu, labels=labels, colors=colors, names=names)
    plt.tight_layout()
    plt.show()



.. image:: examples/dpul_plot_pca_6_output_12_0.png


The legend can be shifted along the y-axis using ``legend_y``
(default=-0.15), useful if the ``figsize`` (default=(5,5)) is adjusted:

.. code:: ipython2

    dpul_plot.pca(df_pu=df_pu, labels=labels, figsize=(6, 8), legend_y=-0.1)
    plt.tight_layout()
    plt.show()



.. image:: examples/dpul_plot_pca_7_output_14_0.png


The scatter plot using the ``args_scatter`` parameter, which is a key
word argument dictionary passed to the internally called the
``plt.scatter`` class:

.. code:: ipython2

    dpul_plot.pca(df_pu=df_pu, labels=labels, args_scatter={"s": 25, "edgecolor": "black"})
    plt.tight_layout()
    plt.show()



.. image:: examples/dpul_plot_pca_8_output_16_0.png


To change the legend, just disable it (setting ``legend=False``) and
re-create it using the ``aa.plot_legend()`` function:

.. code:: ipython2

    DICT_COLOR = aa.plot_get_cdict()
    dict_color = {"Neg": DICT_COLOR["SAMPLES_REL_NEG"], "Pos": DICT_COLOR["SAMPLES_POS"], "Unl": DICT_COLOR["SAMPLES_UNL"]}
    dpul_plot.pca(df_pu=df_pu, labels=labels, legend=False)
    aa.plot_legend(dict_color=dict_color, y=1.2, handlelength=1, marker="o")
    plt.tight_layout()
    plt.show()



.. image:: examples/dpul_plot_pca_9_output_18_0.png


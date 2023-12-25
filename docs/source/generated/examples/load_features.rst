You can load features corresponding to a specific dataset using the
``load_feature()`` function:

.. code:: ipython2

    import aaanalysis as aa
    df_feat = aa.load_features(name="DOM_GSEC")
    aa.display_df(df_feat)



.. raw:: html

    <style type="text/css">
    #T_10a98 thead th {
      background-color: white;
      color: black;
    }
    #T_10a98 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_10a98 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_10a98 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_10a98  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_10a98 table {
      font-size: 12px;
    }
    </style>
    <table id="T_10a98" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_10a98_level0_col0" class="col_heading level0 col0" >feature</th>
          <th id="T_10a98_level0_col1" class="col_heading level0 col1" >category</th>
          <th id="T_10a98_level0_col2" class="col_heading level0 col2" >subcategory</th>
          <th id="T_10a98_level0_col3" class="col_heading level0 col3" >scale_name</th>
          <th id="T_10a98_level0_col4" class="col_heading level0 col4" >scale_description</th>
          <th id="T_10a98_level0_col5" class="col_heading level0 col5" >abs_auc</th>
          <th id="T_10a98_level0_col6" class="col_heading level0 col6" >abs_mean_dif</th>
          <th id="T_10a98_level0_col7" class="col_heading level0 col7" >mean_dif</th>
          <th id="T_10a98_level0_col8" class="col_heading level0 col8" >std_test</th>
          <th id="T_10a98_level0_col9" class="col_heading level0 col9" >std_ref</th>
          <th id="T_10a98_level0_col10" class="col_heading level0 col10" >p_val_mann_whitney</th>
          <th id="T_10a98_level0_col11" class="col_heading level0 col11" >p_val_fdr_bh</th>
          <th id="T_10a98_level0_col12" class="col_heading level0 col12" >positions</th>
          <th id="T_10a98_level0_col13" class="col_heading level0 col13" >feat_importance</th>
          <th id="T_10a98_level0_col14" class="col_heading level0 col14" >feat_importance_std</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_10a98_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_10a98_row0_col0" class="data row0 col0" >TMD_C_JMD_C-...)-KLEP840101</td>
          <td id="T_10a98_row0_col1" class="data row0 col1" >Energy</td>
          <td id="T_10a98_row0_col2" class="data row0 col2" >Charge</td>
          <td id="T_10a98_row0_col3" class="data row0 col3" >Charge</td>
          <td id="T_10a98_row0_col4" class="data row0 col4" >Net charge (...t al., 1984)</td>
          <td id="T_10a98_row0_col5" class="data row0 col5" >0.244000</td>
          <td id="T_10a98_row0_col6" class="data row0 col6" >0.103666</td>
          <td id="T_10a98_row0_col7" class="data row0 col7" >0.103666</td>
          <td id="T_10a98_row0_col8" class="data row0 col8" >0.106692</td>
          <td id="T_10a98_row0_col9" class="data row0 col9" >0.110506</td>
          <td id="T_10a98_row0_col10" class="data row0 col10" >0.000000</td>
          <td id="T_10a98_row0_col11" class="data row0 col11" >0.000000</td>
          <td id="T_10a98_row0_col12" class="data row0 col12" >31,32,33,34,35</td>
          <td id="T_10a98_row0_col13" class="data row0 col13" >0.970400</td>
          <td id="T_10a98_row0_col14" class="data row0 col14" >1.438918</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_10a98_row1_col0" class="data row1 col0" >TMD_C_JMD_C-...)-FINA910104</td>
          <td id="T_10a98_row1_col1" class="data row1 col1" >Conformation</td>
          <td id="T_10a98_row1_col2" class="data row1 col2" >α-helix (C-cap)</td>
          <td id="T_10a98_row1_col3" class="data row1 col3" >α-helix termination</td>
          <td id="T_10a98_row1_col4" class="data row1 col4" >Helix termin...t al., 1991)</td>
          <td id="T_10a98_row1_col5" class="data row1 col5" >0.243000</td>
          <td id="T_10a98_row1_col6" class="data row1 col6" >0.085064</td>
          <td id="T_10a98_row1_col7" class="data row1 col7" >0.085064</td>
          <td id="T_10a98_row1_col8" class="data row1 col8" >0.098774</td>
          <td id="T_10a98_row1_col9" class="data row1 col9" >0.096946</td>
          <td id="T_10a98_row1_col10" class="data row1 col10" >0.000000</td>
          <td id="T_10a98_row1_col11" class="data row1 col11" >0.000000</td>
          <td id="T_10a98_row1_col12" class="data row1 col12" >31,32,33,34,35</td>
          <td id="T_10a98_row1_col13" class="data row1 col13" >0.000000</td>
          <td id="T_10a98_row1_col14" class="data row1 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_10a98_row2_col0" class="data row2 col0" >TMD_C_JMD_C-...)-LEVM760105</td>
          <td id="T_10a98_row2_col1" class="data row2 col1" >Shape</td>
          <td id="T_10a98_row2_col2" class="data row2 col2" >Side chain length</td>
          <td id="T_10a98_row2_col3" class="data row2 col3" >Side chain length</td>
          <td id="T_10a98_row2_col4" class="data row2 col4" >Radius of gy...evitt, 1976)</td>
          <td id="T_10a98_row2_col5" class="data row2 col5" >0.233000</td>
          <td id="T_10a98_row2_col6" class="data row2 col6" >0.137044</td>
          <td id="T_10a98_row2_col7" class="data row2 col7" >0.137044</td>
          <td id="T_10a98_row2_col8" class="data row2 col8" >0.161683</td>
          <td id="T_10a98_row2_col9" class="data row2 col9" >0.176964</td>
          <td id="T_10a98_row2_col10" class="data row2 col10" >0.000000</td>
          <td id="T_10a98_row2_col11" class="data row2 col11" >0.000001</td>
          <td id="T_10a98_row2_col12" class="data row2 col12" >32,33</td>
          <td id="T_10a98_row2_col13" class="data row2 col13" >1.554800</td>
          <td id="T_10a98_row2_col14" class="data row2 col14" >2.109848</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_10a98_row3_col0" class="data row3 col0" >TMD_C_JMD_C-...)-HUTJ700102</td>
          <td id="T_10a98_row3_col1" class="data row3 col1" >Energy</td>
          <td id="T_10a98_row3_col2" class="data row3 col2" >Entropy</td>
          <td id="T_10a98_row3_col3" class="data row3 col3" >Entropy</td>
          <td id="T_10a98_row3_col4" class="data row3 col4" >Absolute ent...chens, 1970)</td>
          <td id="T_10a98_row3_col5" class="data row3 col5" >0.229000</td>
          <td id="T_10a98_row3_col6" class="data row3 col6" >0.098224</td>
          <td id="T_10a98_row3_col7" class="data row3 col7" >0.098224</td>
          <td id="T_10a98_row3_col8" class="data row3 col8" >0.106865</td>
          <td id="T_10a98_row3_col9" class="data row3 col9" >0.124608</td>
          <td id="T_10a98_row3_col10" class="data row3 col10" >0.000000</td>
          <td id="T_10a98_row3_col11" class="data row3 col11" >0.000001</td>
          <td id="T_10a98_row3_col12" class="data row3 col12" >31,32,33,34,35</td>
          <td id="T_10a98_row3_col13" class="data row3 col13" >3.111200</td>
          <td id="T_10a98_row3_col14" class="data row3 col14" >3.109955</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_10a98_row4_col0" class="data row4 col0" >TMD_C_JMD_C-...)-RADA880106</td>
          <td id="T_10a98_row4_col1" class="data row4 col1" >ASA/Volume</td>
          <td id="T_10a98_row4_col2" class="data row4 col2" >Volume</td>
          <td id="T_10a98_row4_col3" class="data row4 col3" >Accessible s...e area (ASA)</td>
          <td id="T_10a98_row4_col4" class="data row4 col4" >Accessible s...enden, 1988)</td>
          <td id="T_10a98_row4_col5" class="data row4 col5" >0.223000</td>
          <td id="T_10a98_row4_col6" class="data row4 col6" >0.095071</td>
          <td id="T_10a98_row4_col7" class="data row4 col7" >0.095071</td>
          <td id="T_10a98_row4_col8" class="data row4 col8" >0.114758</td>
          <td id="T_10a98_row4_col9" class="data row4 col9" >0.132829</td>
          <td id="T_10a98_row4_col10" class="data row4 col10" >0.000000</td>
          <td id="T_10a98_row4_col11" class="data row4 col11" >0.000002</td>
          <td id="T_10a98_row4_col12" class="data row4 col12" >32,33</td>
          <td id="T_10a98_row4_col13" class="data row4 col13" >0.000000</td>
          <td id="T_10a98_row4_col14" class="data row4 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_10a98_row5_col0" class="data row5 col0" >TMD_C_JMD_C-...)-KLEP840101</td>
          <td id="T_10a98_row5_col1" class="data row5 col1" >Energy</td>
          <td id="T_10a98_row5_col2" class="data row5 col2" >Charge</td>
          <td id="T_10a98_row5_col3" class="data row5 col3" >Charge</td>
          <td id="T_10a98_row5_col4" class="data row5 col4" >Net charge (...t al., 1984)</td>
          <td id="T_10a98_row5_col5" class="data row5 col5" >0.222000</td>
          <td id="T_10a98_row5_col6" class="data row5 col6" >0.058671</td>
          <td id="T_10a98_row5_col7" class="data row5 col7" >0.058671</td>
          <td id="T_10a98_row5_col8" class="data row5 col8" >0.064895</td>
          <td id="T_10a98_row5_col9" class="data row5 col9" >0.069547</td>
          <td id="T_10a98_row5_col10" class="data row5 col10" >0.000000</td>
          <td id="T_10a98_row5_col11" class="data row5 col11" >0.000001</td>
          <td id="T_10a98_row5_col12" class="data row5 col12" >27,28,29,30,31,32,33</td>
          <td id="T_10a98_row5_col13" class="data row5 col13" >0.000000</td>
          <td id="T_10a98_row5_col14" class="data row5 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_10a98_row6_col0" class="data row6 col0" >TMD_C_JMD_C-...)-FAUJ880109</td>
          <td id="T_10a98_row6_col1" class="data row6 col1" >Energy</td>
          <td id="T_10a98_row6_col2" class="data row6 col2" >Isoelectric point</td>
          <td id="T_10a98_row6_col3" class="data row6 col3" >Number hydro... bond donors</td>
          <td id="T_10a98_row6_col4" class="data row6 col4" >Number of hy...t al., 1988)</td>
          <td id="T_10a98_row6_col5" class="data row6 col5" >0.215000</td>
          <td id="T_10a98_row6_col6" class="data row6 col6" >0.146661</td>
          <td id="T_10a98_row6_col7" class="data row6 col7" >0.146661</td>
          <td id="T_10a98_row6_col8" class="data row6 col8" >0.174609</td>
          <td id="T_10a98_row6_col9" class="data row6 col9" >0.188034</td>
          <td id="T_10a98_row6_col10" class="data row6 col10" >0.000000</td>
          <td id="T_10a98_row6_col11" class="data row6 col11" >0.000004</td>
          <td id="T_10a98_row6_col12" class="data row6 col12" >33,34,35,36</td>
          <td id="T_10a98_row6_col13" class="data row6 col13" >1.032400</td>
          <td id="T_10a98_row6_col14" class="data row6 col14" >1.510722</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_10a98_row7_col0" class="data row7 col0" >TMD_C_JMD_C-...)-JANJ780101</td>
          <td id="T_10a98_row7_col1" class="data row7 col1" >ASA/Volume</td>
          <td id="T_10a98_row7_col2" class="data row7 col2" >Accessible s...e area (ASA)</td>
          <td id="T_10a98_row7_col3" class="data row7 col3" >ASA (folded protein)</td>
          <td id="T_10a98_row7_col4" class="data row7 col4" >Average acce...t al., 1978)</td>
          <td id="T_10a98_row7_col5" class="data row7 col5" >0.215000</td>
          <td id="T_10a98_row7_col6" class="data row7 col6" >0.124317</td>
          <td id="T_10a98_row7_col7" class="data row7 col7" >0.124317</td>
          <td id="T_10a98_row7_col8" class="data row7 col8" >0.166309</td>
          <td id="T_10a98_row7_col9" class="data row7 col9" >0.153364</td>
          <td id="T_10a98_row7_col10" class="data row7 col10" >0.000000</td>
          <td id="T_10a98_row7_col11" class="data row7 col11" >0.000004</td>
          <td id="T_10a98_row7_col12" class="data row7 col12" >31,32,33,34,35</td>
          <td id="T_10a98_row7_col13" class="data row7 col13" >1.080400</td>
          <td id="T_10a98_row7_col14" class="data row7 col14" >1.296094</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_10a98_row8_col0" class="data row8 col0" >TMD_C_JMD_C-...)-WILM950103</td>
          <td id="T_10a98_row8_col1" class="data row8 col1" >Polarity</td>
          <td id="T_10a98_row8_col2" class="data row8 col2" >Hydrophobici... (interface)</td>
          <td id="T_10a98_row8_col3" class="data row8 col3" >Hydrophobici... (interface)</td>
          <td id="T_10a98_row8_col4" class="data row8 col4" >Hydrophobici...t al., 1995)</td>
          <td id="T_10a98_row8_col5" class="data row8 col5" >0.212000</td>
          <td id="T_10a98_row8_col6" class="data row8 col6" >0.141305</td>
          <td id="T_10a98_row8_col7" class="data row8 col7" >-0.141305</td>
          <td id="T_10a98_row8_col8" class="data row8 col8" >0.168603</td>
          <td id="T_10a98_row8_col9" class="data row8 col9" >0.217235</td>
          <td id="T_10a98_row8_col10" class="data row8 col10" >0.000000</td>
          <td id="T_10a98_row8_col11" class="data row8 col11" >0.000005</td>
          <td id="T_10a98_row8_col12" class="data row8 col12" >33,34</td>
          <td id="T_10a98_row8_col13" class="data row8 col13" >1.747200</td>
          <td id="T_10a98_row8_col14" class="data row8 col14" >2.150664</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_10a98_row9_col0" class="data row9 col0" >TMD_C_JMD_C-...)-AURR980110</td>
          <td id="T_10a98_row9_col1" class="data row9 col1" >Conformation</td>
          <td id="T_10a98_row9_col2" class="data row9 col2" >α-helix</td>
          <td id="T_10a98_row9_col3" class="data row9 col3" >α-helix (middle)</td>
          <td id="T_10a98_row9_col4" class="data row9 col4" >Normalized p...-Rose, 1998)</td>
          <td id="T_10a98_row9_col5" class="data row9 col5" >0.211000</td>
          <td id="T_10a98_row9_col6" class="data row9 col6" >0.125350</td>
          <td id="T_10a98_row9_col7" class="data row9 col7" >0.125350</td>
          <td id="T_10a98_row9_col8" class="data row9 col8" >0.160819</td>
          <td id="T_10a98_row9_col9" class="data row9 col9" >0.174121</td>
          <td id="T_10a98_row9_col10" class="data row9 col10" >0.000000</td>
          <td id="T_10a98_row9_col11" class="data row9 col11" >0.000005</td>
          <td id="T_10a98_row9_col12" class="data row9 col12" >32,33</td>
          <td id="T_10a98_row9_col13" class="data row9 col13" >1.788800</td>
          <td id="T_10a98_row9_col14" class="data row9 col14" >2.700803</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row10" class="row_heading level0 row10" >11</th>
          <td id="T_10a98_row10_col0" class="data row10 col0" >TMD_C_JMD_C-...)-AURR980110</td>
          <td id="T_10a98_row10_col1" class="data row10 col1" >Conformation</td>
          <td id="T_10a98_row10_col2" class="data row10 col2" >α-helix</td>
          <td id="T_10a98_row10_col3" class="data row10 col3" >α-helix (middle)</td>
          <td id="T_10a98_row10_col4" class="data row10 col4" >Normalized p...-Rose, 1998)</td>
          <td id="T_10a98_row10_col5" class="data row10 col5" >0.211000</td>
          <td id="T_10a98_row10_col6" class="data row10 col6" >0.077355</td>
          <td id="T_10a98_row10_col7" class="data row10 col7" >0.077355</td>
          <td id="T_10a98_row10_col8" class="data row10 col8" >0.102965</td>
          <td id="T_10a98_row10_col9" class="data row10 col9" >0.107453</td>
          <td id="T_10a98_row10_col10" class="data row10 col10" >0.000000</td>
          <td id="T_10a98_row10_col11" class="data row10 col11" >0.000005</td>
          <td id="T_10a98_row10_col12" class="data row10 col12" >27,28,29,30,31,32,33</td>
          <td id="T_10a98_row10_col13" class="data row10 col13" >3.048800</td>
          <td id="T_10a98_row10_col14" class="data row10 col14" >3.623912</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row11" class="row_heading level0 row11" >12</th>
          <td id="T_10a98_row11_col0" class="data row11 col0" >TMD_C_JMD_C-...)-JANJ790102</td>
          <td id="T_10a98_row11_col1" class="data row11 col1" >Energy</td>
          <td id="T_10a98_row11_col2" class="data row11 col2" >Free energy (unfolding)</td>
          <td id="T_10a98_row11_col3" class="data row11 col3" >Transfer fre...E) to inside</td>
          <td id="T_10a98_row11_col4" class="data row11 col4" >Transfer fre...Janin, 1979)</td>
          <td id="T_10a98_row11_col5" class="data row11 col5" >0.206000</td>
          <td id="T_10a98_row11_col6" class="data row11 col6" >0.111462</td>
          <td id="T_10a98_row11_col7" class="data row11 col7" >-0.111462</td>
          <td id="T_10a98_row11_col8" class="data row11 col8" >0.159718</td>
          <td id="T_10a98_row11_col9" class="data row11 col9" >0.144989</td>
          <td id="T_10a98_row11_col10" class="data row11 col10" >0.000000</td>
          <td id="T_10a98_row11_col11" class="data row11 col11" >0.000009</td>
          <td id="T_10a98_row11_col12" class="data row11 col12" >31,32,33,34,35</td>
          <td id="T_10a98_row11_col13" class="data row11 col13" >0.000000</td>
          <td id="T_10a98_row11_col14" class="data row11 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row12" class="row_heading level0 row12" >13</th>
          <td id="T_10a98_row12_col0" class="data row12 col0" >TMD_C_JMD_C-...)-CHOC760103</td>
          <td id="T_10a98_row12_col1" class="data row12 col1" >ASA/Volume</td>
          <td id="T_10a98_row12_col2" class="data row12 col2" >Buried</td>
          <td id="T_10a98_row12_col3" class="data row12 col3" >Buried</td>
          <td id="T_10a98_row12_col4" class="data row12 col4" >Proportion o...othia, 1976)</td>
          <td id="T_10a98_row12_col5" class="data row12 col5" >0.205000</td>
          <td id="T_10a98_row12_col6" class="data row12 col6" >0.125868</td>
          <td id="T_10a98_row12_col7" class="data row12 col7" >-0.125868</td>
          <td id="T_10a98_row12_col8" class="data row12 col8" >0.172165</td>
          <td id="T_10a98_row12_col9" class="data row12 col9" >0.188333</td>
          <td id="T_10a98_row12_col10" class="data row12 col10" >0.000000</td>
          <td id="T_10a98_row12_col11" class="data row12 col11" >0.000009</td>
          <td id="T_10a98_row12_col12" class="data row12 col12" >32,33</td>
          <td id="T_10a98_row12_col13" class="data row12 col13" >0.000000</td>
          <td id="T_10a98_row12_col14" class="data row12 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row13" class="row_heading level0 row13" >14</th>
          <td id="T_10a98_row13_col0" class="data row13 col0" >TMD_C_JMD_C-...)-LEVM760105</td>
          <td id="T_10a98_row13_col1" class="data row13 col1" >Shape</td>
          <td id="T_10a98_row13_col2" class="data row13 col2" >Side chain length</td>
          <td id="T_10a98_row13_col3" class="data row13 col3" >Side chain length</td>
          <td id="T_10a98_row13_col4" class="data row13 col4" >Radius of gy...evitt, 1976)</td>
          <td id="T_10a98_row13_col5" class="data row13 col5" >0.204000</td>
          <td id="T_10a98_row13_col6" class="data row13 col6" >0.105513</td>
          <td id="T_10a98_row13_col7" class="data row13 col7" >0.105513</td>
          <td id="T_10a98_row13_col8" class="data row13 col8" >0.132849</td>
          <td id="T_10a98_row13_col9" class="data row13 col9" >0.145219</td>
          <td id="T_10a98_row13_col10" class="data row13 col10" >0.000000</td>
          <td id="T_10a98_row13_col11" class="data row13 col11" >0.000009</td>
          <td id="T_10a98_row13_col12" class="data row13 col12" >33,34,35,36</td>
          <td id="T_10a98_row13_col13" class="data row13 col13" >1.992000</td>
          <td id="T_10a98_row13_col14" class="data row13 col14" >2.929460</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row14" class="row_heading level0 row14" >15</th>
          <td id="T_10a98_row14_col0" class="data row14 col0" >TMD_C_JMD_C-...)-DESM900102</td>
          <td id="T_10a98_row14_col1" class="data row14 col1" >Polarity</td>
          <td id="T_10a98_row14_col2" class="data row14 col2" >Amphiphilicity (α-helix)</td>
          <td id="T_10a98_row14_col3" class="data row14 col3" >Membrane preference</td>
          <td id="T_10a98_row14_col4" class="data row14 col4" >Average memb...t al., 1990)</td>
          <td id="T_10a98_row14_col5" class="data row14 col5" >0.200000</td>
          <td id="T_10a98_row14_col6" class="data row14 col6" >0.132693</td>
          <td id="T_10a98_row14_col7" class="data row14 col7" >-0.132693</td>
          <td id="T_10a98_row14_col8" class="data row14 col8" >0.184359</td>
          <td id="T_10a98_row14_col9" class="data row14 col9" >0.209008</td>
          <td id="T_10a98_row14_col10" class="data row14 col10" >0.000000</td>
          <td id="T_10a98_row14_col11" class="data row14 col11" >0.000015</td>
          <td id="T_10a98_row14_col12" class="data row14 col12" >32,33</td>
          <td id="T_10a98_row14_col13" class="data row14 col13" >0.000000</td>
          <td id="T_10a98_row14_col14" class="data row14 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row15" class="row_heading level0 row15" >16</th>
          <td id="T_10a98_row15_col0" class="data row15 col0" >TMD_C_JMD_C-...)-PRAM820102</td>
          <td id="T_10a98_row15_col1" class="data row15 col1" >Shape</td>
          <td id="T_10a98_row15_col2" class="data row15 col2" >Shape and Surface</td>
          <td id="T_10a98_row15_col3" class="data row15 col3" >Slope in Regression</td>
          <td id="T_10a98_row15_col4" class="data row15 col4" >Slope in Reg...swamy, 1982)</td>
          <td id="T_10a98_row15_col5" class="data row15 col5" >0.199000</td>
          <td id="T_10a98_row15_col6" class="data row15 col6" >0.073023</td>
          <td id="T_10a98_row15_col7" class="data row15 col7" >-0.073023</td>
          <td id="T_10a98_row15_col8" class="data row15 col8" >0.087336</td>
          <td id="T_10a98_row15_col9" class="data row15 col9" >0.107750</td>
          <td id="T_10a98_row15_col10" class="data row15 col10" >0.000000</td>
          <td id="T_10a98_row15_col11" class="data row15 col11" >0.000017</td>
          <td id="T_10a98_row15_col12" class="data row15 col12" >31,32,33,34,35</td>
          <td id="T_10a98_row15_col13" class="data row15 col13" >0.616000</td>
          <td id="T_10a98_row15_col14" class="data row15 col14" >0.847660</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row16" class="row_heading level0 row16" >17</th>
          <td id="T_10a98_row16_col0" class="data row16 col0" >TMD_C_JMD_C-...)-CHOP780212</td>
          <td id="T_10a98_row16_col1" class="data row16 col1" >Conformation</td>
          <td id="T_10a98_row16_col2" class="data row16 col2" >β-sheet (C-term)</td>
          <td id="T_10a98_row16_col3" class="data row16 col3" >β-turn (1st residue)</td>
          <td id="T_10a98_row16_col4" class="data row16 col4" >Frequency of...sman, 1978b)</td>
          <td id="T_10a98_row16_col5" class="data row16 col5" >0.199000</td>
          <td id="T_10a98_row16_col6" class="data row16 col6" >0.065983</td>
          <td id="T_10a98_row16_col7" class="data row16 col7" >-0.065983</td>
          <td id="T_10a98_row16_col8" class="data row16 col8" >0.087814</td>
          <td id="T_10a98_row16_col9" class="data row16 col9" >0.105835</td>
          <td id="T_10a98_row16_col10" class="data row16 col10" >0.000000</td>
          <td id="T_10a98_row16_col11" class="data row16 col11" >0.000016</td>
          <td id="T_10a98_row16_col12" class="data row16 col12" >27,28,29,30,31,32,33</td>
          <td id="T_10a98_row16_col13" class="data row16 col13" >4.106000</td>
          <td id="T_10a98_row16_col14" class="data row16 col14" >5.236574</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row17" class="row_heading level0 row17" >18</th>
          <td id="T_10a98_row17_col0" class="data row17 col0" >TMD_C_JMD_C-...)-RICJ880113</td>
          <td id="T_10a98_row17_col1" class="data row17 col1" >Conformation</td>
          <td id="T_10a98_row17_col2" class="data row17 col2" >α-helix (C-cap)</td>
          <td id="T_10a98_row17_col3" class="data row17 col3" >α-helix (C-t...nal, inside)</td>
          <td id="T_10a98_row17_col4" class="data row17 col4" >Relative pre...rdson, 1988)</td>
          <td id="T_10a98_row17_col5" class="data row17 col5" >0.198000</td>
          <td id="T_10a98_row17_col6" class="data row17 col6" >0.138293</td>
          <td id="T_10a98_row17_col7" class="data row17 col7" >0.138293</td>
          <td id="T_10a98_row17_col8" class="data row17 col8" >0.172194</td>
          <td id="T_10a98_row17_col9" class="data row17 col9" >0.198814</td>
          <td id="T_10a98_row17_col10" class="data row17 col10" >0.000000</td>
          <td id="T_10a98_row17_col11" class="data row17 col11" >0.000017</td>
          <td id="T_10a98_row17_col12" class="data row17 col12" >32,33</td>
          <td id="T_10a98_row17_col13" class="data row17 col13" >0.832400</td>
          <td id="T_10a98_row17_col14" class="data row17 col14" >1.383718</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row18" class="row_heading level0 row18" >19</th>
          <td id="T_10a98_row18_col0" class="data row18 col0" >TMD_C_JMD_C-...)-RADA880104</td>
          <td id="T_10a98_row18_col1" class="data row18 col1" >Energy</td>
          <td id="T_10a98_row18_col2" class="data row18 col2" >Free energy (unfolding)</td>
          <td id="T_10a98_row18_col3" class="data row18 col3" >Transfer fre...E) to inside</td>
          <td id="T_10a98_row18_col4" class="data row18 col4" >Transfer fre...enden, 1988)</td>
          <td id="T_10a98_row18_col5" class="data row18 col5" >0.197000</td>
          <td id="T_10a98_row18_col6" class="data row18 col6" >0.060758</td>
          <td id="T_10a98_row18_col7" class="data row18 col7" >0.060758</td>
          <td id="T_10a98_row18_col8" class="data row18 col8" >0.050818</td>
          <td id="T_10a98_row18_col9" class="data row18 col9" >0.095267</td>
          <td id="T_10a98_row18_col10" class="data row18 col10" >0.000000</td>
          <td id="T_10a98_row18_col11" class="data row18 col11" >0.000019</td>
          <td id="T_10a98_row18_col12" class="data row18 col12" >25,28</td>
          <td id="T_10a98_row18_col13" class="data row18 col13" >1.658800</td>
          <td id="T_10a98_row18_col14" class="data row18 col14" >3.421774</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row19" class="row_heading level0 row19" >20</th>
          <td id="T_10a98_row19_col0" class="data row19 col0" >JMD_N_TMD_N-...)-KARP850101</td>
          <td id="T_10a98_row19_col1" class="data row19 col1" >Structure-Activity</td>
          <td id="T_10a98_row19_col2" class="data row19 col2" >Flexibility</td>
          <td id="T_10a98_row19_col3" class="data row19 col3" >Flexibility ...d neighbors)</td>
          <td id="T_10a98_row19_col4" class="data row19 col4" >Flexibility ...chulz, 1985)</td>
          <td id="T_10a98_row19_col5" class="data row19 col5" >0.196000</td>
          <td id="T_10a98_row19_col6" class="data row19 col6" >0.062671</td>
          <td id="T_10a98_row19_col7" class="data row19 col7" >0.062671</td>
          <td id="T_10a98_row19_col8" class="data row19 col8" >0.083456</td>
          <td id="T_10a98_row19_col9" class="data row19 col9" >0.090427</td>
          <td id="T_10a98_row19_col10" class="data row19 col10" >0.000000</td>
          <td id="T_10a98_row19_col11" class="data row19 col11" >0.000023</td>
          <td id="T_10a98_row19_col12" class="data row19 col12" >1,2,3,4,5,6,7,8,9,10</td>
          <td id="T_10a98_row19_col13" class="data row19 col13" >1.574400</td>
          <td id="T_10a98_row19_col14" class="data row19 col14" >1.835403</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row20" class="row_heading level0 row20" >21</th>
          <td id="T_10a98_row20_col0" class="data row20 col0" >TMD-Pattern(...)-RADA880104</td>
          <td id="T_10a98_row20_col1" class="data row20 col1" >Energy</td>
          <td id="T_10a98_row20_col2" class="data row20 col2" >Free energy (unfolding)</td>
          <td id="T_10a98_row20_col3" class="data row20 col3" >Transfer fre...E) to inside</td>
          <td id="T_10a98_row20_col4" class="data row20 col4" >Transfer fre...enden, 1988)</td>
          <td id="T_10a98_row20_col5" class="data row20 col5" >0.195000</td>
          <td id="T_10a98_row20_col6" class="data row20 col6" >0.060096</td>
          <td id="T_10a98_row20_col7" class="data row20 col7" >0.060096</td>
          <td id="T_10a98_row20_col8" class="data row20 col8" >0.050818</td>
          <td id="T_10a98_row20_col9" class="data row20 col9" >0.095039</td>
          <td id="T_10a98_row20_col10" class="data row20 col10" >0.000000</td>
          <td id="T_10a98_row20_col11" class="data row20 col11" >0.000023</td>
          <td id="T_10a98_row20_col12" class="data row20 col12" >24,27</td>
          <td id="T_10a98_row20_col13" class="data row20 col13" >0.000000</td>
          <td id="T_10a98_row20_col14" class="data row20 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row21" class="row_heading level0 row21" >22</th>
          <td id="T_10a98_row21_col0" class="data row21 col0" >TMD_C_JMD_C-...)-RADA880106</td>
          <td id="T_10a98_row21_col1" class="data row21 col1" >ASA/Volume</td>
          <td id="T_10a98_row21_col2" class="data row21 col2" >Volume</td>
          <td id="T_10a98_row21_col3" class="data row21 col3" >Accessible s...e area (ASA)</td>
          <td id="T_10a98_row21_col4" class="data row21 col4" >Accessible s...enden, 1988)</td>
          <td id="T_10a98_row21_col5" class="data row21 col5" >0.193000</td>
          <td id="T_10a98_row21_col6" class="data row21 col6" >0.076770</td>
          <td id="T_10a98_row21_col7" class="data row21 col7" >0.076770</td>
          <td id="T_10a98_row21_col8" class="data row21 col8" >0.092804</td>
          <td id="T_10a98_row21_col9" class="data row21 col9" >0.114150</td>
          <td id="T_10a98_row21_col10" class="data row21 col10" >0.000000</td>
          <td id="T_10a98_row21_col11" class="data row21 col11" >0.000027</td>
          <td id="T_10a98_row21_col12" class="data row21 col12" >33,34,35,36</td>
          <td id="T_10a98_row21_col13" class="data row21 col13" >0.000000</td>
          <td id="T_10a98_row21_col14" class="data row21 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row22" class="row_heading level0 row22" >23</th>
          <td id="T_10a98_row22_col0" class="data row22 col0" >TMD_C_JMD_C-...)-FAUJ880109</td>
          <td id="T_10a98_row22_col1" class="data row22 col1" >Energy</td>
          <td id="T_10a98_row22_col2" class="data row22 col2" >Isoelectric point</td>
          <td id="T_10a98_row22_col3" class="data row22 col3" >Number hydro... bond donors</td>
          <td id="T_10a98_row22_col4" class="data row22 col4" >Number of hy...t al., 1988)</td>
          <td id="T_10a98_row22_col5" class="data row22 col5" >0.192000</td>
          <td id="T_10a98_row22_col6" class="data row22 col6" >0.125521</td>
          <td id="T_10a98_row22_col7" class="data row22 col7" >0.125521</td>
          <td id="T_10a98_row22_col8" class="data row22 col8" >0.188795</td>
          <td id="T_10a98_row22_col9" class="data row22 col9" >0.177163</td>
          <td id="T_10a98_row22_col10" class="data row22 col10" >0.000000</td>
          <td id="T_10a98_row22_col11" class="data row22 col11" >0.000027</td>
          <td id="T_10a98_row22_col12" class="data row22 col12" >31,32,33</td>
          <td id="T_10a98_row22_col13" class="data row22 col13" >0.000000</td>
          <td id="T_10a98_row22_col14" class="data row22 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row23" class="row_heading level0 row23" >24</th>
          <td id="T_10a98_row23_col0" class="data row23 col0" >TMD_C_JMD_C-...)-LIFS790102</td>
          <td id="T_10a98_row23_col1" class="data row23 col1" >Conformation</td>
          <td id="T_10a98_row23_col2" class="data row23 col2" >β-strand</td>
          <td id="T_10a98_row23_col3" class="data row23 col3" >β-strand</td>
          <td id="T_10a98_row23_col4" class="data row23 col4" >Conformation...ander, 1979)</td>
          <td id="T_10a98_row23_col5" class="data row23 col5" >0.189000</td>
          <td id="T_10a98_row23_col6" class="data row23 col6" >0.125674</td>
          <td id="T_10a98_row23_col7" class="data row23 col7" >0.125674</td>
          <td id="T_10a98_row23_col8" class="data row23 col8" >0.183876</td>
          <td id="T_10a98_row23_col9" class="data row23 col9" >0.218813</td>
          <td id="T_10a98_row23_col10" class="data row23 col10" >0.000001</td>
          <td id="T_10a98_row23_col11" class="data row23 col11" >0.000039</td>
          <td id="T_10a98_row23_col12" class="data row23 col12" >28,29</td>
          <td id="T_10a98_row23_col13" class="data row23 col13" >4.729200</td>
          <td id="T_10a98_row23_col14" class="data row23 col14" >4.776785</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row24" class="row_heading level0 row24" >25</th>
          <td id="T_10a98_row24_col0" class="data row24 col0" >TMD_C_JMD_C-...)-TANS770106</td>
          <td id="T_10a98_row24_col1" class="data row24 col1" >Conformation</td>
          <td id="T_10a98_row24_col2" class="data row24 col2" >β-turn (TM helix)</td>
          <td id="T_10a98_row24_col3" class="data row24 col3" >β-turn in double bend</td>
          <td id="T_10a98_row24_col4" class="data row24 col4" >Normalized f...eraga, 1977)</td>
          <td id="T_10a98_row24_col5" class="data row24 col5" >0.189000</td>
          <td id="T_10a98_row24_col6" class="data row24 col6" >0.093759</td>
          <td id="T_10a98_row24_col7" class="data row24 col7" >0.093759</td>
          <td id="T_10a98_row24_col8" class="data row24 col8" >0.136715</td>
          <td id="T_10a98_row24_col9" class="data row24 col9" >0.137320</td>
          <td id="T_10a98_row24_col10" class="data row24 col10" >0.000001</td>
          <td id="T_10a98_row24_col11" class="data row24 col11" >0.000039</td>
          <td id="T_10a98_row24_col12" class="data row24 col12" >32,33</td>
          <td id="T_10a98_row24_col13" class="data row24 col13" >0.000000</td>
          <td id="T_10a98_row24_col14" class="data row24 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row25" class="row_heading level0 row25" >26</th>
          <td id="T_10a98_row25_col0" class="data row25 col0" >TMD_C_JMD_C-...)-JANJ790102</td>
          <td id="T_10a98_row25_col1" class="data row25 col1" >Energy</td>
          <td id="T_10a98_row25_col2" class="data row25 col2" >Free energy (unfolding)</td>
          <td id="T_10a98_row25_col3" class="data row25 col3" >Transfer fre...E) to inside</td>
          <td id="T_10a98_row25_col4" class="data row25 col4" >Transfer fre...Janin, 1979)</td>
          <td id="T_10a98_row25_col5" class="data row25 col5" >0.187000</td>
          <td id="T_10a98_row25_col6" class="data row25 col6" >0.144354</td>
          <td id="T_10a98_row25_col7" class="data row25 col7" >-0.144354</td>
          <td id="T_10a98_row25_col8" class="data row25 col8" >0.181777</td>
          <td id="T_10a98_row25_col9" class="data row25 col9" >0.233103</td>
          <td id="T_10a98_row25_col10" class="data row25 col10" >0.000001</td>
          <td id="T_10a98_row25_col11" class="data row25 col11" >0.000049</td>
          <td id="T_10a98_row25_col12" class="data row25 col12" >33,37</td>
          <td id="T_10a98_row25_col13" class="data row25 col13" >2.833600</td>
          <td id="T_10a98_row25_col14" class="data row25 col14" >3.640617</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row26" class="row_heading level0 row26" >27</th>
          <td id="T_10a98_row26_col0" class="data row26 col0" >TMD_C_JMD_C-...)-CHOC760103</td>
          <td id="T_10a98_row26_col1" class="data row26 col1" >ASA/Volume</td>
          <td id="T_10a98_row26_col2" class="data row26 col2" >Buried</td>
          <td id="T_10a98_row26_col3" class="data row26 col3" >Buried</td>
          <td id="T_10a98_row26_col4" class="data row26 col4" >Proportion o...othia, 1976)</td>
          <td id="T_10a98_row26_col5" class="data row26 col5" >0.185000</td>
          <td id="T_10a98_row26_col6" class="data row26 col6" >0.105474</td>
          <td id="T_10a98_row26_col7" class="data row26 col7" >-0.105474</td>
          <td id="T_10a98_row26_col8" class="data row26 col8" >0.157535</td>
          <td id="T_10a98_row26_col9" class="data row26 col9" >0.163039</td>
          <td id="T_10a98_row26_col10" class="data row26 col10" >0.000001</td>
          <td id="T_10a98_row26_col11" class="data row26 col11" >0.000059</td>
          <td id="T_10a98_row26_col12" class="data row26 col12" >33,34,35,36</td>
          <td id="T_10a98_row26_col13" class="data row26 col13" >0.000000</td>
          <td id="T_10a98_row26_col14" class="data row26 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row27" class="row_heading level0 row27" >28</th>
          <td id="T_10a98_row27_col0" class="data row27 col0" >TMD_C_JMD_C-...)-MITS020101</td>
          <td id="T_10a98_row27_col1" class="data row27 col1" >Polarity</td>
          <td id="T_10a98_row27_col2" class="data row27 col2" >Amphiphilicity</td>
          <td id="T_10a98_row27_col3" class="data row27 col3" >Amphiphilicity</td>
          <td id="T_10a98_row27_col4" class="data row27 col4" >Amphiphilici...t al., 2002)</td>
          <td id="T_10a98_row27_col5" class="data row27 col5" >0.185000</td>
          <td id="T_10a98_row27_col6" class="data row27 col6" >0.101798</td>
          <td id="T_10a98_row27_col7" class="data row27 col7" >0.101798</td>
          <td id="T_10a98_row27_col8" class="data row27 col8" >0.145676</td>
          <td id="T_10a98_row27_col9" class="data row27 col9" >0.155096</td>
          <td id="T_10a98_row27_col10" class="data row27 col10" >0.000001</td>
          <td id="T_10a98_row27_col11" class="data row27 col11" >0.000054</td>
          <td id="T_10a98_row27_col12" class="data row27 col12" >32,33</td>
          <td id="T_10a98_row27_col13" class="data row27 col13" >0.000000</td>
          <td id="T_10a98_row27_col14" class="data row27 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row28" class="row_heading level0 row28" >29</th>
          <td id="T_10a98_row28_col0" class="data row28 col0" >JMD_N_TMD_N-...)-AURR980116</td>
          <td id="T_10a98_row28_col1" class="data row28 col1" >Conformation</td>
          <td id="T_10a98_row28_col2" class="data row28 col2" >α-helix (C-cap)</td>
          <td id="T_10a98_row28_col3" class="data row28 col3" >α-helix (C-t...inal, C-cap)</td>
          <td id="T_10a98_row28_col4" class="data row28 col4" >Normalized p...-Rose, 1998)</td>
          <td id="T_10a98_row28_col5" class="data row28 col5" >0.184000</td>
          <td id="T_10a98_row28_col6" class="data row28 col6" >0.112728</td>
          <td id="T_10a98_row28_col7" class="data row28 col7" >-0.112728</td>
          <td id="T_10a98_row28_col8" class="data row28 col8" >0.166431</td>
          <td id="T_10a98_row28_col9" class="data row28 col9" >0.183800</td>
          <td id="T_10a98_row28_col10" class="data row28 col10" >0.000001</td>
          <td id="T_10a98_row28_col11" class="data row28 col11" >0.000061</td>
          <td id="T_10a98_row28_col12" class="data row28 col12" >11,15</td>
          <td id="T_10a98_row28_col13" class="data row28 col13" >0.857600</td>
          <td id="T_10a98_row28_col14" class="data row28 col14" >1.339550</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row29" class="row_heading level0 row29" >30</th>
          <td id="T_10a98_row29_col0" class="data row29 col0" >TMD_C_JMD_C-...)-FINA910104</td>
          <td id="T_10a98_row29_col1" class="data row29 col1" >Conformation</td>
          <td id="T_10a98_row29_col2" class="data row29 col2" >α-helix (C-cap)</td>
          <td id="T_10a98_row29_col3" class="data row29 col3" >α-helix termination</td>
          <td id="T_10a98_row29_col4" class="data row29 col4" >Helix termin...t al., 1991)</td>
          <td id="T_10a98_row29_col5" class="data row29 col5" >0.184000</td>
          <td id="T_10a98_row29_col6" class="data row29 col6" >0.062096</td>
          <td id="T_10a98_row29_col7" class="data row29 col7" >0.062096</td>
          <td id="T_10a98_row29_col8" class="data row29 col8" >0.078809</td>
          <td id="T_10a98_row29_col9" class="data row29 col9" >0.091271</td>
          <td id="T_10a98_row29_col10" class="data row29 col10" >0.000000</td>
          <td id="T_10a98_row29_col11" class="data row29 col11" >0.000017</td>
          <td id="T_10a98_row29_col12" class="data row29 col12" >26,30,33</td>
          <td id="T_10a98_row29_col13" class="data row29 col13" >0.147200</td>
          <td id="T_10a98_row29_col14" class="data row29 col14" >0.345306</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row30" class="row_heading level0 row30" >31</th>
          <td id="T_10a98_row30_col0" class="data row30 col0" >JMD_N_TMD_N-...)-ZHOH040101</td>
          <td id="T_10a98_row30_col1" class="data row30 col1" >Structure-Activity</td>
          <td id="T_10a98_row30_col2" class="data row30 col2" >Stability</td>
          <td id="T_10a98_row30_col3" class="data row30 col3" >Stability</td>
          <td id="T_10a98_row30_col4" class="data row30 col4" >The stabilit...-Zhou, 2004)</td>
          <td id="T_10a98_row30_col5" class="data row30 col5" >0.183000</td>
          <td id="T_10a98_row30_col6" class="data row30 col6" >0.063902</td>
          <td id="T_10a98_row30_col7" class="data row30 col7" >-0.063902</td>
          <td id="T_10a98_row30_col8" class="data row30 col8" >0.090842</td>
          <td id="T_10a98_row30_col9" class="data row30 col9" >0.101427</td>
          <td id="T_10a98_row30_col10" class="data row30 col10" >0.000002</td>
          <td id="T_10a98_row30_col11" class="data row30 col11" >0.000068</td>
          <td id="T_10a98_row30_col12" class="data row30 col12" >6,7,8,9,10</td>
          <td id="T_10a98_row30_col13" class="data row30 col13" >0.823200</td>
          <td id="T_10a98_row30_col14" class="data row30 col14" >1.404583</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row31" class="row_heading level0 row31" >32</th>
          <td id="T_10a98_row31_col0" class="data row31 col0" >TMD_C_JMD_C-...)-RICJ880113</td>
          <td id="T_10a98_row31_col1" class="data row31 col1" >Conformation</td>
          <td id="T_10a98_row31_col2" class="data row31 col2" >α-helix (C-cap)</td>
          <td id="T_10a98_row31_col3" class="data row31 col3" >α-helix (C-t...nal, inside)</td>
          <td id="T_10a98_row31_col4" class="data row31 col4" >Relative pre...rdson, 1988)</td>
          <td id="T_10a98_row31_col5" class="data row31 col5" >0.182000</td>
          <td id="T_10a98_row31_col6" class="data row31 col6" >0.121315</td>
          <td id="T_10a98_row31_col7" class="data row31 col7" >0.121315</td>
          <td id="T_10a98_row31_col8" class="data row31 col8" >0.147184</td>
          <td id="T_10a98_row31_col9" class="data row31 col9" >0.184212</td>
          <td id="T_10a98_row31_col10" class="data row31 col10" >0.000002</td>
          <td id="T_10a98_row31_col11" class="data row31 col11" >0.000070</td>
          <td id="T_10a98_row31_col12" class="data row31 col12" >33,34,35,36</td>
          <td id="T_10a98_row31_col13" class="data row31 col13" >0.865200</td>
          <td id="T_10a98_row31_col14" class="data row31 col14" >1.553379</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row32" class="row_heading level0 row32" >33</th>
          <td id="T_10a98_row32_col0" class="data row32 col0" >TMD-Pattern(...)-ANDN920101</td>
          <td id="T_10a98_row32_col1" class="data row32 col1" >Structure-Activity</td>
          <td id="T_10a98_row32_col2" class="data row32 col2" >Backbone-dynamics (-CH)</td>
          <td id="T_10a98_row32_col3" class="data row32 col3" >α-CH chemica...ne-dynamics)</td>
          <td id="T_10a98_row32_col4" class="data row32 col4" >alpha-CH che...t al., 1992)</td>
          <td id="T_10a98_row32_col5" class="data row32 col5" >0.182000</td>
          <td id="T_10a98_row32_col6" class="data row32 col6" >0.098529</td>
          <td id="T_10a98_row32_col7" class="data row32 col7" >-0.098529</td>
          <td id="T_10a98_row32_col8" class="data row32 col8" >0.141641</td>
          <td id="T_10a98_row32_col9" class="data row32 col9" >0.162412</td>
          <td id="T_10a98_row32_col10" class="data row32 col10" >0.000002</td>
          <td id="T_10a98_row32_col11" class="data row32 col11" >0.000072</td>
          <td id="T_10a98_row32_col12" class="data row32 col12" >16,20,24,28</td>
          <td id="T_10a98_row32_col13" class="data row32 col13" >0.221200</td>
          <td id="T_10a98_row32_col14" class="data row32 col14" >0.519240</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row33" class="row_heading level0 row33" >34</th>
          <td id="T_10a98_row33_col0" class="data row33 col0" >TMD-Pattern(...)-LIFS790102</td>
          <td id="T_10a98_row33_col1" class="data row33 col1" >Conformation</td>
          <td id="T_10a98_row33_col2" class="data row33 col2" >β-strand</td>
          <td id="T_10a98_row33_col3" class="data row33 col3" >β-strand</td>
          <td id="T_10a98_row33_col4" class="data row33 col4" >Conformation...ander, 1979)</td>
          <td id="T_10a98_row33_col5" class="data row33 col5" >0.182000</td>
          <td id="T_10a98_row33_col6" class="data row33 col6" >0.096246</td>
          <td id="T_10a98_row33_col7" class="data row33 col7" >0.096246</td>
          <td id="T_10a98_row33_col8" class="data row33 col8" >0.160859</td>
          <td id="T_10a98_row33_col9" class="data row33 col9" >0.159538</td>
          <td id="T_10a98_row33_col10" class="data row33 col10" >0.000002</td>
          <td id="T_10a98_row33_col11" class="data row33 col11" >0.000070</td>
          <td id="T_10a98_row33_col12" class="data row33 col12" >16,20,24,28</td>
          <td id="T_10a98_row33_col13" class="data row33 col13" >0.508400</td>
          <td id="T_10a98_row33_col14" class="data row33 col14" >0.738667</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row34" class="row_heading level0 row34" >35</th>
          <td id="T_10a98_row34_col0" class="data row34 col0" >JMD_N_TMD_N-...)-CIDH920102</td>
          <td id="T_10a98_row34_col1" class="data row34 col1" >Polarity</td>
          <td id="T_10a98_row34_col2" class="data row34 col2" >Hydrophobicity</td>
          <td id="T_10a98_row34_col3" class="data row34 col3" >Hydrophobicity</td>
          <td id="T_10a98_row34_col4" class="data row34 col4" >Normalized h...t al., 1992)</td>
          <td id="T_10a98_row34_col5" class="data row34 col5" >0.182000</td>
          <td id="T_10a98_row34_col6" class="data row34 col6" >0.066394</td>
          <td id="T_10a98_row34_col7" class="data row34 col7" >-0.066394</td>
          <td id="T_10a98_row34_col8" class="data row34 col8" >0.097857</td>
          <td id="T_10a98_row34_col9" class="data row34 col9" >0.103426</td>
          <td id="T_10a98_row34_col10" class="data row34 col10" >0.000002</td>
          <td id="T_10a98_row34_col11" class="data row34 col11" >0.000070</td>
          <td id="T_10a98_row34_col12" class="data row34 col12" >6,7,8,9,10</td>
          <td id="T_10a98_row34_col13" class="data row34 col13" >0.000000</td>
          <td id="T_10a98_row34_col14" class="data row34 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row35" class="row_heading level0 row35" >36</th>
          <td id="T_10a98_row35_col0" class="data row35 col0" >TMD_C_JMD_C-...)-JANJ780101</td>
          <td id="T_10a98_row35_col1" class="data row35 col1" >ASA/Volume</td>
          <td id="T_10a98_row35_col2" class="data row35 col2" >Accessible s...e area (ASA)</td>
          <td id="T_10a98_row35_col3" class="data row35 col3" >ASA (folded protein)</td>
          <td id="T_10a98_row35_col4" class="data row35 col4" >Average acce...t al., 1978)</td>
          <td id="T_10a98_row35_col5" class="data row35 col5" >0.182000</td>
          <td id="T_10a98_row35_col6" class="data row35 col6" >0.063819</td>
          <td id="T_10a98_row35_col7" class="data row35 col7" >0.063819</td>
          <td id="T_10a98_row35_col8" class="data row35 col8" >0.101691</td>
          <td id="T_10a98_row35_col9" class="data row35 col9" >0.105987</td>
          <td id="T_10a98_row35_col10" class="data row35 col10" >0.000002</td>
          <td id="T_10a98_row35_col11" class="data row35 col11" >0.000071</td>
          <td id="T_10a98_row35_col12" class="data row35 col12" >27,28,29,30,31,32,33</td>
          <td id="T_10a98_row35_col13" class="data row35 col13" >0.000000</td>
          <td id="T_10a98_row35_col14" class="data row35 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row36" class="row_heading level0 row36" >37</th>
          <td id="T_10a98_row36_col0" class="data row36 col0" >TMD-Pattern(...)-AURR980116</td>
          <td id="T_10a98_row36_col1" class="data row36 col1" >Conformation</td>
          <td id="T_10a98_row36_col2" class="data row36 col2" >α-helix (C-cap)</td>
          <td id="T_10a98_row36_col3" class="data row36 col3" >α-helix (C-t...inal, C-cap)</td>
          <td id="T_10a98_row36_col4" class="data row36 col4" >Normalized p...-Rose, 1998)</td>
          <td id="T_10a98_row36_col5" class="data row36 col5" >0.181000</td>
          <td id="T_10a98_row36_col6" class="data row36 col6" >0.118349</td>
          <td id="T_10a98_row36_col7" class="data row36 col7" >-0.118349</td>
          <td id="T_10a98_row36_col8" class="data row36 col8" >0.169282</td>
          <td id="T_10a98_row36_col9" class="data row36 col9" >0.185522</td>
          <td id="T_10a98_row36_col10" class="data row36 col10" >0.000002</td>
          <td id="T_10a98_row36_col11" class="data row36 col11" >0.000078</td>
          <td id="T_10a98_row36_col12" class="data row36 col12" >14,17</td>
          <td id="T_10a98_row36_col13" class="data row36 col13" >1.226400</td>
          <td id="T_10a98_row36_col14" class="data row36 col14" >1.510986</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row37" class="row_heading level0 row37" >38</th>
          <td id="T_10a98_row37_col0" class="data row37 col0" >TMD_C_JMD_C-...)-QIAN880134</td>
          <td id="T_10a98_row37_col1" class="data row37 col1" >Conformation</td>
          <td id="T_10a98_row37_col2" class="data row37 col2" >Coil</td>
          <td id="T_10a98_row37_col3" class="data row37 col3" >Coil</td>
          <td id="T_10a98_row37_col4" class="data row37 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row37_col5" class="data row37 col5" >0.181000</td>
          <td id="T_10a98_row37_col6" class="data row37 col6" >0.057287</td>
          <td id="T_10a98_row37_col7" class="data row37 col7" >-0.057287</td>
          <td id="T_10a98_row37_col8" class="data row37 col8" >0.072234</td>
          <td id="T_10a98_row37_col9" class="data row37 col9" >0.106512</td>
          <td id="T_10a98_row37_col10" class="data row37 col10" >0.000002</td>
          <td id="T_10a98_row37_col11" class="data row37 col11" >0.000076</td>
          <td id="T_10a98_row37_col12" class="data row37 col12" >28,29</td>
          <td id="T_10a98_row37_col13" class="data row37 col13" >1.919600</td>
          <td id="T_10a98_row37_col14" class="data row37 col14" >2.094497</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row38" class="row_heading level0 row38" >39</th>
          <td id="T_10a98_row38_col0" class="data row38 col0" >TMD_C_JMD_C-...)-ANDN920101</td>
          <td id="T_10a98_row38_col1" class="data row38 col1" >Structure-Activity</td>
          <td id="T_10a98_row38_col2" class="data row38 col2" >Backbone-dynamics (-CH)</td>
          <td id="T_10a98_row38_col3" class="data row38 col3" >α-CH chemica...ne-dynamics)</td>
          <td id="T_10a98_row38_col4" class="data row38 col4" >alpha-CH che...t al., 1992)</td>
          <td id="T_10a98_row38_col5" class="data row38 col5" >0.180000</td>
          <td id="T_10a98_row38_col6" class="data row38 col6" >0.096784</td>
          <td id="T_10a98_row38_col7" class="data row38 col7" >-0.096784</td>
          <td id="T_10a98_row38_col8" class="data row38 col8" >0.151260</td>
          <td id="T_10a98_row38_col9" class="data row38 col9" >0.170153</td>
          <td id="T_10a98_row38_col10" class="data row38 col10" >0.000002</td>
          <td id="T_10a98_row38_col11" class="data row38 col11" >0.000084</td>
          <td id="T_10a98_row38_col12" class="data row38 col12" >25,29,32</td>
          <td id="T_10a98_row38_col13" class="data row38 col13" >0.356800</td>
          <td id="T_10a98_row38_col14" class="data row38 col14" >0.617224</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row39" class="row_heading level0 row39" >40</th>
          <td id="T_10a98_row39_col0" class="data row39 col0" >TMD-Periodic...)-VELV850101</td>
          <td id="T_10a98_row39_col1" class="data row39 col1" >Energy</td>
          <td id="T_10a98_row39_col2" class="data row39 col2" >Electron-ion...raction pot.</td>
          <td id="T_10a98_row39_col3" class="data row39 col3" >Electron-ion...on potential</td>
          <td id="T_10a98_row39_col4" class="data row39 col4" >Electron-ion...t al., 1985)</td>
          <td id="T_10a98_row39_col5" class="data row39 col5" >0.180000</td>
          <td id="T_10a98_row39_col6" class="data row39 col6" >0.069277</td>
          <td id="T_10a98_row39_col7" class="data row39 col7" >-0.069277</td>
          <td id="T_10a98_row39_col8" class="data row39 col8" >0.094949</td>
          <td id="T_10a98_row39_col9" class="data row39 col9" >0.119524</td>
          <td id="T_10a98_row39_col10" class="data row39 col10" >0.000002</td>
          <td id="T_10a98_row39_col11" class="data row39 col11" >0.000082</td>
          <td id="T_10a98_row39_col12" class="data row39 col12" >13,16,20,23,27</td>
          <td id="T_10a98_row39_col13" class="data row39 col13" >1.818000</td>
          <td id="T_10a98_row39_col14" class="data row39 col14" >2.308293</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row40" class="row_heading level0 row40" >41</th>
          <td id="T_10a98_row40_col0" class="data row40 col0" >TMD_C_JMD_C-...)-FAUJ880109</td>
          <td id="T_10a98_row40_col1" class="data row40 col1" >Energy</td>
          <td id="T_10a98_row40_col2" class="data row40 col2" >Isoelectric point</td>
          <td id="T_10a98_row40_col3" class="data row40 col3" >Number hydro... bond donors</td>
          <td id="T_10a98_row40_col4" class="data row40 col4" >Number of hy...t al., 1988)</td>
          <td id="T_10a98_row40_col5" class="data row40 col5" >0.180000</td>
          <td id="T_10a98_row40_col6" class="data row40 col6" >0.067391</td>
          <td id="T_10a98_row40_col7" class="data row40 col7" >0.067391</td>
          <td id="T_10a98_row40_col8" class="data row40 col8" >0.098544</td>
          <td id="T_10a98_row40_col9" class="data row40 col9" >0.113033</td>
          <td id="T_10a98_row40_col10" class="data row40 col10" >0.000002</td>
          <td id="T_10a98_row40_col11" class="data row40 col11" >0.000080</td>
          <td id="T_10a98_row40_col12" class="data row40 col12" >31,32,33,34,...,37,38,39,40</td>
          <td id="T_10a98_row40_col13" class="data row40 col13" >0.635200</td>
          <td id="T_10a98_row40_col14" class="data row40 col14" >1.205475</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row41" class="row_heading level0 row41" >42</th>
          <td id="T_10a98_row41_col0" class="data row41 col0" >JMD_N_TMD_N-...)-CHAM830104</td>
          <td id="T_10a98_row41_col1" class="data row41 col1" >Shape</td>
          <td id="T_10a98_row41_col2" class="data row41 col2" >Side chain length</td>
          <td id="T_10a98_row41_col3" class="data row41 col3" >n atoms in s... chain (2+1)</td>
          <td id="T_10a98_row41_col4" class="data row41 col4" >The number o...arton, 1983)</td>
          <td id="T_10a98_row41_col5" class="data row41 col5" >0.179000</td>
          <td id="T_10a98_row41_col6" class="data row41 col6" >0.115042</td>
          <td id="T_10a98_row41_col7" class="data row41 col7" >-0.115042</td>
          <td id="T_10a98_row41_col8" class="data row41 col8" >0.151938</td>
          <td id="T_10a98_row41_col9" class="data row41 col9" >0.189623</td>
          <td id="T_10a98_row41_col10" class="data row41 col10" >0.000002</td>
          <td id="T_10a98_row41_col11" class="data row41 col11" >0.000068</td>
          <td id="T_10a98_row41_col12" class="data row41 col12" >6,9,12,15</td>
          <td id="T_10a98_row41_col13" class="data row41 col13" >0.648400</td>
          <td id="T_10a98_row41_col14" class="data row41 col14" >1.061142</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row42" class="row_heading level0 row42" >43</th>
          <td id="T_10a98_row42_col0" class="data row42 col0" >JMD_N_TMD_N-...)-QIAN880138</td>
          <td id="T_10a98_row42_col1" class="data row42 col1" >Conformation</td>
          <td id="T_10a98_row42_col2" class="data row42 col2" >Coil (C-term)</td>
          <td id="T_10a98_row42_col3" class="data row42 col3" >Coil (C-terminal)</td>
          <td id="T_10a98_row42_col4" class="data row42 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row42_col5" class="data row42 col5" >0.179000</td>
          <td id="T_10a98_row42_col6" class="data row42 col6" >0.069852</td>
          <td id="T_10a98_row42_col7" class="data row42 col7" >0.069852</td>
          <td id="T_10a98_row42_col8" class="data row42 col8" >0.103576</td>
          <td id="T_10a98_row42_col9" class="data row42 col9" >0.116589</td>
          <td id="T_10a98_row42_col10" class="data row42 col10" >0.000003</td>
          <td id="T_10a98_row42_col11" class="data row42 col11" >0.000093</td>
          <td id="T_10a98_row42_col12" class="data row42 col12" >3,6,10,13,17,20</td>
          <td id="T_10a98_row42_col13" class="data row42 col13" >0.385200</td>
          <td id="T_10a98_row42_col14" class="data row42 col14" >0.555965</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row43" class="row_heading level0 row43" >44</th>
          <td id="T_10a98_row43_col0" class="data row43 col0" >TMD-Pattern(...)-LIFS790102</td>
          <td id="T_10a98_row43_col1" class="data row43 col1" >Conformation</td>
          <td id="T_10a98_row43_col2" class="data row43 col2" >β-strand</td>
          <td id="T_10a98_row43_col3" class="data row43 col3" >β-strand</td>
          <td id="T_10a98_row43_col4" class="data row43 col4" >Conformation...ander, 1979)</td>
          <td id="T_10a98_row43_col5" class="data row43 col5" >0.176000</td>
          <td id="T_10a98_row43_col6" class="data row43 col6" >0.120892</td>
          <td id="T_10a98_row43_col7" class="data row43 col7" >0.120892</td>
          <td id="T_10a98_row43_col8" class="data row43 col8" >0.198986</td>
          <td id="T_10a98_row43_col9" class="data row43 col9" >0.216030</td>
          <td id="T_10a98_row43_col10" class="data row43 col10" >0.000004</td>
          <td id="T_10a98_row43_col11" class="data row43 col11" >0.000113</td>
          <td id="T_10a98_row43_col12" class="data row43 col12" >24,27</td>
          <td id="T_10a98_row43_col13" class="data row43 col13" >0.714800</td>
          <td id="T_10a98_row43_col14" class="data row43 col14" >1.118149</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row44" class="row_heading level0 row44" >45</th>
          <td id="T_10a98_row44_col0" class="data row44 col0" >TMD_C_JMD_C-...)-KANM800103</td>
          <td id="T_10a98_row44_col1" class="data row44 col1" >Conformation</td>
          <td id="T_10a98_row44_col2" class="data row44 col2" >α-helix</td>
          <td id="T_10a98_row44_col3" class="data row44 col3" >α-helix</td>
          <td id="T_10a98_row44_col4" class="data row44 col4" >Average rela...Tsong, 1980)</td>
          <td id="T_10a98_row44_col5" class="data row44 col5" >0.176000</td>
          <td id="T_10a98_row44_col6" class="data row44 col6" >0.087846</td>
          <td id="T_10a98_row44_col7" class="data row44 col7" >0.087846</td>
          <td id="T_10a98_row44_col8" class="data row44 col8" >0.140464</td>
          <td id="T_10a98_row44_col9" class="data row44 col9" >0.157561</td>
          <td id="T_10a98_row44_col10" class="data row44 col10" >0.000004</td>
          <td id="T_10a98_row44_col11" class="data row44 col11" >0.000113</td>
          <td id="T_10a98_row44_col12" class="data row44 col12" >24,28</td>
          <td id="T_10a98_row44_col13" class="data row44 col13" >2.704000</td>
          <td id="T_10a98_row44_col14" class="data row44 col14" >4.076269</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row45" class="row_heading level0 row45" >46</th>
          <td id="T_10a98_row45_col0" class="data row45 col0" >TMD_C_JMD_C-...)-FAUJ880108</td>
          <td id="T_10a98_row45_col1" class="data row45 col1" >Energy</td>
          <td id="T_10a98_row45_col2" class="data row45 col2" >Electron-ion...raction pot.</td>
          <td id="T_10a98_row45_col3" class="data row45 col3" >Electrical Effect</td>
          <td id="T_10a98_row45_col4" class="data row45 col4" >Localized El...t al., 1988)</td>
          <td id="T_10a98_row45_col5" class="data row45 col5" >0.176000</td>
          <td id="T_10a98_row45_col6" class="data row45 col6" >0.064253</td>
          <td id="T_10a98_row45_col7" class="data row45 col7" >-0.064253</td>
          <td id="T_10a98_row45_col8" class="data row45 col8" >0.092619</td>
          <td id="T_10a98_row45_col9" class="data row45 col9" >0.113588</td>
          <td id="T_10a98_row45_col10" class="data row45 col10" >0.000004</td>
          <td id="T_10a98_row45_col11" class="data row45 col11" >0.000113</td>
          <td id="T_10a98_row45_col12" class="data row45 col12" >21,24,28,32</td>
          <td id="T_10a98_row45_col13" class="data row45 col13" >0.826400</td>
          <td id="T_10a98_row45_col14" class="data row45 col14" >1.303426</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row46" class="row_heading level0 row46" >47</th>
          <td id="T_10a98_row46_col0" class="data row46 col0" >TMD-Pattern(...)-QIAN880134</td>
          <td id="T_10a98_row46_col1" class="data row46 col1" >Conformation</td>
          <td id="T_10a98_row46_col2" class="data row46 col2" >Coil</td>
          <td id="T_10a98_row46_col3" class="data row46 col3" >Coil</td>
          <td id="T_10a98_row46_col4" class="data row46 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row46_col5" class="data row46 col5" >0.176000</td>
          <td id="T_10a98_row46_col6" class="data row46 col6" >0.056675</td>
          <td id="T_10a98_row46_col7" class="data row46 col7" >-0.056675</td>
          <td id="T_10a98_row46_col8" class="data row46 col8" >0.099355</td>
          <td id="T_10a98_row46_col9" class="data row46 col9" >0.114698</td>
          <td id="T_10a98_row46_col10" class="data row46 col10" >0.000004</td>
          <td id="T_10a98_row46_col11" class="data row46 col11" >0.000113</td>
          <td id="T_10a98_row46_col12" class="data row46 col12" >24,27</td>
          <td id="T_10a98_row46_col13" class="data row46 col13" >0.372000</td>
          <td id="T_10a98_row46_col14" class="data row46 col14" >0.882270</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row47" class="row_heading level0 row47" >48</th>
          <td id="T_10a98_row47_col0" class="data row47 col0" >TMD_C_JMD_C-...)-TANS770106</td>
          <td id="T_10a98_row47_col1" class="data row47 col1" >Conformation</td>
          <td id="T_10a98_row47_col2" class="data row47 col2" >β-turn (TM helix)</td>
          <td id="T_10a98_row47_col3" class="data row47 col3" >β-turn in double bend</td>
          <td id="T_10a98_row47_col4" class="data row47 col4" >Normalized f...eraga, 1977)</td>
          <td id="T_10a98_row47_col5" class="data row47 col5" >0.175000</td>
          <td id="T_10a98_row47_col6" class="data row47 col6" >0.078020</td>
          <td id="T_10a98_row47_col7" class="data row47 col7" >0.078020</td>
          <td id="T_10a98_row47_col8" class="data row47 col8" >0.113536</td>
          <td id="T_10a98_row47_col9" class="data row47 col9" >0.125285</td>
          <td id="T_10a98_row47_col10" class="data row47 col10" >0.000005</td>
          <td id="T_10a98_row47_col11" class="data row47 col11" >0.000129</td>
          <td id="T_10a98_row47_col12" class="data row47 col12" >33,34,35,36</td>
          <td id="T_10a98_row47_col13" class="data row47 col13" >0.000000</td>
          <td id="T_10a98_row47_col14" class="data row47 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row48" class="row_heading level0 row48" >49</th>
          <td id="T_10a98_row48_col0" class="data row48 col0" >TMD_C_JMD_C-...)-WILM950103</td>
          <td id="T_10a98_row48_col1" class="data row48 col1" >Polarity</td>
          <td id="T_10a98_row48_col2" class="data row48 col2" >Hydrophobici... (interface)</td>
          <td id="T_10a98_row48_col3" class="data row48 col3" >Hydrophobici... (interface)</td>
          <td id="T_10a98_row48_col4" class="data row48 col4" >Hydrophobici...t al., 1995)</td>
          <td id="T_10a98_row48_col5" class="data row48 col5" >0.175000</td>
          <td id="T_10a98_row48_col6" class="data row48 col6" >0.055597</td>
          <td id="T_10a98_row48_col7" class="data row48 col7" >-0.055597</td>
          <td id="T_10a98_row48_col8" class="data row48 col8" >0.089100</td>
          <td id="T_10a98_row48_col9" class="data row48 col9" >0.105827</td>
          <td id="T_10a98_row48_col10" class="data row48 col10" >0.000005</td>
          <td id="T_10a98_row48_col11" class="data row48 col11" >0.000126</td>
          <td id="T_10a98_row48_col12" class="data row48 col12" >27,28,29,30,31,32,33</td>
          <td id="T_10a98_row48_col13" class="data row48 col13" >0.664000</td>
          <td id="T_10a98_row48_col14" class="data row48 col14" >1.089536</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row49" class="row_heading level0 row49" >50</th>
          <td id="T_10a98_row49_col0" class="data row49 col0" >TMD_C_JMD_C-...)-YUTK870103</td>
          <td id="T_10a98_row49_col1" class="data row49 col1" >Energy</td>
          <td id="T_10a98_row49_col2" class="data row49 col2" >Free energy (unfolding)</td>
          <td id="T_10a98_row49_col3" class="data row49 col3" >Free energy (unfolding)</td>
          <td id="T_10a98_row49_col4" class="data row49 col4" >Activation G...t al., 1987)</td>
          <td id="T_10a98_row49_col5" class="data row49 col5" >0.174000</td>
          <td id="T_10a98_row49_col6" class="data row49 col6" >0.123237</td>
          <td id="T_10a98_row49_col7" class="data row49 col7" >-0.123237</td>
          <td id="T_10a98_row49_col8" class="data row49 col8" >0.192743</td>
          <td id="T_10a98_row49_col9" class="data row49 col9" >0.197652</td>
          <td id="T_10a98_row49_col10" class="data row49 col10" >0.000005</td>
          <td id="T_10a98_row49_col11" class="data row49 col11" >0.000134</td>
          <td id="T_10a98_row49_col12" class="data row49 col12" >33,34,35,36</td>
          <td id="T_10a98_row49_col13" class="data row49 col13" >0.000000</td>
          <td id="T_10a98_row49_col14" class="data row49 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row50" class="row_heading level0 row50" >51</th>
          <td id="T_10a98_row50_col0" class="data row50 col0" >JMD_N_TMD_N-...)-QIAN880138</td>
          <td id="T_10a98_row50_col1" class="data row50 col1" >Conformation</td>
          <td id="T_10a98_row50_col2" class="data row50 col2" >Coil (C-term)</td>
          <td id="T_10a98_row50_col3" class="data row50 col3" >Coil (C-terminal)</td>
          <td id="T_10a98_row50_col4" class="data row50 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row50_col5" class="data row50 col5" >0.174000</td>
          <td id="T_10a98_row50_col6" class="data row50 col6" >0.067216</td>
          <td id="T_10a98_row50_col7" class="data row50 col7" >0.067216</td>
          <td id="T_10a98_row50_col8" class="data row50 col8" >0.105047</td>
          <td id="T_10a98_row50_col9" class="data row50 col9" >0.116197</td>
          <td id="T_10a98_row50_col10" class="data row50 col10" >0.000005</td>
          <td id="T_10a98_row50_col11" class="data row50 col11" >0.000133</td>
          <td id="T_10a98_row50_col12" class="data row50 col12" >1,4,8,11,15,18</td>
          <td id="T_10a98_row50_col13" class="data row50 col13" >0.000000</td>
          <td id="T_10a98_row50_col14" class="data row50 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row51" class="row_heading level0 row51" >52</th>
          <td id="T_10a98_row51_col0" class="data row51 col0" >TMD_C_JMD_C-...)-QIAN880122</td>
          <td id="T_10a98_row51_col1" class="data row51 col1" >Conformation</td>
          <td id="T_10a98_row51_col2" class="data row51 col2" >β-strand</td>
          <td id="T_10a98_row51_col3" class="data row51 col3" >β-sheet</td>
          <td id="T_10a98_row51_col4" class="data row51 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row51_col5" class="data row51 col5" >0.173000</td>
          <td id="T_10a98_row51_col6" class="data row51 col6" >0.056328</td>
          <td id="T_10a98_row51_col7" class="data row51 col7" >0.056328</td>
          <td id="T_10a98_row51_col8" class="data row51 col8" >0.067428</td>
          <td id="T_10a98_row51_col9" class="data row51 col9" >0.094795</td>
          <td id="T_10a98_row51_col10" class="data row51 col10" >0.000006</td>
          <td id="T_10a98_row51_col11" class="data row51 col11" >0.000147</td>
          <td id="T_10a98_row51_col12" class="data row51 col12" >25,28,31</td>
          <td id="T_10a98_row51_col13" class="data row51 col13" >0.483200</td>
          <td id="T_10a98_row51_col14" class="data row51 col14" >0.913371</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row52" class="row_heading level0 row52" >53</th>
          <td id="T_10a98_row52_col0" class="data row52 col0" >JMD_N_TMD_N-...)-CHAM830104</td>
          <td id="T_10a98_row52_col1" class="data row52 col1" >Shape</td>
          <td id="T_10a98_row52_col2" class="data row52 col2" >Side chain length</td>
          <td id="T_10a98_row52_col3" class="data row52 col3" >n atoms in s... chain (2+1)</td>
          <td id="T_10a98_row52_col4" class="data row52 col4" >The number o...arton, 1983)</td>
          <td id="T_10a98_row52_col5" class="data row52 col5" >0.172000</td>
          <td id="T_10a98_row52_col6" class="data row52 col6" >0.087470</td>
          <td id="T_10a98_row52_col7" class="data row52 col7" >-0.087470</td>
          <td id="T_10a98_row52_col8" class="data row52 col8" >0.135114</td>
          <td id="T_10a98_row52_col9" class="data row52 col9" >0.144731</td>
          <td id="T_10a98_row52_col10" class="data row52 col10" >0.000005</td>
          <td id="T_10a98_row52_col11" class="data row52 col11" >0.000137</td>
          <td id="T_10a98_row52_col12" class="data row52 col12" >2,5,8,11,14,17,20</td>
          <td id="T_10a98_row52_col13" class="data row52 col13" >0.444000</td>
          <td id="T_10a98_row52_col14" class="data row52 col14" >0.721620</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row53" class="row_heading level0 row53" >54</th>
          <td id="T_10a98_row53_col0" class="data row53 col0" >TMD_C_JMD_C-...)-PRAM820102</td>
          <td id="T_10a98_row53_col1" class="data row53 col1" >Shape</td>
          <td id="T_10a98_row53_col2" class="data row53 col2" >Shape and Surface</td>
          <td id="T_10a98_row53_col3" class="data row53 col3" >Slope in Regression</td>
          <td id="T_10a98_row53_col4" class="data row53 col4" >Slope in Reg...swamy, 1982)</td>
          <td id="T_10a98_row53_col5" class="data row53 col5" >0.172000</td>
          <td id="T_10a98_row53_col6" class="data row53 col6" >0.056268</td>
          <td id="T_10a98_row53_col7" class="data row53 col7" >-0.056268</td>
          <td id="T_10a98_row53_col8" class="data row53 col8" >0.074692</td>
          <td id="T_10a98_row53_col9" class="data row53 col9" >0.093571</td>
          <td id="T_10a98_row53_col10" class="data row53 col10" >0.000006</td>
          <td id="T_10a98_row53_col11" class="data row53 col11" >0.000151</td>
          <td id="T_10a98_row53_col12" class="data row53 col12" >27,28,29,30,31,32,33</td>
          <td id="T_10a98_row53_col13" class="data row53 col13" >0.303600</td>
          <td id="T_10a98_row53_col14" class="data row53 col14" >0.618242</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row54" class="row_heading level0 row54" >55</th>
          <td id="T_10a98_row54_col0" class="data row54 col0" >TMD_C_JMD_C-...)-YUTK870101</td>
          <td id="T_10a98_row54_col1" class="data row54 col1" >Energy</td>
          <td id="T_10a98_row54_col2" class="data row54 col2" >Free energy (unfolding)</td>
          <td id="T_10a98_row54_col3" class="data row54 col3" >Free energy (unfolding)</td>
          <td id="T_10a98_row54_col4" class="data row54 col4" >Unfolding Gi...t al., 1987)</td>
          <td id="T_10a98_row54_col5" class="data row54 col5" >0.171000</td>
          <td id="T_10a98_row54_col6" class="data row54 col6" >0.074769</td>
          <td id="T_10a98_row54_col7" class="data row54 col7" >-0.074769</td>
          <td id="T_10a98_row54_col8" class="data row54 col8" >0.122674</td>
          <td id="T_10a98_row54_col9" class="data row54 col9" >0.130394</td>
          <td id="T_10a98_row54_col10" class="data row54 col10" >0.000007</td>
          <td id="T_10a98_row54_col11" class="data row54 col11" >0.000164</td>
          <td id="T_10a98_row54_col12" class="data row54 col12" >26,30,33</td>
          <td id="T_10a98_row54_col13" class="data row54 col13" >0.757200</td>
          <td id="T_10a98_row54_col14" class="data row54 col14" >0.884324</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row55" class="row_heading level0 row55" >56</th>
          <td id="T_10a98_row55_col0" class="data row55 col0" >TMD_C_JMD_C-...)-HUTJ700102</td>
          <td id="T_10a98_row55_col1" class="data row55 col1" >Energy</td>
          <td id="T_10a98_row55_col2" class="data row55 col2" >Entropy</td>
          <td id="T_10a98_row55_col3" class="data row55 col3" >Entropy</td>
          <td id="T_10a98_row55_col4" class="data row55 col4" >Absolute ent...chens, 1970)</td>
          <td id="T_10a98_row55_col5" class="data row55 col5" >0.170000</td>
          <td id="T_10a98_row55_col6" class="data row55 col6" >0.057307</td>
          <td id="T_10a98_row55_col7" class="data row55 col7" >0.057307</td>
          <td id="T_10a98_row55_col8" class="data row55 col8" >0.087801</td>
          <td id="T_10a98_row55_col9" class="data row55 col9" >0.097929</td>
          <td id="T_10a98_row55_col10" class="data row55 col10" >0.000008</td>
          <td id="T_10a98_row55_col11" class="data row55 col11" >0.000175</td>
          <td id="T_10a98_row55_col12" class="data row55 col12" >27,28,29,30,31,32,33</td>
          <td id="T_10a98_row55_col13" class="data row55 col13" >0.000000</td>
          <td id="T_10a98_row55_col14" class="data row55 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row56" class="row_heading level0 row56" >57</th>
          <td id="T_10a98_row56_col0" class="data row56 col0" >TMD_C_JMD_C-...)-JANJ790102</td>
          <td id="T_10a98_row56_col1" class="data row56 col1" >Energy</td>
          <td id="T_10a98_row56_col2" class="data row56 col2" >Free energy (unfolding)</td>
          <td id="T_10a98_row56_col3" class="data row56 col3" >Transfer fre...E) to inside</td>
          <td id="T_10a98_row56_col4" class="data row56 col4" >Transfer fre...Janin, 1979)</td>
          <td id="T_10a98_row56_col5" class="data row56 col5" >0.169000</td>
          <td id="T_10a98_row56_col6" class="data row56 col6" >0.054819</td>
          <td id="T_10a98_row56_col7" class="data row56 col7" >-0.054819</td>
          <td id="T_10a98_row56_col8" class="data row56 col8" >0.095746</td>
          <td id="T_10a98_row56_col9" class="data row56 col9" >0.100423</td>
          <td id="T_10a98_row56_col10" class="data row56 col10" >0.000010</td>
          <td id="T_10a98_row56_col11" class="data row56 col11" >0.000201</td>
          <td id="T_10a98_row56_col12" class="data row56 col12" >27,28,29,30,31,32,33</td>
          <td id="T_10a98_row56_col13" class="data row56 col13" >0.000000</td>
          <td id="T_10a98_row56_col14" class="data row56 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row57" class="row_heading level0 row57" >58</th>
          <td id="T_10a98_row57_col0" class="data row57 col0" >TMD_C_JMD_C-...)-DESM900102</td>
          <td id="T_10a98_row57_col1" class="data row57 col1" >Polarity</td>
          <td id="T_10a98_row57_col2" class="data row57 col2" >Amphiphilicity (α-helix)</td>
          <td id="T_10a98_row57_col3" class="data row57 col3" >Membrane preference</td>
          <td id="T_10a98_row57_col4" class="data row57 col4" >Average memb...t al., 1990)</td>
          <td id="T_10a98_row57_col5" class="data row57 col5" >0.168000</td>
          <td id="T_10a98_row57_col6" class="data row57 col6" >0.106440</td>
          <td id="T_10a98_row57_col7" class="data row57 col7" >-0.106440</td>
          <td id="T_10a98_row57_col8" class="data row57 col8" >0.168586</td>
          <td id="T_10a98_row57_col9" class="data row57 col9" >0.175701</td>
          <td id="T_10a98_row57_col10" class="data row57 col10" >0.000011</td>
          <td id="T_10a98_row57_col11" class="data row57 col11" >0.000216</td>
          <td id="T_10a98_row57_col12" class="data row57 col12" >33,34,35,36</td>
          <td id="T_10a98_row57_col13" class="data row57 col13" >0.000000</td>
          <td id="T_10a98_row57_col14" class="data row57 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row58" class="row_heading level0 row58" >59</th>
          <td id="T_10a98_row58_col0" class="data row58 col0" >TMD_C_JMD_C-...)-KLEP840101</td>
          <td id="T_10a98_row58_col1" class="data row58 col1" >Energy</td>
          <td id="T_10a98_row58_col2" class="data row58 col2" >Charge</td>
          <td id="T_10a98_row58_col3" class="data row58 col3" >Charge</td>
          <td id="T_10a98_row58_col4" class="data row58 col4" >Net charge (...t al., 1984)</td>
          <td id="T_10a98_row58_col5" class="data row58 col5" >0.168000</td>
          <td id="T_10a98_row58_col6" class="data row58 col6" >0.086323</td>
          <td id="T_10a98_row58_col7" class="data row58 col7" >0.086323</td>
          <td id="T_10a98_row58_col8" class="data row58 col8" >0.121405</td>
          <td id="T_10a98_row58_col9" class="data row58 col9" >0.138577</td>
          <td id="T_10a98_row58_col10" class="data row58 col10" >0.000000</td>
          <td id="T_10a98_row58_col11" class="data row58 col11" >0.000030</td>
          <td id="T_10a98_row58_col12" class="data row58 col12" >30,34</td>
          <td id="T_10a98_row58_col13" class="data row58 col13" >0.140400</td>
          <td id="T_10a98_row58_col14" class="data row58 col14" >0.391229</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row59" class="row_heading level0 row59" >60</th>
          <td id="T_10a98_row59_col0" class="data row59 col0" >JMD_N_TMD_N-...)-QIAN880127</td>
          <td id="T_10a98_row59_col1" class="data row59 col1" >Conformation</td>
          <td id="T_10a98_row59_col2" class="data row59 col2" >Coil (N-term)</td>
          <td id="T_10a98_row59_col3" class="data row59 col3" >Coil (N-terminal)</td>
          <td id="T_10a98_row59_col4" class="data row59 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row59_col5" class="data row59 col5" >0.168000</td>
          <td id="T_10a98_row59_col6" class="data row59 col6" >0.071770</td>
          <td id="T_10a98_row59_col7" class="data row59 col7" >-0.071770</td>
          <td id="T_10a98_row59_col8" class="data row59 col8" >0.116934</td>
          <td id="T_10a98_row59_col9" class="data row59 col9" >0.123667</td>
          <td id="T_10a98_row59_col10" class="data row59 col10" >0.000011</td>
          <td id="T_10a98_row59_col11" class="data row59 col11" >0.000216</td>
          <td id="T_10a98_row59_col12" class="data row59 col12" >4,7,11</td>
          <td id="T_10a98_row59_col13" class="data row59 col13" >0.616400</td>
          <td id="T_10a98_row59_col14" class="data row59 col14" >1.124195</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row60" class="row_heading level0 row60" >61</th>
          <td id="T_10a98_row60_col0" class="data row60 col0" >TMD_C_JMD_C-...)-RICJ880113</td>
          <td id="T_10a98_row60_col1" class="data row60 col1" >Conformation</td>
          <td id="T_10a98_row60_col2" class="data row60 col2" >α-helix (C-cap)</td>
          <td id="T_10a98_row60_col3" class="data row60 col3" >α-helix (C-t...nal, inside)</td>
          <td id="T_10a98_row60_col4" class="data row60 col4" >Relative pre...rdson, 1988)</td>
          <td id="T_10a98_row60_col5" class="data row60 col5" >0.168000</td>
          <td id="T_10a98_row60_col6" class="data row60 col6" >0.067627</td>
          <td id="T_10a98_row60_col7" class="data row60 col7" >0.067627</td>
          <td id="T_10a98_row60_col8" class="data row60 col8" >0.098469</td>
          <td id="T_10a98_row60_col9" class="data row60 col9" >0.110321</td>
          <td id="T_10a98_row60_col10" class="data row60 col10" >0.000011</td>
          <td id="T_10a98_row60_col11" class="data row60 col11" >0.000215</td>
          <td id="T_10a98_row60_col12" class="data row60 col12" >31,32,33,34,...,37,38,39,40</td>
          <td id="T_10a98_row60_col13" class="data row60 col13" >1.105200</td>
          <td id="T_10a98_row60_col14" class="data row60 col14" >1.425601</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row61" class="row_heading level0 row61" >62</th>
          <td id="T_10a98_row61_col0" class="data row61 col0" >TMD_C_JMD_C-...)-MITS020101</td>
          <td id="T_10a98_row61_col1" class="data row61 col1" >Polarity</td>
          <td id="T_10a98_row61_col2" class="data row61 col2" >Amphiphilicity</td>
          <td id="T_10a98_row61_col3" class="data row61 col3" >Amphiphilicity</td>
          <td id="T_10a98_row61_col4" class="data row61 col4" >Amphiphilici...t al., 2002)</td>
          <td id="T_10a98_row61_col5" class="data row61 col5" >0.167000</td>
          <td id="T_10a98_row61_col6" class="data row61 col6" >0.080568</td>
          <td id="T_10a98_row61_col7" class="data row61 col7" >0.080568</td>
          <td id="T_10a98_row61_col8" class="data row61 col8" >0.128898</td>
          <td id="T_10a98_row61_col9" class="data row61 col9" >0.128726</td>
          <td id="T_10a98_row61_col10" class="data row61 col10" >0.000011</td>
          <td id="T_10a98_row61_col11" class="data row61 col11" >0.000218</td>
          <td id="T_10a98_row61_col12" class="data row61 col12" >33,34,35,36</td>
          <td id="T_10a98_row61_col13" class="data row61 col13" >1.299200</td>
          <td id="T_10a98_row61_col14" class="data row61 col14" >2.159535</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row62" class="row_heading level0 row62" >63</th>
          <td id="T_10a98_row62_col0" class="data row62 col0" >TMD-Pattern(...)-PRAM820102</td>
          <td id="T_10a98_row62_col1" class="data row62 col1" >Shape</td>
          <td id="T_10a98_row62_col2" class="data row62 col2" >Shape and Surface</td>
          <td id="T_10a98_row62_col3" class="data row62 col3" >Slope in Regression</td>
          <td id="T_10a98_row62_col4" class="data row62 col4" >Slope in Reg...swamy, 1982)</td>
          <td id="T_10a98_row62_col5" class="data row62 col5" >0.167000</td>
          <td id="T_10a98_row62_col6" class="data row62 col6" >0.077343</td>
          <td id="T_10a98_row62_col7" class="data row62 col7" >0.077343</td>
          <td id="T_10a98_row62_col8" class="data row62 col8" >0.135340</td>
          <td id="T_10a98_row62_col9" class="data row62 col9" >0.134263</td>
          <td id="T_10a98_row62_col10" class="data row62 col10" >0.000012</td>
          <td id="T_10a98_row62_col11" class="data row62 col11" >0.000228</td>
          <td id="T_10a98_row62_col12" class="data row62 col12" >19,22,26</td>
          <td id="T_10a98_row62_col13" class="data row62 col13" >1.301600</td>
          <td id="T_10a98_row62_col14" class="data row62 col14" >1.697263</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row63" class="row_heading level0 row63" >64</th>
          <td id="T_10a98_row63_col0" class="data row63 col0" >TMD_C_JMD_C-...)-FINA770101</td>
          <td id="T_10a98_row63_col1" class="data row63 col1" >Structure-Activity</td>
          <td id="T_10a98_row63_col2" class="data row63 col2" >Stability (helix-coil)</td>
          <td id="T_10a98_row63_col3" class="data row63 col3" >Stability (helix-coil)</td>
          <td id="T_10a98_row63_col4" class="data row63 col4" >Helix-coil e...itsyn, 1977)</td>
          <td id="T_10a98_row63_col5" class="data row63 col5" >0.167000</td>
          <td id="T_10a98_row63_col6" class="data row63 col6" >0.070129</td>
          <td id="T_10a98_row63_col7" class="data row63 col7" >0.070129</td>
          <td id="T_10a98_row63_col8" class="data row63 col8" >0.114947</td>
          <td id="T_10a98_row63_col9" class="data row63 col9" >0.126566</td>
          <td id="T_10a98_row63_col10" class="data row63 col10" >0.000012</td>
          <td id="T_10a98_row63_col11" class="data row63 col11" >0.000223</td>
          <td id="T_10a98_row63_col12" class="data row63 col12" >21,24,28,32</td>
          <td id="T_10a98_row63_col13" class="data row63 col13" >1.199200</td>
          <td id="T_10a98_row63_col14" class="data row63 col14" >1.554845</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row64" class="row_heading level0 row64" >65</th>
          <td id="T_10a98_row64_col0" class="data row64 col0" >TMD_C_JMD_C-...)-AURR980110</td>
          <td id="T_10a98_row64_col1" class="data row64 col1" >Conformation</td>
          <td id="T_10a98_row64_col2" class="data row64 col2" >α-helix</td>
          <td id="T_10a98_row64_col3" class="data row64 col3" >α-helix (middle)</td>
          <td id="T_10a98_row64_col4" class="data row64 col4" >Normalized p...-Rose, 1998)</td>
          <td id="T_10a98_row64_col5" class="data row64 col5" >0.166000</td>
          <td id="T_10a98_row64_col6" class="data row64 col6" >0.081797</td>
          <td id="T_10a98_row64_col7" class="data row64 col7" >0.081797</td>
          <td id="T_10a98_row64_col8" class="data row64 col8" >0.121170</td>
          <td id="T_10a98_row64_col9" class="data row64 col9" >0.149555</td>
          <td id="T_10a98_row64_col10" class="data row64 col10" >0.000013</td>
          <td id="T_10a98_row64_col11" class="data row64 col11" >0.000239</td>
          <td id="T_10a98_row64_col12" class="data row64 col12" >33,34,35,36</td>
          <td id="T_10a98_row64_col13" class="data row64 col13" >1.295200</td>
          <td id="T_10a98_row64_col14" class="data row64 col14" >2.225137</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row65" class="row_heading level0 row65" >66</th>
          <td id="T_10a98_row65_col0" class="data row65 col0" >TMD-Pattern(...)-VELV850101</td>
          <td id="T_10a98_row65_col1" class="data row65 col1" >Energy</td>
          <td id="T_10a98_row65_col2" class="data row65 col2" >Electron-ion...raction pot.</td>
          <td id="T_10a98_row65_col3" class="data row65 col3" >Electron-ion...on potential</td>
          <td id="T_10a98_row65_col4" class="data row65 col4" >Electron-ion...t al., 1985)</td>
          <td id="T_10a98_row65_col5" class="data row65 col5" >0.165000</td>
          <td id="T_10a98_row65_col6" class="data row65 col6" >0.121210</td>
          <td id="T_10a98_row65_col7" class="data row65 col7" >-0.121210</td>
          <td id="T_10a98_row65_col8" class="data row65 col8" >0.143560</td>
          <td id="T_10a98_row65_col9" class="data row65 col9" >0.207767</td>
          <td id="T_10a98_row65_col10" class="data row65 col10" >0.000015</td>
          <td id="T_10a98_row65_col11" class="data row65 col11" >0.000254</td>
          <td id="T_10a98_row65_col12" class="data row65 col12" >24,27</td>
          <td id="T_10a98_row65_col13" class="data row65 col13" >1.302000</td>
          <td id="T_10a98_row65_col14" class="data row65 col14" >1.466618</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row66" class="row_heading level0 row66" >67</th>
          <td id="T_10a98_row66_col0" class="data row66 col0" >TMD_C_JMD_C-...)-VELV850101</td>
          <td id="T_10a98_row66_col1" class="data row66 col1" >Energy</td>
          <td id="T_10a98_row66_col2" class="data row66 col2" >Electron-ion...raction pot.</td>
          <td id="T_10a98_row66_col3" class="data row66 col3" >Electron-ion...on potential</td>
          <td id="T_10a98_row66_col4" class="data row66 col4" >Electron-ion...t al., 1985)</td>
          <td id="T_10a98_row66_col5" class="data row66 col5" >0.165000</td>
          <td id="T_10a98_row66_col6" class="data row66 col6" >0.119568</td>
          <td id="T_10a98_row66_col7" class="data row66 col7" >-0.119568</td>
          <td id="T_10a98_row66_col8" class="data row66 col8" >0.143560</td>
          <td id="T_10a98_row66_col9" class="data row66 col9" >0.205817</td>
          <td id="T_10a98_row66_col10" class="data row66 col10" >0.000014</td>
          <td id="T_10a98_row66_col11" class="data row66 col11" >0.000253</td>
          <td id="T_10a98_row66_col12" class="data row66 col12" >25,28</td>
          <td id="T_10a98_row66_col13" class="data row66 col13" >0.000000</td>
          <td id="T_10a98_row66_col14" class="data row66 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row67" class="row_heading level0 row67" >68</th>
          <td id="T_10a98_row67_col0" class="data row67 col0" >TMD-Pattern(...)-HUTJ700102</td>
          <td id="T_10a98_row67_col1" class="data row67 col1" >Energy</td>
          <td id="T_10a98_row67_col2" class="data row67 col2" >Entropy</td>
          <td id="T_10a98_row67_col3" class="data row67 col3" >Entropy</td>
          <td id="T_10a98_row67_col4" class="data row67 col4" >Absolute ent...chens, 1970)</td>
          <td id="T_10a98_row67_col5" class="data row67 col5" >0.165000</td>
          <td id="T_10a98_row67_col6" class="data row67 col6" >0.063134</td>
          <td id="T_10a98_row67_col7" class="data row67 col7" >-0.063134</td>
          <td id="T_10a98_row67_col8" class="data row67 col8" >0.104624</td>
          <td id="T_10a98_row67_col9" class="data row67 col9" >0.113955</td>
          <td id="T_10a98_row67_col10" class="data row67 col10" >0.000015</td>
          <td id="T_10a98_row67_col11" class="data row67 col11" >0.000258</td>
          <td id="T_10a98_row67_col12" class="data row67 col12" >19,22,26</td>
          <td id="T_10a98_row67_col13" class="data row67 col13" >0.000000</td>
          <td id="T_10a98_row67_col14" class="data row67 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row68" class="row_heading level0 row68" >69</th>
          <td id="T_10a98_row68_col0" class="data row68 col0" >TMD_C_JMD_C-...)-TANS770108</td>
          <td id="T_10a98_row68_col1" class="data row68 col1" >Conformation</td>
          <td id="T_10a98_row68_col2" class="data row68 col2" >β/α-bridge</td>
          <td id="T_10a98_row68_col3" class="data row68 col3" >β/α-bridge</td>
          <td id="T_10a98_row68_col4" class="data row68 col4" >Normalized f...eraga, 1977)</td>
          <td id="T_10a98_row68_col5" class="data row68 col5" >0.164000</td>
          <td id="T_10a98_row68_col6" class="data row68 col6" >0.079708</td>
          <td id="T_10a98_row68_col7" class="data row68 col7" >0.079708</td>
          <td id="T_10a98_row68_col8" class="data row68 col8" >0.135324</td>
          <td id="T_10a98_row68_col9" class="data row68 col9" >0.137910</td>
          <td id="T_10a98_row68_col10" class="data row68 col10" >0.000016</td>
          <td id="T_10a98_row68_col11" class="data row68 col11" >0.000271</td>
          <td id="T_10a98_row68_col12" class="data row68 col12" >32,33,34</td>
          <td id="T_10a98_row68_col13" class="data row68 col13" >0.462400</td>
          <td id="T_10a98_row68_col14" class="data row68 col14" >0.706967</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row69" class="row_heading level0 row69" >70</th>
          <td id="T_10a98_row69_col0" class="data row69 col0" >TMD_C_JMD_C-...)-CHOP780212</td>
          <td id="T_10a98_row69_col1" class="data row69 col1" >Conformation</td>
          <td id="T_10a98_row69_col2" class="data row69 col2" >β-sheet (C-term)</td>
          <td id="T_10a98_row69_col3" class="data row69 col3" >β-turn (1st residue)</td>
          <td id="T_10a98_row69_col4" class="data row69 col4" >Frequency of...sman, 1978b)</td>
          <td id="T_10a98_row69_col5" class="data row69 col5" >0.164000</td>
          <td id="T_10a98_row69_col6" class="data row69 col6" >0.076207</td>
          <td id="T_10a98_row69_col7" class="data row69 col7" >-0.076207</td>
          <td id="T_10a98_row69_col8" class="data row69 col8" >0.125506</td>
          <td id="T_10a98_row69_col9" class="data row69 col9" >0.147002</td>
          <td id="T_10a98_row69_col10" class="data row69 col10" >0.000016</td>
          <td id="T_10a98_row69_col11" class="data row69 col11" >0.000267</td>
          <td id="T_10a98_row69_col12" class="data row69 col12" >24,28,32</td>
          <td id="T_10a98_row69_col13" class="data row69 col13" >1.095600</td>
          <td id="T_10a98_row69_col14" class="data row69 col14" >1.575630</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row70" class="row_heading level0 row70" >71</th>
          <td id="T_10a98_row70_col0" class="data row70 col0" >TMD-Periodic...)-COHE430101</td>
          <td id="T_10a98_row70_col1" class="data row70 col1" >ASA/Volume</td>
          <td id="T_10a98_row70_col2" class="data row70 col2" >Partial specific volume</td>
          <td id="T_10a98_row70_col3" class="data row70 col3" >Partial specific volume</td>
          <td id="T_10a98_row70_col4" class="data row70 col4" >Partial spec...dsall, 1943)</td>
          <td id="T_10a98_row70_col5" class="data row70 col5" >0.164000</td>
          <td id="T_10a98_row70_col6" class="data row70 col6" >0.058745</td>
          <td id="T_10a98_row70_col7" class="data row70 col7" >0.058745</td>
          <td id="T_10a98_row70_col8" class="data row70 col8" >0.092103</td>
          <td id="T_10a98_row70_col9" class="data row70 col9" >0.106413</td>
          <td id="T_10a98_row70_col10" class="data row70 col10" >0.000017</td>
          <td id="T_10a98_row70_col11" class="data row70 col11" >0.000276</td>
          <td id="T_10a98_row70_col12" class="data row70 col12" >12,15,18,21,24,27,30</td>
          <td id="T_10a98_row70_col13" class="data row70 col13" >1.141600</td>
          <td id="T_10a98_row70_col14" class="data row70 col14" >1.375595</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row71" class="row_heading level0 row71" >72</th>
          <td id="T_10a98_row71_col0" class="data row71 col0" >TMD-Pattern(...)-OOBM770105</td>
          <td id="T_10a98_row71_col1" class="data row71 col1" >Energy</td>
          <td id="T_10a98_row71_col2" class="data row71 col2" >Non-bonded energy</td>
          <td id="T_10a98_row71_col3" class="data row71 col3" >Non-bonded e... per residue</td>
          <td id="T_10a98_row71_col4" class="data row71 col4" >Short and me...e-Ooi, 1977)</td>
          <td id="T_10a98_row71_col5" class="data row71 col5" >0.164000</td>
          <td id="T_10a98_row71_col6" class="data row71 col6" >0.056983</td>
          <td id="T_10a98_row71_col7" class="data row71 col7" >0.056983</td>
          <td id="T_10a98_row71_col8" class="data row71 col8" >0.099221</td>
          <td id="T_10a98_row71_col9" class="data row71 col9" >0.102039</td>
          <td id="T_10a98_row71_col10" class="data row71 col10" >0.000017</td>
          <td id="T_10a98_row71_col11" class="data row71 col11" >0.000274</td>
          <td id="T_10a98_row71_col12" class="data row71 col12" >16,19,22,26</td>
          <td id="T_10a98_row71_col13" class="data row71 col13" >1.305600</td>
          <td id="T_10a98_row71_col14" class="data row71 col14" >1.643621</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row72" class="row_heading level0 row72" >73</th>
          <td id="T_10a98_row72_col0" class="data row72 col0" >TMD-Pattern(...)-ANDN920101</td>
          <td id="T_10a98_row72_col1" class="data row72 col1" >Structure-Activity</td>
          <td id="T_10a98_row72_col2" class="data row72 col2" >Backbone-dynamics (-CH)</td>
          <td id="T_10a98_row72_col3" class="data row72 col3" >α-CH chemica...ne-dynamics)</td>
          <td id="T_10a98_row72_col4" class="data row72 col4" >alpha-CH che...t al., 1992)</td>
          <td id="T_10a98_row72_col5" class="data row72 col5" >0.163000</td>
          <td id="T_10a98_row72_col6" class="data row72 col6" >0.128817</td>
          <td id="T_10a98_row72_col7" class="data row72 col7" >-0.128817</td>
          <td id="T_10a98_row72_col8" class="data row72 col8" >0.184672</td>
          <td id="T_10a98_row72_col9" class="data row72 col9" >0.227780</td>
          <td id="T_10a98_row72_col10" class="data row72 col10" >0.000020</td>
          <td id="T_10a98_row72_col11" class="data row72 col11" >0.000293</td>
          <td id="T_10a98_row72_col12" class="data row72 col12" >24,27</td>
          <td id="T_10a98_row72_col13" class="data row72 col13" >0.872800</td>
          <td id="T_10a98_row72_col14" class="data row72 col14" >1.063156</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row73" class="row_heading level0 row73" >74</th>
          <td id="T_10a98_row73_col0" class="data row73 col0" >TMD_C_JMD_C-...)-EISD860101</td>
          <td id="T_10a98_row73_col1" class="data row73 col1" >Polarity</td>
          <td id="T_10a98_row73_col2" class="data row73 col2" >Hydrophobicity</td>
          <td id="T_10a98_row73_col3" class="data row73 col3" >Solvation free energy</td>
          <td id="T_10a98_row73_col4" class="data row73 col4" >Solvation fr...chlan, 1986)</td>
          <td id="T_10a98_row73_col5" class="data row73 col5" >0.162000</td>
          <td id="T_10a98_row73_col6" class="data row73 col6" >0.083936</td>
          <td id="T_10a98_row73_col7" class="data row73 col7" >-0.083936</td>
          <td id="T_10a98_row73_col8" class="data row73 col8" >0.143338</td>
          <td id="T_10a98_row73_col9" class="data row73 col9" >0.147948</td>
          <td id="T_10a98_row73_col10" class="data row73 col10" >0.000021</td>
          <td id="T_10a98_row73_col11" class="data row73 col11" >0.000304</td>
          <td id="T_10a98_row73_col12" class="data row73 col12" >30,33,37</td>
          <td id="T_10a98_row73_col13" class="data row73 col13" >0.330400</td>
          <td id="T_10a98_row73_col14" class="data row73 col14" >0.377566</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row74" class="row_heading level0 row74" >75</th>
          <td id="T_10a98_row74_col0" class="data row74 col0" >TMD_C_JMD_C-...)-QIAN880130</td>
          <td id="T_10a98_row74_col1" class="data row74 col1" >Conformation</td>
          <td id="T_10a98_row74_col2" class="data row74 col2" >Coil</td>
          <td id="T_10a98_row74_col3" class="data row74 col3" >Coil</td>
          <td id="T_10a98_row74_col4" class="data row74 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row74_col5" class="data row74 col5" >0.162000</td>
          <td id="T_10a98_row74_col6" class="data row74 col6" >0.070292</td>
          <td id="T_10a98_row74_col7" class="data row74 col7" >-0.070292</td>
          <td id="T_10a98_row74_col8" class="data row74 col8" >0.096915</td>
          <td id="T_10a98_row74_col9" class="data row74 col9" >0.128362</td>
          <td id="T_10a98_row74_col10" class="data row74 col10" >0.000020</td>
          <td id="T_10a98_row74_col11" class="data row74 col11" >0.000302</td>
          <td id="T_10a98_row74_col12" class="data row74 col12" >21,25,28</td>
          <td id="T_10a98_row74_col13" class="data row74 col13" >1.528400</td>
          <td id="T_10a98_row74_col14" class="data row74 col14" >2.418922</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row75" class="row_heading level0 row75" >76</th>
          <td id="T_10a98_row75_col0" class="data row75 col0" >TMD-Pattern(...)-QIAN880130</td>
          <td id="T_10a98_row75_col1" class="data row75 col1" >Conformation</td>
          <td id="T_10a98_row75_col2" class="data row75 col2" >Coil</td>
          <td id="T_10a98_row75_col3" class="data row75 col3" >Coil</td>
          <td id="T_10a98_row75_col4" class="data row75 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row75_col5" class="data row75 col5" >0.161000</td>
          <td id="T_10a98_row75_col6" class="data row75 col6" >0.068424</td>
          <td id="T_10a98_row75_col7" class="data row75 col7" >-0.068424</td>
          <td id="T_10a98_row75_col8" class="data row75 col8" >0.096915</td>
          <td id="T_10a98_row75_col9" class="data row75 col9" >0.126975</td>
          <td id="T_10a98_row75_col10" class="data row75 col10" >0.000024</td>
          <td id="T_10a98_row75_col11" class="data row75 col11" >0.000332</td>
          <td id="T_10a98_row75_col12" class="data row75 col12" >20,24,27</td>
          <td id="T_10a98_row75_col13" class="data row75 col13" >0.000000</td>
          <td id="T_10a98_row75_col14" class="data row75 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row76" class="row_heading level0 row76" >77</th>
          <td id="T_10a98_row76_col0" class="data row76 col0" >JMD_N_TMD_N-...)-BIOV880101</td>
          <td id="T_10a98_row76_col1" class="data row76 col1" >ASA/Volume</td>
          <td id="T_10a98_row76_col2" class="data row76 col2" >Buried</td>
          <td id="T_10a98_row76_col3" class="data row76 col3" >Buriability</td>
          <td id="T_10a98_row76_col4" class="data row76 col4" >Information ...t al., 1988)</td>
          <td id="T_10a98_row76_col5" class="data row76 col5" >0.161000</td>
          <td id="T_10a98_row76_col6" class="data row76 col6" >0.058976</td>
          <td id="T_10a98_row76_col7" class="data row76 col7" >-0.058976</td>
          <td id="T_10a98_row76_col8" class="data row76 col8" >0.096823</td>
          <td id="T_10a98_row76_col9" class="data row76 col9" >0.114647</td>
          <td id="T_10a98_row76_col10" class="data row76 col10" >0.000025</td>
          <td id="T_10a98_row76_col11" class="data row76 col11" >0.000335</td>
          <td id="T_10a98_row76_col12" class="data row76 col12" >6,7,8,9,10</td>
          <td id="T_10a98_row76_col13" class="data row76 col13" >0.000000</td>
          <td id="T_10a98_row76_col14" class="data row76 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row77" class="row_heading level0 row77" >78</th>
          <td id="T_10a98_row77_col0" class="data row77 col0" >JMD_N_TMD_N-...)-PRAM820103</td>
          <td id="T_10a98_row77_col1" class="data row77 col1" >Shape</td>
          <td id="T_10a98_row77_col2" class="data row77 col2" >Shape and Surface</td>
          <td id="T_10a98_row77_col3" class="data row77 col3" >Correlation ...n regression</td>
          <td id="T_10a98_row77_col4" class="data row77 col4" >Correlation ...swamy, 1982)</td>
          <td id="T_10a98_row77_col5" class="data row77 col5" >0.161000</td>
          <td id="T_10a98_row77_col6" class="data row77 col6" >0.057828</td>
          <td id="T_10a98_row77_col7" class="data row77 col7" >0.057828</td>
          <td id="T_10a98_row77_col8" class="data row77 col8" >0.088362</td>
          <td id="T_10a98_row77_col9" class="data row77 col9" >0.106085</td>
          <td id="T_10a98_row77_col10" class="data row77 col10" >0.000024</td>
          <td id="T_10a98_row77_col11" class="data row77 col11" >0.000328</td>
          <td id="T_10a98_row77_col12" class="data row77 col12" >1,5,8,11</td>
          <td id="T_10a98_row77_col13" class="data row77 col13" >1.304400</td>
          <td id="T_10a98_row77_col14" class="data row77 col14" >1.657101</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row78" class="row_heading level0 row78" >79</th>
          <td id="T_10a98_row78_col0" class="data row78 col0" >TMD_C_JMD_C-...)-ROSM880103</td>
          <td id="T_10a98_row78_col1" class="data row78 col1" >Structure-Activity</td>
          <td id="T_10a98_row78_col2" class="data row78 col2" >Backbone-dynamics (-CH)</td>
          <td id="T_10a98_row78_col3" class="data row78 col3" >Loss of hydr...ix formation</td>
          <td id="T_10a98_row78_col4" class="data row78 col4" >Loss of Side...seman, 1988)</td>
          <td id="T_10a98_row78_col5" class="data row78 col5" >0.160000</td>
          <td id="T_10a98_row78_col6" class="data row78 col6" >0.059281</td>
          <td id="T_10a98_row78_col7" class="data row78 col7" >-0.059281</td>
          <td id="T_10a98_row78_col8" class="data row78 col8" >0.100693</td>
          <td id="T_10a98_row78_col9" class="data row78 col9" >0.120806</td>
          <td id="T_10a98_row78_col10" class="data row78 col10" >0.000027</td>
          <td id="T_10a98_row78_col11" class="data row78 col11" >0.000359</td>
          <td id="T_10a98_row78_col12" class="data row78 col12" >32,33,34</td>
          <td id="T_10a98_row78_col13" class="data row78 col13" >0.757200</td>
          <td id="T_10a98_row78_col14" class="data row78 col14" >1.471249</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row79" class="row_heading level0 row79" >80</th>
          <td id="T_10a98_row79_col0" class="data row79 col0" >TMD_C_JMD_C-...)-FINA910104</td>
          <td id="T_10a98_row79_col1" class="data row79 col1" >Conformation</td>
          <td id="T_10a98_row79_col2" class="data row79 col2" >α-helix (C-cap)</td>
          <td id="T_10a98_row79_col3" class="data row79 col3" >α-helix termination</td>
          <td id="T_10a98_row79_col4" class="data row79 col4" >Helix termin...t al., 1991)</td>
          <td id="T_10a98_row79_col5" class="data row79 col5" >0.159000</td>
          <td id="T_10a98_row79_col6" class="data row79 col6" >0.103808</td>
          <td id="T_10a98_row79_col7" class="data row79 col7" >0.103808</td>
          <td id="T_10a98_row79_col8" class="data row79 col8" >0.140977</td>
          <td id="T_10a98_row79_col9" class="data row79 col9" >0.179008</td>
          <td id="T_10a98_row79_col10" class="data row79 col10" >0.000014</td>
          <td id="T_10a98_row79_col11" class="data row79 col11" >0.000248</td>
          <td id="T_10a98_row79_col12" class="data row79 col12" >33,37</td>
          <td id="T_10a98_row79_col13" class="data row79 col13" >0.233200</td>
          <td id="T_10a98_row79_col14" class="data row79 col14" >0.593921</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row80" class="row_heading level0 row80" >81</th>
          <td id="T_10a98_row80_col0" class="data row80 col0" >TMD-Pattern(...)-FINA770101</td>
          <td id="T_10a98_row80_col1" class="data row80 col1" >Structure-Activity</td>
          <td id="T_10a98_row80_col2" class="data row80 col2" >Stability (helix-coil)</td>
          <td id="T_10a98_row80_col3" class="data row80 col3" >Stability (helix-coil)</td>
          <td id="T_10a98_row80_col4" class="data row80 col4" >Helix-coil e...itsyn, 1977)</td>
          <td id="T_10a98_row80_col5" class="data row80 col5" >0.158000</td>
          <td id="T_10a98_row80_col6" class="data row80 col6" >0.089108</td>
          <td id="T_10a98_row80_col7" class="data row80 col7" >-0.089108</td>
          <td id="T_10a98_row80_col8" class="data row80 col8" >0.164444</td>
          <td id="T_10a98_row80_col9" class="data row80 col9" >0.163909</td>
          <td id="T_10a98_row80_col10" class="data row80 col10" >0.000035</td>
          <td id="T_10a98_row80_col11" class="data row80 col11" >0.000420</td>
          <td id="T_10a98_row80_col12" class="data row80 col12" >11,14,17</td>
          <td id="T_10a98_row80_col13" class="data row80 col13" >1.014000</td>
          <td id="T_10a98_row80_col14" class="data row80 col14" >1.539338</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row81" class="row_heading level0 row81" >82</th>
          <td id="T_10a98_row81_col0" class="data row81 col0" >JMD_N_TMD_N-...)-ZHOH040101</td>
          <td id="T_10a98_row81_col1" class="data row81 col1" >Structure-Activity</td>
          <td id="T_10a98_row81_col2" class="data row81 col2" >Stability</td>
          <td id="T_10a98_row81_col3" class="data row81 col3" >Stability</td>
          <td id="T_10a98_row81_col4" class="data row81 col4" >The stabilit...-Zhou, 2004)</td>
          <td id="T_10a98_row81_col5" class="data row81 col5" >0.157000</td>
          <td id="T_10a98_row81_col6" class="data row81 col6" >0.127895</td>
          <td id="T_10a98_row81_col7" class="data row81 col7" >-0.127895</td>
          <td id="T_10a98_row81_col8" class="data row81 col8" >0.151304</td>
          <td id="T_10a98_row81_col9" class="data row81 col9" >0.258491</td>
          <td id="T_10a98_row81_col10" class="data row81 col10" >0.000035</td>
          <td id="T_10a98_row81_col11" class="data row81 col11" >0.000420</td>
          <td id="T_10a98_row81_col12" class="data row81 col12" >5,6</td>
          <td id="T_10a98_row81_col13" class="data row81 col13" >0.833200</td>
          <td id="T_10a98_row81_col14" class="data row81 col14" >1.360696</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row82" class="row_heading level0 row82" >83</th>
          <td id="T_10a98_row82_col0" class="data row82 col0" >TMD_C_JMD_C-...)-MONM990101</td>
          <td id="T_10a98_row82_col1" class="data row82 col1" >Conformation</td>
          <td id="T_10a98_row82_col2" class="data row82 col2" >β-turn (TM helix)</td>
          <td id="T_10a98_row82_col3" class="data row82 col3" >β-turn (TM helix)</td>
          <td id="T_10a98_row82_col4" class="data row82 col4" >Turn propens...t al., 1999)</td>
          <td id="T_10a98_row82_col5" class="data row82 col5" >0.157000</td>
          <td id="T_10a98_row82_col6" class="data row82 col6" >0.093480</td>
          <td id="T_10a98_row82_col7" class="data row82 col7" >0.093480</td>
          <td id="T_10a98_row82_col8" class="data row82 col8" >0.163797</td>
          <td id="T_10a98_row82_col9" class="data row82 col9" >0.179896</td>
          <td id="T_10a98_row82_col10" class="data row82 col10" >0.000039</td>
          <td id="T_10a98_row82_col11" class="data row82 col11" >0.000449</td>
          <td id="T_10a98_row82_col12" class="data row82 col12" >31,32,33,34,35</td>
          <td id="T_10a98_row82_col13" class="data row82 col13" >0.000000</td>
          <td id="T_10a98_row82_col14" class="data row82 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row83" class="row_heading level0 row83" >84</th>
          <td id="T_10a98_row83_col0" class="data row83 col0" >JMD_N_TMD_N-...)-FAUJ880101</td>
          <td id="T_10a98_row83_col1" class="data row83 col1" >Shape</td>
          <td id="T_10a98_row83_col2" class="data row83 col2" >Steric parameter</td>
          <td id="T_10a98_row83_col3" class="data row83 col3" >Shape Index</td>
          <td id="T_10a98_row83_col4" class="data row83 col4" >Graph shape ...t al., 1988)</td>
          <td id="T_10a98_row83_col5" class="data row83 col5" >0.156000</td>
          <td id="T_10a98_row83_col6" class="data row83 col6" >0.093808</td>
          <td id="T_10a98_row83_col7" class="data row83 col7" >-0.093808</td>
          <td id="T_10a98_row83_col8" class="data row83 col8" >0.157361</td>
          <td id="T_10a98_row83_col9" class="data row83 col9" >0.168800</td>
          <td id="T_10a98_row83_col10" class="data row83 col10" >0.000042</td>
          <td id="T_10a98_row83_col11" class="data row83 col11" >0.000458</td>
          <td id="T_10a98_row83_col12" class="data row83 col12" >6</td>
          <td id="T_10a98_row83_col13" class="data row83 col13" >0.379600</td>
          <td id="T_10a98_row83_col14" class="data row83 col14" >0.843756</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row84" class="row_heading level0 row84" >85</th>
          <td id="T_10a98_row84_col0" class="data row84 col0" >TMD_C_JMD_C-...)-JANJ780101</td>
          <td id="T_10a98_row84_col1" class="data row84 col1" >ASA/Volume</td>
          <td id="T_10a98_row84_col2" class="data row84 col2" >Accessible s...e area (ASA)</td>
          <td id="T_10a98_row84_col3" class="data row84 col3" >ASA (folded protein)</td>
          <td id="T_10a98_row84_col4" class="data row84 col4" >Average acce...t al., 1978)</td>
          <td id="T_10a98_row84_col5" class="data row84 col5" >0.155000</td>
          <td id="T_10a98_row84_col6" class="data row84 col6" >0.110281</td>
          <td id="T_10a98_row84_col7" class="data row84 col7" >0.110281</td>
          <td id="T_10a98_row84_col8" class="data row84 col8" >0.178578</td>
          <td id="T_10a98_row84_col9" class="data row84 col9" >0.202098</td>
          <td id="T_10a98_row84_col10" class="data row84 col10" >0.000046</td>
          <td id="T_10a98_row84_col11" class="data row84 col11" >0.000486</td>
          <td id="T_10a98_row84_col12" class="data row84 col12" >33,37,40</td>
          <td id="T_10a98_row84_col13" class="data row84 col13" >0.272400</td>
          <td id="T_10a98_row84_col14" class="data row84 col14" >0.623809</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row85" class="row_heading level0 row85" >86</th>
          <td id="T_10a98_row85_col0" class="data row85 col0" >JMD_N_TMD_N-...)-FINA770101</td>
          <td id="T_10a98_row85_col1" class="data row85 col1" >Structure-Activity</td>
          <td id="T_10a98_row85_col2" class="data row85 col2" >Stability (helix-coil)</td>
          <td id="T_10a98_row85_col3" class="data row85 col3" >Stability (helix-coil)</td>
          <td id="T_10a98_row85_col4" class="data row85 col4" >Helix-coil e...itsyn, 1977)</td>
          <td id="T_10a98_row85_col5" class="data row85 col5" >0.155000</td>
          <td id="T_10a98_row85_col6" class="data row85 col6" >0.087029</td>
          <td id="T_10a98_row85_col7" class="data row85 col7" >-0.087029</td>
          <td id="T_10a98_row85_col8" class="data row85 col8" >0.163718</td>
          <td id="T_10a98_row85_col9" class="data row85 col9" >0.162392</td>
          <td id="T_10a98_row85_col10" class="data row85 col10" >0.000049</td>
          <td id="T_10a98_row85_col11" class="data row85 col11" >0.000508</td>
          <td id="T_10a98_row85_col12" class="data row85 col12" >9,12,15</td>
          <td id="T_10a98_row85_col13" class="data row85 col13" >0.000000</td>
          <td id="T_10a98_row85_col14" class="data row85 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row86" class="row_heading level0 row86" >87</th>
          <td id="T_10a98_row86_col0" class="data row86 col0" >JMD_N_TMD_N-...)-RICJ880107</td>
          <td id="T_10a98_row86_col1" class="data row86 col1" >Conformation</td>
          <td id="T_10a98_row86_col2" class="data row86 col2" >π-helix</td>
          <td id="T_10a98_row86_col3" class="data row86 col3" >α-helix</td>
          <td id="T_10a98_row86_col4" class="data row86 col4" >Relative pre...rdson, 1988)</td>
          <td id="T_10a98_row86_col5" class="data row86 col5" >0.155000</td>
          <td id="T_10a98_row86_col6" class="data row86 col6" >0.066867</td>
          <td id="T_10a98_row86_col7" class="data row86 col7" >-0.066867</td>
          <td id="T_10a98_row86_col8" class="data row86 col8" >0.105803</td>
          <td id="T_10a98_row86_col9" class="data row86 col9" >0.129430</td>
          <td id="T_10a98_row86_col10" class="data row86 col10" >0.000047</td>
          <td id="T_10a98_row86_col11" class="data row86 col11" >0.000496</td>
          <td id="T_10a98_row86_col12" class="data row86 col12" >3,6,9,13</td>
          <td id="T_10a98_row86_col13" class="data row86 col13" >0.335200</td>
          <td id="T_10a98_row86_col14" class="data row86 col14" >0.649905</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row87" class="row_heading level0 row87" >88</th>
          <td id="T_10a98_row87_col0" class="data row87 col0" >JMD_N_TMD_N-...)-RADA880106</td>
          <td id="T_10a98_row87_col1" class="data row87 col1" >ASA/Volume</td>
          <td id="T_10a98_row87_col2" class="data row87 col2" >Volume</td>
          <td id="T_10a98_row87_col3" class="data row87 col3" >Accessible s...e area (ASA)</td>
          <td id="T_10a98_row87_col4" class="data row87 col4" >Accessible s...enden, 1988)</td>
          <td id="T_10a98_row87_col5" class="data row87 col5" >0.155000</td>
          <td id="T_10a98_row87_col6" class="data row87 col6" >0.059593</td>
          <td id="T_10a98_row87_col7" class="data row87 col7" >-0.059593</td>
          <td id="T_10a98_row87_col8" class="data row87 col8" >0.104862</td>
          <td id="T_10a98_row87_col9" class="data row87 col9" >0.110749</td>
          <td id="T_10a98_row87_col10" class="data row87 col10" >0.000050</td>
          <td id="T_10a98_row87_col11" class="data row87 col11" >0.000508</td>
          <td id="T_10a98_row87_col12" class="data row87 col12" >6,9,12,15</td>
          <td id="T_10a98_row87_col13" class="data row87 col13" >0.482000</td>
          <td id="T_10a98_row87_col14" class="data row87 col14" >0.672000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row88" class="row_heading level0 row88" >89</th>
          <td id="T_10a98_row88_col0" class="data row88 col0" >JMD_N_TMD_N-...)-ARGP820101</td>
          <td id="T_10a98_row88_col1" class="data row88 col1" >Polarity</td>
          <td id="T_10a98_row88_col2" class="data row88 col2" >Hydrophobicity</td>
          <td id="T_10a98_row88_col3" class="data row88 col3" >Hydrophobicity</td>
          <td id="T_10a98_row88_col4" class="data row88 col4" >Hydrophobici...t al., 1982)</td>
          <td id="T_10a98_row88_col5" class="data row88 col5" >0.154000</td>
          <td id="T_10a98_row88_col6" class="data row88 col6" >0.092099</td>
          <td id="T_10a98_row88_col7" class="data row88 col7" >-0.092099</td>
          <td id="T_10a98_row88_col8" class="data row88 col8" >0.142836</td>
          <td id="T_10a98_row88_col9" class="data row88 col9" >0.171547</td>
          <td id="T_10a98_row88_col10" class="data row88 col10" >0.000052</td>
          <td id="T_10a98_row88_col11" class="data row88 col11" >0.000520</td>
          <td id="T_10a98_row88_col12" class="data row88 col12" >4,7,11</td>
          <td id="T_10a98_row88_col13" class="data row88 col13" >1.065200</td>
          <td id="T_10a98_row88_col14" class="data row88 col14" >1.916900</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row89" class="row_heading level0 row89" >90</th>
          <td id="T_10a98_row89_col0" class="data row89 col0" >TMD_C_JMD_C-...)-CHAM820102</td>
          <td id="T_10a98_row89_col1" class="data row89 col1" >Polarity</td>
          <td id="T_10a98_row89_col2" class="data row89 col2" >Hydrophobici... (interface)</td>
          <td id="T_10a98_row89_col3" class="data row89 col3" >Free energy (interface)</td>
          <td id="T_10a98_row89_col4" class="data row89 col4" >Free energy ...arton, 1982)</td>
          <td id="T_10a98_row89_col5" class="data row89 col5" >0.154000</td>
          <td id="T_10a98_row89_col6" class="data row89 col6" >0.082300</td>
          <td id="T_10a98_row89_col7" class="data row89 col7" >-0.082300</td>
          <td id="T_10a98_row89_col8" class="data row89 col8" >0.136264</td>
          <td id="T_10a98_row89_col9" class="data row89 col9" >0.177551</td>
          <td id="T_10a98_row89_col10" class="data row89 col10" >0.000050</td>
          <td id="T_10a98_row89_col11" class="data row89 col11" >0.000508</td>
          <td id="T_10a98_row89_col12" class="data row89 col12" >33,34</td>
          <td id="T_10a98_row89_col13" class="data row89 col13" >0.366800</td>
          <td id="T_10a98_row89_col14" class="data row89 col14" >0.691767</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row90" class="row_heading level0 row90" >91</th>
          <td id="T_10a98_row90_col0" class="data row90 col0" >TMD-Pattern(...)-MAXF760105</td>
          <td id="T_10a98_row90_col1" class="data row90 col1" >Conformation</td>
          <td id="T_10a98_row90_col2" class="data row90 col2" >α-helix (left-handed)</td>
          <td id="T_10a98_row90_col3" class="data row90 col3" >α-helix (left-handed)</td>
          <td id="T_10a98_row90_col4" class="data row90 col4" >Normalized f...eraga, 1976)</td>
          <td id="T_10a98_row90_col5" class="data row90 col5" >0.154000</td>
          <td id="T_10a98_row90_col6" class="data row90 col6" >0.062226</td>
          <td id="T_10a98_row90_col7" class="data row90 col7" >0.062226</td>
          <td id="T_10a98_row90_col8" class="data row90 col8" >0.144085</td>
          <td id="T_10a98_row90_col9" class="data row90 col9" >0.119863</td>
          <td id="T_10a98_row90_col10" class="data row90 col10" >0.000057</td>
          <td id="T_10a98_row90_col11" class="data row90 col11" >0.000548</td>
          <td id="T_10a98_row90_col12" class="data row90 col12" >19,22,26</td>
          <td id="T_10a98_row90_col13" class="data row90 col13" >0.715200</td>
          <td id="T_10a98_row90_col14" class="data row90 col14" >1.186306</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row91" class="row_heading level0 row91" >92</th>
          <td id="T_10a98_row91_col0" class="data row91 col0" >TMD_C_JMD_C-...)-BIOV880101</td>
          <td id="T_10a98_row91_col1" class="data row91 col1" >ASA/Volume</td>
          <td id="T_10a98_row91_col2" class="data row91 col2" >Buried</td>
          <td id="T_10a98_row91_col3" class="data row91 col3" >Buriability</td>
          <td id="T_10a98_row91_col4" class="data row91 col4" >Information ...t al., 1988)</td>
          <td id="T_10a98_row91_col5" class="data row91 col5" >0.153000</td>
          <td id="T_10a98_row91_col6" class="data row91 col6" >0.085041</td>
          <td id="T_10a98_row91_col7" class="data row91 col7" >-0.085041</td>
          <td id="T_10a98_row91_col8" class="data row91 col8" >0.135864</td>
          <td id="T_10a98_row91_col9" class="data row91 col9" >0.161279</td>
          <td id="T_10a98_row91_col10" class="data row91 col10" >0.000059</td>
          <td id="T_10a98_row91_col11" class="data row91 col11" >0.000561</td>
          <td id="T_10a98_row91_col12" class="data row91 col12" >30,33,37</td>
          <td id="T_10a98_row91_col13" class="data row91 col13" >0.473600</td>
          <td id="T_10a98_row91_col14" class="data row91 col14" >0.930690</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row92" class="row_heading level0 row92" >93</th>
          <td id="T_10a98_row92_col0" class="data row92 col0" >TMD_C_JMD_C-...)-WILM950103</td>
          <td id="T_10a98_row92_col1" class="data row92 col1" >Polarity</td>
          <td id="T_10a98_row92_col2" class="data row92 col2" >Hydrophobici... (interface)</td>
          <td id="T_10a98_row92_col3" class="data row92 col3" >Hydrophobici... (interface)</td>
          <td id="T_10a98_row92_col4" class="data row92 col4" >Hydrophobici...t al., 1995)</td>
          <td id="T_10a98_row92_col5" class="data row92 col5" >0.153000</td>
          <td id="T_10a98_row92_col6" class="data row92 col6" >0.069595</td>
          <td id="T_10a98_row92_col7" class="data row92 col7" >-0.069595</td>
          <td id="T_10a98_row92_col8" class="data row92 col8" >0.107314</td>
          <td id="T_10a98_row92_col9" class="data row92 col9" >0.134698</td>
          <td id="T_10a98_row92_col10" class="data row92 col10" >0.000060</td>
          <td id="T_10a98_row92_col11" class="data row92 col11" >0.000566</td>
          <td id="T_10a98_row92_col12" class="data row92 col12" >26,29,33</td>
          <td id="T_10a98_row92_col13" class="data row92 col13" >0.770800</td>
          <td id="T_10a98_row92_col14" class="data row92 col14" >1.299178</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row93" class="row_heading level0 row93" >94</th>
          <td id="T_10a98_row93_col0" class="data row93 col0" >TMD-Pattern(...)-RACS820101</td>
          <td id="T_10a98_row93_col1" class="data row93 col1" >Conformation</td>
          <td id="T_10a98_row93_col2" class="data row93 col2" >β-sheet (N-term)</td>
          <td id="T_10a98_row93_col3" class="data row93 col3" >α-helix with...tructure (i)</td>
          <td id="T_10a98_row93_col4" class="data row93 col4" >Average rela...eraga, 1982)</td>
          <td id="T_10a98_row93_col5" class="data row93 col5" >0.153000</td>
          <td id="T_10a98_row93_col6" class="data row93 col6" >0.062678</td>
          <td id="T_10a98_row93_col7" class="data row93 col7" >-0.062678</td>
          <td id="T_10a98_row93_col8" class="data row93 col8" >0.109868</td>
          <td id="T_10a98_row93_col9" class="data row93 col9" >0.123054</td>
          <td id="T_10a98_row93_col10" class="data row93 col10" >0.000061</td>
          <td id="T_10a98_row93_col11" class="data row93 col11" >0.000570</td>
          <td id="T_10a98_row93_col12" class="data row93 col12" >12,15,18,21</td>
          <td id="T_10a98_row93_col13" class="data row93 col13" >0.333600</td>
          <td id="T_10a98_row93_col14" class="data row93 col14" >0.598524</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row94" class="row_heading level0 row94" >95</th>
          <td id="T_10a98_row94_col0" class="data row94 col0" >JMD_N_TMD_N-...)-FAUJ880101</td>
          <td id="T_10a98_row94_col1" class="data row94 col1" >Shape</td>
          <td id="T_10a98_row94_col2" class="data row94 col2" >Steric parameter</td>
          <td id="T_10a98_row94_col3" class="data row94 col3" >Shape Index</td>
          <td id="T_10a98_row94_col4" class="data row94 col4" >Graph shape ...t al., 1988)</td>
          <td id="T_10a98_row94_col5" class="data row94 col5" >0.152000</td>
          <td id="T_10a98_row94_col6" class="data row94 col6" >0.070636</td>
          <td id="T_10a98_row94_col7" class="data row94 col7" >-0.070636</td>
          <td id="T_10a98_row94_col8" class="data row94 col8" >0.127109</td>
          <td id="T_10a98_row94_col9" class="data row94 col9" >0.134520</td>
          <td id="T_10a98_row94_col10" class="data row94 col10" >0.000071</td>
          <td id="T_10a98_row94_col11" class="data row94 col11" >0.000626</td>
          <td id="T_10a98_row94_col12" class="data row94 col12" >6,7</td>
          <td id="T_10a98_row94_col13" class="data row94 col13" >0.184800</td>
          <td id="T_10a98_row94_col14" class="data row94 col14" >0.448866</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row95" class="row_heading level0 row95" >96</th>
          <td id="T_10a98_row95_col0" class="data row95 col0" >TMD_C_JMD_C-...)-LINS030106</td>
          <td id="T_10a98_row95_col1" class="data row95 col1" >ASA/Volume</td>
          <td id="T_10a98_row95_col2" class="data row95 col2" >Accessible s...e area (ASA)</td>
          <td id="T_10a98_row95_col3" class="data row95 col3" >Hydrophilic ...ed proteins)</td>
          <td id="T_10a98_row95_col4" class="data row95 col4" >Hydrophilic ...t al., 2003)</td>
          <td id="T_10a98_row95_col5" class="data row95 col5" >0.151000</td>
          <td id="T_10a98_row95_col6" class="data row95 col6" >0.071208</td>
          <td id="T_10a98_row95_col7" class="data row95 col7" >0.071208</td>
          <td id="T_10a98_row95_col8" class="data row95 col8" >0.136279</td>
          <td id="T_10a98_row95_col9" class="data row95 col9" >0.155749</td>
          <td id="T_10a98_row95_col10" class="data row95 col10" >0.000078</td>
          <td id="T_10a98_row95_col11" class="data row95 col11" >0.000657</td>
          <td id="T_10a98_row95_col12" class="data row95 col12" >26,30,33</td>
          <td id="T_10a98_row95_col13" class="data row95 col13" >0.326400</td>
          <td id="T_10a98_row95_col14" class="data row95 col14" >0.451202</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row96" class="row_heading level0 row96" >97</th>
          <td id="T_10a98_row96_col0" class="data row96 col0" >JMD_N_TMD_N-...)-MEIH800101</td>
          <td id="T_10a98_row96_col1" class="data row96 col1" >Shape</td>
          <td id="T_10a98_row96_col2" class="data row96 col2" >Reduced distance</td>
          <td id="T_10a98_row96_col3" class="data row96 col3" >Reduced distance (C-α)</td>
          <td id="T_10a98_row96_col4" class="data row96 col4" >Average redu...t al., 1980)</td>
          <td id="T_10a98_row96_col5" class="data row96 col5" >0.151000</td>
          <td id="T_10a98_row96_col6" class="data row96 col6" >0.060954</td>
          <td id="T_10a98_row96_col7" class="data row96 col7" >0.060954</td>
          <td id="T_10a98_row96_col8" class="data row96 col8" >0.099010</td>
          <td id="T_10a98_row96_col9" class="data row96 col9" >0.121406</td>
          <td id="T_10a98_row96_col10" class="data row96 col10" >0.000072</td>
          <td id="T_10a98_row96_col11" class="data row96 col11" >0.000631</td>
          <td id="T_10a98_row96_col12" class="data row96 col12" >6,7,8,9,10</td>
          <td id="T_10a98_row96_col13" class="data row96 col13" >0.000000</td>
          <td id="T_10a98_row96_col14" class="data row96 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row97" class="row_heading level0 row97" >98</th>
          <td id="T_10a98_row97_col0" class="data row97 col0" >TMD-Pattern(...)-TANS770102</td>
          <td id="T_10a98_row97_col1" class="data row97 col1" >Conformation</td>
          <td id="T_10a98_row97_col2" class="data row97 col2" >α-helix (C-term, out)</td>
          <td id="T_10a98_row97_col3" class="data row97 col3" >α-helix (C-t...al, outside)</td>
          <td id="T_10a98_row97_col4" class="data row97 col4" >Normalized f...eraga, 1977)</td>
          <td id="T_10a98_row97_col5" class="data row97 col5" >0.150000</td>
          <td id="T_10a98_row97_col6" class="data row97 col6" >0.056439</td>
          <td id="T_10a98_row97_col7" class="data row97 col7" >-0.056439</td>
          <td id="T_10a98_row97_col8" class="data row97 col8" >0.094520</td>
          <td id="T_10a98_row97_col9" class="data row97 col9" >0.108682</td>
          <td id="T_10a98_row97_col10" class="data row97 col10" >0.000084</td>
          <td id="T_10a98_row97_col11" class="data row97 col11" >0.000685</td>
          <td id="T_10a98_row97_col12" class="data row97 col12" >17,20,24,28</td>
          <td id="T_10a98_row97_col13" class="data row97 col13" >0.684400</td>
          <td id="T_10a98_row97_col14" class="data row97 col14" >0.941892</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row98" class="row_heading level0 row98" >99</th>
          <td id="T_10a98_row98_col0" class="data row98 col0" >JMD_N_TMD_N-...)-MEIH800101</td>
          <td id="T_10a98_row98_col1" class="data row98 col1" >Shape</td>
          <td id="T_10a98_row98_col2" class="data row98 col2" >Reduced distance</td>
          <td id="T_10a98_row98_col3" class="data row98 col3" >Reduced distance (C-α)</td>
          <td id="T_10a98_row98_col4" class="data row98 col4" >Average redu...t al., 1980)</td>
          <td id="T_10a98_row98_col5" class="data row98 col5" >0.149000</td>
          <td id="T_10a98_row98_col6" class="data row98 col6" >0.075519</td>
          <td id="T_10a98_row98_col7" class="data row98 col7" >0.075519</td>
          <td id="T_10a98_row98_col8" class="data row98 col8" >0.131494</td>
          <td id="T_10a98_row98_col9" class="data row98 col9" >0.155207</td>
          <td id="T_10a98_row98_col10" class="data row98 col10" >0.000091</td>
          <td id="T_10a98_row98_col11" class="data row98 col11" >0.000715</td>
          <td id="T_10a98_row98_col12" class="data row98 col12" >4,5,6</td>
          <td id="T_10a98_row98_col13" class="data row98 col13" >0.376000</td>
          <td id="T_10a98_row98_col14" class="data row98 col14" >0.526004</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row99" class="row_heading level0 row99" >100</th>
          <td id="T_10a98_row99_col0" class="data row99 col0" >TMD_C_JMD_C-...)-LEVM760105</td>
          <td id="T_10a98_row99_col1" class="data row99 col1" >Shape</td>
          <td id="T_10a98_row99_col2" class="data row99 col2" >Side chain length</td>
          <td id="T_10a98_row99_col3" class="data row99 col3" >Side chain length</td>
          <td id="T_10a98_row99_col4" class="data row99 col4" >Radius of gy...evitt, 1976)</td>
          <td id="T_10a98_row99_col5" class="data row99 col5" >0.149000</td>
          <td id="T_10a98_row99_col6" class="data row99 col6" >0.073526</td>
          <td id="T_10a98_row99_col7" class="data row99 col7" >0.073526</td>
          <td id="T_10a98_row99_col8" class="data row99 col8" >0.133612</td>
          <td id="T_10a98_row99_col9" class="data row99 col9" >0.157088</td>
          <td id="T_10a98_row99_col10" class="data row99 col10" >0.000090</td>
          <td id="T_10a98_row99_col11" class="data row99 col11" >0.000714</td>
          <td id="T_10a98_row99_col12" class="data row99 col12" >31,34,38</td>
          <td id="T_10a98_row99_col13" class="data row99 col13" >2.050800</td>
          <td id="T_10a98_row99_col14" class="data row99 col14" >2.338278</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row100" class="row_heading level0 row100" >101</th>
          <td id="T_10a98_row100_col0" class="data row100 col0" >TMD_C_JMD_C-...)-CHOP780212</td>
          <td id="T_10a98_row100_col1" class="data row100 col1" >Conformation</td>
          <td id="T_10a98_row100_col2" class="data row100 col2" >β-sheet (C-term)</td>
          <td id="T_10a98_row100_col3" class="data row100 col3" >β-turn (1st residue)</td>
          <td id="T_10a98_row100_col4" class="data row100 col4" >Frequency of...sman, 1978b)</td>
          <td id="T_10a98_row100_col5" class="data row100 col5" >0.149000</td>
          <td id="T_10a98_row100_col6" class="data row100 col6" >0.069627</td>
          <td id="T_10a98_row100_col7" class="data row100 col7" >-0.069627</td>
          <td id="T_10a98_row100_col8" class="data row100 col8" >0.113251</td>
          <td id="T_10a98_row100_col9" class="data row100 col9" >0.143949</td>
          <td id="T_10a98_row100_col10" class="data row100 col10" >0.000093</td>
          <td id="T_10a98_row100_col11" class="data row100 col11" >0.000725</td>
          <td id="T_10a98_row100_col12" class="data row100 col12" >26,29,33</td>
          <td id="T_10a98_row100_col13" class="data row100 col13" >0.842800</td>
          <td id="T_10a98_row100_col14" class="data row100 col14" >1.314094</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row101" class="row_heading level0 row101" >102</th>
          <td id="T_10a98_row101_col0" class="data row101 col0" >TMD_C_JMD_C-...)-OOBM770105</td>
          <td id="T_10a98_row101_col1" class="data row101 col1" >Energy</td>
          <td id="T_10a98_row101_col2" class="data row101 col2" >Non-bonded energy</td>
          <td id="T_10a98_row101_col3" class="data row101 col3" >Non-bonded e... per residue</td>
          <td id="T_10a98_row101_col4" class="data row101 col4" >Short and me...e-Ooi, 1977)</td>
          <td id="T_10a98_row101_col5" class="data row101 col5" >0.149000</td>
          <td id="T_10a98_row101_col6" class="data row101 col6" >0.067384</td>
          <td id="T_10a98_row101_col7" class="data row101 col7" >0.067384</td>
          <td id="T_10a98_row101_col8" class="data row101 col8" >0.134029</td>
          <td id="T_10a98_row101_col9" class="data row101 col9" >0.143995</td>
          <td id="T_10a98_row101_col10" class="data row101 col10" >0.000095</td>
          <td id="T_10a98_row101_col11" class="data row101 col11" >0.000731</td>
          <td id="T_10a98_row101_col12" class="data row101 col12" >23,27</td>
          <td id="T_10a98_row101_col13" class="data row101 col13" >0.868400</td>
          <td id="T_10a98_row101_col14" class="data row101 col14" >1.469467</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row102" class="row_heading level0 row102" >103</th>
          <td id="T_10a98_row102_col0" class="data row102 col0" >JMD_N_TMD_N-...)-RACS820101</td>
          <td id="T_10a98_row102_col1" class="data row102 col1" >Conformation</td>
          <td id="T_10a98_row102_col2" class="data row102 col2" >β-sheet (N-term)</td>
          <td id="T_10a98_row102_col3" class="data row102 col3" >α-helix with...tructure (i)</td>
          <td id="T_10a98_row102_col4" class="data row102 col4" >Average rela...eraga, 1982)</td>
          <td id="T_10a98_row102_col5" class="data row102 col5" >0.149000</td>
          <td id="T_10a98_row102_col6" class="data row102 col6" >0.063073</td>
          <td id="T_10a98_row102_col7" class="data row102 col7" >-0.063073</td>
          <td id="T_10a98_row102_col8" class="data row102 col8" >0.107731</td>
          <td id="T_10a98_row102_col9" class="data row102 col9" >0.126806</td>
          <td id="T_10a98_row102_col10" class="data row102 col10" >0.000091</td>
          <td id="T_10a98_row102_col11" class="data row102 col11" >0.000716</td>
          <td id="T_10a98_row102_col12" class="data row102 col12" >10,13,16,19</td>
          <td id="T_10a98_row102_col13" class="data row102 col13" >0.000000</td>
          <td id="T_10a98_row102_col14" class="data row102 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row103" class="row_heading level0 row103" >104</th>
          <td id="T_10a98_row103_col0" class="data row103 col0" >JMD_N_TMD_N-...)-ARGP820101</td>
          <td id="T_10a98_row103_col1" class="data row103 col1" >Polarity</td>
          <td id="T_10a98_row103_col2" class="data row103 col2" >Hydrophobicity</td>
          <td id="T_10a98_row103_col3" class="data row103 col3" >Hydrophobicity</td>
          <td id="T_10a98_row103_col4" class="data row103 col4" >Hydrophobici...t al., 1982)</td>
          <td id="T_10a98_row103_col5" class="data row103 col5" >0.148000</td>
          <td id="T_10a98_row103_col6" class="data row103 col6" >0.076361</td>
          <td id="T_10a98_row103_col7" class="data row103 col7" >-0.076361</td>
          <td id="T_10a98_row103_col8" class="data row103 col8" >0.140513</td>
          <td id="T_10a98_row103_col9" class="data row103 col9" >0.148387</td>
          <td id="T_10a98_row103_col10" class="data row103 col10" >0.000108</td>
          <td id="T_10a98_row103_col11" class="data row103 col11" >0.000790</td>
          <td id="T_10a98_row103_col12" class="data row103 col12" >4,5,6</td>
          <td id="T_10a98_row103_col13" class="data row103 col13" >0.537200</td>
          <td id="T_10a98_row103_col14" class="data row103 col14" >1.041739</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row104" class="row_heading level0 row104" >105</th>
          <td id="T_10a98_row104_col0" class="data row104 col0" >JMD_N_TMD_N-...)-BROC820101</td>
          <td id="T_10a98_row104_col1" class="data row104 col1" >Polarity</td>
          <td id="T_10a98_row104_col2" class="data row104 col2" >Hydrophobicity</td>
          <td id="T_10a98_row104_col3" class="data row104 col3" >Hydrophobici...coefficient)</td>
          <td id="T_10a98_row104_col4" class="data row104 col4" >Retention Co...t al., 1982)</td>
          <td id="T_10a98_row104_col5" class="data row104 col5" >0.148000</td>
          <td id="T_10a98_row104_col6" class="data row104 col6" >0.067069</td>
          <td id="T_10a98_row104_col7" class="data row104 col7" >-0.067069</td>
          <td id="T_10a98_row104_col8" class="data row104 col8" >0.120409</td>
          <td id="T_10a98_row104_col9" class="data row104 col9" >0.137261</td>
          <td id="T_10a98_row104_col10" class="data row104 col10" >0.000103</td>
          <td id="T_10a98_row104_col11" class="data row104 col11" >0.000768</td>
          <td id="T_10a98_row104_col12" class="data row104 col12" >6,9,12,15</td>
          <td id="T_10a98_row104_col13" class="data row104 col13" >0.106400</td>
          <td id="T_10a98_row104_col14" class="data row104 col14" >0.249766</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row105" class="row_heading level0 row105" >106</th>
          <td id="T_10a98_row105_col0" class="data row105 col0" >TMD_C_JMD_C-...)-CORJ870107</td>
          <td id="T_10a98_row105_col1" class="data row105 col1" >Polarity</td>
          <td id="T_10a98_row105_col2" class="data row105 col2" >Amphiphilicity (α-helix)</td>
          <td id="T_10a98_row105_col3" class="data row105 col3" >Amphiphilicity (α-helix)</td>
          <td id="T_10a98_row105_col4" class="data row105 col4" >TOTFT index ...t al., 1987)</td>
          <td id="T_10a98_row105_col5" class="data row105 col5" >0.147000</td>
          <td id="T_10a98_row105_col6" class="data row105 col6" >0.086326</td>
          <td id="T_10a98_row105_col7" class="data row105 col7" >0.086326</td>
          <td id="T_10a98_row105_col8" class="data row105 col8" >0.163515</td>
          <td id="T_10a98_row105_col9" class="data row105 col9" >0.196429</td>
          <td id="T_10a98_row105_col10" class="data row105 col10" >0.000111</td>
          <td id="T_10a98_row105_col11" class="data row105 col11" >0.000799</td>
          <td id="T_10a98_row105_col12" class="data row105 col12" >24,28</td>
          <td id="T_10a98_row105_col13" class="data row105 col13" >0.420400</td>
          <td id="T_10a98_row105_col14" class="data row105 col14" >0.765639</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row106" class="row_heading level0 row106" >107</th>
          <td id="T_10a98_row106_col0" class="data row106 col0" >TMD_C_JMD_C-...)-ANDN920101</td>
          <td id="T_10a98_row106_col1" class="data row106 col1" >Structure-Activity</td>
          <td id="T_10a98_row106_col2" class="data row106 col2" >Backbone-dynamics (-CH)</td>
          <td id="T_10a98_row106_col3" class="data row106 col3" >α-CH chemica...ne-dynamics)</td>
          <td id="T_10a98_row106_col4" class="data row106 col4" >alpha-CH che...t al., 1992)</td>
          <td id="T_10a98_row106_col5" class="data row106 col5" >0.147000</td>
          <td id="T_10a98_row106_col6" class="data row106 col6" >0.079575</td>
          <td id="T_10a98_row106_col7" class="data row106 col7" >-0.079575</td>
          <td id="T_10a98_row106_col8" class="data row106 col8" >0.145620</td>
          <td id="T_10a98_row106_col9" class="data row106 col9" >0.160200</td>
          <td id="T_10a98_row106_col10" class="data row106 col10" >0.000115</td>
          <td id="T_10a98_row106_col11" class="data row106 col11" >0.000811</td>
          <td id="T_10a98_row106_col12" class="data row106 col12" >25,26,27,28</td>
          <td id="T_10a98_row106_col13" class="data row106 col13" >0.322000</td>
          <td id="T_10a98_row106_col14" class="data row106 col14" >0.559943</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row107" class="row_heading level0 row107" >108</th>
          <td id="T_10a98_row107_col0" class="data row107 col0" >TMD-Pattern(...)-RICJ880107</td>
          <td id="T_10a98_row107_col1" class="data row107 col1" >Conformation</td>
          <td id="T_10a98_row107_col2" class="data row107 col2" >π-helix</td>
          <td id="T_10a98_row107_col3" class="data row107 col3" >α-helix</td>
          <td id="T_10a98_row107_col4" class="data row107 col4" >Relative pre...rdson, 1988)</td>
          <td id="T_10a98_row107_col5" class="data row107 col5" >0.146000</td>
          <td id="T_10a98_row107_col6" class="data row107 col6" >0.068957</td>
          <td id="T_10a98_row107_col7" class="data row107 col7" >0.068957</td>
          <td id="T_10a98_row107_col8" class="data row107 col8" >0.131400</td>
          <td id="T_10a98_row107_col9" class="data row107 col9" >0.140413</td>
          <td id="T_10a98_row107_col10" class="data row107 col10" >0.000131</td>
          <td id="T_10a98_row107_col11" class="data row107 col11" >0.000868</td>
          <td id="T_10a98_row107_col12" class="data row107 col12" >20,23,27</td>
          <td id="T_10a98_row107_col13" class="data row107 col13" >0.697200</td>
          <td id="T_10a98_row107_col14" class="data row107 col14" >1.056350</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row108" class="row_heading level0 row108" >109</th>
          <td id="T_10a98_row108_col0" class="data row108 col0" >TMD_C_JMD_C-...)-COHE430101</td>
          <td id="T_10a98_row108_col1" class="data row108 col1" >ASA/Volume</td>
          <td id="T_10a98_row108_col2" class="data row108 col2" >Partial specific volume</td>
          <td id="T_10a98_row108_col3" class="data row108 col3" >Partial specific volume</td>
          <td id="T_10a98_row108_col4" class="data row108 col4" >Partial spec...dsall, 1943)</td>
          <td id="T_10a98_row108_col5" class="data row108 col5" >0.145000</td>
          <td id="T_10a98_row108_col6" class="data row108 col6" >0.124999</td>
          <td id="T_10a98_row108_col7" class="data row108 col7" >0.124999</td>
          <td id="T_10a98_row108_col8" class="data row108 col8" >0.180151</td>
          <td id="T_10a98_row108_col9" class="data row108 col9" >0.242281</td>
          <td id="T_10a98_row108_col10" class="data row108 col10" >0.000145</td>
          <td id="T_10a98_row108_col11" class="data row108 col11" >0.000912</td>
          <td id="T_10a98_row108_col12" class="data row108 col12" >28,29</td>
          <td id="T_10a98_row108_col13" class="data row108 col13" >1.740800</td>
          <td id="T_10a98_row108_col14" class="data row108 col14" >2.317117</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row109" class="row_heading level0 row109" >110</th>
          <td id="T_10a98_row109_col0" class="data row109 col0" >TMD_C_JMD_C-...)-MEIH800101</td>
          <td id="T_10a98_row109_col1" class="data row109 col1" >Shape</td>
          <td id="T_10a98_row109_col2" class="data row109 col2" >Reduced distance</td>
          <td id="T_10a98_row109_col3" class="data row109 col3" >Reduced distance (C-α)</td>
          <td id="T_10a98_row109_col4" class="data row109 col4" >Average redu...t al., 1980)</td>
          <td id="T_10a98_row109_col5" class="data row109 col5" >0.144000</td>
          <td id="T_10a98_row109_col6" class="data row109 col6" >0.101763</td>
          <td id="T_10a98_row109_col7" class="data row109 col7" >0.101763</td>
          <td id="T_10a98_row109_col8" class="data row109 col8" >0.161290</td>
          <td id="T_10a98_row109_col9" class="data row109 col9" >0.209898</td>
          <td id="T_10a98_row109_col10" class="data row109 col10" >0.000156</td>
          <td id="T_10a98_row109_col11" class="data row109 col11" >0.000951</td>
          <td id="T_10a98_row109_col12" class="data row109 col12" >33,37</td>
          <td id="T_10a98_row109_col13" class="data row109 col13" >0.000000</td>
          <td id="T_10a98_row109_col14" class="data row109 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row110" class="row_heading level0 row110" >111</th>
          <td id="T_10a98_row110_col0" class="data row110 col0" >JMD_N_TMD_N-...)-RICJ880111</td>
          <td id="T_10a98_row110_col1" class="data row110 col1" >Conformation</td>
          <td id="T_10a98_row110_col2" class="data row110 col2" >π-helix</td>
          <td id="T_10a98_row110_col3" class="data row110 col3" >α-helix</td>
          <td id="T_10a98_row110_col4" class="data row110 col4" >Relative pre...rdson, 1988)</td>
          <td id="T_10a98_row110_col5" class="data row110 col5" >0.143000</td>
          <td id="T_10a98_row110_col6" class="data row110 col6" >0.069574</td>
          <td id="T_10a98_row110_col7" class="data row110 col7" >-0.069574</td>
          <td id="T_10a98_row110_col8" class="data row110 col8" >0.126051</td>
          <td id="T_10a98_row110_col9" class="data row110 col9" >0.159340</td>
          <td id="T_10a98_row110_col10" class="data row110 col10" >0.000173</td>
          <td id="T_10a98_row110_col11" class="data row110 col11" >0.001007</td>
          <td id="T_10a98_row110_col12" class="data row110 col12" >5,6</td>
          <td id="T_10a98_row110_col13" class="data row110 col13" >0.363200</td>
          <td id="T_10a98_row110_col14" class="data row110 col14" >0.326260</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row111" class="row_heading level0 row111" >112</th>
          <td id="T_10a98_row111_col0" class="data row111 col0" >JMD_N_TMD_N-...)-BIGC670101</td>
          <td id="T_10a98_row111_col1" class="data row111 col1" >ASA/Volume</td>
          <td id="T_10a98_row111_col2" class="data row111 col2" >Volume</td>
          <td id="T_10a98_row111_col3" class="data row111 col3" >Volume</td>
          <td id="T_10a98_row111_col4" class="data row111 col4" >Residue volu...gelow, 1967)</td>
          <td id="T_10a98_row111_col5" class="data row111 col5" >0.143000</td>
          <td id="T_10a98_row111_col6" class="data row111 col6" >0.067181</td>
          <td id="T_10a98_row111_col7" class="data row111 col7" >-0.067181</td>
          <td id="T_10a98_row111_col8" class="data row111 col8" >0.141579</td>
          <td id="T_10a98_row111_col9" class="data row111 col9" >0.135502</td>
          <td id="T_10a98_row111_col10" class="data row111 col10" >0.000184</td>
          <td id="T_10a98_row111_col11" class="data row111 col11" >0.001045</td>
          <td id="T_10a98_row111_col12" class="data row111 col12" >5,8,11</td>
          <td id="T_10a98_row111_col13" class="data row111 col13" >0.382000</td>
          <td id="T_10a98_row111_col14" class="data row111 col14" >0.675082</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row112" class="row_heading level0 row112" >113</th>
          <td id="T_10a98_row112_col0" class="data row112 col0" >JMD_N_TMD_N-...)-CIDH920102</td>
          <td id="T_10a98_row112_col1" class="data row112 col1" >Polarity</td>
          <td id="T_10a98_row112_col2" class="data row112 col2" >Hydrophobicity</td>
          <td id="T_10a98_row112_col3" class="data row112 col3" >Hydrophobicity</td>
          <td id="T_10a98_row112_col4" class="data row112 col4" >Normalized h...t al., 1992)</td>
          <td id="T_10a98_row112_col5" class="data row112 col5" >0.142000</td>
          <td id="T_10a98_row112_col6" class="data row112 col6" >0.070908</td>
          <td id="T_10a98_row112_col7" class="data row112 col7" >-0.070908</td>
          <td id="T_10a98_row112_col8" class="data row112 col8" >0.135389</td>
          <td id="T_10a98_row112_col9" class="data row112 col9" >0.144272</td>
          <td id="T_10a98_row112_col10" class="data row112 col10" >0.000190</td>
          <td id="T_10a98_row112_col11" class="data row112 col11" >0.001062</td>
          <td id="T_10a98_row112_col12" class="data row112 col12" >5,8,11</td>
          <td id="T_10a98_row112_col13" class="data row112 col13" >0.384400</td>
          <td id="T_10a98_row112_col14" class="data row112 col14" >0.570074</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row113" class="row_heading level0 row113" >114</th>
          <td id="T_10a98_row113_col0" class="data row113 col0" >JMD_N_TMD_N-...)-BAEK050101</td>
          <td id="T_10a98_row113_col1" class="data row113 col1" >Conformation</td>
          <td id="T_10a98_row113_col2" class="data row113 col2" >β-strand</td>
          <td id="T_10a98_row113_col3" class="data row113 col3" >Linker index...long region)</td>
          <td id="T_10a98_row113_col4" class="data row113 col4" >Linker index...t al., 2005)</td>
          <td id="T_10a98_row113_col5" class="data row113 col5" >0.142000</td>
          <td id="T_10a98_row113_col6" class="data row113 col6" >0.058743</td>
          <td id="T_10a98_row113_col7" class="data row113 col7" >-0.058743</td>
          <td id="T_10a98_row113_col8" class="data row113 col8" >0.117342</td>
          <td id="T_10a98_row113_col9" class="data row113 col9" >0.120311</td>
          <td id="T_10a98_row113_col10" class="data row113 col10" >0.000187</td>
          <td id="T_10a98_row113_col11" class="data row113 col11" >0.001056</td>
          <td id="T_10a98_row113_col12" class="data row113 col12" >6,10,14</td>
          <td id="T_10a98_row113_col13" class="data row113 col13" >0.197200</td>
          <td id="T_10a98_row113_col14" class="data row113 col14" >0.344958</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row114" class="row_heading level0 row114" >115</th>
          <td id="T_10a98_row114_col0" class="data row114 col0" >TMD-Pattern(...)-QIAN880113</td>
          <td id="T_10a98_row114_col1" class="data row114 col1" >Conformation</td>
          <td id="T_10a98_row114_col2" class="data row114 col2" >π-helix</td>
          <td id="T_10a98_row114_col3" class="data row114 col3" >α-helix (C-terminal)</td>
          <td id="T_10a98_row114_col4" class="data row114 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row114_col5" class="data row114 col5" >0.141000</td>
          <td id="T_10a98_row114_col6" class="data row114 col6" >0.070553</td>
          <td id="T_10a98_row114_col7" class="data row114 col7" >-0.070553</td>
          <td id="T_10a98_row114_col8" class="data row114 col8" >0.164819</td>
          <td id="T_10a98_row114_col9" class="data row114 col9" >0.154840</td>
          <td id="T_10a98_row114_col10" class="data row114 col10" >0.000217</td>
          <td id="T_10a98_row114_col11" class="data row114 col11" >0.001151</td>
          <td id="T_10a98_row114_col12" class="data row114 col12" >14,17</td>
          <td id="T_10a98_row114_col13" class="data row114 col13" >0.634800</td>
          <td id="T_10a98_row114_col14" class="data row114 col14" >0.816456</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row115" class="row_heading level0 row115" >116</th>
          <td id="T_10a98_row115_col0" class="data row115 col0" >JMD_N_TMD_N-...)-VASM830102</td>
          <td id="T_10a98_row115_col1" class="data row115 col1" >Energy</td>
          <td id="T_10a98_row115_col2" class="data row115 col2" >Non-bonded energy</td>
          <td id="T_10a98_row115_col3" class="data row115 col3" >Free energy (Extended)</td>
          <td id="T_10a98_row115_col4" class="data row115 col4" >Relative pop...t al., 1983)</td>
          <td id="T_10a98_row115_col5" class="data row115 col5" >0.141000</td>
          <td id="T_10a98_row115_col6" class="data row115 col6" >0.067593</td>
          <td id="T_10a98_row115_col7" class="data row115 col7" >0.067593</td>
          <td id="T_10a98_row115_col8" class="data row115 col8" >0.146572</td>
          <td id="T_10a98_row115_col9" class="data row115 col9" >0.140332</td>
          <td id="T_10a98_row115_col10" class="data row115 col10" >0.000225</td>
          <td id="T_10a98_row115_col11" class="data row115 col11" >0.001173</td>
          <td id="T_10a98_row115_col12" class="data row115 col12" >7,8,9,10</td>
          <td id="T_10a98_row115_col13" class="data row115 col13" >0.484800</td>
          <td id="T_10a98_row115_col14" class="data row115 col14" >0.832789</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row116" class="row_heading level0 row116" >117</th>
          <td id="T_10a98_row116_col0" class="data row116 col0" >TMD_C_JMD_C-...)-HUTJ700102</td>
          <td id="T_10a98_row116_col1" class="data row116 col1" >Energy</td>
          <td id="T_10a98_row116_col2" class="data row116 col2" >Entropy</td>
          <td id="T_10a98_row116_col3" class="data row116 col3" >Entropy</td>
          <td id="T_10a98_row116_col4" class="data row116 col4" >Absolute ent...chens, 1970)</td>
          <td id="T_10a98_row116_col5" class="data row116 col5" >0.140000</td>
          <td id="T_10a98_row116_col6" class="data row116 col6" >0.086678</td>
          <td id="T_10a98_row116_col7" class="data row116 col7" >0.086678</td>
          <td id="T_10a98_row116_col8" class="data row116 col8" >0.145086</td>
          <td id="T_10a98_row116_col9" class="data row116 col9" >0.183353</td>
          <td id="T_10a98_row116_col10" class="data row116 col10" >0.000250</td>
          <td id="T_10a98_row116_col11" class="data row116 col11" >0.001252</td>
          <td id="T_10a98_row116_col12" class="data row116 col12" >30,34</td>
          <td id="T_10a98_row116_col13" class="data row116 col13" >0.162800</td>
          <td id="T_10a98_row116_col14" class="data row116 col14" >0.333317</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row117" class="row_heading level0 row117" >118</th>
          <td id="T_10a98_row117_col0" class="data row117 col0" >TMD-Pattern(...)-FINA770101</td>
          <td id="T_10a98_row117_col1" class="data row117 col1" >Structure-Activity</td>
          <td id="T_10a98_row117_col2" class="data row117 col2" >Stability (helix-coil)</td>
          <td id="T_10a98_row117_col3" class="data row117 col3" >Stability (helix-coil)</td>
          <td id="T_10a98_row117_col4" class="data row117 col4" >Helix-coil e...itsyn, 1977)</td>
          <td id="T_10a98_row117_col5" class="data row117 col5" >0.140000</td>
          <td id="T_10a98_row117_col6" class="data row117 col6" >0.070899</td>
          <td id="T_10a98_row117_col7" class="data row117 col7" >0.070899</td>
          <td id="T_10a98_row117_col8" class="data row117 col8" >0.135187</td>
          <td id="T_10a98_row117_col9" class="data row117 col9" >0.153279</td>
          <td id="T_10a98_row117_col10" class="data row117 col10" >0.000233</td>
          <td id="T_10a98_row117_col11" class="data row117 col11" >0.001204</td>
          <td id="T_10a98_row117_col12" class="data row117 col12" >20,23,27</td>
          <td id="T_10a98_row117_col13" class="data row117 col13" >0.000000</td>
          <td id="T_10a98_row117_col14" class="data row117 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row118" class="row_heading level0 row118" >119</th>
          <td id="T_10a98_row118_col0" class="data row118 col0" >TMD_C_JMD_C-...)-MITS020101</td>
          <td id="T_10a98_row118_col1" class="data row118 col1" >Polarity</td>
          <td id="T_10a98_row118_col2" class="data row118 col2" >Amphiphilicity</td>
          <td id="T_10a98_row118_col3" class="data row118 col3" >Amphiphilicity</td>
          <td id="T_10a98_row118_col4" class="data row118 col4" >Amphiphilici...t al., 2002)</td>
          <td id="T_10a98_row118_col5" class="data row118 col5" >0.140000</td>
          <td id="T_10a98_row118_col6" class="data row118 col6" >0.066859</td>
          <td id="T_10a98_row118_col7" class="data row118 col7" >0.066859</td>
          <td id="T_10a98_row118_col8" class="data row118 col8" >0.130397</td>
          <td id="T_10a98_row118_col9" class="data row118 col9" >0.147129</td>
          <td id="T_10a98_row118_col10" class="data row118 col10" >0.000229</td>
          <td id="T_10a98_row118_col11" class="data row118 col11" >0.001185</td>
          <td id="T_10a98_row118_col12" class="data row118 col12" >33,37,40</td>
          <td id="T_10a98_row118_col13" class="data row118 col13" >0.334800</td>
          <td id="T_10a98_row118_col14" class="data row118 col14" >0.632640</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row119" class="row_heading level0 row119" >120</th>
          <td id="T_10a98_row119_col0" class="data row119 col0" >JMD_N_TMD_N-...)-ZHOH040101</td>
          <td id="T_10a98_row119_col1" class="data row119 col1" >Structure-Activity</td>
          <td id="T_10a98_row119_col2" class="data row119 col2" >Stability</td>
          <td id="T_10a98_row119_col3" class="data row119 col3" >Stability</td>
          <td id="T_10a98_row119_col4" class="data row119 col4" >The stabilit...-Zhou, 2004)</td>
          <td id="T_10a98_row119_col5" class="data row119 col5" >0.139000</td>
          <td id="T_10a98_row119_col6" class="data row119 col6" >0.070195</td>
          <td id="T_10a98_row119_col7" class="data row119 col7" >-0.070195</td>
          <td id="T_10a98_row119_col8" class="data row119 col8" >0.113589</td>
          <td id="T_10a98_row119_col9" class="data row119 col9" >0.146944</td>
          <td id="T_10a98_row119_col10" class="data row119 col10" >0.000259</td>
          <td id="T_10a98_row119_col11" class="data row119 col11" >0.001276</td>
          <td id="T_10a98_row119_col12" class="data row119 col12" >3,4,5</td>
          <td id="T_10a98_row119_col13" class="data row119 col13" >0.498800</td>
          <td id="T_10a98_row119_col14" class="data row119 col14" >0.924962</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row120" class="row_heading level0 row120" >121</th>
          <td id="T_10a98_row120_col0" class="data row120 col0" >TMD_C_JMD_C-...)-QIAN880114</td>
          <td id="T_10a98_row120_col1" class="data row120 col1" >Conformation</td>
          <td id="T_10a98_row120_col2" class="data row120 col2" >β-sheet (N-term)</td>
          <td id="T_10a98_row120_col3" class="data row120 col3" >β-sheet (N-terminal)</td>
          <td id="T_10a98_row120_col4" class="data row120 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row120_col5" class="data row120 col5" >0.138000</td>
          <td id="T_10a98_row120_col6" class="data row120 col6" >0.070821</td>
          <td id="T_10a98_row120_col7" class="data row120 col7" >-0.070821</td>
          <td id="T_10a98_row120_col8" class="data row120 col8" >0.121293</td>
          <td id="T_10a98_row120_col9" class="data row120 col9" >0.151868</td>
          <td id="T_10a98_row120_col10" class="data row120 col10" >0.000310</td>
          <td id="T_10a98_row120_col11" class="data row120 col11" >0.001400</td>
          <td id="T_10a98_row120_col12" class="data row120 col12" >24,28,32</td>
          <td id="T_10a98_row120_col13" class="data row120 col13" >0.718800</td>
          <td id="T_10a98_row120_col14" class="data row120 col14" >1.295090</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row121" class="row_heading level0 row121" >122</th>
          <td id="T_10a98_row121_col0" class="data row121 col0" >JMD_N_TMD_N-...)-RACS770103</td>
          <td id="T_10a98_row121_col1" class="data row121 col1" >ASA/Volume</td>
          <td id="T_10a98_row121_col2" class="data row121 col2" >Accessible s...e area (ASA)</td>
          <td id="T_10a98_row121_col3" class="data row121 col3" >Side chain orientation</td>
          <td id="T_10a98_row121_col4" class="data row121 col4" >Side chain o...eraga, 1977)</td>
          <td id="T_10a98_row121_col5" class="data row121 col5" >0.138000</td>
          <td id="T_10a98_row121_col6" class="data row121 col6" >0.069674</td>
          <td id="T_10a98_row121_col7" class="data row121 col7" >0.069674</td>
          <td id="T_10a98_row121_col8" class="data row121 col8" >0.151437</td>
          <td id="T_10a98_row121_col9" class="data row121 col9" >0.143090</td>
          <td id="T_10a98_row121_col10" class="data row121 col10" >0.000308</td>
          <td id="T_10a98_row121_col11" class="data row121 col11" >0.001398</td>
          <td id="T_10a98_row121_col12" class="data row121 col12" >4,5,6</td>
          <td id="T_10a98_row121_col13" class="data row121 col13" >0.214400</td>
          <td id="T_10a98_row121_col14" class="data row121 col14" >0.501327</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row122" class="row_heading level0 row122" >123</th>
          <td id="T_10a98_row122_col0" class="data row122 col0" >JMD_N_TMD_N-...)-HUTJ700102</td>
          <td id="T_10a98_row122_col1" class="data row122 col1" >Energy</td>
          <td id="T_10a98_row122_col2" class="data row122 col2" >Entropy</td>
          <td id="T_10a98_row122_col3" class="data row122 col3" >Entropy</td>
          <td id="T_10a98_row122_col4" class="data row122 col4" >Absolute ent...chens, 1970)</td>
          <td id="T_10a98_row122_col5" class="data row122 col5" >0.138000</td>
          <td id="T_10a98_row122_col6" class="data row122 col6" >0.061121</td>
          <td id="T_10a98_row122_col7" class="data row122 col7" >-0.061121</td>
          <td id="T_10a98_row122_col8" class="data row122 col8" >0.128877</td>
          <td id="T_10a98_row122_col9" class="data row122 col9" >0.124138</td>
          <td id="T_10a98_row122_col10" class="data row122 col10" >0.000304</td>
          <td id="T_10a98_row122_col11" class="data row122 col11" >0.001387</td>
          <td id="T_10a98_row122_col12" class="data row122 col12" >1,5,8,11</td>
          <td id="T_10a98_row122_col13" class="data row122 col13" >0.352400</td>
          <td id="T_10a98_row122_col14" class="data row122 col14" >0.457395</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row123" class="row_heading level0 row123" >124</th>
          <td id="T_10a98_row123_col0" class="data row123 col0" >TMD_C_JMD_C-...)-MONM990101</td>
          <td id="T_10a98_row123_col1" class="data row123 col1" >Conformation</td>
          <td id="T_10a98_row123_col2" class="data row123 col2" >β-turn (TM helix)</td>
          <td id="T_10a98_row123_col3" class="data row123 col3" >β-turn (TM helix)</td>
          <td id="T_10a98_row123_col4" class="data row123 col4" >Turn propens...t al., 1999)</td>
          <td id="T_10a98_row123_col5" class="data row123 col5" >0.137000</td>
          <td id="T_10a98_row123_col6" class="data row123 col6" >0.073116</td>
          <td id="T_10a98_row123_col7" class="data row123 col7" >0.073116</td>
          <td id="T_10a98_row123_col8" class="data row123 col8" >0.137320</td>
          <td id="T_10a98_row123_col9" class="data row123 col9" >0.169283</td>
          <td id="T_10a98_row123_col10" class="data row123 col10" >0.000297</td>
          <td id="T_10a98_row123_col11" class="data row123 col11" >0.001373</td>
          <td id="T_10a98_row123_col12" class="data row123 col12" >26,30,33</td>
          <td id="T_10a98_row123_col13" class="data row123 col13" >0.180400</td>
          <td id="T_10a98_row123_col14" class="data row123 col14" >0.297153</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row124" class="row_heading level0 row124" >125</th>
          <td id="T_10a98_row124_col0" class="data row124 col0" >TMD_C_JMD_C-...)-MEIH800101</td>
          <td id="T_10a98_row124_col1" class="data row124 col1" >Shape</td>
          <td id="T_10a98_row124_col2" class="data row124 col2" >Reduced distance</td>
          <td id="T_10a98_row124_col3" class="data row124 col3" >Reduced distance (C-α)</td>
          <td id="T_10a98_row124_col4" class="data row124 col4" >Average redu...t al., 1980)</td>
          <td id="T_10a98_row124_col5" class="data row124 col5" >0.137000</td>
          <td id="T_10a98_row124_col6" class="data row124 col6" >0.072970</td>
          <td id="T_10a98_row124_col7" class="data row124 col7" >-0.072970</td>
          <td id="T_10a98_row124_col8" class="data row124 col8" >0.120345</td>
          <td id="T_10a98_row124_col9" class="data row124 col9" >0.157741</td>
          <td id="T_10a98_row124_col10" class="data row124 col10" >0.000342</td>
          <td id="T_10a98_row124_col11" class="data row124 col11" >0.001501</td>
          <td id="T_10a98_row124_col12" class="data row124 col12" >28,29,30</td>
          <td id="T_10a98_row124_col13" class="data row124 col13" >0.298000</td>
          <td id="T_10a98_row124_col14" class="data row124 col14" >1.061395</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row125" class="row_heading level0 row125" >126</th>
          <td id="T_10a98_row125_col0" class="data row125 col0" >TMD_C_JMD_C-...)-QIAN880138</td>
          <td id="T_10a98_row125_col1" class="data row125 col1" >Conformation</td>
          <td id="T_10a98_row125_col2" class="data row125 col2" >Coil (C-term)</td>
          <td id="T_10a98_row125_col3" class="data row125 col3" >Coil (C-terminal)</td>
          <td id="T_10a98_row125_col4" class="data row125 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row125_col5" class="data row125 col5" >0.137000</td>
          <td id="T_10a98_row125_col6" class="data row125 col6" >0.065719</td>
          <td id="T_10a98_row125_col7" class="data row125 col7" >-0.065719</td>
          <td id="T_10a98_row125_col8" class="data row125 col8" >0.114425</td>
          <td id="T_10a98_row125_col9" class="data row125 col9" >0.146722</td>
          <td id="T_10a98_row125_col10" class="data row125 col10" >0.000312</td>
          <td id="T_10a98_row125_col11" class="data row125 col11" >0.001404</td>
          <td id="T_10a98_row125_col12" class="data row125 col12" >31,35,39</td>
          <td id="T_10a98_row125_col13" class="data row125 col13" >0.360800</td>
          <td id="T_10a98_row125_col14" class="data row125 col14" >0.882718</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row126" class="row_heading level0 row126" >127</th>
          <td id="T_10a98_row126_col0" class="data row126 col0" >JMD_N_TMD_N-...)-KARS160114</td>
          <td id="T_10a98_row126_col1" class="data row126 col1" >Shape</td>
          <td id="T_10a98_row126_col2" class="data row126 col2" >Side chain length</td>
          <td id="T_10a98_row126_col3" class="data row126 col3" >Eccentricity (average)</td>
          <td id="T_10a98_row126_col4" class="data row126 col4" >Average weig...isley, 2016)</td>
          <td id="T_10a98_row126_col5" class="data row126 col5" >0.137000</td>
          <td id="T_10a98_row126_col6" class="data row126 col6" >0.056352</td>
          <td id="T_10a98_row126_col7" class="data row126 col7" >-0.056352</td>
          <td id="T_10a98_row126_col8" class="data row126 col8" >0.122287</td>
          <td id="T_10a98_row126_col9" class="data row126 col9" >0.122893</td>
          <td id="T_10a98_row126_col10" class="data row126 col10" >0.000322</td>
          <td id="T_10a98_row126_col11" class="data row126 col11" >0.001432</td>
          <td id="T_10a98_row126_col12" class="data row126 col12" >16,17</td>
          <td id="T_10a98_row126_col13" class="data row126 col13" >1.170800</td>
          <td id="T_10a98_row126_col14" class="data row126 col14" >1.925978</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row127" class="row_heading level0 row127" >128</th>
          <td id="T_10a98_row127_col0" class="data row127 col0" >TMD_C_JMD_C-...)-ROSM880103</td>
          <td id="T_10a98_row127_col1" class="data row127 col1" >Structure-Activity</td>
          <td id="T_10a98_row127_col2" class="data row127 col2" >Backbone-dynamics (-CH)</td>
          <td id="T_10a98_row127_col3" class="data row127 col3" >Loss of hydr...ix formation</td>
          <td id="T_10a98_row127_col4" class="data row127 col4" >Loss of Side...seman, 1988)</td>
          <td id="T_10a98_row127_col5" class="data row127 col5" >0.136000</td>
          <td id="T_10a98_row127_col6" class="data row127 col6" >0.080537</td>
          <td id="T_10a98_row127_col7" class="data row127 col7" >-0.080537</td>
          <td id="T_10a98_row127_col8" class="data row127 col8" >0.194254</td>
          <td id="T_10a98_row127_col9" class="data row127 col9" >0.165343</td>
          <td id="T_10a98_row127_col10" class="data row127 col10" >0.000150</td>
          <td id="T_10a98_row127_col11" class="data row127 col11" >0.000932</td>
          <td id="T_10a98_row127_col12" class="data row127 col12" >26,27</td>
          <td id="T_10a98_row127_col13" class="data row127 col13" >0.638000</td>
          <td id="T_10a98_row127_col14" class="data row127 col14" >0.796859</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row128" class="row_heading level0 row128" >129</th>
          <td id="T_10a98_row128_col0" class="data row128 col0" >TMD_C_JMD_C-...)-BAEK050101</td>
          <td id="T_10a98_row128_col1" class="data row128 col1" >Conformation</td>
          <td id="T_10a98_row128_col2" class="data row128 col2" >β-strand</td>
          <td id="T_10a98_row128_col3" class="data row128 col3" >Linker index...long region)</td>
          <td id="T_10a98_row128_col4" class="data row128 col4" >Linker index...t al., 2005)</td>
          <td id="T_10a98_row128_col5" class="data row128 col5" >0.136000</td>
          <td id="T_10a98_row128_col6" class="data row128 col6" >0.072267</td>
          <td id="T_10a98_row128_col7" class="data row128 col7" >-0.072267</td>
          <td id="T_10a98_row128_col8" class="data row128 col8" >0.142246</td>
          <td id="T_10a98_row128_col9" class="data row128 col9" >0.173638</td>
          <td id="T_10a98_row128_col10" class="data row128 col10" >0.000352</td>
          <td id="T_10a98_row128_col11" class="data row128 col11" >0.001527</td>
          <td id="T_10a98_row128_col12" class="data row128 col12" >33,34</td>
          <td id="T_10a98_row128_col13" class="data row128 col13" >1.013200</td>
          <td id="T_10a98_row128_col14" class="data row128 col14" >1.315181</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row129" class="row_heading level0 row129" >130</th>
          <td id="T_10a98_row129_col0" class="data row129 col0" >TMD_C_JMD_C-...)-LINS030107</td>
          <td id="T_10a98_row129_col1" class="data row129 col1" >ASA/Volume</td>
          <td id="T_10a98_row129_col2" class="data row129 col2" >Accessible s...e area (ASA)</td>
          <td id="T_10a98_row129_col3" class="data row129 col3" >ASA (folded protein)</td>
          <td id="T_10a98_row129_col4" class="data row129 col4" >% total acce...t al., 2003)</td>
          <td id="T_10a98_row129_col5" class="data row129 col5" >0.136000</td>
          <td id="T_10a98_row129_col6" class="data row129 col6" >0.064864</td>
          <td id="T_10a98_row129_col7" class="data row129 col7" >-0.064864</td>
          <td id="T_10a98_row129_col8" class="data row129 col8" >0.078387</td>
          <td id="T_10a98_row129_col9" class="data row129 col9" >0.131618</td>
          <td id="T_10a98_row129_col10" class="data row129 col10" >0.000367</td>
          <td id="T_10a98_row129_col11" class="data row129 col11" >0.001565</td>
          <td id="T_10a98_row129_col12" class="data row129 col12" >25,28</td>
          <td id="T_10a98_row129_col13" class="data row129 col13" >0.842000</td>
          <td id="T_10a98_row129_col14" class="data row129 col14" >0.904274</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row130" class="row_heading level0 row130" >131</th>
          <td id="T_10a98_row130_col0" class="data row130 col0" >TMD_C_JMD_C-...)-PALJ810113</td>
          <td id="T_10a98_row130_col1" class="data row130 col1" >Conformation</td>
          <td id="T_10a98_row130_col2" class="data row130 col2" >α-helix (left-handed)</td>
          <td id="T_10a98_row130_col3" class="data row130 col3" >β-turn (α class)</td>
          <td id="T_10a98_row130_col4" class="data row130 col4" >Normalized f...t al., 1981)</td>
          <td id="T_10a98_row130_col5" class="data row130 col5" >0.135000</td>
          <td id="T_10a98_row130_col6" class="data row130 col6" >0.072992</td>
          <td id="T_10a98_row130_col7" class="data row130 col7" >-0.072992</td>
          <td id="T_10a98_row130_col8" class="data row130 col8" >0.138972</td>
          <td id="T_10a98_row130_col9" class="data row130 col9" >0.165851</td>
          <td id="T_10a98_row130_col10" class="data row130 col10" >0.000412</td>
          <td id="T_10a98_row130_col11" class="data row130 col11" >0.001667</td>
          <td id="T_10a98_row130_col12" class="data row130 col12" >32,33</td>
          <td id="T_10a98_row130_col13" class="data row130 col13" >0.292400</td>
          <td id="T_10a98_row130_col14" class="data row130 col14" >0.546994</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row131" class="row_heading level0 row131" >132</th>
          <td id="T_10a98_row131_col0" class="data row131 col0" >TMD_C_JMD_C-...)-YUTK870103</td>
          <td id="T_10a98_row131_col1" class="data row131 col1" >Energy</td>
          <td id="T_10a98_row131_col2" class="data row131 col2" >Free energy (unfolding)</td>
          <td id="T_10a98_row131_col3" class="data row131 col3" >Free energy (unfolding)</td>
          <td id="T_10a98_row131_col4" class="data row131 col4" >Activation G...t al., 1987)</td>
          <td id="T_10a98_row131_col5" class="data row131 col5" >0.135000</td>
          <td id="T_10a98_row131_col6" class="data row131 col6" >0.067074</td>
          <td id="T_10a98_row131_col7" class="data row131 col7" >-0.067074</td>
          <td id="T_10a98_row131_col8" class="data row131 col8" >0.127768</td>
          <td id="T_10a98_row131_col9" class="data row131 col9" >0.106936</td>
          <td id="T_10a98_row131_col10" class="data row131 col10" >0.000409</td>
          <td id="T_10a98_row131_col11" class="data row131 col11" >0.001662</td>
          <td id="T_10a98_row131_col12" class="data row131 col12" >27,28,29,30,31,32,33</td>
          <td id="T_10a98_row131_col13" class="data row131 col13" >0.000000</td>
          <td id="T_10a98_row131_col14" class="data row131 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row132" class="row_heading level0 row132" >133</th>
          <td id="T_10a98_row132_col0" class="data row132 col0" >JMD_N_TMD_N-...)-ZHOH040101</td>
          <td id="T_10a98_row132_col1" class="data row132 col1" >Structure-Activity</td>
          <td id="T_10a98_row132_col2" class="data row132 col2" >Stability</td>
          <td id="T_10a98_row132_col3" class="data row132 col3" >Stability</td>
          <td id="T_10a98_row132_col4" class="data row132 col4" >The stabilit...-Zhou, 2004)</td>
          <td id="T_10a98_row132_col5" class="data row132 col5" >0.135000</td>
          <td id="T_10a98_row132_col6" class="data row132 col6" >0.062723</td>
          <td id="T_10a98_row132_col7" class="data row132 col7" >-0.062723</td>
          <td id="T_10a98_row132_col8" class="data row132 col8" >0.120282</td>
          <td id="T_10a98_row132_col9" class="data row132 col9" >0.141044</td>
          <td id="T_10a98_row132_col10" class="data row132 col10" >0.000396</td>
          <td id="T_10a98_row132_col11" class="data row132 col11" >0.001638</td>
          <td id="T_10a98_row132_col12" class="data row132 col12" >3,6,9</td>
          <td id="T_10a98_row132_col13" class="data row132 col13" >0.696800</td>
          <td id="T_10a98_row132_col14" class="data row132 col14" >1.062095</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row133" class="row_heading level0 row133" >134</th>
          <td id="T_10a98_row133_col0" class="data row133 col0" >TMD-Pattern(...)-RADA880106</td>
          <td id="T_10a98_row133_col1" class="data row133 col1" >ASA/Volume</td>
          <td id="T_10a98_row133_col2" class="data row133 col2" >Volume</td>
          <td id="T_10a98_row133_col3" class="data row133 col3" >Accessible s...e area (ASA)</td>
          <td id="T_10a98_row133_col4" class="data row133 col4" >Accessible s...enden, 1988)</td>
          <td id="T_10a98_row133_col5" class="data row133 col5" >0.135000</td>
          <td id="T_10a98_row133_col6" class="data row133 col6" >0.058024</td>
          <td id="T_10a98_row133_col7" class="data row133 col7" >-0.058024</td>
          <td id="T_10a98_row133_col8" class="data row133 col8" >0.115415</td>
          <td id="T_10a98_row133_col9" class="data row133 col9" >0.124556</td>
          <td id="T_10a98_row133_col10" class="data row133 col10" >0.000385</td>
          <td id="T_10a98_row133_col11" class="data row133 col11" >0.001610</td>
          <td id="T_10a98_row133_col12" class="data row133 col12" >11,14,17</td>
          <td id="T_10a98_row133_col13" class="data row133 col13" >0.244400</td>
          <td id="T_10a98_row133_col14" class="data row133 col14" >0.503183</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row134" class="row_heading level0 row134" >135</th>
          <td id="T_10a98_row134_col0" class="data row134 col0" >JMD_N_TMD_N-...)-CRAJ730102</td>
          <td id="T_10a98_row134_col1" class="data row134 col1" >Conformation</td>
          <td id="T_10a98_row134_col2" class="data row134 col2" >β-sheet</td>
          <td id="T_10a98_row134_col3" class="data row134 col3" >β-sheet</td>
          <td id="T_10a98_row134_col4" class="data row134 col4" >Normalized f...t al., 1973)</td>
          <td id="T_10a98_row134_col5" class="data row134 col5" >0.134000</td>
          <td id="T_10a98_row134_col6" class="data row134 col6" >0.096792</td>
          <td id="T_10a98_row134_col7" class="data row134 col7" >-0.096792</td>
          <td id="T_10a98_row134_col8" class="data row134 col8" >0.182935</td>
          <td id="T_10a98_row134_col9" class="data row134 col9" >0.210285</td>
          <td id="T_10a98_row134_col10" class="data row134 col10" >0.000461</td>
          <td id="T_10a98_row134_col11" class="data row134 col11" >0.001775</td>
          <td id="T_10a98_row134_col12" class="data row134 col12" >5,6</td>
          <td id="T_10a98_row134_col13" class="data row134 col13" >0.485600</td>
          <td id="T_10a98_row134_col14" class="data row134 col14" >0.792949</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row135" class="row_heading level0 row135" >136</th>
          <td id="T_10a98_row135_col0" class="data row135 col0" >JMD_N_TMD_N-...)-QIAN880134</td>
          <td id="T_10a98_row135_col1" class="data row135 col1" >Conformation</td>
          <td id="T_10a98_row135_col2" class="data row135 col2" >Coil</td>
          <td id="T_10a98_row135_col3" class="data row135 col3" >Coil</td>
          <td id="T_10a98_row135_col4" class="data row135 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row135_col5" class="data row135 col5" >0.133000</td>
          <td id="T_10a98_row135_col6" class="data row135 col6" >0.071020</td>
          <td id="T_10a98_row135_col7" class="data row135 col7" >0.071020</td>
          <td id="T_10a98_row135_col8" class="data row135 col8" >0.161372</td>
          <td id="T_10a98_row135_col9" class="data row135 col9" >0.138873</td>
          <td id="T_10a98_row135_col10" class="data row135 col10" >0.000491</td>
          <td id="T_10a98_row135_col11" class="data row135 col11" >0.001836</td>
          <td id="T_10a98_row135_col12" class="data row135 col12" >6,10,14</td>
          <td id="T_10a98_row135_col13" class="data row135 col13" >0.000000</td>
          <td id="T_10a98_row135_col14" class="data row135 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row136" class="row_heading level0 row136" >137</th>
          <td id="T_10a98_row136_col0" class="data row136 col0" >JMD_N_TMD_N-...)-RICJ880111</td>
          <td id="T_10a98_row136_col1" class="data row136 col1" >Conformation</td>
          <td id="T_10a98_row136_col2" class="data row136 col2" >π-helix</td>
          <td id="T_10a98_row136_col3" class="data row136 col3" >α-helix</td>
          <td id="T_10a98_row136_col4" class="data row136 col4" >Relative pre...rdson, 1988)</td>
          <td id="T_10a98_row136_col5" class="data row136 col5" >0.133000</td>
          <td id="T_10a98_row136_col6" class="data row136 col6" >0.054446</td>
          <td id="T_10a98_row136_col7" class="data row136 col7" >-0.054446</td>
          <td id="T_10a98_row136_col8" class="data row136 col8" >0.113232</td>
          <td id="T_10a98_row136_col9" class="data row136 col9" >0.124195</td>
          <td id="T_10a98_row136_col10" class="data row136 col10" >0.000466</td>
          <td id="T_10a98_row136_col11" class="data row136 col11" >0.001785</td>
          <td id="T_10a98_row136_col12" class="data row136 col12" >9,12,15</td>
          <td id="T_10a98_row136_col13" class="data row136 col13" >0.113200</td>
          <td id="T_10a98_row136_col14" class="data row136 col14" >0.232615</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row137" class="row_heading level0 row137" >138</th>
          <td id="T_10a98_row137_col0" class="data row137 col0" >JMD_N_TMD_N-...)-BURA740102</td>
          <td id="T_10a98_row137_col1" class="data row137 col1" >Conformation</td>
          <td id="T_10a98_row137_col2" class="data row137 col2" >β-strand</td>
          <td id="T_10a98_row137_col3" class="data row137 col3" >Extended</td>
          <td id="T_10a98_row137_col4" class="data row137 col4" >Normalized f...t al., 1974)</td>
          <td id="T_10a98_row137_col5" class="data row137 col5" >0.132000</td>
          <td id="T_10a98_row137_col6" class="data row137 col6" >0.056043</td>
          <td id="T_10a98_row137_col7" class="data row137 col7" >0.056043</td>
          <td id="T_10a98_row137_col8" class="data row137 col8" >0.119813</td>
          <td id="T_10a98_row137_col9" class="data row137 col9" >0.123454</td>
          <td id="T_10a98_row137_col10" class="data row137 col10" >0.000562</td>
          <td id="T_10a98_row137_col11" class="data row137 col11" >0.001981</td>
          <td id="T_10a98_row137_col12" class="data row137 col12" >14,15</td>
          <td id="T_10a98_row137_col13" class="data row137 col13" >0.231600</td>
          <td id="T_10a98_row137_col14" class="data row137 col14" >0.356019</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row138" class="row_heading level0 row138" >139</th>
          <td id="T_10a98_row138_col0" class="data row138 col0" >TMD-Segment(...)-TANS770102</td>
          <td id="T_10a98_row138_col1" class="data row138 col1" >Conformation</td>
          <td id="T_10a98_row138_col2" class="data row138 col2" >α-helix (C-term, out)</td>
          <td id="T_10a98_row138_col3" class="data row138 col3" >α-helix (C-t...al, outside)</td>
          <td id="T_10a98_row138_col4" class="data row138 col4" >Normalized f...eraga, 1977)</td>
          <td id="T_10a98_row138_col5" class="data row138 col5" >0.132000</td>
          <td id="T_10a98_row138_col6" class="data row138 col6" >0.055783</td>
          <td id="T_10a98_row138_col7" class="data row138 col7" >-0.055783</td>
          <td id="T_10a98_row138_col8" class="data row138 col8" >0.129933</td>
          <td id="T_10a98_row138_col9" class="data row138 col9" >0.133383</td>
          <td id="T_10a98_row138_col10" class="data row138 col10" >0.000558</td>
          <td id="T_10a98_row138_col11" class="data row138 col11" >0.001977</td>
          <td id="T_10a98_row138_col12" class="data row138 col12" >16,17</td>
          <td id="T_10a98_row138_col13" class="data row138 col13" >0.502400</td>
          <td id="T_10a98_row138_col14" class="data row138 col14" >0.761626</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row139" class="row_heading level0 row139" >140</th>
          <td id="T_10a98_row139_col0" class="data row139 col0" >TMD-Pattern(...)-QIAN880124</td>
          <td id="T_10a98_row139_col1" class="data row139 col1" >Conformation</td>
          <td id="T_10a98_row139_col2" class="data row139 col2" >β-sheet (C-term)</td>
          <td id="T_10a98_row139_col3" class="data row139 col3" >β-sheet (C-terminal)</td>
          <td id="T_10a98_row139_col4" class="data row139 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row139_col5" class="data row139 col5" >0.131000</td>
          <td id="T_10a98_row139_col6" class="data row139 col6" >0.069857</td>
          <td id="T_10a98_row139_col7" class="data row139 col7" >0.069857</td>
          <td id="T_10a98_row139_col8" class="data row139 col8" >0.157078</td>
          <td id="T_10a98_row139_col9" class="data row139 col9" >0.159138</td>
          <td id="T_10a98_row139_col10" class="data row139 col10" >0.000580</td>
          <td id="T_10a98_row139_col11" class="data row139 col11" >0.002008</td>
          <td id="T_10a98_row139_col12" class="data row139 col12" >11,14,17,20</td>
          <td id="T_10a98_row139_col13" class="data row139 col13" >0.502800</td>
          <td id="T_10a98_row139_col14" class="data row139 col14" >0.811308</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row140" class="row_heading level0 row140" >141</th>
          <td id="T_10a98_row140_col0" class="data row140 col0" >TMD_C_JMD_C-...)-TANS770106</td>
          <td id="T_10a98_row140_col1" class="data row140 col1" >Conformation</td>
          <td id="T_10a98_row140_col2" class="data row140 col2" >β-turn (TM helix)</td>
          <td id="T_10a98_row140_col3" class="data row140 col3" >β-turn in double bend</td>
          <td id="T_10a98_row140_col4" class="data row140 col4" >Normalized f...eraga, 1977)</td>
          <td id="T_10a98_row140_col5" class="data row140 col5" >0.131000</td>
          <td id="T_10a98_row140_col6" class="data row140 col6" >0.056621</td>
          <td id="T_10a98_row140_col7" class="data row140 col7" >0.056621</td>
          <td id="T_10a98_row140_col8" class="data row140 col8" >0.144377</td>
          <td id="T_10a98_row140_col9" class="data row140 col9" >0.128425</td>
          <td id="T_10a98_row140_col10" class="data row140 col10" >0.000597</td>
          <td id="T_10a98_row140_col11" class="data row140 col11" >0.002043</td>
          <td id="T_10a98_row140_col12" class="data row140 col12" >31,34,38</td>
          <td id="T_10a98_row140_col13" class="data row140 col13" >0.726800</td>
          <td id="T_10a98_row140_col14" class="data row140 col14" >0.885807</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row141" class="row_heading level0 row141" >142</th>
          <td id="T_10a98_row141_col0" class="data row141 col0" >JMD_N_TMD_N-...)-ANDN920101</td>
          <td id="T_10a98_row141_col1" class="data row141 col1" >Structure-Activity</td>
          <td id="T_10a98_row141_col2" class="data row141 col2" >Backbone-dynamics (-CH)</td>
          <td id="T_10a98_row141_col3" class="data row141 col3" >α-CH chemica...ne-dynamics)</td>
          <td id="T_10a98_row141_col4" class="data row141 col4" >alpha-CH che...t al., 1992)</td>
          <td id="T_10a98_row141_col5" class="data row141 col5" >0.130000</td>
          <td id="T_10a98_row141_col6" class="data row141 col6" >0.087733</td>
          <td id="T_10a98_row141_col7" class="data row141 col7" >-0.087733</td>
          <td id="T_10a98_row141_col8" class="data row141 col8" >0.180612</td>
          <td id="T_10a98_row141_col9" class="data row141 col9" >0.187328</td>
          <td id="T_10a98_row141_col10" class="data row141 col10" >0.000674</td>
          <td id="T_10a98_row141_col11" class="data row141 col11" >0.002190</td>
          <td id="T_10a98_row141_col12" class="data row141 col12" >10,13,17</td>
          <td id="T_10a98_row141_col13" class="data row141 col13" >0.420000</td>
          <td id="T_10a98_row141_col14" class="data row141 col14" >0.643453</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row142" class="row_heading level0 row142" >143</th>
          <td id="T_10a98_row142_col0" class="data row142 col0" >JMD_N_TMD_N-...)-BIOV880101</td>
          <td id="T_10a98_row142_col1" class="data row142 col1" >ASA/Volume</td>
          <td id="T_10a98_row142_col2" class="data row142 col2" >Buried</td>
          <td id="T_10a98_row142_col3" class="data row142 col3" >Buriability</td>
          <td id="T_10a98_row142_col4" class="data row142 col4" >Information ...t al., 1988)</td>
          <td id="T_10a98_row142_col5" class="data row142 col5" >0.130000</td>
          <td id="T_10a98_row142_col6" class="data row142 col6" >0.067433</td>
          <td id="T_10a98_row142_col7" class="data row142 col7" >-0.067433</td>
          <td id="T_10a98_row142_col8" class="data row142 col8" >0.133237</td>
          <td id="T_10a98_row142_col9" class="data row142 col9" >0.146065</td>
          <td id="T_10a98_row142_col10" class="data row142 col10" >0.000642</td>
          <td id="T_10a98_row142_col11" class="data row142 col11" >0.002130</td>
          <td id="T_10a98_row142_col12" class="data row142 col12" >7,11,14</td>
          <td id="T_10a98_row142_col13" class="data row142 col13" >0.306800</td>
          <td id="T_10a98_row142_col14" class="data row142 col14" >0.574245</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row143" class="row_heading level0 row143" >144</th>
          <td id="T_10a98_row143_col0" class="data row143 col0" >JMD_N_TMD_N-...)-QIAN880114</td>
          <td id="T_10a98_row143_col1" class="data row143 col1" >Conformation</td>
          <td id="T_10a98_row143_col2" class="data row143 col2" >β-sheet (N-term)</td>
          <td id="T_10a98_row143_col3" class="data row143 col3" >β-sheet (N-terminal)</td>
          <td id="T_10a98_row143_col4" class="data row143 col4" >Weights for ...owski, 1988)</td>
          <td id="T_10a98_row143_col5" class="data row143 col5" >0.130000</td>
          <td id="T_10a98_row143_col6" class="data row143 col6" >0.058210</td>
          <td id="T_10a98_row143_col7" class="data row143 col7" >0.058210</td>
          <td id="T_10a98_row143_col8" class="data row143 col8" >0.127516</td>
          <td id="T_10a98_row143_col9" class="data row143 col9" >0.112411</td>
          <td id="T_10a98_row143_col10" class="data row143 col10" >0.000633</td>
          <td id="T_10a98_row143_col11" class="data row143 col11" >0.002111</td>
          <td id="T_10a98_row143_col12" class="data row143 col12" >6,7,8,9,10</td>
          <td id="T_10a98_row143_col13" class="data row143 col13" >0.140800</td>
          <td id="T_10a98_row143_col14" class="data row143 col14" >0.376807</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row144" class="row_heading level0 row144" >145</th>
          <td id="T_10a98_row144_col0" class="data row144 col0" >JMD_N_TMD_N-...)-VASM830102</td>
          <td id="T_10a98_row144_col1" class="data row144 col1" >Energy</td>
          <td id="T_10a98_row144_col2" class="data row144 col2" >Non-bonded energy</td>
          <td id="T_10a98_row144_col3" class="data row144 col3" >Free energy (Extended)</td>
          <td id="T_10a98_row144_col4" class="data row144 col4" >Relative pop...t al., 1983)</td>
          <td id="T_10a98_row144_col5" class="data row144 col5" >0.129000</td>
          <td id="T_10a98_row144_col6" class="data row144 col6" >0.077724</td>
          <td id="T_10a98_row144_col7" class="data row144 col7" >0.077724</td>
          <td id="T_10a98_row144_col8" class="data row144 col8" >0.148907</td>
          <td id="T_10a98_row144_col9" class="data row144 col9" >0.164954</td>
          <td id="T_10a98_row144_col10" class="data row144 col10" >0.000708</td>
          <td id="T_10a98_row144_col11" class="data row144 col11" >0.002247</td>
          <td id="T_10a98_row144_col12" class="data row144 col12" >4,8,11</td>
          <td id="T_10a98_row144_col13" class="data row144 col13" >0.160400</td>
          <td id="T_10a98_row144_col14" class="data row144 col14" >0.302939</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row145" class="row_heading level0 row145" >146</th>
          <td id="T_10a98_row145_col0" class="data row145 col0" >TMD_C_JMD_C-...)-MAXF760105</td>
          <td id="T_10a98_row145_col1" class="data row145 col1" >Conformation</td>
          <td id="T_10a98_row145_col2" class="data row145 col2" >α-helix (left-handed)</td>
          <td id="T_10a98_row145_col3" class="data row145 col3" >α-helix (left-handed)</td>
          <td id="T_10a98_row145_col4" class="data row145 col4" >Normalized f...eraga, 1976)</td>
          <td id="T_10a98_row145_col5" class="data row145 col5" >0.129000</td>
          <td id="T_10a98_row145_col6" class="data row145 col6" >0.071374</td>
          <td id="T_10a98_row145_col7" class="data row145 col7" >0.071374</td>
          <td id="T_10a98_row145_col8" class="data row145 col8" >0.180851</td>
          <td id="T_10a98_row145_col9" class="data row145 col9" >0.152571</td>
          <td id="T_10a98_row145_col10" class="data row145 col10" >0.000727</td>
          <td id="T_10a98_row145_col11" class="data row145 col11" >0.002285</td>
          <td id="T_10a98_row145_col12" class="data row145 col12" >23,27</td>
          <td id="T_10a98_row145_col13" class="data row145 col13" >0.000000</td>
          <td id="T_10a98_row145_col14" class="data row145 col14" >0.000000</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row146" class="row_heading level0 row146" >147</th>
          <td id="T_10a98_row146_col0" class="data row146 col0" >JMD_N_TMD_N-...)-KOEP990102</td>
          <td id="T_10a98_row146_col1" class="data row146 col1" >Conformation</td>
          <td id="T_10a98_row146_col2" class="data row146 col2" >β-sheet (N-term)</td>
          <td id="T_10a98_row146_col3" class="data row146 col3" >Extended (de...ned β-sheet)</td>
          <td id="T_10a98_row146_col4" class="data row146 col4" >Beta-sheet p...evitt, 1999)</td>
          <td id="T_10a98_row146_col5" class="data row146 col5" >0.128000</td>
          <td id="T_10a98_row146_col6" class="data row146 col6" >0.086726</td>
          <td id="T_10a98_row146_col7" class="data row146 col7" >0.086726</td>
          <td id="T_10a98_row146_col8" class="data row146 col8" >0.184173</td>
          <td id="T_10a98_row146_col9" class="data row146 col9" >0.184291</td>
          <td id="T_10a98_row146_col10" class="data row146 col10" >0.000769</td>
          <td id="T_10a98_row146_col11" class="data row146 col11" >0.002364</td>
          <td id="T_10a98_row146_col12" class="data row146 col12" >5,6</td>
          <td id="T_10a98_row146_col13" class="data row146 col13" >0.565600</td>
          <td id="T_10a98_row146_col14" class="data row146 col14" >0.778424</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row147" class="row_heading level0 row147" >148</th>
          <td id="T_10a98_row147_col0" class="data row147 col0" >TMD_C_JMD_C-...)-TANS770102</td>
          <td id="T_10a98_row147_col1" class="data row147 col1" >Conformation</td>
          <td id="T_10a98_row147_col2" class="data row147 col2" >α-helix (C-term, out)</td>
          <td id="T_10a98_row147_col3" class="data row147 col3" >α-helix (C-t...al, outside)</td>
          <td id="T_10a98_row147_col4" class="data row147 col4" >Normalized f...eraga, 1977)</td>
          <td id="T_10a98_row147_col5" class="data row147 col5" >0.128000</td>
          <td id="T_10a98_row147_col6" class="data row147 col6" >0.062708</td>
          <td id="T_10a98_row147_col7" class="data row147 col7" >-0.062708</td>
          <td id="T_10a98_row147_col8" class="data row147 col8" >0.113629</td>
          <td id="T_10a98_row147_col9" class="data row147 col9" >0.123346</td>
          <td id="T_10a98_row147_col10" class="data row147 col10" >0.000767</td>
          <td id="T_10a98_row147_col11" class="data row147 col11" >0.002362</td>
          <td id="T_10a98_row147_col12" class="data row147 col12" >25,28,31</td>
          <td id="T_10a98_row147_col13" class="data row147 col13" >0.407200</td>
          <td id="T_10a98_row147_col14" class="data row147 col14" >0.686822</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row148" class="row_heading level0 row148" >149</th>
          <td id="T_10a98_row148_col0" class="data row148 col0" >JMD_N_TMD_N-...)-CHAM830105</td>
          <td id="T_10a98_row148_col1" class="data row148 col1" >Shape</td>
          <td id="T_10a98_row148_col2" class="data row148 col2" >Side chain length</td>
          <td id="T_10a98_row148_col3" class="data row148 col3" >n atoms in s... chain (3+1)</td>
          <td id="T_10a98_row148_col4" class="data row148 col4" >The number o...arton, 1983)</td>
          <td id="T_10a98_row148_col5" class="data row148 col5" >0.128000</td>
          <td id="T_10a98_row148_col6" class="data row148 col6" >0.057140</td>
          <td id="T_10a98_row148_col7" class="data row148 col7" >-0.057140</td>
          <td id="T_10a98_row148_col8" class="data row148 col8" >0.128493</td>
          <td id="T_10a98_row148_col9" class="data row148 col9" >0.130946</td>
          <td id="T_10a98_row148_col10" class="data row148 col10" >0.000672</td>
          <td id="T_10a98_row148_col11" class="data row148 col11" >0.002187</td>
          <td id="T_10a98_row148_col12" class="data row148 col12" >7,8,9,10,11,12,13</td>
          <td id="T_10a98_row148_col13" class="data row148 col13" >0.121600</td>
          <td id="T_10a98_row148_col14" class="data row148 col14" >0.273037</td>
        </tr>
        <tr>
          <th id="T_10a98_level0_row149" class="row_heading level0 row149" >150</th>
          <td id="T_10a98_row149_col0" class="data row149 col0" >JMD_N_TMD_N-...)-ISOY800102</td>
          <td id="T_10a98_row149_col1" class="data row149 col1" >Conformation</td>
          <td id="T_10a98_row149_col2" class="data row149 col2" >β-strand</td>
          <td id="T_10a98_row149_col3" class="data row149 col3" >Extended</td>
          <td id="T_10a98_row149_col4" class="data row149 col4" >Normalized r...t al., 1980)</td>
          <td id="T_10a98_row149_col5" class="data row149 col5" >0.126000</td>
          <td id="T_10a98_row149_col6" class="data row149 col6" >0.079975</td>
          <td id="T_10a98_row149_col7" class="data row149 col7" >-0.079975</td>
          <td id="T_10a98_row149_col8" class="data row149 col8" >0.169167</td>
          <td id="T_10a98_row149_col9" class="data row149 col9" >0.182954</td>
          <td id="T_10a98_row149_col10" class="data row149 col10" >0.000926</td>
          <td id="T_10a98_row149_col11" class="data row149 col11" >0.002636</td>
          <td id="T_10a98_row149_col12" class="data row149 col12" >5,6</td>
          <td id="T_10a98_row149_col13" class="data row149 col13" >1.002000</td>
          <td id="T_10a98_row149_col14" class="data row149 col14" >1.075427</td>
        </tr>
      </tbody>
    </table>



Feature sets are made available for datasets that have been rigorously
tested and documented through scientific research.

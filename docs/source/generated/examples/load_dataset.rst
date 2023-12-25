An overview dataset table is provided as default:

.. code:: ipython2

    import aaanalysis as aa
    df_info = aa.load_dataset()
    print(df_info)
    aa.display_df(df=df_info, show_shape=True)


.. parsed-literal::

             Level        Dataset  # Sequences  # Amino acids  # Positives  \
    0   Amino acid    AA_CASPASE3          233         185605          705   
    1   Amino acid       AA_FURIN           71          59003          163   
    2   Amino acid         AA_LDR          342         118248        35469   
    3   Amino acid        AA_MMP2          573         312976         2416   
    4   Amino acid     AA_RNABIND          221          55001         6492   
    5   Amino acid          AA_SA          233         185605       101082   
    6     Sequence      SEQ_AMYLO         1414           8484          511   
    7     Sequence     SEQ_CAPSID         7935        3364680         3864   
    8     Sequence  SEQ_DISULFIDE         2547         614470          897   
    9     Sequence   SEQ_LOCATION         1835         732398         1045   
    10    Sequence    SEQ_SOLUBLE        17408        4432269         8704   
    11    Sequence       SEQ_TAIL         6668        2671690         2574   
    12      Domain       DOM_GSEC          126          92964           63   
    13      Domain    DOM_GSEC_PU          694         494524           63   
    
        # Negatives    Predictor  \
    0        184900   PROSPERous   
    1         58840   PROSPERous   
    2         82779  IDP-Seq2Seq   
    3        310560   PROSPERous   
    4         48509    GMKSVM-RU   
    5         84523   PROSPERous   
    6           903    ReRF-Pred   
    7          4071     VIRALpro   
    8          1650        Dipro   
    9           790          NaN   
    10         8704       SOLpro   
    11         4094     VIRALpro   
    12           63          NaN   
    13            0          NaN   
    
                                              Description              Reference  \
    0               Prediction of caspase-3 cleavage site      Song et al., 2018   
    1                   Prediction of furin cleavage site      Song et al., 2018   
    2   Prediction of long intrinsically disordered re...      Tang et al., 2020   
    3   Prediction of Matrix metallopeptidase-2 (MMP2)...      Song et al., 2018   
    4   Prediction of RNA-binding protein residues (RB...      Yang et al., 2021   
    5   Prediction of solvent accessibility (SA) of re...      Song et al., 2018   
    6                Prediction of amyloidognenic regions       Teng et al. 2021   
    7                      Prediction of capdsid proteins    Galiez et al., 2016   
    8        Prediction of disulfide bridges in sequences     Cheng et al., 2006   
    9   Prediction of subcellular location of protein ...      Shen et al., 2019   
    10       Prediction of soluble and insoluble proteins    Magnan et al., 2009   
    11                        Prediction of tail proteins    Galiez et al., 2016   
    12           Prediction of gamma-secretase substrates  Breimann et al, 2024c   
    13  Prediction of gamma-secretase substrates (PU d...  Breimann et al, 2024c   
    
                                                    Label  
    0   1 (adjacent to cleavage site), 0 (not adjacent...  
    1   1 (adjacent to cleavage site), 0 (not adjacent...  
    2                         1 (disordered), 0 (ordered)  
    3   1 (adjacent to cleavage site), 0 (not adjacent...  
    4                        1 (binding), 0 (non-binding)  
    5   1 (exposed/accessible), 0 (buried/non-accessible)  
    6            1 (amyloidogenic), 0 (non-amyloidogenic)  
    7          1 (capsid protein), 0 (non-capsid protein)  
    8   1 (sequence with SS bond), 0 (sequence without...  
    9   1 (protein in cytoplasm), 0 (protein in plasma...  
    10                         1 (soluble), 0 (insoluble)  
    11             1 (tail protein), 0 (non-tail protein)  
    12                   1 (substrate), 0 (non-substrate)  
    13        1 (substrate), 2 (unknown substrate status)  
    DataFrame shape: (14, 10)



.. raw:: html

    <style type="text/css">
    #T_31b87 thead th {
      background-color: white;
      color: black;
    }
    #T_31b87 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_31b87 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_31b87 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_31b87  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_31b87 table {
      font-size: 12px;
    }
    </style>
    <table id="T_31b87" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_31b87_level0_col0" class="col_heading level0 col0" >Level</th>
          <th id="T_31b87_level0_col1" class="col_heading level0 col1" >Dataset</th>
          <th id="T_31b87_level0_col2" class="col_heading level0 col2" ># Sequences</th>
          <th id="T_31b87_level0_col3" class="col_heading level0 col3" ># Amino acids</th>
          <th id="T_31b87_level0_col4" class="col_heading level0 col4" ># Positives</th>
          <th id="T_31b87_level0_col5" class="col_heading level0 col5" ># Negatives</th>
          <th id="T_31b87_level0_col6" class="col_heading level0 col6" >Predictor</th>
          <th id="T_31b87_level0_col7" class="col_heading level0 col7" >Description</th>
          <th id="T_31b87_level0_col8" class="col_heading level0 col8" >Reference</th>
          <th id="T_31b87_level0_col9" class="col_heading level0 col9" >Label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_31b87_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_31b87_row0_col0" class="data row0 col0" >Amino acid</td>
          <td id="T_31b87_row0_col1" class="data row0 col1" >AA_CASPASE3</td>
          <td id="T_31b87_row0_col2" class="data row0 col2" >233</td>
          <td id="T_31b87_row0_col3" class="data row0 col3" >185605</td>
          <td id="T_31b87_row0_col4" class="data row0 col4" >705</td>
          <td id="T_31b87_row0_col5" class="data row0 col5" >184900</td>
          <td id="T_31b87_row0_col6" class="data row0 col6" >PROSPERous</td>
          <td id="T_31b87_row0_col7" class="data row0 col7" >Prediction o...leavage site</td>
          <td id="T_31b87_row0_col8" class="data row0 col8" >Song et al., 2018</td>
          <td id="T_31b87_row0_col9" class="data row0 col9" >1 (adjacent ...eavage site)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_31b87_row1_col0" class="data row1 col0" >Amino acid</td>
          <td id="T_31b87_row1_col1" class="data row1 col1" >AA_FURIN</td>
          <td id="T_31b87_row1_col2" class="data row1 col2" >71</td>
          <td id="T_31b87_row1_col3" class="data row1 col3" >59003</td>
          <td id="T_31b87_row1_col4" class="data row1 col4" >163</td>
          <td id="T_31b87_row1_col5" class="data row1 col5" >58840</td>
          <td id="T_31b87_row1_col6" class="data row1 col6" >PROSPERous</td>
          <td id="T_31b87_row1_col7" class="data row1 col7" >Prediction o...leavage site</td>
          <td id="T_31b87_row1_col8" class="data row1 col8" >Song et al., 2018</td>
          <td id="T_31b87_row1_col9" class="data row1 col9" >1 (adjacent ...eavage site)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_31b87_row2_col0" class="data row2 col0" >Amino acid</td>
          <td id="T_31b87_row2_col1" class="data row2 col1" >AA_LDR</td>
          <td id="T_31b87_row2_col2" class="data row2 col2" >342</td>
          <td id="T_31b87_row2_col3" class="data row2 col3" >118248</td>
          <td id="T_31b87_row2_col4" class="data row2 col4" >35469</td>
          <td id="T_31b87_row2_col5" class="data row2 col5" >82779</td>
          <td id="T_31b87_row2_col6" class="data row2 col6" >IDP-Seq2Seq</td>
          <td id="T_31b87_row2_col7" class="data row2 col7" >Prediction o...egions (LDR)</td>
          <td id="T_31b87_row2_col8" class="data row2 col8" >Tang et al., 2020</td>
          <td id="T_31b87_row2_col9" class="data row2 col9" >1 (disordere... 0 (ordered)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_31b87_row3_col0" class="data row3 col0" >Amino acid</td>
          <td id="T_31b87_row3_col1" class="data row3 col1" >AA_MMP2</td>
          <td id="T_31b87_row3_col2" class="data row3 col2" >573</td>
          <td id="T_31b87_row3_col3" class="data row3 col3" >312976</td>
          <td id="T_31b87_row3_col4" class="data row3 col4" >2416</td>
          <td id="T_31b87_row3_col5" class="data row3 col5" >310560</td>
          <td id="T_31b87_row3_col6" class="data row3 col6" >PROSPERous</td>
          <td id="T_31b87_row3_col7" class="data row3 col7" >Prediction o...leavage site</td>
          <td id="T_31b87_row3_col8" class="data row3 col8" >Song et al., 2018</td>
          <td id="T_31b87_row3_col9" class="data row3 col9" >1 (adjacent ...eavage site)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_31b87_row4_col0" class="data row4 col0" >Amino acid</td>
          <td id="T_31b87_row4_col1" class="data row4 col1" >AA_RNABIND</td>
          <td id="T_31b87_row4_col2" class="data row4 col2" >221</td>
          <td id="T_31b87_row4_col3" class="data row4 col3" >55001</td>
          <td id="T_31b87_row4_col4" class="data row4 col4" >6492</td>
          <td id="T_31b87_row4_col5" class="data row4 col5" >48509</td>
          <td id="T_31b87_row4_col6" class="data row4 col6" >GMKSVM-RU</td>
          <td id="T_31b87_row4_col7" class="data row4 col7" >Prediction o...P60 dataset)</td>
          <td id="T_31b87_row4_col8" class="data row4 col8" >Yang et al., 2021</td>
          <td id="T_31b87_row4_col9" class="data row4 col9" >1 (binding),...non-binding)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_31b87_row5_col0" class="data row5 col0" >Amino acid</td>
          <td id="T_31b87_row5_col1" class="data row5 col1" >AA_SA</td>
          <td id="T_31b87_row5_col2" class="data row5 col2" >233</td>
          <td id="T_31b87_row5_col3" class="data row5 col3" >185605</td>
          <td id="T_31b87_row5_col4" class="data row5 col4" >101082</td>
          <td id="T_31b87_row5_col5" class="data row5 col5" >84523</td>
          <td id="T_31b87_row5_col6" class="data row5 col6" >PROSPERous</td>
          <td id="T_31b87_row5_col7" class="data row5 col7" >Prediction o...E3 data set)</td>
          <td id="T_31b87_row5_col8" class="data row5 col8" >Song et al., 2018</td>
          <td id="T_31b87_row5_col9" class="data row5 col9" >1 (exposed/a...-accessible)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_31b87_row6_col0" class="data row6 col0" >Sequence</td>
          <td id="T_31b87_row6_col1" class="data row6 col1" >SEQ_AMYLO</td>
          <td id="T_31b87_row6_col2" class="data row6 col2" >1414</td>
          <td id="T_31b87_row6_col3" class="data row6 col3" >8484</td>
          <td id="T_31b87_row6_col4" class="data row6 col4" >511</td>
          <td id="T_31b87_row6_col5" class="data row6 col5" >903</td>
          <td id="T_31b87_row6_col6" class="data row6 col6" >ReRF-Pred</td>
          <td id="T_31b87_row6_col7" class="data row6 col7" >Prediction o...enic regions</td>
          <td id="T_31b87_row6_col8" class="data row6 col8" >Teng et al. 2021</td>
          <td id="T_31b87_row6_col9" class="data row6 col9" >1 (amyloidog...yloidogenic)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_31b87_row7_col0" class="data row7 col0" >Sequence</td>
          <td id="T_31b87_row7_col1" class="data row7 col1" >SEQ_CAPSID</td>
          <td id="T_31b87_row7_col2" class="data row7 col2" >7935</td>
          <td id="T_31b87_row7_col3" class="data row7 col3" >3364680</td>
          <td id="T_31b87_row7_col4" class="data row7 col4" >3864</td>
          <td id="T_31b87_row7_col5" class="data row7 col5" >4071</td>
          <td id="T_31b87_row7_col6" class="data row7 col6" >VIRALpro</td>
          <td id="T_31b87_row7_col7" class="data row7 col7" >Prediction o...sid proteins</td>
          <td id="T_31b87_row7_col8" class="data row7 col8" >Galiez et al., 2016</td>
          <td id="T_31b87_row7_col9" class="data row7 col9" >1 (capsid pr...sid protein)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_31b87_row8_col0" class="data row8 col0" >Sequence</td>
          <td id="T_31b87_row8_col1" class="data row8 col1" >SEQ_DISULFIDE</td>
          <td id="T_31b87_row8_col2" class="data row8 col2" >2547</td>
          <td id="T_31b87_row8_col3" class="data row8 col3" >614470</td>
          <td id="T_31b87_row8_col4" class="data row8 col4" >897</td>
          <td id="T_31b87_row8_col5" class="data row8 col5" >1650</td>
          <td id="T_31b87_row8_col6" class="data row8 col6" >Dipro</td>
          <td id="T_31b87_row8_col7" class="data row8 col7" >Prediction o...in sequences</td>
          <td id="T_31b87_row8_col8" class="data row8 col8" >Cheng et al., 2006</td>
          <td id="T_31b87_row8_col9" class="data row8 col9" >1 (sequence ...out SS bond)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_31b87_row9_col0" class="data row9 col0" >Sequence</td>
          <td id="T_31b87_row9_col1" class="data row9 col1" >SEQ_LOCATION</td>
          <td id="T_31b87_row9_col2" class="data row9 col2" >1835</td>
          <td id="T_31b87_row9_col3" class="data row9 col3" >732398</td>
          <td id="T_31b87_row9_col4" class="data row9 col4" >1045</td>
          <td id="T_31b87_row9_col5" class="data row9 col5" >790</td>
          <td id="T_31b87_row9_col6" class="data row9 col6" >nan</td>
          <td id="T_31b87_row9_col7" class="data row9 col7" >Prediction o...ma membrane)</td>
          <td id="T_31b87_row9_col8" class="data row9 col8" >Shen et al., 2019</td>
          <td id="T_31b87_row9_col9" class="data row9 col9" >1 (protein i...a membrane) </td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row10" class="row_heading level0 row10" >11</th>
          <td id="T_31b87_row10_col0" class="data row10 col0" >Sequence</td>
          <td id="T_31b87_row10_col1" class="data row10 col1" >SEQ_SOLUBLE</td>
          <td id="T_31b87_row10_col2" class="data row10 col2" >17408</td>
          <td id="T_31b87_row10_col3" class="data row10 col3" >4432269</td>
          <td id="T_31b87_row10_col4" class="data row10 col4" >8704</td>
          <td id="T_31b87_row10_col5" class="data row10 col5" >8704</td>
          <td id="T_31b87_row10_col6" class="data row10 col6" >SOLpro</td>
          <td id="T_31b87_row10_col7" class="data row10 col7" >Prediction o...ble proteins</td>
          <td id="T_31b87_row10_col8" class="data row10 col8" >Magnan et al., 2009</td>
          <td id="T_31b87_row10_col9" class="data row10 col9" >1 (soluble),... (insoluble)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row11" class="row_heading level0 row11" >12</th>
          <td id="T_31b87_row11_col0" class="data row11 col0" >Sequence</td>
          <td id="T_31b87_row11_col1" class="data row11 col1" >SEQ_TAIL</td>
          <td id="T_31b87_row11_col2" class="data row11 col2" >6668</td>
          <td id="T_31b87_row11_col3" class="data row11 col3" >2671690</td>
          <td id="T_31b87_row11_col4" class="data row11 col4" >2574</td>
          <td id="T_31b87_row11_col5" class="data row11 col5" >4094</td>
          <td id="T_31b87_row11_col6" class="data row11 col6" >VIRALpro</td>
          <td id="T_31b87_row11_col7" class="data row11 col7" >Prediction o...ail proteins</td>
          <td id="T_31b87_row11_col8" class="data row11 col8" >Galiez et al., 2016</td>
          <td id="T_31b87_row11_col9" class="data row11 col9" >1 (tail prot...ail protein)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row12" class="row_heading level0 row12" >13</th>
          <td id="T_31b87_row12_col0" class="data row12 col0" >Domain</td>
          <td id="T_31b87_row12_col1" class="data row12 col1" >DOM_GSEC</td>
          <td id="T_31b87_row12_col2" class="data row12 col2" >126</td>
          <td id="T_31b87_row12_col3" class="data row12 col3" >92964</td>
          <td id="T_31b87_row12_col4" class="data row12 col4" >63</td>
          <td id="T_31b87_row12_col5" class="data row12 col5" >63</td>
          <td id="T_31b87_row12_col6" class="data row12 col6" >nan</td>
          <td id="T_31b87_row12_col7" class="data row12 col7" >Prediction o...e substrates</td>
          <td id="T_31b87_row12_col8" class="data row12 col8" >Breimann et al, 2024c</td>
          <td id="T_31b87_row12_col9" class="data row12 col9" >1 (substrate...n-substrate)</td>
        </tr>
        <tr>
          <th id="T_31b87_level0_row13" class="row_heading level0 row13" >14</th>
          <td id="T_31b87_row13_col0" class="data row13 col0" >Domain</td>
          <td id="T_31b87_row13_col1" class="data row13 col1" >DOM_GSEC_PU</td>
          <td id="T_31b87_row13_col2" class="data row13 col2" >694</td>
          <td id="T_31b87_row13_col3" class="data row13 col3" >494524</td>
          <td id="T_31b87_row13_col4" class="data row13 col4" >63</td>
          <td id="T_31b87_row13_col5" class="data row13 col5" >0</td>
          <td id="T_31b87_row13_col6" class="data row13 col6" >nan</td>
          <td id="T_31b87_row13_col7" class="data row13 col7" >Prediction o...(PU dataset)</td>
          <td id="T_31b87_row13_col8" class="data row13 col8" >Breimann et al, 2024c</td>
          <td id="T_31b87_row13_col9" class="data row13 col9" >1 (substrate...rate status)</td>
        </tr>
      </tbody>
    </table>



Load one of the datasets from the overview table by using a name from the 'Dataset' column (e.g., 'SEQ_CAPSID'). The number of proteins per class can be adjusted by the 'n' parameter:

.. code:: ipython2

    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=2)
    aa.display_df(df=df_seq, char_limit=40)



.. raw:: html

    <style type="text/css">
    #T_07840 thead th {
      background-color: white;
      color: black;
    }
    #T_07840 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_07840 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_07840 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_07840  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_07840 table {
      font-size: 12px;
    }
    </style>
    <table id="T_07840" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_07840_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_07840_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_07840_level0_col2" class="col_heading level0 col2" >label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_07840_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_07840_row0_col0" class="data row0 col0" >CAPSID_1</td>
          <td id="T_07840_row0_col1" class="data row0 col1" >MVTHNVKINKHVTRRSYSSA...KGDDDDTPRIPATKLDEENV</td>
          <td id="T_07840_row0_col2" class="data row0 col2" >0</td>
        </tr>
        <tr>
          <th id="T_07840_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_07840_row1_col0" class="data row1 col0" >CAPSID_2</td>
          <td id="T_07840_row1_col1" class="data row1 col1" >MKKRQKKMTLSNFTDTSFQD...VFMRMAMLEAVINARHFGEE</td>
          <td id="T_07840_row1_col2" class="data row1 col2" >0</td>
        </tr>
        <tr>
          <th id="T_07840_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_07840_row2_col0" class="data row2 col0" >CAPSID_4072</td>
          <td id="T_07840_row2_col1" class="data row2 col1" >MALTTNDVITEDFVRETVEE...IFTRKAWKAIFPEAAVKVDA</td>
          <td id="T_07840_row2_col2" class="data row2 col2" >1</td>
        </tr>
        <tr>
          <th id="T_07840_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_07840_row3_col0" class="data row3 col0" >CAPSID_4073</td>
          <td id="T_07840_row3_col1" class="data row3 col1" >MGELTDNGVQLAKAQIGKHQ...TIGQLTCTNPAAHAKIRDLK</td>
          <td id="T_07840_row3_col2" class="data row3 col2" >1</td>
        </tr>
      </tbody>
    </table>



Samples for amino acid ('AA') level datasets are provided by specyfing their amino acid window size using the  TODO ...

.. code:: ipython2

    df_aa = aa.load_dataset(name="AA_CASPASE3", n=2)
    aa.display_df(df=df_aa)



.. raw:: html

    <style type="text/css">
    #T_32a39 thead th {
      background-color: white;
      color: black;
    }
    #T_32a39 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_32a39 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_32a39 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_32a39  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_32a39 table {
      font-size: 12px;
    }
    </style>
    <table id="T_32a39" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_32a39_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_32a39_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_32a39_level0_col2" class="col_heading level0 col2" >label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_32a39_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_32a39_row0_col0" class="data row0 col0" >CASPASE3_1_pos4</td>
          <td id="T_32a39_row0_col1" class="data row0 col1" >MSLFDLFRG</td>
          <td id="T_32a39_row0_col2" class="data row0 col2" >0</td>
        </tr>
        <tr>
          <th id="T_32a39_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_32a39_row1_col0" class="data row1 col0" >CASPASE3_1_pos5</td>
          <td id="T_32a39_row1_col1" class="data row1 col1" >SLFDLFRGF</td>
          <td id="T_32a39_row1_col2" class="data row1 col2" >0</td>
        </tr>
        <tr>
          <th id="T_32a39_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_32a39_row2_col0" class="data row2 col0" >CASPASE3_1_pos126</td>
          <td id="T_32a39_row2_col1" class="data row2 col1" >QTLRDSMLK</td>
          <td id="T_32a39_row2_col2" class="data row2 col2" >1</td>
        </tr>
        <tr>
          <th id="T_32a39_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_32a39_row3_col0" class="data row3 col0" >CASPASE3_1_pos127</td>
          <td id="T_32a39_row3_col1" class="data row3 col1" >TLRDSMLKY</td>
          <td id="T_32a39_row3_col2" class="data row3 col2" >1</td>
        </tr>
      </tbody>
    </table>



.. code:: ipython2

    for name in df_info["Dataset"]:
        n_unfiltered = len(aa.load_dataset(name=name, non_canonical_aa="keep"))
        n =len(aa.load_dataset(name=name))
        print(name, n_unfiltered, n)    


.. parsed-literal::

    AA_CASPASE3 183741 183741
    AA_FURIN 58435 58435
    AA_LDR 115512 115512
    AA_MMP2 308392 308392
    AA_RNABIND 53233 53233
    AA_SA 183741 183741
    SEQ_AMYLO 1414 1414
    SEQ_CAPSID 7935 7862
    SEQ_DISULFIDE 2547 2202
    SEQ_LOCATION 1835 1835
    SEQ_SOLUBLE 17408 16902
    SEQ_TAIL 6668 6640
    DOM_GSEC 126 126
    DOM_GSEC_PU 694 694



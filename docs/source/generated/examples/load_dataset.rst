An overview dataset table is provided as default, where the suffix in
the ‘Dataset’ (‘AA’, ‘SEQ’, and ‘DOM’) column corresponds to the ‘Level’
values (‘Amino acid’, ‘Sequence’, and ‘Domain’ level). Load datasets
using the ``load_dataset()`` function:

.. code:: ipython2

    import aaanalysis as aa
    df_info = aa.load_dataset()
    aa.display_df(df=df_info, show_shape=True)


.. parsed-literal::

    DataFrame shape: (14, 10)



.. raw:: html

    <style type="text/css">
    #T_d83ba thead th {
      background-color: white;
      color: black;
    }
    #T_d83ba tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_d83ba tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_d83ba th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_d83ba  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_d83ba table {
      font-size: 12px;
    }
    </style>
    <table id="T_d83ba" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_d83ba_level0_col0" class="col_heading level0 col0" >Level</th>
          <th id="T_d83ba_level0_col1" class="col_heading level0 col1" >Dataset</th>
          <th id="T_d83ba_level0_col2" class="col_heading level0 col2" ># Sequences</th>
          <th id="T_d83ba_level0_col3" class="col_heading level0 col3" ># Amino acids</th>
          <th id="T_d83ba_level0_col4" class="col_heading level0 col4" ># Positives</th>
          <th id="T_d83ba_level0_col5" class="col_heading level0 col5" ># Negatives</th>
          <th id="T_d83ba_level0_col6" class="col_heading level0 col6" >Predictor</th>
          <th id="T_d83ba_level0_col7" class="col_heading level0 col7" >Description</th>
          <th id="T_d83ba_level0_col8" class="col_heading level0 col8" >Reference</th>
          <th id="T_d83ba_level0_col9" class="col_heading level0 col9" >Label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_d83ba_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_d83ba_row0_col0" class="data row0 col0" >Amino acid</td>
          <td id="T_d83ba_row0_col1" class="data row0 col1" >AA_CASPASE3</td>
          <td id="T_d83ba_row0_col2" class="data row0 col2" >233</td>
          <td id="T_d83ba_row0_col3" class="data row0 col3" >185605</td>
          <td id="T_d83ba_row0_col4" class="data row0 col4" >705</td>
          <td id="T_d83ba_row0_col5" class="data row0 col5" >184900</td>
          <td id="T_d83ba_row0_col6" class="data row0 col6" >PROSPERous</td>
          <td id="T_d83ba_row0_col7" class="data row0 col7" >Prediction o...leavage site</td>
          <td id="T_d83ba_row0_col8" class="data row0 col8" >Song et al., 2018</td>
          <td id="T_d83ba_row0_col9" class="data row0 col9" >1 (adjacent ...eavage site)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_d83ba_row1_col0" class="data row1 col0" >Amino acid</td>
          <td id="T_d83ba_row1_col1" class="data row1 col1" >AA_FURIN</td>
          <td id="T_d83ba_row1_col2" class="data row1 col2" >71</td>
          <td id="T_d83ba_row1_col3" class="data row1 col3" >59003</td>
          <td id="T_d83ba_row1_col4" class="data row1 col4" >163</td>
          <td id="T_d83ba_row1_col5" class="data row1 col5" >58840</td>
          <td id="T_d83ba_row1_col6" class="data row1 col6" >PROSPERous</td>
          <td id="T_d83ba_row1_col7" class="data row1 col7" >Prediction o...leavage site</td>
          <td id="T_d83ba_row1_col8" class="data row1 col8" >Song et al., 2018</td>
          <td id="T_d83ba_row1_col9" class="data row1 col9" >1 (adjacent ...eavage site)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_d83ba_row2_col0" class="data row2 col0" >Amino acid</td>
          <td id="T_d83ba_row2_col1" class="data row2 col1" >AA_LDR</td>
          <td id="T_d83ba_row2_col2" class="data row2 col2" >342</td>
          <td id="T_d83ba_row2_col3" class="data row2 col3" >118248</td>
          <td id="T_d83ba_row2_col4" class="data row2 col4" >35469</td>
          <td id="T_d83ba_row2_col5" class="data row2 col5" >82779</td>
          <td id="T_d83ba_row2_col6" class="data row2 col6" >IDP-Seq2Seq</td>
          <td id="T_d83ba_row2_col7" class="data row2 col7" >Prediction o...egions (LDR)</td>
          <td id="T_d83ba_row2_col8" class="data row2 col8" >Tang et al., 2020</td>
          <td id="T_d83ba_row2_col9" class="data row2 col9" >1 (disordere... 0 (ordered)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_d83ba_row3_col0" class="data row3 col0" >Amino acid</td>
          <td id="T_d83ba_row3_col1" class="data row3 col1" >AA_MMP2</td>
          <td id="T_d83ba_row3_col2" class="data row3 col2" >573</td>
          <td id="T_d83ba_row3_col3" class="data row3 col3" >312976</td>
          <td id="T_d83ba_row3_col4" class="data row3 col4" >2416</td>
          <td id="T_d83ba_row3_col5" class="data row3 col5" >310560</td>
          <td id="T_d83ba_row3_col6" class="data row3 col6" >PROSPERous</td>
          <td id="T_d83ba_row3_col7" class="data row3 col7" >Prediction o...leavage site</td>
          <td id="T_d83ba_row3_col8" class="data row3 col8" >Song et al., 2018</td>
          <td id="T_d83ba_row3_col9" class="data row3 col9" >1 (adjacent ...eavage site)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_d83ba_row4_col0" class="data row4 col0" >Amino acid</td>
          <td id="T_d83ba_row4_col1" class="data row4 col1" >AA_RNABIND</td>
          <td id="T_d83ba_row4_col2" class="data row4 col2" >221</td>
          <td id="T_d83ba_row4_col3" class="data row4 col3" >55001</td>
          <td id="T_d83ba_row4_col4" class="data row4 col4" >6492</td>
          <td id="T_d83ba_row4_col5" class="data row4 col5" >48509</td>
          <td id="T_d83ba_row4_col6" class="data row4 col6" >GMKSVM-RU</td>
          <td id="T_d83ba_row4_col7" class="data row4 col7" >Prediction o...P60 dataset)</td>
          <td id="T_d83ba_row4_col8" class="data row4 col8" >Yang et al., 2021</td>
          <td id="T_d83ba_row4_col9" class="data row4 col9" >1 (binding),...non-binding)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_d83ba_row5_col0" class="data row5 col0" >Amino acid</td>
          <td id="T_d83ba_row5_col1" class="data row5 col1" >AA_SA</td>
          <td id="T_d83ba_row5_col2" class="data row5 col2" >233</td>
          <td id="T_d83ba_row5_col3" class="data row5 col3" >185605</td>
          <td id="T_d83ba_row5_col4" class="data row5 col4" >101082</td>
          <td id="T_d83ba_row5_col5" class="data row5 col5" >84523</td>
          <td id="T_d83ba_row5_col6" class="data row5 col6" >PROSPERous</td>
          <td id="T_d83ba_row5_col7" class="data row5 col7" >Prediction o...E3 data set)</td>
          <td id="T_d83ba_row5_col8" class="data row5 col8" >Song et al., 2018</td>
          <td id="T_d83ba_row5_col9" class="data row5 col9" >1 (exposed/a...-accessible)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_d83ba_row6_col0" class="data row6 col0" >Sequence</td>
          <td id="T_d83ba_row6_col1" class="data row6 col1" >SEQ_AMYLO</td>
          <td id="T_d83ba_row6_col2" class="data row6 col2" >1414</td>
          <td id="T_d83ba_row6_col3" class="data row6 col3" >8484</td>
          <td id="T_d83ba_row6_col4" class="data row6 col4" >511</td>
          <td id="T_d83ba_row6_col5" class="data row6 col5" >903</td>
          <td id="T_d83ba_row6_col6" class="data row6 col6" >ReRF-Pred</td>
          <td id="T_d83ba_row6_col7" class="data row6 col7" >Prediction o...enic regions</td>
          <td id="T_d83ba_row6_col8" class="data row6 col8" >Teng et al. 2021</td>
          <td id="T_d83ba_row6_col9" class="data row6 col9" >1 (amyloidog...yloidogenic)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_d83ba_row7_col0" class="data row7 col0" >Sequence</td>
          <td id="T_d83ba_row7_col1" class="data row7 col1" >SEQ_CAPSID</td>
          <td id="T_d83ba_row7_col2" class="data row7 col2" >7935</td>
          <td id="T_d83ba_row7_col3" class="data row7 col3" >3364680</td>
          <td id="T_d83ba_row7_col4" class="data row7 col4" >3864</td>
          <td id="T_d83ba_row7_col5" class="data row7 col5" >4071</td>
          <td id="T_d83ba_row7_col6" class="data row7 col6" >VIRALpro</td>
          <td id="T_d83ba_row7_col7" class="data row7 col7" >Prediction o...sid proteins</td>
          <td id="T_d83ba_row7_col8" class="data row7 col8" >Galiez et al., 2016</td>
          <td id="T_d83ba_row7_col9" class="data row7 col9" >1 (capsid pr...sid protein)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_d83ba_row8_col0" class="data row8 col0" >Sequence</td>
          <td id="T_d83ba_row8_col1" class="data row8 col1" >SEQ_DISULFIDE</td>
          <td id="T_d83ba_row8_col2" class="data row8 col2" >2547</td>
          <td id="T_d83ba_row8_col3" class="data row8 col3" >614470</td>
          <td id="T_d83ba_row8_col4" class="data row8 col4" >897</td>
          <td id="T_d83ba_row8_col5" class="data row8 col5" >1650</td>
          <td id="T_d83ba_row8_col6" class="data row8 col6" >Dipro</td>
          <td id="T_d83ba_row8_col7" class="data row8 col7" >Prediction o...in sequences</td>
          <td id="T_d83ba_row8_col8" class="data row8 col8" >Cheng et al., 2006</td>
          <td id="T_d83ba_row8_col9" class="data row8 col9" >1 (sequence ...out SS bond)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_d83ba_row9_col0" class="data row9 col0" >Sequence</td>
          <td id="T_d83ba_row9_col1" class="data row9 col1" >SEQ_LOCATION</td>
          <td id="T_d83ba_row9_col2" class="data row9 col2" >1835</td>
          <td id="T_d83ba_row9_col3" class="data row9 col3" >732398</td>
          <td id="T_d83ba_row9_col4" class="data row9 col4" >1045</td>
          <td id="T_d83ba_row9_col5" class="data row9 col5" >790</td>
          <td id="T_d83ba_row9_col6" class="data row9 col6" >nan</td>
          <td id="T_d83ba_row9_col7" class="data row9 col7" >Prediction o...ma membrane)</td>
          <td id="T_d83ba_row9_col8" class="data row9 col8" >Shen et al., 2019</td>
          <td id="T_d83ba_row9_col9" class="data row9 col9" >1 (protein i...a membrane) </td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row10" class="row_heading level0 row10" >11</th>
          <td id="T_d83ba_row10_col0" class="data row10 col0" >Sequence</td>
          <td id="T_d83ba_row10_col1" class="data row10 col1" >SEQ_SOLUBLE</td>
          <td id="T_d83ba_row10_col2" class="data row10 col2" >17408</td>
          <td id="T_d83ba_row10_col3" class="data row10 col3" >4432269</td>
          <td id="T_d83ba_row10_col4" class="data row10 col4" >8704</td>
          <td id="T_d83ba_row10_col5" class="data row10 col5" >8704</td>
          <td id="T_d83ba_row10_col6" class="data row10 col6" >SOLpro</td>
          <td id="T_d83ba_row10_col7" class="data row10 col7" >Prediction o...ble proteins</td>
          <td id="T_d83ba_row10_col8" class="data row10 col8" >Magnan et al., 2009</td>
          <td id="T_d83ba_row10_col9" class="data row10 col9" >1 (soluble),... (insoluble)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row11" class="row_heading level0 row11" >12</th>
          <td id="T_d83ba_row11_col0" class="data row11 col0" >Sequence</td>
          <td id="T_d83ba_row11_col1" class="data row11 col1" >SEQ_TAIL</td>
          <td id="T_d83ba_row11_col2" class="data row11 col2" >6668</td>
          <td id="T_d83ba_row11_col3" class="data row11 col3" >2671690</td>
          <td id="T_d83ba_row11_col4" class="data row11 col4" >2574</td>
          <td id="T_d83ba_row11_col5" class="data row11 col5" >4094</td>
          <td id="T_d83ba_row11_col6" class="data row11 col6" >VIRALpro</td>
          <td id="T_d83ba_row11_col7" class="data row11 col7" >Prediction o...ail proteins</td>
          <td id="T_d83ba_row11_col8" class="data row11 col8" >Galiez et al., 2016</td>
          <td id="T_d83ba_row11_col9" class="data row11 col9" >1 (tail prot...ail protein)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row12" class="row_heading level0 row12" >13</th>
          <td id="T_d83ba_row12_col0" class="data row12 col0" >Domain</td>
          <td id="T_d83ba_row12_col1" class="data row12 col1" >DOM_GSEC</td>
          <td id="T_d83ba_row12_col2" class="data row12 col2" >126</td>
          <td id="T_d83ba_row12_col3" class="data row12 col3" >92964</td>
          <td id="T_d83ba_row12_col4" class="data row12 col4" >63</td>
          <td id="T_d83ba_row12_col5" class="data row12 col5" >63</td>
          <td id="T_d83ba_row12_col6" class="data row12 col6" >nan</td>
          <td id="T_d83ba_row12_col7" class="data row12 col7" >Prediction o...e substrates</td>
          <td id="T_d83ba_row12_col8" class="data row12 col8" >Breimann et al, 2024c</td>
          <td id="T_d83ba_row12_col9" class="data row12 col9" >1 (substrate...n-substrate)</td>
        </tr>
        <tr>
          <th id="T_d83ba_level0_row13" class="row_heading level0 row13" >14</th>
          <td id="T_d83ba_row13_col0" class="data row13 col0" >Domain</td>
          <td id="T_d83ba_row13_col1" class="data row13 col1" >DOM_GSEC_PU</td>
          <td id="T_d83ba_row13_col2" class="data row13 col2" >694</td>
          <td id="T_d83ba_row13_col3" class="data row13 col3" >494524</td>
          <td id="T_d83ba_row13_col4" class="data row13 col4" >63</td>
          <td id="T_d83ba_row13_col5" class="data row13 col5" >0</td>
          <td id="T_d83ba_row13_col6" class="data row13 col6" >nan</td>
          <td id="T_d83ba_row13_col7" class="data row13 col7" >Prediction o...(PU dataset)</td>
          <td id="T_d83ba_row13_col8" class="data row13 col8" >Breimann et al, 2024c</td>
          <td id="T_d83ba_row13_col9" class="data row13 col9" >1 (substrate...rate status)</td>
        </tr>
      </tbody>
    </table>



Load one of the datasets from the overview table by using a name from
the ‘Dataset’ column (e.g., ``name='SEQ_CAPSID'``). The number of
proteins per class can be adjusted by the ``n`` parameter:

.. code:: ipython2

    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=2)
    aa.display_df(df=df_seq)



.. raw:: html

    <style type="text/css">
    #T_e40e0 thead th {
      background-color: white;
      color: black;
    }
    #T_e40e0 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_e40e0 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_e40e0 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_e40e0  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_e40e0 table {
      font-size: 12px;
    }
    </style>
    <table id="T_e40e0" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_e40e0_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_e40e0_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_e40e0_level0_col2" class="col_heading level0 col2" >label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_e40e0_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_e40e0_row0_col0" class="data row0 col0" >CAPSID_1</td>
          <td id="T_e40e0_row0_col1" class="data row0 col1" >MVTHNVKINKHV...RIPATKLDEENV</td>
          <td id="T_e40e0_row0_col2" class="data row0 col2" >0</td>
        </tr>
        <tr>
          <th id="T_e40e0_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_e40e0_row1_col0" class="data row1 col0" >CAPSID_2</td>
          <td id="T_e40e0_row1_col1" class="data row1 col1" >MKKRQKKMTLSN...EAVINARHFGEE</td>
          <td id="T_e40e0_row1_col2" class="data row1 col2" >0</td>
        </tr>
        <tr>
          <th id="T_e40e0_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_e40e0_row2_col0" class="data row2 col0" >CAPSID_4072</td>
          <td id="T_e40e0_row2_col1" class="data row2 col1" >MALTTNDVITED...AIFPEAAVKVDA</td>
          <td id="T_e40e0_row2_col2" class="data row2 col2" >1</td>
        </tr>
        <tr>
          <th id="T_e40e0_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_e40e0_row3_col0" class="data row3 col0" >CAPSID_4073</td>
          <td id="T_e40e0_row3_col1" class="data row3 col1" >MGELTDNGVQLA...NPAAHAKIRDLK</td>
          <td id="T_e40e0_row3_col2" class="data row3 col2" >1</td>
        </tr>
      </tbody>
    </table>



The sampling can be performed randomly by setting ``random=True``:

.. code:: ipython2

    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=2, random=True)
    aa.display_df(df=df_seq)



.. raw:: html

    <style type="text/css">
    #T_7f31d thead th {
      background-color: white;
      color: black;
    }
    #T_7f31d tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_7f31d tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_7f31d th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_7f31d  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_7f31d table {
      font-size: 12px;
    }
    </style>
    <table id="T_7f31d" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_7f31d_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_7f31d_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_7f31d_level0_col2" class="col_heading level0 col2" >label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_7f31d_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_7f31d_row0_col0" class="data row0 col0" >CAPSID_1899</td>
          <td id="T_7f31d_row0_col1" class="data row0 col1" >MLSINPNEQTEK...KLGEQFELVRPI</td>
          <td id="T_7f31d_row0_col2" class="data row0 col2" >0</td>
        </tr>
        <tr>
          <th id="T_7f31d_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_7f31d_row1_col0" class="data row1 col0" >CAPSID_61</td>
          <td id="T_7f31d_row1_col1" class="data row1 col1" >MLIEDEPNIIEA...LDQVRALMAETS</td>
          <td id="T_7f31d_row1_col2" class="data row1 col2" >0</td>
        </tr>
        <tr>
          <th id="T_7f31d_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_7f31d_row2_col0" class="data row2 col0" >CAPSID_4286</td>
          <td id="T_7f31d_row2_col1" class="data row2 col1" >MNPADHPSVYVA...VDVFINQMMAHQ</td>
          <td id="T_7f31d_row2_col2" class="data row2 col2" >1</td>
        </tr>
        <tr>
          <th id="T_7f31d_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_7f31d_row3_col0" class="data row3 col0" >CAPSID_4135</td>
          <td id="T_7f31d_row3_col1" class="data row3 col1" >MSASHGVLTVPR...LLRRRKRRYMWN</td>
          <td id="T_7f31d_row3_col2" class="data row3 col2" >1</td>
        </tr>
      </tbody>
    </table>



Sequences with non-canonical amino acids are by default removed, which
can be disabled by setting ``non_canonical_aa='keep'`` or
``non_canonical_aa='gap'``:

.. code:: ipython2

    n_unfiltered = len(aa.load_dataset(name='SEQ_DISULFIDE', non_canonical_aa="keep"))
    n = len(aa.load_dataset(name='SEQ_DISULFIDE'))
    print(f"'SEQ_DISULFIDE' contain {n_unfiltered} proteins and {n} after filtering.")    


.. parsed-literal::

    'SEQ_DISULFIDE' contain 2547 proteins and 2202 after filtering.


Datasets can be filtered for the minimum and maximum sequence length
using ``min_len`` and ``max_len``:

.. code:: ipython2

    n_len_filtered = len(aa.load_dataset(name='SEQ_DISULFIDE', min_len=100, max_len=200))
    print(f"'SEQ_DISULFIDE' contain {n_unfiltered} proteins, of which {n_len_filtered} have a length between 100 and 200 residues.")   



.. parsed-literal::

    'SEQ_DISULFIDE' contain 2547 proteins, of which 644 have a length between 100 and 200 residues.


For the ‘Amino acid level’ datasets, the size of the amino acid window
can be adjusted using the ``aa_window_size`` parameter:

.. code:: ipython2

    df_aa = aa.load_dataset(name="AA_CASPASE3", n=2, aa_window_size=5)
    aa.display_df(df=df_aa)



.. raw:: html

    <style type="text/css">
    #T_0e439 thead th {
      background-color: white;
      color: black;
    }
    #T_0e439 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_0e439 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_0e439 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_0e439  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_0e439 table {
      font-size: 12px;
    }
    </style>
    <table id="T_0e439" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_0e439_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_0e439_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_0e439_level0_col2" class="col_heading level0 col2" >label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_0e439_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_0e439_row0_col0" class="data row0 col0" >CASPASE3_1_pos126</td>
          <td id="T_0e439_row0_col1" class="data row0 col1" >LRDSM</td>
          <td id="T_0e439_row0_col2" class="data row0 col2" >1</td>
        </tr>
        <tr>
          <th id="T_0e439_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_0e439_row1_col0" class="data row1 col0" >CASPASE3_1_pos127</td>
          <td id="T_0e439_row1_col1" class="data row1 col1" >RDSML</td>
          <td id="T_0e439_row1_col2" class="data row1 col2" >1</td>
        </tr>
        <tr>
          <th id="T_0e439_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_0e439_row2_col0" class="data row2 col0" >CASPASE3_1_pos2</td>
          <td id="T_0e439_row2_col1" class="data row2 col1" >MSLFD</td>
          <td id="T_0e439_row2_col2" class="data row2 col2" >0</td>
        </tr>
        <tr>
          <th id="T_0e439_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_0e439_row3_col0" class="data row3 col0" >CASPASE3_1_pos3</td>
          <td id="T_0e439_row3_col1" class="data row3 col1" >SLFDL</td>
          <td id="T_0e439_row3_col2" class="data row3 col2" >0</td>
        </tr>
      </tbody>
    </table>



For Positive-Unlabeled (PU) learning, datasets are provided containing
only positive (labeled by ‘1’) and unlabeled data (‘2’), indicated by a
‘PU’ suffix in the ‘Dataset’ column name:

.. code:: ipython2

    df_seq = aa.load_dataset(name="DOM_GSEC_PU", n=10)
    aa.display_df(df=df_seq)



.. raw:: html

    <style type="text/css">
    #T_57f20 thead th {
      background-color: white;
      color: black;
    }
    #T_57f20 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_57f20 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_57f20 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_57f20  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_57f20 table {
      font-size: 12px;
    }
    </style>
    <table id="T_57f20" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_57f20_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_57f20_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_57f20_level0_col2" class="col_heading level0 col2" >label</th>
          <th id="T_57f20_level0_col3" class="col_heading level0 col3" >tmd_start</th>
          <th id="T_57f20_level0_col4" class="col_heading level0 col4" >tmd_stop</th>
          <th id="T_57f20_level0_col5" class="col_heading level0 col5" >jmd_n</th>
          <th id="T_57f20_level0_col6" class="col_heading level0 col6" >tmd</th>
          <th id="T_57f20_level0_col7" class="col_heading level0 col7" >jmd_c</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_57f20_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_57f20_row0_col0" class="data row0 col0" >P05067</td>
          <td id="T_57f20_row0_col1" class="data row0 col1" >MLPGLALLLLAA...NPTYKFFEQMQN</td>
          <td id="T_57f20_row0_col2" class="data row0 col2" >1</td>
          <td id="T_57f20_row0_col3" class="data row0 col3" >701</td>
          <td id="T_57f20_row0_col4" class="data row0 col4" >723</td>
          <td id="T_57f20_row0_col5" class="data row0 col5" >FAEDVGSNKG</td>
          <td id="T_57f20_row0_col6" class="data row0 col6" >AIIGLMVGGVVIATVIVITLVML</td>
          <td id="T_57f20_row0_col7" class="data row0 col7" >KKKQYTSIHH</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_57f20_row1_col0" class="data row1 col0" >P14925</td>
          <td id="T_57f20_row1_col1" class="data row1 col1" >MAGRARSGLLLL...YSAPLPKPAPSS</td>
          <td id="T_57f20_row1_col2" class="data row1 col2" >1</td>
          <td id="T_57f20_row1_col3" class="data row1 col3" >868</td>
          <td id="T_57f20_row1_col4" class="data row1 col4" >890</td>
          <td id="T_57f20_row1_col5" class="data row1 col5" >KLSTEPGSGV</td>
          <td id="T_57f20_row1_col6" class="data row1 col6" >SVVLITTLLVIPVLVLLAIVMFI</td>
          <td id="T_57f20_row1_col7" class="data row1 col7" >RWKKSRAFGD</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_57f20_row2_col0" class="data row2 col0" >P70180</td>
          <td id="T_57f20_row2_col1" class="data row2 col1" >MRSLLLFTFSAC...REDSIRSHFSVA</td>
          <td id="T_57f20_row2_col2" class="data row2 col2" >1</td>
          <td id="T_57f20_row2_col3" class="data row2 col3" >477</td>
          <td id="T_57f20_row2_col4" class="data row2 col4" >499</td>
          <td id="T_57f20_row2_col5" class="data row2 col5" >PCKSSGGLEE</td>
          <td id="T_57f20_row2_col6" class="data row2 col6" >SAVTGIVVGALLGAGLLMAFYFF</td>
          <td id="T_57f20_row2_col7" class="data row2 col7" >RKKYRITIER</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_57f20_row3_col0" class="data row3 col0" >Q03157</td>
          <td id="T_57f20_row3_col1" class="data row3 col1" >MGPTSPAARGQG...ENPTYRFLEERP</td>
          <td id="T_57f20_row3_col2" class="data row3 col2" >1</td>
          <td id="T_57f20_row3_col3" class="data row3 col3" >585</td>
          <td id="T_57f20_row3_col4" class="data row3 col4" >607</td>
          <td id="T_57f20_row3_col5" class="data row3 col5" >APSGTGVSRE</td>
          <td id="T_57f20_row3_col6" class="data row3 col6" >ALSGLLIMGAGGGSLIVLSLLLL</td>
          <td id="T_57f20_row3_col7" class="data row3 col7" >RKKKPYGTIS</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_57f20_row4_col0" class="data row4 col0" >Q06481</td>
          <td id="T_57f20_row4_col1" class="data row4 col1" >MAATGTAAAAAT...NPTYKYLEQMQI</td>
          <td id="T_57f20_row4_col2" class="data row4 col2" >1</td>
          <td id="T_57f20_row4_col3" class="data row4 col3" >694</td>
          <td id="T_57f20_row4_col4" class="data row4 col4" >716</td>
          <td id="T_57f20_row4_col5" class="data row4 col5" >LREDFSLSSS</td>
          <td id="T_57f20_row4_col6" class="data row4 col6" >ALIGLLVIAVAIATVIVISLVML</td>
          <td id="T_57f20_row4_col7" class="data row4 col7" >RKRQYGTISH</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_57f20_row5_col0" class="data row5 col0" >P35613</td>
          <td id="T_57f20_row5_col1" class="data row5 col1" >MAAALFVLLGFA...DKGKNVRQRNSS</td>
          <td id="T_57f20_row5_col2" class="data row5 col2" >1</td>
          <td id="T_57f20_row5_col3" class="data row5 col3" >323</td>
          <td id="T_57f20_row5_col4" class="data row5 col4" >345</td>
          <td id="T_57f20_row5_col5" class="data row5 col5" >IITLRVRSHL</td>
          <td id="T_57f20_row5_col6" class="data row5 col6" >AALWPFLGIVAEVLVLVTIIFIY</td>
          <td id="T_57f20_row5_col7" class="data row5 col7" >EKRRKPEDVL</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_57f20_row6_col0" class="data row6 col0" >P35070</td>
          <td id="T_57f20_row6_col1" class="data row6 col1" >MDRAARCSGASS...PINEDIEETNIA</td>
          <td id="T_57f20_row6_col2" class="data row6 col2" >1</td>
          <td id="T_57f20_row6_col3" class="data row6 col3" >119</td>
          <td id="T_57f20_row6_col4" class="data row6 col4" >141</td>
          <td id="T_57f20_row6_col5" class="data row6 col5" >LFYLRGDRGQ</td>
          <td id="T_57f20_row6_col6" class="data row6 col6" >ILVICLIAVMVVFIILVIGVCTC</td>
          <td id="T_57f20_row6_col7" class="data row6 col7" >CHPLRKRRKR</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_57f20_row7_col0" class="data row7 col0" >P09803</td>
          <td id="T_57f20_row7_col1" class="data row7 col1" >MGARCRSFSALL...KLADMYGGGEDD</td>
          <td id="T_57f20_row7_col2" class="data row7 col2" >1</td>
          <td id="T_57f20_row7_col3" class="data row7 col3" >711</td>
          <td id="T_57f20_row7_col4" class="data row7 col4" >733</td>
          <td id="T_57f20_row7_col5" class="data row7 col5" >GIVAAGLQVP</td>
          <td id="T_57f20_row7_col6" class="data row7 col6" >AILGILGGILALLILILLLLLFL</td>
          <td id="T_57f20_row7_col7" class="data row7 col7" >RRRTVVKEPL</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_57f20_row8_col0" class="data row8 col0" >P19022</td>
          <td id="T_57f20_row8_col1" class="data row8 col1" >MCRIAGALRTLL...KKLADMYGGGDD</td>
          <td id="T_57f20_row8_col2" class="data row8 col2" >1</td>
          <td id="T_57f20_row8_col3" class="data row8 col3" >724</td>
          <td id="T_57f20_row8_col4" class="data row8 col4" >746</td>
          <td id="T_57f20_row8_col5" class="data row8 col5" >RIVGAGLGTG</td>
          <td id="T_57f20_row8_col6" class="data row8 col6" >AIIAILLCIIILLILVLMFVVWM</td>
          <td id="T_57f20_row8_col7" class="data row8 col7" >KRRDKERQAK</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_57f20_row9_col0" class="data row9 col0" >P16070</td>
          <td id="T_57f20_row9_col1" class="data row9 col1" >MDKFWWHAAWGL...RNLQNVDMKIGV</td>
          <td id="T_57f20_row9_col2" class="data row9 col2" >1</td>
          <td id="T_57f20_row9_col3" class="data row9 col3" >650</td>
          <td id="T_57f20_row9_col4" class="data row9 col4" >672</td>
          <td id="T_57f20_row9_col5" class="data row9 col5" >GPIRTPQIPE</td>
          <td id="T_57f20_row9_col6" class="data row9 col6" >WLIILASLLALALILAVCIAVNS</td>
          <td id="T_57f20_row9_col7" class="data row9 col7" >RRRCGQKKKL</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row10" class="row_heading level0 row10" >11</th>
          <td id="T_57f20_row10_col0" class="data row10 col0" >P12821</td>
          <td id="T_57f20_row10_col1" class="data row10 col1" >MGAASGRRGPGL...PQFGSEVELRHS</td>
          <td id="T_57f20_row10_col2" class="data row10 col2" >2</td>
          <td id="T_57f20_row10_col3" class="data row10 col3" >1257</td>
          <td id="T_57f20_row10_col4" class="data row10 col4" >1276</td>
          <td id="T_57f20_row10_col5" class="data row10 col5" >GLDLDAQQAR</td>
          <td id="T_57f20_row10_col6" class="data row10 col6" >VGQWLLLFLGIALLVATLGL</td>
          <td id="T_57f20_row10_col7" class="data row10 col7" >SQRLFSIRHR</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row11" class="row_heading level0 row11" >12</th>
          <td id="T_57f20_row11_col0" class="data row11 col0" >P36896</td>
          <td id="T_57f20_row11_col1" class="data row11 col1" >MAESAGASSFFP...LSQLSVQEDVKI</td>
          <td id="T_57f20_row11_col2" class="data row11 col2" >2</td>
          <td id="T_57f20_row11_col3" class="data row11 col3" >127</td>
          <td id="T_57f20_row11_col4" class="data row11 col4" >149</td>
          <td id="T_57f20_row11_col5" class="data row11 col5" >EHPSMWGPVE</td>
          <td id="T_57f20_row11_col6" class="data row11 col6" >LVGIIAGPVFLLFLIIIIVFLVI</td>
          <td id="T_57f20_row11_col7" class="data row11 col7" >NYHQRVYHNR</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row12" class="row_heading level0 row12" >13</th>
          <td id="T_57f20_row12_col0" class="data row12 col0" >Q8NER5</td>
          <td id="T_57f20_row12_col1" class="data row12 col1" >MTRALCSALRQA...ISQLCVKEDCKA</td>
          <td id="T_57f20_row12_col2" class="data row12 col2" >2</td>
          <td id="T_57f20_row12_col3" class="data row12 col3" >114</td>
          <td id="T_57f20_row12_col4" class="data row12 col4" >136</td>
          <td id="T_57f20_row12_col5" class="data row12 col5" >PNAPKLGPME</td>
          <td id="T_57f20_row12_col6" class="data row12 col6" >LAIIITVPVCLLSIAAMLTVWAC</td>
          <td id="T_57f20_row12_col7" class="data row12 col7" >QGRQCSYRKK</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row13" class="row_heading level0 row13" >14</th>
          <td id="T_57f20_row13_col0" class="data row13 col0" >P37023</td>
          <td id="T_57f20_row13_col1" class="data row13 col1" >MTLGSPRKGLLM...ISNSPEKPKVIQ</td>
          <td id="T_57f20_row13_col2" class="data row13 col2" >2</td>
          <td id="T_57f20_row13_col3" class="data row13 col3" >119</td>
          <td id="T_57f20_row13_col4" class="data row13 col4" >141</td>
          <td id="T_57f20_row13_col5" class="data row13 col5" >PSEQPGTDGQ</td>
          <td id="T_57f20_row13_col6" class="data row13 col6" >LALILGPVLALLALVALGVLGLW</td>
          <td id="T_57f20_row13_col7" class="data row13 col7" >HVRRRQEKQR</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row14" class="row_heading level0 row14" >15</th>
          <td id="T_57f20_row14_col0" class="data row14 col0" >O43184</td>
          <td id="T_57f20_row14_col1" class="data row14 col1" >MAARPLPVSPAR...QVPRSTHTAYIK</td>
          <td id="T_57f20_row14_col2" class="data row14 col2" >2</td>
          <td id="T_57f20_row14_col3" class="data row14 col3" >707</td>
          <td id="T_57f20_row14_col4" class="data row14 col4" >729</td>
          <td id="T_57f20_row14_col5" class="data row14 col5" >DSGPIRQADN</td>
          <td id="T_57f20_row14_col6" class="data row14 col6" >QGLTIGILVTILCLLAAGFVVYL</td>
          <td id="T_57f20_row14_col7" class="data row14 col7" >KRKTLIRLLF</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row15" class="row_heading level0 row15" >16</th>
          <td id="T_57f20_row15_col0" class="data row15 col0" >Q13444</td>
          <td id="T_57f20_row15_col1" class="data row15 col1" >MRLALLWALGLL...APPPPTVSSLYL</td>
          <td id="T_57f20_row15_col2" class="data row15 col2" >2</td>
          <td id="T_57f20_row15_col3" class="data row15 col3" >695</td>
          <td id="T_57f20_row15_col4" class="data row15 col4" >717</td>
          <td id="T_57f20_row15_col5" class="data row15 col5" >TTQLKATSSL</td>
          <td id="T_57f20_row15_col6" class="data row15 col6" >TTGLLLSLLVLLVLVMLGASYWY</td>
          <td id="T_57f20_row15_col7" class="data row15 col7" >RARLHQRLCQ</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row16" class="row_heading level0 row16" >17</th>
          <td id="T_57f20_row16_col0" class="data row16 col0" >Q9Z0F8</td>
          <td id="T_57f20_row16_col1" class="data row16 col1" >MRRRLLILTTLV...RQSRVDSKETEC</td>
          <td id="T_57f20_row16_col2" class="data row16 col2" >2</td>
          <td id="T_57f20_row16_col3" class="data row16 col3" >672</td>
          <td id="T_57f20_row16_col4" class="data row16 col4" >694</td>
          <td id="T_57f20_row16_col5" class="data row16 col5" >NTFGKFLADN</td>
          <td id="T_57f20_row16_col6" class="data row16 col6" >IVGSVLVFSLIFWIPFSILVHCV</td>
          <td id="T_57f20_row16_col7" class="data row16 col7" >DKKLDKQYES</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row17" class="row_heading level0 row17" >18</th>
          <td id="T_57f20_row17_col0" class="data row17 col0" >Q9Y3Q7</td>
          <td id="T_57f20_row17_col1" class="data row17 col1" >MFLLLALLTELG...SSVVSESDDVGH</td>
          <td id="T_57f20_row17_col2" class="data row17 col2" >2</td>
          <td id="T_57f20_row17_col3" class="data row17 col3" >685</td>
          <td id="T_57f20_row17_col4" class="data row17 col4" >707</td>
          <td id="T_57f20_row17_col5" class="data row17 col5" >FYTEKGYNTH</td>
          <td id="T_57f20_row17_col6" class="data row17 col6" >WNNWFILSFCIFLPFFIVFTTVI</td>
          <td id="T_57f20_row17_col7" class="data row17 col7" >FKRNEISKSC</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row18" class="row_heading level0 row18" >19</th>
          <td id="T_57f20_row18_col0" class="data row18 col0" >Q9R157</td>
          <td id="T_57f20_row18_col1" class="data row18 col1" >MPLLFILAELAM...ERKIVPQGEHKI</td>
          <td id="T_57f20_row18_col2" class="data row18 col2" >2</td>
          <td id="T_57f20_row18_col3" class="data row18 col3" >684</td>
          <td id="T_57f20_row18_col4" class="data row18 col4" >703</td>
          <td id="T_57f20_row18_col5" class="data row18 col5" >TKRLSKNEDS</td>
          <td id="T_57f20_row18_col6" class="data row18 col6" >WVILGFFIFLPFIVTFLVGI</td>
          <td id="T_57f20_row18_col7" class="data row18 col7" >MKRNERKIVP</td>
        </tr>
        <tr>
          <th id="T_57f20_level0_row19" class="row_heading level0 row19" >20</th>
          <td id="T_57f20_row19_col0" class="data row19 col0" >O35674</td>
          <td id="T_57f20_row19_col1" class="data row19 col1" >MPGRAGVARFCL...SQRVGAIISSKI</td>
          <td id="T_57f20_row19_col2" class="data row19 col2" >2</td>
          <td id="T_57f20_row19_col3" class="data row19 col3" >704</td>
          <td id="T_57f20_row19_col4" class="data row19 col4" >726</td>
          <td id="T_57f20_row19_col5" class="data row19 col5" >VDSGPLPPKS</td>
          <td id="T_57f20_row19_col6" class="data row19 col6" >VGPVIAGVFSALFVLAVLVLLCH</td>
          <td id="T_57f20_row19_col7" class="data row19 col7" >CYRQSHKLGK</td>
        </tr>
      </tbody>
    </table>



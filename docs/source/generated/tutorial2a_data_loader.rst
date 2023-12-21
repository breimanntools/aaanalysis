Data Loading Tutorial
=====================

This is a tutorial on loading of protein benchmark datasets.

Loading of protein benchmarks
-----------------------------

Load the overview table of protein benchmark datasets using the default
settings:

.. code:: ipython2

    import aaanalysis as aa
    df_info = aa.load_dataset()
    aa.display_df(df=df_info)



.. raw:: html

    <style type="text/css">
    #T_541b0 thead th {
      background-color: white;
      color: black;
    }
    #T_541b0 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_541b0 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_541b0 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_541b0  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_541b0 table {
      font-size: 12px;
    }
    </style>
    <table id="T_541b0" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_541b0_level0_col0" class="col_heading level0 col0" >Level</th>
          <th id="T_541b0_level0_col1" class="col_heading level0 col1" >Dataset</th>
          <th id="T_541b0_level0_col2" class="col_heading level0 col2" ># Sequences</th>
          <th id="T_541b0_level0_col3" class="col_heading level0 col3" ># Amino acids</th>
          <th id="T_541b0_level0_col4" class="col_heading level0 col4" ># Positives</th>
          <th id="T_541b0_level0_col5" class="col_heading level0 col5" ># Negatives</th>
          <th id="T_541b0_level0_col6" class="col_heading level0 col6" >Predictor</th>
          <th id="T_541b0_level0_col7" class="col_heading level0 col7" >Description</th>
          <th id="T_541b0_level0_col8" class="col_heading level0 col8" >Reference</th>
          <th id="T_541b0_level0_col9" class="col_heading level0 col9" >Label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_541b0_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_541b0_row0_col0" class="data row0 col0" >Amino acid</td>
          <td id="T_541b0_row0_col1" class="data row0 col1" >AA_CASPASE3</td>
          <td id="T_541b0_row0_col2" class="data row0 col2" >233</td>
          <td id="T_541b0_row0_col3" class="data row0 col3" >185605</td>
          <td id="T_541b0_row0_col4" class="data row0 col4" >705</td>
          <td id="T_541b0_row0_col5" class="data row0 col5" >184900</td>
          <td id="T_541b0_row0_col6" class="data row0 col6" >PROSPERous</td>
          <td id="T_541b0_row0_col7" class="data row0 col7" >Prediction of caspase-3 cleavage site</td>
          <td id="T_541b0_row0_col8" class="data row0 col8" >Song et al., 2018</td>
          <td id="T_541b0_row0_col9" class="data row0 col9" >1 (adjacent to cleavage site), 0 (not adjacent to cleavage site)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_541b0_row1_col0" class="data row1 col0" >Amino acid</td>
          <td id="T_541b0_row1_col1" class="data row1 col1" >AA_FURIN</td>
          <td id="T_541b0_row1_col2" class="data row1 col2" >71</td>
          <td id="T_541b0_row1_col3" class="data row1 col3" >59003</td>
          <td id="T_541b0_row1_col4" class="data row1 col4" >163</td>
          <td id="T_541b0_row1_col5" class="data row1 col5" >58840</td>
          <td id="T_541b0_row1_col6" class="data row1 col6" >PROSPERous</td>
          <td id="T_541b0_row1_col7" class="data row1 col7" >Prediction of furin cleavage site</td>
          <td id="T_541b0_row1_col8" class="data row1 col8" >Song et al., 2018</td>
          <td id="T_541b0_row1_col9" class="data row1 col9" >1 (adjacent to cleavage site), 0 (not adjacent to cleavage site)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_541b0_row2_col0" class="data row2 col0" >Amino acid</td>
          <td id="T_541b0_row2_col1" class="data row2 col1" >AA_LDR</td>
          <td id="T_541b0_row2_col2" class="data row2 col2" >342</td>
          <td id="T_541b0_row2_col3" class="data row2 col3" >118248</td>
          <td id="T_541b0_row2_col4" class="data row2 col4" >35469</td>
          <td id="T_541b0_row2_col5" class="data row2 col5" >82779</td>
          <td id="T_541b0_row2_col6" class="data row2 col6" >IDP-Seq2Seq</td>
          <td id="T_541b0_row2_col7" class="data row2 col7" >Prediction of long intrinsically disordered regions (LDR)</td>
          <td id="T_541b0_row2_col8" class="data row2 col8" >Tang et al., 2020</td>
          <td id="T_541b0_row2_col9" class="data row2 col9" >1 (disordered), 0 (ordered)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_541b0_row3_col0" class="data row3 col0" >Amino acid</td>
          <td id="T_541b0_row3_col1" class="data row3 col1" >AA_MMP2</td>
          <td id="T_541b0_row3_col2" class="data row3 col2" >573</td>
          <td id="T_541b0_row3_col3" class="data row3 col3" >312976</td>
          <td id="T_541b0_row3_col4" class="data row3 col4" >2416</td>
          <td id="T_541b0_row3_col5" class="data row3 col5" >310560</td>
          <td id="T_541b0_row3_col6" class="data row3 col6" >PROSPERous</td>
          <td id="T_541b0_row3_col7" class="data row3 col7" >Prediction of Matrix metallopeptidase-2 (MMP2) cleavage site</td>
          <td id="T_541b0_row3_col8" class="data row3 col8" >Song et al., 2018</td>
          <td id="T_541b0_row3_col9" class="data row3 col9" >1 (adjacent to cleavage site), 0 (not adjacent to cleavage site)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_541b0_row4_col0" class="data row4 col0" >Amino acid</td>
          <td id="T_541b0_row4_col1" class="data row4 col1" >AA_RNABIND</td>
          <td id="T_541b0_row4_col2" class="data row4 col2" >221</td>
          <td id="T_541b0_row4_col3" class="data row4 col3" >55001</td>
          <td id="T_541b0_row4_col4" class="data row4 col4" >6492</td>
          <td id="T_541b0_row4_col5" class="data row4 col5" >48509</td>
          <td id="T_541b0_row4_col6" class="data row4 col6" >GMKSVM-RU</td>
          <td id="T_541b0_row4_col7" class="data row4 col7" >Prediction of RNA-binding protein residues (RBP60 dataset)</td>
          <td id="T_541b0_row4_col8" class="data row4 col8" >Yang et al., 2021</td>
          <td id="T_541b0_row4_col9" class="data row4 col9" >1 (binding), 0 (non-binding)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_541b0_row5_col0" class="data row5 col0" >Amino acid</td>
          <td id="T_541b0_row5_col1" class="data row5 col1" >AA_SA</td>
          <td id="T_541b0_row5_col2" class="data row5 col2" >233</td>
          <td id="T_541b0_row5_col3" class="data row5 col3" >185605</td>
          <td id="T_541b0_row5_col4" class="data row5 col4" >101082</td>
          <td id="T_541b0_row5_col5" class="data row5 col5" >84523</td>
          <td id="T_541b0_row5_col6" class="data row5 col6" >PROSPERous</td>
          <td id="T_541b0_row5_col7" class="data row5 col7" >Prediction of solvent accessibility (SA) of residue (AA_CASPASE3 data set)</td>
          <td id="T_541b0_row5_col8" class="data row5 col8" >Song et al., 2018</td>
          <td id="T_541b0_row5_col9" class="data row5 col9" >1 (exposed/accessible), 0 (buried/non-accessible)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_541b0_row6_col0" class="data row6 col0" >Sequence</td>
          <td id="T_541b0_row6_col1" class="data row6 col1" >SEQ_AMYLO</td>
          <td id="T_541b0_row6_col2" class="data row6 col2" >1414</td>
          <td id="T_541b0_row6_col3" class="data row6 col3" >8484</td>
          <td id="T_541b0_row6_col4" class="data row6 col4" >511</td>
          <td id="T_541b0_row6_col5" class="data row6 col5" >903</td>
          <td id="T_541b0_row6_col6" class="data row6 col6" >ReRF-Pred</td>
          <td id="T_541b0_row6_col7" class="data row6 col7" >Prediction of amyloidognenic regions</td>
          <td id="T_541b0_row6_col8" class="data row6 col8" >Teng et al. 2021</td>
          <td id="T_541b0_row6_col9" class="data row6 col9" >1 (amyloidogenic), 0 (non-amyloidogenic)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_541b0_row7_col0" class="data row7 col0" >Sequence</td>
          <td id="T_541b0_row7_col1" class="data row7 col1" >SEQ_CAPSID</td>
          <td id="T_541b0_row7_col2" class="data row7 col2" >7935</td>
          <td id="T_541b0_row7_col3" class="data row7 col3" >3364680</td>
          <td id="T_541b0_row7_col4" class="data row7 col4" >3864</td>
          <td id="T_541b0_row7_col5" class="data row7 col5" >4071</td>
          <td id="T_541b0_row7_col6" class="data row7 col6" >VIRALpro</td>
          <td id="T_541b0_row7_col7" class="data row7 col7" >Prediction of capdsid proteins</td>
          <td id="T_541b0_row7_col8" class="data row7 col8" >Galiez et al., 2016</td>
          <td id="T_541b0_row7_col9" class="data row7 col9" >1 (capsid protein), 0 (non-capsid protein)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_541b0_row8_col0" class="data row8 col0" >Sequence</td>
          <td id="T_541b0_row8_col1" class="data row8 col1" >SEQ_DISULFIDE</td>
          <td id="T_541b0_row8_col2" class="data row8 col2" >2547</td>
          <td id="T_541b0_row8_col3" class="data row8 col3" >614470</td>
          <td id="T_541b0_row8_col4" class="data row8 col4" >897</td>
          <td id="T_541b0_row8_col5" class="data row8 col5" >1650</td>
          <td id="T_541b0_row8_col6" class="data row8 col6" >Dipro</td>
          <td id="T_541b0_row8_col7" class="data row8 col7" >Prediction of disulfide bridges in sequences</td>
          <td id="T_541b0_row8_col8" class="data row8 col8" >Cheng et al., 2006</td>
          <td id="T_541b0_row8_col9" class="data row8 col9" >1 (sequence with SS bond), 0 (sequence without SS bond)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_541b0_row9_col0" class="data row9 col0" >Sequence</td>
          <td id="T_541b0_row9_col1" class="data row9 col1" >SEQ_LOCATION</td>
          <td id="T_541b0_row9_col2" class="data row9 col2" >1835</td>
          <td id="T_541b0_row9_col3" class="data row9 col3" >732398</td>
          <td id="T_541b0_row9_col4" class="data row9 col4" >1045</td>
          <td id="T_541b0_row9_col5" class="data row9 col5" >790</td>
          <td id="T_541b0_row9_col6" class="data row9 col6" >nan</td>
          <td id="T_541b0_row9_col7" class="data row9 col7" >Prediction of subcellular location of protein (cytoplasm vs plasma membrane)</td>
          <td id="T_541b0_row9_col8" class="data row9 col8" >Shen et al., 2019</td>
          <td id="T_541b0_row9_col9" class="data row9 col9" >1 (protein in cytoplasm), 0 (protein in plasma membrane) </td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row10" class="row_heading level0 row10" >11</th>
          <td id="T_541b0_row10_col0" class="data row10 col0" >Sequence</td>
          <td id="T_541b0_row10_col1" class="data row10 col1" >SEQ_SOLUBLE</td>
          <td id="T_541b0_row10_col2" class="data row10 col2" >17408</td>
          <td id="T_541b0_row10_col3" class="data row10 col3" >4432269</td>
          <td id="T_541b0_row10_col4" class="data row10 col4" >8704</td>
          <td id="T_541b0_row10_col5" class="data row10 col5" >8704</td>
          <td id="T_541b0_row10_col6" class="data row10 col6" >SOLpro</td>
          <td id="T_541b0_row10_col7" class="data row10 col7" >Prediction of soluble and insoluble proteins</td>
          <td id="T_541b0_row10_col8" class="data row10 col8" >Magnan et al., 2009</td>
          <td id="T_541b0_row10_col9" class="data row10 col9" >1 (soluble), 0 (insoluble)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row11" class="row_heading level0 row11" >12</th>
          <td id="T_541b0_row11_col0" class="data row11 col0" >Sequence</td>
          <td id="T_541b0_row11_col1" class="data row11 col1" >SEQ_TAIL</td>
          <td id="T_541b0_row11_col2" class="data row11 col2" >6668</td>
          <td id="T_541b0_row11_col3" class="data row11 col3" >2671690</td>
          <td id="T_541b0_row11_col4" class="data row11 col4" >2574</td>
          <td id="T_541b0_row11_col5" class="data row11 col5" >4094</td>
          <td id="T_541b0_row11_col6" class="data row11 col6" >VIRALpro</td>
          <td id="T_541b0_row11_col7" class="data row11 col7" >Prediction of tail proteins</td>
          <td id="T_541b0_row11_col8" class="data row11 col8" >Galiez et al., 2016</td>
          <td id="T_541b0_row11_col9" class="data row11 col9" >1 (tail protein), 0 (non-tail protein)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row12" class="row_heading level0 row12" >13</th>
          <td id="T_541b0_row12_col0" class="data row12 col0" >Domain</td>
          <td id="T_541b0_row12_col1" class="data row12 col1" >DOM_GSEC</td>
          <td id="T_541b0_row12_col2" class="data row12 col2" >126</td>
          <td id="T_541b0_row12_col3" class="data row12 col3" >92964</td>
          <td id="T_541b0_row12_col4" class="data row12 col4" >63</td>
          <td id="T_541b0_row12_col5" class="data row12 col5" >63</td>
          <td id="T_541b0_row12_col6" class="data row12 col6" >nan</td>
          <td id="T_541b0_row12_col7" class="data row12 col7" >Prediction of gamma-secretase substrates</td>
          <td id="T_541b0_row12_col8" class="data row12 col8" >Breimann et al, 2023c</td>
          <td id="T_541b0_row12_col9" class="data row12 col9" >1 (substrate), 0 (non-substrate)</td>
        </tr>
        <tr>
          <th id="T_541b0_level0_row13" class="row_heading level0 row13" >14</th>
          <td id="T_541b0_row13_col0" class="data row13 col0" >Domain</td>
          <td id="T_541b0_row13_col1" class="data row13 col1" >DOM_GSEC_PU</td>
          <td id="T_541b0_row13_col2" class="data row13 col2" >694</td>
          <td id="T_541b0_row13_col3" class="data row13 col3" >494524</td>
          <td id="T_541b0_row13_col4" class="data row13 col4" >63</td>
          <td id="T_541b0_row13_col5" class="data row13 col5" >0</td>
          <td id="T_541b0_row13_col6" class="data row13 col6" >nan</td>
          <td id="T_541b0_row13_col7" class="data row13 col7" >Prediction of gamma-secretase substrates (PU dataset)</td>
          <td id="T_541b0_row13_col8" class="data row13 col8" >Breimann et al, 2023c</td>
          <td id="T_541b0_row13_col9" class="data row13 col9" >1 (substrate), 2 (unknown substrate status)</td>
        </tr>
      </tbody>
    </table>



The benchmark datasets are categorized into amino acid (‘AA’), domain
(‘DOM’), and sequence (‘SEQ’) level datasets, indicated by their
``name`` prefix, as exemplified here.

.. code:: ipython2

    df_seq1 = aa.load_dataset(name="AA_CASPASE3")
    df_seq2 = aa.load_dataset(name="SEQ_CAPSID")
    df_seq3 = aa.load_dataset(name="DOM_GSEC")
    aa.display_df(df=df_seq3.head(5), char_limit=25)
    # Compare columns of three types



.. raw:: html

    <style type="text/css">
    #T_7169f thead th {
      background-color: white;
      color: black;
    }
    #T_7169f tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_7169f tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_7169f th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_7169f  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_7169f table {
      font-size: 12px;
    }
    </style>
    <table id="T_7169f" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_7169f_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_7169f_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_7169f_level0_col2" class="col_heading level0 col2" >label</th>
          <th id="T_7169f_level0_col3" class="col_heading level0 col3" >tmd_start</th>
          <th id="T_7169f_level0_col4" class="col_heading level0 col4" >tmd_stop</th>
          <th id="T_7169f_level0_col5" class="col_heading level0 col5" >jmd_n</th>
          <th id="T_7169f_level0_col6" class="col_heading level0 col6" >tmd</th>
          <th id="T_7169f_level0_col7" class="col_heading level0 col7" >jmd_c</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_7169f_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_7169f_row0_col0" class="data row0 col0" >P05067</td>
          <td id="T_7169f_row0_col1" class="data row0 col1" >MLPGLALLLLAA...NPTYKFFEQMQN</td>
          <td id="T_7169f_row0_col2" class="data row0 col2" >1</td>
          <td id="T_7169f_row0_col3" class="data row0 col3" >701</td>
          <td id="T_7169f_row0_col4" class="data row0 col4" >723</td>
          <td id="T_7169f_row0_col5" class="data row0 col5" >FAEDVGSNKG</td>
          <td id="T_7169f_row0_col6" class="data row0 col6" >AIIGLMVGGVVIATVIVITLVML</td>
          <td id="T_7169f_row0_col7" class="data row0 col7" >KKKQYTSIHH</td>
        </tr>
        <tr>
          <th id="T_7169f_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_7169f_row1_col0" class="data row1 col0" >P14925</td>
          <td id="T_7169f_row1_col1" class="data row1 col1" >MAGRARSGLLLL...YSAPLPKPAPSS</td>
          <td id="T_7169f_row1_col2" class="data row1 col2" >1</td>
          <td id="T_7169f_row1_col3" class="data row1 col3" >868</td>
          <td id="T_7169f_row1_col4" class="data row1 col4" >890</td>
          <td id="T_7169f_row1_col5" class="data row1 col5" >KLSTEPGSGV</td>
          <td id="T_7169f_row1_col6" class="data row1 col6" >SVVLITTLLVIPVLVLLAIVMFI</td>
          <td id="T_7169f_row1_col7" class="data row1 col7" >RWKKSRAFGD</td>
        </tr>
        <tr>
          <th id="T_7169f_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_7169f_row2_col0" class="data row2 col0" >P70180</td>
          <td id="T_7169f_row2_col1" class="data row2 col1" >MRSLLLFTFSAC...REDSIRSHFSVA</td>
          <td id="T_7169f_row2_col2" class="data row2 col2" >1</td>
          <td id="T_7169f_row2_col3" class="data row2 col3" >477</td>
          <td id="T_7169f_row2_col4" class="data row2 col4" >499</td>
          <td id="T_7169f_row2_col5" class="data row2 col5" >PCKSSGGLEE</td>
          <td id="T_7169f_row2_col6" class="data row2 col6" >SAVTGIVVGALLGAGLLMAFYFF</td>
          <td id="T_7169f_row2_col7" class="data row2 col7" >RKKYRITIER</td>
        </tr>
        <tr>
          <th id="T_7169f_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_7169f_row3_col0" class="data row3 col0" >Q03157</td>
          <td id="T_7169f_row3_col1" class="data row3 col1" >MGPTSPAARGQG...ENPTYRFLEERP</td>
          <td id="T_7169f_row3_col2" class="data row3 col2" >1</td>
          <td id="T_7169f_row3_col3" class="data row3 col3" >585</td>
          <td id="T_7169f_row3_col4" class="data row3 col4" >607</td>
          <td id="T_7169f_row3_col5" class="data row3 col5" >APSGTGVSRE</td>
          <td id="T_7169f_row3_col6" class="data row3 col6" >ALSGLLIMGAGGGSLIVLSLLLL</td>
          <td id="T_7169f_row3_col7" class="data row3 col7" >RKKKPYGTIS</td>
        </tr>
        <tr>
          <th id="T_7169f_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_7169f_row4_col0" class="data row4 col0" >Q06481</td>
          <td id="T_7169f_row4_col1" class="data row4 col1" >MAATGTAAAAAT...NPTYKYLEQMQI</td>
          <td id="T_7169f_row4_col2" class="data row4 col2" >1</td>
          <td id="T_7169f_row4_col3" class="data row4 col3" >694</td>
          <td id="T_7169f_row4_col4" class="data row4 col4" >716</td>
          <td id="T_7169f_row4_col5" class="data row4 col5" >LREDFSLSSS</td>
          <td id="T_7169f_row4_col6" class="data row4 col6" >ALIGLLVIAVAIATVIVISLVML</td>
          <td id="T_7169f_row4_col7" class="data row4 col7" >RKRQYGTISH</td>
        </tr>
      </tbody>
    </table>



Each dataset can be utilized for a binary classification, with labels
being positive (1) or negative (0). A balanced number of samples can be
chosen by the ``n`` parameter, defining the sample number per class.

.. code:: ipython2

    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=100)
    # Returns 200 samples, 100 positives and 100 negatives
    df_seq["label"].value_counts()




.. parsed-literal::

    label
    0    100
    1    100
    Name: count, dtype: int64



Or randomly selected using ``random=True``:

.. code:: ipython2

    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=100, random=True)

The protein sequences can have varying length:

.. code:: ipython2

    # Plot distribution
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Utility AAanalysis function for publication ready plots
    aa.plot_settings(font_scale=1.2) 
    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=100)
    list_seq_lens = df_seq["sequence"].apply(len)
    sns.histplot(list_seq_lens, binwidth=50)
    sns.despine()
    plt.xlim(0, 1500)
    plt.show()



.. image:: tutorial2a_data_loader_1_output_9_0.png


Which can be easily filtered using ``min_len`` and ``max_len``
parameters:

.. code:: ipython2

    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=100, min_len=200, max_len=800)
    list_seq_lens = df_seq["sequence"].apply(len)
    aa.plot_settings(font_scale=1.2)  # Utility AAanalysis function for publication ready plots
    sns.histplot(list_seq_lens, binwidth=50)
    sns.despine()
    plt.xlim(0, 1500)
    plt.show()



.. image:: tutorial2a_data_loader_2_output_11_0.png


Loading of protein benchmarks: Amino acid window size
-----------------------------------------------------

For amino acid level datasets, labels are provided for each residue
position, which can be seen by setting ``aa_window_size=None``:

.. code:: ipython2

    df_seq = aa.load_dataset(name="AA_CASPASE3", aa_window_size=None)
    aa.display_df(df=df_seq.head(10), char_limit=25)



.. raw:: html

    <style type="text/css">
    #T_a9434 thead th {
      background-color: white;
      color: black;
    }
    #T_a9434 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_a9434 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_a9434 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_a9434  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_a9434 table {
      font-size: 12px;
    }
    </style>
    <table id="T_a9434" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_a9434_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_a9434_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_a9434_level0_col2" class="col_heading level0 col2" >label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_a9434_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_a9434_row0_col0" class="data row0 col0" >CASPASE3_1</td>
          <td id="T_a9434_row0_col1" class="data row0 col1" >MSLFDLFRGFFG...LDLFLGRWFRSR</td>
          <td id="T_a9434_row0_col2" class="data row0 col2" >0,0,0,0,0,0,...,0,0,0,0,0,0</td>
        </tr>
        <tr>
          <th id="T_a9434_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_a9434_row1_col0" class="data row1 col0" >CASPASE3_2</td>
          <td id="T_a9434_row1_col1" class="data row1 col1" >MEVTGDAGVPES...LQNPKRARQDPT</td>
          <td id="T_a9434_row1_col2" class="data row1 col2" >0,0,0,0,0,0,...,0,0,0,0,0,0</td>
        </tr>
        <tr>
          <th id="T_a9434_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_a9434_row2_col0" class="data row2 col0" >CASPASE3_3</td>
          <td id="T_a9434_row2_col1" class="data row2 col1" >MRARSGARGALL...EMLVAMTTDGDC</td>
          <td id="T_a9434_row2_col2" class="data row2 col2" >0,0,0,0,0,0,...,0,0,0,0,0,0</td>
        </tr>
        <tr>
          <th id="T_a9434_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_a9434_row3_col0" class="data row3 col0" >CASPASE3_4</td>
          <td id="T_a9434_row3_col1" class="data row3 col1" >MDAKARNCLLQH...NLGILYILQTLE</td>
          <td id="T_a9434_row3_col2" class="data row3 col2" >0,0,0,0,0,0,...,0,0,0,0,0,0</td>
        </tr>
        <tr>
          <th id="T_a9434_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_a9434_row4_col0" class="data row4 col0" >CASPASE3_5</td>
          <td id="T_a9434_row4_col1" class="data row4 col1" >MTSFSTSAQCST...KEIQLVIKVFIA</td>
          <td id="T_a9434_row4_col2" class="data row4 col2" >0,0,0,0,0,0,...,0,0,0,0,0,0</td>
        </tr>
        <tr>
          <th id="T_a9434_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_a9434_row5_col0" class="data row5 col0" >CASPASE3_6</td>
          <td id="T_a9434_row5_col1" class="data row5 col1" >MGLGASSEQPAG...PDPEPGLCEGPW</td>
          <td id="T_a9434_row5_col2" class="data row5 col2" >0,0,0,0,0,0,...,0,0,0,0,0,0</td>
        </tr>
        <tr>
          <th id="T_a9434_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_a9434_row6_col0" class="data row6 col0" >CASPASE3_7</td>
          <td id="T_a9434_row6_col1" class="data row6 col1" >MANQVNGNAVQL...EFYQDTYGQQWK</td>
          <td id="T_a9434_row6_col2" class="data row6 col2" >0,0,0,0,0,0,...,0,0,0,0,0,0</td>
        </tr>
        <tr>
          <th id="T_a9434_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_a9434_row7_col0" class="data row7 col0" >CASPASE3_8</td>
          <td id="T_a9434_row7_col1" class="data row7 col1" >MAKQPSDVSSEC...LRYIVRLVWRMH</td>
          <td id="T_a9434_row7_col2" class="data row7 col2" >0,0,0,0,0,0,...,0,0,0,0,0,0</td>
        </tr>
        <tr>
          <th id="T_a9434_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_a9434_row8_col0" class="data row8 col0" >CASPASE3_9</td>
          <td id="T_a9434_row8_col1" class="data row8 col1" >MCTALSPKVRSG...VSASYKAKKEIK</td>
          <td id="T_a9434_row8_col2" class="data row8 col2" >0,0,0,0,0,0,...,0,0,0,0,0,0</td>
        </tr>
        <tr>
          <th id="T_a9434_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_a9434_row9_col0" class="data row9 col0" >CASPASE3_10</td>
          <td id="T_a9434_row9_col1" class="data row9 col1" >MFYAHFVLSKRG...IIATPGPRFHII</td>
          <td id="T_a9434_row9_col2" class="data row9 col2" >0,0,0,0,0,0,...,0,0,0,0,0,0</td>
        </tr>
      </tbody>
    </table>



For convenience, we provide an “amino acid window” of length n. This
window represents a specific amino acid, which is flanked by (n-1)/2
residues on both its N-terminal and C-terminal sides. It’s essential for
n to be odd, ensuring equal residues on both sides. While the default
window size is 9, sizes between 5 and 15 are also popular.

.. code:: ipython2

    df_seq = aa.load_dataset(name="AA_CASPASE3", n=2)
    aa.display_df(df=df_seq, char_limit=25)



.. raw:: html

    <style type="text/css">
    #T_b0d1e thead th {
      background-color: white;
      color: black;
    }
    #T_b0d1e tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_b0d1e tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_b0d1e th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_b0d1e  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_b0d1e table {
      font-size: 12px;
    }
    </style>
    <table id="T_b0d1e" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_b0d1e_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_b0d1e_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_b0d1e_level0_col2" class="col_heading level0 col2" >label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_b0d1e_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_b0d1e_row0_col0" class="data row0 col0" >CASPASE3_1_pos126</td>
          <td id="T_b0d1e_row0_col1" class="data row0 col1" >QTLRDSMLK</td>
          <td id="T_b0d1e_row0_col2" class="data row0 col2" >1</td>
        </tr>
        <tr>
          <th id="T_b0d1e_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_b0d1e_row1_col0" class="data row1 col0" >CASPASE3_1_pos127</td>
          <td id="T_b0d1e_row1_col1" class="data row1 col1" >TLRDSMLKY</td>
          <td id="T_b0d1e_row1_col2" class="data row1 col2" >1</td>
        </tr>
        <tr>
          <th id="T_b0d1e_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_b0d1e_row2_col0" class="data row2 col0" >CASPASE3_1_pos4</td>
          <td id="T_b0d1e_row2_col1" class="data row2 col1" >MSLFDLFRG</td>
          <td id="T_b0d1e_row2_col2" class="data row2 col2" >0</td>
        </tr>
        <tr>
          <th id="T_b0d1e_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_b0d1e_row3_col0" class="data row3 col0" >CASPASE3_1_pos5</td>
          <td id="T_b0d1e_row3_col1" class="data row3 col1" >SLFDLFRGF</td>
          <td id="T_b0d1e_row3_col2" class="data row3 col2" >0</td>
        </tr>
      </tbody>
    </table>



Sequences can be pre-filtered using ``min_len`` and ``max_len`` and
``n`` residues can be randomly selected by ``random`` with different
``aa_window_size``\ s.

.. code:: ipython2

    df_seq = aa.load_dataset(name="AA_CASPASE3", min_len=20, n=2, random=True, aa_window_size=21)
    aa.display_df(df=df_seq, char_limit=25)



.. raw:: html

    <style type="text/css">
    #T_4212f thead th {
      background-color: white;
      color: black;
    }
    #T_4212f tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_4212f tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_4212f th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_4212f  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_4212f table {
      font-size: 12px;
    }
    </style>
    <table id="T_4212f" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_4212f_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_4212f_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_4212f_level0_col2" class="col_heading level0 col2" >label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_4212f_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_4212f_row0_col0" class="data row0 col0" >CASPASE3_199_pos224</td>
          <td id="T_4212f_row0_col1" class="data row0 col1" >SPEAKLTEVDNYHFYSSIPSM</td>
          <td id="T_4212f_row0_col2" class="data row0 col2" >1</td>
        </tr>
        <tr>
          <th id="T_4212f_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_4212f_row1_col0" class="data row1 col0" >CASPASE3_102_pos149</td>
          <td id="T_4212f_row1_col1" class="data row1 col1" >RKRRQTSMTDFYHSKRRLIFS</td>
          <td id="T_4212f_row1_col2" class="data row1 col2" >1</td>
        </tr>
        <tr>
          <th id="T_4212f_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_4212f_row2_col0" class="data row2 col0" >CASPASE3_64_pos523</td>
          <td id="T_4212f_row2_col1" class="data row2 col1" >ACPPVAAPGSTPFSSQPNLAD</td>
          <td id="T_4212f_row2_col2" class="data row2 col2" >0</td>
        </tr>
        <tr>
          <th id="T_4212f_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_4212f_row3_col0" class="data row3 col0" >CASPASE3_78_pos576</td>
          <td id="T_4212f_row3_col1" class="data row3 col1" >DQDSRSAPEPKKPEENPASKF</td>
          <td id="T_4212f_row3_col2" class="data row3 col2" >0</td>
        </tr>
      </tbody>
    </table>



Loading of protein benchmarks: Positive-Unlabeled (PU) datasets
---------------------------------------------------------------

In typical binary classification, data is labeled as positive (1) or
negative (0). But with many protein sequence datasets, we face
challenges: they might be small, unbalanced, or lack a clear negative
class. For datasets with only positive and unlabeled samples (2), we use
PU learning. This approach identifies reliable negatives from the
unlabeled data to make binary classification possible. We offer
benchmark datasets for this scenario, denoted by the ``_PU`` suffix. For
example, the ``DOM_GSEC_PU`` dataset corresponds to the
``DOM_GSEC set``.

.. code:: ipython2

    df_seq = aa.load_dataset(name="DOM_GSEC", n=2)
    aa.display_df(df=df_seq, char_limit=25)



.. raw:: html

    <style type="text/css">
    #T_57c5d thead th {
      background-color: white;
      color: black;
    }
    #T_57c5d tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_57c5d tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_57c5d th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_57c5d  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_57c5d table {
      font-size: 12px;
    }
    </style>
    <table id="T_57c5d" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_57c5d_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_57c5d_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_57c5d_level0_col2" class="col_heading level0 col2" >label</th>
          <th id="T_57c5d_level0_col3" class="col_heading level0 col3" >tmd_start</th>
          <th id="T_57c5d_level0_col4" class="col_heading level0 col4" >tmd_stop</th>
          <th id="T_57c5d_level0_col5" class="col_heading level0 col5" >jmd_n</th>
          <th id="T_57c5d_level0_col6" class="col_heading level0 col6" >tmd</th>
          <th id="T_57c5d_level0_col7" class="col_heading level0 col7" >jmd_c</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_57c5d_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_57c5d_row0_col0" class="data row0 col0" >Q14802</td>
          <td id="T_57c5d_row0_col1" class="data row0 col1" >MQKVTLGLLVFL...TPPLITPGSAQS</td>
          <td id="T_57c5d_row0_col2" class="data row0 col2" >0</td>
          <td id="T_57c5d_row0_col3" class="data row0 col3" >37</td>
          <td id="T_57c5d_row0_col4" class="data row0 col4" >59</td>
          <td id="T_57c5d_row0_col5" class="data row0 col5" >NSPFYYDWHS</td>
          <td id="T_57c5d_row0_col6" class="data row0 col6" >LQVGGLICAGVLCAMGIIIVMSA</td>
          <td id="T_57c5d_row0_col7" class="data row0 col7" >KCKCKFGQKS</td>
        </tr>
        <tr>
          <th id="T_57c5d_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_57c5d_row1_col0" class="data row1 col0" >Q86UE4</td>
          <td id="T_57c5d_row1_col1" class="data row1 col1" >MAARSWQDELAQ...QIKKKKKARRET</td>
          <td id="T_57c5d_row1_col2" class="data row1 col2" >0</td>
          <td id="T_57c5d_row1_col3" class="data row1 col3" >50</td>
          <td id="T_57c5d_row1_col4" class="data row1 col4" >72</td>
          <td id="T_57c5d_row1_col5" class="data row1 col5" >LGLEPKRYPG</td>
          <td id="T_57c5d_row1_col6" class="data row1 col6" >WVILVGTGALGLLLLFLLGYGWA</td>
          <td id="T_57c5d_row1_col7" class="data row1 col7" >AACAGARKKR</td>
        </tr>
        <tr>
          <th id="T_57c5d_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_57c5d_row2_col0" class="data row2 col0" >P05067</td>
          <td id="T_57c5d_row2_col1" class="data row2 col1" >MLPGLALLLLAA...NPTYKFFEQMQN</td>
          <td id="T_57c5d_row2_col2" class="data row2 col2" >1</td>
          <td id="T_57c5d_row2_col3" class="data row2 col3" >701</td>
          <td id="T_57c5d_row2_col4" class="data row2 col4" >723</td>
          <td id="T_57c5d_row2_col5" class="data row2 col5" >FAEDVGSNKG</td>
          <td id="T_57c5d_row2_col6" class="data row2 col6" >AIIGLMVGGVVIATVIVITLVML</td>
          <td id="T_57c5d_row2_col7" class="data row2 col7" >KKKQYTSIHH</td>
        </tr>
        <tr>
          <th id="T_57c5d_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_57c5d_row3_col0" class="data row3 col0" >P14925</td>
          <td id="T_57c5d_row3_col1" class="data row3 col1" >MAGRARSGLLLL...YSAPLPKPAPSS</td>
          <td id="T_57c5d_row3_col2" class="data row3 col2" >1</td>
          <td id="T_57c5d_row3_col3" class="data row3 col3" >868</td>
          <td id="T_57c5d_row3_col4" class="data row3 col4" >890</td>
          <td id="T_57c5d_row3_col5" class="data row3 col5" >KLSTEPGSGV</td>
          <td id="T_57c5d_row3_col6" class="data row3 col6" >SVVLITTLLVIPVLVLLAIVMFI</td>
          <td id="T_57c5d_row3_col7" class="data row3 col7" >RWKKSRAFGD</td>
        </tr>
      </tbody>
    </table>



.. code:: ipython2

    df_seq_pu = aa.load_dataset(name="DOM_GSEC_PU", n=2)
    aa.display_df(df=df_seq_pu, char_limit=25)



.. raw:: html

    <style type="text/css">
    #T_46087 thead th {
      background-color: white;
      color: black;
    }
    #T_46087 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_46087 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_46087 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_46087  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_46087 table {
      font-size: 12px;
    }
    </style>
    <table id="T_46087" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_46087_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_46087_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_46087_level0_col2" class="col_heading level0 col2" >label</th>
          <th id="T_46087_level0_col3" class="col_heading level0 col3" >tmd_start</th>
          <th id="T_46087_level0_col4" class="col_heading level0 col4" >tmd_stop</th>
          <th id="T_46087_level0_col5" class="col_heading level0 col5" >jmd_n</th>
          <th id="T_46087_level0_col6" class="col_heading level0 col6" >tmd</th>
          <th id="T_46087_level0_col7" class="col_heading level0 col7" >jmd_c</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_46087_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_46087_row0_col0" class="data row0 col0" >P05067</td>
          <td id="T_46087_row0_col1" class="data row0 col1" >MLPGLALLLLAA...NPTYKFFEQMQN</td>
          <td id="T_46087_row0_col2" class="data row0 col2" >1</td>
          <td id="T_46087_row0_col3" class="data row0 col3" >701</td>
          <td id="T_46087_row0_col4" class="data row0 col4" >723</td>
          <td id="T_46087_row0_col5" class="data row0 col5" >FAEDVGSNKG</td>
          <td id="T_46087_row0_col6" class="data row0 col6" >AIIGLMVGGVVIATVIVITLVML</td>
          <td id="T_46087_row0_col7" class="data row0 col7" >KKKQYTSIHH</td>
        </tr>
        <tr>
          <th id="T_46087_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_46087_row1_col0" class="data row1 col0" >P14925</td>
          <td id="T_46087_row1_col1" class="data row1 col1" >MAGRARSGLLLL...YSAPLPKPAPSS</td>
          <td id="T_46087_row1_col2" class="data row1 col2" >1</td>
          <td id="T_46087_row1_col3" class="data row1 col3" >868</td>
          <td id="T_46087_row1_col4" class="data row1 col4" >890</td>
          <td id="T_46087_row1_col5" class="data row1 col5" >KLSTEPGSGV</td>
          <td id="T_46087_row1_col6" class="data row1 col6" >SVVLITTLLVIPVLVLLAIVMFI</td>
          <td id="T_46087_row1_col7" class="data row1 col7" >RWKKSRAFGD</td>
        </tr>
        <tr>
          <th id="T_46087_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_46087_row2_col0" class="data row2 col0" >P12821</td>
          <td id="T_46087_row2_col1" class="data row2 col1" >MGAASGRRGPGL...PQFGSEVELRHS</td>
          <td id="T_46087_row2_col2" class="data row2 col2" >2</td>
          <td id="T_46087_row2_col3" class="data row2 col3" >1257</td>
          <td id="T_46087_row2_col4" class="data row2 col4" >1276</td>
          <td id="T_46087_row2_col5" class="data row2 col5" >GLDLDAQQAR</td>
          <td id="T_46087_row2_col6" class="data row2 col6" >VGQWLLLFLGIALLVATLGL</td>
          <td id="T_46087_row2_col7" class="data row2 col7" >SQRLFSIRHR</td>
        </tr>
        <tr>
          <th id="T_46087_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_46087_row3_col0" class="data row3 col0" >P36896</td>
          <td id="T_46087_row3_col1" class="data row3 col1" >MAESAGASSFFP...LSQLSVQEDVKI</td>
          <td id="T_46087_row3_col2" class="data row3 col2" >2</td>
          <td id="T_46087_row3_col3" class="data row3 col3" >127</td>
          <td id="T_46087_row3_col4" class="data row3 col4" >149</td>
          <td id="T_46087_row3_col5" class="data row3 col5" >EHPSMWGPVE</td>
          <td id="T_46087_row3_col6" class="data row3 col6" >LVGIIAGPVFLLFLIIIIVFLVI</td>
          <td id="T_46087_row3_col7" class="data row3 col7" >NYHQRVYHNR</td>
        </tr>
      </tbody>
    </table>




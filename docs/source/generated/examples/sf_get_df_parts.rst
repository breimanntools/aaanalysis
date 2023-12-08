Get sequence parts from df_seq with 'tmd_e', and 'tmd_jmd' as parts and jmd length of 10:

.. code:: ipython2

    import aaanalysis as aa
    sf = aa.SequenceFeature()
    df_seq = aa.load_dataset(name='DOM_GSEC')
    df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd_e", "tmd_jmd"], jmd_n_len=10, jmd_c_len=10)
    aa.display_df(df=df_parts.head(5))



.. raw:: html

    <style type="text/css">
    #T_4cdd2 thead th {
      background-color: white;
      color: black;
    }
    #T_4cdd2 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_4cdd2 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_4cdd2 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_4cdd2  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_4cdd2 table {
      font-size: 12px;
    }
    </style>
    <table id="T_4cdd2" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_4cdd2_level0_col0" class="col_heading level0 col0" >tmd_e</th>
          <th id="T_4cdd2_level0_col1" class="col_heading level0 col1" >tmd_jmd</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_4cdd2_level0_row0" class="row_heading level0 row0" >P05067</th>
          <td id="T_4cdd2_row0_col0" class="data row0 col0" >AIIGLMVGGVVIATVIVITLVML</td>
          <td id="T_4cdd2_row0_col1" class="data row0 col1" >FAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYTSIHH</td>
        </tr>
        <tr>
          <th id="T_4cdd2_level0_row1" class="row_heading level0 row1" >P14925</th>
          <td id="T_4cdd2_row1_col0" class="data row1 col0" >SVVLITTLLVIPVLVLLAIVMFI</td>
          <td id="T_4cdd2_row1_col1" class="data row1 col1" >KLSTEPGSGVSVVLITTLLVIPVLVLLAIVMFIRWKKSRAFGD</td>
        </tr>
        <tr>
          <th id="T_4cdd2_level0_row2" class="row_heading level0 row2" >P70180</th>
          <td id="T_4cdd2_row2_col0" class="data row2 col0" >SAVTGIVVGALLGAGLLMAFYFF</td>
          <td id="T_4cdd2_row2_col1" class="data row2 col1" >PCKSSGGLEESAVTGIVVGALLGAGLLMAFYFFRKKYRITIER</td>
        </tr>
        <tr>
          <th id="T_4cdd2_level0_row3" class="row_heading level0 row3" >Q03157</th>
          <td id="T_4cdd2_row3_col0" class="data row3 col0" >ALSGLLIMGAGGGSLIVLSLLLL</td>
          <td id="T_4cdd2_row3_col1" class="data row3 col1" >APSGTGVSREALSGLLIMGAGGGSLIVLSLLLLRKKKPYGTIS</td>
        </tr>
        <tr>
          <th id="T_4cdd2_level0_row4" class="row_heading level0 row4" >Q06481</th>
          <td id="T_4cdd2_row4_col0" class="data row4 col0" >ALIGLLVIAVAIATVIVISLVML</td>
          <td id="T_4cdd2_row4_col1" class="data row4 col1" >LREDFSLSSSALIGLLVIAVAIATVIVISLVMLRKRQYGTISH</td>
        </tr>
      </tbody>
    </table>





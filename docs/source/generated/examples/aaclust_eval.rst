Different clustering results can be evaluated and compared using the
``AAclust().eval()`` method. We perform five clusterings with
``n_clusters`` for 5, 10, 25, 50, and 100 utilizing a Python
comprehension list:

.. code:: ipython2

    import aaanalysis as aa
    aa.options["verbose"] = False
    X = aa.load_scales().T
    aac = aa.AAclust()
    list_labels = [aac.fit(X, n_clusters=n).labels_ for n in [5, 10, 25, 50, 100]]
    df_eval = aac.eval(X, list_labels=list_labels)
    aa.display_df(df_eval)



.. raw:: html

    <style type="text/css">
    #T_6a472 thead th {
      background-color: white;
      color: black;
    }
    #T_6a472 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_6a472 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_6a472 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_6a472  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_6a472 table {
      font-size: 12px;
    }
    </style>
    <table id="T_6a472" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_6a472_level0_col0" class="col_heading level0 col0" >name</th>
          <th id="T_6a472_level0_col1" class="col_heading level0 col1" >n_clusters</th>
          <th id="T_6a472_level0_col2" class="col_heading level0 col2" >BIC</th>
          <th id="T_6a472_level0_col3" class="col_heading level0 col3" >CH</th>
          <th id="T_6a472_level0_col4" class="col_heading level0 col4" >SC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_6a472_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_6a472_row0_col0" class="data row0 col0" >Set 1</td>
          <td id="T_6a472_row0_col1" class="data row0 col1" >5</td>
          <td id="T_6a472_row0_col2" class="data row0 col2" >-235.010922</td>
          <td id="T_6a472_row0_col3" class="data row0 col3" >119.193092</td>
          <td id="T_6a472_row0_col4" class="data row0 col4" >0.187077</td>
        </tr>
        <tr>
          <th id="T_6a472_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_6a472_row1_col0" class="data row1 col0" >Set 2</td>
          <td id="T_6a472_row1_col1" class="data row1 col1" >10</td>
          <td id="T_6a472_row1_col2" class="data row1 col2" >503.637472</td>
          <td id="T_6a472_row1_col3" class="data row1 col3" >85.200434</td>
          <td id="T_6a472_row1_col4" class="data row1 col4" >0.186997</td>
        </tr>
        <tr>
          <th id="T_6a472_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_6a472_row2_col0" class="data row2 col0" >Set 3</td>
          <td id="T_6a472_row2_col1" class="data row2 col1" >25</td>
          <td id="T_6a472_row2_col2" class="data row2 col2" >893.834929</td>
          <td id="T_6a472_row2_col3" class="data row2 col3" >52.008981</td>
          <td id="T_6a472_row2_col4" class="data row2 col4" >0.171144</td>
        </tr>
        <tr>
          <th id="T_6a472_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_6a472_row3_col0" class="data row3 col0" >Set 4</td>
          <td id="T_6a472_row3_col1" class="data row3 col1" >50</td>
          <td id="T_6a472_row3_col2" class="data row3 col2" >279.310802</td>
          <td id="T_6a472_row3_col3" class="data row3 col3" >34.231471</td>
          <td id="T_6a472_row3_col4" class="data row3 col4" >0.139191</td>
        </tr>
        <tr>
          <th id="T_6a472_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_6a472_row4_col0" class="data row4 col0" >Set 5</td>
          <td id="T_6a472_row4_col1" class="data row4 col1" >100</td>
          <td id="T_6a472_row4_col2" class="data row4 col2" >-1485.636265</td>
          <td id="T_6a472_row4_col3" class="data row4 col3" >23.665256</td>
          <td id="T_6a472_row4_col4" class="data row4 col4" >0.150045</td>
        </tr>
      </tbody>
    </table>



The name of the scale sets can be provided using the ``names_datasets``
parameter, which must match with the number of evaluated cluster sets:

.. code:: ipython2

    names = [f"Clustering {i}" for i in range(1, 6)]
    df_eval = aac.eval(X, list_labels=list_labels, names_datasets=names)
    aa.display_df(df_eval)



.. raw:: html

    <style type="text/css">
    #T_54308 thead th {
      background-color: white;
      color: black;
    }
    #T_54308 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_54308 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_54308 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_54308  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_54308 table {
      font-size: 12px;
    }
    </style>
    <table id="T_54308" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_54308_level0_col0" class="col_heading level0 col0" >name</th>
          <th id="T_54308_level0_col1" class="col_heading level0 col1" >n_clusters</th>
          <th id="T_54308_level0_col2" class="col_heading level0 col2" >BIC</th>
          <th id="T_54308_level0_col3" class="col_heading level0 col3" >CH</th>
          <th id="T_54308_level0_col4" class="col_heading level0 col4" >SC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_54308_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_54308_row0_col0" class="data row0 col0" >Clustering 1</td>
          <td id="T_54308_row0_col1" class="data row0 col1" >5</td>
          <td id="T_54308_row0_col2" class="data row0 col2" >-235.010922</td>
          <td id="T_54308_row0_col3" class="data row0 col3" >119.193092</td>
          <td id="T_54308_row0_col4" class="data row0 col4" >0.187077</td>
        </tr>
        <tr>
          <th id="T_54308_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_54308_row1_col0" class="data row1 col0" >Clustering 2</td>
          <td id="T_54308_row1_col1" class="data row1 col1" >10</td>
          <td id="T_54308_row1_col2" class="data row1 col2" >503.637472</td>
          <td id="T_54308_row1_col3" class="data row1 col3" >85.200434</td>
          <td id="T_54308_row1_col4" class="data row1 col4" >0.186997</td>
        </tr>
        <tr>
          <th id="T_54308_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_54308_row2_col0" class="data row2 col0" >Clustering 3</td>
          <td id="T_54308_row2_col1" class="data row2 col1" >25</td>
          <td id="T_54308_row2_col2" class="data row2 col2" >893.834929</td>
          <td id="T_54308_row2_col3" class="data row2 col3" >52.008981</td>
          <td id="T_54308_row2_col4" class="data row2 col4" >0.171144</td>
        </tr>
        <tr>
          <th id="T_54308_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_54308_row3_col0" class="data row3 col0" >Clustering 4</td>
          <td id="T_54308_row3_col1" class="data row3 col1" >50</td>
          <td id="T_54308_row3_col2" class="data row3 col2" >279.310802</td>
          <td id="T_54308_row3_col3" class="data row3 col3" >34.231471</td>
          <td id="T_54308_row3_col4" class="data row3 col4" >0.139191</td>
        </tr>
        <tr>
          <th id="T_54308_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_54308_row4_col0" class="data row4 col0" >Clustering 5</td>
          <td id="T_54308_row4_col1" class="data row4 col1" >100</td>
          <td id="T_54308_row4_col2" class="data row4 col2" >-1485.636265</td>
          <td id="T_54308_row4_col3" class="data row4 col3" >23.665256</td>
          <td id="T_54308_row4_col4" class="data row4 col4" >0.150045</td>
        </tr>
      </tbody>
    </table>



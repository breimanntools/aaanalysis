Create a small example dataset for dPUlearn containing positive (1),
unlabeled (2) data samples and the identified negatives (0):

.. code:: ipython2

    import aaanalysis as aa
    import pandas as pd
    import numpy as np
    aa.options["verbose"] = False
    X = np.array([[0.2, 0.1], [0.1, 0.15], [0.25, 0.2], [0.2, 0.3], [0.5, 0.7]])
    # Three different sets of labels 
    list_labels = [[1, 1, 2, 0, 0], [1, 1, 0, 2, 0], [1, 1, 0, 0, 2]]

Use the ``dPULearn.eval()`` method to obtain the evaluation for each
label set:

.. code:: ipython2

    dpul = aa.dPULearn()
    df_eval = dpul.eval(X, list_labels=list_labels)
    aa.display_df(df_eval)



.. raw:: html

    <style type="text/css">
    #T_28951 thead th {
      background-color: white;
      color: black;
    }
    #T_28951 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_28951 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_28951 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_28951  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_28951 table {
      font-size: 12px;
    }
    </style>
    <table id="T_28951" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_28951_level0_col0" class="col_heading level0 col0" >name</th>
          <th id="T_28951_level0_col1" class="col_heading level0 col1" >n_rel_neg</th>
          <th id="T_28951_level0_col2" class="col_heading level0 col2" >avg_STD</th>
          <th id="T_28951_level0_col3" class="col_heading level0 col3" >avg_IQR</th>
          <th id="T_28951_level0_col4" class="col_heading level0 col4" >avg_abs_AUC_pos</th>
          <th id="T_28951_level0_col5" class="col_heading level0 col5" >avg_abs_AUC_unl</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_28951_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_28951_row0_col0" class="data row0 col0" >Set 1</td>
          <td id="T_28951_row0_col1" class="data row0 col1" >2</td>
          <td id="T_28951_row0_col2" class="data row0 col2" >0.175000</td>
          <td id="T_28951_row0_col3" class="data row0 col3" >0.175000</td>
          <td id="T_28951_row0_col4" class="data row0 col4" >0.437500</td>
          <td id="T_28951_row0_col5" class="data row0 col5" >0.250000</td>
        </tr>
        <tr>
          <th id="T_28951_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_28951_row1_col0" class="data row1 col0" >Set 2</td>
          <td id="T_28951_row1_col1" class="data row1 col1" >2</td>
          <td id="T_28951_row1_col2" class="data row1 col2" >0.187500</td>
          <td id="T_28951_row1_col3" class="data row1 col3" >0.187500</td>
          <td id="T_28951_row1_col4" class="data row1 col4" >0.500000</td>
          <td id="T_28951_row1_col5" class="data row1 col5" >0.250000</td>
        </tr>
        <tr>
          <th id="T_28951_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_28951_row2_col0" class="data row2 col0" >Set 3</td>
          <td id="T_28951_row2_col1" class="data row2 col1" >2</td>
          <td id="T_28951_row2_col2" class="data row2 col2" >0.037500</td>
          <td id="T_28951_row2_col3" class="data row2 col3" >0.037500</td>
          <td id="T_28951_row2_col4" class="data row2 col4" >0.437500</td>
          <td id="T_28951_row2_col5" class="data row2 col5" >0.500000</td>
        </tr>
      </tbody>
    </table>



The dataset names given in the ‘name’ column or can be customized,
typically using the name of the identification method, e.g., ‘euclidean’
for Euclidean distance-based. This can be achieved by setting
``names_datasets``:

.. code:: ipython2

    names_datasets = ["Dataset 1", "Dataset 2", "Dataset 3"]
    df_eval = dpul.eval(X, list_labels=list_labels, names_datasets=names_datasets)
    aa.display_df(df_eval)



.. raw:: html

    <style type="text/css">
    #T_0eddb thead th {
      background-color: white;
      color: black;
    }
    #T_0eddb tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_0eddb tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_0eddb th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_0eddb  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_0eddb table {
      font-size: 12px;
    }
    </style>
    <table id="T_0eddb" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_0eddb_level0_col0" class="col_heading level0 col0" >name</th>
          <th id="T_0eddb_level0_col1" class="col_heading level0 col1" >n_rel_neg</th>
          <th id="T_0eddb_level0_col2" class="col_heading level0 col2" >avg_STD</th>
          <th id="T_0eddb_level0_col3" class="col_heading level0 col3" >avg_IQR</th>
          <th id="T_0eddb_level0_col4" class="col_heading level0 col4" >avg_abs_AUC_pos</th>
          <th id="T_0eddb_level0_col5" class="col_heading level0 col5" >avg_abs_AUC_unl</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_0eddb_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_0eddb_row0_col0" class="data row0 col0" >Dataset 1</td>
          <td id="T_0eddb_row0_col1" class="data row0 col1" >2</td>
          <td id="T_0eddb_row0_col2" class="data row0 col2" >0.175000</td>
          <td id="T_0eddb_row0_col3" class="data row0 col3" >0.175000</td>
          <td id="T_0eddb_row0_col4" class="data row0 col4" >0.437500</td>
          <td id="T_0eddb_row0_col5" class="data row0 col5" >0.250000</td>
        </tr>
        <tr>
          <th id="T_0eddb_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_0eddb_row1_col0" class="data row1 col0" >Dataset 2</td>
          <td id="T_0eddb_row1_col1" class="data row1 col1" >2</td>
          <td id="T_0eddb_row1_col2" class="data row1 col2" >0.187500</td>
          <td id="T_0eddb_row1_col3" class="data row1 col3" >0.187500</td>
          <td id="T_0eddb_row1_col4" class="data row1 col4" >0.500000</td>
          <td id="T_0eddb_row1_col5" class="data row1 col5" >0.250000</td>
        </tr>
        <tr>
          <th id="T_0eddb_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_0eddb_row2_col0" class="data row2 col0" >Dataset 3</td>
          <td id="T_0eddb_row2_col1" class="data row2 col1" >2</td>
          <td id="T_0eddb_row2_col2" class="data row2 col2" >0.037500</td>
          <td id="T_0eddb_row2_col3" class="data row2 col3" >0.037500</td>
          <td id="T_0eddb_row2_col4" class="data row2 col4" >0.437500</td>
          <td id="T_0eddb_row2_col5" class="data row2 col5" >0.500000</td>
        </tr>
      </tbody>
    </table>



The ``df_eval`` DataFrame provides two categories of quality measures:

1. **Homogeneity Within Negatives**: Measured by ‘avg_STD’ and
   ‘avg_IQR’, indicating the uniformity and spread of identified
   negatives.
2. **Dissimilarity With Other Groups**: Represented here by
   ‘avg_abs_AUC_pos/unl’, comparing identified negatives with positives
   (‘pos’, label 1) and unlabeled samples (‘unl’, label 2).

For a more comprehensive analysis, include ``X_neg`` as a feature matrix
of ground-truth negatives to assess their dissimilarity with the
identified negatives:

.. code:: ipython2

    X_neg = [[0.5, 0.8], [0.4, 0.4]]
    df_eval = dpul.eval(X, list_labels=list_labels, names_datasets=names_datasets, X_neg=X_neg)
    aa.display_df(df_eval)



.. raw:: html

    <style type="text/css">
    #T_d8be1 thead th {
      background-color: white;
      color: black;
    }
    #T_d8be1 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_d8be1 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_d8be1 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_d8be1  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_d8be1 table {
      font-size: 12px;
    }
    </style>
    <table id="T_d8be1" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_d8be1_level0_col0" class="col_heading level0 col0" >name</th>
          <th id="T_d8be1_level0_col1" class="col_heading level0 col1" >n_rel_neg</th>
          <th id="T_d8be1_level0_col2" class="col_heading level0 col2" >avg_STD</th>
          <th id="T_d8be1_level0_col3" class="col_heading level0 col3" >avg_IQR</th>
          <th id="T_d8be1_level0_col4" class="col_heading level0 col4" >avg_abs_AUC_pos</th>
          <th id="T_d8be1_level0_col5" class="col_heading level0 col5" >avg_abs_AUC_unl</th>
          <th id="T_d8be1_level0_col6" class="col_heading level0 col6" >avg_abs_AUC_neg</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_d8be1_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_d8be1_row0_col0" class="data row0 col0" >Dataset 1</td>
          <td id="T_d8be1_row0_col1" class="data row0 col1" >2</td>
          <td id="T_d8be1_row0_col2" class="data row0 col2" >0.175000</td>
          <td id="T_d8be1_row0_col3" class="data row0 col3" >0.175000</td>
          <td id="T_d8be1_row0_col4" class="data row0 col4" >0.437500</td>
          <td id="T_d8be1_row0_col5" class="data row0 col5" >0.250000</td>
          <td id="T_d8be1_row0_col6" class="data row0 col6" >0.187500</td>
        </tr>
        <tr>
          <th id="T_d8be1_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_d8be1_row1_col0" class="data row1 col0" >Dataset 2</td>
          <td id="T_d8be1_row1_col1" class="data row1 col1" >2</td>
          <td id="T_d8be1_row1_col2" class="data row1 col2" >0.187500</td>
          <td id="T_d8be1_row1_col3" class="data row1 col3" >0.187500</td>
          <td id="T_d8be1_row1_col4" class="data row1 col4" >0.500000</td>
          <td id="T_d8be1_row1_col5" class="data row1 col5" >0.250000</td>
          <td id="T_d8be1_row1_col6" class="data row1 col6" >0.187500</td>
        </tr>
        <tr>
          <th id="T_d8be1_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_d8be1_row2_col0" class="data row2 col0" >Dataset 3</td>
          <td id="T_d8be1_row2_col1" class="data row2 col1" >2</td>
          <td id="T_d8be1_row2_col2" class="data row2 col2" >0.037500</td>
          <td id="T_d8be1_row2_col3" class="data row2 col3" >0.037500</td>
          <td id="T_d8be1_row2_col4" class="data row2 col4" >0.437500</td>
          <td id="T_d8be1_row2_col5" class="data row2 col5" >0.500000</td>
          <td id="T_d8be1_row2_col6" class="data row2 col6" >0.500000</td>
        </tr>
      </tbody>
    </table>



If the variance within the data is high enough, the Kullback-Leibler
Divergence (KLD) can be computed to assess the dissimilarity of
distributions between the identified negatives and the other groups:

.. code:: ipython2

    # Extend the unlabeled group by one sample to fulfill variance requirements
    X = np.array([[0.2, 0.1], [0.1, 0.15], [0.25, 0.2], [0.2, 0.3], [0.5, 0.7], [0.6, 0.8]])
    list_labels = [[1, 1, 2, 0, 0, 2], [1, 1, 0, 2, 0, 2], [1, 1, 0, 0, 2, 2]]
    df_eval = dpul.eval(X, list_labels=list_labels, names_datasets=names_datasets, X_neg=X_neg, comp_kld=True)
    aa.display_df(df_eval)



.. raw:: html

    <style type="text/css">
    #T_6bb29 thead th {
      background-color: white;
      color: black;
    }
    #T_6bb29 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_6bb29 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_6bb29 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_6bb29  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_6bb29 table {
      font-size: 12px;
    }
    </style>
    <table id="T_6bb29" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_6bb29_level0_col0" class="col_heading level0 col0" >name</th>
          <th id="T_6bb29_level0_col1" class="col_heading level0 col1" >n_rel_neg</th>
          <th id="T_6bb29_level0_col2" class="col_heading level0 col2" >avg_STD</th>
          <th id="T_6bb29_level0_col3" class="col_heading level0 col3" >avg_IQR</th>
          <th id="T_6bb29_level0_col4" class="col_heading level0 col4" >avg_abs_AUC_pos</th>
          <th id="T_6bb29_level0_col5" class="col_heading level0 col5" >avg_KLD_pos</th>
          <th id="T_6bb29_level0_col6" class="col_heading level0 col6" >avg_abs_AUC_unl</th>
          <th id="T_6bb29_level0_col7" class="col_heading level0 col7" >avg_KLD_unl</th>
          <th id="T_6bb29_level0_col8" class="col_heading level0 col8" >avg_abs_AUC_neg</th>
          <th id="T_6bb29_level0_col9" class="col_heading level0 col9" >avg_KLD_neg</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_6bb29_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_6bb29_row0_col0" class="data row0 col0" >Dataset 1</td>
          <td id="T_6bb29_row0_col1" class="data row0 col1" >2</td>
          <td id="T_6bb29_row0_col2" class="data row0 col2" >0.175000</td>
          <td id="T_6bb29_row0_col3" class="data row0 col3" >0.175000</td>
          <td id="T_6bb29_row0_col4" class="data row0 col4" >0.437500</td>
          <td id="T_6bb29_row0_col5" class="data row0 col5" >1.414400</td>
          <td id="T_6bb29_row0_col6" class="data row0 col6" >0.125000</td>
          <td id="T_6bb29_row0_col7" class="data row0 col7" >0.003100</td>
          <td id="T_6bb29_row0_col8" class="data row0 col8" >0.187500</td>
          <td id="T_6bb29_row0_col9" class="data row0 col9" >0.181300</td>
        </tr>
        <tr>
          <th id="T_6bb29_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_6bb29_row1_col0" class="data row1 col0" >Dataset 2</td>
          <td id="T_6bb29_row1_col1" class="data row1 col1" >2</td>
          <td id="T_6bb29_row1_col2" class="data row1 col2" >0.187500</td>
          <td id="T_6bb29_row1_col3" class="data row1 col3" >0.187500</td>
          <td id="T_6bb29_row1_col4" class="data row1 col4" >0.500000</td>
          <td id="T_6bb29_row1_col5" class="data row1 col5" >1.366900</td>
          <td id="T_6bb29_row1_col6" class="data row1 col6" >0.125000</td>
          <td id="T_6bb29_row1_col7" class="data row1 col7" >0.003300</td>
          <td id="T_6bb29_row1_col8" class="data row1 col8" >0.187500</td>
          <td id="T_6bb29_row1_col9" class="data row1 col9" >0.104100</td>
        </tr>
        <tr>
          <th id="T_6bb29_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_6bb29_row2_col0" class="data row2 col0" >Dataset 3</td>
          <td id="T_6bb29_row2_col1" class="data row2 col1" >2</td>
          <td id="T_6bb29_row2_col2" class="data row2 col2" >0.037500</td>
          <td id="T_6bb29_row2_col3" class="data row2 col3" >0.037500</td>
          <td id="T_6bb29_row2_col4" class="data row2 col4" >0.437500</td>
          <td id="T_6bb29_row2_col5" class="data row2 col5" >1.016800</td>
          <td id="T_6bb29_row2_col6" class="data row2 col6" >0.500000</td>
          <td id="T_6bb29_row2_col7" class="data row2 col7" >30.317900</td>
          <td id="T_6bb29_row2_col8" class="data row2 col8" >0.500000</td>
          <td id="T_6bb29_row2_col9" class="data row2 col9" >12.020200</td>
        </tr>
      </tbody>
    </table>



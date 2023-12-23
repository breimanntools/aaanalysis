The ``dPULearn.compare_sets_negatives`` method facilitates the
comparison of identified negative samples across datasets. Providing
identified negatives represented by ‘0’ in the ``list_labels`` input, it
returns a DataFrame (typically named ``df_neg_comp``) where each row is
a sample and each column a dataset, indicating whether the sample is
identified as a negative (True) or not (False) in the respective
dataset:

.. code:: ipython2

    import aaanalysis as aa
    list_labels = [[1, 1, 0, 0, 2], [1, 1, 0, 2, 0], [1, 1, 2, 0, 0]]
    dpul = aa.dPULearn()
    df_neg_comp = dpul.compare_sets_negatives(list_labels=list_labels)
    aa.display_df(df_neg_comp)



.. raw:: html

    <style type="text/css">
    #T_eceeb thead th {
      background-color: white;
      color: black;
    }
    #T_eceeb tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_eceeb tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_eceeb th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_eceeb  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_eceeb table {
      font-size: 12px;
    }
    </style>
    <table id="T_eceeb" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_eceeb_level0_col0" class="col_heading level0 col0" >Set 1</th>
          <th id="T_eceeb_level0_col1" class="col_heading level0 col1" >Set 2</th>
          <th id="T_eceeb_level0_col2" class="col_heading level0 col2" >Set 3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_eceeb_level0_row0" class="row_heading level0 row0" >3</th>
          <td id="T_eceeb_row0_col0" class="data row0 col0" >True</td>
          <td id="T_eceeb_row0_col1" class="data row0 col1" >True</td>
          <td id="T_eceeb_row0_col2" class="data row0 col2" >False</td>
        </tr>
        <tr>
          <th id="T_eceeb_level0_row1" class="row_heading level0 row1" >4</th>
          <td id="T_eceeb_row1_col0" class="data row1 col0" >True</td>
          <td id="T_eceeb_row1_col1" class="data row1 col1" >False</td>
          <td id="T_eceeb_row1_col2" class="data row1 col2" >True</td>
        </tr>
        <tr>
          <th id="T_eceeb_level0_row2" class="row_heading level0 row2" >5</th>
          <td id="T_eceeb_row2_col0" class="data row2 col0" >False</td>
          <td id="T_eceeb_row2_col1" class="data row2 col1" >True</td>
          <td id="T_eceeb_row2_col2" class="data row2 col2" >True</td>
        </tr>
      </tbody>
    </table>



By default, only rows containing at least one identified negative are
returned. To return all rows, set ``remove_non_neg=False``:

.. code:: ipython2

    df_neg_comp = dpul.compare_sets_negatives(list_labels=list_labels, remove_non_neg=False)
    aa.display_df(df_neg_comp)



.. raw:: html

    <style type="text/css">
    #T_12ac1 thead th {
      background-color: white;
      color: black;
    }
    #T_12ac1 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_12ac1 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_12ac1 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_12ac1  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_12ac1 table {
      font-size: 12px;
    }
    </style>
    <table id="T_12ac1" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_12ac1_level0_col0" class="col_heading level0 col0" >Set 1</th>
          <th id="T_12ac1_level0_col1" class="col_heading level0 col1" >Set 2</th>
          <th id="T_12ac1_level0_col2" class="col_heading level0 col2" >Set 3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_12ac1_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_12ac1_row0_col0" class="data row0 col0" >False</td>
          <td id="T_12ac1_row0_col1" class="data row0 col1" >False</td>
          <td id="T_12ac1_row0_col2" class="data row0 col2" >False</td>
        </tr>
        <tr>
          <th id="T_12ac1_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_12ac1_row1_col0" class="data row1 col0" >False</td>
          <td id="T_12ac1_row1_col1" class="data row1 col1" >False</td>
          <td id="T_12ac1_row1_col2" class="data row1 col2" >False</td>
        </tr>
        <tr>
          <th id="T_12ac1_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_12ac1_row2_col0" class="data row2 col0" >True</td>
          <td id="T_12ac1_row2_col1" class="data row2 col1" >True</td>
          <td id="T_12ac1_row2_col2" class="data row2 col2" >False</td>
        </tr>
        <tr>
          <th id="T_12ac1_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_12ac1_row3_col0" class="data row3 col0" >True</td>
          <td id="T_12ac1_row3_col1" class="data row3 col1" >False</td>
          <td id="T_12ac1_row3_col2" class="data row3 col2" >True</td>
        </tr>
        <tr>
          <th id="T_12ac1_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_12ac1_row4_col0" class="data row4 col0" >False</td>
          <td id="T_12ac1_row4_col1" class="data row4 col1" >True</td>
          <td id="T_12ac1_row4_col2" class="data row4 col2" >True</td>
        </tr>
      </tbody>
    </table>



Names of the datasets can be provided by the ``names`` argument:

.. code:: ipython2

    names = ["Dataset 1", "Dataset 2", "Dataset 3"]
    df_neg_comp = dpul.compare_sets_negatives(list_labels=list_labels, names=names)
    aa.display_df(df_neg_comp)



.. raw:: html

    <style type="text/css">
    #T_3ce19 thead th {
      background-color: white;
      color: black;
    }
    #T_3ce19 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_3ce19 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_3ce19 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_3ce19  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_3ce19 table {
      font-size: 12px;
    }
    </style>
    <table id="T_3ce19" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_3ce19_level0_col0" class="col_heading level0 col0" >Dataset 1</th>
          <th id="T_3ce19_level0_col1" class="col_heading level0 col1" >Dataset 2</th>
          <th id="T_3ce19_level0_col2" class="col_heading level0 col2" >Dataset 3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_3ce19_level0_row0" class="row_heading level0 row0" >3</th>
          <td id="T_3ce19_row0_col0" class="data row0 col0" >True</td>
          <td id="T_3ce19_row0_col1" class="data row0 col1" >True</td>
          <td id="T_3ce19_row0_col2" class="data row0 col2" >False</td>
        </tr>
        <tr>
          <th id="T_3ce19_level0_row1" class="row_heading level0 row1" >4</th>
          <td id="T_3ce19_row1_col0" class="data row1 col0" >True</td>
          <td id="T_3ce19_row1_col1" class="data row1 col1" >False</td>
          <td id="T_3ce19_row1_col2" class="data row1 col2" >True</td>
        </tr>
        <tr>
          <th id="T_3ce19_level0_row2" class="row_heading level0 row2" >5</th>
          <td id="T_3ce19_row2_col0" class="data row2 col0" >False</td>
          <td id="T_3ce19_row2_col1" class="data row2 col1" >True</td>
          <td id="T_3ce19_row2_col2" class="data row2 col2" >True</td>
        </tr>
      </tbody>
    </table>



A DataFrame with sequence information (``df_seq``) and an requiered
‘entry’ column can be provdied, which is then merged with the
``df_neg_comp`` output DataFrame:

.. code:: ipython2

    import pandas as pd
    df_seq = pd.DataFrame([("entry1", "AA"), ("entry2", "BB"), ("entry3", "CC"), ("entry4", "DD"), ("entry5", "EE")], columns=["entry", "sequence"])
    df_neg_comp = dpul.compare_sets_negatives(list_labels=list_labels, df_seq=df_seq)
    aa.display_df(df_neg_comp)



.. raw:: html

    <style type="text/css">
    #T_aa59d thead th {
      background-color: white;
      color: black;
    }
    #T_aa59d tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_aa59d tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_aa59d th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_aa59d  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_aa59d table {
      font-size: 12px;
    }
    </style>
    <table id="T_aa59d" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_aa59d_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_aa59d_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_aa59d_level0_col2" class="col_heading level0 col2" >Set 1</th>
          <th id="T_aa59d_level0_col3" class="col_heading level0 col3" >Set 2</th>
          <th id="T_aa59d_level0_col4" class="col_heading level0 col4" >Set 3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_aa59d_level0_row0" class="row_heading level0 row0" >3</th>
          <td id="T_aa59d_row0_col0" class="data row0 col0" >entry3</td>
          <td id="T_aa59d_row0_col1" class="data row0 col1" >CC</td>
          <td id="T_aa59d_row0_col2" class="data row0 col2" >True</td>
          <td id="T_aa59d_row0_col3" class="data row0 col3" >True</td>
          <td id="T_aa59d_row0_col4" class="data row0 col4" >False</td>
        </tr>
        <tr>
          <th id="T_aa59d_level0_row1" class="row_heading level0 row1" >4</th>
          <td id="T_aa59d_row1_col0" class="data row1 col0" >entry4</td>
          <td id="T_aa59d_row1_col1" class="data row1 col1" >DD</td>
          <td id="T_aa59d_row1_col2" class="data row1 col2" >True</td>
          <td id="T_aa59d_row1_col3" class="data row1 col3" >False</td>
          <td id="T_aa59d_row1_col4" class="data row1 col4" >True</td>
        </tr>
        <tr>
          <th id="T_aa59d_level0_row2" class="row_heading level0 row2" >5</th>
          <td id="T_aa59d_row2_col0" class="data row2 col0" >entry5</td>
          <td id="T_aa59d_row2_col1" class="data row2 col1" >EE</td>
          <td id="T_aa59d_row2_col2" class="data row2 col2" >False</td>
          <td id="T_aa59d_row2_col3" class="data row2 col3" >True</td>
          <td id="T_aa59d_row2_col4" class="data row2 col4" >True</td>
        </tr>
      </tbody>
    </table>



Such overlaps are conveniently visualized using Venn diagrams, but they
are limited to a maximum of three datasets. For comparing more than
three datasets, an Upset Plot is a better choice. To facilitate this,
set ``return_upset_data=True`` to generate a data structure directly
compatible with the Upset Plot visualizations:

.. code:: ipython2

    from upsetplot import plot
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    list_labels = [[1, 1, 0, 2, 2], [1, 1, 2, 0, 0], [1, 1, 2, 0, 0], [1, 1, 0, 0, 0]]
    upset_data = dpul.compare_sets_negatives(list_labels=list_labels, return_upset_data=True)
    plot(upset_data, show_counts='%d')
    plt.suptitle("Overlap of identified negatives in different datasets")
    plt.show()



.. image:: examples/dpul_compare_sets_negatives_1_output_9_0.png


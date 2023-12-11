An overview dataset table is provided as default:

.. code:: ipython2

    import aaanalysis as aa
    df_info = aa.load_dataset()
    aa.display_df(df=df_info, show_shape=True)


::


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 import aaanalysis as aa
          2 df_info = aa.load_dataset()
          3 aa.display_df(df=df_info, show_shape=True)


    File ~/Programming/Pycharm_Projekte/1Packages/aaanalysis/aaanalysis/__init__.py:2
          1 from .data_handling import load_dataset, load_scales, load_features, to_fasta
    ----> 2 from .feature_engineering import AAclust, AAclustPlot, SequenceFeature, CPP, CPPPlot
          3 from .pu_learning import dPULearn
          4 from .explainable_ai import TreeModel, ShapModel


    File ~/Programming/Pycharm_Projekte/1Packages/aaanalysis/aaanalysis/feature_engineering/__init__.py:3
          1 from ._aaclust import AAclust
          2 from ._aaclust_plot import AAclustPlot
    ----> 3 from ._cpp_plot import CPPPlot
          4 from ._cpp import CPP
          5 from ._sequence_feature import SequenceFeature


    File ~/Programming/Pycharm_Projekte/1Packages/aaanalysis/aaanalysis/feature_engineering/_cpp_plot.py:14
         11 import aaanalysis as aa
         12 import aaanalysis.utils as ut
    ---> 14 from ._backend.check_feature import (check_split_kws,
         15                                      check_parts_len, check_match_features_seq_parts,
         16                                      check_df_seq,
         17                                      check_df_parts, check_match_df_parts_features, check_match_df_parts_list_parts,
         18                                      check_df_scales, check_match_df_scales_features,
         19                                      check_df_cat, check_match_df_cat_features,
         20                                      check_match_df_parts_df_scales, check_match_df_scales_df_cat)
         21 from ._backend.cpp.utils_cpp_plot import get_optimal_fontsize
         23 from ._backend.cpp.cpp_plot_feature import plot_feature


    File ~/Programming/Pycharm_Projekte/1Packages/aaanalysis/aaanalysis/feature_engineering/_backend/check_feature.py:9
          6 import warnings
          8 import aaanalysis.utils as ut
    ----> 9 from .cpp._utils_feature import get_parts
         12 # Helper functions
         13 def _get_min_pos_split(split=None):


    ModuleNotFoundError: No module named 'aaanalysis.feature_engineering._backend.cpp._utils_feature'


Load one of the datasets from the overview table by using a name from the 'Dataset' column (e.g., 'SEQ_CAPSID'). The number of proteins per class can be adjusted by the 'n' parameter:

.. code:: ipython2

    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=2)
    aa.display_df(df=df_seq, char_limit=40)



.. raw:: html

    <style type="text/css">
    #T_69131 thead th {
      background-color: white;
      color: black;
    }
    #T_69131 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_69131 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_69131 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_69131  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_69131 table {
      font-size: 12px;
    }
    </style>
    <table id="T_69131" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_69131_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_69131_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_69131_level0_col2" class="col_heading level0 col2" >label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_69131_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_69131_row0_col0" class="data row0 col0" >CAPSID_1</td>
          <td id="T_69131_row0_col1" class="data row0 col1" >MVTHNVKINKHVTRRSYSSA...KGDDDDTPRIPATKLDEENV</td>
          <td id="T_69131_row0_col2" class="data row0 col2" >0</td>
        </tr>
        <tr>
          <th id="T_69131_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_69131_row1_col0" class="data row1 col0" >CAPSID_2</td>
          <td id="T_69131_row1_col1" class="data row1 col1" >MKKRQKKMTLSNFTDTSFQD...VFMRMAMLEAVINARHFGEE</td>
          <td id="T_69131_row1_col2" class="data row1 col2" >0</td>
        </tr>
        <tr>
          <th id="T_69131_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_69131_row2_col0" class="data row2 col0" >CAPSID_4072</td>
          <td id="T_69131_row2_col1" class="data row2 col1" >MALTTNDVITEDFVRETVEE...IFTRKAWKAIFPEAAVKVDA</td>
          <td id="T_69131_row2_col2" class="data row2 col2" >1</td>
        </tr>
        <tr>
          <th id="T_69131_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_69131_row3_col0" class="data row3 col0" >CAPSID_4073</td>
          <td id="T_69131_row3_col1" class="data row3 col1" >MGELTDNGVQLAKAQIGKHQ...TIGQLTCTNPAAHAKIRDLK</td>
          <td id="T_69131_row3_col2" class="data row3 col2" >1</td>
        </tr>
      </tbody>
    </table>



Samples for amino acid ('AA') level datasets are provided by specyfing their amino acid window size using the  TODO ...

.. code:: ipython2

    df_aa = aa.load_dataset(name="AA_CASPASE3", n=2)
    aa.display_df(df=df_aa)



.. raw:: html

    <style type="text/css">
    #T_a8670 thead th {
      background-color: white;
      color: black;
    }
    #T_a8670 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_a8670 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_a8670 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_a8670  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_a8670 table {
      font-size: 12px;
    }
    </style>
    <table id="T_a8670" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_a8670_level0_col0" class="col_heading level0 col0" >entry</th>
          <th id="T_a8670_level0_col1" class="col_heading level0 col1" >sequence</th>
          <th id="T_a8670_level0_col2" class="col_heading level0 col2" >label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_a8670_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_a8670_row0_col0" class="data row0 col0" >CASPASE3_1_pos126</td>
          <td id="T_a8670_row0_col1" class="data row0 col1" >QTLRDSMLK</td>
          <td id="T_a8670_row0_col2" class="data row0 col2" >1</td>
        </tr>
        <tr>
          <th id="T_a8670_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_a8670_row1_col0" class="data row1 col0" >CASPASE3_1_pos127</td>
          <td id="T_a8670_row1_col1" class="data row1 col1" >TLRDSMLKY</td>
          <td id="T_a8670_row1_col2" class="data row1 col2" >1</td>
        </tr>
        <tr>
          <th id="T_a8670_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_a8670_row2_col0" class="data row2 col0" >CASPASE3_1_pos4</td>
          <td id="T_a8670_row2_col1" class="data row2 col1" >MSLFDLFRG</td>
          <td id="T_a8670_row2_col2" class="data row2 col2" >0</td>
        </tr>
        <tr>
          <th id="T_a8670_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_a8670_row3_col0" class="data row3 col0" >CASPASE3_1_pos5</td>
          <td id="T_a8670_row3_col1" class="data row3 col1" >SLFDLFRGF</td>
          <td id="T_a8670_row3_col2" class="data row3 col2" >0</td>
        </tr>
      </tbody>
    </table>



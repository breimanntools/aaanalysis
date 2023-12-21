Quick Start with AAanalysis
===========================

**AAanalysis** is a Python framework for sequence-based protein
prediction, centered around the *Comparative Physical Profiling (CPP)*
algorithm for interpretable feature engineering.

First, import some third-party packages and ``aanalsis``:

.. code:: ipython3

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    import aaanalysis as aa
    aa.options["verbose"] = False

We can load a dataset of amino acid scales and an example dataset for
γ-secretase of 50 substrates and 50 non-substrates:

.. code:: ipython3

    df_scales = aa.load_scales()
    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)

Starting the **Feature Engineering**, we can utilize the ``AAclust``
model for pre-selecting a redundancy-reduced set of amino acid scales:

.. code:: ipython3

    aac = aa.AAclust()
    X = np.array(df_scales).T
    scales = aac.fit(X, names=list(df_scales), n_clusters=100).medoid_names_ 
    df_scales = df_scales[scales]

We can now use the ``CPP`` algorithm, which aims at identifying a set of
features most discriminant between two sets of sequences. Its core idea
is the CPP feature concept, defined as a combination of *Parts*,
*Splits*, and *Scales*. Parts and Splits can be obtained using
``SequenceFeature``:

.. code:: ipython3

    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd"])
    split_kws = sf.get_split_kws(n_split_max=1, split_types=["Segment"])

Running CPP creates all ``Part-Split-Scale`` combinations and filters a
selected maximum of non-redundant features. As a baseline approach, we
use CPP without filtering (``max_cor=1``) to compute the average values
for the 100 selected scales over the entire TMD sequences:

.. code:: ipython3

    # Small set of CPP features (100 features are created)
    y = list(df_seq["label"])
    cpp = aa.CPP(df_scales=df_scales, df_parts=df_parts, split_kws=split_kws)
    df_feat = cpp.run(labels=y, max_cor=1)
    aa.display_df(df=df_feat.head(10))



.. raw:: html

    <style type="text/css">
    #T_c93a4 thead th {
      background-color: white;
      color: black;
    }
    #T_c93a4 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_c93a4 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_c93a4 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_c93a4  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_c93a4 table {
      font-size: 12px;
    }
    </style>
    <table id="T_c93a4" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_c93a4_level0_col0" class="col_heading level0 col0" >feature</th>
          <th id="T_c93a4_level0_col1" class="col_heading level0 col1" >category</th>
          <th id="T_c93a4_level0_col2" class="col_heading level0 col2" >subcategory</th>
          <th id="T_c93a4_level0_col3" class="col_heading level0 col3" >scale_name</th>
          <th id="T_c93a4_level0_col4" class="col_heading level0 col4" >scale_description</th>
          <th id="T_c93a4_level0_col5" class="col_heading level0 col5" >abs_auc</th>
          <th id="T_c93a4_level0_col6" class="col_heading level0 col6" >abs_mean_dif</th>
          <th id="T_c93a4_level0_col7" class="col_heading level0 col7" >mean_dif</th>
          <th id="T_c93a4_level0_col8" class="col_heading level0 col8" >std_test</th>
          <th id="T_c93a4_level0_col9" class="col_heading level0 col9" >std_ref</th>
          <th id="T_c93a4_level0_col10" class="col_heading level0 col10" >p_val_mann_whitney</th>
          <th id="T_c93a4_level0_col11" class="col_heading level0 col11" >p_val_fdr_bh</th>
          <th id="T_c93a4_level0_col12" class="col_heading level0 col12" >positions</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_c93a4_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_c93a4_row0_col0" class="data row0 col0" >TMD-Segment(1,1)-WOLR790101</td>
          <td id="T_c93a4_row0_col1" class="data row0 col1" >Polarity</td>
          <td id="T_c93a4_row0_col2" class="data row0 col2" >Hydrophobicity (surrounding)</td>
          <td id="T_c93a4_row0_col3" class="data row0 col3" >Hydration potential</td>
          <td id="T_c93a4_row0_col4" class="data row0 col4" >Hydrophobicity index (Wolfenden et al., 1979)</td>
          <td id="T_c93a4_row0_col5" class="data row0 col5" >0.246000</td>
          <td id="T_c93a4_row0_col6" class="data row0 col6" >0.032767</td>
          <td id="T_c93a4_row0_col7" class="data row0 col7" >0.032767</td>
          <td id="T_c93a4_row0_col8" class="data row0 col8" >0.028962</td>
          <td id="T_c93a4_row0_col9" class="data row0 col9" >0.037213</td>
          <td id="T_c93a4_row0_col10" class="data row0 col10" >0.000022</td>
          <td id="T_c93a4_row0_col11" class="data row0 col11" >0.002203</td>
          <td id="T_c93a4_row0_col12" class="data row0 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c93a4_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_c93a4_row1_col0" class="data row1 col0" >TMD-Segment(1,1)-FAUJ880109</td>
          <td id="T_c93a4_row1_col1" class="data row1 col1" >Energy</td>
          <td id="T_c93a4_row1_col2" class="data row1 col2" >Isoelectric point</td>
          <td id="T_c93a4_row1_col3" class="data row1 col3" >Number hydrogen bond donors</td>
          <td id="T_c93a4_row1_col4" class="data row1 col4" >Number of hydrogen bond donors (Fauchere et al., 1988)</td>
          <td id="T_c93a4_row1_col5" class="data row1 col5" >0.222000</td>
          <td id="T_c93a4_row1_col6" class="data row1 col6" >0.020946</td>
          <td id="T_c93a4_row1_col7" class="data row1 col7" >-0.020946</td>
          <td id="T_c93a4_row1_col8" class="data row1 col8" >0.020626</td>
          <td id="T_c93a4_row1_col9" class="data row1 col9" >0.026994</td>
          <td id="T_c93a4_row1_col10" class="data row1 col10" >0.000110</td>
          <td id="T_c93a4_row1_col11" class="data row1 col11" >0.005485</td>
          <td id="T_c93a4_row1_col12" class="data row1 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c93a4_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_c93a4_row2_col0" class="data row2 col0" >TMD-Segment(1,1)-CHOC760103</td>
          <td id="T_c93a4_row2_col1" class="data row2 col1" >ASA/Volume</td>
          <td id="T_c93a4_row2_col2" class="data row2 col2" >Buried</td>
          <td id="T_c93a4_row2_col3" class="data row2 col3" >Buried</td>
          <td id="T_c93a4_row2_col4" class="data row2 col4" >Proportion of residues 95% buried (Chothia, 1976)</td>
          <td id="T_c93a4_row2_col5" class="data row2 col5" >0.218000</td>
          <td id="T_c93a4_row2_col6" class="data row2 col6" >0.040137</td>
          <td id="T_c93a4_row2_col7" class="data row2 col7" >0.040137</td>
          <td id="T_c93a4_row2_col8" class="data row2 col8" >0.044550</td>
          <td id="T_c93a4_row2_col9" class="data row2 col9" >0.055506</td>
          <td id="T_c93a4_row2_col10" class="data row2 col10" >0.000174</td>
          <td id="T_c93a4_row2_col11" class="data row2 col11" >0.005728</td>
          <td id="T_c93a4_row2_col12" class="data row2 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c93a4_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_c93a4_row3_col0" class="data row3 col0" >TMD-Segment(1,1)-PRAM900101</td>
          <td id="T_c93a4_row3_col1" class="data row3 col1" >Polarity</td>
          <td id="T_c93a4_row3_col2" class="data row3 col2" >Hydrophilicity</td>
          <td id="T_c93a4_row3_col3" class="data row3 col3" >Polarity (hydrophilicity)</td>
          <td id="T_c93a4_row3_col4" class="data row3 col4" >Hydrophobicity (Prabhakaran, 1990)</td>
          <td id="T_c93a4_row3_col5" class="data row3 col5" >0.209000</td>
          <td id="T_c93a4_row3_col6" class="data row3 col6" >0.017235</td>
          <td id="T_c93a4_row3_col7" class="data row3 col7" >-0.017235</td>
          <td id="T_c93a4_row3_col8" class="data row3 col8" >0.016659</td>
          <td id="T_c93a4_row3_col9" class="data row3 col9" >0.025221</td>
          <td id="T_c93a4_row3_col10" class="data row3 col10" >0.000312</td>
          <td id="T_c93a4_row3_col11" class="data row3 col11" >0.005728</td>
          <td id="T_c93a4_row3_col12" class="data row3 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c93a4_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_c93a4_row4_col0" class="data row4 col0" >TMD-Segment(1,1)-YUTK870104</td>
          <td id="T_c93a4_row4_col1" class="data row4 col1" >Energy</td>
          <td id="T_c93a4_row4_col2" class="data row4 col2" >Free energy (unfolding)</td>
          <td id="T_c93a4_row4_col3" class="data row4 col3" >Free energy (unfolding)</td>
          <td id="T_c93a4_row4_col4" class="data row4 col4" >Activation Gibbs energy of unfolding, pH9.0 (Yutani et al., 1987)</td>
          <td id="T_c93a4_row4_col5" class="data row4 col5" >0.209000</td>
          <td id="T_c93a4_row4_col6" class="data row4 col6" >0.007919</td>
          <td id="T_c93a4_row4_col7" class="data row4 col7" >0.007919</td>
          <td id="T_c93a4_row4_col8" class="data row4 col8" >0.011043</td>
          <td id="T_c93a4_row4_col9" class="data row4 col9" >0.016763</td>
          <td id="T_c93a4_row4_col10" class="data row4 col10" >0.000311</td>
          <td id="T_c93a4_row4_col11" class="data row4 col11" >0.005728</td>
          <td id="T_c93a4_row4_col12" class="data row4 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c93a4_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_c93a4_row5_col0" class="data row5 col0" >TMD-Segment(1,1)-FUKS010106</td>
          <td id="T_c93a4_row5_col1" class="data row5 col1" >Composition</td>
          <td id="T_c93a4_row5_col2" class="data row5 col2" >Membrane proteins (MPs)</td>
          <td id="T_c93a4_row5_col3" class="data row5 col3" >Proteins of mesophiles (INT)</td>
          <td id="T_c93a4_row5_col4" class="data row5 col4" >Interior composition of amino acids in intracellular proteins of mesophiles [%] (Fukuchi-Nishikawa, 2001)</td>
          <td id="T_c93a4_row5_col5" class="data row5 col5" >0.206000</td>
          <td id="T_c93a4_row5_col6" class="data row5 col6" >0.058909</td>
          <td id="T_c93a4_row5_col7" class="data row5 col7" >0.058909</td>
          <td id="T_c93a4_row5_col8" class="data row5 col8" >0.068070</td>
          <td id="T_c93a4_row5_col9" class="data row5 col9" >0.081967</td>
          <td id="T_c93a4_row5_col10" class="data row5 col10" >0.000380</td>
          <td id="T_c93a4_row5_col11" class="data row5 col11" >0.005728</td>
          <td id="T_c93a4_row5_col12" class="data row5 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c93a4_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_c93a4_row6_col0" class="data row6 col0" >TMD-Segment(1,1)-VELV850101</td>
          <td id="T_c93a4_row6_col1" class="data row6 col1" >Energy</td>
          <td id="T_c93a4_row6_col2" class="data row6 col2" >Electron-ion interaction pot.</td>
          <td id="T_c93a4_row6_col3" class="data row6 col3" >Electron-ion interaction potential</td>
          <td id="T_c93a4_row6_col4" class="data row6 col4" >Electron-ion interaction potential (Veljkovic et al., 1985)</td>
          <td id="T_c93a4_row6_col5" class="data row6 col5" >0.203000</td>
          <td id="T_c93a4_row6_col6" class="data row6 col6" >0.045725</td>
          <td id="T_c93a4_row6_col7" class="data row6 col7" >-0.045725</td>
          <td id="T_c93a4_row6_col8" class="data row6 col8" >0.059791</td>
          <td id="T_c93a4_row6_col9" class="data row6 col9" >0.068804</td>
          <td id="T_c93a4_row6_col10" class="data row6 col10" >0.000480</td>
          <td id="T_c93a4_row6_col11" class="data row6 col11" >0.005728</td>
          <td id="T_c93a4_row6_col12" class="data row6 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c93a4_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_c93a4_row7_col0" class="data row7 col0" >TMD-Segment(1,1)-ROBB760109</td>
          <td id="T_c93a4_row7_col1" class="data row7 col1" >Conformation</td>
          <td id="T_c93a4_row7_col2" class="data row7 col2" >β-turn (N-term)</td>
          <td id="T_c93a4_row7_col3" class="data row7 col3" >β-turn (1st residue)</td>
          <td id="T_c93a4_row7_col4" class="data row7 col4" >Information measure for N-terminal turn (Robson-Suzuki, 1976)</td>
          <td id="T_c93a4_row7_col5" class="data row7 col5" >0.202000</td>
          <td id="T_c93a4_row7_col6" class="data row7 col6" >0.035480</td>
          <td id="T_c93a4_row7_col7" class="data row7 col7" >-0.035480</td>
          <td id="T_c93a4_row7_col8" class="data row7 col8" >0.039526</td>
          <td id="T_c93a4_row7_col9" class="data row7 col9" >0.049378</td>
          <td id="T_c93a4_row7_col10" class="data row7 col10" >0.000499</td>
          <td id="T_c93a4_row7_col11" class="data row7 col11" >0.005728</td>
          <td id="T_c93a4_row7_col12" class="data row7 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c93a4_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_c93a4_row8_col0" class="data row8 col0" >TMD-Segment(1,1)-CHAM830108</td>
          <td id="T_c93a4_row8_col1" class="data row8 col1" >Energy</td>
          <td id="T_c93a4_row8_col2" class="data row8 col2" >Charge</td>
          <td id="T_c93a4_row8_col3" class="data row8 col3" >Charge (donor)</td>
          <td id="T_c93a4_row8_col4" class="data row8 col4" >A parameter of charge transfer donor capability (Charton-Charton, 1983)</td>
          <td id="T_c93a4_row8_col5" class="data row8 col5" >0.200000</td>
          <td id="T_c93a4_row8_col6" class="data row8 col6" >0.071838</td>
          <td id="T_c93a4_row8_col7" class="data row8 col7" >-0.071838</td>
          <td id="T_c93a4_row8_col8" class="data row8 col8" >0.090338</td>
          <td id="T_c93a4_row8_col9" class="data row8 col9" >0.101652</td>
          <td id="T_c93a4_row8_col10" class="data row8 col10" >0.000516</td>
          <td id="T_c93a4_row8_col11" class="data row8 col11" >0.005728</td>
          <td id="T_c93a4_row8_col12" class="data row8 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c93a4_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_c93a4_row9_col0" class="data row9 col0" >TMD-Segment(1,1)-LINS030109</td>
          <td id="T_c93a4_row9_col1" class="data row9 col1" >ASA/Volume</td>
          <td id="T_c93a4_row9_col2" class="data row9 col2" >Accessible surface area (ASA)</td>
          <td id="T_c93a4_row9_col3" class="data row9 col3" >Hydrophilic ASA (folded proteins)</td>
          <td id="T_c93a4_row9_col4" class="data row9 col4" >% Hydrophilic accessible surfaces vs win3 of whole residues from folded proteins (Lins et al., 2003)</td>
          <td id="T_c93a4_row9_col5" class="data row9 col5" >0.200000</td>
          <td id="T_c93a4_row9_col6" class="data row9 col6" >0.026014</td>
          <td id="T_c93a4_row9_col7" class="data row9 col7" >-0.026014</td>
          <td id="T_c93a4_row9_col8" class="data row9 col8" >0.032572</td>
          <td id="T_c93a4_row9_col9" class="data row9 col9" >0.038047</td>
          <td id="T_c93a4_row9_col10" class="data row9 col10" >0.000574</td>
          <td id="T_c93a4_row9_col11" class="data row9 col11" >0.005742</td>
          <td id="T_c93a4_row9_col12" class="data row9 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
      </tbody>
    </table>



For **Machine Learning**, a feature matrix from a given set of CPP
features can be created using ``sf.feature_matrix``:

.. code:: ipython3

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    X = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"])
    rf = RandomForestClassifier()
    cv_base = cross_val_score(rf, X, y, scoring="accuracy", cv=5)
    print(f"Mean accuracy of {round(np.mean(cv_base), 2)}")


.. parsed-literal::

    Mean accuracy of 0.6


Creating more features with CPP will take a little time but improve
prediction performance:

.. code:: ipython3

    # CPP features with default parts and splits (around 100.000 features)
    df_parts = sf.get_df_parts(df_seq=df_seq)
    cpp = aa.CPP(df_scales=df_scales, df_parts=df_parts)
    df_feat = cpp.run(labels=y)
    aa.display_df(df=df_feat.head(10))
    
    X = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"])
    rf = RandomForestClassifier()
    cv = cross_val_score(rf, X, y, scoring="accuracy", cv=5) 
    print(f"Mean accuracy of {round(np.mean(cv), 2)}")
    
    # Plot comparison of two approaches
    aa.plot_settings()
    sns.barplot(pd.DataFrame({"Baseline": cv_base, "CPP": cv}), palette=["tab:blue", "tab:red"])
    plt.ylabel("Mean accuracy", size=aa.plot_gcfs()+1)
    plt.ylim(0, 1)
    plt.title("Comparison of Feature Engineering Methods", size=aa.plot_gcfs()-1)
    sns.despine()
    plt.show()



.. raw:: html

    <style type="text/css">
    #T_3243e thead th {
      background-color: white;
      color: black;
    }
    #T_3243e tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_3243e tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_3243e th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_3243e  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_3243e table {
      font-size: 12px;
    }
    </style>
    <table id="T_3243e" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_3243e_level0_col0" class="col_heading level0 col0" >feature</th>
          <th id="T_3243e_level0_col1" class="col_heading level0 col1" >category</th>
          <th id="T_3243e_level0_col2" class="col_heading level0 col2" >subcategory</th>
          <th id="T_3243e_level0_col3" class="col_heading level0 col3" >scale_name</th>
          <th id="T_3243e_level0_col4" class="col_heading level0 col4" >scale_description</th>
          <th id="T_3243e_level0_col5" class="col_heading level0 col5" >abs_auc</th>
          <th id="T_3243e_level0_col6" class="col_heading level0 col6" >abs_mean_dif</th>
          <th id="T_3243e_level0_col7" class="col_heading level0 col7" >mean_dif</th>
          <th id="T_3243e_level0_col8" class="col_heading level0 col8" >std_test</th>
          <th id="T_3243e_level0_col9" class="col_heading level0 col9" >std_ref</th>
          <th id="T_3243e_level0_col10" class="col_heading level0 col10" >p_val_mann_whitney</th>
          <th id="T_3243e_level0_col11" class="col_heading level0 col11" >p_val_fdr_bh</th>
          <th id="T_3243e_level0_col12" class="col_heading level0 col12" >positions</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_3243e_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_3243e_row0_col0" class="data row0 col0" >TMD_C_JMD_C-Pattern(N,1,5,8,12)-ROBB760109</td>
          <td id="T_3243e_row0_col1" class="data row0 col1" >Conformation</td>
          <td id="T_3243e_row0_col2" class="data row0 col2" >β-turn (N-term)</td>
          <td id="T_3243e_row0_col3" class="data row0 col3" >β-turn (1st residue)</td>
          <td id="T_3243e_row0_col4" class="data row0 col4" >Information measure for N-terminal turn (Robson-Suzuki, 1976)</td>
          <td id="T_3243e_row0_col5" class="data row0 col5" >0.377000</td>
          <td id="T_3243e_row0_col6" class="data row0 col6" >0.126610</td>
          <td id="T_3243e_row0_col7" class="data row0 col7" >-0.126610</td>
          <td id="T_3243e_row0_col8" class="data row0 col8" >0.062139</td>
          <td id="T_3243e_row0_col9" class="data row0 col9" >0.087645</td>
          <td id="T_3243e_row0_col10" class="data row0 col10" >0.000000</td>
          <td id="T_3243e_row0_col11" class="data row0 col11" >0.000000</td>
          <td id="T_3243e_row0_col12" class="data row0 col12" >21,25,28,32</td>
        </tr>
        <tr>
          <th id="T_3243e_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_3243e_row1_col0" class="data row1 col0" >TMD_C_JMD_C-Segment(4,5)-ZIMJ680104</td>
          <td id="T_3243e_row1_col1" class="data row1 col1" >Energy</td>
          <td id="T_3243e_row1_col2" class="data row1 col2" >Isoelectric point</td>
          <td id="T_3243e_row1_col3" class="data row1 col3" >Isoelectric point</td>
          <td id="T_3243e_row1_col4" class="data row1 col4" >Isoelectric point (Zimmerman et al., 1968)</td>
          <td id="T_3243e_row1_col5" class="data row1 col5" >0.373000</td>
          <td id="T_3243e_row1_col6" class="data row1 col6" >0.220000</td>
          <td id="T_3243e_row1_col7" class="data row1 col7" >0.220000</td>
          <td id="T_3243e_row1_col8" class="data row1 col8" >0.123716</td>
          <td id="T_3243e_row1_col9" class="data row1 col9" >0.137350</td>
          <td id="T_3243e_row1_col10" class="data row1 col10" >0.000000</td>
          <td id="T_3243e_row1_col11" class="data row1 col11" >0.000000</td>
          <td id="T_3243e_row1_col12" class="data row1 col12" >33,34,35,36</td>
        </tr>
        <tr>
          <th id="T_3243e_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_3243e_row2_col0" class="data row2 col0" >TMD_C_JMD_C-Segment(5,7)-LINS030101</td>
          <td id="T_3243e_row2_col1" class="data row2 col1" >ASA/Volume</td>
          <td id="T_3243e_row2_col2" class="data row2 col2" >Volume</td>
          <td id="T_3243e_row2_col3" class="data row2 col3" >Accessible surface area (ASA)</td>
          <td id="T_3243e_row2_col4" class="data row2 col4" >Total accessible surfaces of whole residues (backbone and lateral chain) calculated with a window 3, expressed in Å²  (Lins et al., 2003)</td>
          <td id="T_3243e_row2_col5" class="data row2 col5" >0.354000</td>
          <td id="T_3243e_row2_col6" class="data row2 col6" >0.237161</td>
          <td id="T_3243e_row2_col7" class="data row2 col7" >0.237161</td>
          <td id="T_3243e_row2_col8" class="data row2 col8" >0.145884</td>
          <td id="T_3243e_row2_col9" class="data row2 col9" >0.164285</td>
          <td id="T_3243e_row2_col10" class="data row2 col10" >0.000000</td>
          <td id="T_3243e_row2_col11" class="data row2 col11" >0.000001</td>
          <td id="T_3243e_row2_col12" class="data row2 col12" >32,33,34</td>
        </tr>
        <tr>
          <th id="T_3243e_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_3243e_row3_col0" class="data row3 col0" >TMD_C_JMD_C-Pattern(N,4,8,12,15)-MUNV940102</td>
          <td id="T_3243e_row3_col1" class="data row3 col1" >Energy</td>
          <td id="T_3243e_row3_col2" class="data row3 col2" >Free energy (folding)</td>
          <td id="T_3243e_row3_col3" class="data row3 col3" >Free energy (α-helix)</td>
          <td id="T_3243e_row3_col4" class="data row3 col4" >Free energy in alpha-helical region (Munoz-Serrano, 1994)</td>
          <td id="T_3243e_row3_col5" class="data row3 col5" >0.353000</td>
          <td id="T_3243e_row3_col6" class="data row3 col6" >0.119820</td>
          <td id="T_3243e_row3_col7" class="data row3 col7" >-0.119820</td>
          <td id="T_3243e_row3_col8" class="data row3 col8" >0.065320</td>
          <td id="T_3243e_row3_col9" class="data row3 col9" >0.098536</td>
          <td id="T_3243e_row3_col10" class="data row3 col10" >0.000000</td>
          <td id="T_3243e_row3_col11" class="data row3 col11" >0.000001</td>
          <td id="T_3243e_row3_col12" class="data row3 col12" >24,28,32,35</td>
        </tr>
        <tr>
          <th id="T_3243e_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_3243e_row4_col0" class="data row4 col0" >TMD_C_JMD_C-Segment(2,3)-VASM830101</td>
          <td id="T_3243e_row4_col1" class="data row4 col1" >Conformation</td>
          <td id="T_3243e_row4_col2" class="data row4 col2" >Unclassified (Conformation)</td>
          <td id="T_3243e_row4_col3" class="data row4 col3" >α-helix</td>
          <td id="T_3243e_row4_col4" class="data row4 col4" >Relative population of conformational state A (Vasquez et al., 1983)</td>
          <td id="T_3243e_row4_col5" class="data row4 col5" >0.345000</td>
          <td id="T_3243e_row4_col6" class="data row4 col6" >0.147010</td>
          <td id="T_3243e_row4_col7" class="data row4 col7" >0.147010</td>
          <td id="T_3243e_row4_col8" class="data row4 col8" >0.110459</td>
          <td id="T_3243e_row4_col9" class="data row4 col9" >0.091564</td>
          <td id="T_3243e_row4_col10" class="data row4 col10" >0.000000</td>
          <td id="T_3243e_row4_col11" class="data row4 col11" >0.000001</td>
          <td id="T_3243e_row4_col12" class="data row4 col12" >27,28,29,30,31,32,33</td>
        </tr>
        <tr>
          <th id="T_3243e_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_3243e_row5_col0" class="data row5 col0" >TMD_C_JMD_C-Segment(6,9)-ZIMJ680104</td>
          <td id="T_3243e_row5_col1" class="data row5 col1" >Energy</td>
          <td id="T_3243e_row5_col2" class="data row5 col2" >Isoelectric point</td>
          <td id="T_3243e_row5_col3" class="data row5 col3" >Isoelectric point</td>
          <td id="T_3243e_row5_col4" class="data row5 col4" >Isoelectric point (Zimmerman et al., 1968)</td>
          <td id="T_3243e_row5_col5" class="data row5 col5" >0.341000</td>
          <td id="T_3243e_row5_col6" class="data row5 col6" >0.263651</td>
          <td id="T_3243e_row5_col7" class="data row5 col7" >0.263651</td>
          <td id="T_3243e_row5_col8" class="data row5 col8" >0.187136</td>
          <td id="T_3243e_row5_col9" class="data row5 col9" >0.171995</td>
          <td id="T_3243e_row5_col10" class="data row5 col10" >0.000000</td>
          <td id="T_3243e_row5_col11" class="data row5 col11" >0.000001</td>
          <td id="T_3243e_row5_col12" class="data row5 col12" >32,33</td>
        </tr>
        <tr>
          <th id="T_3243e_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_3243e_row6_col0" class="data row6 col0" >TMD-Pattern(C,4,7,11)-ROBB760109</td>
          <td id="T_3243e_row6_col1" class="data row6 col1" >Conformation</td>
          <td id="T_3243e_row6_col2" class="data row6 col2" >β-turn (N-term)</td>
          <td id="T_3243e_row6_col3" class="data row6 col3" >β-turn (1st residue)</td>
          <td id="T_3243e_row6_col4" class="data row6 col4" >Information measure for N-terminal turn (Robson-Suzuki, 1976)</td>
          <td id="T_3243e_row6_col5" class="data row6 col5" >0.339000</td>
          <td id="T_3243e_row6_col6" class="data row6 col6" >0.133913</td>
          <td id="T_3243e_row6_col7" class="data row6 col7" >-0.133913</td>
          <td id="T_3243e_row6_col8" class="data row6 col8" >0.079916</td>
          <td id="T_3243e_row6_col9" class="data row6 col9" >0.106153</td>
          <td id="T_3243e_row6_col10" class="data row6 col10" >0.000000</td>
          <td id="T_3243e_row6_col11" class="data row6 col11" >0.000001</td>
          <td id="T_3243e_row6_col12" class="data row6 col12" >20,24,27</td>
        </tr>
        <tr>
          <th id="T_3243e_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_3243e_row7_col0" class="data row7 col0" >TMD_C_JMD_C-Pattern(N,4,8,12,15)-KANM800103</td>
          <td id="T_3243e_row7_col1" class="data row7 col1" >Conformation</td>
          <td id="T_3243e_row7_col2" class="data row7 col2" >α-helix</td>
          <td id="T_3243e_row7_col3" class="data row7 col3" >α-helix</td>
          <td id="T_3243e_row7_col4" class="data row7 col4" >Average relative probability of inner helix (Kanehisa-Tsong, 1980)</td>
          <td id="T_3243e_row7_col5" class="data row7 col5" >0.338000</td>
          <td id="T_3243e_row7_col6" class="data row7 col6" >0.145650</td>
          <td id="T_3243e_row7_col7" class="data row7 col7" >0.145650</td>
          <td id="T_3243e_row7_col8" class="data row7 col8" >0.094896</td>
          <td id="T_3243e_row7_col9" class="data row7 col9" >0.109870</td>
          <td id="T_3243e_row7_col10" class="data row7 col10" >0.000000</td>
          <td id="T_3243e_row7_col11" class="data row7 col11" >0.000001</td>
          <td id="T_3243e_row7_col12" class="data row7 col12" >24,28,32,35</td>
        </tr>
        <tr>
          <th id="T_3243e_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_3243e_row8_col0" class="data row8 col0" >TMD_C_JMD_C-Pattern(C,4,8)-CHOC760103</td>
          <td id="T_3243e_row8_col1" class="data row8 col1" >ASA/Volume</td>
          <td id="T_3243e_row8_col2" class="data row8 col2" >Buried</td>
          <td id="T_3243e_row8_col3" class="data row8 col3" >Buried</td>
          <td id="T_3243e_row8_col4" class="data row8 col4" >Proportion of residues 95% buried (Chothia, 1976)</td>
          <td id="T_3243e_row8_col5" class="data row8 col5" >0.337000</td>
          <td id="T_3243e_row8_col6" class="data row8 col6" >0.267280</td>
          <td id="T_3243e_row8_col7" class="data row8 col7" >-0.267280</td>
          <td id="T_3243e_row8_col8" class="data row8 col8" >0.133790</td>
          <td id="T_3243e_row8_col9" class="data row8 col9" >0.229053</td>
          <td id="T_3243e_row8_col10" class="data row8 col10" >0.000000</td>
          <td id="T_3243e_row8_col11" class="data row8 col11" >0.000001</td>
          <td id="T_3243e_row8_col12" class="data row8 col12" >33,37</td>
        </tr>
        <tr>
          <th id="T_3243e_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_3243e_row9_col0" class="data row9 col0" >TMD_C_JMD_C-Segment(2,2)-ZIMJ680104</td>
          <td id="T_3243e_row9_col1" class="data row9 col1" >Energy</td>
          <td id="T_3243e_row9_col2" class="data row9 col2" >Isoelectric point</td>
          <td id="T_3243e_row9_col3" class="data row9 col3" >Isoelectric point</td>
          <td id="T_3243e_row9_col4" class="data row9 col4" >Isoelectric point (Zimmerman et al., 1968)</td>
          <td id="T_3243e_row9_col5" class="data row9 col5" >0.337000</td>
          <td id="T_3243e_row9_col6" class="data row9 col6" >0.106262</td>
          <td id="T_3243e_row9_col7" class="data row9 col7" >0.106262</td>
          <td id="T_3243e_row9_col8" class="data row9 col8" >0.070618</td>
          <td id="T_3243e_row9_col9" class="data row9 col9" >0.082016</td>
          <td id="T_3243e_row9_col10" class="data row9 col10" >0.000000</td>
          <td id="T_3243e_row9_col11" class="data row9 col11" >0.000001</td>
          <td id="T_3243e_row9_col12" class="data row9 col12" >31,32,33,34,35,36,37,38,39,40</td>
        </tr>
      </tbody>
    </table>



.. parsed-literal::

    Mean accuracy of 0.9



.. image:: tutorial1_quick_start_1_output_13_2.png



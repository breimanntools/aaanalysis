Slow Start with AAanalysis
==========================

Dive into the powerful capabilities of **AAanalysis**—a Python framework
dedicated to sequence-based, alignment-free protein prediction. In this
tutorial, we’ll focus on extracting interpretable features from protein
sequences using the ``AAclust`` and ``CPP`` models and how they can be
harnessed for binary classification tasks.

What You Will Learn:
--------------------

1. **Loading Sequences and Scales**: Load protein sequences and
   selections of amino acid scales.
2. **Feature Engineering**: Extract essential features using the
   ``AAclust`` and ``CPP`` models.
3. **Protein Prediction**: Make predictions using the RandomForest
   model.
4. **Explainable AI**: Interpret predictions at the group and individual
   levels by combining ``CPP`` with ``SHAP``.

.. code:: ipython3

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    import aaanalysis as aa
    aa.options["verbose"] = False

1. Loading Sequences and Scales
-------------------------------

With AAanalysis, you have access to numerous benchmark datasets for
protein sequence analysis. Using our γ-secretase substrates and
non-substrates dataset as a hands-on example, you can effortlessly
retrieve these datasets using the ``aa.load_dataset()`` function.
Furthermore, amino acid scales, predominantly from AAindex, along with
their hierarchical classification (known as ``AAontology``), are
available at your fingertips with the ``aa.load_scales()`` function.

We now load the scales dataset and a dataset of 50 γ-secretase
substrates and non-substrates:

.. code:: ipython3

    df_scales = aa.load_scales()
    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)

2. Feature Engineering
----------------------

The centerpiece of AAanalysis is the Comparative Physicochemical
Profiling (``CPP``) model, which is supported by ``AAclust`` for the
pre-selection of amino acid scales.

AAclust
~~~~~~~

Since redundancy is an essential problem for machine learning tasks, the
``AAclust`` object provides a lightweight wrapper for sklearn clustering
algorithms such as Agglomerative clustering. AAclust clusters a set of
scales and selects for each cluster the most representative scale (i.e.,
the scale closes to the cluster center). We will use AAclust to obtain a
set of 100 scales, as defined by the ``n_clusters`` parameters:

.. code:: ipython3

    from sklearn.cluster import KMeans
    
    aac = aa.AAclust(model_class=KMeans)
    X = np.array(df_scales).T
    scales = aac.fit(X, names=list(df_scales), n_clusters=100).medoid_names_ 
    df_scales = df_scales[scales]

Comparative Physicochemical Profiling (CPP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPP is a sequence-based feature engineering algorithm. It aims at
identifying a set of features most discriminant between two sets of
sequences: the test set and the reference set. Supported by the
``SequenceFeature`` object (``sf``), A CPP feature integrates:

-  **Parts**: Are combination of a target middle domain (TMD) and N- and
   C-terminal adjacent regions (JMD-N and JMD-C, respectively), obtained
   ``sf.get_df_parts``.
-  **Splits**: These Parts can be split into various continuous segments
   or discontinuous patterns, specified ``sf.get_split_kws()``.
-  **Scales**: Sets of amino acid scales.

We use SequenceFeature to obtain Parts and Splits. These together with
the Scales are used by CPP as input to identify the set of
characteristic features to discriminate between γ-secretase substrates
and non-substrates:

.. code:: ipython3

    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd"])
    split_kws = sf.get_split_kws(n_split_max=1, split_types=["Segment"])

Running the CPP algorithm creates all Part, Split, Split combinations
and filters a selected maximum of non-redundant features. As a baseline
approach, we use CPP without filtering (``max_cor=1``) to compute the
average values for the 100 selected scales over the entire TMD sequences
(corresponding to the transmembrane domain of γ-secretase substrates and
non-substrates):

.. code:: ipython3

    # Small set of CPP features (100 features are created)
    y = list(df_seq["label"])
    cpp = aa.CPP(df_scales=df_scales, df_parts=df_parts, split_kws=split_kws)
    df_feat = cpp.run(labels=y, max_cor=1) 
    aa.display_df(df=df_feat.head(10))



.. raw:: html

    <style type="text/css">
    #T_c0b87 thead th {
      background-color: white;
      color: black;
    }
    #T_c0b87 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_c0b87 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_c0b87 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_c0b87  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_c0b87 table {
      font-size: 12px;
    }
    </style>
    <table id="T_c0b87" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_c0b87_level0_col0" class="col_heading level0 col0" >feature</th>
          <th id="T_c0b87_level0_col1" class="col_heading level0 col1" >category</th>
          <th id="T_c0b87_level0_col2" class="col_heading level0 col2" >subcategory</th>
          <th id="T_c0b87_level0_col3" class="col_heading level0 col3" >scale_name</th>
          <th id="T_c0b87_level0_col4" class="col_heading level0 col4" >scale_description</th>
          <th id="T_c0b87_level0_col5" class="col_heading level0 col5" >abs_auc</th>
          <th id="T_c0b87_level0_col6" class="col_heading level0 col6" >abs_mean_dif</th>
          <th id="T_c0b87_level0_col7" class="col_heading level0 col7" >mean_dif</th>
          <th id="T_c0b87_level0_col8" class="col_heading level0 col8" >std_test</th>
          <th id="T_c0b87_level0_col9" class="col_heading level0 col9" >std_ref</th>
          <th id="T_c0b87_level0_col10" class="col_heading level0 col10" >p_val_mann_whitney</th>
          <th id="T_c0b87_level0_col11" class="col_heading level0 col11" >p_val_fdr_bh</th>
          <th id="T_c0b87_level0_col12" class="col_heading level0 col12" >positions</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_c0b87_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_c0b87_row0_col0" class="data row0 col0" >TMD-Segment(1,1)-WOLR790101</td>
          <td id="T_c0b87_row0_col1" class="data row0 col1" >Polarity</td>
          <td id="T_c0b87_row0_col2" class="data row0 col2" >Hydrophobicity (surrounding)</td>
          <td id="T_c0b87_row0_col3" class="data row0 col3" >Hydration potential</td>
          <td id="T_c0b87_row0_col4" class="data row0 col4" >Hydrophobicity index (Wolfenden et al., 1979)</td>
          <td id="T_c0b87_row0_col5" class="data row0 col5" >0.246000</td>
          <td id="T_c0b87_row0_col6" class="data row0 col6" >0.032767</td>
          <td id="T_c0b87_row0_col7" class="data row0 col7" >0.032767</td>
          <td id="T_c0b87_row0_col8" class="data row0 col8" >0.028962</td>
          <td id="T_c0b87_row0_col9" class="data row0 col9" >0.037213</td>
          <td id="T_c0b87_row0_col10" class="data row0 col10" >0.000022</td>
          <td id="T_c0b87_row0_col11" class="data row0 col11" >0.002203</td>
          <td id="T_c0b87_row0_col12" class="data row0 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c0b87_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_c0b87_row1_col0" class="data row1 col0" >TMD-Segment(1,1)-ANDN920101</td>
          <td id="T_c0b87_row1_col1" class="data row1 col1" >Structure-Activity</td>
          <td id="T_c0b87_row1_col2" class="data row1 col2" >Backbone-dynamics (-CH)</td>
          <td id="T_c0b87_row1_col3" class="data row1 col3" >α-CH chemical shifts (backbone-dynamics)</td>
          <td id="T_c0b87_row1_col4" class="data row1 col4" >alpha-CH chemical shifts (Andersen et al., 1992)</td>
          <td id="T_c0b87_row1_col5" class="data row1 col5" >0.230000</td>
          <td id="T_c0b87_row1_col6" class="data row1 col6" >0.064845</td>
          <td id="T_c0b87_row1_col7" class="data row1 col7" >-0.064845</td>
          <td id="T_c0b87_row1_col8" class="data row1 col8" >0.072291</td>
          <td id="T_c0b87_row1_col9" class="data row1 col9" >0.077702</td>
          <td id="T_c0b87_row1_col10" class="data row1 col10" >0.000076</td>
          <td id="T_c0b87_row1_col11" class="data row1 col11" >0.002528</td>
          <td id="T_c0b87_row1_col12" class="data row1 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c0b87_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_c0b87_row2_col0" class="data row2 col0" >TMD-Segment(1,1)-KOEH090103</td>
          <td id="T_c0b87_row2_col1" class="data row2 col1" >Polarity</td>
          <td id="T_c0b87_row2_col2" class="data row2 col2" >Hydrophilicity</td>
          <td id="T_c0b87_row2_col3" class="data row2 col3" >Polarity (hydrophilicity)</td>
          <td id="T_c0b87_row2_col4" class="data row2 col4" >Hydrophobicity scale (Eisenberg-Weiss, 1984), Inverted to match the direction of transfer from water to bilyer by Koehler et al. (2009)</td>
          <td id="T_c0b87_row2_col5" class="data row2 col5" >0.227000</td>
          <td id="T_c0b87_row2_col6" class="data row2 col6" >0.023491</td>
          <td id="T_c0b87_row2_col7" class="data row2 col7" >-0.023491</td>
          <td id="T_c0b87_row2_col8" class="data row2 col8" >0.024560</td>
          <td id="T_c0b87_row2_col9" class="data row2 col9" >0.032585</td>
          <td id="T_c0b87_row2_col10" class="data row2 col10" >0.000094</td>
          <td id="T_c0b87_row2_col11" class="data row2 col11" >0.002528</td>
          <td id="T_c0b87_row2_col12" class="data row2 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c0b87_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_c0b87_row3_col0" class="data row3 col0" >TMD-Segment(1,1)-LINS030104</td>
          <td id="T_c0b87_row3_col1" class="data row3 col1" >ASA/Volume</td>
          <td id="T_c0b87_row3_col2" class="data row3 col2" >Accessible surface area (ASA)</td>
          <td id="T_c0b87_row3_col3" class="data row3 col3" >ASA (folded protein)</td>
          <td id="T_c0b87_row3_col4" class="data row3 col4" >Total median accessible surfaces of whole residues from folded proteins, expressed in Å²  (Lins et al., 2003)</td>
          <td id="T_c0b87_row3_col5" class="data row3 col5" >0.223000</td>
          <td id="T_c0b87_row3_col6" class="data row3 col6" >0.025527</td>
          <td id="T_c0b87_row3_col7" class="data row3 col7" >-0.025527</td>
          <td id="T_c0b87_row3_col8" class="data row3 col8" >0.024048</td>
          <td id="T_c0b87_row3_col9" class="data row3 col9" >0.033495</td>
          <td id="T_c0b87_row3_col10" class="data row3 col10" >0.000120</td>
          <td id="T_c0b87_row3_col11" class="data row3 col11" >0.002528</td>
          <td id="T_c0b87_row3_col12" class="data row3 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c0b87_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_c0b87_row4_col0" class="data row4 col0" >TMD-Segment(1,1)-LINS030106</td>
          <td id="T_c0b87_row4_col1" class="data row4 col1" >ASA/Volume</td>
          <td id="T_c0b87_row4_col2" class="data row4 col2" >Accessible surface area (ASA)</td>
          <td id="T_c0b87_row4_col3" class="data row4 col3" >Hydrophilic ASA (folded proteins)</td>
          <td id="T_c0b87_row4_col4" class="data row4 col4" >Hydrophilic median accessible surfaces of whole residues from folded proteins, expressed in Å²  (Lins et al., 2003)</td>
          <td id="T_c0b87_row4_col5" class="data row4 col5" >0.222000</td>
          <td id="T_c0b87_row4_col6" class="data row4 col6" >0.024464</td>
          <td id="T_c0b87_row4_col7" class="data row4 col7" >-0.024464</td>
          <td id="T_c0b87_row4_col8" class="data row4 col8" >0.023645</td>
          <td id="T_c0b87_row4_col9" class="data row4 col9" >0.032067</td>
          <td id="T_c0b87_row4_col10" class="data row4 col10" >0.000136</td>
          <td id="T_c0b87_row4_col11" class="data row4 col11" >0.002528</td>
          <td id="T_c0b87_row4_col12" class="data row4 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c0b87_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_c0b87_row5_col0" class="data row5 col0" >TMD-Segment(1,1)-CHOC760104</td>
          <td id="T_c0b87_row5_col1" class="data row5 col1" >ASA/Volume</td>
          <td id="T_c0b87_row5_col2" class="data row5 col2" >Buried</td>
          <td id="T_c0b87_row5_col3" class="data row5 col3" >Buried</td>
          <td id="T_c0b87_row5_col4" class="data row5 col4" >Proportion of residues 100% buried (Chothia, 1976)</td>
          <td id="T_c0b87_row5_col5" class="data row5 col5" >0.220000</td>
          <td id="T_c0b87_row5_col6" class="data row5 col6" >0.046652</td>
          <td id="T_c0b87_row5_col7" class="data row5 col7" >0.046652</td>
          <td id="T_c0b87_row5_col8" class="data row5 col8" >0.046731</td>
          <td id="T_c0b87_row5_col9" class="data row5 col9" >0.062379</td>
          <td id="T_c0b87_row5_col10" class="data row5 col10" >0.000152</td>
          <td id="T_c0b87_row5_col11" class="data row5 col11" >0.002528</td>
          <td id="T_c0b87_row5_col12" class="data row5 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c0b87_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_c0b87_row6_col0" class="data row6 col0" >TMD-Segment(1,1)-JURD980101</td>
          <td id="T_c0b87_row6_col1" class="data row6 col1" >Polarity</td>
          <td id="T_c0b87_row6_col2" class="data row6 col2" >Hydrophobicity</td>
          <td id="T_c0b87_row6_col3" class="data row6 col3" >Hydrophobicity</td>
          <td id="T_c0b87_row6_col4" class="data row6 col4" >Modified Kyte-Doolittle hydrophobicity scale (Juretic et al., 1998)</td>
          <td id="T_c0b87_row6_col5" class="data row6 col5" >0.209000</td>
          <td id="T_c0b87_row6_col6" class="data row6 col6" >0.038451</td>
          <td id="T_c0b87_row6_col7" class="data row6 col7" >0.038451</td>
          <td id="T_c0b87_row6_col8" class="data row6 col8" >0.044812</td>
          <td id="T_c0b87_row6_col9" class="data row6 col9" >0.055053</td>
          <td id="T_c0b87_row6_col10" class="data row6 col10" >0.000324</td>
          <td id="T_c0b87_row6_col11" class="data row6 col11" >0.004388</td>
          <td id="T_c0b87_row6_col12" class="data row6 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c0b87_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_c0b87_row7_col0" class="data row7 col0" >TMD-Segment(1,1)-FUKS010106</td>
          <td id="T_c0b87_row7_col1" class="data row7 col1" >Composition</td>
          <td id="T_c0b87_row7_col2" class="data row7 col2" >Membrane proteins (MPs)</td>
          <td id="T_c0b87_row7_col3" class="data row7 col3" >Proteins of mesophiles (INT)</td>
          <td id="T_c0b87_row7_col4" class="data row7 col4" >Interior composition of amino acids in intracellular proteins of mesophiles [%] (Fukuchi-Nishikawa, 2001)</td>
          <td id="T_c0b87_row7_col5" class="data row7 col5" >0.206000</td>
          <td id="T_c0b87_row7_col6" class="data row7 col6" >0.058909</td>
          <td id="T_c0b87_row7_col7" class="data row7 col7" >0.058909</td>
          <td id="T_c0b87_row7_col8" class="data row7 col8" >0.068070</td>
          <td id="T_c0b87_row7_col9" class="data row7 col9" >0.081967</td>
          <td id="T_c0b87_row7_col10" class="data row7 col10" >0.000380</td>
          <td id="T_c0b87_row7_col11" class="data row7 col11" >0.004388</td>
          <td id="T_c0b87_row7_col12" class="data row7 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c0b87_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_c0b87_row8_col0" class="data row8 col0" >TMD-Segment(1,1)-KOEH090102</td>
          <td id="T_c0b87_row8_col1" class="data row8 col1" >Polarity</td>
          <td id="T_c0b87_row8_col2" class="data row8 col2" >Hydrophilicity</td>
          <td id="T_c0b87_row8_col3" class="data row8 col3" >Polarity (hydrophilicity)</td>
          <td id="T_c0b87_row8_col4" class="data row8 col4" >Hydrophobicity scale (Hessa et al., 2007), Cited by Koehler et al. (2009)</td>
          <td id="T_c0b87_row8_col5" class="data row8 col5" >0.206000</td>
          <td id="T_c0b87_row8_col6" class="data row8 col6" >0.025848</td>
          <td id="T_c0b87_row8_col7" class="data row8 col7" >-0.025848</td>
          <td id="T_c0b87_row8_col8" class="data row8 col8" >0.028664</td>
          <td id="T_c0b87_row8_col9" class="data row8 col9" >0.039067</td>
          <td id="T_c0b87_row8_col10" class="data row8 col10" >0.000395</td>
          <td id="T_c0b87_row8_col11" class="data row8 col11" >0.004388</td>
          <td id="T_c0b87_row8_col12" class="data row8 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
        <tr>
          <th id="T_c0b87_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_c0b87_row9_col0" class="data row9 col0" >TMD-Segment(1,1)-VELV850101</td>
          <td id="T_c0b87_row9_col1" class="data row9 col1" >Energy</td>
          <td id="T_c0b87_row9_col2" class="data row9 col2" >Electron-ion interaction pot.</td>
          <td id="T_c0b87_row9_col3" class="data row9 col3" >Electron-ion interaction potential</td>
          <td id="T_c0b87_row9_col4" class="data row9 col4" >Electron-ion interaction potential (Veljkovic et al., 1985)</td>
          <td id="T_c0b87_row9_col5" class="data row9 col5" >0.203000</td>
          <td id="T_c0b87_row9_col6" class="data row9 col6" >0.045725</td>
          <td id="T_c0b87_row9_col7" class="data row9 col7" >-0.045725</td>
          <td id="T_c0b87_row9_col8" class="data row9 col8" >0.059791</td>
          <td id="T_c0b87_row9_col9" class="data row9 col9" >0.068804</td>
          <td id="T_c0b87_row9_col10" class="data row9 col10" >0.000480</td>
          <td id="T_c0b87_row9_col11" class="data row9 col11" >0.004687</td>
          <td id="T_c0b87_row9_col12" class="data row9 col12" >11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30</td>
        </tr>
      </tbody>
    </table>



3. Protein Prediction
---------------------

A feature matrix from a given set of CPP features can be created using
``sf.feat_matrix`` and used for machine learning:

.. code:: ipython3

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    X = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"])
    rf = RandomForestClassifier()
    cv_base = cross_val_score(rf, X, y, scoring="accuracy", cv=5)
    print(f"Mean accuracy of {round(np.mean(cv_base), 2)}")


.. parsed-literal::

    Mean accuracy of 0.56


CPP uses 330 Splits and 3 Parts by default, yielding around 100.000
features for 100 Scales. Creating and filtering all these
Part-Split-Scale combinations will take a little time (~1.5 minutes for
this example on an i7-10510U (4 cores, 8 threads) with multiprocessing),
but improve prediction performance:

.. code:: ipython3

    # CPP features with default splits and parts (around 100.000 features)
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
    #T_ce980 thead th {
      background-color: white;
      color: black;
    }
    #T_ce980 tbody tr:nth-child(odd) {
      background-color: #f2f2f2;
    }
    #T_ce980 tbody tr:nth-child(even) {
      background-color: white;
    }
    #T_ce980 th {
      padding: 5px;
      white-space: nowrap;
    }
    #T_ce980  td {
      padding: 5px;
      white-space: nowrap;
    }
    #T_ce980 table {
      font-size: 12px;
    }
    </style>
    <table id="T_ce980" style='display:block; max-height: 300px; max-width: 100%; overflow-x: auto; overflow-y: auto;'>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_ce980_level0_col0" class="col_heading level0 col0" >feature</th>
          <th id="T_ce980_level0_col1" class="col_heading level0 col1" >category</th>
          <th id="T_ce980_level0_col2" class="col_heading level0 col2" >subcategory</th>
          <th id="T_ce980_level0_col3" class="col_heading level0 col3" >scale_name</th>
          <th id="T_ce980_level0_col4" class="col_heading level0 col4" >scale_description</th>
          <th id="T_ce980_level0_col5" class="col_heading level0 col5" >abs_auc</th>
          <th id="T_ce980_level0_col6" class="col_heading level0 col6" >abs_mean_dif</th>
          <th id="T_ce980_level0_col7" class="col_heading level0 col7" >mean_dif</th>
          <th id="T_ce980_level0_col8" class="col_heading level0 col8" >std_test</th>
          <th id="T_ce980_level0_col9" class="col_heading level0 col9" >std_ref</th>
          <th id="T_ce980_level0_col10" class="col_heading level0 col10" >p_val_mann_whitney</th>
          <th id="T_ce980_level0_col11" class="col_heading level0 col11" >p_val_fdr_bh</th>
          <th id="T_ce980_level0_col12" class="col_heading level0 col12" >positions</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_ce980_level0_row0" class="row_heading level0 row0" >1</th>
          <td id="T_ce980_row0_col0" class="data row0 col0" >TMD_C_JMD_C-Segment(2,3)-QIAN880106</td>
          <td id="T_ce980_row0_col1" class="data row0 col1" >Conformation</td>
          <td id="T_ce980_row0_col2" class="data row0 col2" >α-helix</td>
          <td id="T_ce980_row0_col3" class="data row0 col3" >α-helix (middle)</td>
          <td id="T_ce980_row0_col4" class="data row0 col4" >Weights for alpha-helix at the window position of -1 (Qian-Sejnowski, 1988)</td>
          <td id="T_ce980_row0_col5" class="data row0 col5" >0.387000</td>
          <td id="T_ce980_row0_col6" class="data row0 col6" >0.121446</td>
          <td id="T_ce980_row0_col7" class="data row0 col7" >0.121446</td>
          <td id="T_ce980_row0_col8" class="data row0 col8" >0.069196</td>
          <td id="T_ce980_row0_col9" class="data row0 col9" >0.085013</td>
          <td id="T_ce980_row0_col10" class="data row0 col10" >0.000000</td>
          <td id="T_ce980_row0_col11" class="data row0 col11" >0.000000</td>
          <td id="T_ce980_row0_col12" class="data row0 col12" >27,28,29,30,31,32,33</td>
        </tr>
        <tr>
          <th id="T_ce980_level0_row1" class="row_heading level0 row1" >2</th>
          <td id="T_ce980_row1_col0" class="data row1 col0" >TMD_C_JMD_C-Segment(5,7)-ONEK900101</td>
          <td id="T_ce980_row1_col1" class="data row1 col1" >Others</td>
          <td id="T_ce980_row1_col2" class="data row1 col2" >Unclassified (Others)</td>
          <td id="T_ce980_row1_col3" class="data row1 col3" >ΔG values in peptides</td>
          <td id="T_ce980_row1_col4" class="data row1 col4" >Delta G values for the peptides extrapolated to 0 M urea (O'Neil-DeGrado, 1990)</td>
          <td id="T_ce980_row1_col5" class="data row1 col5" >0.373000</td>
          <td id="T_ce980_row1_col6" class="data row1 col6" >0.115273</td>
          <td id="T_ce980_row1_col7" class="data row1 col7" >0.115273</td>
          <td id="T_ce980_row1_col8" class="data row1 col8" >0.065969</td>
          <td id="T_ce980_row1_col9" class="data row1 col9" >0.112756</td>
          <td id="T_ce980_row1_col10" class="data row1 col10" >0.000000</td>
          <td id="T_ce980_row1_col11" class="data row1 col11" >0.000000</td>
          <td id="T_ce980_row1_col12" class="data row1 col12" >32,33,34</td>
        </tr>
        <tr>
          <th id="T_ce980_level0_row2" class="row_heading level0 row2" >3</th>
          <td id="T_ce980_row2_col0" class="data row2 col0" >TMD_C_JMD_C-Segment(5,7)-LEVM760105</td>
          <td id="T_ce980_row2_col1" class="data row2 col1" >Shape</td>
          <td id="T_ce980_row2_col2" class="data row2 col2" >Side chain length</td>
          <td id="T_ce980_row2_col3" class="data row2 col3" >Side chain length</td>
          <td id="T_ce980_row2_col4" class="data row2 col4" >Radius of gyration of side chain (Levitt, 1976)</td>
          <td id="T_ce980_row2_col5" class="data row2 col5" >0.360000</td>
          <td id="T_ce980_row2_col6" class="data row2 col6" >0.250473</td>
          <td id="T_ce980_row2_col7" class="data row2 col7" >0.250473</td>
          <td id="T_ce980_row2_col8" class="data row2 col8" >0.148804</td>
          <td id="T_ce980_row2_col9" class="data row2 col9" >0.174650</td>
          <td id="T_ce980_row2_col10" class="data row2 col10" >0.000000</td>
          <td id="T_ce980_row2_col11" class="data row2 col11" >0.000000</td>
          <td id="T_ce980_row2_col12" class="data row2 col12" >32,33,34</td>
        </tr>
        <tr>
          <th id="T_ce980_level0_row3" class="row_heading level0 row3" >4</th>
          <td id="T_ce980_row3_col0" class="data row3 col0" >TMD_C_JMD_C-Pattern(N,5,8,12,15)-QIAN880106</td>
          <td id="T_ce980_row3_col1" class="data row3 col1" >Conformation</td>
          <td id="T_ce980_row3_col2" class="data row3 col2" >α-helix</td>
          <td id="T_ce980_row3_col3" class="data row3 col3" >α-helix (middle)</td>
          <td id="T_ce980_row3_col4" class="data row3 col4" >Weights for alpha-helix at the window position of -1 (Qian-Sejnowski, 1988)</td>
          <td id="T_ce980_row3_col5" class="data row3 col5" >0.358000</td>
          <td id="T_ce980_row3_col6" class="data row3 col6" >0.144860</td>
          <td id="T_ce980_row3_col7" class="data row3 col7" >0.144860</td>
          <td id="T_ce980_row3_col8" class="data row3 col8" >0.079321</td>
          <td id="T_ce980_row3_col9" class="data row3 col9" >0.117515</td>
          <td id="T_ce980_row3_col10" class="data row3 col10" >0.000000</td>
          <td id="T_ce980_row3_col11" class="data row3 col11" >0.000000</td>
          <td id="T_ce980_row3_col12" class="data row3 col12" >25,28,32,35</td>
        </tr>
        <tr>
          <th id="T_ce980_level0_row4" class="row_heading level0 row4" >5</th>
          <td id="T_ce980_row4_col0" class="data row4 col0" >TMD_C_JMD_C-Segment(4,5)-KLEP840101</td>
          <td id="T_ce980_row4_col1" class="data row4 col1" >Energy</td>
          <td id="T_ce980_row4_col2" class="data row4 col2" >Charge</td>
          <td id="T_ce980_row4_col3" class="data row4 col3" >Charge</td>
          <td id="T_ce980_row4_col4" class="data row4 col4" >Net charge (Klein et al., 1984)</td>
          <td id="T_ce980_row4_col5" class="data row4 col5" >0.354000</td>
          <td id="T_ce980_row4_col6" class="data row4 col6" >0.192500</td>
          <td id="T_ce980_row4_col7" class="data row4 col7" >0.192500</td>
          <td id="T_ce980_row4_col8" class="data row4 col8" >0.111915</td>
          <td id="T_ce980_row4_col9" class="data row4 col9" >0.127009</td>
          <td id="T_ce980_row4_col10" class="data row4 col10" >0.000000</td>
          <td id="T_ce980_row4_col11" class="data row4 col11" >0.000000</td>
          <td id="T_ce980_row4_col12" class="data row4 col12" >33,34,35,36</td>
        </tr>
        <tr>
          <th id="T_ce980_level0_row5" class="row_heading level0 row5" >6</th>
          <td id="T_ce980_row5_col0" class="data row5 col0" >TMD_C_JMD_C-Pattern(N,4,8,12,15)-MUNV940102</td>
          <td id="T_ce980_row5_col1" class="data row5 col1" >Energy</td>
          <td id="T_ce980_row5_col2" class="data row5 col2" >Free energy (folding)</td>
          <td id="T_ce980_row5_col3" class="data row5 col3" >Free energy (α-helix)</td>
          <td id="T_ce980_row5_col4" class="data row5 col4" >Free energy in alpha-helical region (Munoz-Serrano, 1994)</td>
          <td id="T_ce980_row5_col5" class="data row5 col5" >0.353000</td>
          <td id="T_ce980_row5_col6" class="data row5 col6" >0.119820</td>
          <td id="T_ce980_row5_col7" class="data row5 col7" >-0.119820</td>
          <td id="T_ce980_row5_col8" class="data row5 col8" >0.065320</td>
          <td id="T_ce980_row5_col9" class="data row5 col9" >0.098536</td>
          <td id="T_ce980_row5_col10" class="data row5 col10" >0.000000</td>
          <td id="T_ce980_row5_col11" class="data row5 col11" >0.000001</td>
          <td id="T_ce980_row5_col12" class="data row5 col12" >24,28,32,35</td>
        </tr>
        <tr>
          <th id="T_ce980_level0_row6" class="row_heading level0 row6" >7</th>
          <td id="T_ce980_row6_col0" class="data row6 col0" >TMD_C_JMD_C-Segment(5,7)-HUTJ700102</td>
          <td id="T_ce980_row6_col1" class="data row6 col1" >Energy</td>
          <td id="T_ce980_row6_col2" class="data row6 col2" >Entropy</td>
          <td id="T_ce980_row6_col3" class="data row6 col3" >Entropy</td>
          <td id="T_ce980_row6_col4" class="data row6 col4" >Absolute entropy (Hutchens, 1970)</td>
          <td id="T_ce980_row6_col5" class="data row6 col5" >0.343000</td>
          <td id="T_ce980_row6_col6" class="data row6 col6" >0.229253</td>
          <td id="T_ce980_row6_col7" class="data row6 col7" >0.229253</td>
          <td id="T_ce980_row6_col8" class="data row6 col8" >0.143391</td>
          <td id="T_ce980_row6_col9" class="data row6 col9" >0.171184</td>
          <td id="T_ce980_row6_col10" class="data row6 col10" >0.000000</td>
          <td id="T_ce980_row6_col11" class="data row6 col11" >0.000001</td>
          <td id="T_ce980_row6_col12" class="data row6 col12" >32,33,34</td>
        </tr>
        <tr>
          <th id="T_ce980_level0_row7" class="row_heading level0 row7" >8</th>
          <td id="T_ce980_row7_col0" class="data row7 col0" >TMD-Segment(12,13)-ROBB760113</td>
          <td id="T_ce980_row7_col1" class="data row7 col1" >Conformation</td>
          <td id="T_ce980_row7_col2" class="data row7 col2" >β-turn</td>
          <td id="T_ce980_row7_col3" class="data row7 col3" >β-turn</td>
          <td id="T_ce980_row7_col4" class="data row7 col4" >Information measure for loop (Robson-Suzuki, 1976)</td>
          <td id="T_ce980_row7_col5" class="data row7 col5" >0.337000</td>
          <td id="T_ce980_row7_col6" class="data row7 col6" >0.319440</td>
          <td id="T_ce980_row7_col7" class="data row7 col7" >-0.319440</td>
          <td id="T_ce980_row7_col8" class="data row7 col8" >0.175203</td>
          <td id="T_ce980_row7_col9" class="data row7 col9" >0.255754</td>
          <td id="T_ce980_row7_col10" class="data row7 col10" >0.000000</td>
          <td id="T_ce980_row7_col11" class="data row7 col11" >0.000001</td>
          <td id="T_ce980_row7_col12" class="data row7 col12" >27,28</td>
        </tr>
        <tr>
          <th id="T_ce980_level0_row8" class="row_heading level0 row8" >9</th>
          <td id="T_ce980_row8_col0" class="data row8 col0" >TMD_C_JMD_C-Segment(4,5)-RICJ880113</td>
          <td id="T_ce980_row8_col1" class="data row8 col1" >Conformation</td>
          <td id="T_ce980_row8_col2" class="data row8 col2" >α-helix (C-cap)</td>
          <td id="T_ce980_row8_col3" class="data row8 col3" >α-helix (C-terminal, inside)</td>
          <td id="T_ce980_row8_col4" class="data row8 col4" >Relative preference value at C2 (Richardson-Richardson, 1988)</td>
          <td id="T_ce980_row8_col5" class="data row8 col5" >0.336000</td>
          <td id="T_ce980_row8_col6" class="data row8 col6" >0.223765</td>
          <td id="T_ce980_row8_col7" class="data row8 col7" >0.223765</td>
          <td id="T_ce980_row8_col8" class="data row8 col8" >0.133513</td>
          <td id="T_ce980_row8_col9" class="data row8 col9" >0.178217</td>
          <td id="T_ce980_row8_col10" class="data row8 col10" >0.000000</td>
          <td id="T_ce980_row8_col11" class="data row8 col11" >0.000001</td>
          <td id="T_ce980_row8_col12" class="data row8 col12" >33,34,35,36</td>
        </tr>
        <tr>
          <th id="T_ce980_level0_row9" class="row_heading level0 row9" >10</th>
          <td id="T_ce980_row9_col0" class="data row9 col0" >TMD_C_JMD_C-Pattern(C,4,8)-JURD980101</td>
          <td id="T_ce980_row9_col1" class="data row9 col1" >Polarity</td>
          <td id="T_ce980_row9_col2" class="data row9 col2" >Hydrophobicity</td>
          <td id="T_ce980_row9_col3" class="data row9 col3" >Hydrophobicity</td>
          <td id="T_ce980_row9_col4" class="data row9 col4" >Modified Kyte-Doolittle hydrophobicity scale (Juretic et al., 1998)</td>
          <td id="T_ce980_row9_col5" class="data row9 col5" >0.329000</td>
          <td id="T_ce980_row9_col6" class="data row9 col6" >0.264720</td>
          <td id="T_ce980_row9_col7" class="data row9 col7" >-0.264720</td>
          <td id="T_ce980_row9_col8" class="data row9 col8" >0.141666</td>
          <td id="T_ce980_row9_col9" class="data row9 col9" >0.233134</td>
          <td id="T_ce980_row9_col10" class="data row9 col10" >0.000000</td>
          <td id="T_ce980_row9_col11" class="data row9 col11" >0.000001</td>
          <td id="T_ce980_row9_col12" class="data row9 col12" >33,37</td>
        </tr>
      </tbody>
    </table>



.. parsed-literal::

    Mean accuracy of 0.9



.. image:: NOTEBOOK_1_output_13_2.png


4. Explainable AI
-----------------

Explainable AI on group level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explainable AI on individual level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

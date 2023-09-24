Quick Start with AAanalysis
===========================

Dive into the powerful capabilities of ``AAanalysis``—a Python framework
dedicated to sequence-based, alignment-free protein prediction. In this
tutorial, using gamma-secretase substrates and non-substrates as an
example, we’ll focus on extracting interpretable features from protein
sequences using the ``AAclust`` and ``CPP`` models and how they can be
harnessed for binary classification tasks.

What You Will Learn:
--------------------

-  ``Loading Sequences and Scales``: How to easily load protein
   sequences and their amino acid scales.
-  ``Feature Engineering``: Extract essential features using the
   ``AAclust`` and ``CPP`` models.
-  ``Protein Prediction``: Make predictions using the RandomForest
   model.
-  ``Explainable AI``: Interpret predictions at the group and individual
   levels by combining ``CPP`` with ``SHAP``.

1. Loading Sequences and Scales
-------------------------------

With AAanalysis, you have access to numerous benchmark datasets for
protein sequence analysis. Using our γ-secretase substrates and
non-substrates dataset as a hands-on example, you can effortlessly
retrieve these datasets using the ``aa.load_dataset()`` function.
Furthermore, amino acid scales, predominantly from AAindex, along with
their hierarchical classification (known as ``AAontology``), are
available at your fingertips with the ``aa.load_scales()`` function.

.. code:: ipython3

    import aaanalysis as aa
    
    df_scales = aa.load_scales()
    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
    df_seq.head(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>entry</th>
          <th>sequence</th>
          <th>label</th>
          <th>tmd_start</th>
          <th>tmd_stop</th>
          <th>jmd_n</th>
          <th>tmd</th>
          <th>jmd_c</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Q14802</td>
          <td>MQKVTLGLLVFLAGFPVLDANDLEDKNSPFYYDWHSLQVGGLICAG...</td>
          <td>0</td>
          <td>37</td>
          <td>59</td>
          <td>NSPFYYDWHS</td>
          <td>LQVGGLICAGVLCAMGIIIVMSA</td>
          <td>KCKCKFGQKS</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Q86UE4</td>
          <td>MAARSWQDELAQQAEEGSARLREMLSVGLGFLRTELGLDLGLEPKR...</td>
          <td>0</td>
          <td>50</td>
          <td>72</td>
          <td>LGLEPKRYPG</td>
          <td>WVILVGTGALGLLLLFLLGYGWA</td>
          <td>AACAGARKKR</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Q969W9</td>
          <td>MHRLMGVNSTAAAAAGQPNVSCTCNCKRSLFQSMEITELEFVQIII...</td>
          <td>0</td>
          <td>41</td>
          <td>63</td>
          <td>FQSMEITELE</td>
          <td>FVQIIIIVVVMMVMVVVITCLLS</td>
          <td>HYKLSARSFI</td>
        </tr>
        <tr>
          <th>3</th>
          <td>P53801</td>
          <td>MAPGVARGPTPYWRLRLGGAALLLLLIPVAAAQEPPGAACSQNTNK...</td>
          <td>0</td>
          <td>97</td>
          <td>119</td>
          <td>RWGVCWVNFE</td>
          <td>ALIITMSVVGGTLLLGIAICCCC</td>
          <td>CCRRKRSRKP</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Q8IUW5</td>
          <td>MAPRALPGSAVLAAAVFVGGAVSSPLVAPDNGSSRTLHSRTETTPS...</td>
          <td>0</td>
          <td>59</td>
          <td>81</td>
          <td>NDTGNGHPEY</td>
          <td>IAYALVPVFFIMGLFGVLICHLL</td>
          <td>KKKGYRCTTE</td>
        </tr>
      </tbody>
    </table>
    </div>



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

    from sklearn.cluster import AgglomerativeClustering
    import numpy as np
    
    aac = aa.AAclust(model=AgglomerativeClustering)
    X = np.array(df_scales)
    scales = aac.fit(X, names=list(df_scales), n_clusters=100) 
    df_scales = df_scales[scales]
    df_scales[scales[0:4]].head(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ANDN920101</th>
          <th>SIMZ760101</th>
          <th>NAKH900106</th>
          <th>AURR980112</th>
        </tr>
        <tr>
          <th>AA</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>A</th>
          <td>0.494</td>
          <td>0.268</td>
          <td>0.237</td>
          <td>0.787</td>
        </tr>
        <tr>
          <th>C</th>
          <td>0.864</td>
          <td>0.258</td>
          <td>0.303</td>
          <td>0.104</td>
        </tr>
        <tr>
          <th>D</th>
          <td>1.000</td>
          <td>0.206</td>
          <td>0.000</td>
          <td>0.451</td>
        </tr>
        <tr>
          <th>E</th>
          <td>0.420</td>
          <td>0.210</td>
          <td>0.090</td>
          <td>0.823</td>
        </tr>
        <tr>
          <th>F</th>
          <td>0.877</td>
          <td>0.887</td>
          <td>0.724</td>
          <td>0.402</td>
        </tr>
      </tbody>
    </table>
    </div>



Comparative Physicochemical Profiling (CPP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPP is a sequence-based feature engineering algorithm. It aims at
identifying a set of features most discriminant between two sets of
sequences: the test set and the reference set. Supported by the
``SequenceFeature`` object (``sf``), A CPP feature integrates: -
``Parts``: Are combination of a target middle domain (TMD) and N- and
C-terminal adjacent regions (JMD-N and JMD-C, respectively), obtained
``sf.get_df_parts``. - ``Splits``: These ``Parts`` can be split into
various continuous segments or discontinuous patterns, specified
``sf.get_split_kws()``. - ``Scales``: Sets of amino acid scales.

We use SequenceFeature to obtain Parts and Splits:

.. code:: ipython3

    y = list(df_seq["label"])
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd_jmd"])
    split_kws = sf.get_split_kws(n_split_max=1, split_types=["Segment"])
    df_parts.head(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>tmd_jmd</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>D3ZZK3</th>
          <td>RIIGDGANSTVLLVSVSGSVVLVVILIAAFVISRRRSKYSQAK</td>
        </tr>
        <tr>
          <th>O14786</th>
          <td>PGNVLKTLDPILITIIAMSALGVLLGAVCGVVLYCACWHNGMS</td>
        </tr>
        <tr>
          <th>O35516</th>
          <td>SELESPRNAQLLYLLAVAVVIILFFILLGVIMAKRKRKHGFLW</td>
        </tr>
        <tr>
          <th>O43914</th>
          <td>DCSCSTVSPGVLAGIVMGDLVLTVLIALAVYFLGRLVPRGRGA</td>
        </tr>
        <tr>
          <th>O75581</th>
          <td>YPTEEPAPQATNTVGSVIGVIVTIFVSGTVYFICQRMLCPRMK</td>
        </tr>
      </tbody>
    </table>
    </div>



Running the CPP algorithm creates all ``Part``, ``Split``, ``Split``
combinations and filters a selected maximum of non-redundant features.
As a baseline approach, we use CPP to compute the average values for the
100 selected scales over the entire TMD-JMD sequences:

.. code:: ipython3

    # Small set of CPP features (100 features are created)
    cpp = aa.CPP(df_scales=df_scales, df_parts=df_parts, split_kws=split_kws, verbose=False)
    df_feat = cpp.run(labels=y) 
    df_feat




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>feature</th>
          <th>category</th>
          <th>subcategory</th>
          <th>scale_name</th>
          <th>scale_description</th>
          <th>abs_auc</th>
          <th>abs_mean_dif</th>
          <th>mean_dif</th>
          <th>std_test</th>
          <th>std_ref</th>
          <th>p_val_mann_whitney</th>
          <th>p_val_fdr_bh</th>
          <th>positions</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>TMD_JMD-Segment(1,1)-ANDN920101</td>
          <td>Structure-Activity</td>
          <td>Backbone-dynamics (-CH)</td>
          <td>α-CH chemical shifts (backbone-dynamics)</td>
          <td>alpha-CH chemical shifts (Andersen et al., 1992)</td>
          <td>0.130</td>
          <td>0.022966</td>
          <td>0.022966</td>
          <td>0.054433</td>
          <td>0.053266</td>
          <td>0.025737</td>
          <td>0.099022</td>
          <td>1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,1...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>TMD_JMD-Segment(1,1)-VASM830101</td>
          <td>Conformation</td>
          <td>Unclassified (Conformation)</td>
          <td>α-helix</td>
          <td>Relative population of conformational state A ...</td>
          <td>0.120</td>
          <td>0.019298</td>
          <td>-0.019298</td>
          <td>0.046755</td>
          <td>0.049127</td>
          <td>0.039609</td>
          <td>0.099022</td>
          <td>1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,1...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>TMD_JMD-Segment(1,1)-ROBB760113</td>
          <td>Conformation</td>
          <td>β-turn</td>
          <td>β-turn</td>
          <td>Information measure for loop (Robson-Suzuki, 1...</td>
          <td>0.108</td>
          <td>0.021958</td>
          <td>0.021958</td>
          <td>0.060658</td>
          <td>0.053190</td>
          <td>0.062212</td>
          <td>0.100670</td>
          <td>1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,1...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>TMD_JMD-Segment(1,1)-RACS820103</td>
          <td>Conformation</td>
          <td>Unclassified (Conformation)</td>
          <td>α-helix (left-handed)</td>
          <td>Average relative fractional occurrence in AL(i...</td>
          <td>0.080</td>
          <td>0.019579</td>
          <td>-0.019579</td>
          <td>0.072260</td>
          <td>0.047452</td>
          <td>0.166907</td>
          <td>0.166907</td>
          <td>1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,1...</td>
        </tr>
      </tbody>
    </table>
    </div>



3. Protein Prediction
---------------------

A feature matrix from a given set of CPP features can be created using
``sf.feat_matrix`` and used for machine learning:

.. code:: ipython3

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    X = sf.feat_matrix(df_parts=df_parts, features=df_feat["feature"])
    rf = RandomForestClassifier()
    cv_base = cross_val_score(rf, X, y, scoring="accuracy")
    print(f"Mean accuracy of {round(np.mean(cv_base), 2)}")


.. parsed-literal::

    Mean accuracy of 0.58


Creating more features with CPP will take some more time. but improve
prediction performance:

.. code:: ipython3

    # CPP features with default splits (around 100.000 features)
    df_parts = sf.get_df_parts(df_seq=df_seq)
    cpp = aa.CPP(df_scales=df_scales, df_parts=df_parts, verbose=False)
    df_feat = cpp.run(labels=y)
    df_feat.head(10)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>feature</th>
          <th>category</th>
          <th>subcategory</th>
          <th>scale_name</th>
          <th>scale_description</th>
          <th>abs_auc</th>
          <th>abs_mean_dif</th>
          <th>mean_dif</th>
          <th>std_test</th>
          <th>std_ref</th>
          <th>p_val_mann_whitney</th>
          <th>p_val_fdr_bh</th>
          <th>positions</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>TMD_C_JMD_C-Segment(2,3)-QIAN880106</td>
          <td>Conformation</td>
          <td>α-helix</td>
          <td>α-helix (middle)</td>
          <td>Weights for alpha-helix at the window position...</td>
          <td>0.387</td>
          <td>0.121446</td>
          <td>0.121446</td>
          <td>0.069196</td>
          <td>0.085013</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>27,28,29,30,31,32,33</td>
        </tr>
        <tr>
          <th>1</th>
          <td>TMD_C_JMD_C-Segment(4,5)-ZIMJ680104</td>
          <td>Energy</td>
          <td>Isoelectric point</td>
          <td>Isoelectric point</td>
          <td>Isoelectric point (Zimmerman et al., 1968)</td>
          <td>0.373</td>
          <td>0.220000</td>
          <td>0.220000</td>
          <td>0.123716</td>
          <td>0.137350</td>
          <td>1.000000e-10</td>
          <td>2.475000e-07</td>
          <td>33,34,35,36</td>
        </tr>
        <tr>
          <th>2</th>
          <td>TMD_C_JMD_C-Pattern(N,5,8,12,15)-QIAN880106</td>
          <td>Conformation</td>
          <td>α-helix</td>
          <td>α-helix (middle)</td>
          <td>Weights for alpha-helix at the window position...</td>
          <td>0.358</td>
          <td>0.144860</td>
          <td>0.144860</td>
          <td>0.079321</td>
          <td>0.117515</td>
          <td>7.000000e-10</td>
          <td>7.150000e-07</td>
          <td>25,28,32,35</td>
        </tr>
        <tr>
          <th>3</th>
          <td>TMD_C_JMD_C-Segment(5,7)-LINS030101</td>
          <td>ASA/Volume</td>
          <td>Volume</td>
          <td>Accessible surface area (ASA)</td>
          <td>Total accessible surfaces of whole residues (b...</td>
          <td>0.354</td>
          <td>0.237161</td>
          <td>0.237161</td>
          <td>0.145884</td>
          <td>0.164285</td>
          <td>1.100000e-09</td>
          <td>7.150000e-07</td>
          <td>32,33,34</td>
        </tr>
        <tr>
          <th>4</th>
          <td>TMD_C_JMD_C-Segment(6,9)-ZIMJ680104</td>
          <td>Energy</td>
          <td>Isoelectric point</td>
          <td>Isoelectric point</td>
          <td>Isoelectric point (Zimmerman et al., 1968)</td>
          <td>0.341</td>
          <td>0.263651</td>
          <td>0.263651</td>
          <td>0.187136</td>
          <td>0.171995</td>
          <td>4.000000e-09</td>
          <td>1.185395e-06</td>
          <td>32,33</td>
        </tr>
        <tr>
          <th>5</th>
          <td>TMD_C_JMD_C-Segment(4,9)-ROBB760113</td>
          <td>Conformation</td>
          <td>β-turn</td>
          <td>β-turn</td>
          <td>Information measure for loop (Robson-Suzuki, 1...</td>
          <td>0.337</td>
          <td>0.319440</td>
          <td>-0.319440</td>
          <td>0.175203</td>
          <td>0.255754</td>
          <td>6.100000e-09</td>
          <td>1.185395e-06</td>
          <td>27,28</td>
        </tr>
        <tr>
          <th>6</th>
          <td>TMD_C_JMD_C-Segment(2,2)-EISD860102</td>
          <td>Energy</td>
          <td>Isoelectric point</td>
          <td>Atom-based hydrophobic moment</td>
          <td>Atom-based hydrophobic moment (Eisenberg-McLac...</td>
          <td>0.337</td>
          <td>0.139567</td>
          <td>0.139567</td>
          <td>0.098917</td>
          <td>0.101842</td>
          <td>6.300000e-09</td>
          <td>1.185395e-06</td>
          <td>31,32,33,34,35,36,37,38,39,40</td>
        </tr>
        <tr>
          <th>7</th>
          <td>TMD_C_JMD_C-Segment(4,5)-RICJ880113</td>
          <td>Conformation</td>
          <td>α-helix (C-cap)</td>
          <td>α-helix (C-terminal, inside)</td>
          <td>Relative preference value at C2 (Richardson-Ri...</td>
          <td>0.336</td>
          <td>0.223765</td>
          <td>0.223765</td>
          <td>0.133513</td>
          <td>0.178217</td>
          <td>7.100000e-09</td>
          <td>1.185395e-06</td>
          <td>33,34,35,36</td>
        </tr>
        <tr>
          <th>8</th>
          <td>TMD_C_JMD_C-Segment(5,7)-KARS160107</td>
          <td>Shape</td>
          <td>Side chain length</td>
          <td>Eccentricity (maximum)</td>
          <td>Diameter (maximum eccentricity) (Karkbara-Knis...</td>
          <td>0.331</td>
          <td>0.217594</td>
          <td>0.217594</td>
          <td>0.136011</td>
          <td>0.172395</td>
          <td>1.130000e-08</td>
          <td>1.331786e-06</td>
          <td>32,33,34</td>
        </tr>
        <tr>
          <th>9</th>
          <td>TMD_C_JMD_C-Pattern(C,4,8)-JURD980101</td>
          <td>Polarity</td>
          <td>Hydrophobicity</td>
          <td>Hydrophobicity</td>
          <td>Modified Kyte-Doolittle hydrophobicity scale (...</td>
          <td>0.329</td>
          <td>0.264720</td>
          <td>-0.264720</td>
          <td>0.141666</td>
          <td>0.233134</td>
          <td>1.480000e-08</td>
          <td>1.425259e-06</td>
          <td>33,37</td>
        </tr>
      </tbody>
    </table>
    </div>



Which can be again used for machine learning:

.. code:: ipython3

    import seaborn as sns
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import matplotlib.pyplot as plt
    import pandas as pd
    
    X = sf.feat_matrix(df_parts=df_parts, features=df_feat["feature"])
    rf = RandomForestClassifier()
    cv = cross_val_score(rf, X, y, scoring="accuracy", cv=5, n_jobs=1) 
    print(f"Mean accuracy of {round(np.mean(cv), 2)}")
    
    aa.plot_settings(font_scale=1.1)
    sns.barplot(pd.DataFrame({"Baseline": cv_base, "CPP": cv}), palette=["tab:blue", "tab:red"])
    plt.ylabel("Mean accuracy", size=aa.plot_gcfs()+1)
    plt.ylim(0, 1)
    sns.despine()
    plt.show()


.. parsed-literal::

    Mean accuracy of 0.9



.. image:: output_13_1.png


4. Explainable AI
-----------------

Explainable AI on group level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explainable AI on individual level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

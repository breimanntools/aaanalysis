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
    # Load scales and scale categories (AAontology) 
    df_scales = aa.load_scales()
    # Load training data
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
    aac = aa.AAclust(model=AgglomerativeClustering, model_kwargs=dict(linkage="ward"))
    X = np.array(df_scales)
    scales = aac.fit(X, n_clusters=100, names=list(df_scales)) 
    df_scales = df_scales[scales]
    df_scales




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
          <th>CORJ870107</th>
          <th>ROBB760113</th>
          <th>MIYS990104</th>
          <th>BIGC670101</th>
          <th>ROSG850102</th>
          <th>ZIMJ680105</th>
          <th>...</th>
          <th>YUTK870102</th>
          <th>SUEM840102</th>
          <th>VASM830102</th>
          <th>VELV850101</th>
          <th>VENT840101</th>
          <th>MONM990101</th>
          <th>GEOR030102</th>
          <th>GEOR030106</th>
          <th>KARS160120</th>
          <th>LINS030117</th>
        </tr>
        <tr>
          <th>AA</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
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
          <td>0.446</td>
          <td>0.101</td>
          <td>0.479</td>
          <td>0.164</td>
          <td>0.564</td>
          <td>0.444</td>
          <td>...</td>
          <td>0.557</td>
          <td>0.103</td>
          <td>0.617</td>
          <td>0.295</td>
          <td>0</td>
          <td>0.077</td>
          <td>0.250</td>
          <td>0.516</td>
          <td>0.952</td>
          <td>0.186</td>
        </tr>
        <tr>
          <th>C</th>
          <td>0.864</td>
          <td>0.258</td>
          <td>0.303</td>
          <td>0.104</td>
          <td>0.725</td>
          <td>0.849</td>
          <td>0.000</td>
          <td>0.323</td>
          <td>1.000</td>
          <td>0.000</td>
          <td>...</td>
          <td>0.680</td>
          <td>0.337</td>
          <td>0.734</td>
          <td>0.657</td>
          <td>0</td>
          <td>0.154</td>
          <td>0.246</td>
          <td>0.000</td>
          <td>0.952</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>D</th>
          <td>1.000</td>
          <td>0.206</td>
          <td>0.000</td>
          <td>0.451</td>
          <td>0.000</td>
          <td>0.790</td>
          <td>0.803</td>
          <td>0.324</td>
          <td>0.256</td>
          <td>0.000</td>
          <td>...</td>
          <td>0.574</td>
          <td>0.909</td>
          <td>0.225</td>
          <td>1.000</td>
          <td>0</td>
          <td>0.923</td>
          <td>0.091</td>
          <td>0.404</td>
          <td>0.952</td>
          <td>0.186</td>
        </tr>
        <tr>
          <th>E</th>
          <td>0.420</td>
          <td>0.210</td>
          <td>0.090</td>
          <td>0.823</td>
          <td>0.233</td>
          <td>0.092</td>
          <td>0.859</td>
          <td>0.488</td>
          <td>0.256</td>
          <td>0.025</td>
          <td>...</td>
          <td>0.402</td>
          <td>0.077</td>
          <td>0.531</td>
          <td>0.046</td>
          <td>0</td>
          <td>0.923</td>
          <td>0.404</td>
          <td>0.610</td>
          <td>0.952</td>
          <td>0.349</td>
        </tr>
        <tr>
          <th>F</th>
          <td>0.877</td>
          <td>0.887</td>
          <td>0.724</td>
          <td>0.402</td>
          <td>0.950</td>
          <td>0.328</td>
          <td>0.000</td>
          <td>0.783</td>
          <td>0.923</td>
          <td>1.000</td>
          <td>...</td>
          <td>0.680</td>
          <td>0.233</td>
          <td>0.023</td>
          <td>0.749</td>
          <td>1</td>
          <td>0.000</td>
          <td>0.536</td>
          <td>0.712</td>
          <td>0.952</td>
          <td>0.326</td>
        </tr>
        <tr>
          <th>G</th>
          <td>0.025</td>
          <td>0.032</td>
          <td>0.259</td>
          <td>0.055</td>
          <td>0.352</td>
          <td>1.000</td>
          <td>0.662</td>
          <td>0.000</td>
          <td>0.513</td>
          <td>0.175</td>
          <td>...</td>
          <td>0.525</td>
          <td>0.000</td>
          <td>0.455</td>
          <td>0.040</td>
          <td>0</td>
          <td>0.692</td>
          <td>0.000</td>
          <td>0.210</td>
          <td>0.952</td>
          <td>0.023</td>
        </tr>
        <tr>
          <th>H</th>
          <td>0.840</td>
          <td>0.387</td>
          <td>0.401</td>
          <td>0.463</td>
          <td>0.610</td>
          <td>0.454</td>
          <td>0.479</td>
          <td>0.561</td>
          <td>0.667</td>
          <td>0.338</td>
          <td>...</td>
          <td>0.754</td>
          <td>0.000</td>
          <td>0.345</td>
          <td>0.191</td>
          <td>0</td>
          <td>0.923</td>
          <td>0.201</td>
          <td>0.612</td>
          <td>0.562</td>
          <td>0.419</td>
        </tr>
        <tr>
          <th>I</th>
          <td>0.000</td>
          <td>0.990</td>
          <td>0.697</td>
          <td>0.512</td>
          <td>0.969</td>
          <td>0.151</td>
          <td>0.056</td>
          <td>0.663</td>
          <td>0.923</td>
          <td>0.894</td>
          <td>...</td>
          <td>0.820</td>
          <td>0.714</td>
          <td>0.070</td>
          <td>0.000</td>
          <td>1</td>
          <td>0.154</td>
          <td>0.161</td>
          <td>0.457</td>
          <td>0.583</td>
          <td>0.140</td>
        </tr>
        <tr>
          <th>K</th>
          <td>0.506</td>
          <td>0.516</td>
          <td>0.127</td>
          <td>0.591</td>
          <td>0.027</td>
          <td>0.613</td>
          <td>1.000</td>
          <td>0.694</td>
          <td>0.000</td>
          <td>0.044</td>
          <td>...</td>
          <td>0.615</td>
          <td>0.012</td>
          <td>0.688</td>
          <td>0.294</td>
          <td>0</td>
          <td>0.923</td>
          <td>0.195</td>
          <td>0.536</td>
          <td>0.912</td>
          <td>1.000</td>
        </tr>
        <tr>
          <th>L</th>
          <td>0.272</td>
          <td>0.835</td>
          <td>0.905</td>
          <td>0.732</td>
          <td>1.000</td>
          <td>0.076</td>
          <td>0.014</td>
          <td>0.663</td>
          <td>0.846</td>
          <td>0.925</td>
          <td>...</td>
          <td>1.000</td>
          <td>0.428</td>
          <td>0.771</td>
          <td>0.000</td>
          <td>1</td>
          <td>0.000</td>
          <td>0.513</td>
          <td>0.690</td>
          <td>0.952</td>
          <td>0.186</td>
        </tr>
        <tr>
          <th>M</th>
          <td>0.704</td>
          <td>0.452</td>
          <td>1.000</td>
          <td>1.000</td>
          <td>0.883</td>
          <td>0.084</td>
          <td>0.113</td>
          <td>0.620</td>
          <td>0.846</td>
          <td>0.756</td>
          <td>...</td>
          <td>0.689</td>
          <td>0.701</td>
          <td>0.512</td>
          <td>0.651</td>
          <td>0</td>
          <td>0.077</td>
          <td>0.151</td>
          <td>0.670</td>
          <td>0.952</td>
          <td>0.372</td>
        </tr>
        <tr>
          <th>N</th>
          <td>0.988</td>
          <td>0.029</td>
          <td>0.381</td>
          <td>0.287</td>
          <td>0.171</td>
          <td>0.924</td>
          <td>0.718</td>
          <td>0.398</td>
          <td>0.282</td>
          <td>0.162</td>
          <td>...</td>
          <td>0.508</td>
          <td>0.000</td>
          <td>0.313</td>
          <td>0.028</td>
          <td>0</td>
          <td>1.000</td>
          <td>0.277</td>
          <td>0.342</td>
          <td>0.952</td>
          <td>0.093</td>
        </tr>
        <tr>
          <th>P</th>
          <td>0.605</td>
          <td>0.871</td>
          <td>0.403</td>
          <td>0.000</td>
          <td>0.130</td>
          <td>0.824</td>
          <td>0.803</td>
          <td>0.376</td>
          <td>0.308</td>
          <td>0.750</td>
          <td>...</td>
          <td>0.566</td>
          <td>0.545</td>
          <td>0.937</td>
          <td>0.157</td>
          <td>0</td>
          <td>1.000</td>
          <td>1.000</td>
          <td>1.000</td>
          <td>0.952</td>
          <td>0.698</td>
        </tr>
        <tr>
          <th>Q</th>
          <td>0.519</td>
          <td>0.000</td>
          <td>0.203</td>
          <td>0.805</td>
          <td>0.238</td>
          <td>0.546</td>
          <td>0.732</td>
          <td>0.539</td>
          <td>0.256</td>
          <td>0.388</td>
          <td>...</td>
          <td>0.697</td>
          <td>0.428</td>
          <td>0.446</td>
          <td>0.602</td>
          <td>0</td>
          <td>0.923</td>
          <td>0.478</td>
          <td>0.530</td>
          <td>0.952</td>
          <td>0.256</td>
        </tr>
        <tr>
          <th>R</th>
          <td>0.531</td>
          <td>0.268</td>
          <td>0.061</td>
          <td>0.738</td>
          <td>0.482</td>
          <td>0.748</td>
          <td>0.634</td>
          <td>0.735</td>
          <td>0.308</td>
          <td>0.112</td>
          <td>...</td>
          <td>0.000</td>
          <td>0.000</td>
          <td>0.550</td>
          <td>0.760</td>
          <td>0</td>
          <td>1.000</td>
          <td>0.549</td>
          <td>0.728</td>
          <td>0.952</td>
          <td>0.372</td>
        </tr>
        <tr>
          <th>S</th>
          <td>0.679</td>
          <td>0.045</td>
          <td>0.450</td>
          <td>0.293</td>
          <td>0.293</td>
          <td>0.798</td>
          <td>0.704</td>
          <td>0.188</td>
          <td>0.359</td>
          <td>0.256</td>
          <td>...</td>
          <td>0.656</td>
          <td>0.000</td>
          <td>0.868</td>
          <td>0.657</td>
          <td>0</td>
          <td>0.231</td>
          <td>0.168</td>
          <td>0.399</td>
          <td>0.952</td>
          <td>0.186</td>
        </tr>
        <tr>
          <th>T</th>
          <td>0.494</td>
          <td>0.174</td>
          <td>0.619</td>
          <td>0.360</td>
          <td>0.279</td>
          <td>0.529</td>
          <td>0.577</td>
          <td>0.352</td>
          <td>0.462</td>
          <td>0.419</td>
          <td>...</td>
          <td>0.574</td>
          <td>0.000</td>
          <td>1.000</td>
          <td>0.745</td>
          <td>0</td>
          <td>0.000</td>
          <td>0.344</td>
          <td>0.513</td>
          <td>0.000</td>
          <td>0.419</td>
        </tr>
        <tr>
          <th>V</th>
          <td>0.000</td>
          <td>0.577</td>
          <td>0.183</td>
          <td>0.451</td>
          <td>0.907</td>
          <td>0.000</td>
          <td>0.127</td>
          <td>0.492</td>
          <td>0.872</td>
          <td>0.719</td>
          <td>...</td>
          <td>0.770</td>
          <td>0.000</td>
          <td>0.408</td>
          <td>0.045</td>
          <td>1</td>
          <td>0.077</td>
          <td>0.151</td>
          <td>0.467</td>
          <td>0.952</td>
          <td>0.163</td>
        </tr>
        <tr>
          <th>W</th>
          <td>0.926</td>
          <td>1.000</td>
          <td>0.707</td>
          <td>0.805</td>
          <td>0.500</td>
          <td>0.773</td>
          <td>0.070</td>
          <td>1.000</td>
          <td>0.846</td>
          <td>0.894</td>
          <td>...</td>
          <td>0.467</td>
          <td>1.000</td>
          <td>0.138</td>
          <td>0.434</td>
          <td>1</td>
          <td>0.231</td>
          <td>0.066</td>
          <td>0.440</td>
          <td>1.000</td>
          <td>0.349</td>
        </tr>
        <tr>
          <th>Y</th>
          <td>0.802</td>
          <td>0.990</td>
          <td>0.425</td>
          <td>0.524</td>
          <td>0.771</td>
          <td>0.798</td>
          <td>0.127</td>
          <td>0.806</td>
          <td>0.615</td>
          <td>0.762</td>
          <td>...</td>
          <td>0.557</td>
          <td>0.857</td>
          <td>0.000</td>
          <td>0.408</td>
          <td>1</td>
          <td>0.154</td>
          <td>0.110</td>
          <td>0.666</td>
          <td>0.736</td>
          <td>0.349</td>
        </tr>
      </tbody>
    </table>
    <p>20 rows × 100 columns</p>
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
``sf.get_split_kws()``. - ``Scales``: Sets of amino acid scales. We
first use SequenceFeature to obtain Parts and Splits:

.. code:: ipython3

    # Feature Engineering
    y = list(df_seq["label"])
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=10, list_parts=["tmd_jmd"])
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

    # Small set of features (100 features created)
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws, verbose=False)
    df_feat = cpp.run(labels=y, tmd_len=20, jmd_n_len=10, jmd_c_len=10, n_filter=100)  # Default values for lengths are used
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
    X = sf.feat_matrix(df_parts=df_parts, df_scales=df_scales, features=df_feat["feature"])
    # ML evaluation
    rf = RandomForestClassifier()
    cv_base = cross_val_score(rf, X, y, scoring="accuracy", cv=5, n_jobs=8) # Set n_jobs=1 to disable multi-processing
    print(f"Mean accuracy of {round(np.mean(cv_base), 2)}")


.. parsed-literal::

    Mean accuracy of 0.57


Creating more features with CPP will take some more time. but improve
prediction performance:

.. code:: ipython3

    # Default CPP features  (around 100.000 features)
    split_kws = sf.get_split_kws()
    df_parts = sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=10)
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws, verbose=False)
    df_feat = cpp.run(labels=y, n_processes=8, n_filter=100)
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
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>95</th>
          <td>JMD_N_TMD_N-Pattern(C,6,9)-NAKH900106</td>
          <td>Composition</td>
          <td>Mitochondrial proteins</td>
          <td>Mitochondrial proteins</td>
          <td>Normalized composition from animal (Nakashima ...</td>
          <td>0.228</td>
          <td>0.172120</td>
          <td>-0.172120</td>
          <td>0.180254</td>
          <td>0.199987</td>
          <td>8.754340e-05</td>
          <td>2.693037e-04</td>
          <td>12,15</td>
        </tr>
        <tr>
          <th>96</th>
          <td>JMD_N_TMD_N-Pattern(C,6,9,12)-ZIMJ680105</td>
          <td>Others</td>
          <td>PC 2</td>
          <td>Principal Component 1 (Zimmerman)</td>
          <td>RF rank (Zimmerman et al., 1968)</td>
          <td>0.227</td>
          <td>0.133867</td>
          <td>-0.133867</td>
          <td>0.160532</td>
          <td>0.161415</td>
          <td>9.118090e-05</td>
          <td>2.778863e-04</td>
          <td>9,12,15</td>
        </tr>
        <tr>
          <th>97</th>
          <td>JMD_N_TMD_N-Segment(7,8)-KARS160107</td>
          <td>Shape</td>
          <td>Side chain length</td>
          <td>Eccentricity (maximum)</td>
          <td>Diameter (maximum eccentricity) (Karkbara-Knis...</td>
          <td>0.227</td>
          <td>0.098674</td>
          <td>-0.098674</td>
          <td>0.104428</td>
          <td>0.124875</td>
          <td>8.945330e-05</td>
          <td>2.740061e-04</td>
          <td>16,17</td>
        </tr>
        <tr>
          <th>98</th>
          <td>JMD_N_TMD_N-Pattern(C,6,9,12)-SIMZ760101</td>
          <td>Polarity</td>
          <td>Hydrophobicity</td>
          <td>Transfer free energy (TFE) to outside</td>
          <td>Transfer free energy (Simon, 1976), Cited by C...</td>
          <td>0.225</td>
          <td>0.161307</td>
          <td>-0.161307</td>
          <td>0.192235</td>
          <td>0.212741</td>
          <td>1.036749e-04</td>
          <td>3.042894e-04</td>
          <td>9,12,15</td>
        </tr>
        <tr>
          <th>99</th>
          <td>JMD_N_TMD_N-Pattern(C,3,6)-TANS770102</td>
          <td>Conformation</td>
          <td>α-helix (C-term, out)</td>
          <td>α-helix (C-terminal, outside)</td>
          <td>Normalized frequency of isolated helix (Tanaka...</td>
          <td>0.224</td>
          <td>0.108020</td>
          <td>-0.108020</td>
          <td>0.133731</td>
          <td>0.139419</td>
          <td>1.143783e-04</td>
          <td>3.272494e-04</td>
          <td>15,18</td>
        </tr>
      </tbody>
    </table>
    <p>100 rows × 13 columns</p>
    </div>



Which can be again used for machine learning:

.. code:: ipython3

    import seaborn as sns
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import matplotlib.pyplot as plt
    import pandas as pd
    X = sf.feat_matrix(df_parts=df_parts, df_scales=df_scales, features=df_feat["feature"])
    # ML evaluation
    rf = RandomForestClassifier()
    cv = cross_val_score(rf, X, y, scoring="accuracy", cv=5, n_jobs=1) 
    print(f"Mean accuracy of {round(np.mean(cv), 2)}")
    aa.plot_settings(font_scale=1.1)
    sns.barplot(pd.DataFrame({"Baseline": cv_base, "CPP": cv}), palette=["tab:blue", "tab:red"])
    plt.ylabel("Mean accuracy", size=aa.plot_gcfs()+1)
    sns.despine()
    plt.show()


.. parsed-literal::

    Mean accuracy of 0.95



.. image:: output_13_1.png


4. Explainable AI
-----------------

Explainable AI on group level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explainable AI on individual level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

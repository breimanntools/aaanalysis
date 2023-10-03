Scale Loading Tutorial
======================

This is a tutorial on the ``load_scales`` function for loading of amino
acid scales sets, their classification (AAontology), or evaluation
(AAclust top60).

Six different datasets can be loading in total by using the ``name``
parameter: ``scales``, ``scales_raw``, ``scales_pc``, ``scales_cat``,
``top60``, and ``top60_eval``.

Three sets of numerical amino acid scales
-----------------------------------------

-  ``scales_raw``: Original amino acid scales sourced from AAindex and
   two other datasets.
-  ``scales``: Min-max normalized version of the raw scales.
-  ``scales_pc``: Scales compressed using principal component analysis
   (PCA).

Amino acid scales are indicated by a unique id (columns) and assign a
numerical value to each canonical amino acid:

.. code:: ipython2

    import aaanalysis as aa
    df_scales = aa.load_scales()
    df_raw = aa.load_scales(name="scales_raw")
    df_pc = aa.load_scales(name="scales_pc")
    df_scales.iloc[:5, :4]




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
          <th>ARGP820101</th>
          <th>ARGP820102</th>
          <th>ARGP820103</th>
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
          <td>0.230</td>
          <td>0.355</td>
          <td>0.504</td>
        </tr>
        <tr>
          <th>C</th>
          <td>0.864</td>
          <td>0.404</td>
          <td>0.579</td>
          <td>0.387</td>
        </tr>
        <tr>
          <th>D</th>
          <td>1.000</td>
          <td>0.174</td>
          <td>0.000</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>E</th>
          <td>0.420</td>
          <td>0.177</td>
          <td>0.019</td>
          <td>0.032</td>
        </tr>
        <tr>
          <th>F</th>
          <td>0.877</td>
          <td>0.762</td>
          <td>0.601</td>
          <td>0.670</td>
        </tr>
      </tbody>
    </table>
    </div>



AAontology
----------

-  ``scales_cat`` provides a two-level classification for all
   ``scales``, termed AAontology.

The entries in the ``scale_id`` column align with the column names of
``df_scales`` and \`\ ``df_raw``. Additional columns detail further
information from AAontology, such as scale category or description.

.. code:: ipython2

    df_cat = aa.load_scales(name="scales_cat")
    df_cat.head(5)




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
          <th>scale_id</th>
          <th>category</th>
          <th>subcategory</th>
          <th>scale_name</th>
          <th>scale_description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>LINS030110</td>
          <td>ASA/Volume</td>
          <td>Accessible surface area (ASA)</td>
          <td>ASA (folded coil/turn)</td>
          <td>Total median accessible surfaces of whole resi...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>LINS030113</td>
          <td>ASA/Volume</td>
          <td>Accessible surface area (ASA)</td>
          <td>ASA (folded coil/turn)</td>
          <td>% total accessible surfaces of whole residues ...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>JANJ780101</td>
          <td>ASA/Volume</td>
          <td>Accessible surface area (ASA)</td>
          <td>ASA (folded protein)</td>
          <td>Average accessible surface area (Janin et al.,...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>JANJ780103</td>
          <td>ASA/Volume</td>
          <td>Accessible surface area (ASA)</td>
          <td>ASA (folded protein)</td>
          <td>Percentage of exposed residues (Janin et al., ...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>LINS030104</td>
          <td>ASA/Volume</td>
          <td>Accessible surface area (ASA)</td>
          <td>ASA (folded protein)</td>
          <td>Total median accessible surfaces of whole resi...</td>
        </tr>
      </tbody>
    </table>
    </div>



Redundancy-reduce scale subsets
-------------------------------

The remaining two datasets stem from an in-depth analysis of
redundancy-reduced subsets of ``scales`` using AAclust.

-  ``top60`` comprises the 60 best performing scale sets, benchmarked on
   our protein datasets available by the ``aa.load_dataset`` function.

These have a unique AAclust id (``top60_id`` index, ‘ACC’ for AAclust)
and the presence (1) or absence (0) of scales is indicated.

.. code:: ipython2

    df_top60 = aa.load_scales(name="top60")
    df_top60.sum(axis=1)    # Shows number of scales in each subset
    df_top60.iloc[:5, :4]




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
          <th>ARGP820101</th>
          <th>ARGP820102</th>
          <th>ARGP820103</th>
        </tr>
        <tr>
          <th>top60_id</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>AAC01</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>AAC02</th>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>AAC03</th>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>AAC04</th>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>AAC05</th>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



-  ``top60_eval`` shows the average accuracy for each protein scale
   subset given by their ids (index) across all tested protein
   benchmarks (columns):

.. code:: ipython2

    df_eval = aa.load_scales(name="top60_eval")
    df_eval.mean(axis=1)  # Shows the overall average performance used for ranking
    df_eval.iloc[:5, :4]




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
          <th>SEQ_AMYLO</th>
          <th>SEQ_CAPSID</th>
          <th>SEQ_DISULFIDE</th>
          <th>SEQ_LOCATION</th>
        </tr>
        <tr>
          <th>top60_id</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>AAC01</th>
          <td>0.761</td>
          <td>0.827</td>
          <td>0.732</td>
          <td>0.746</td>
        </tr>
        <tr>
          <th>AAC02</th>
          <td>0.747</td>
          <td>0.830</td>
          <td>0.733</td>
          <td>0.742</td>
        </tr>
        <tr>
          <th>AAC03</th>
          <td>0.741</td>
          <td>0.829</td>
          <td>0.734</td>
          <td>0.746</td>
        </tr>
        <tr>
          <th>AAC04</th>
          <td>0.747</td>
          <td>0.828</td>
          <td>0.731</td>
          <td>0.747</td>
        </tr>
        <tr>
          <th>AAC05</th>
          <td>0.739</td>
          <td>0.830</td>
          <td>0.735</td>
          <td>0.752</td>
        </tr>
      </tbody>
    </table>
    </div>



Use the ``top60_n`` parameters to select the n-th best scale set, either
as ``scales``, ``scales_raw``, or ``scales_cat``

.. code:: ipython2

    df_cat_1 = aa.load_scales(name="scales_cat", top60_n=1)
    df_raw_1 = aa.load_scales(name="scales_raw", top60_n=1)
    df_scales_1 = aa.load_scales(top60_n=1)
    # Which is the same as 
    df_top60 = aa.load_scales(name="top60")
    selected_scales = df_top60.columns[df_top60.loc["AAC01"] == 1].tolist()
    df_aac1 = df_scales[selected_scales] 

Filtering of scales
-------------------

Two parameters are provided to filter ``df_scales``, ``df_cat``, and
``df_raw``. You can exclude scales from the other two data sources
(i.e., scales not contained in AAindex) setting ``just_aaindex=True``,
which is disabled by default. AAontology comprises scales that were not
subordinated to any subcategory (‘unclassified’ scales), which can be
excluded by setting ``unclassified_out=True``:

.. code:: ipython2

    df_scales = aa.load_scales(just_aaindex=True, unclassified_out=True)
    df_raw = aa.load_scales(name="scales_raw")
    df_cat = aa.load_scales(name="scales_cat", just_aaindex=True, unclassified_out=True)


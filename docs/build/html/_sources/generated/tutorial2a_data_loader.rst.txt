Data Loading Tutorial
=====================

This is a tutorial on loading of protein benchmark datasets.

Loading of protein benchmarks
-----------------------------

Load the overview table of protein benchmark datasets using the default
settings:

.. code:: ipython2

    import aaanalysis as aa
    df_info = aa.load_dataset()
    df_info.iloc[:13, :7]




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
          <th>Level</th>
          <th>Dataset</th>
          <th># Sequences</th>
          <th># Amino acids</th>
          <th># Positives</th>
          <th># Negatives</th>
          <th>Predictor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Amino acid</td>
          <td>AA_CASPASE3</td>
          <td>233</td>
          <td>185605</td>
          <td>705</td>
          <td>184900</td>
          <td>PROSPERous</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Amino acid</td>
          <td>AA_FURIN</td>
          <td>71</td>
          <td>59003</td>
          <td>163</td>
          <td>58840</td>
          <td>PROSPERous</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Amino acid</td>
          <td>AA_LDR</td>
          <td>342</td>
          <td>118248</td>
          <td>35469</td>
          <td>82779</td>
          <td>IDP-Seq2Seq</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Amino acid</td>
          <td>AA_MMP2</td>
          <td>573</td>
          <td>312976</td>
          <td>2416</td>
          <td>310560</td>
          <td>PROSPERous</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Amino acid</td>
          <td>AA_RNABIND</td>
          <td>221</td>
          <td>55001</td>
          <td>6492</td>
          <td>48509</td>
          <td>GMKSVM-RU</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Amino acid</td>
          <td>AA_SA</td>
          <td>233</td>
          <td>185605</td>
          <td>101082</td>
          <td>84523</td>
          <td>PROSPERous</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Sequence</td>
          <td>SEQ_AMYLO</td>
          <td>1414</td>
          <td>8484</td>
          <td>511</td>
          <td>903</td>
          <td>ReRF-Pred</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Sequence</td>
          <td>SEQ_CAPSID</td>
          <td>7935</td>
          <td>3364680</td>
          <td>3864</td>
          <td>4071</td>
          <td>VIRALpro</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Sequence</td>
          <td>SEQ_DISULFIDE</td>
          <td>2547</td>
          <td>614470</td>
          <td>897</td>
          <td>1650</td>
          <td>Dipro</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Sequence</td>
          <td>SEQ_LOCATION</td>
          <td>1835</td>
          <td>732398</td>
          <td>1045</td>
          <td>790</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>10</th>
          <td>Sequence</td>
          <td>SEQ_SOLUBLE</td>
          <td>17408</td>
          <td>4432269</td>
          <td>8704</td>
          <td>8704</td>
          <td>SOLpro</td>
        </tr>
        <tr>
          <th>11</th>
          <td>Sequence</td>
          <td>SEQ_TAIL</td>
          <td>6668</td>
          <td>2671690</td>
          <td>2574</td>
          <td>4094</td>
          <td>VIRALpro</td>
        </tr>
        <tr>
          <th>12</th>
          <td>Domain</td>
          <td>DOM_GSEC</td>
          <td>126</td>
          <td>92964</td>
          <td>63</td>
          <td>63</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



The benchmark datasets are categorized into amino acid (‘AA’), domain
(‘DOM’), and sequence (‘SEQ’) level datasets, indicated by their
``name`` prefix, as exemplified here.

.. code:: ipython2

    df_seq1 = aa.load_dataset(name="AA_CASPASE3")
    df_seq2 = aa.load_dataset(name="SEQ_CAPSID")
    df_seq3 = aa.load_dataset(name="DOM_GSEC")
    df_seq2.head(2)
    # Compare columns of three types




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
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>CAPSID_1</td>
          <td>MVTHNVKINKHVTRRSYSSAKEVLEIPPLTEVQTASYKWFMDKGIK...</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>CAPSID_2</td>
          <td>MKKRQKKMTLSNFTDTSFQDFVSAEQVDDKSAMALINRAEDFKAGQ...</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



Each dataset can be utilized for a binary classification, with labels
being positive (1) or negative (0). A balanced number of samples can be
chosen by the ``n`` parameter, defining the sample number per class.

.. code:: ipython2

    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=100)
    # Returns 200 samples, 100 positives and 100 negatives
    df_seq["label"].value_counts()




.. parsed-literal::

    label
    0    100
    1    100
    Name: count, dtype: int64



Or randomly selected using ``random=True``:

.. code:: ipython2

    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=100, random=True)

The protein sequences can have varying length:

.. code:: ipython2

    # Plot distribution
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Utility AAanalysis function for publication ready plots
    aa.plot_settings(font_scale=1.2) 
    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=100)
    list_seq_lens = df_seq["sequence"].apply(len)
    sns.histplot(list_seq_lens, binwidth=50)
    sns.despine()
    plt.xlim(0, 1500)
    plt.show()



.. image:: NOTEBOOK_1_output_9_0.png


Which can be easily filtered using ``min_len`` and ``max_len``
parameters:

.. code:: ipython2

    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=100, min_len=200, max_len=800)
    list_seq_lens = df_seq["sequence"].apply(len)
    aa.plot_settings(font_scale=1.2)  # Utility AAanalysis function for publication ready plots
    sns.histplot(list_seq_lens, binwidth=50)
    sns.despine()
    plt.xlim(0, 1500)
    plt.show()



.. image:: NOTEBOOK_2_output_11_0.png


Loading of protein benchmarks: Amino acid window size
-----------------------------------------------------

For amino acid level datasets, labels are provided for each residue
position, which can be seen by setting ``aa_window_size=None``:

.. code:: ipython2

    df_seq = aa.load_dataset(name="AA_CASPASE3", aa_window_size=None)
    df_seq.head(4)




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
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>CASPASE3_1</td>
          <td>MSLFDLFRGFFGFPGPRSHRDPFFGGMTRDEDDDEEEEEEGGSWGR...</td>
          <td>0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>CASPASE3_2</td>
          <td>MEVTGDAGVPESGEIRTLKPCLLRRNYSREQHGVAASCLEDLRSKA...</td>
          <td>0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>CASPASE3_3</td>
          <td>MRARSGARGALLLALLLCWDPTPSLAGIDSGGQALPDSFPSAPAEQ...</td>
          <td>0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>CASPASE3_4</td>
          <td>MDAKARNCLLQHREALEKDIKTSYIMDHMISDGFLTISEEEKVRNE...</td>
          <td>0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,...</td>
        </tr>
      </tbody>
    </table>
    </div>



For convenience, we provide an “amino acid window” of length n. This
window represents a specific amino acid, which is flanked by (n-1)/2
residues on both its N-terminal and C-terminal sides. It’s essential for
n to be odd, ensuring equal residues on both sides. While the default
window size is 9, sizes between 5 and 15 are also popular.

.. code:: ipython2

    df_seq = aa.load_dataset(name="AA_CASPASE3", n=2)
    df_seq




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
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>CASPASE3_1_pos126</td>
          <td>QTLRDSMLK</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>CASPASE3_1_pos127</td>
          <td>TLRDSMLKY</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>CASPASE3_1_pos4</td>
          <td>MSLFDLFRG</td>
          <td>0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>CASPASE3_1_pos5</td>
          <td>SLFDLFRGF</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



Sequences can be pre-filtered using ``min_len`` and ``max_len`` and
``n`` residues can be randomly selected by ``random`` with different
``aa_window_size``\ s.

.. code:: ipython2

    df_seq = aa.load_dataset(name="AA_CASPASE3", min_len=20, n=2, random=True, aa_window_size=21)
    df_seq




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
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>CASPASE3_94_pos31</td>
          <td>VSHWQQQSYLDSGIHSGATTT</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>CASPASE3_129_pos530</td>
          <td>WFNKVLEDKTDDASTPATDTS</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>CASPASE3_76_pos554</td>
          <td>QLLRGVKHLHDNWILHRDLKT</td>
          <td>0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>CASPASE3_19_pos163</td>
          <td>GHRGNSLDRRSQGGPHLSGAV</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



Loading of protein benchmarks: Positive-Unlabeled (PU) datasets
---------------------------------------------------------------

In typical binary classification, data is labeled as positive (1) or
negative (0). But with many protein sequence datasets, we face
challenges: they might be small, unbalanced, or lack a clear negative
class. For datasets with only positive and unlabeled samples (2), we use
PU learning. This approach identifies reliable negatives from the
unlabeled data to make binary classification possible. We offer
benchmark datasets for this scenario, denoted by the ``_PU`` suffix. For
example, the ``DOM_GSEC_PU`` dataset corresponds to the
``DOM_GSEC set``.

.. code:: ipython2

    df_seq = aa.load_dataset(name="DOM_GSEC", n=2)
    df_seq




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
          <td>P05067</td>
          <td>MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMN...</td>
          <td>1</td>
          <td>701</td>
          <td>723</td>
          <td>FAEDVGSNKG</td>
          <td>AIIGLMVGGVVIATVIVITLVML</td>
          <td>KKKQYTSIHH</td>
        </tr>
        <tr>
          <th>3</th>
          <td>P14925</td>
          <td>MAGRARSGLLLLLLGLLALQSSCLAFRSPLSVFKRFKETTRSFSNE...</td>
          <td>1</td>
          <td>868</td>
          <td>890</td>
          <td>KLSTEPGSGV</td>
          <td>SVVLITTLLVIPVLVLLAIVMFI</td>
          <td>RWKKSRAFGD</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    df_seq_pu = aa.load_dataset(name="DOM_GSEC_PU", n=2)
    df_seq_pu




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
          <td>P05067</td>
          <td>MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMN...</td>
          <td>1</td>
          <td>701</td>
          <td>723</td>
          <td>FAEDVGSNKG</td>
          <td>AIIGLMVGGVVIATVIVITLVML</td>
          <td>KKKQYTSIHH</td>
        </tr>
        <tr>
          <th>1</th>
          <td>P14925</td>
          <td>MAGRARSGLLLLLLGLLALQSSCLAFRSPLSVFKRFKETTRSFSNE...</td>
          <td>1</td>
          <td>868</td>
          <td>890</td>
          <td>KLSTEPGSGV</td>
          <td>SVVLITTLLVIPVLVLLAIVMFI</td>
          <td>RWKKSRAFGD</td>
        </tr>
        <tr>
          <th>2</th>
          <td>P12821</td>
          <td>MGAASGRRGPGLLLPLPLLLLLPPQPALALDPGLQPGNFSADEAGA...</td>
          <td>2</td>
          <td>1257</td>
          <td>1276</td>
          <td>GLDLDAQQAR</td>
          <td>VGQWLLLFLGIALLVATLGL</td>
          <td>SQRLFSIRHR</td>
        </tr>
        <tr>
          <th>3</th>
          <td>P36896</td>
          <td>MAESAGASSFFPLVVLLLAGSGGSGPRGVQALLCACTSCLQANYTC...</td>
          <td>2</td>
          <td>127</td>
          <td>149</td>
          <td>EHPSMWGPVE</td>
          <td>LVGIIAGPVFLLFLIIIIVFLVI</td>
          <td>NYHQRVYHNR</td>
        </tr>
      </tbody>
    </table>
    </div>




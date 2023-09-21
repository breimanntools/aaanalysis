Data loading
============

An overview of the benchmark datasets are provided by default:

.. code:: ipython2

    import aaanalysis as aa
    df_info = aa.load_dataset()
    print(df_info.iloc[:, :7])


.. parsed-literal::

             Level        Dataset  # Sequences  # Amino acids  # Positives  # Negatives    Predictor
    0   Amino acid    AA_CASPASE3          233         185605          705       184900   PROSPERous
    1   Amino acid       AA_FURIN           71          59003          163        58840   PROSPERous
    2   Amino acid         AA_LDR          342         118248        35469        82779  IDP-Seq2Seq
    3   Amino acid        AA_MMP2          573         312976         2416       310560   PROSPERous
    4   Amino acid     AA_RNABIND          221          55001         6492        48509    GMKSVM-RU
    5   Amino acid          AA_SA          233         185605       101082        84523   PROSPERous
    6     Sequence      SEQ_AMYLO         1414           8484          511          903    ReRF-Pred
    7     Sequence     SEQ_CAPSID         7935        3364680         3864         4071     VIRALpro
    8     Sequence  SEQ_DISULFIDE         2547         614470          897         1650        Dipro
    9     Sequence   SEQ_LOCATION         1835         732398         1045          790          NaN
    10    Sequence    SEQ_SOLUBLE        17408        4432269         8704         8704       SOLpro
    11    Sequence       SEQ_TAIL         6668        2671690         2574         4094     VIRALpro
    12      Domain       DOM_GSEC          126          92964           63           63          NaN
    13      Domain    DOM_GSEC_PU          694         494524           63            0          NaN


Benchmark datasets are categorized into amino acid (‘AA’), domain
(‘DOM’), and sequence (‘SEQ’) level datasets, indicated name suffix.

.. code:: ipython2

    df_seq1 = aa.load_dataset(name="AA_FURIN")
    df_seq2 = aa.load_dataset(name="SEQ_AMYLO")
    df_seq3 = aa.load_dataset(name="DOM_GSEC")
    print(df_seq3.head(5))


.. parsed-literal::

        entry                                           sequence  label  tmd_start  tmd_stop       jmd_n                      tmd       jmd_c
    0  P05067  MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMN...      1        701       723  FAEDVGSNKG  AIIGLMVGGVVIATVIVITLVML  KKKQYTSIHH
    1  P14925  MAGRARSGLLLLLLGLLALQSSCLAFRSPLSVFKRFKETTRSFSNE...      1        868       890  KLSTEPGSGV  SVVLITTLLVIPVLVLLAIVMFI  RWKKSRAFGD
    2  P70180  MRSLLLFTFSACVLLARVLLAGGASSGAGDTRPGSRRRAREALAAQ...      1        477       499  PCKSSGGLEE  SAVTGIVVGALLGAGLLMAFYFF  RKKYRITIER
    3  Q03157  MGPTSPAARGQGRRWRPPPLPLLLPLSLLLLRAQLAVGNLAVGSPS...      1        585       607  APSGTGVSRE  ALSGLLIMGAGGGSLIVLSLLLL  RKKKPYGTIS
    4  Q06481  MAATGTAAAAATGRLLLLLLVGLTAPALALAGYIEALAANAGTGFA...      1        694       716  LREDFSLSSS  ALIGLLVIAVAIATVIVISLVML  RKRQYGTISH


For some datasets, an additional version of it is provided for
positive-unlabeled (PU) learning containing only positive (1) and
unlabeled (2) data samples, as indicated by *dataset_name_PU* (e.g.,
‘DOM_GSEC_PU’).

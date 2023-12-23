You can evaluate different sets of identified negative samples using the
``dPULearn.eval()`` method. Load first one of our example datasets with
its respective features:

.. code:: ipython2

    import aaanalysis as aa
    aa.options["verbose"] = False
    # Dataset with positive (γ-secretase substrates)
    # and unlabeled data (proteins with unknown substrate status)
    df_seq = aa.load_dataset(name="DOM_GSEC_PU")
    labels = df_seq["label"].to_numpy()
    n_pos = sum([x == 1 for x in labels])
    df_feat = aa.load_features(name="DOM_GSEC")
    aa.display_df(df_seq)

Create features using


.. code:: ipython2

    _df_seq = aa.load_dataset(name="DOM_GSEC")


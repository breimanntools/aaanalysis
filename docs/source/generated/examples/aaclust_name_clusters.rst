We first create an example dataset of 100 scales and obtain their
``AAontolgy`` subcategory names to showcase the automatic cluster naming
by ``AAclust.name_clusters()`` method:

.. code:: ipython2

    import aaanalysis as aa
    # Create example dataset comprising 100 scales
    df_scales = aa.load_scales().T.sample(100).T
    X = df_scales.T
    df_cat = aa.load_scales(name="scales_cat")
    dict_scale_name = dict(zip(df_cat["scale_id"], df_cat["subcategory"]))
    names = [dict_scale_name[s] for s in list(df_scales)]
    # Fit AAclust model and obtain clustering label for 10 clusters
    aac = aa.AAclust()
    aac.fit(X, n_clusters=10)
    labels = aac.labels_

We can now provide the feature matrix ``X``, ``names``, and ``labels``
to the ``AAclust.name_clusters()`` method:

.. code:: ipython2

    cluster_names = aac.name_clusters(X, labels=labels, names=names)
    print("Name of clusters: ", list(sorted(set(cluster_names))))


.. parsed-literal::

    Name of clusters:  ['AA composition', 'Accessible surface area', 'Free energy', 'Hydrophobicity', 'MPs', 'Non-bonded energy', 'Volume', 'α-helix', 'β-strand', 'β-turn']


These names are automatically shorten, which can be disabled by setting
``shorten_names=False``:

.. code:: ipython2

    cluster_names = aac.name_clusters(X, labels=labels, names=names, shorten_names=False)
    print("Longer names: ", list(sorted(set(cluster_names))))


.. parsed-literal::

    Longer names:  ['AA composition', 'Accessible surface area (ASA)', 'Free energy (unfolding)', 'Hydrophobicity', 'Non-bonded energy', 'Volume', 'α-helix', 'β-strand', 'β-turn', 'β-turn (N-term)']


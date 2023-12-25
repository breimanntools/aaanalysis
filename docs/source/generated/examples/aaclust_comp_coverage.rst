The overlap of unique names between two list of names (in percentage)
can be computed by ``AAclust().comp_coverage()`` method:

.. code:: ipython2

    import aaanalysis as aa
    df_cat = aa.load_scales(name="scales_cat")
    names_ref = df_cat["subcategory"].to_list()
    names = names_ref[0:50]
    coverage = aa.AAclust().comp_coverage(names=names, names_ref=names_ref)
    print(f"The scale subcategories of the first 50 scales cover {coverage}% of all scale subcategories from AAontology.")


.. parsed-literal::

    The scale subcategories of the first 50 scales cover 6.76% of all scale subcategories from AAontology.


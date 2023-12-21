Adjusting figures using ``aa.plot_settings`` could change the fontsize
to a non-integer number:

.. code:: ipython2

    import matplotlib.pyplot as plt
    import seaborn as sns
    import aaanalysis as aa
    data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 27, 43]}
    colors = aa.plot_get_clist()
    aa.plot_settings(font_scale=0.9)
    sns.barplot(y='Classes', x='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    fontsize = aa.plot_gcfs()
    plt.title(f"Title fontsize: {fontsize}", fontsize=fontsize)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_gcfs_1_output_1_0.png


Which can be consistently adjusted using ``aa.plot_gcfs()``:

.. code:: ipython2

    sns.barplot(y='Classes', x='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    fontsize = aa.plot_gcfs() + 4
    plt.title(f"Title fontsize: {fontsize}", fontsize=fontsize)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_gcfs_2_output_3_0.png


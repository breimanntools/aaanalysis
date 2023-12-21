We provide two default ``AAanalysis`` default color dictionaries, which
can be accessed via ``aa.plot_get_cdict()``. First, colors for plot
elements such as SHAP plots can be retrieved by ``name='DICT_COLOR'``:

.. code:: ipython2

    import matplotlib.pyplot as plt
    import seaborn as sns
    import aaanalysis as aa
    dict_color = aa.plot_get_cdict(name="DICT_COLOR")
    data = {"Plot Elements": list(dict_color.keys()), 'Values': [1] * len(dict_color) }
    aa.plot_settings(weight_bold=False)
    ax = sns.barplot(data=data, x="Values", y="Plot Elements", palette=dict_color, hue="Plot Elements")
    ax.xaxis.set_visible(False)
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_get_cdict_1_output_1_0.png


The other dictionary comprises the default colors for the scale
categories from AAontology, retrieved by ``name='DICT_CAT'``:

.. code:: ipython2

    dict_color = aa.plot_get_cdict(name="DICT_CAT")
    data = {"Scale Categories": list(dict_color.keys()), 'Values': [1] * len(dict_color) }
    aa.plot_settings(weight_bold=False)
    ax = sns.barplot(data=data, x="Values", y="Scale Categories", palette=dict_color, hue="Scale Categories")
    ax.xaxis.set_visible(False)
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_get_cdict_2_output_3_0.png


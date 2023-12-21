Continuous color maps (cmap) for ‘CPP plots’ and ‘CPP-SHAP plots’ can be
retrieved using ``aa.plot_get_cmap()``, where the number of colors is
set using ``n_colors``:

.. code:: ipython2

    import matplotlib.pyplot as plt
    import seaborn as sns
    import aaanalysis as aa
    data = {'Classes': ['Negative values', 'Middle value (0)', 'Positive values',], 'Values': [13, 23, 33]}
    aa.plot_settings(font_scale=0.9)
    colors = aa.plot_get_cmap(n_colors=3)
    sns.barplot(data=data, x='Classes', y='Values', palette=colors, hue="Classes")
    plt.show()



.. image:: examples/plot_get_cmap_1_output_1_0.png


For ‘CPP plots’, we recommend using a white facecolor using
``facecolor_dark=False``:

.. code:: ipython2

    colors = aa.plot_get_cmap(name="CPP", n_colors=3, facecolor_dark=False)
    sns.barplot(data=data, x='Classes', y='Values', palette=colors, hue="Classes",
                edgecolor="black")
    plt.show()



.. image:: examples/plot_get_cmap_2_output_3_0.png


For ‘CPP-SHAP plots’, we recommend using a dark facecolor:

.. code:: ipython2

    colors = aa.plot_get_cmap(name="SHAP", n_colors=3, facecolor_dark=True)
    sns.barplot(data=data, x='Classes', y='Values', palette=colors, hue="Classes")
    plt.show()



.. image:: examples/plot_get_cmap_3_output_5_0.png


The number of colors steps can be adjusted to any number integer number:

.. code:: ipython2

    n = 11
    colors = aa.plot_get_cmap(n_colors=n)
    sns.palplot(colors)
    plt.show()
    colors = aa.plot_get_cmap(n_colors=n, facecolor_dark=False)
    sns.palplot(colors)
    plt.show()
    colors = aa.plot_get_cmap(n_colors=n, facecolor_dark=True)
    sns.palplot(colors)
    plt.show()



.. image:: examples/plot_get_cmap_4_output_7_0.png



.. image:: examples/plot_get_cmap_5_output_7_1.png



.. image:: examples/plot_get_cmap_6_output_7_2.png


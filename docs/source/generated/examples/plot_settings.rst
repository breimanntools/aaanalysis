A default seaborn barplot can be created as follows:

.. code:: ipython2

    import matplotlib.pyplot as plt
    import seaborn as sns
    data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 27, 43]}
    sns.barplot(x="Classes", y="Values", data=data)
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_1_output_1_0.png


Adjust plots with AAanalysis using ``aa.plot_settings()``:

.. code:: ipython2

    import aaanalysis as aa
    aa.plot_settings()
    sns.barplot(x="Classes", y="Values", data=data)
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_2_output_3_0.png


You can add our default colors using the ``aa.plot_get_clist()`` method:

.. code:: ipython2

    colors = aa.plot_get_clist(n_colors=3)
    sns.barplot(x="Classes", y="Values", data=data, palette=colors, hue="Classes")
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_3_output_5_0.png


Adjust the font scale for all plot texts using a scaling factor called
``font_scale``:

.. code:: ipython2

    aa.plot_settings(font_scale=1.5)
    sns.barplot(x="Classes", y="Values", data=data, palette=colors, hue="Classes")
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_4_output_7_0.png


The font type and style can be adjusted by the ``font`` and
``weight_bold`` arguments:

.. code:: ipython2

    aa.plot_settings(font="DejaVu Sans", weight_bold=False)
    sns.barplot(x="Classes", y="Values", data=data, palette=colors, hue="Classes")
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_5_output_9_0.png


If you only want to change the ``font`` type, you can set
``adjust_only_font=True``:

.. code:: ipython2

    aa.plot_settings(adjust_only_font=True, font="DejaVu Sans")
    sns.barplot(x="Classes", y="Values", data=data, palette=colors, hue="Classes")
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_6_output_11_0.png


Grid can be enabled by ``grid=True`` and the ``grid-axis`` can be ‘x’,
‘y’, or ‘both’:

.. code:: ipython2

    aa.plot_settings(grid=True, grid_axis="both")
    sns.barplot(x="Classes", y="Values", data=data, palette=colors, hue="Classes")
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_7_output_13_0.png


The x- any y-ticks can be easily adjusted. Remove all ticks by
``no_ticks=True``:

.. code:: ipython2

    aa.plot_settings(no_ticks=True)
    sns.barplot(x="Classes", y="Values", data=data, palette=colors, hue="Classes")
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_8_output_15_0.png


Or shorten all via ``short_ticks=True``:

.. code:: ipython2

    aa.plot_settings(short_ticks=True)
    sns.barplot(x="Classes", y="Values", data=data, palette=colors, hue="Classes")
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_9_output_17_0.png


This can as well be applied separately for the x- and y-axis:

.. code:: ipython2

    aa.plot_settings(short_ticks_x=True, no_ticks_y=True)
    sns.barplot(x="Classes", y="Values", data=data, palette=colors, hue="Classes")
    sns.despine()
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_10_output_19_0.png


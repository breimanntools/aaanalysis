AAanalysis provides the capability to create a legend independently of
the plotted object, enabling flexible legend creation using the
``aa.plot_legend()`` method. First, create a default seaborn plot:

.. code:: ipython2

    import matplotlib.pyplot as plt
    import seaborn as sns
    import aaanalysis as aa
    data = {'Classes': ['A', 'B', 'C'], 'Values': [23, 27, 43]}
    colors = aa.plot_get_clist()
    aa.plot_settings()
    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()

You then just need to provide a color dictionary (``dict_color``):

.. code:: ipython2

    list_cat = ["A", "B", "C"]
    dict_color = dict(zip(list_cat, colors))
    aa.plot_legend(dict_color=dict_color)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_1_output_3_0.png


You can adjust the location by using the ``loc`` parameter:

.. code:: ipython2

    list_cat = ["A", "B", "C"]
    dict_color = dict(zip(list_cat, colors))
    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, loc="center left")
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_2_output_5_0.png


You can adjust the number of columns (``n_cols``) or the y-axis position
(``y=1.1``, top):

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, y=1.1)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_3_output_7_0.png


Setting the legend on the right middle can be achieved by using ``x=1``
and ``y=0.5``.

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=1, y=0.5, x=1)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_4_output_9_0.png


Categories can be independently labeled using ``labels``:

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    labels = ["Cat A", "Cat B", "Cat C"]
    aa.plot_legend(dict_color=dict_color, ncol=1, y=0.5, x=1, labels=labels)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_5_output_11_0.png


The legend can be directly set in the left under the plot using
``loc_out=True``:

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_6_output_13_0.png


We provide four spacing and length options. First, ``labelspacing``
(default=0.2):

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, labelspacing=1)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_7_output_15_0.png


Second, ``columnspacing`` (default=1.0):

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, columnspacing=5)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_8_output_17_0.png


Third, spacing between handles (i.e., the colored legend boxes) and the
legend text labels using ``handletextpad`` (default=0.8):

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, handletextpad=0)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_9_output_19_0.png


Fourth, the length of the legend handles can be adjusted using
``handlelength`` (default=2):

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, handlelength=1, handletextpad=0)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_10_output_21_0.png


The ``title`` of the legend can be set and automatically aligned to the
left using ``title_align_left=True``:

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, title="Categories", title_align_left=True)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_11_output_23_0.png


Adjust the general fontsize and weight using ``fontsize`` (default=None,
i.e., default fontsize of matplotlib or fontsize adjusted by
``aa.plot_settings()``) and ``fontsize_weight`` (default=‘normal’):

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, title="Categories", title_align_left=True, fontsize=25, fontsize_weight="bold")
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_12_output_25_0.png


Or you can adjust only the font of the legend title using
``fontsize_title`` and ``title_weight``:

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, title="Categories", title_align_left=True, fontsize_title=25, title_weight="bold")
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_13_output_27_0.png


The edges of the handles can be adjusted using linewidth (``lw``) and
``edgecolor``:

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, title="Categories", title_align_left=True, fontsize_title=25, title_weight="bold", lw=2, edgecolor="black")
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_14_output_29_0.png


The legend handle (here called ‘markers’) can be adjusted using
``markers`` (e.g., ‘-’ for lines) and ``marker_size`` (default=10):

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, title="Categories", marker='*', marker_size=15)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_15_output_31_0.png


Lines can be selected using ``marker='-'`` if linewidth (``lw``,
default=0) is >0:

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, title="Categories", marker='-', lw=2)
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_16_output_33_0.png


The style of the lines can be adjusted for each line individually by
using ``linestyle``:

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    sns.despine()
    aa.plot_legend(dict_color=dict_color, ncol=2, loc_out=True, labels=labels, title="Categories", marker='-', lw=2, linestyle=["-", ":", "--"])
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_17_output_35_0.png


Finally, you can add a ``hatch`` (i.e., filling pattern of markers) and
adjust their ``hatchcolor``:

.. code:: ipython2

    ax = sns.barplot(x='Classes', y='Values', data=data, palette=colors, hue="Classes", legend=False)
    # Create hatches
    hatches = ['/', '.', '.']
    for bars, hatch in zip(ax.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    sns.despine()
    dict_color = {"Group 1": "black", "Group 2": "black"}
    aa.plot_legend(dict_color=dict_color, ncol=2, y=1.1, hatch=["/", "."])
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_legend_18_output_37_0.png


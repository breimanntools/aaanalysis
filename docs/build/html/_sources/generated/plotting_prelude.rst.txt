Plotting prelude
================

These are some of our utility plotting functions to make
publication-ready visualizations with a view extra lines of code.

Let us first make all imports and create some data

.. code:: ipython2

    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    
    data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 27, 43]}

The default seaborn output with removed top and right spines looks like
this:

.. code:: ipython2

    sns.barplot(x='Classes', y='Values', data=data)
    sns.despine()
    plt.title("Seaborn default")
    plt.tight_layout()
    plt.show()



.. image:: output_3_0.png


Just call our ``aa.plot_setting`` function with our optimized color set
to get this:

.. code:: ipython2

    import aaanalysis as aa
    colors = aa.plot_get_clist()
    aa.plot_settings()
    sns.barplot(x='Classes', y='Values', data=data, palette=colors)
    sns.despine()
    plt.title("Adjusted by AAanalysis 1")
    plt.tight_layout()
    plt.show()



.. image:: output_5_0.png


The settings can be easily adjusted and colors are provided for up to 9
classes:

.. code:: ipython2

    data = {'Classes': ['Class A', 'Class B', 'Class C', "Class D", "Class E"], 'Values': [23, 27, 43, 9, 14]}
    colors = aa.plot_get_clist(n_colors=5)
    aa.plot_settings(no_ticks_x=True, short_ticks_y=True, grid=True, grid_axis="y")
    sns.barplot(x='Classes', y='Values', data=data, palette=colors)
    sns.despine()
    plt.title("Adjusted by AAanalysis 2")
    plt.tight_layout()
    plt.show()



.. image:: output_7_0.png


Retrieve the set font size and create an independent legend like this:

.. code:: ipython2

    data = {'Classes': ['Class A', 'Class B', 'Class C', "Class D", "Class E"], 'Values': [23, 27, 43, 9, 14]}
    colors = aa.plot_get_clist(n_colors=5)
    aa.plot_settings(no_ticks_x=True, short_ticks_y=True)
    sns.barplot(x='Classes', y='Values', data=data, palette=colors, hatch=["/", "/", "/", ".", "."])
    fontsize = aa.plot_gcfs()
    sns.despine()
    plt.title("Adjusted by AAanalysis 3")
    dict_color = {"Group 1": "black", "Group 2": "black"}
    aa.plot_set_legend(dict_color=dict_color, ncol=1, x=0.7, y=0.9, hatch=["/", "."])
    plt.tight_layout()
    plt.show()



.. image:: output_9_0.png



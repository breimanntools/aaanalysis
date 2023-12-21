You can retrieve a list of n colors by using the ‘n_colors’ parameter:

.. code:: ipython2

    import matplotlib.pyplot as plt
    import seaborn as sns
    import aaanalysis as aa
    colors = aa.plot_get_clist(n_colors=2)
    sns.palplot(colors)
    plt.show()



.. image:: examples/plot_get_clist_1_output_1_0.png


We assembled 8 different color lists for 2 to 9 colors:

.. code:: ipython2

    for n in range(3, 9):
        colors = aa.plot_get_clist(n_colors=n)
        sns.palplot(colors)
        plt.show()



.. image:: examples/plot_get_clist_2_output_3_0.png



.. image:: examples/plot_get_clist_3_output_3_1.png



.. image:: examples/plot_get_clist_4_output_3_2.png



.. image:: examples/plot_get_clist_5_output_3_3.png



.. image:: examples/plot_get_clist_6_output_3_4.png



.. image:: examples/plot_get_clist_7_output_3_5.png


For more than 9 colors, we provide the ‘husl’ default color palette of
the :func:``seaborn.color_palette`` fuction:

.. code:: ipython2

    for n in [10, 15, 20]:
        colors = aa.plot_get_clist(n_colors=n)
        sns.palplot(colors)
        plt.show()



.. image:: examples/plot_get_clist_8_output_5_0.png



.. image:: examples/plot_get_clist_9_output_5_1.png



.. image:: examples/plot_get_clist_10_output_5_2.png


A default seaborn barplot can be created as follows:

.. code:: ipython2

    import matplotlib.pyplot as plt
    import seaborn as sns
    data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 27, 43]}
    args = dict(x="Classes", y="Values", data=data)
    sns.barplot(**args)
    sns.despine()
    plt.title("Seaborn default")
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_1_output_1_0.png


Adjust plots with AAanalysis using ``aa.plot_settings()``:

.. code:: ipython2

    import aaanalysis as aa
    aa.plot_settings()
    sns.barplot(**args)
    sns.despine()
    plt.title("Default adjusted")
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_2_output_3_0.png


You can add our default colors using the ``aa.plot_get_clist()`` method:

.. code:: ipython2

    colors = aa.plot_get_clist(n_colors=3)
    sns.barplot(**args, palette=colors, hue="Classes")
    sns.despine()
    plt.title("Adjusted")
    plt.tight_layout()
    plt.show()



.. image:: examples/plot_settings_3_output_5_0.png



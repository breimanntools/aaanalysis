The ``Bayesian Information Criterion (BIC)`` [-∞, ∞] for a given set of
clusters in the dataset ``X`` can be computed using the
``comp_bic_score()`` function. As introduced in [Breimann24a]\_, the BIC
was adjusted so that higher values indicate better clustering results:

.. code:: ipython2

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import aaanalysis as aa
    
    # Generate random data for two groups
    group_blue = np.random.normal(-2, 0.5, 1000)
    group_red = np.random.normal(2, 0.5, 1000)
    
    # Combine data into a single dataset and reshape it
    X = np.hstack([group_blue, group_red]).reshape(-1, 1)  # Reshape to 2D array
    labels = np.array([1]*1000 + [0]*1000)
    bic_score = round(aa.comp_bic_score(X, labels), 3)
    
    # Plot
    aa.plot_settings()
    sns.histplot(group_blue, color="blue", kde=True, label='Group 1', alpha=0.5)
    sns.histplot(group_red, color="red", kde=True, label='Group 2', alpha=0.5)
    plt.title(f"BIC = {bic_score} (Perfect labeling)")
    sns.despine()
    plt.show()



.. image:: examples/comp_bic_score_1_output_1_0.png


Labeling both groups randomly is dramatically decreasing the
``bic_score``:

.. code:: ipython2

    group_blue = np.random.normal(-2, 0.5, 1000)
    group_red = np.random.normal(2, 0.5, 1000)
    X = np.hstack([group_blue, group_red]).reshape(-1, 1)  # Reshape to 2D array
    labels = np.array([1]*1000 + [0]*1000)
    np.random.shuffle(labels)
    bic_score = round(aa.comp_bic_score(X, labels), 3)
    
    # Plot
    aa.plot_settings()
    sns.histplot(group_blue, color="blue", kde=True, label='Group 1', alpha=0.5)
    sns.histplot(group_red, color="red", kde=True, label='Group 2', alpha=0.5)
    plt.title(f"BIC = {bic_score} (Random labeling)")
    sns.despine()
    plt.show()



.. image:: examples/comp_bic_score_2_output_3_0.png


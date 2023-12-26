To demonstrate the ``AAclustPlot().eval()`` method, we create an example
dataset:

.. code:: ipython2

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import aaanalysis as aa
    aa.options["verbose"] = False
    # Obtain example scale dataset 
    df_scales = aa.load_scales()
    X = df_scales.T
    # Fit AAclust model and retrieve labels for evaluation
    aac = aa.AAclust()
    list_labels = [aac.fit(X, n_clusters=n).labels_ for n in [3, 5, 10, 25, 50, 100, 150, 200]]
    df_eval = aac.eval(X, list_labels=list_labels)

And can visualize now all results of the \`df_eval`\`. The clustering
results are ranked in from top to down by the average ranking over all
three quality measures (BIC, SC, and CH):

.. code:: ipython2

    aac_plot = aa.AAclustPlot(model_class=PCA)
    fig, ax = aac_plot.eval(df_eval=df_eval)
    plt.show()



.. image:: examples/aac_plot_eval_1_output_3_0.png


You can adjust the x-axis limits of the three quality measures using the
``dict_xlims`` parameter:

.. code:: ipython2

    dict_xlims = dict(n_clusters=(0, 250), BIC=(-7500, 7500), CH=(0, 200), SC=(0, 0.4))
    aac_plot.eval(df_eval=df_eval, dict_xlims=dict_xlims)
    plt.show()



.. image:: examples/aac_plot_eval_2_output_5_0.png


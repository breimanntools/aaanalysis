Get default arguments for all splits types (Segment, Pattern, PeriodicPattern):

.. code:: ipython2

    import aaanalysis as aa
    sf = aa.SequenceFeature()
    split_kws = sf.get_split_kws()
    split_kws




.. parsed-literal::

    {'Segment': {'n_split_min': 1, 'n_split_max': 15},
     'Pattern': {'steps': [3, 4], 'n_min': 2, 'n_max': 4, 'len_max': 15},
     'PeriodicPattern': {'steps': [3, 4]}}



You can also retrieve arguments for specifc split types ('Segment', 'Pattern', 'PeriodicPattern'):

.. code:: ipython2

    split_kws = sf.get_split_kws(split_types=["Segment", "Pattern"])
    split_kws




.. parsed-literal::

    {'Segment': {'n_split_min': 1, 'n_split_max': 15},
     'Pattern': {'steps': [3, 4], 'n_min': 2, 'n_max': 4, 'len_max': 15}}



The arguments for each split type can be adjusted. For 'Segments', their minimum and maximum length can be chagned:

.. code:: ipython2

    split_kws = sf.get_split_kws(split_types="Segment", n_split_min=5, n_split_max=10)
    split_kws




.. parsed-literal::

    {'Segment': {'n_split_min': 5, 'n_split_max': 10}}



For 'PeriodicPattern', the step size of each odd and even step can be specified as follows:

.. code:: ipython2

    split_kws = sf.get_split_kws(split_types="PeriodicPattern", steps_periodicpattern=[5, 10])
    split_kws




.. parsed-literal::

    {'PeriodicPattern': {'steps': [5, 10]}}



And for 'Patterns', the step size, the minimum and maximum number of steps, and the maximum residue size of the pattern can be adjusted: 

.. code:: ipython2

    split_kws = sf.get_split_kws(split_types="Pattern", steps_pattern=[5, 10], n_min=3, n_max=5, len_max=30)
    split_kws




.. parsed-literal::

    {'Pattern': {'steps': [5, 10], 'n_min': 3, 'n_max': 5, 'len_max': 30}}



Learning from unbalanced and small data
=======================================

Unbalanced and small datasets are everywhere in life science ....

In a standard binary classification setup, data with positive (1) and negative (0) labels are provided, which can be
used for training by machine learning models. If only a view samples of the negative class exist, data augmentation
techniques (e.g., SMOTE) can be used to extend the negative dataset by artificially generated sequences. Such approaches
are very popular for deep learning-based image recognition, but not feasible for protein sequence prediction tasks
because slight amino acid mutations (sequence alterations or perturbations) can already have dramatic biological effects.
Alternatively, negatives samples can be identified from unlabeled samples (2), which often exist in great quantities.
These unlabeled samples should be biologically as similar as possible to the positive class, beside not containing
the features distinguishing the positive from the negative class. For example, .

What is PU learning?
--------------------
Positive Unlabeled (PU) learning is a subfield of machine learning ...
Learning from unbalanced and small data
=======================================
In the life sciences, the prevalence of unbalanced and small datasets often leads to challenges in machine learning,
particularly when dealing with a rare class of interest. Typically, binary classification involves labeled positive
and negative data. However, in scenarios where negative samples are scarce, conventional data augmentation methods
like SMOTE may fall short due to generating artificial samples. This issue is especially pertinent in fields like
protein sequence prediction, where even minor alterations can lead to significant biological implications.
A practical solution is to harness the abundance of unlabeled data to identify negative samples, a strategy
that becomes essential when obtaining a balanced dataset is challenging.

What is PU learning?
--------------------
Positive-Unlabeled (PU) learning, a subfield of machine learning, is tailored for situations with only positive and
unlabeled data. It is gaining relevance in bioinformatics and similar fields, where negative data often remain
unlabeled and undiscovered. PU learning algorithms aim to identify negative data from unlabeled data based on statistical
comparison with the positive data, coupled with iterative learning strategies. This approach adeptly handles the
inherent data asymmetry in applications where labeled negatives are unattainable.

Challenges and Potential Biases in PU Learning
----------------------------------------------
PU learning faces specific challenges, that can be summarized as follows:

- **Homogeneity in Identified Negatives**: Similarity among identified negatives can indicate algorithm bias or
  training data diversity issues, potentially leading to overfitting.
- **Biased Negative Selection**: Selection methods for negatives can significantly influence model performance.
- **Overfitting to Positive Data**: Training predominantly on positive data risks failing to accurately identify true negatives.
- **Assumptions in Class Distribution**: Incorrect assumptions about class distribution can result in misclassification.
- **Data Quality**: The quality and representativeness of data are critical for model accuracy.

To enhance model robustness and accuracy, it is crucial to ensure a high diversity among identified negatives. This
diversity promotes better generalization, as models learn to recognize a broader spectrum of negative features.
Additionally, negatives near the decision boundary are particularly valuable for the model's ability to distinguish
between positives and negatives. Addressing these challenges requires careful algorithm selection, data preprocessing,
and diverse evaluation strategies.

Evaluation Strategies for PU Learning
-------------------------------------
Evaluating PU learning models in the absence of ground truth can be performed in two steps. First, the population
of identified negatives can be assessed regarding their statistical properties and compared against the other data
classes. Second, the prediction performance of the machine learning models trained on the obtained dataset can be assessed.
The availability of a minority ground-truth negative dataset can significantly aid this process.

**1. Statistical Evaluation of Identified Negatives**
This step involves analyzing the characteristics of the negatives identified by the PU learning model using
distinct statistical measures:

- **Homogeneity in Identified Negatives**: Examine the extent to which identified negatives are similar
  to each other, which can indicate potential biases or lack of diversity in the training data. This can be achieved
  by using:

  - *Coefficient of Variation (CV)* [0, ∞): Evaluates the variability within the identified negatives,
    with high CV (> 1) indicating more diversity and low CV (< 1) indicating homogeneity
  - *Entropy* [0, 1]: Measures the diversity within the identified negatives, where higher entropy signifies greater
    variability and lower entropy indicates homogeneity

- **Distribution Alignment**: Determine the similarity between the identified negatives and the other data classes,
  by assessing the alignment of their distributions. This is especially useful if a minority ground-truth dataset
  of negatives exists. To this end, various statistical measures can be used:

  - *Area Under the Curve (AUC)*: Assess the difference between two datasets, with a higher AUC indicating
    a higher discrimination. We use an adjusted AUC [-0.5, 0.5] such that 0 indicates no differences and the
    sign indicates which dataset comprises higher values.
  - *Kullback-Leibler Divergence* [0, ∞): Measures the divergence between the distributions of identified negatives
    and unlabeled data, with lower values (near 0) indicating better alignment.
  - *Mann-Whitney U Test*: Non-parametric test to compare distributions, useful for assessing if the identified negatives
    differ significantly from the positive or unlabeled data.

- **Assessing Reproducibility**: To evaluate the reproducibility of a PU learning algorithm, perform the method multiple
  times and compare the overlap in the resulting sets of identified negatives.

**2. Machine Learning Model Evaluation Strategies**
Once the negatives are identified, the following strategies can be employed to evaluate the trained models using
standard evaluation measures such as accuracy or balanced accuracy (preferred if dataset is still unbalanced):

- **Proxy Ground Truth**: Utilize a small, accurately labeled dataset as a proxy for ground truth to provide
  a limited yet insightful evaluation. If a minority ground truth negative dataset is available, you can train only
  using the known positive and identified negative samples, and evaluate on the ground truth negative data.
- **Consistency and Stability**: Assess the model’s consistency across multiple runs or data subsets,
  indicating reliability.
- **Confidence Analysis**: Evaluate the model's prediction confidence scores. High confidence may indicate effectiveness,
  but beware of overfitting or bias.
- **External Validation**: Use external knowledge or domain expertise to validate model predictions,
  ensuring they align with real-world scenarios.
- **Ablation and Sensitivity Analysis**: Conduct ablation studies to understand model dependencies, and test the
  model's robustness to data perturbations. You can assess the impact of removing or altering features or parameters,
  indicating model robustness.

Combining these strategies offers a comprehensive view of a PU learning model’s performance, especially in scenarios
lacking complete ground truth data.

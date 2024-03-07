.. _eval_pu_learning:

Evaluating PU Learning
======================
PU (Positive-Unlabeled) Learning addresses the problem of learning from only positive and unlabeled data, without
negative instances. This scenario, common in many real-world applications, necessitates innovative evaluation
strategies due to the lack of comprehensive ground truth. This section delves into evaluating PU learning models,
focusing on the statistical assessment of identified negatives and the overall predictive performance.
It highlights the importance of overcoming biases and challenges unique to PU learning to ensure model accuracy
and applicability, thus guiding the development of reliable and insightful machine learning solutions in unlabeled
data environments.


Challenges in PU Learning
-------------------------
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


1. Statistical Evaluation of Identified Negatives
#################################################
This step involves analyzing the characteristics of the negatives identified by the PU learning model using
distinct statistical measures:

**Homogeneity in Identified Negatives**: Examine the extent to which identified negatives are similar to each other, which can indicate potential biases
or lack of diversity in the training data, by using:

- **Standard Deviation (STD)**: Measures variability within the identified negatives, ideal for normally distributed data.
  A higher STD suggests more diversity, and a lower one indicates homogeneity. While sensitive to outliers, this
  sensitivity can be informative based on your data's nature.
- **Inter Quantile Range (IQR)**: Assesses the spread of the middle 50% of the identified negatives, offering robustness
  against outliers and suitability for both normally and non-normally distributed variables.  Higher IQR values signal
  more diversity, focusing on the dataset's central tendency rather than extremes

Using both STD and IQR offers a balanced view of data variability, blending STD's sensitivity to overall spread with
IQR's robustness to outliers, thereby enriching the reliability of homogeneity assessments

**Distribution Alignment**: Determine the similarity between the identified negatives and the other data classes,
by assessing the alignment of their distributions. This analysis is crucial when ground-truth negative data is available.
Key statistical measures include:

- **Area Under the Curve (AUC)**: Assess the difference between two datasets, with a higher AUC indicating
  a greater discrimination. We adjusted the AUC between [-0.5, 0.5] so that a value of 0 implies no difference,
  while the sign indicates which dataset has higher values.
- **Kullback-Leibler Divergence (KLD)** [0, ∞): Measures the divergence between the distributions of two datasets, with
  values near 0 indicating better alignment.

Using AUC and KLD together enriches distribution alignment analysis since AUC quantifies comparative discrimination,
while KLD directly measures distribution divergence.

**Assessing Reproducibility**: To evaluate the reproducibility of a PU learning algorithm, perform the method multiple
times and compare the overlap in the resulting sets of identified negatives.

2. Machine Learning Model Evaluation Strategies
###############################################
Once the negatives are identified, the following strategies can be employed to evaluate the trained models using
standard evaluation measures such as accuracy or balanced accuracy (preferred if dataset is still unbalanced):

- **Confidence Analysis and Ground Truth**: Assess the model's prediction confidence. High confidence suggests
  effectiveness, but be cautious of overfitting or bias. When available, use a minority ground-truth negative dataset
  for training with known positives and identified negatives, and evaluate against this ground truth.
- **External Validation**: Use external knowledge or domain expertise to validate model predictions and identified
  negatives, ensuring they align with real-world scenarios.
- **Consistency and Stability**: Evaluate the model's reliability by checking its consistency across multiple
  runs or subsets of data.
- **Ablation and Sensitivity Analysis**: Perform ablation studies to determine the model's dependence on specific
  features or parameters, and test its resilience to data variations. This includes assessing the effects of removing
  or modifying features or parameters to gauge model robustness.

These strategies collectively provide a thorough evaluation of PU learning models, particularly valuable in settings
lacking full ground-truth data.

"""
This is a script for the backend of the dPULearn.fit() method.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import math
from sklearn.decomposition import PCA

import aaanalysis.utils as ut

# II Main Functions
def get_neg_via_distance(X=None, labels=None, metric="euclidean", n_unl_to_neg=None,
                         label_neg=0, label_pos=1):
    """Identify distant samples from positive mean as reliable negatives based on a specified distance metric."""
    col_dif = f'{metric}_dif'
    col_dif_abs = f"{metric}_abs_dif"
    mask_pos = labels == label_pos
    mask_unl = labels != label_pos
    # Compute the distances to the average value of the positive datapoints
    dif_to_pos_mean = pairwise_distances(X[mask_pos], X, metric=metric).mean(axis=0)
    abs_dif_to_pos_mean = np.abs(dif_to_pos_mean)
    # Create a DataFrame with the distances
    df_pu = pd.DataFrame({col_dif: dif_to_pos_mean, col_dif_abs: abs_dif_to_pos_mean})
    # Select negatives based on largest average distance to positives
    top_indices = df_pu[mask_unl].sort_values(by=col_dif_abs).tail(n_unl_to_neg).index
    new_labels = labels.copy()
    new_labels[top_indices] = label_neg
    # Adjust df distance
    df_pu = df_pu.round(4)
    df_pu.insert(0, ut.COL_SELECTION_VIA, [metric if l == 0 else None for l in new_labels])
    return new_labels, df_pu


def get_neg_via_pca(X=None, labels=None, n_components=0.8, n_unl_to_neg=None,
                    label_neg=0, label_pos=1, **pca_kwargs):
    """Identify distant samples from positive mean as reliable negatives in PCA-compressed feature spaces."""
    # Principal component analysis
    pca = PCA(n_components=n_components, **pca_kwargs)
    pca.fit(X.T)
    list_exp_var = pca.explained_variance_ratio_
    columns_pca = [f"PC{n+1} ({round(exp_var*100, 1)}%)" for n, exp_var in enumerate(list_exp_var)]
    # Determine number of negatives based on explained variance
    _list_n_neg = [math.ceil(n_unl_to_neg * x / sum(list_exp_var)) for x in list_exp_var]
    _list_n_cumsum = np.cumsum(np.array(_list_n_neg))
    list_n_neg = [n for n, cs in zip(_list_n_neg, _list_n_cumsum) if cs <= n_unl_to_neg]
    if sum(list_n_neg) != n_unl_to_neg:
        list_n_neg.append(n_unl_to_neg - sum(list_n_neg))
    columns_pca = columns_pca[:len(list_n_neg)]
    # Create df_pu based on PCA components
    df_pu = pd.DataFrame(pca.components_.T[:, :len(columns_pca)], columns=columns_pca)
    # Get mean of positive data for each component
    mask_pos = labels == label_pos
    mask_unl = labels != label_pos
    pc_means = df_pu[mask_pos].mean(axis=0)
    # Select negatives based on absolute difference to mean of positives for each component
    df_pu.insert(0, ut.COL_SELECTION_VIA, None)  # New column to store the PC information
    new_labels = labels.copy()
    for col_pc, mean_pc, n in zip(columns_pca, pc_means, list_n_neg):
        col_abs_dif = f"{col_pc}_abs_dif"
        # Calculate absolute difference to the mean for each sample in the component
        df_pu[col_abs_dif] = np.abs(df_pu[col_pc] - mean_pc)
        # Sort and take top n indices
        top_indices = df_pu[mask_unl].sort_values(by=col_abs_dif).tail(n).index
        # Update labels and masks
        new_labels[top_indices] = label_neg
        mask_unl[top_indices] = False
        # Record the PC by which the negatives are selected
        df_pu.loc[top_indices, ut.COL_SELECTION_VIA] = col_pc.split(' ')[0]
    # Adjust df
    cols = [x for x in list(df_pu) if x != ut.COL_SELECTION_VIA]
    df_pu[cols] = df_pu[cols].round(4)
    return new_labels, df_pu


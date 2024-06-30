"""This is the backend for the ShapModel().add_sample_mean_dif() method"""
import numpy as np
import pandas as pd
import aaanalysis.utils as ut

def add_sample_mean_dif_(X, labels=None, label_ref=0, df_feat=None, drop=False,
                         sample_positions=None, names=None, group_average=False):
    """ Compute the feature value difference between selected samples and a reference group average,
     and include into feature DataFrame."""
    if drop:
        # Keep original mean_dif
        columns = [x for x in list(df_feat) if ut.COL_MEAN_DIF not in x or x in [ut.COL_MEAN_DIF, ut.COL_ABS_MEAN_DIF]]
        df_feat = df_feat[columns]
    # Compute mean of reference group
    ref_group_mask = np.asarray([l == label_ref for l in labels])
    ref_group_mean = X[ref_group_mask].mean(axis=0)
    # Generate column names based on 'names'
    if not group_average:
        col_names = [f'{ut.COL_MEAN_DIF}_{name}' for name in names]
    else:
        col_names = [f'{ut.COL_MEAN_DIF}_group' if not names else f'{ut.COL_MEAN_DIF}_{names}']
    # Compute the feature value mean differences
    mean_diffs = []
    if group_average:
        group_mean = X[sample_positions].mean(axis=0)
        mean_diffs.append(group_mean - ref_group_mean)
    else:
        for pos in sample_positions:
            sample_mean = X[pos]
            mean_diff = sample_mean - ref_group_mean
            mean_diffs.append(mean_diff)
    # Insert new columns into DataFrame
    df_feat_mean_diff = pd.DataFrame(data=np.array(mean_diffs).T, columns=col_names)
    df_feat = pd.concat([df_feat, df_feat_mean_diff], axis=1)
    return df_feat

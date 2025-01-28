"""
This is a script for the backend of the ShapModel.add_feat_impact() and ShapModel.add_feat_importance() methods.
"""
import numpy as np
import pandas as pd
import warnings

import aaanalysis.utils as ut


# I Helper Functions
def _abs_normalize(values=None):
    total_impact = sum(np.abs(values))
    values = [round(x / total_impact * 100, 2) for x in values]
    return values


# II Main Functions
def _drop_feat_columns(df_feat=None):
    """Drop all feature impact and feature importance columns"""
    f = lambda x: x not in [ut.COL_FEAT_IMPORT, ut.COL_FEAT_IMPORT_STD] and ut.COL_FEAT_IMPACT not in x
    columns = [x for x in list(df_feat) if f(x)]
    df_feat = df_feat[columns]
    return df_feat


def _comp_sample_shap_feat_impact(shap_values=None, i=None, normalize=True):
    """Compute the shap feature impact for the i-th sample"""
    shap_value_sample = shap_values[i]
    feat_impact = shap_value_sample / np.nansum(np.abs(shap_value_sample), axis=0) * 100
    if normalize:
        feat_impact = _abs_normalize(values=feat_impact)
    return feat_impact


def _comp_group_shap_feat_impact(shap_values=None, list_i=None, normalize=True, verbose=True):
    """Compute the shap feature impact for a group of samples"""
    shap_value_samples = shap_values[list_i]
    mean_shap_value_samples = shap_value_samples.mean(axis=0)
    std_shap_values_samples = shap_value_samples.std(axis=0)
    feat_impact = mean_shap_value_samples / np.nansum(np.abs(mean_shap_value_samples)) * 100
    feat_impact_std = std_shap_values_samples / np.nansum(np.abs(mean_shap_value_samples)) * 100
    if normalize:
        abs_sum_before_norm = sum(np.abs(feat_impact))
        feat_impact = _abs_normalize(values=feat_impact)
        abs_sum_after_norm = sum(np.abs(feat_impact))
        # Scale std (assumes linear scaling factor)
        feat_impact_std *= abs_sum_after_norm/abs_sum_before_norm
    if verbose:
        max_std, max_impact = round(max(np.abs(feat_impact_std)), 2), round(max(np.abs(feat_impact)), 2)
        if max_std > max_impact * 5:
            warnings.warn(f"Absolute maximum of 'feat_impact_std' ({max_std}) >> 'feat_impact' ({max_impact}). Grouping might be invalid.")
    return feat_impact, feat_impact_std


# II Main functions
# Feature importance
def comp_shap_feature_importance(shap_values=None, normalize=True):
    """
    Compute shap feature importance (absolute mean for all samples)
    See: https://github.com/slundberg/shap/issues/538
    """
    feat_importance = np.abs(shap_values).mean(axis=0) * 100
    if normalize:
        feat_importance = _abs_normalize(values=feat_importance)
    return feat_importance


def insert_shap_feature_importance(df_feat=None, feat_importance=None, drop=False):
    """Insert shap explainer-based feature importance"""
    df_feat = df_feat.copy()
    if drop:
        df_feat = _drop_feat_columns(df_feat=df_feat)
    args = dict(allow_duplicates=False)
    df_feat.insert(loc=len(df_feat.columns), column=ut.COL_FEAT_IMPORT, value=feat_importance, **args)
    return df_feat


# Feature impact
def comp_shap_feature_impact(shap_values, sample_positions=None, normalize=True, group_average=False, verbose=True):
    """
    Compute SHAP feature impact for different scenarios:
        a) For a single sample, returning its feature impact.
        b) For multiple samples, returning each sample's feature impact.
        c) For a group of samples, returning the group average feature impact.
    """
    # Single sample
    if isinstance(sample_positions, int):
        return _comp_sample_shap_feat_impact(shap_values, i=sample_positions, normalize=normalize)
    # Multiple samples
    elif isinstance(sample_positions, list) and not group_average:
        impacts = [_comp_sample_shap_feat_impact(shap_values, i, normalize) for i in sample_positions]
        return np.array(impacts)
    # Group average
    elif isinstance(sample_positions, list) and group_average:
        feat_impact, feat_impact_std = _comp_group_shap_feat_impact(shap_values, list_i=sample_positions, normalize=normalize,
                                                                    verbose=verbose)
        return np.array([feat_impact, feat_impact_std])


def insert_shap_feature_impact(df_feat=None, feat_impact=None, names=None, group_average=False, drop=False):
    """Insert shap explainer-based feature importance"""
    df_feat = df_feat.copy()
    if drop:
        df_feat = _drop_feat_columns(df_feat=df_feat)
    # Single sample or multiple samples
    if not group_average:
        col_names = [f'{ut.COL_FEAT_IMPACT}_{col_name}' for col_name in names]
    # Group average
    else:
        col_names = [f'{ut.COL_FEAT_IMPACT}_{names}', f'{ut.COL_FEAT_IMPACT_STD}_{names}']
    df_feat_impact = pd.DataFrame(data=feat_impact.T, columns=col_names)
    df_feat = pd.concat([df_feat, df_feat_impact], axis=1)
    return df_feat

"""
This is a script for the backend of the ShapModel.add_feat_impact() and ShapModel.add_feat_importance() methods.
"""
import numpy as np
import aaanalysis.utils as ut

# I Helper Functions
def _abs_normalize(values=None):
    total_impact = sum(np.abs(values))
    values = [round(x / total_impact * 100, 2) for x in values]
    return values


# II Main Functions
def _comp_sample_shap_feat_impact(shap_values=None, i=None, normalize=True):
    """Compute the shap feature impact for the i-th sample"""
    shap_value_sample = shap_values[i]
    feat_impact = shap_value_sample / np.nansum(np.abs(shap_value_sample), axis=0) * 100
    if normalize:
        feat_impact = _abs_normalize(values=feat_impact)
    return feat_impact


def _comp_group_shap_feat_impact(shap_values=None, list_i=None, normalize=True):
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
        # Scale std
        feat_impact_std *= abs_sum_after_norm/abs_sum_before_norm
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
        columns = [x for x in list(df_feat) if x not in [ut.COL_FEAT_IMPORT, ut.COL_FEAT_IMPORT_STD]]
        df_feat = df_feat[columns]
    args = dict(allow_duplicates=False)
    df_feat.insert(loc=len(df_feat.columns), column=ut.COL_FEAT_IMPORT, value=feat_importance, **args)
    return df_feat


# Feature impact
def comp_shap_feature_impact(shap_values, pos=None, normalize=True, group_average=False):
    """
    Compute SHAP feature impact for different scenarios:
        a) For a single sample, returning its feature impact.
        b) For multiple samples, returning each sample's feature impact.
        c) For a group of samples, returning the group average feature impact.
    """
    # Case a: Single sample
    if isinstance(pos, int):
        return _comp_sample_shap_feat_impact(shap_values, i=pos, normalize=normalize)
    # Case b: Multiple samples
    elif isinstance(pos, list) and not group_average:
        impacts = [_comp_sample_shap_feat_impact(shap_values, i, normalize) for i in pos]
        return np.array(impacts)
    # Case c: Group average
    elif isinstance(pos, list) and group_average:
        return _comp_group_shap_feat_impact(shap_values, list_i=pos, normalize=normalize)


def insert_shap_feature_impact(df_feat=None, feat_impact=None, pos=None, names=None, drop=False):
    """Insert shap explainer-based feature importance"""
    df_feat = df_feat.copy()
    if drop:
        columns = [x for x in list(df_feat) if ut.COL_FEAT_IMPACT not in x]
        df_feat = df_feat[columns]
    for n, col_name in zip(pos, names):
        column_name = f'{ut.COL_FEAT_IMPACT}_{col_name}'
        df_feat[column_name] = feat_impact[n] if isinstance(pos, list) else feat_impact
    return df_feat

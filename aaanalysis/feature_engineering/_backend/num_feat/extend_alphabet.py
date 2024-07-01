"""
This is a script for the backend of the NumericalFeature.extend_alphabet() method.
"""


# II Main Functions
def extend_alphabet_(df_scales=None, new_letter=None, value_type="mean"):
    """Extend amino acid alphabet of df_scales by new letter."""
    # Compute the statistic for each scale
    if value_type == "min":
        new_values = df_scales.min()
    elif value_type == "mean":
        new_values = df_scales.mean()
    elif value_type == "median":
        new_values = df_scales.median()
    else:
        new_values = df_scales.max()
    # Add the new letter to the DataFrame
    df_scales.loc[new_letter] = new_values
    return df_scales

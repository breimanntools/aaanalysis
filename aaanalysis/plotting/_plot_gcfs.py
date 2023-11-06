"""
This is a script for getting current font size of figures.
"""
import seaborn as sns

# Main function
def plot_gcfs(option='font.size'):
    """
    Gets current font size (or axes linewdith).

    This font size can be set by :func:`plot_settings` function.

    Examples
    --------
    Here are the default colors used in CPP and CPP-SHAP plots:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import seaborn as sns
        >>> import aaanalysis as aa
        >>> data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 27, 43]}
        >>> colors = aa.plot_get_clist()
        >>> aa.plot_settings()
        >>> sns.barplot(y='Classes', x='Values', data=data, palette=colors, hue="Classes", legend=False)
        >>> sns.despine()
        >>> plt.title("Two points bigger title", size=aa.plot_gcfs()+2)
        >>> plt.tight_layout()
        >>> plt.show()

    See Also
    --------
    - Our `Plotting Prelude <plotting_prelude.html>`_.
    """
    allowed_options = ["font.size", "axes.linewidth"]
    if option not in allowed_options:
        return ValueError(f"'option' should be one of following: {allowed_options}")
    # Get the current plotting context
    current_context = sns.plotting_context()
    option_value = current_context[option]  # Typically font_size
    return option_value

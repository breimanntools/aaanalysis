"""
This is a script for getting current font size of figures.
"""
import seaborn as sns

# Main function
def plot_gcfs():
    """
    Gets current font size.

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
        >>> sns.barplot(y='Classes', x='Values', data=data, palette=colors)
        >>> sns.despine()
        >>> plt.title("Big Title (+4 bigger than rest)", size=aa.plot_gcfs()+4)
        >>> plt.tight_layout()
        >>> plt.show()

    See Also
    --------
    * Our `Plotting Prelude <plotting_prelude.html>`_.
    """
    # Get the current plotting context
    current_context = sns.plotting_context()
    font_size = current_context['font.size']
    return font_size

"""
This is a script for getting current font size of figures.
"""
import seaborn as sns


# Main function
def plot_gcfs(option: str = 'font.size') -> int:
    """
    Get the current font size (or axes linewidth).

    This font size can be set by :func:`plot_settings` function.

    Parameters
    ----------
    option : str, default='font.size'
        Figure setting to get default value from. Either 'font.size' or 'axes.linewidth'

    Returns
    -------
    option_value : int
        Numerical value for selected option.

    See Also
    --------
    * `Plotting Prelude <plotting_prelude.html>`_.

    Examples
    --------
    .. include:: examples/plot_gcfs.rst
    """
    # Check input
    allowed_options = ["font.size", "axes.linewidth"]
    if option not in allowed_options:
        raise ValueError(f"'option' should be one of: {allowed_options}")
    # Get the current plotting context
    current_context = sns.plotting_context()
    option_value = current_context[option]
    return option_value

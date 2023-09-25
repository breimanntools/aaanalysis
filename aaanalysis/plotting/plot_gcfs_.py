"""
This is a script for getting current font size of figures.
"""
import seaborn as sns

# Main function
def plot_gcfs():
    """Get current font size, which is set by :func:`plot_settings` function."""
    # Get the current plotting context
    current_context = sns.plotting_context()
    font_size = current_context['font.size']
    return font_size

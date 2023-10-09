"""
This is a script for internal plotting utility functions used in the backend.
"""
import seaborn as sns

# Main function
def plot_gco(option='font.size', show_options=False):
    """Get current option from plotting context"""
    current_context = sns.plotting_context()
    if show_options:
        print(current_context)
    option_value = current_context[option]  # Typically font_size
    return option_value

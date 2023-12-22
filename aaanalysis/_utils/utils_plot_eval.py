"""
This is a script for the backend of evaluation plotting functions.
"""
import seaborn as sns
import matplotlib.ticker as mticker


# I Helper Functions



# Plotting helper functions
def adjust_spines(ax=None):
    """Adjust spines to be in middle if data range from <0 to >0"""
    min_val, max_val = ax.get_xlim()
    if max_val > 0 and min_val >= 0:
        sns.despine(ax=ax)
    else:
        sns.despine(ax=ax, left=True)
        current_lw = ax.spines['bottom'].get_linewidth()
        ax.axvline(0, color='black', linewidth=current_lw)
        val = max([abs(min_val), abs(max_val)])
        ax.set_xlim(-val, val)
    return ax


def x_ticks_0(ax):
    """Apply custom formatting for x-axis ticks."""
    def custom_x_ticks(x, pos):
        """Format x-axis ticks."""
        return f'{x:.2f}' if x else f'{x:.0f}'
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(custom_x_ticks))

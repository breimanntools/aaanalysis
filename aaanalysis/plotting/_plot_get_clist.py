"""
Plotting utility function to obtain AAanalysis color list.
"""
from typing import List
from aaanalysis import utils as ut



# II Main function
def plot_get_clist(n_colors: int = 3) -> List[str]:
    """
    Returns list of 2 to 9 colors.

    This fuctions returns one of eight different colorl lists optimized
    for appealing visualization of categories.

    Parameters
    ----------
    n_colors
        Number of colors. Must be between 2 and 9.
    Returns
    -------
    list
        List with colors given as matplotlib color names.

    Examples
    --------
    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import seaborn as sns
        >>> import aaanalysis as aa
        >>> colors = aa.plot_get_clist(n_colors=3)
        >>> data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [10, 23, 33]}
        >>> aa.plot_settings()
        >>> sns.barplot(data=data, x='Classes', y='Values', palette=colors, hue="Classes", legend=False)
        >>> plt.show()

    See Also
    --------
    - The example notebooks in `Plotting Prelude <plotting_prelude.html>`_.
    - `Matplotlib color names <https://matplotlib.org/stable/gallery/color/named_colors.html>`_
    - :func:`seaborn.color_palette` function to generate a color palette in seaborn.
    """
    # Check input
    ut.check_number_range(name="n_colors", val=n_colors, min_val=2, max_val=9, just_int=True)

    # Base lists
    list_colors_3_to_4 = ["tab:gray", "tab:blue", "tab:red", "tab:orange"]
    list_colors_5_to_6 = ["tab:blue", "tab:cyan", "tab:gray","tab:red",
                          "tab:orange", "tab:brown"]
    list_colors_8_to_9 = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                          "tab:gray", "gold", "tab:cyan", "tab:brown",
                          "tab:purple"]
    # Two classes
    if n_colors == 2:
        return ["tab:blue", "tab:red"]
    # Control/base + 2-3 classes
    elif n_colors in [3, 4]:
        return list_colors_3_to_4[0:n_colors]
    # 5-7 classes (gray in middle as visual "breather")
    elif n_colors in [5, 6]:
        return list_colors_5_to_6[0:n_colors]
    elif n_colors == 7:
        return ["tab:blue", "tab:cyan", "tab:purple", "tab:gray",
                "tab:red", "tab:orange", "tab:brown"]
    # 8-9 classes (colors from scale categories)
    elif n_colors in [8, 9]:
        return list_colors_8_to_9[0:n_colors]


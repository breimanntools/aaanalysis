"""
This is a script for the backend utility functions for the CPPPlot class.
"""

# I Helper Functions


# II Main Functions
def get_sorted_list_cat_(df_cat=None, list_cat=None, col_cat=None):
    """Get list of categories/subcategories sorted as in df_cat"""
    list_cat_ = list(df_cat[col_cat].drop_duplicates())
    sorted_cat = [x for x in list_cat_ if x in list_cat]
    return sorted_cat

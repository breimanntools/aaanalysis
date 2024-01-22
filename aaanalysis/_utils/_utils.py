"""This is a script for utility functions for utility functions."""

def add_str(str_error=None, str_add=None):
    """Add additional error message 'str_add' to default error message ('add_str')"""
    if str_add:
        str_error += "\n " + str_add
    return str_error

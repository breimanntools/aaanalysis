"""
This a script for general decorators used in AAanalysis
"""
import warnings
import traceback
import inspect
from sklearn.exceptions import ConvergenceWarning
import functools
import re


# Helper functions


# Document common interfaces
import re
from functools import wraps


# TODO remove since signature is not shown in runtime if used!!
# TODO use runtime decorator only for internal methods since they destroy the signature for some IDEs
def doc_params(**kwargs):
    def decorator(func):
        # If the docstring is None (e.g., if -OO was passed to the interpreter), return the unchanged function
        if func.__doc__ is None:
            return func

        # Wraps is used to copy metadata from the original function to the wrapper function
        @wraps(func)
        def wrapper(*args, **wrapper_kwargs):
            return func(*args, **wrapper_kwargs)

        # Regular expression to find replacement fields and their indentation
        pattern = re.compile(r'(?P<indent>\s*)\{(?P<key>\w+)\}')

        # Function to adjust indentation
        def adjust_indent(match):
            key = match.group('key')
            indent = match.group('indent')
            if key in kwargs:
                # Split the replacement string into lines and indent each one
                replacement_lines = kwargs[key].strip().split('\n')
                indented_replacement = '\n'.join(indent + line for line in replacement_lines)
                return indented_replacement
            else:
                # Key not provided in kwargs, keep original
                return match.group(0)

        # Replace all matching strings in doc
        wrapper.__doc__ = pattern.sub(adjust_indent, wrapper.__doc__)

        return wrapper

    return decorator


# Catch Runtime
class CatchRuntimeWarnings:
    """Context manager to catch RuntimeWarnings and store them in a list."""
    def __enter__(self):
        self._warn_list = []
        self._other_warnings = []
        self._showwarning_orig = warnings.showwarning
        warnings.showwarning = self._catch_warning
        return self

    def __exit__(self, exc_type, exc_value, tb):
        warnings.showwarning = self._showwarning_orig
        # Re-issue any other warnings that were caught but not RuntimeWarning
        for warn_message, warn_category, filename, lineno in self._other_warnings:
            warnings.warn_explicit(warn_message, warn_category, filename, lineno)

    def _catch_warning(self, message, category, filename, lineno, file=None, line=None):
        if category == RuntimeWarning:
            line_content = traceback.format_list([(filename, lineno, "", line)])[0].strip()
            warning_msg = f"{message}: {line_content.split(', in')[1]}"
            self._warn_list.append(warning_msg)
        else:
            # Store other warnings for re-issuing later
            self._other_warnings.append((message, category, filename, lineno))

    def get_warnings(self):
        return self._warn_list


def catch_runtime_warnings():
    """Decorator to catch RuntimeWarnings and store them in a list.

    Returns
    -------
    decorated_func : method
        The decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with CatchRuntimeWarnings() as crw:
                result = func(*args, **kwargs)
            if crw.get_warnings():
                list_warnings = crw.get_warnings()
                n = len(list_warnings)
                summary_msg = f"The following {n} 'RuntimeWarnings' were caught:\n" + "\nRuntimeWarning: ".join(crw.get_warnings())
                warnings.warn(summary_msg, RuntimeWarning)
            return result
        return wrapper

    return decorator


# Catch convergence
class ClusteringConvergenceException(Exception):
    def __init__(self, message, distinct_clusters):
        super().__init__(message)
        self.distinct_clusters = distinct_clusters


def catch_convergence_warning():
    """Decorator to catch ConvergenceWarnings and raise custom exceptions.

    Returns
    -------
    decorated_func : method
        The decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings(record=True) as w:
                # Trigger the "always" behavior for ConvergenceWarning
                warnings.simplefilter("always", ConvergenceWarning)
                result = func(*args, **kwargs)  # Call the original function

                # Check if the warning is the one we're interested in
                for warn in w:
                    if issubclass(warn.category, ConvergenceWarning):
                        message = str(warn.message)
                        if "Number of distinct clusters" in message:
                            distinct_clusters = int(message.split("(")[1].split(")")[0].split()[0])
                            raise ClusteringConvergenceException(f"Process stopped due to ConvergenceWarning.", distinct_clusters)
            return result
        return wrapper

    return decorator


# Catch invalid division (could be added to AAclust().comp_medoids())
class InvalidDivisionException(Exception):
    pass


def catch_invalid_divide_warning():
    """Decorator to catch specific RuntimeWarnings related to invalid division
       and raise custom exceptions.

    Returns
    -------
    decorated_func : method
        The decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with CatchRuntimeWarnings() as crw:
                result = func(*args, **kwargs)
            if crw.get_warnings():
                raise InvalidDivisionException(f"\nError due to RuntimeWarning: {crw.get_warnings()[0]}")
            return result
        return wrapper
    return decorator



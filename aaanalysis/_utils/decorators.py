"""
This a script for general decorators used in AAanalysis.
# Dev: use runtime decorator only for internal methods since they destroy the signature for some IDEs
# Dev: use backend processing error decorator only for backend
"""
import warnings
import traceback
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
import functools
import re
import textwrap

# Helper functions
def _build_deprecation_message(name=None, reason=None, version_removed=None):
    """Assemble the user-facing DeprecationWarning text for a deprecated symbol."""
    msg = f"'{name}' is deprecated"
    if version_removed is not None:
        msg += f" and will be removed in version {version_removed}"
    msg += "."
    if reason is not None:
        msg += f" {reason}"
    return msg


def _prepend_deprecation_note(doc=None, reason=None, version_removed=None):
    """Prepend a ``.. admonition:: Deprecated`` block to an existing docstring.

    Sphinx renders it as a highlighted box in the API docs, so the deprecation is
    visible both at call time (the warning) and in the rendered documentation. An
    ``admonition`` (rather than the ``.. deprecated::`` directive) is used on
    purpose: the latter requires a "deprecated since" version argument, whereas we
    track the *removal* target — and an argument-less ``.. deprecated::`` is a
    Sphinx build error. The body is always non-empty so the directive is valid.
    """
    body = []
    if reason is not None:
        body.append(reason)
    if version_removed is not None:
        body.append(f"Scheduled for removal in version {version_removed}.")
    if not body:
        body.append("This API is deprecated and will be removed in a future release.")
    # Indent every physical line (a multi-line ``reason`` must stay inside the
    # directive body, or Sphinx drops the continuation lines). textwrap.indent
    # leaves blank lines un-prefixed, so no trailing whitespace creeps in.
    indented = textwrap.indent("\n".join(body), "   ")
    block = f".. admonition:: Deprecated\n\n{indented}"
    if not doc:
        return block + "\n"
    return f"{block}\n\n{doc}"


# Deprecation
def deprecated(reason=None, version_removed=None):
    """Mark a public function, method, or class as deprecated (semver policy).

    Emits a :class:`DeprecationWarning` whenever the wrapped callable is invoked
    (for a class, on instantiation) and prepends a ``.. admonition:: Deprecated``
    note to its docstring. Per the project's strict-semver policy, a symbol in
    ``aaanalysis.__all__`` must ship at least one minor release carrying this
    decorator before it is renamed or removed.

    Parameters
    ----------
    reason
        Human-readable explanation and migration hint (e.g. the replacement
        symbol). Appended to the warning message and the docstring note.
    version_removed
        The version in which the symbol is scheduled to be removed (e.g.
        ``"1.2.0"``). Surfaced in both the warning and the docstring note.

    Returns
    -------
    decorator
        A decorator that wraps the target callable/class with the deprecation
        warning while preserving its signature and metadata.
    """
    def decorator(obj):
        msg = _build_deprecation_message(name=obj.__name__, reason=reason,
                                         version_removed=version_removed)
        doc = _prepend_deprecation_note(doc=obj.__doc__, reason=reason,
                                        version_removed=version_removed)
        if isinstance(obj, type):
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                orig_init(self, *args, **kwargs)

            obj.__init__ = new_init
            obj.__doc__ = doc
            return obj

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        wrapper.__doc__ = doc
        return wrapper

    return decorator


# Catch backend errors
class BackendProcessingError(Exception):
    """Custom exception for backend processing errors."""
    def __init__(self, message, cause=None):
        super().__init__(message)
        self.cause = cause  # Store the original exception

    def __str__(self):
        if self.cause:
            return f"{self.args[0]} (Caused by: {self.cause})"
        return self.args[0]


def catch_backend_processing_error():
    """
    Decorator to catch all exceptions, wrap them as BackendProcessingError with context.

    This exception is intended for use in main backend functions that are exposed to the frontend.
    It serves as a catch-all for potential errors originating in the backend, ensuring that issues
    are encapsulated and communicated effectively to the frontend.

    Note:
    - This class should be used sparingly and only for backend logic errors or unexpected issues.
    - Input-dependent errors should be handled directly in the frontend to maintain clear separation
      of concerns and improve input validation workflows.

    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Execute the decorated function
                return func(*args, **kwargs)
            except Exception as error:
                # Catch any exception and wrap it in BackendProcessingError
                raise BackendProcessingError(
                    f"Error in backend function '{func.__name__}'", cause=error
                ) from error
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


def catch_runtime_warnings(suppress=False):
    """Decorator to catch RuntimeWarnings and store them in a list."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with CatchRuntimeWarnings() as crw:
                result = func(*args, **kwargs)
            if crw.get_warnings():
                list_warnings = crw.get_warnings()
                n = len(list_warnings)
                summary_msg = f"The following {n} 'RuntimeWarnings' were caught:\n" + "\nRuntimeWarning: ".join(crw.get_warnings())
                if not suppress:
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
    """Decorator to catch ConvergenceWarnings and raise custom exceptions."""
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
       and raise custom exceptions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with CatchRuntimeWarnings() as crw:
                result = func(*args, **kwargs)
            if crw.get_warnings():
                raise InvalidDivisionException(f"\nError due to 'RuntimeWarning': {crw.get_warnings()[0]}")
            return result
        return wrapper
    return decorator


# Catch UndefinedMetricWarnings
class CatchUndefinedMetricWarning:
    """Context manager to catch and aggregate UndefinedMetricWarnings."""
    def __enter__(self):
        self._warn_set = set()
        self._other_warnings = []
        self._showwarning_orig = warnings.showwarning
        warnings.showwarning = self._catch_warning
        return self

    def __exit__(self, exc_type, exc_value, tb):
        warnings.showwarning = self._showwarning_orig
        for warn_message, warn_category, filename, lineno in self._other_warnings:
            warnings.warn_explicit(warn_message, warn_category, filename, lineno)

    def _catch_warning(self, message, category, filename, lineno, file=None, line=None):
        if category == UndefinedMetricWarning:
            self._warn_set.add(str(message))  # Add message to set (duplicates are automatically handled)
        else:
            self._other_warnings.append((message, category, filename, lineno))

    def get_warnings(self):
        return list(self._warn_set)


def catch_undefined_metric_warning():
    """Decorator to catch and report UndefinedMetricWarnings once per unique message."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with CatchUndefinedMetricWarning() as cumw:
                result = func(*args, **kwargs)
            if cumw.get_warnings():
                summary_msg = "The following 'UndefinedMetricWarning' was caught:\n" + "\n".join(cumw.get_warnings())
                summary_msg += ("\n This warning was likely triggered due to 'precision' or 'f1' metrics and "
                                "an imbalanced and/or small dataset.")
                warnings.warn(summary_msg, UndefinedMetricWarning)
            return result
        return wrapper
    return decorator

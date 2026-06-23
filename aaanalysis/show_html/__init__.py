"""
Notebook display helper (``dev`` extra).

Public objects: display_df.
``display_df`` renders a DataFrame as a clean, width-bounded HTML table with its shape —
the house presentation for every example / tutorial notebook. Gated behind the ``dev``
extra (needs ``IPython``); imported lazily from the top-level package and replaced by an
install-hint stub when absent.

See ``.claude/rules/notebooks.md`` for the notebook presentation rule, ``CONTEXT.md``
for domain terms.
"""
from ._display_df import display_df


__all__ = [
    "display_df",
]

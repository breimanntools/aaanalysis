"""
This is a script for the backend of the CPP.run() method.

This is the key algorithm of CPP and for AAanalysis.

Stage implementations were moved into ``_filters/`` modules in PR 1
(filter-stage extraction). This file remains as a backward-compatibility
re-export so existing internal imports keep working — the public surface
is unchanged.

Stage map:
- ``assign_scale_values_to_seq``  →  ``_filters._assign``
- ``pre_filtering_info``          →  ``_filters._stat_filter``
- ``pre_filtering``               →  ``_filters._pre_filter``
- ``filtering``, ``filtering_info_`` → ``_filters._redundancy_filter``
- ``add_stat``                    →  ``_filters._add_stat``

Multiprocessing / shared-progress infrastructure lives in
``_filters._progress`` (module-level globals are owned by that single module
so all stages share the same MP state).
"""
from ._filters._assign import assign_scale_values_to_seq
from ._filters._stat_filter import pre_filtering_info
from ._filters._pre_filter import pre_filtering
from ._filters._redundancy_filter import filtering, filtering_info_
from ._filters._add_stat import add_stat

__all__ = [
    "assign_scale_values_to_seq",
    "pre_filtering_info",
    "pre_filtering",
    "filtering",
    "filtering_info_",
    "add_stat",
]

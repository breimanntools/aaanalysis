"""
Sequence analysis: window sampling and amino-acid sequence logos.

Public objects: AALogo(+Plot), AAWindowSampler.
``AAWindowSampler`` turns ``df_seq`` (from ``data_handling.load_dataset``) into
fixed-length windows / labels for ``feature_engineering.SequenceFeature``; ``AALogo``
(+ ``AALogoPlot``) renders position-specific sequence logos.

See ``.claude/rules/code-conventions.md`` for conventions, ``reproducibility.md`` for the
``seed`` contract, ``CONTEXT.md`` for domain terms (window sampling vocabulary).
"""
from ._aalogo import AALogo
from ._aalogo_plot import AALogoPlot
from ._aa_window_sampler import AAWindowSampler

__all__ = [
    "AALogo",
    "AALogoPlot",
    "AAWindowSampler",
]

# Backward-compatible aliases for the pre-PascalCase names (AAlogo/AAlogoPlot),
# resolved lazily so they stay out of ``__all__``.
_DEPRECATED_CLASS_ALIASES = {"AAlogo": "AALogo", "AAlogoPlot": "AALogoPlot"}


def __getattr__(name):
    canonical = _DEPRECATED_CLASS_ALIASES.get(name)
    if canonical is not None:
        import warnings
        warnings.warn(f"'{name}' is deprecated; use '{canonical}'.",
                      DeprecationWarning, stacklevel=2)
        return globals()[canonical]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

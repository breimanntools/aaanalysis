from .data_handling import (load_dataset, load_scales, load_features,
                            read_fasta, to_fasta,
                            SequencePreprocessor,
                            EmbeddingPreprocessor,
                            combine_dict_nums)
from .seq_analysis import AAlogo, AAlogoPlot, AAWindowSampler
from .feature_engineering import AAclust, AAclustPlot, SequenceFeature, NumericalFeature, CPP, CPPGrid, CPPPlot
from .pu_learning import dPULearn, dPULearnPlot
from .explainable_ai import TreeModel
from .protein_design import AAMut, AAMutPlot, SeqMut, SeqMutPlot, SeqOpt, SeqOptPlot
from .plotting import (plot_get_clist, plot_get_cmap, plot_get_cdict,
                       plot_settings, plot_legend, plot_gcfs, plot_rank)
from .metrics import (comp_auc_adjusted, comp_bic_score, comp_kld,
                      comp_per_protein_ap, comp_detection_metrics,
                      comp_bootstrap_ci, comp_smooth_scores)
from .config import options

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("aaanalysis")
except PackageNotFoundError:        # not installed (e.g. source tree without metadata)
    __version__ = "0.0.0+unknown"


# Import of base version features
__all__ = [
    "load_dataset",
    "load_scales",
    "load_features",
    "read_fasta",
    "to_fasta",
    "SequencePreprocessor",
    "EmbeddingPreprocessor",
    "combine_dict_nums",
    # "comp_seq_sim",       BioPython
    # "filter_seq",         BioPython
    # "scan_motif",         MEME Suite
    "AAlogo",
    "AAlogoPlot",
    "AAWindowSampler",
    # "StructurePreprocessor",   # DSSP + biopython (pro)
    # "AnnotationPreprocessor",  # UniProt fetch + requests (pro)
    "AAclust",
    "AAclustPlot",
    "SequenceFeature",
    "NumericalFeature",
    "CPP",
    "CPPGrid",
    "CPPPlot",
    "dPULearn",
    "dPULearnPlot",
    "AAMut",
    "AAMutPlot",
    "SeqMut",
    "SeqMutPlot",
    "SeqOpt",
    "SeqOptPlot",
    "TreeModel",
    # "ShapModel"       # SHAP
    "plot_get_clist",
    "plot_get_cmap",
    "plot_get_cdict",
    "plot_settings",
    "plot_legend",
    "plot_gcfs",
    "plot_rank",
    "comp_auc_adjusted",
    "comp_bic_score",
    "comp_kld",
    "comp_per_protein_ap",
    "comp_detection_metrics",
    "comp_bootstrap_ci",
    "comp_smooth_scores",
    "options"
]


# Dynamically import professional (pro) or development (dev) features if dependencies are available.
# Optional-dependency import names per install extra. missing_feature_stub uses these to tell a
# genuinely-missing optional dependency apart from a real bug inside an extra module: it branches on
# the ImportError's ``.name`` (reliable for ModuleNotFoundError on the Python 3.11+ floor), never on a
# substring of the message. See .claude/rules/pro-core-boundary.md.
_EXTRA_MODULES = {
    "pro": {"shap", "Bio", "biopython", "upsetplot", "UpSetPlot", "requests", "afragmenter", "py3Dmol", "ipywidgets"},
    "embed": {"torch", "transformers", "sentencepiece", "huggingface_hub"},
    "dev": {"IPython"},
}
_EXTRA_INSTALL = {"pro": "aaanalysis[pro]", "embed": "aaanalysis[embed]", "dev": "aaanalysis[dev]"}


def _raise_missing_feature(feature_name, error, mode):
    """Raise an install hint if ``error`` names a known optional dependency, else re-raise it unchanged."""
    if getattr(error, "name", None) in _EXTRA_MODULES[mode]:
        raise ImportError(
            f"'{feature_name}' needs additional dependencies. Install via:\n"
            f"\n\tpip install '{_EXTRA_INSTALL[mode]}'"
        ) from error
    raise error


def missing_feature_stub(feature_name, error, mode="pro"):
    """Return a callable that raises an ImportError when used.

    This acts as a stub — a placeholder function that represents a missing feature.
    Instead of breaking the import, the stub delays the error until the user tries to use
    the unavailable feature. When invoked, it raises a friendly install hint only if the
    original ImportError names a known optional dependency for ``mode`` (via ``error.name``);
    otherwise it re-raises the original error unchanged so real bugs surface with a full traceback.
    """
    if mode not in _EXTRA_MODULES:
        raise ValueError("mode must be 'pro' or 'dev'")
    return lambda *args, **kwargs: _raise_missing_feature(feature_name, error, mode)

try:
    from .explainable_ai_pro import ShapModel
    __all__.append("ShapModel")
except ImportError as e:
    ShapModel = None
    globals()["ShapModel"] = missing_feature_stub("ShapModel", e, mode="pro")


try:
    from .seq_analysis_pro import comp_seq_sim
    __all__.append("comp_seq_sim")
except ImportError as e:
    comp_seq_sim = None
    globals()["comp_seq_sim"] = missing_feature_stub("comp_seq_sim", e, mode="pro")


try:
    from .seq_analysis_pro import filter_seq
    __all__.append("filter_seq")
except ImportError as e:
    filter_seq = None
    globals()["filter_seq"] = missing_feature_stub("filter_seq", e, mode="pro")


try:
    from .seq_analysis_pro import scan_motif
    __all__.append("scan_motif")
except ImportError as e:
    scan_motif = None
    globals()["scan_motif"] = missing_feature_stub(
        "scan_motif", e, mode="pro")


try:
    from .data_handling_pro import StructurePreprocessor
    __all__.append("StructurePreprocessor")
except ImportError as e:
    StructurePreprocessor = None
    globals()["StructurePreprocessor"] = missing_feature_stub(
        "StructurePreprocessor", e, mode="pro")


try:
    from .data_handling_pro import AnnotationPreprocessor
    __all__.append("AnnotationPreprocessor")
except ImportError as e:
    AnnotationPreprocessor = None
    globals()["AnnotationPreprocessor"] = missing_feature_stub(
        "AnnotationPreprocessor", e, mode="pro")


try:
    from .feature_engineering_pro import CPPStructurePlot
    __all__.append("CPPStructurePlot")
except ImportError as e:
    CPPStructurePlot = None
    globals()["CPPStructurePlot"] = missing_feature_stub(
        "CPPStructurePlot", e, mode="pro")


try:
    from .show_html import display_df
    __all__.append("display_df")
except ImportError as e:
    display_df = None
    globals()["display_df"] = missing_feature_stub("display_df", e, mode="dev")

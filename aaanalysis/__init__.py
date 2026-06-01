from .data_handling import (load_dataset, load_scales, load_features,
                            read_fasta, to_fasta,
                            SequencePreprocessor,
                            EmbeddingPreprocessor,
                            combine_dict_nums)
from .seq_analysis import AAlogo, AAlogoPlot, AAWindowSampler
from .feature_engineering import AAclust, AAclustPlot, SequenceFeature, NumericalFeature, CPP, CPPGrid, CPPPlot
from .pu_learning import dPULearn, dPULearnPlot
from .explainable_ai import TreeModel
from .protein_design import AAMut, AAMutPlot, SeqMut, SeqMutPlot
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


# Dynamically import professional (pro) or development (dev) features if dependencies are available
def raise_error_pro(feature_name=None, error_message=None):
    """Raise an informative ImportError with optional instructions for missing pro features."""
    msg = str(error_message)
    if "No module named" in msg:
        raise ImportError(
            f"'{feature_name}' needs additional dependencies. Install AAanalysis Professional via:\n"
            f"\n\tpip install 'aaanalysis[pro]'"
        )
    raise ImportError(error_message)


def raise_error_dev(feature_name=None, error_message=None):
    """Raise an informative ImportError with optional instructions for missing dev features."""
    msg = str(error_message)
    if "No module named" in msg:
        raise ImportError(
            f"'{feature_name}' needs additional dependencies. Install AAanalysis Development via:\n"
            f"\n\tpip install 'aaanalysis[dev]'"
        )
    raise ImportError(error_message)


def missing_feature_stub(feature_name, error, mode="pro"):
    """Return a callable that raises an ImportError when used.

    This acts as a stub — a placeholder function that represents a missing feature.
    Instead of breaking the import, the stub delays the error until the user tries to use
    the unavailable feature.
    """
    if mode == "pro":
        return lambda *args, **kwargs: raise_error_pro(feature_name, error)
    elif mode == "dev":
        return lambda *args, **kwargs: raise_error_dev(feature_name, error)
    else:
        raise ValueError("mode must be 'pro' or 'dev'")

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
    from .show_html import display_df
    __all__.append("display_df")
except ImportError as e:
    display_df = None
    globals()["display_df"] = missing_feature_stub("display_df", e, mode="dev")

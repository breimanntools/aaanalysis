from .data_handling import (load_dataset, load_scales, load_features,
                            read_fasta, to_fasta,
                            SequencePreprocessor)
from .seq_analysis import AALogo, AALogoPlot
from .feature_engineering import AAclust, AAclustPlot, SequenceFeature, NumericalFeature, CPP, CPPPlot
from .pu_learning import dPULearn, dPULearnPlot
from .explainable_ai import TreeModel
from .protein_design import AAMut, AAMutPlot, SeqMut, SeqMutPlot
from .plotting import (plot_get_clist, plot_get_cmap, plot_get_cdict,
                       plot_settings, plot_legend, plot_gcfs)
from .metrics import (comp_auc_adjusted, comp_bic_score, comp_kld)
from .config import options


# Import of base version features
__all__ = [
    "load_dataset",
    "load_scales",
    "load_features",
    "read_fasta",
    "to_fasta",
    "SequencePreprocessor",
    # "comp_seq_sim",       BioPython
    # "filter_seq",         BioPython
    "AALogo",
    "AALogoPlot",
    "AAclust",
    "AAclustPlot",
    "SequenceFeature",
    "NumericalFeature",
    "CPP",
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
    "comp_auc_adjusted",
    "comp_bic_score",
    "comp_kld",
    "options"
]


# Dynamically import professional (pro) features if dependencies are available
def raise_error(feature_name=None, error_message=None):
    """Raise an informative ImportError with optional instructions for missing pro features."""
    str_error_message = str(error_message)
    if "No module named" in str_error_message:
        str_error = (f"'{feature_name}' needs additional dependencies. Install AAanalysis Professional via: "
                     f"\n\tpip install aaanalysis[pro].")
        raise ImportError(str_error)
    else:
        raise ImportError(error_message)


def missing_feature_stub(feature_name, error):
    """Return a callable that raises an ImportError when used

    This acts as a **stub** — a placeholder function that represents a missing feature.
    Instead of breaking the import, the stub delays the error until the user tries to use
    the unavailable feature (e.g., `ShapModel()`), providing a clear message on how to fix it
    """
    return lambda *args, **kwargs: raise_error(feature_name=feature_name, error_message=error)


try:
    from .explainable_ai_pro import ShapModel
    __all__.append("ShapModel")
except ImportError as e:
    ShapModel = None
    globals()["ShapModel"] = missing_feature_stub("ShapModel", e)


try:
    from .seq_analysis_pro import comp_seq_sim
    __all__.append("comp_seq_sim")
except ImportError as e:
    comp_seq_sim = None
    globals()["comp_seq_sim"] = missing_feature_stub("comp_seq_sim", e)


try:
    from .seq_analysis_pro import filter_seq
    __all__.append("filter_seq")
except ImportError as e:
    filter_seq = None
    globals()["filter_seq"] = missing_feature_stub("filter_seq", e)


try:
    from .show_html import display_df
    __all__.append("display_df")
except ImportError as e:
    display_df = None
    globals()["display_df"] = missing_feature_stub("display_df", e)

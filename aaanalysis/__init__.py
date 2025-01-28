from .data_handling import (load_dataset, load_scales, load_features,
                            read_fasta, to_fasta,
                            SequencePreprocessor)
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
    # "comp_seq_sim",       BioPython
    # "filter_seq",         BioPython
    "SequencePreprocessor",
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
    """Dynamically set the fallback for missing features and raise ImportError"""
    str_error_message = str(error_message)
    if "No module named" in str_error_message:
        str_error = (f"'{feature_name}' needs additional dependencies. Install AAanalysis Professional via: "
                     f"\n\tpip install aaanalysis[pro].")
        raise ImportError(str_error)
    else:
        raise ImportError(e)


try:
    from .explainable_ai_pro import ShapModel
    __all__.append("ShapModel")
except ImportError as e:
    ShapModel = None  # For IDE navigation
    globals()["ShapModel"] = raise_error(feature_name="ShapModel", error_message=e)


try:
    from .data_handling_pro import comp_seq_sim
    __all__.append("comp_seq_sim")
except ImportError as e:
    comp_seq_sim = None
    globals()["comp_seq_sim"] = raise_error(feature_name="comp_seq_sim", error_message=e)


try:
    from .data_handling_pro import filter_seq
    __all__.append("filter_seq")
except ImportError as e:
    comp_seq_sim = None
    globals()["filter_seq"] = raise_error(feature_name="filter_seq", error_message=e)

try:
    from .show_html import display_df
    __all__.append("display_df")
except ImportError as e:
    comp_seq_sim = None
    globals()["display_df"] = raise_error(feature_name="display_df", error_message=e)

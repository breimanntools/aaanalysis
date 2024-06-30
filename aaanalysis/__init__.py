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


# Import of professional (pro) version features if dependencies are available
try:
    from .explainable_ai_pro import ShapModel
    from .data_handling_pro import comp_seq_sim, filter_seq
    from .show_html import display_df
    # Extend the __all__ list with pro features if successful
    __all__.extend(["ShapModel",
                    "display_df",
                    "comp_seq_sim",
                    "filter_seq"])

except ImportError as e:
    # Define a factory function to create a class or function placeholder
    def make_pro_feature(feature_name):
        str_error = (f"'{feature_name}' needs additional dependencies. Install AAanalysis Professional via:"
                     f"\n\tpip install aaanalysis[pro].")

        class UnavailableFeature:
            def __init__(self, *args, **kwargs):
                raise ImportError(str_error)

            def __call__(self, *args, **kwargs):
                raise ImportError(str_error)

        # Return the custom class for the unavailable feature
        return UnavailableFeature

    # Use the factory function to create placeholders for pro features
    ShapModel = make_pro_feature("ShapModel")
    display_df = make_pro_feature("display_df")
    comp_seq_sim = make_pro_feature("comp_seq_sim")
    filter_seq = make_pro_feature("filter_seq")

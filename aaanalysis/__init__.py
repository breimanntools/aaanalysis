from .data_handling import load_dataset, load_scales, load_features, to_fasta
from .feature_engineering import AAclust, AAclustPlot, SequenceFeature, NumericalFeature, CPP, CPPPlot
from .pu_learning import dPULearn, dPULearnPlot
from .pertubation import AAMut, AAMutPlot, SeqMut, SeqMutPlot
from .plotting import (plot_get_clist, plot_get_cmap, plot_get_cdict,
                       plot_settings, plot_legend, plot_gcfs)
from .metrics import (comp_auc_adjusted, comp_bic_score, comp_kld)
from .config import options


# Import of base version features
__all__ = [
    "load_dataset",
    "load_scales",
    "load_features",
    "to_fasta",
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
    from .explainable_ai import TreeModel, ShapExplainer
    from .plotting import display_df
    # Extend the __all__ list with pro features if successful
    __all__.extend(["TreeModel",
                    "ShapExplainer",
                    "display_df"])

except ImportError as e:
    # Define a factory function to create a class or function placeholder
    def make_unavailable_feature(feature_name):
        str_error = (f"'{feature_name}' requires additional dependencies. Please install the professional version of AAanalysis via:"
                     f"\n\tpip install aaanalysis[pro]")

        class UnavailableFeature:
            def __init__(self, *args, **kwargs):
                raise ImportError(str_error)

            def __call__(self, *args, **kwargs):
                raise ImportError(str_error)

        # Return the custom class for the unavailable feature
        return UnavailableFeature

    # Use the factory function to create placeholders for pro features
    TreeModel = make_unavailable_feature("TreeModel")
    ShapExplainer = make_unavailable_feature("ShapExplainer")
    display_df = make_unavailable_feature("display_df")

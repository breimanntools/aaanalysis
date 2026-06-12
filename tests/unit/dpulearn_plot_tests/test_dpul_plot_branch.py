"""This is a script to test residual dPULearnPlot branch arms via the public API."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa

aa.options["verbose"] = False


def _df_pu(n_samples=8, n_pc=3):
    """A minimal df_pu with PC + abs_dif + selection_via columns."""
    rng = np.random.default_rng(0)
    data = {}
    for i in range(1, n_pc + 1):
        data[f"PC{i}"] = rng.random(n_samples)
        data[f"PC{i}_abs_dif"] = rng.random(n_samples)
    df = pd.DataFrame(data)
    df["selection_via"] = ["PC1", "PC2"] + [None] * (n_samples - 2)
    return df


def _labels(n_samples=8, n_neg=2):
    """0 (identified neg), 1 (pos), 2 (unl) labels of length n_samples."""
    labels = [0] * n_neg + [1, 1] + [2] * (n_samples - n_neg - 2)
    return labels


# I dPULearnPlot.pca branch arms
class TestPcaBranch:
    """Cover residual arms reachable through dPULearnPlot.pca()."""

    def test_too_few_pcs(self):
        # _dpulearn_plot.py:236 — df_pu with < 2 PCs raises
        df_pu = _df_pu(n_pc=1)
        labels = _labels()
        with pytest.raises(ValueError, match="at least two PCs"):
            aa.dPULearnPlot.pca(df_pu=df_pu, labels=labels)
        plt.close("all")

    def test_color_list_via_kwargs_scatterplot(self):
        # _dpulearn_plot.py:31 — args_scatter 'c' as a non-str sequence -> colors = list
        df_pu = _df_pu()
        labels = _labels()
        colors_list = ["#111111", "#222222", "#333333"]
        ax = aa.dPULearnPlot.pca(df_pu=df_pu, labels=labels,
                                 kwargs_scatterplot={"c": colors_list})
        assert ax is not None
        plt.close("all")

    def test_names_colors_length_mismatch(self):
        # _dpulearn_plot.py:59 — check_match_names_colors raises on length mismatch
        df_pu = _df_pu()
        labels = _labels()
        with pytest.raises(ValueError, match="does not match"):
            aa.dPULearnPlot.pca(df_pu=df_pu, labels=labels,
                                names=["a", "b", "c"], colors=["#111111", "#222222"])
        plt.close("all")


# II dPULearnPlot.eval branch arm (dict_xlims gating)
class TestEvalBranch:
    """Cover the dict_xlims axis-gating arm of plot_eval."""

    def _df_eval(self, with_neg=False, with_kld=False):
        data = {
            "name": ["PCA", "dist"],
            "avg_STD": [0.5, 0.6],
            "avg_IQR": [0.4, 0.5],
            "avg_abs_AUC_pos": [0.7, 0.6],
            "avg_abs_AUC_unl": [0.65, 0.55],
        }
        if with_neg:
            data["avg_abs_AUC_neg"] = [0.6, 0.5]
        if with_kld:
            data["avg_KLD_pos"] = [0.2, 0.3]
            data["avg_KLD_unl"] = [0.25, 0.35]
        return pd.DataFrame(data)

    def test_dict_xlims_applied(self):
        # dpul_plot.py:80 — dict_xlims key i <= n_colss is applied
        df_eval = self._df_eval(with_kld=True)
        fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval, dict_xlims={0: (0, 1), 1: (0, 1)})
        assert axes is not None
        plt.close("all")

    def test_eval_no_kld(self):
        # Dissimilarity title without KLD axis (kld_in False branch)
        df_eval = self._df_eval(with_kld=False)
        fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval)
        assert axes is not None
        plt.close("all")

"""This is a testing script for the dPULearnPlot.pca() method"""
import hypothesis.strategies as some
from hypothesis import given, settings
import pytest
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import aaanalysis as aa
import random
import warnings


def _create_sample_df_pu(n_samples=10, n_pcs=7, include_abs_diff=True, include_selection_via=True,
                         n_neg=5, random_order=False):
    """Create random df_pu"""
    # Base columns for principal components (PCs)
    _pc_columns = [f'PC{i + 1} ({random.uniform(1, 60):.1f}%)' for i in range(n_pcs)]
    pc_columns = [x.split(" ")[0] for x in _pc_columns]
    # Generate random data for principal components
    data_matrix = np.random.rand(n_samples, n_pcs)
    # Create DataFrame with PC columns
    df_pu = pd.DataFrame(data_matrix, columns=_pc_columns)
    # Add columns for absolute differences if needed
    if include_abs_diff:
        abs_diff_columns = [f'{col}_abs_dif' for col in pc_columns]
        abs_diff_data = np.random.rand(n_samples, n_pcs)
        df_pu[abs_diff_columns] = pd.DataFrame(abs_diff_data, columns=abs_diff_columns)
    # Add 'selection_via' column if needed
    if include_selection_via:
        # Ensure n_neg does not exceed the length of pc_columns
        n_neg = min(n_neg, len(pc_columns))
        # Create a list of random PC labels and None, with the sum of PC labels equal to n_neg
        selection_labels = random.sample(pc_columns, n_neg) + [None] * (n_samples - n_neg)
        random.shuffle(selection_labels)
        df_pu['selection_via'] = selection_labels
    # Randomly shuffle columns if required
    if random_order:
        shuffled_cols = random.sample(list(df_pu.columns), len(df_pu.columns))
        df_pu = df_pu[shuffled_cols]
    return df_pu


def _create_labels(n=10, n_neg=5, include_unl=True, labels_selection_via=None):
    """Create random label set with a specified number of negative labels."""
    # If labels_selection_via is provided, it overrides n
    n = len(labels_selection_via) if labels_selection_via is not None else n
    # Initialize labels with 1 or 2
    labels = list(np.random.choice([1, 2], size=n)) if include_unl else [1] * n
    # Override labels based on labels_selection_via
    if labels_selection_via is not None:
        labels = [0 if x is not None else label for x, label in zip(labels_selection_via, labels)]
    else:
        # Randomly assign 0 to n_neg elements in labels
        indices_to_change = random.sample(range(n), n_neg)
        for i in indices_to_change:
            labels[i] = 0
    return labels


def _get_random_int():
    """"""
    n_samples = random.randint(20, 100)
    n_pc = random.randint(3, 5)
    n_neg = random.randint(5, int(n_samples / 2))
    return n_samples, n_pc, n_neg


class TestdPULearnPlotPCA:
    """Positive tests for the dPULearnPlot.pca() function."""

    # Positive tests
    def test_df_pu_valid(self):
        for i in range(20):
            n_samples, n_pc, n_neg = _get_random_int()
            df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc,
                                         include_abs_diff=True, n_neg=n_neg, random_order=True)
            labels = _create_labels(labels_selection_via=df_pu["selection_via"])
            ax = aa.dPULearnPlot.pca(df_pu=df_pu, labels=labels)
            assert isinstance(ax, plt.Axes)
            plt.close()


    def test_figsize_valid(self):
        for i in range(20):
            n_samples, n_pc, n_neg = _get_random_int()
            df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                         random_order=True)
            labels = _create_labels(labels_selection_via=df_pu["selection_via"])
            figsize = (random.uniform(4, 20), random.uniform(4, 20))
            ax = aa.dPULearnPlot.pca(df_pu=df_pu, labels=labels, figsize=figsize)
            assert isinstance(ax, plt.Axes)
            plt.close()


    def test_labels_valid(self):
        for i in range(20):
            n_samples, n_pc, n_neg = _get_random_int()
            df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                         random_order=True)
            labels = _create_labels(labels_selection_via=df_pu["selection_via"])
            ax = aa.dPULearnPlot.pca(df_pu, labels=labels)
            assert isinstance(ax, plt.Axes)
            plt.close()


    def test_pc_x_pc_y_valid(self):
        for i in range(20):
            n_samples, n_pc, n_neg = _get_random_int()
            df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                         random_order=True)
            pc_x = random.randint(1, 3)
            pc_y = random.randint(1, 3)
            labels = _create_labels(labels_selection_via=df_pu["selection_via"])
            ax = aa.dPULearnPlot.pca(df_pu, pc_x=pc_x, pc_y=pc_y, labels=labels)
            assert isinstance(ax, plt.Axes)
            plt.close()


    @settings(max_examples=10, deadline=2000)
    @given(show_pos_mean_x=some.booleans(), show_pos_mean_y=some.booleans())
    def test_show_pos_mean_valid(self, show_pos_mean_x, show_pos_mean_y):
        n_samples, n_pc, n_neg = _get_random_int()
        df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                     random_order=True)
        labels = _create_labels(labels_selection_via=df_pu["selection_via"])
        ax = aa.dPULearnPlot.pca(df_pu, show_pos_mean_x=show_pos_mean_x, show_pos_mean_y=show_pos_mean_y, labels=labels)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=10, deadline=2000)
    @given(colors=some.lists(some.sampled_from(['blue', 'green', 'red']), min_size=3, max_size=3, unique=True))
    def test_colors_valid(self, colors):
        n_samples, n_pc, n_neg = _get_random_int()
        df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                     random_order=True)
        labels = _create_labels(labels_selection_via=df_pu["selection_via"])
        ax = aa.dPULearnPlot.pca(df_pu, colors=colors, labels=labels)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=10, deadline=2000)
    @given(names=some.lists(some.text(), min_size=3, max_size=3, unique=True))
    def test_names_valid(self, names):
        with warnings.catch_warnings():
            # Suppress specific matplotlib UserWarnings about missing glyphs
            warnings.filterwarnings("ignore", category=UserWarning)
            n_samples, n_pc, n_neg = _get_random_int()
            df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                         random_order=True)
            labels = _create_labels(labels_selection_via=df_pu["selection_via"])

            ax = aa.dPULearnPlot.pca(df_pu, names=names, labels=labels)
            assert isinstance(ax, plt.Axes)
            plt.close()


    @settings(max_examples=10, deadline=2000)
    @given(legend=some.booleans())
    def test_legend_valid(self, legend):
        n_samples, n_pc, n_neg = _get_random_int()
        df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                     random_order=True)
        labels = _create_labels(labels_selection_via=df_pu["selection_via"])
        ax = aa.dPULearnPlot.pca(df_pu, legend=legend, labels=labels)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=10, deadline=2000)
    @given(legend_y=some.floats(min_value=-1, max_value=1))
    def test_legend_y_valid(self, legend_y):
        n_samples, n_pc, n_neg = _get_random_int()
        df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                     random_order=True)
        labels = _create_labels(labels_selection_via=df_pu["selection_via"])
        ax = aa.dPULearnPlot.pca(df_pu, legend_y=legend_y, labels=labels)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_args_scatter_valid(self):
        list_args_scatter = [{"color": "red", "marker": "o"},
                             {"alpha": 0.5, "edgecolors": "black"},
                             {"s": 100,  "marker": "o"},
                             {"c": ["red", "green", "blue"],"linewidths": 2},
                             {"color": "cyan", "marker": "^", "alpha": 0.7, "edgecolors": "none", "s": 50}]
        for args_scatter in list_args_scatter:
            n_samples, n_pc, n_neg = _get_random_int()
            df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                         random_order=True)
            labels = _create_labels(labels_selection_via=df_pu["selection_via"])
            ax = aa.dPULearnPlot.pca(df_pu, args_scatter=args_scatter, labels=labels)
            assert isinstance(ax, plt.Axes)
            plt.close()

    # Negative tests
    def test_df_pu_invalid_type(self):
        """Test passing an invalid type for df_pu."""
        labels = _create_labels(n_neg=2, n=4)
        with pytest.raises(ValueError):
            aa.dPULearnPlot.pca(df_pu="invalid_type", labels=labels)

    def test_df_pu_missing_columns(self):
        """Test passing a DataFrame missing required columns."""
        labels = _create_labels(n_neg=1, n=3)
        df_pu = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        with pytest.raises(ValueError):
            aa.dPULearnPlot.pca(df_pu=df_pu, labels=labels)

    def test_labels_invalid_length(self):
        """Test passing labels of incorrect length."""
        for i in range(3, 20):
            n_samples, n_pc, n_neg = _get_random_int()
            df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                         random_order=True)
            labels = [1, 0]  # Incorrect length
            with pytest.raises(ValueError):
                aa.dPULearnPlot.pca(df_pu, labels=labels)

    def test_figsize_invalid_type(self):
        """Test passing an invalid type for figsize."""
        for i in range(20):
            n_samples, n_pc, n_neg = _get_random_int()
            df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                         random_order=True)
            labels = _create_labels(labels_selection_via=df_pu["selection_via"])
            with pytest.raises(ValueError):
                aa.dPULearnPlot.pca(df_pu=df_pu, labels=labels, figsize="invalid_type")

    def test_figsize_negative_values(self):
        """Test passing negative values for figsize."""
        for i in range(20):
            n_samples, n_pc, n_neg = _get_random_int()
            df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                         random_order=True)
            labels = _create_labels(labels_selection_via=df_pu["selection_via"])
            with pytest.raises(ValueError):
                aa.dPULearnPlot.pca(df_pu=df_pu, labels=labels, figsize=(-5, -10))

    def test_invalid_pc_x_pc_y(self):
        """Test passing invalid pc_x and pc_y values."""
        for i in range(20):
            n_samples, n_pc, n_neg = _get_random_int()
            df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                         random_order=True)
            labels = _create_labels(labels_selection_via=df_pu["selection_via"])
            with pytest.raises(ValueError):
                aa.dPULearnPlot.pca(df_pu, pc_x=100, pc_y=100, labels=labels)  # Values too large

    def test_invalid_args_scatter(self):
        """Test passing invalid args_scatter."""
        n_samples, n_pc, n_neg = _get_random_int()
        df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg,
                                     random_order=True)
        labels = _create_labels(labels_selection_via=df_pu["selection_via"])
        args_scatter = {"invalid_param": "invalid_value"}
        with pytest.raises(ValueError):
            aa.dPULearnPlot.pca(df_pu=df_pu, labels=labels, args_scatter=args_scatter)

class TestdPULearnPlotPCAComplex:
    """Complex tests for the dPULearnPlot.pca() function."""

    @settings(max_examples=10, deadline=2000)
    @given(
        n_samples=some.integers(min_value=20, max_value=100),
        n_pc=some.integers(min_value=3, max_value=7),
        n_neg=some.integers(min_value=1, max_value=10),
        figsize=some.tuples(some.floats(min_value=4, max_value=20), some.floats(min_value=4, max_value=20)),
        pc_x=some.integers(min_value=1, max_value=3),
        pc_y=some.integers(min_value=1, max_value=3),
        show_pos_mean_x=some.booleans(),
        show_pos_mean_y=some.booleans(),
        legend=some.booleans(),
        legend_y=some.floats(min_value=-1, max_value=1),
        colors=some.lists(some.sampled_from(['blue', 'green', 'red']), min_size=3, max_size=3, unique=True),
        names=some.lists(some.text(), min_size=3, max_size=3, unique=True),
        args_scatter=some.fixed_dictionaries({
            "marker": some.sampled_from(["o", "x", "^"]),
            "alpha": some.floats(min_value=0, max_value=1)
        })
    )
    def test_complex_scenario(self, n_samples, n_pc, n_neg, figsize, pc_x, pc_y, show_pos_mean_x, show_pos_mean_y, legend, legend_y, colors, names, args_scatter):
        """Test dPULearnPlot.pca() with a complex combination of parameters."""
        df_pu = _create_sample_df_pu(n_samples=n_samples, n_pcs=n_pc, include_abs_diff=True, n_neg=n_neg, random_order=True)
        labels = _create_labels(n=n_samples, n_neg=n_neg, include_unl=True, labels_selection_via=df_pu["selection_via"])

        with warnings.catch_warnings():
            # Suppress specific matplotlib UserWarnings about missing glyphs
            warnings.filterwarnings("ignore", category=UserWarning)

            ax = aa.dPULearnPlot.pca(df_pu, labels=labels, figsize=figsize, pc_x=pc_x, pc_y=pc_y, show_pos_mean_x=show_pos_mean_x, show_pos_mean_y=show_pos_mean_y, names=names, colors=colors, legend=legend, legend_y=legend_y, args_scatter=args_scatter)
            assert isinstance(ax, plt.Axes)
            plt.close()
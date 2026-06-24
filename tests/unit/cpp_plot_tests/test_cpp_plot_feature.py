"""
This script tests the feature method for plotting CPP feature distributions.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hypothesis import given, settings
import hypothesis.strategies as st
import warnings
import pytest
import aaanalysis as aa
import random

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Helper functions and common setups
def get_input():
    """"""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=25)
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features()
    return df_seq, labels, df_feat


VALID_FEATURE = 'TMD_C_JMD_C-Segment(3,4)-KLEP840101'


class TestCPPPlotFeature:
    """Test class for feature method, focusing on individual parameters."""

    def test_feature(self):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        random_features = random.sample(features, 10)
        cpp_plot = aa.CPPPlot()
        for feature in random_features:
            _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels)
            assert isinstance(ax, plt.Axes)
            plt.close()

    def test_feature_list(self):
        # feature accepts a list of feature ids; feat_rank selects which one
        df_seq, labels, df_feat = get_input()
        features = random.sample(df_feat["feature"].to_list(), 5)
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=features, df_seq=df_seq, labels=labels)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_feature_df_feat(self):
        # feature accepts a whole df_feat; its 'feature' column supplies the ids
        df_seq, labels, df_feat = get_input()
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=df_feat.head(10), df_seq=df_seq, labels=labels)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_feat_rank(self):
        # feat_rank is 1-based; each valid rank selects a feature and plots
        df_seq, labels, df_feat = get_input()
        features = random.sample(df_feat["feature"].to_list(), 5)
        cpp_plot = aa.CPPPlot()
        for feat_rank in range(1, len(features) + 1):
            _, ax = cpp_plot.feature(feature=features, df_seq=df_seq, labels=labels, feat_rank=feat_rank)
            assert isinstance(ax, plt.Axes)
            plt.close()

    def test_feat_rank_selects_expected(self):
        # feature=list with feat_rank=k plots the SAME data as passing the k-th id directly
        # (as-given order, 1-based), and distinct features plot distinct curves.
        def _kde_signature(ax):
            # Fingerprint each plotted curve by both grid (x) AND density (y): x alone only
            # encodes the data range, so two distinct features sharing a range would collide.
            return tuple(
                (tuple(round(float(x), 6) for x in ln.get_xdata()),
                 tuple(round(float(y), 6) for y in ln.get_ydata()))
                for ln in ax.lines
            )

        df_seq, labels, df_feat = get_input()
        cpp_plot = aa.CPPPlot()
        # Deterministically gather 4 features whose plotted curves are pairwise distinct, so the
        # distinctness assertion below is a valid precondition rather than a sampling coin flip.
        rng = random.Random(0)
        pool = df_feat["feature"].to_list()
        rng.shuffle(pool)
        features, seen = [], set()
        for feature in pool:
            sig = _kde_signature(cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels)[1])
            plt.close("all")
            if sig not in seen:
                seen.add(sig)
                features.append(feature)
            if len(features) == 4:
                break
        assert len(features) == 4

        sigs = []
        for k in range(1, len(features) + 1):
            _, ax_rank = cpp_plot.feature(feature=features, df_seq=df_seq, labels=labels, feat_rank=k)
            _, ax_direct = cpp_plot.feature(feature=features[k - 1], df_seq=df_seq, labels=labels)
            sig_rank = _kde_signature(ax_rank)
            assert sig_rank == _kde_signature(ax_direct)
            sigs.append(sig_rank)
            plt.close("all")
        # Distinct features (distinct ranks) yield distinct plotted curves
        assert len(set(sigs)) == len(sigs)

    def test_invalid_feat_rank(self):
        df_seq, labels, df_feat = get_input()
        features = random.sample(df_feat["feature"].to_list(), 3)
        cpp_plot = aa.CPPPlot()
        # Out of range (1-based): 0, negative, and beyond the number of features
        for feat_rank in [0, -1, len(features) + 1, 99]:
            with pytest.raises(ValueError):
                cpp_plot.feature(feature=features, df_seq=df_seq, labels=labels, feat_rank=feat_rank)
        # Wrong type
        for feat_rank in [1.5, "1", None, [1]]:
            with pytest.raises(ValueError):
                cpp_plot.feature(feature=features, df_seq=df_seq, labels=labels, feat_rank=feat_rank)
        # A single feature only admits feat_rank=1
        with pytest.raises(ValueError):
            cpp_plot.feature(feature=features[0], df_seq=df_seq, labels=labels, feat_rank=2)

    def test_valid_df_seq_labels(self):
        df_seq, labels, df_feat = get_input()
        all_data_set_names = [x for x in aa.load_dataset()["Dataset"].to_list() if "AA" not in x
                              and "AMYLO" not in x and "PU" not in x]
        features = df_feat["feature"].to_list()
        random_features = random.sample(features, 3)
        sampled_names = random.sample(all_data_set_names, 3)
        cpp_plot = aa.CPPPlot()
        for name in sampled_names:
            df_seq = aa.load_dataset(name=name, n=10, min_len=50)
            labels = df_seq["label"].to_list()
            for feature in random_features:
                _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels)
                assert isinstance(ax, plt.Axes)
                plt.close()

    def test_valid_label_test(self):
        df_seq, labels, df_feat = get_input()
        labels = [10 if l == 1 else l for l in labels]
        features = df_feat["feature"].to_list()
        random_features = random.sample(features, 5)
        cpp_plot = aa.CPPPlot()
        for feature in random_features:
            _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, label_test=10)
            assert isinstance(ax, plt.Axes)
            plt.close()

    def test_valid_label_ref(self):
        df_seq, labels, df_feat = get_input()
        labels = [10 if l == 0 else l for l in labels]
        features = df_feat["feature"].to_list()
        random_features = random.sample(features, 5)
        cpp_plot = aa.CPPPlot()
        for feature in random_features:
            _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, label_ref=10)
            assert isinstance(ax, plt.Axes)
            plt.close()

    def test_ax(self):
        df_seq, labels, df_feat = get_input()
        feature = df_feat["feature"].to_list()[0]
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, ax=None)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)
        fig, ax_in = plt.subplots()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, ax=ax_in)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    @settings(max_examples=5, deadline=None)
    @given(figsize=st.tuples(st.floats(min_value=1, max_value=10), st.floats(min_value=1, max_value=20)))
    def test_figsize(self, figsize):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.sample(features, 10)[0]
        cpp_plot = aa.CPPPlot()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, figsize=figsize)
            assert isinstance(ax, plt.Axes)
            plt.close()

    
    def test_names_to_show(self):
        df_seq, labels, df_feat = get_input()
        list_names = [f"Protein {i}" for i in range(len(df_seq))]
        df_seq["name"] = list_names
        features = df_feat["feature"].to_list()
        feature = random.choice(features)
        # Randomly sample up to 6 names from list_names
        names_to_show = random.sample(list_names, min(len(list_names), 6))
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, names_to_show=names_to_show)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=5, deadline=None)
    @given(
        name_test=st.text(alphabet=st.characters(blacklist_characters=["$", "{", "}"], min_codepoint=32, max_codepoint=126),
                          min_size=1, max_size=10),
        name_ref=st.text(alphabet=st.characters(blacklist_characters=["$", "{", "}"], min_codepoint=32, max_codepoint=126),
                         min_size=1, max_size=10)
    )
    def test_name_test_ref(self, name_test, name_ref):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.choice(features)
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, name_test=name_test, name_ref=name_ref)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=5, deadline=None)
    @given(color_test=st.sampled_from(["blue", "green", "red", "yellow"]),
           color_ref=st.sampled_from(["gray", "black", "orange", "pink"]))
    def test_color_test_ref(self, color_test, color_ref):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.choice(features)
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, color_test=color_test, color_ref=color_ref)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=5, deadline=None)
    @given(show_seq=st.booleans())
    def test_show_seq(self, show_seq):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.sample(features, 10)[0]
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, show_seq=show_seq)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @given(show_title=st.booleans())
    def test_show_title(self, show_title):
        df_seq, labels, df_feat = get_input()
        cpp_plot = aa.CPPPlot()
        plt.close("all")
        _, ax = cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, show_title=show_title)
        assert isinstance(ax, plt.Axes)
        # Title present iff show_title; when present it is the wrapped feature description
        if show_title:
            assert ax.get_title() != ""
            sf = aa.SequenceFeature()
            expected = sf.get_feature_descriptions(features=[VALID_FEATURE])[0]
            assert ax.get_title().replace("\n", " ") == expected
        else:
            assert ax.get_title() == ""
        plt.close()

    @settings(max_examples=5, deadline=None)
    @given(title_wrap_width=st.integers(min_value=10, max_value=200))
    def test_title_wrap_width(self, title_wrap_width):
        df_seq, labels, df_feat = get_input()
        cpp_plot = aa.CPPPlot()
        plt.close("all")
        _, ax = cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels,
                              title_wrap_width=title_wrap_width)
        assert isinstance(ax, plt.Axes)
        # Every wrapped line stays within the requested width
        assert all(len(line) <= title_wrap_width for line in ax.get_title().split("\n"))
        plt.close()

    @settings(max_examples=5, deadline=None)
    @given(histplot=st.booleans())
    def test_histplot(self, histplot):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.sample(features, 10)[0]
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, histplot=histplot)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(alpha_hist=st.floats(min_value=0, max_value=1), alpha_dif=st.floats(min_value=0, max_value=1))
    def test_alpha_params(self, alpha_hist, alpha_dif):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.choice(features)
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, alpha_hist=alpha_hist, alpha_dif=alpha_dif)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(fontsize_mean_dif=st.one_of(st.none(), st.floats(min_value=1, max_value=20)))
    def test_fontsize_mean_dif(self, fontsize_mean_dif):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.choice(features)
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, fontsize_mean_dif=fontsize_mean_dif)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(fontsize_name_test=st.one_of(st.none(), st.floats(min_value=1, max_value=20)))
    def test_fontsize_name_test(self, fontsize_name_test):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.choice(features)
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, fontsize_name_test=fontsize_name_test)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(fontsize_name_ref=st.one_of(st.none(), st.floats(min_value=1, max_value=20)))
    def test_fontsize_name_ref(self, fontsize_name_ref):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.choice(features)
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, fontsize_name_ref=fontsize_name_ref)
        assert isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(fontsize_names_to_show=st.one_of(st.none(), st.floats(min_value=1, max_value=20)))
    def test_fontsize_names_to_show(self, fontsize_names_to_show):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.choice(features)
        cpp_plot = aa.CPPPlot()
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels,
                              fontsize_names_to_show=fontsize_names_to_show)
        assert isinstance(ax, plt.Axes)
        plt.close()

    # Negative test cases
    def test_invalid_feature(self):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.choice(features)
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature="invalid_feature", df_seq=df_seq, labels=labels)
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature=feature.lower(), df_seq=df_seq, labels=labels)
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature=feature.upper(), df_seq=df_seq, labels=labels)
        # A list/df_feat containing an invalid feature id is still rejected
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature=[feature, "invalid_feature"], df_seq=df_seq, labels=labels, feat_rank=2)
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature=None, df_seq=df_seq, labels=labels)
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature=1, df_seq=df_seq, labels=labels)
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature=1, df_seq=df_seq, labels=labels)
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature='TMD_C_JMD-C-Segment(3,4)-KLEP840101', df_seq=df_seq, labels=labels)
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature='TMD_C_JMD_C-Segment(3,4)-KLEP84010', df_seq=df_seq, labels=labels)
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature='TMD_C_JMD_C-Segment(3,2)-KLEP840101', df_seq=df_seq, labels=labels)

    def test_invalid_df_seq(self):
        _, labels, _ = get_input()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature=VALID_FEATURE, df_seq=None, labels=labels)

    def test_invalid_labels(self):
        df_seq, _, _ = get_input()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=None)

    def test_invalid_figsize(self):
        df_seq, labels, _ = get_input()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, figsize=(-5, -5))

    def test_invalid_names_to_show(self):
        df_seq, labels, _ = get_input()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            ax = cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, names_to_show=["invalid_name"])

    def test_invalid_label_test(self):
        df_seq, labels, _ = get_input()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, label_test="invalid")

    def test_invalid_label_ref(self):
        df_seq, labels, _ = get_input()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, label_ref="invalid")

    def test_invalid_alpha_hist(self):
        df_seq, labels, _ = get_input()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, alpha_hist=-0.5)
        with pytest.raises(ValueError):
            cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, alpha_hist=1.5)

    def test_invalid_histplot(self):
        df_seq, labels, _ = get_input()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, histplot="invalid")

    def test_invalid_show_seq(self):
        df_seq, labels, _ = get_input()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, show_seq="invalid")

    def test_invalid_show_title(self):
        df_seq, labels, _ = get_input()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, show_title="invalid")

    def test_invalid_title_wrap_width(self):
        df_seq, labels, _ = get_input()
        cpp_plot = aa.CPPPlot()
        for bad in [0, -5, "invalid", 12.5]:
            with pytest.raises(ValueError):
                cpp_plot.feature(feature=VALID_FEATURE, df_seq=df_seq, labels=labels, title_wrap_width=bad)


class TestComplexCPPPlotFeature:
    """Test complex class for feature method, focusing on individual parameters."""


    def test_complex_positive(self):
        df_seq, labels, df_feat = get_input()
        features = df_feat["feature"].to_list()
        feature = random.choice(features)
        cpp_plot = aa.CPPPlot()

        # Complex setup: Custom figsize, specific labels, custom colors, and displaying sequences
        figsize = (10, 8)
        label_test = 1
        label_ref = 0
        color_test = "darkblue"
        color_ref = "darkred"
        show_seq = True
        names_to_show = random.sample([f"Protein {i}" for i in range(len(df_seq))], 3)
        df_seq["name"] = [f"Protein {i}" for i in range(len(df_seq))]

        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels, label_test=label_test, label_ref=label_ref,
                              figsize=figsize, color_test=color_test, color_ref=color_ref, show_seq=show_seq,
                              names_to_show=names_to_show)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_complex_negative(self):
        df_seq, labels, df_feat = get_input()
        cpp_plot = aa.CPPPlot()

        # Complex setup: Invalid feature name, incorrect label_test and label_ref types, and invalid figsize
        invalid_feature = "NonExistentFeature123"
        invalid_label_test = "test"
        invalid_label_ref = "ref"
        invalid_figsize = (-10, -8)

        with pytest.raises(ValueError):
            cpp_plot.feature(feature=invalid_feature, df_seq=df_seq, labels=labels,
                             label_test=invalid_label_test,
                             label_ref=invalid_label_ref, figsize=invalid_figsize)
            plt.close()


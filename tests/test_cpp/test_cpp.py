"""
This is a script for CPP object

Recommended testing commands:
    a) General:     pytest -v -p no:warnings --tb=no test_cpp.py
    b) Function:    pytest -v -p no:warnings --tb=no test_cpp.py::TestCPP:test_add_stat
    c) Doctest:     pytest -v --doctest-modules -p no:warnings cpp_to
"""
import pandas as pd
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import matplotlib as mpl
import matplotlib.pyplot as plt

from cpp_tools import CPP, SequenceFeature
from cpp_tools.cpp import _get_df_pos
import cpp_tools._utils as ut


@pytest.fixture(params=["a", 3, dict(), list(), pd.DataFrame(), -0])
def wrong_input_cpp(request):
    return request.param


@pytest.fixture(params=["a", dict(), list(), pd.DataFrame()])
def wrong_input(request):
    return request.param


@pytest.fixture
def cpp(df_scales, df_cat, df_parts, split_kws):
    return CPP(df_scales=df_scales, df_cat=df_cat, df_parts=df_parts, split_kws=split_kws)


# I Unit Tests
class TestCPP:
    """Test CPP class interface"""

    # Positive unit test
    def test_cpp_call(self, df_scales, df_cat, df_parts, split_kws):
        cpp = CPP(df_scales=df_scales, df_cat=df_cat,
                  df_parts=df_parts, split_kws=split_kws)
        assert isinstance(cpp, object)
        cpp = CPP(df_parts=df_parts, )
        assert isinstance(cpp, object)

    # Negative unit test
    def test_missing_input(self, df_scales, df_cat, df_parts, split_kws):
        with pytest.raises(ValueError):
            CPP()

    def test_wrong_df_scales(self, wrong_input_cpp, df_cat, df_parts, split_kws):
        with pytest.raises(ValueError):
            CPP(df_scales=wrong_input_cpp, df_cat=df_cat, df_parts=df_parts, split_kws=split_kws)

    def test_wrong_df_cat(self, df_scales, wrong_input_cpp, df_parts, split_kws):
        with pytest.raises(ValueError):
            CPP(df_scales=df_scales, df_cat=wrong_input_cpp, df_parts=df_parts, split_kws=split_kws)

    def test_wrong_df_parts(self, df_scales, df_cat, wrong_input_cpp, split_kws):
        with pytest.raises(ValueError):
            CPP(df_scales=df_scales, df_cat=df_cat, df_parts=wrong_input_cpp, split_kws=split_kws)

    def test_wrong_split_kws(self, df_scales, df_cat, df_parts, wrong_input_cpp):
        with pytest.raises(ValueError):
            CPP(df_scales=df_scales, df_cat=df_cat, df_parts=df_parts, split_kws=wrong_input_cpp)


class TestAddStat:
    """Test adding statistics of features to DataFrame"""

    # Positive unit tests
    def test_add_stat(self, cpp, df_feat, labels):
        assert isinstance(cpp.add_stat(df_feat=df_feat, labels=labels, parametric=True), pd.DataFrame)
        assert isinstance(cpp.add_stat(df_feat=df_feat, labels=labels, parametric=False), pd.DataFrame)
        df_feat = df_feat[["feature"]]
        assert isinstance(cpp.add_stat(df_feat=df_feat, labels=labels, parametric=True), pd.DataFrame)
        assert isinstance(cpp.add_stat(df_feat=df_feat, labels=labels, parametric=False), pd.DataFrame)

    # Negative unit tests
    def test_wrong_df_feat(self, cpp, labels, wrong_df):
        with pytest.raises(ValueError):
            cpp.add_stat(df_feat=wrong_df, labels=labels)

    def test_corrupted_labels(self, cpp, corrupted_labels, df_feat):
        with pytest.raises(ValueError):
            cpp.add_stat(df_feat=df_feat, labels=corrupted_labels)


class TestAddPositions:
    """Test add_positions method"""

    # Positive unit tests
    def test_add_positions(self, df_feat, cpp):
        df_feat = cpp.add_positions(df_feat=df_feat, tmd_len=30)
        assert isinstance(df_feat, pd.DataFrame)
        assert "positions" in list(df_feat)

    # Property based testing
    @given(tmd_len=some.integers(min_value=15, max_value=100),
           jmd_n_len=some.integers(min_value=5, max_value=20),
           jmd_c_len=some.integers(min_value=5, max_value=20),
           ext_len=some.integers(min_value=1, max_value=4),
           start=some.integers(min_value=0, max_value=50))
    @settings(max_examples=10, deadline=None)
    def test_add_position_tmd_len(self, df_feat_module_scope, df_parts, tmd_len, jmd_n_len, jmd_c_len, ext_len, start):
        cpp = CPP(df_parts=df_parts)
        df_feat = cpp.add_positions(df_feat=df_feat_module_scope, tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                    ext_len=ext_len, start=start)
        assert isinstance(df_feat, pd.DataFrame)

    # Negative unit tests
    def test_wrong_tmd_len(self, df_feat, cpp, wrong_input):
        with pytest.raises(ValueError):
            cpp.add_positions(df_feat=df_feat, tmd_len=wrong_input)

    def test_wrong_jmd_len(self, df_feat, cpp, wrong_input):
        with pytest.raises(ValueError):
            cpp.add_positions(df_feat=df_feat, jmd_n_len=wrong_input)
        with pytest.raises(ValueError):
            cpp.add_positions(df_feat=df_feat, jmd_c_len=wrong_input)

    def test_wrong_ext_len(self, df_feat, cpp, wrong_input):
        with pytest.raises(ValueError):
            cpp.add_positions(df_feat=df_feat, ext_len=wrong_input)
        # ext_len >= jmd_n_len or jmd_c_len
        with pytest.raises(ValueError):
            cpp.add_positions(df_feat=df_feat, ext_len=5, jmd_n_len=3)
        with pytest.raises(ValueError):
            cpp.add_positions(df_feat=df_feat, ext_len=5, jmd_c_len=3)

    def test_wrong_start(self, df_feat, cpp, wrong_input):
        with pytest.raises(ValueError):
            cpp.add_positions(df_feat=df_feat, start=wrong_input)
        with pytest.raises(ValueError):
            cpp.add_positions(df_feat=df_feat, start=-4)


class TestAddScaleCategory:
    """Test add_positions method"""

    # Positive unit tests
    def test_add_scale_category(self, df_feat, cpp):
        assert df_feat.equals(cpp.add_scale_info(df_feat=df_feat))
        df_no_cat = df_feat.drop([ut.COL_CAT, ut.COL_SUBCAT], axis=1)
        df_with_cat = cpp.add_scale_info(df_feat=df_no_cat)
        assert df_feat.equals(df_with_cat)

    # Negative unit tests
    def test_wrong_input(self, cpp, wrong_input):
        with pytest.raises(ValueError):
            cpp.add_scale_info(df_feat=wrong_input)

    def test_missing_feature(self, cpp, df_feat):
        df_no_cat = df_feat.drop([ut.COL_CAT, ut.COL_SUBCAT, ut.COL_FEATURE], axis=1)
        with pytest.raises(ValueError):
            cpp.add_scale_info(df_feat=df_no_cat)


class TestAddFeatureImpact:
    """Test adding feature impact to feature DataFrame"""

    # Positive unit tests
    def test_add_feat_impact(self, cpp, df_feat, df_parts, df_scales, labels):
        from sklearn.ensemble import RandomForestClassifier
        import shap
        sf = SequenceFeature()
        X = sf.feat_matrix(features=list(df_feat["feature"]), df_parts=df_parts, df_scales=df_scales)
        assert isinstance(X, np.ndarray)
        model = RandomForestClassifier().fit(X=X, y=labels)
        # compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X, y=labels)
        df_feat["shap_value"] = shap_values[1][0]
        df_feat = cpp.add_shap(df_feat=df_feat)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = cpp.add_shap(df_feat=df_feat, name_feat_impact="Test")
        assert isinstance(df_feat, pd.DataFrame)
        df_feat["shap_value"] = [0.4] * (len(df_feat) - 1) + [np.NaN]
        df_feat = cpp.add_shap(df_feat=df_feat, name_feat_impact="test")
        assert isinstance(df_feat, pd.DataFrame)

    # Negative unit tests
    def test_wrong_shap_value(self, cpp, df_feat):
        with pytest.raises(ValueError):
            df_feat = cpp.add_shap(df_feat=df_feat)
        df_feat["shap_value"] = "wrong"
        with pytest.raises(ValueError):
            df_feat = cpp.add_shap(df_feat=df_feat)


class TestAddSampleDif:
    """Test adding differences of sample and reference mean to feature DataFrame"""

    # Positive unit tests
    def test_add_sample(self, df_feat, df_seq, labels, cpp):
        list_names = list(df_seq[ut.COL_NAME])[0:2]
        ref_group = 0
        # Test all names
        for prot_name in list_names:
            df_feat = cpp.add_sample_dif(df_feat=df_feat, df_seq=df_seq, labels=labels,
                                         sample_name=prot_name, ref_group=ref_group)
            assert isinstance(df_feat, pd.DataFrame)

    # Negative unit tests
    def test_wrong_input(self, df_feat, df_seq, labels, cpp):
        args = dict(df_feat=df_feat, df_seq=df_seq, labels=labels)
        name = "A4_HUMAN"
        ref_group = 0
        with pytest.raises(ValueError):
            cpp.add_sample_dif(**args, sample_name=name.lower(), ref_group=ref_group)
        with pytest.raises(ValueError):
            cpp.add_sample_dif(**args, sample_name=1, ref_group=ref_group)
        with pytest.raises(ValueError):
            cpp.add_sample_dif(**args, sample_name=name, ref_group=5)
        with pytest.raises(ValueError):
            cpp.add_sample_dif(**args, sample_name=name, ref_group=[0, 1])

    def test_corrupted_df_seq(self, df_feat, wrong_df, labels, cpp):
        name = "A4_HUMAN"
        ref_group = 0
        with pytest.raises(ValueError):
            cpp.add_sample_dif(df_feat=df_feat, df_seq=wrong_df,
                               labels=labels, sample_name=name, ref_group=ref_group)

    def test_corrupted_labels(self, df_feat, df_seq, corrupted_labels, cpp):
        name = "A4_HUMAN"
        ref_group = 0
        with pytest.raises(ValueError):
            cpp.add_sample_dif(df_feat=df_feat, df_seq=df_seq,
                               labels=corrupted_labels, sample_name=name, ref_group=ref_group)


class TestRun:
    """Test add_positions method"""

    # Positive unit tests
    def test_cpp_run(self):
        sf = SequenceFeature()
        df_seq = sf.load_sequences(n_in_class=2)
        labels = [1 if x == "SUBEXPERT" else 0 for x in df_seq["class"]]
        df_parts = sf.get_df_parts(df_seq=df_seq)
        df_cat = sf.load_categories()
        df_scales = sf.load_scales()
        list_scales = list(df_scales)[0:2]
        df_scales = df_scales[list_scales]
        cpp = CPP(df_parts=df_parts, df_scales=df_scales, df_cat=df_cat)
        args = dict(verbose=False, labels=labels)
        assert isinstance(cpp.run(**args), pd.DataFrame)
        """
        assert isinstance(cpp.run(parametric=True, **args), pd.DataFrame)
        assert isinstance(cpp.run(n_filter=1000, **args), pd.DataFrame)
        assert isinstance(cpp.run(n_pre_filter=1000, **args), pd.DataFrame)
        assert isinstance(cpp.run(accept_gaps=True, **args), pd.DataFrame)
        assert isinstance(cpp.run(pct_pre_filter=20, **args), pd.DataFrame)
        """

    # Negative unit tests
    def test_corrupted_labels(self, cpp, corrupted_labels):
        with pytest.raises(ValueError):
            cpp.run(verbose=False, labels=corrupted_labels)

    def test_wrong_n_filter(self, cpp, labels):
        for n in ["a", -3, list(), np.NaN]:
            with pytest.raises(ValueError):
                cpp.run(verbose=False, labels=labels, n_filter=n)
        # Should be non-negative int > 1 and not None
        for n in [-1, 0, -100, 0.5, None]:
            with pytest.raises(ValueError):
                cpp.run(verbose=False, labels=labels, n_filter=n)

    def test_wrong_n_pre_filter(self, cpp, labels):
        for n in ["a", -3, list(), np.NaN]:
            with pytest.raises(ValueError):
                cpp.run(verbose=False, labels=labels, n_pre_filter=n)
        # Should be non-negative int > 1 (None accepted)
        for n in [-1, 0, -100, 0.5]:
            with pytest.raises(ValueError):
                cpp.run(verbose=False, labels=labels, n_pre_filter=n)

    def test_wrong_pct_pre_filter(self, cpp, labels):
        for n in ["a", -3, list(), np.NaN]:
            with pytest.raises(ValueError):
                cpp.run(verbose=False, labels=labels, pct_pre_filter=n)
        # Should be non-negative int >= 5 and not None
        for n in [-1, 0, -100, 0.5, 4, 3, None]:
            with pytest.raises(ValueError):
                cpp.run(verbose=False, labels=labels, pct_pre_filter=n)

    def test_wrong_max_std(self, cpp, labels):
        for n in ["a", -3, list(), np.NaN]:
            with pytest.raises(ValueError):
                cpp.run(verbose=False, labels=labels, max_std_test=n)
        # Should be non-negative int >= 5 and not None
        for n in [-1, 100, -100, 4, 3, None]:
            with pytest.raises(ValueError):
                cpp.run(verbose=False, labels=labels, max_std_test=n)


class TestGetDfPos:
    """Test common interface of CPP plotting methods
    """

    # Positive unit tests
    def test_get_df_pos(self, df_feat, df_cat):
        df_pos = _get_df_pos(df_feat=df_feat, df_cat=df_cat)
        assert isinstance(df_pos, pd.DataFrame)
        for i in ["count", "mean", "sum", "std"]:
            assert isinstance(_get_df_pos(df_feat=df_feat, df_cat=df_cat, value_type=i), pd.DataFrame)

    # Property based testing
    @given(tmd_len=some.integers(min_value=15, max_value=100),
           jmd_n_len=some.integers(min_value=5, max_value=20),
           jmd_c_len=some.integers(min_value=5, max_value=20),
           start=some.integers(min_value=0, max_value=50))
    @settings(max_examples=10, deadline=None)
    def test_get_df_pos_len(self, df_feat_module_scope, df_cat, tmd_len, jmd_n_len, jmd_c_len, start):
        df_pos = _get_df_pos(df_feat=df_feat_module_scope, df_cat=df_cat,
                             tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                             start=start)
        assert isinstance(df_pos, pd.DataFrame)

    # Negative unit tests
    def test_wrong_value_type(self, df_feat, df_cat):
        with pytest.raises(ValueError):
            _get_df_pos(df_feat=df_feat, df_cat=df_cat, tmd_len=wrong_input)

    def test_wrong_tmd_len(self, df_feat, df_cat, wrong_input):
        with pytest.raises(ValueError):
            _get_df_pos(df_feat=df_feat, df_cat=df_cat, tmd_len=wrong_input)

    def test_wrong_jmd_len(self, df_feat, df_cat, wrong_input):
        with pytest.raises(ValueError):
            _get_df_pos(df_feat=df_feat, df_cat=df_cat, jmd_n_len=wrong_input)
        with pytest.raises(ValueError):
            _get_df_pos(df_feat=df_feat, df_cat=df_cat, jmd_c_len=wrong_input)

    def test_wrong_start(self, df_feat, df_cat, wrong_input):
        with pytest.raises(ValueError):
            _get_df_pos(df_feat=df_feat, start=wrong_input)
        with pytest.raises(ValueError):
            _get_df_pos(df_feat=df_feat, df_cat=df_cat, start=-4)

    def test_wrong_normalize(self, df_feat, df_cat, wrong_input):
        with pytest.raises(ValueError):
            _get_df_pos(df_feat=df_feat, df_cat=df_cat, normalize=wrong_input)

    def test_wrong_value_type(self, df_feat, df_cat, wrong_input):
        with pytest.raises(ValueError):
            _get_df_pos(df_feat=df_feat, df_cat=df_cat, value_type=wrong_input)

    def test_wrong_value_col(self, df_feat, df_cat, wrong_input):
        with pytest.raises(ValueError):
            _get_df_pos(df_feat=df_feat, df_cat=df_cat, val_col=wrong_input)

    def test_wrong_y(self, df_feat, df_cat, wrong_input):
        with pytest.raises(ValueError):
            _get_df_pos(df_feat=df_feat, df_cat=df_cat, y=wrong_input)


class TestPlotMethods:
    """General test for plotting methods (using heatmap)"""

    # Positive & Negative unit tests
    def test_df_feat(self, df_feat, cpp):
        """Positive unit Test main arguments: df_feat, y, val_col, value_type, normalize"""
        assert isinstance(cpp.plot_heatmap(df_feat=df_feat), mpl.axes.Axes)
        for y in ["category", "subcategory", "scale_name"]:
            assert isinstance(cpp.plot_heatmap(df_feat=df_feat, y=y), mpl.axes.Axes)
        for val_col in ["abs_auc", "abs_mean_dif", "mean_dif", "std_test", "p_val_fdr_bh"]:
            assert isinstance(cpp.plot_heatmap(df_feat=df_feat, val_col=val_col), mpl.axes.Axes)
        for val_type in ["sum", "mean", "std"]:
            for normalize in [True, False, "positions"]:
                assert isinstance(cpp.plot_heatmap(df_feat=df_feat, val_type=val_type, normalize=normalize),
                                  mpl.axes.Axes)

    def test_wrong_df_feat(self, df_feat, cpp):
        for y in ["categorY", "sub__category", "Scale", "feature", 1, list, "abs_mean_dif"]:
            with pytest.raises(ValueError):
                cpp.plot_heatmap(df_feat=df_feat, y=y)
        for val_col in ["subcategory", "Abs_mean_dif", "p_val", 1, list]:
            with pytest.raises(ValueError):
                cpp.plot_heatmap(df_feat=df_feat, val_col=val_col)
        for val_type in ["SUM", "man", 1, 2]:
            for normalize in ["positions", True]:
                with pytest.raises(ValueError):
                    cpp.plot_heatmap(df_feat=df_feat, val_type=val_type, normalize=normalize)

    def test_plotting(self, df_feat, df_parts):
        """Test main plotting arguments: figsize, title, title_kws"""
        cpp = CPP(df_parts=df_parts)
        # Figsize and title checked by matplotlib
        title_kws = {'fontsize': 11,
                     'fontweight': "bold",
                     'verticalalignment': 'baseline',
                     'horizontalalignment': "center"}
        assert isinstance(cpp.plot_heatmap(df_feat=df_feat, title="Test", title_kws=title_kws)
                          , mpl.axes.Axes)

    def test_figsize(self, df_feat, df_parts):
        """Test figsize"""
        cpp = CPP(df_parts=df_parts)
        args = dict(df_feat=df_feat, figsize=(10, 5))
        assert isinstance(cpp.plot_heatmap(**args), mpl.axes.Axes)
        assert isinstance(cpp.plot_bargraph(**args), mpl.axes.Axes)
        assert isinstance(cpp.plot_profile(**args), mpl.axes.Axes)

    def test_wrong_figsize(self, df_feat, df_parts):
        """Test wrong figsize"""
        cpp = CPP(df_parts=df_parts)
        for figsize in [(0, 10), "a", [1, 2], (10, "a")]:
            args = dict(df_feat=df_feat, figsize=figsize)
            with pytest.raises(ValueError):
                cpp.plot_heatmap(**args)
            with pytest.raises(ValueError):
                cpp.plot_bargraph(**args)
            with pytest.raises(ValueError):
                cpp.plot_profile(**args)

    def test_dict_color(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        sf = SequenceFeature()
        dict_color = sf.load_colors()
        assert isinstance(cpp.plot_heatmap(df_feat=df_feat, dict_color=dict_color)
                          , mpl.axes.Axes)

    def test_wrong_dict_color(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        sf = SequenceFeature()
        dict_color = sf.load_colors()
        for i in [1, dict(), "asdf", 0.1]:
            with pytest.raises(ValueError):
                cpp.plot_heatmap(df_feat=df_feat, dict_color=i)
        dict_color["Composition"] = 1
        with pytest.raises(ValueError):
            cpp.plot_heatmap(df_feat=df_feat, dict_color=dict_color)
        dict_color = {"Composition": "blue"}
        with pytest.raises(ValueError):
            cpp.plot_heatmap(df_feat=df_feat, dict_color=dict_color)

    def test_sequences(self, df_feat, df_parts):
        """Test sequence input: tmd_seq, jmd_n_seq, jmd_c_seq"""
        # Length input tested in TestGetDfPos
        jmd_c_seq = "AAAAAAAAAAa"
        jmd_n_seq = "aa"*10
        cpp = CPP(df_parts=df_parts, jmd_n_len=len(jmd_n_seq), jmd_c_len=len(jmd_c_seq))
        assert isinstance(cpp.plot_heatmap(df_feat=df_feat,
                                           tmd_seq="AA"*16, jmd_c_seq=jmd_c_seq, jmd_n_seq=jmd_n_seq)
                          , mpl.axes.Axes)

    def test_wrong_sequences(self, df_feat, df_parts):
        """Test sequence input: tmd_seq, jmd_n_seq, jmd_c_seq"""
        # Length input tested in TestGetDfPos
        cpp = CPP(df_parts=df_parts)
        wrong_seq = [1, None, list, dict]
        tmd_seq = "A" * 20
        jmd_c_seq = "B" * 10
        jmd_n_seq = "C" * 10
        for w in wrong_seq:
            with pytest.raises(ValueError):
                cpp.plot_heatmap(df_feat=df_feat, tmd_seq=w, jmd_c_seq=jmd_c_seq, jmd_n_seq=jmd_n_seq)
            with pytest.raises(ValueError):
                cpp.plot_heatmap(df_feat=df_feat, tmd_seq=tmd_seq, jmd_c_seq=w, jmd_n_seq=jmd_n_seq)
            with pytest.raises(ValueError):
                cpp.plot_heatmap(df_feat=df_feat, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq, jmd_n_seq=w)
        with pytest.raises(ValueError):
            cpp.plot_heatmap(df_feat=df_feat, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
        with pytest.raises(ValueError):
            cpp.plot_heatmap(df_feat=df_feat, tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq)
        with pytest.raises(ValueError):
            cpp.plot_heatmap(df_feat=df_feat, jmd_c_seq=jmd_c_seq, jmd_n_seq=jmd_n_seq)
        jmd_c_seq = "AAAAAAAAAAa"
        jmd_n_seq = "aa"*10
        with pytest.raises(ValueError):
            cpp.plot_heatmap(df_feat=df_feat, jmd_c_seq=jmd_c_seq, jmd_n_seq=jmd_n_seq)

    def test_size(self, df_feat, df_parts):
        """Test size input: seq_size, tmd_fontsize, jmd_fontsize"""
        cpp = CPP(df_parts=df_parts)
        assert isinstance(cpp.plot_heatmap(df_feat=df_feat,
                                           tmd_seq=11, jmd_fontsize=12, tmd_fontsize=11)
                          , mpl.axes.Axes)
        # Simple check function -> No negative test

    def test_color(self, df_feat, df_parts):
        """Test color input: tmd_color, jmd_color, tmd_seq_color, jmd_seq_color"""
        cpp = CPP(df_parts=df_parts)
        args = dict(df_feat=df_feat, tmd_color="b")
        assert isinstance(cpp.plot_heatmap(**args), mpl.axes.Axes)
        assert isinstance(cpp.plot_heatmap(**args), mpl.axes.Axes)
        args = dict(df_feat=df_feat, jmd_seq_color="b")
        assert isinstance(cpp.plot_heatmap(**args), mpl.axes.Axes)
        assert isinstance(cpp.plot_heatmap(**args), mpl.axes.Axes)
        # Simple check function -> No negative test

    def test_ticks(self, df_feat, df_parts):
        """Test xtick input: xtick_size, xtick_width, xtick_length"""
        cpp = CPP(df_parts=df_parts)
        assert isinstance(cpp.plot_heatmap(df_feat=df_feat, xtick_size=11, xtick_width=2,
                                           xtick_length=5, ytick_size=11)
                          , mpl.axes.Axes)
        # Simple check function -> No negative test

    def test_legend(self, df_feat, df_parts):
        """Test legend args for heatmap and profile: add_legend_cat, legend_kws"""
        cpp = CPP(df_parts=df_parts)
        assert isinstance(cpp.plot_heatmap(df_feat=df_feat, legend_kws=dict(fontsize=11))
                          , mpl.axes.Axes)
        # Simple check function -> No negative test


class TestPlotHeatmap:
    """Test additional interface of heatmap"""

    # Positive and negative unit tests
    def test_vmin_vmax(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        for vmin, vmax in zip([-10, -5, 1, 2, 0.1], [0, 10, 2, 3, 0.2]):
            args = dict(df_feat=df_feat, vmin=vmin, vmax=vmax)
            assert isinstance(cpp.plot_heatmap(**args), mpl.axes.Axes)

    def test_wrong_vmin_vmax(self, df_feat, cpp):
        for vmin, vmax in zip([1, "a", -10, 2], [0, 1, -100, "2"]):
            with pytest.raises(ValueError):
                cpp.plot_heatmap(df_feat=df_feat, vmin=vmin, vmax=vmax)


class TestPlotGraphProfile:
    """Test additional interface of bargraph"""

    # Positive and negative unit tests
    def test_color(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        args = dict(df_feat=df_feat, bar_color="r", edge_color="b")
        assert isinstance(cpp.plot_bargraph(**args), mpl.axes.Axes)
        args = dict(df_feat=df_feat, edge_color="b")
        assert isinstance(cpp.plot_profile(**args), mpl.axes.Axes)

    def test_wrong_color(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        args = dict(df_feat=df_feat, bar_color="a", edge_color=1)
        with pytest.raises(ValueError):
            cpp.plot_bargraph(**args)
        args = dict(df_feat=df_feat, edge_color=1)
        with pytest.raises(ValueError):
            cpp.plot_profile(**args)

    def test_ylim(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        args = dict(df_feat=df_feat, ylim=(0, 100))
        assert isinstance(cpp.plot_bargraph(**args), mpl.axes.Axes)
        assert isinstance(cpp.plot_profile(**args), mpl.axes.Axes)

    def test_wrong_ylim(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        for ylim in [1, "a", [1, 40], (0), (0, 2), (-10, "a")]:
            args = dict(df_feat=df_feat, ylim=ylim)
            with pytest.raises(ValueError):
                cpp.plot_bargraph(**args)
            with pytest.raises(ValueError):
                cpp.plot_profile(**args)

    def test_highlight_alpha(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        args = dict(df_feat=df_feat, highlight_alpha=0.5)
        assert isinstance(cpp.plot_bargraph(**args), mpl.axes.Axes)
        assert isinstance(cpp.plot_profile(**args), mpl.axes.Axes)

    def test_wrong_highlight_alpha(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        for i in ["a", 10, list]:
            args = dict(df_feat=df_feat, highlight_alpha=i)
            with pytest.raises(ValueError):
                cpp.plot_bargraph(**args)
            with pytest.raises(ValueError):
                cpp.plot_profile(**args)

    def test_grid_axis(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        for grid_axis in ["x", "y", "both"]:
            args = dict(df_feat=df_feat, grid_axis=grid_axis)
            assert isinstance(cpp.plot_bargraph(**args), mpl.axes.Axes)
            assert isinstance(cpp.plot_profile(**args), mpl.axes.Axes)

    def test_wrong_grid_axis(self, df_feat, df_parts):
        cpp = CPP(df_parts=df_parts)
        for grid_axis in ["X", 1, None, list, "XY"]:
            args = dict(df_feat=df_feat, grid_axis=grid_axis)
            with pytest.raises(ValueError):
                cpp.plot_bargraph(**args)
            with pytest.raises(ValueError):
                cpp.plot_profile(**args)


class TestPlotStat:
    """Test additional interface of stat plot"""

    # Positive unit tests

    # Negative unit tests


# II Regression/Functional test
def test_add_pipeline(df_feat):
    # TODO check
    sf = SequenceFeature()
    df_seq = sf.load_sequences(n_in_class=50)
    labels = [1 if x == "SUBEXPERT" else 0 for x in df_seq["class"]]
    df_parts = sf.get_df_parts(df_seq=df_seq)
    cpp = CPP(df_parts=df_parts)
    df = df_feat.copy()
    cols = [x for x in list(df) if "p_val" not in x]
    df = cpp.add_scale_info(df_feat=df)
    assert df_feat[cols].equals(df[cols])
    df = cpp.add_stat(df_feat=df, labels=labels)
    df = cpp.add_positions(df_feat=df)
    assert df_feat[cols].equals(df[cols])
    df = cpp.add_scale_info(df_feat=df)
    assert df_feat[cols].equals(df[cols])
    df = cpp.add_positions(df_feat=df)
    df = cpp.add_stat(df_feat=df, labels=labels)
    assert df_feat[cols].equals(df[cols])
    df = cpp.add_positions(df_feat=df)
    assert df_feat[cols].equals(df[cols])


def test_cpp_pipeline():
    pass


def test_cpp_with_shap():
    pass

"""This is a script to test the AFragmenter backend wrapper functions
(check_afragmenter_available / _domains_to_chopping / run_afragmenter_on_pae).

These are backend helpers under ``data_handling_pro/_backend/struct_preproc``.
``afragmenter`` is an optional pro dependency that may not be installed; the
wrapper lazy-imports it. To cover the wrapper's clustering-dispatch and
output-normalization internals deterministically we inject a fake AFragmenter
module via ``_try_import_afragmenter`` — a deliberate, narrow exception to the
otherwise frontend-driven testing convention.
"""
from unittest.mock import patch

import pytest

from aaanalysis.data_handling_pro._backend.struct_preproc._afragmenter import (
    check_afragmenter_available,
    run_afragmenter_on_pae,
    _domains_to_chopping,
    _try_import_afragmenter,
)

MODULE = "aaanalysis.data_handling_pro._backend.struct_preproc._afragmenter"
IMPORTER = f"{MODULE}._try_import_afragmenter"


# I Helper Functions
class _FakeSeg:
    """Stand-in for an AFragmenter instance built from a PAE path."""

    def __init__(self, domains, method="cluster"):
        self._domains = domains
        self._method = method

    def cluster(self, resolution=0.7):
        if self._method != "cluster":
            raise AttributeError("cluster")
        return self._domains

    def run_clustering(self, resolution=0.7):
        return self._domains


class _FakeSegNoMethod:
    """AFragmenter instance exposing neither .cluster nor .run_clustering."""

    def __init__(self, *a, **k):
        pass


def _fake_module(domains, method="cluster", seg_cls=None):
    """Build a fake ``afragmenter`` module whose AFragmenter(...) returns a seg."""

    class _Mod:
        __name__ = "afragmenter"

        @staticmethod
        def AFragmenter(path, threshold=2.0):
            cls = seg_cls or _FakeSeg
            if seg_cls is _FakeSegNoMethod:
                return _FakeSegNoMethod()
            return _FakeSeg(domains, method=method)

    return _Mod


def _module_without_class():
    class _Mod:
        __name__ = "afragmenter"

    return _Mod


# II Test Classes
class TestTryImportAfragmenter:
    """Exercise the real lazy-importer (not the patched stub)."""

    def test_returns_module_or_none(self):
        # Calls the genuine importlib path: a module if 'afragmenter' is
        # installed, else None via the ImportError branch. Either is valid.
        out = _try_import_afragmenter()
        assert out is None or hasattr(out, "__name__")

    def test_import_error_returns_none(self):
        # Force the ImportError branch regardless of whether afragmenter is
        # actually installed in this environment.
        with patch("importlib.import_module", side_effect=ImportError("no afr")):
            assert _try_import_afragmenter() is None


class TestCheckAfragmenterAvailable:
    """check_afragmenter_available: present is silent, absent raises."""

    # ----- POSITIVES -----
    def test_valid_present_no_raise(self):
        with patch(IMPORTER, return_value=object()):
            assert check_afragmenter_available() is None

    def test_valid_present_module_like(self):
        with patch(IMPORTER, return_value=_fake_module([])):
            check_afragmenter_available()  # no exception

    # ----- NEGATIVES -----
    def test_invalid_missing_raises(self):
        with patch(IMPORTER, return_value=None):
            with pytest.raises(RuntimeError, match="not installed"):
                check_afragmenter_available()

    def test_invalid_missing_hint_has_install(self):
        with patch(IMPORTER, return_value=None):
            with pytest.raises(RuntimeError, match=r"aaanalysis\[pro\]"):
                check_afragmenter_available()


class TestDomainsToChopping:
    """_domains_to_chopping: format list-of-list-of-(start,end)."""

    # ----- POSITIVES -----
    def test_valid_single_contiguous(self):
        assert _domains_to_chopping([[(1, 50)]]) == "1-50"

    def test_valid_two_domains(self):
        assert _domains_to_chopping([[(1, 50)], [(55, 120)]]) == "1-50,55-120"

    def test_valid_discontinuous_segments(self):
        out = _domains_to_chopping([[(1, 50), (60, 80)], [(90, 120)]])
        assert out == "1-50_60-80,90-120"

    def test_valid_empty_domains(self):
        assert _domains_to_chopping([]) == ""

    def test_valid_skips_empty_segment_list(self):
        assert _domains_to_chopping([[], [(5, 9)]]) == "5-9"

    # ----- NEGATIVES -----
    def test_invalid_non_iterable_pair(self):
        with pytest.raises((TypeError, ValueError)):
            _domains_to_chopping([[(1,)]])  # not a 2-tuple unpack

    def test_invalid_none_input(self):
        with pytest.raises(TypeError):
            _domains_to_chopping(None)


class TestRunAfragmenterOnPae:
    """run_afragmenter_on_pae: dispatch + normalization + error branches."""

    # ----- POSITIVES -----
    def test_valid_cluster_method(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        mod = _fake_module([[(1, 50)], [(55, 120)]], method="cluster")
        with patch(IMPORTER, return_value=mod):
            out = run_afragmenter_on_pae(pae)
        assert out == "1-50,55-120"

    def test_valid_run_clustering_fallback(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        mod = _fake_module([[(1, 30)]], method="run_clustering")

        class _Seg:
            def run_clustering(self, resolution=0.7):
                return [[(1, 30)]]

        class _Mod:
            __name__ = "afragmenter"

            @staticmethod
            def AFragmenter(path, threshold=2.0):
                return _Seg()

        with patch(IMPORTER, return_value=_Mod):
            out = run_afragmenter_on_pae(pae)
        assert out == "1-30"

    def test_valid_flat_tuple_domains_normalized(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        # Upstream returns a flat list of (start, end) tuples (one per domain).
        mod = _fake_module([(1, 50), (55, 120)])
        with patch(IMPORTER, return_value=mod):
            out = run_afragmenter_on_pae(pae)
        assert out == "1-50,55-120"

    def test_valid_empty_domains(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        mod = _fake_module([])
        with patch(IMPORTER, return_value=mod):
            out = run_afragmenter_on_pae(pae)
        assert out == ""

    def test_valid_custom_resolution_threshold(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        mod = _fake_module([[(1, 10)]])
        with patch(IMPORTER, return_value=mod):
            out = run_afragmenter_on_pae(pae, resolution=1.2, threshold=3.0)
        assert out == "1-10"

    # ----- NEGATIVES -----
    def test_invalid_not_installed(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        with patch(IMPORTER, return_value=None):
            with pytest.raises(RuntimeError, match="not installed"):
                run_afragmenter_on_pae(pae)

    def test_invalid_no_afragmenter_class(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        with patch(IMPORTER, return_value=_module_without_class()):
            with pytest.raises(RuntimeError, match="no AFragmenter class"):
                run_afragmenter_on_pae(pae)

    def test_invalid_no_cluster_methods(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        mod = _fake_module([], seg_cls=_FakeSegNoMethod)
        with patch(IMPORTER, return_value=mod):
            with pytest.raises(RuntimeError, match="neither"):
                run_afragmenter_on_pae(pae)

    def test_invalid_non_list_return(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        mod = _fake_module("not-a-list")
        with patch(IMPORTER, return_value=mod):
            with pytest.raises(RuntimeError, match="expected list"):
                run_afragmenter_on_pae(pae)

    def test_invalid_unknown_domain_item(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        mod = _fake_module([42])  # neither a (s,e) tuple nor a list
        with patch(IMPORTER, return_value=mod):
            with pytest.raises(RuntimeError, match="not understood"):
                run_afragmenter_on_pae(pae)


class TestRunAfragmenterComplex:
    """Cross-cutting combinations for run_afragmenter_on_pae."""

    def test_complex_clustering_raises_wrapped(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")

        class _Seg:
            def cluster(self, resolution=0.7):
                raise ValueError("internal boom")

        class _Mod:
            __name__ = "afragmenter"

            @staticmethod
            def AFragmenter(path, threshold=2.0):
                return _Seg()

        with patch(IMPORTER, return_value=_Mod):
            with pytest.raises(RuntimeError, match="AFragmenter failed"):
                run_afragmenter_on_pae(pae)

    def test_complex_mixed_flat_and_nested(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        # one flat tuple domain + one nested multi-segment domain
        mod = _fake_module([(1, 20), [(30, 40), (45, 50)]])
        with patch(IMPORTER, return_value=mod):
            out = run_afragmenter_on_pae(pae)
        assert out == "1-20,30-40_45-50"

    def test_complex_int_coercion(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")
        # numpy-like float coords should coerce to int in the chopping string.
        mod = _fake_module([[(1.0, 50.0)]])
        with patch(IMPORTER, return_value=mod):
            out = run_afragmenter_on_pae(pae)
        assert out == "1-50"

    def test_complex_alias_lowercase_class(self, tmp_path):
        pae = tmp_path / "P1.json"
        pae.write_text("{}")

        class _Seg:
            def cluster(self, resolution=0.7):
                return [[(1, 9)]]

        class _Mod:
            __name__ = "afragmenter"
            afragmenter = staticmethod(lambda path, threshold=2.0: _Seg())

        # No capital-A AFragmenter attr -> falls back to lowercase alias.
        with patch(IMPORTER, return_value=_Mod):
            out = run_afragmenter_on_pae(pae)
        assert out == "1-9"

    def test_complex_wrapped_error_includes_path(self, tmp_path):
        pae = tmp_path / "WEIRD.json"
        pae.write_text("{}")

        class _Seg:
            def cluster(self, resolution=0.7):
                raise RuntimeError("x")

        class _Mod:
            __name__ = "afragmenter"

            @staticmethod
            def AFragmenter(path, threshold=2.0):
                return _Seg()

        with patch(IMPORTER, return_value=_Mod):
            with pytest.raises(RuntimeError, match="WEIRD.json"):
                run_afragmenter_on_pae(pae)

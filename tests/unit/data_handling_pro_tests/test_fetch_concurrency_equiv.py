"""Same-output tests for the opt-in concurrent fetch path (issue #180 Part B).

The pro web-fetch backends gained a ``max_workers`` thread-pool option behind a
shared, order-preserving runner (``_fetch.run_in_order_``). These tests prove:

* ``run_in_order_`` preserves input order independent of completion time and is
  byte-identical serial vs. concurrent, and propagates the first exception;
* ``fetch_alphafold_bulk`` / ``fetch_and_map`` return identical results (status
  table / ``df_annot`` and on-disk files) at any worker count, in input order;
* the frontend ``max_workers`` validation rejects bad values.

Per-accession mocks (a side-effect *function*, not a call-order list) keep the
expected response deterministic regardless of the order threads make calls.
"""
import time
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis import StructurePreprocessor, AnnotationPreprocessor
from aaanalysis.data_handling_pro._backend import _fetch
from aaanalysis.data_handling_pro._backend.struct_preproc import _alphafold as af
from aaanalysis.data_handling_pro._backend.annot_preproc import _uniprot as up

AF_MODULE = "aaanalysis.data_handling_pro._backend.struct_preproc._alphafold"
UP_MODULE = "aaanalysis.data_handling_pro._backend.annot_preproc._uniprot"
SEQ = "ACDEFGHIKLMNPQRSTVWY"


# I Helper Functions
def _resp(status=200, content=b"DATA"):
    m = MagicMock()
    m.status_code = status
    m.content = content
    return m


def _af_http(url, timeout=30.0):
    """Deterministic per-URL response (content = URL bytes), thread-safe."""
    return _resp(200, content=url.encode())


def _af_resolve(entry, file_format, timeout):
    """Per-entry model/PAE URLs so each accession's files are distinct."""
    return f"https://af/{entry}/model", f"https://af/{entry}/pae"


def _uniprot_record(acc, timeout=30.0):
    """Deterministic per-accession UniProt record: one phospho site."""
    pos = (int(acc[1:]) % len(SEQ)) + 1
    feat = {"type": "Modified residue", "description": "Phosphoserine",
            "location": {"start": {"value": pos}, "end": {"value": pos}},
            "evidences": [{"evidenceCode": "ECO:0000269"}]}
    return {"sequence": {"value": SEQ}, "features": [feat]}


# II Test Classes
class TestRunInOrder:
    """The shared order-preserving, opt-in-concurrent runner."""

    def test_serial_none_preserves_order(self):
        assert _fetch.run_in_order_(lambda x: x * 2, [1, 2, 3, 4],
                                    max_workers=None) == [2, 4, 6, 8]

    def test_serial_one_preserves_order(self):
        assert _fetch.run_in_order_(lambda x: x * 2, [1, 2, 3, 4],
                                    max_workers=1) == [2, 4, 6, 8]

    @pytest.mark.parametrize("max_workers", [2, 4, 8])
    def test_concurrent_matches_serial(self, max_workers):
        items = list(range(20))
        serial = _fetch.run_in_order_(lambda x: x * x, items, max_workers=1)
        concurrent = _fetch.run_in_order_(lambda x: x * x, items,
                                          max_workers=max_workers)
        assert concurrent == serial == [x * x for x in items]

    def test_order_independent_of_completion_time(self):
        # Earlier items sleep the longest; the output must still be in input
        # order, proving order comes from the submission index, not completion.
        def slow(i):
            time.sleep(0.01 * (5 - i))
            return i
        assert _fetch.run_in_order_(slow, [0, 1, 2, 3, 4],
                                    max_workers=5) == [0, 1, 2, 3, 4]

    def test_empty_items(self):
        assert _fetch.run_in_order_(lambda x: x, [], max_workers=4) == []

    @pytest.mark.parametrize("max_workers", [1, 4])
    def test_exception_propagates(self, max_workers):
        def boom(x):
            if x == 2:
                raise RuntimeError("kaboom")
            return x
        with pytest.raises(RuntimeError, match="kaboom"):
            _fetch.run_in_order_(boom, [0, 1, 2, 3], max_workers=max_workers)


class TestAlphafoldConcurrencyEquivalence:
    """fetch_alphafold_bulk: concurrent == sequential (status table + files)."""

    @pytest.fixture(autouse=True)
    def _stub_resolve(self):
        with patch(f"{AF_MODULE}._af_resolve_urls", side_effect=_af_resolve):
            yield

    @pytest.mark.parametrize("max_workers", [2, 4, 8])
    def test_identical_status_table_and_order(self, tmp_path, max_workers):
        entries = [f"P{i}" for i in range(12)]
        folder = tmp_path / "af"
        folder.mkdir()
        with patch(f"{AF_MODULE}.http_get_", side_effect=_af_http):
            serial = af.fetch_alphafold_bulk(entries, folder, "pdb", 30.0,
                                             False, False, max_workers=1)
            concurrent = af.fetch_alphafold_bulk(entries, folder, "pdb", 30.0,
                                                 False, False,
                                                 max_workers=max_workers)
        # Byte-identical status table, including row (input) order.
        pd.testing.assert_frame_equal(serial, concurrent)
        assert list(concurrent[ut.COL_ENTRY]) == entries
        assert concurrent["alphafold_ok"].all()

    def test_files_written_with_expected_content(self, tmp_path):
        entries = [f"P{i}" for i in range(6)]
        folder = tmp_path / "af"
        folder.mkdir()
        with patch(f"{AF_MODULE}.http_get_", side_effect=_af_http):
            af.fetch_alphafold_bulk(entries, folder, "pdb", 30.0, False, False,
                                    max_workers=4)
        for entry in entries:
            model = folder / f"{entry}.pdb"
            assert model.is_file()
            assert model.read_bytes() == f"https://af/{entry}/model".encode()


class TestUniprotConcurrencyEquivalence:
    """fetch_and_map: concurrent == sequential (df_annot rows + order)."""

    @pytest.mark.parametrize("max_workers", [2, 4, 8])
    def test_identical_df_and_order(self, max_workers):
        entries = [f"P{i}" for i in range(12)]
        with patch(f"{UP_MODULE}.fetch_uniprot_json", side_effect=_uniprot_record):
            serial = up.fetch_and_map(entries, None, None, 30.0, False,
                                      max_workers=1)
            concurrent = up.fetch_and_map(entries, None, None, 30.0, False,
                                          max_workers=max_workers)
        pd.testing.assert_frame_equal(serial, concurrent)
        # One row per entry, concatenated in input order.
        assert list(concurrent[ut.COL_PROTEIN_ID]) == entries


class TestFrontendMaxWorkers:
    """Frontend plumbing + validation of the new ``max_workers`` parameter."""

    def _df_seq(self, entries=("P1",)):
        return pd.DataFrame({ut.COL_ENTRY: list(entries),
                             ut.COL_SEQ: [SEQ] * len(entries)})

    @pytest.mark.parametrize("bad", [0, -1, 2.5])
    def test_fetch_alphafold_invalid_max_workers_raises(self, tmp_path, bad):
        stp = StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.fetch_alphafold(df_seq=self._df_seq(), out_folder=tmp_path / "o",
                                max_workers=bad)

    @pytest.mark.parametrize("bad", [0, -1, 2.5])
    def test_fetch_uniprot_invalid_max_workers_raises(self, bad):
        ap = AnnotationPreprocessor(verbose=False)
        with pytest.raises(ValueError):
            ap.fetch_uniprot(df_seq=self._df_seq(), max_workers=bad)

    def test_fetch_alphafold_max_workers_plumbs_through(self, tmp_path):
        stp = StructurePreprocessor(verbose=False)
        entries = ["P1", "P2", "P3"]
        with patch(f"{AF_MODULE}._af_resolve_urls", side_effect=_af_resolve), \
                patch(f"{AF_MODULE}.http_get_", side_effect=_af_http):
            df = stp.fetch_alphafold(df_seq=self._df_seq(entries),
                                     out_folder=tmp_path / "o", skip_existing=False,
                                     max_workers=4)
        assert list(df[ut.COL_ENTRY]) == entries
        assert df["alphafold_ok"].all()

    def test_fetch_uniprot_max_workers_plumbs_through(self):
        ap = AnnotationPreprocessor(verbose=False)
        entries = ["P1", "P2", "P3"]
        with patch(f"{UP_MODULE}.fetch_uniprot_json", side_effect=_uniprot_record):
            df = ap.fetch_uniprot(df_seq=self._df_seq(entries), evidence="all",
                                  max_workers=4)
        assert list(df[ut.COL_PROTEIN_ID]) == entries

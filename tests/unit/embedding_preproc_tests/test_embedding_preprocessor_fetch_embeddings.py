"""Tests for EmbeddingPreprocessor.fetch_embeddings / pool_embeddings.

Compute logic is exercised with a deterministic fake model loader (no torch, no
download). A single real-model integration test runs only if the 'embed' extra is
installed and is marked slow."""
import importlib.util

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
from aaanalysis.data_handling._backend.embed_preproc import fetch


HAS_TORCH = importlib.util.find_spec("torch") is not None


def _df(n=3):
    seqs = ["MKVLAA", "MMGGWWKKLL", "ACDEFGHIKLMN"][:n]
    return pd.DataFrame({"entry": [f"P{i}" for i in range(n)],
                         "sequence": seqs, "label": [1, 0, 1][:n]})


# Deterministic fake model loader: hidden[token_pos] == token_pos (broadcast over D)
class _FakeT:
    def __init__(self, a): self.a = np.asarray(a, dtype=float)
    def to(self, device): return self
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


class _NoGrad:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeTorch:
    def no_grad(self): return _NoGrad()


class _FakeTok:
    def __init__(self, has_cls): self.has_cls = has_cls
    def __call__(self, proc, return_tensors=None, padding=None):
        residues = [p.replace(" ", "") for p in proc]   # undo t5 spacing
        extra = 2 if self.has_cls else 1
        max_t = max(len(r) for r in residues) + extra
        return {"input_ids": _FakeT(np.zeros((len(residues), max_t)))}


class _FakeModel:
    def __init__(self, dim, has_cls, fail=False):
        self.dim, self.has_cls, self.fail = dim, has_cls, fail
    def __call__(self, input_ids=None, output_hidden_states=False, **kw):
        if self.fail:
            raise RuntimeError("forced failure")
        b, t = input_ids.a.shape
        h = np.tile(np.arange(t).reshape(1, t, 1), (b, 1, self.dim)).astype(float)
        out = type("Out", (), {})()
        out.last_hidden_state = _FakeT(h)
        out.hidden_states = [_FakeT(h)]
        return out


def _patch_loader(monkeypatch, fail=False):
    def fake_load(mdl, device):
        meta = fetch.REGISTRY[mdl]   # match whatever model the frontend requested
        return _FakeModel(meta["dim"], meta["has_cls"], fail=fail), _FakeTok(meta["has_cls"]), _FakeTorch()
    monkeypatch.setattr(fetch, "_load_model_and_tokenizer", fake_load)


class TestFetchValidation:
    """Validate block — no compute reached."""

    def test_bad_mode(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().fetch_embeddings(_df(), mode="seq")

    def test_bad_model(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().fetch_embeddings(_df(), model="gpt4")

    def test_bad_pooling(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().fetch_embeddings(_df(), pooling="sum")

    def test_bad_source_uniprot_rejected(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().fetch_embeddings(_df(), source="uniprot")

    def test_bad_batch_size(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().fetch_embeddings(_df(), batch_size=0)

    def test_bad_on_failure(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().fetch_embeddings(_df(), on_failure="ignore")

    def test_cls_on_t5_rejected(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().fetch_embeddings(_df(), mode="protein",
                                                        pooling="cls", model="prott5_xl_u50")

    def test_embeddings_ok_collision(self):
        df = _df()
        df["embeddings_ok"] = True
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().fetch_embeddings(df)


class TestFetchComputeFake:
    """Compute logic via the fake loader (no torch, no download)."""

    def test_protein_shape_and_mean(self, monkeypatch):
        _patch_loader(monkeypatch)
        df = _df()
        X = aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(df, mode="protein", model="esm2_t6_8M")
        assert X.shape == (3, fetch.REGISTRY["esm2_t6_8M"]["dim"])
        # has_cls: residues are token rows 1..L (values 1..L) -> mean = (L+1)/2
        for i, seq in enumerate(df["sequence"]):
            np.testing.assert_allclose(X[i], (len(seq) + 1) / 2)

    def test_protein_max_pooling(self, monkeypatch):
        _patch_loader(monkeypatch)
        df = _df()
        X = aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(
            df, mode="protein", pooling="max", model="esm2_t6_8M")
        for i, seq in enumerate(df["sequence"]):
            np.testing.assert_allclose(X[i], float(len(seq)))   # max residue row == L

    def test_residue_shapes(self, monkeypatch):
        _patch_loader(monkeypatch)
        df = _df()
        emb = aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(df, mode="residue", model="esm2_t6_8M")
        dim = fetch.REGISTRY["esm2_t6_8M"]["dim"]
        for entry, seq in zip(df["entry"], df["sequence"]):
            assert emb[entry].shape == (len(seq), dim)

    def test_return_df_adds_ok_column(self, monkeypatch):
        _patch_loader(monkeypatch)
        X, df_out = aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(_df(), return_df=True)
        assert "embeddings_ok" in df_out.columns
        assert bool(df_out["embeddings_ok"].all())

    def test_on_failure_nan(self, monkeypatch):
        _patch_loader(monkeypatch, fail=True)
        X, df_out = aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(
            _df(), on_failure="nan", return_df=True)
        assert np.isnan(X).all()
        assert not df_out["embeddings_ok"].any()

    def test_on_failure_raise(self, monkeypatch):
        _patch_loader(monkeypatch, fail=True)
        with pytest.raises(RuntimeError):
            aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(_df(), on_failure="raise")

    def test_on_failure_drop(self, monkeypatch):
        _patch_loader(monkeypatch, fail=True)
        X = aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(_df(), on_failure="drop")
        assert X.shape[0] == 0

    def test_cls_pooling(self, monkeypatch):
        _patch_loader(monkeypatch)
        # fake hidden[token_pos] == token_pos; the CLS token is row 0 -> all zeros
        X = aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(
            _df(), mode="protein", pooling="cls", model="esm2_t6_8M")
        np.testing.assert_allclose(X, 0.0)

    def test_residue_on_failure_nan(self, monkeypatch):
        _patch_loader(monkeypatch, fail=True)
        emb = aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(
            _df(), mode="residue", on_failure="nan")
        assert all(np.isnan(v).all() for v in emb.values())

    def test_residue_on_failure_drop(self, monkeypatch):
        _patch_loader(monkeypatch, fail=True)
        emb = aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(
            _df(), mode="residue", on_failure="drop")
        assert emb == {}

    def test_oversized_model_warns(self, monkeypatch):
        _patch_loader(monkeypatch)
        monkeypatch.setattr(fetch, "detect_hardware", lambda: dict(
            device="cpu", has_cuda=False, has_mps=False, total_ram_gb=0.01, free_vram_gb=None))
        with pytest.warns(RuntimeWarning):
            aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(_df(), model="esm2_t6_8M")

    def test_verbose_prints_recommendation(self, monkeypatch, capsys):
        _patch_loader(monkeypatch)
        aa.EmbeddingPreprocessor(verbose=True).fetch_embeddings(_df(), model="esm2_t6_8M")

    def test_layer_and_allow_oversized_pass_through(self, monkeypatch):
        _patch_loader(monkeypatch)
        X = aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(
            _df(), model="esm2_t6_8M", max_length=50, layer=0, allow_oversized=True)
        assert X.shape == (3, 320)

    def test_max_length_truncates_with_warning(self, monkeypatch):
        _patch_loader(monkeypatch)
        with pytest.warns(UserWarning):
            aa.EmbeddingPreprocessor(verbose=False).fetch_embeddings(
                _df(), model="esm2_t6_8M", max_length=3)


class TestPoolEmbeddings:

    def test_dict_mean(self):
        emb = {"A": np.array([[0.0, 2.0], [2.0, 4.0]])}
        pooled = aa.EmbeddingPreprocessor().pool_embeddings(emb, pooling="mean")
        np.testing.assert_allclose(pooled["A"], [1.0, 3.0])

    def test_matrix_when_df_seq_given(self):
        emb = {"P0": np.zeros((4, 5)), "P1": np.ones((3, 5))}
        df = pd.DataFrame({"entry": ["P0", "P1"], "sequence": ["AAAA", "GGG"], "label": [1, 0]})
        X = aa.EmbeddingPreprocessor().pool_embeddings(emb, df_seq=df)
        assert X.shape == (2, 5)
        np.testing.assert_allclose(X[1], 1.0)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().pool_embeddings({})

    def test_missing_entry_rejected(self):
        emb = {"P0": np.zeros((4, 5))}
        df = pd.DataFrame({"entry": ["P0", "P1"], "sequence": ["AAAA", "GGG"], "label": [1, 0]})
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().pool_embeddings(emb, df_seq=df)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TORCH, reason="requires the 'embed' extra (torch/transformers)")
class TestFetchRealModel:
    """End-to-end with the smallest real model (downloads esm2_t6_8M once)."""

    def test_protein_and_residue(self):
        df = _df(2)
        ep = aa.EmbeddingPreprocessor(verbose=False)
        X = ep.fetch_embeddings(df, mode="protein", model="esm2_t6_8M")
        assert X.shape == (2, 320) and not np.isnan(X).any()
        emb = ep.fetch_embeddings(df, mode="residue", model="esm2_t6_8M")
        for entry, seq in zip(df["entry"], df["sequence"]):
            assert emb[entry].shape == (len(seq), 320)
        # pool_embeddings reproduces mean pooling
        X_pool = ep.pool_embeddings(emb, df_seq=df)
        np.testing.assert_allclose(X_pool, X, rtol=1e-4, atol=1e-4)

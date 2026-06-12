"""Networkless unit tests for the fetch_embeddings backend foundation
(registry, hardware detection, recommendation, pooling). No torch required."""
import numpy as np
import pytest

from aaanalysis.data_handling._backend.embed_preproc import fetch


REQUIRED_KEYS = {"repo_id", "params_m", "dim", "ram_gb", "vram_gb",
                 "max_len", "has_cls", "tokenizer", "license", "note"}


class TestRegistry:
    """REGISTRY integrity."""

    def test_has_eight_models(self):
        assert len(fetch.REGISTRY) == 8
        assert fetch.LIST_MODELS == list(fetch.REGISTRY)

    def test_every_entry_has_required_keys(self):
        for key, meta in fetch.REGISTRY.items():
            assert REQUIRED_KEYS.issubset(meta), f"{key} missing keys"

    def test_metadata_value_types(self):
        for key, meta in fetch.REGISTRY.items():
            assert meta["params_m"] > 0
            assert meta["dim"] > 0
            assert meta["ram_gb"] > 0 and meta["vram_gb"] > 0
            assert isinstance(meta["has_cls"], bool)
            assert meta["tokenizer"] in ("bert", "t5")
            assert meta["max_len"] is None or meta["max_len"] > 0

    def test_esm1b_has_length_cap(self):
        assert fetch.REGISTRY["esm1b"]["max_len"] == 1022

    def test_t5_models_have_no_cls(self):
        assert fetch.REGISTRY["prott5_xl_u50"]["has_cls"] is False
        assert fetch.REGISTRY["prostt5"]["has_cls"] is False


class TestDetectHardware:
    """Hardware probe (runs on the CI host; torch usually absent)."""

    def test_returns_expected_shape(self):
        info = fetch.detect_hardware()
        assert set(info) == {"device", "has_cuda", "has_mps", "total_ram_gb", "free_vram_gb"}
        assert info["device"] in ("cpu", "cuda", "mps")

    def test_ram_is_none_or_positive(self):
        info = fetch.detect_hardware()
        assert info["total_ram_gb"] is None or info["total_ram_gb"] > 0


class TestEstimateFootprint:

    def test_positive_and_monotonic_in_batch(self):
        a = fetch.estimate_footprint_gb("esm2_t12_35M", device="cpu", batch_size=1)
        b = fetch.estimate_footprint_gb("esm2_t12_35M", device="cpu", batch_size=32)
        assert 0 < a < b

    def test_gpu_floor_below_cpu_floor(self):
        cpu = fetch.estimate_footprint_gb("esm2_t33_650M", device="cpu")
        gpu = fetch.estimate_footprint_gb("esm2_t33_650M", device="cuda")
        assert gpu < cpu


class TestRecommendModel:

    def test_none_memory_returns_smallest(self):
        assert fetch.recommend_model(mem_gb=None) == "esm2_t6_8M"

    def test_tiny_memory_returns_smallest(self):
        assert fetch.recommend_model(mem_gb=0.01, device="cpu") == "esm2_t6_8M"

    def test_large_memory_returns_largest(self):
        assert fetch.recommend_model(mem_gb=1000, device="cuda") == "esm2_t36_3B"

    def test_monotonic_more_memory_not_smaller(self):
        prev = -1.0
        for mem in [0.1, 0.5, 1.0, 3.0, 10.0, 1000.0]:
            params = fetch.REGISTRY[fetch.recommend_model(mem_gb=mem, device="cpu")]["params_m"]
            assert params >= prev
            prev = params


class TestPoolResidue:

    def test_mean(self):
        arr = np.array([[0.0, 2.0], [2.0, 4.0]])
        np.testing.assert_allclose(fetch.pool_residue_(arr, "mean"), [1.0, 3.0])

    def test_max(self):
        arr = np.array([[0.0, 5.0], [3.0, 1.0]])
        np.testing.assert_allclose(fetch.pool_residue_(arr, "max"), [3.0, 5.0])

    def test_output_shape_is_dim(self):
        arr = np.zeros((7, 11))
        assert fetch.pool_residue_(arr, "mean").shape == (11,)

    def test_cls_rejected_on_residue_array(self):
        with pytest.raises(ValueError):
            fetch.pool_residue_(np.zeros((3, 4)), "cls")

    def test_bad_pooling_rejected(self):
        with pytest.raises(ValueError):
            fetch.pool_residue_(np.zeros((3, 4)), "sum")

    def test_non_2d_rejected(self):
        with pytest.raises(ValueError):
            fetch.pool_residue_(np.zeros((3,)), "mean")

"""
This is a script to test the zero-split guard in CPP's vectorized recompute path.
"""
import numpy as np

from aaanalysis.feature_engineering._backend.cpp._filters._recompute import (
    iter_scale_chunks, build_position_buffer,
)
from aaanalysis.feature_engineering._backend.cpp._split import SplitRange


class TestIterScaleChunksZeroSplits:
    """The guard: iter_scale_chunks yields nothing (no ZeroDivisionError) on a
    zero-width split axis or zero samples, and still yields when n_splits > 0."""

    def test_zero_splits_yields_nothing(self):
        n_samples, L, D = 4, 10, 3
        arr_3d = np.zeros((n_samples, L, D), dtype=np.float64)
        pos_buf = np.full((n_samples, 0, 1), L, dtype=np.int64)  # zero splits
        chunks = list(iter_scale_chunks(arr_3d=arr_3d, scale_indices=[0, 1, 2], pos_buf=pos_buf))
        assert chunks == []

    def test_zero_samples_yields_nothing(self):
        arr_3d = np.zeros((0, 10, 3), dtype=np.float64)
        pos_buf = np.full((0, 2, 1), 10, dtype=np.int64)
        chunks = list(iter_scale_chunks(arr_3d=arr_3d, scale_indices=[0, 1, 2], pos_buf=pos_buf))
        assert chunks == []

    def test_empty_scale_indices_yields_nothing(self):
        arr_3d = np.zeros((4, 10, 3), dtype=np.float64)
        pos_buf = np.zeros((4, 2, 1), dtype=np.int64)
        chunks = list(iter_scale_chunks(arr_3d=arr_3d, scale_indices=[], pos_buf=pos_buf))
        assert chunks == []

    def test_nonzero_splits_still_yields(self):
        n_samples, L, D = 4, 10, 3
        arr_3d = np.random.RandomState(0).rand(n_samples, L, D)
        pos_buf = np.zeros((n_samples, 2, 1), dtype=np.int64)  # 2 splits at position 0
        chunks = list(iter_scale_chunks(arr_3d=arr_3d, scale_indices=[0, 1, 2], pos_buf=pos_buf))
        assert len(chunks) >= 1
        total_scales = sum(cm.shape[2] for _, cm in chunks)
        assert total_scales == 3


class TestBuildPositionBufferEmptyPattern:
    """build_position_buffer returns a zero-width buffer for a Pattern config
    that produces no valid splits (cumsum always exceeds len_max)."""

    def test_empty_pattern_config_zero_splits(self):
        spr = SplitRange(split_type_str=False)
        seq_lens = np.array([4, 10, 4], dtype=np.int64)
        pos_buf, labels_splits, max_split_len = build_position_buffer(
            split_type="Pattern",
            split_type_args=dict(steps=[3], n_min=2, n_max=3, len_max=4),
            seq_lens=seq_lens, L_max=10, spr=spr,
        )
        assert labels_splits == []
        assert pos_buf.shape[1] == 0
        # The end-to-end invariant: feeding this buffer to iter_scale_chunks
        # must not raise — it simply yields nothing.
        arr_3d = np.zeros((3, 10, 2), dtype=np.float64)
        assert list(iter_scale_chunks(arr_3d=arr_3d, scale_indices=[0, 1], pos_buf=pos_buf)) == []

    def test_nonempty_pattern_config_has_splits(self):
        spr = SplitRange(split_type_str=False)
        seq_lens = np.array([10, 10], dtype=np.int64)
        pos_buf, labels_splits, _ = build_position_buffer(
            split_type="Pattern",
            split_type_args=dict(steps=[3, 4], n_min=2, n_max=3, len_max=8),
            seq_lens=seq_lens, L_max=10, spr=spr,
        )
        assert len(labels_splits) > 0
        assert pos_buf.shape[1] == len(labels_splits)

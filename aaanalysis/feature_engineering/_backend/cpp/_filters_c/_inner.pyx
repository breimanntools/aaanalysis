# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""
Bit-exact Cython kernel for per-(sample, feature) mean in CPP feature
matrix construction.

The arithmetic is identical to numpy's ``np.mean`` because:

1. We replicate numpy's pairwise summation exactly (see ``pairwise_sum``):
   - linear sum for n < 8
   - 8-way unrolled accumulator for 8 <= n <= 128 (numpy's
     ``PW_BLOCKSIZE``), with the SAME final reduction tree
     ``((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))``
   - recursive halving for n > 128 (rare; our segments are <= ~30)
2. The mean is ``sum / count`` (same as ``np.mean``).
3. Output is rounded to 5 decimals via ``np.round`` in the Python wrapper
   to match the legacy ``_feature_value`` rounding (banker's rounding).

These three steps preserve bit-exact parity with ``np.mean`` for arrays of
length <= 128 and the recursion structure matches numpy for longer arrays.
"""
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport NAN


cnp.import_array()


cdef inline double pairwise_nansum(const double* a, Py_ssize_t n,
                                    Py_ssize_t* out_count) noexcept nogil:
    """Bit-exact replica of numpy's ``np.nansum`` step inside ``np.nanmean``.

    Replaces NaN entries with 0 (using the IEEE 754 ``v != v`` check, which
    is true only for NaN) and accumulates with the SAME pairwise summation
    tree as ``pairwise_sum``. Also counts non-NaN entries → ``out_count``.

    Skipping a NaN entry vs adding 0 is bit-identical (0 + r = r exactly in
    IEEE 754), so the result matches ``np.nansum``.
    """
    cdef double r0, r1, r2, r3, r4, r5, r6, r7, res, v
    cdef Py_ssize_t i, n2, cnt, cnt_left, cnt_right
    if n < 8:
        res = 0.0
        cnt = 0
        for i in range(n):
            v = a[i]
            if v == v:  # not NaN
                res = res + v
                cnt = cnt + 1
        out_count[0] = cnt
        return res
    elif n <= 128:
        cnt = 0
        # Initial 8 reads: NaN → 0, count non-NaN.
        v = a[0]
        if v == v:
            r0 = v; cnt += 1
        else:
            r0 = 0.0
        v = a[1]
        if v == v:
            r1 = v; cnt += 1
        else:
            r1 = 0.0
        v = a[2]
        if v == v:
            r2 = v; cnt += 1
        else:
            r2 = 0.0
        v = a[3]
        if v == v:
            r3 = v; cnt += 1
        else:
            r3 = 0.0
        v = a[4]
        if v == v:
            r4 = v; cnt += 1
        else:
            r4 = 0.0
        v = a[5]
        if v == v:
            r5 = v; cnt += 1
        else:
            r5 = 0.0
        v = a[6]
        if v == v:
            r6 = v; cnt += 1
        else:
            r6 = 0.0
        v = a[7]
        if v == v:
            r7 = v; cnt += 1
        else:
            r7 = 0.0
        # Accumulator loop: 8-element chunks, skip NaN.
        i = 8
        while i + 8 <= n:
            v = a[i + 0]
            if v == v:
                r0 = r0 + v; cnt += 1
            v = a[i + 1]
            if v == v:
                r1 = r1 + v; cnt += 1
            v = a[i + 2]
            if v == v:
                r2 = r2 + v; cnt += 1
            v = a[i + 3]
            if v == v:
                r3 = r3 + v; cnt += 1
            v = a[i + 4]
            if v == v:
                r4 = r4 + v; cnt += 1
            v = a[i + 5]
            if v == v:
                r5 = r5 + v; cnt += 1
            v = a[i + 6]
            if v == v:
                r6 = r6 + v; cnt += 1
            v = a[i + 7]
            if v == v:
                r7 = r7 + v; cnt += 1
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            v = a[i]
            if v == v:
                res = res + v; cnt += 1
            i += 1
        out_count[0] = cnt
        return res
    else:
        n2 = (n // 2) - ((n // 2) % 8)
        res = pairwise_nansum(a, n2, &cnt_left)
        res = res + pairwise_nansum(a + n2, n - n2, &cnt_right)
        out_count[0] = cnt_left + cnt_right
        return res


cdef inline double pairwise_sum(const double* a, Py_ssize_t n) noexcept nogil:
    """Bit-exact replica of numpy's pairwise_sum_DOUBLE for length n.

    See ``numpy/_core/src/umath/loops_utils.h.src`` for the original.
    """
    cdef double r0, r1, r2, r3, r4, r5, r6, r7, res
    cdef Py_ssize_t i, n2
    if n < 8:
        res = 0.0
        for i in range(n):
            res = res + a[i]
        return res
    elif n <= 128:
        r0 = a[0]; r1 = a[1]; r2 = a[2]; r3 = a[3]
        r4 = a[4]; r5 = a[5]; r6 = a[6]; r7 = a[7]
        i = 8
        while i + 8 <= n:
            r0 = r0 + a[i + 0]; r1 = r1 + a[i + 1]
            r2 = r2 + a[i + 2]; r3 = r3 + a[i + 3]
            r4 = r4 + a[i + 4]; r5 = r5 + a[i + 5]
            r6 = r6 + a[i + 6]; r7 = r7 + a[i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res = res + a[i]
            i += 1
        return res
    else:
        # n > 128 — recursive halving. Split at largest multiple of 8 <= n/2
        # to keep the 8-block path on both halves.
        n2 = (n // 2) - ((n // 2) % 8)
        return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2)


def compute_segment_mean(double[:, ::1] arr_2d not None,
                          long[::1] seq_lens not None,
                          int i_th, int n_split):
    """Per-sample mean of ``arr_2d[i, start:end]`` where (start, end) come from
    legacy ``SplitVec.segment`` (``start = int(L*(i_th-1)/n_split)``,
    ``end = int(L*i_th/n_split)``).

    Bit-exact with ``np.mean(arr_2d[i, start:end])`` per sample.
    """
    cdef Py_ssize_t n = arr_2d.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef Py_ssize_t i
    cdef long L_i, start, end, k
    cdef double s
    with nogil:
        for i in range(n):
            L_i = seq_lens[i]
            # Match Python's ``int(L_i / n_split * (i_th - 1))`` via integer ops
            # (positive divisor → floor == trunc → equivalent).
            start = (L_i * (i_th - 1)) // n_split
            end = (L_i * i_th) // n_split
            k = end - start
            if k > 0:
                s = pairwise_sum(&arr_2d[i, start], k)
                out_v[i] = s / <double>k
            else:
                out_v[i] = 0.0
    np.round(out, 5, out=out)
    return out


def compute_pattern_n_mean(double[:, ::1] arr_2d not None,
                            long[::1] positions not None):
    """Pattern N-terminus: positions = ``list_pos - 1`` (fixed across samples).

    Bit-exact with ``np.mean(arr_2d[i, positions])``: builds a per-row
    contiguous gather buffer, then pairwise-sums.
    """
    cdef Py_ssize_t n = arr_2d.shape[0]
    cdef Py_ssize_t k = positions.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef Py_ssize_t i, j
    cdef double s
    cdef double* buf = <double*>malloc(k * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    try:
        with nogil:
            for i in range(n):
                for j in range(k):
                    buf[j] = arr_2d[i, positions[j]]
                s = pairwise_sum(buf, k)
                out_v[i] = s / <double>k
    finally:
        free(buf)
    np.round(out, 5, out=out)
    return out


def compute_pattern_c_mean(double[:, ::1] arr_2d not None,
                            long[::1] seq_lens not None,
                            long[::1] list_pos not None):
    """Pattern C-terminus: per sample, positions = ``L_i - list_pos``.

    Bit-exact with the per-sample ``np.mean(arr_2d[i, L_i - list_pos])``.
    """
    cdef Py_ssize_t n = arr_2d.shape[0]
    cdef Py_ssize_t k = list_pos.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef Py_ssize_t i, j
    cdef long L_i
    cdef double s
    cdef double* buf = <double*>malloc(k * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    try:
        with nogil:
            for i in range(n):
                L_i = seq_lens[i]
                for j in range(k):
                    buf[j] = arr_2d[i, L_i - list_pos[j]]
                s = pairwise_sum(buf, k)
                out_v[i] = s / <double>k
    finally:
        free(buf)
    np.round(out, 5, out=out)
    return out


# ---------------------------------------------------------------------------
# NaN-aware variants (accept_gaps=True): bit-exact with ``np.nanmean``.
# ---------------------------------------------------------------------------
def compute_segment_nanmean(double[:, ::1] arr_2d not None,
                             long[::1] seq_lens not None,
                             int i_th, int n_split):
    """Bit-exact with per-sample ``np.nanmean(arr_2d[i, start:end])``."""
    cdef Py_ssize_t n = arr_2d.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef Py_ssize_t i, cnt
    cdef long L_i, start, end, k
    cdef double s
    with nogil:
        for i in range(n):
            L_i = seq_lens[i]
            start = (L_i * (i_th - 1)) // n_split
            end = (L_i * i_th) // n_split
            k = end - start
            if k > 0:
                s = pairwise_nansum(&arr_2d[i, start], k, &cnt)
                if cnt > 0:
                    out_v[i] = s / <double>cnt
                else:
                    out_v[i] = NAN
            else:
                out_v[i] = 0.0
    np.round(out, 5, out=out)
    return out


def compute_pattern_n_nanmean(double[:, ::1] arr_2d not None,
                               long[::1] positions not None):
    """Bit-exact with per-sample ``np.nanmean(arr_2d[i, positions])``."""
    cdef Py_ssize_t n = arr_2d.shape[0]
    cdef Py_ssize_t k = positions.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef Py_ssize_t i, j, cnt
    cdef double s
    cdef double* buf = <double*>malloc(k * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    try:
        with nogil:
            for i in range(n):
                for j in range(k):
                    buf[j] = arr_2d[i, positions[j]]
                s = pairwise_nansum(buf, k, &cnt)
                if cnt > 0:
                    out_v[i] = s / <double>cnt
                else:
                    out_v[i] = NAN
    finally:
        free(buf)
    np.round(out, 5, out=out)
    return out


def compute_pattern_c_nanmean(double[:, ::1] arr_2d not None,
                               long[::1] seq_lens not None,
                               long[::1] list_pos not None):
    """Bit-exact with per-sample ``np.nanmean(arr_2d[i, L_i - list_pos])``."""
    cdef Py_ssize_t n = arr_2d.shape[0]
    cdef Py_ssize_t k = list_pos.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef Py_ssize_t i, j, cnt
    cdef long L_i
    cdef double s
    cdef double* buf = <double*>malloc(k * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    try:
        with nogil:
            for i in range(n):
                L_i = seq_lens[i]
                for j in range(k):
                    buf[j] = arr_2d[i, L_i - list_pos[j]]
                s = pairwise_nansum(buf, k, &cnt)
                if cnt > 0:
                    out_v[i] = s / <double>cnt
                else:
                    out_v[i] = NAN
    finally:
        free(buf)
    np.round(out, 5, out=out)
    return out

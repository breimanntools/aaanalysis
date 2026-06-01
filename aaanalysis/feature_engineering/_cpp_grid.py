"""
This is a script for the frontend CPPGrid class running grid-style CPP configuration sweeps.
"""
from typing import Optional, List, Dict, Tuple, Union
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import aaanalysis.utils as ut

from ._cpp import CPP
from ._sequence_feature import SequenceFeature
from ._numerical_feature import NumericalFeature


# Map the public backend name to the joblib backend.
_JOBLIB_BACKEND = {"threads": "threading", "loky": "loky"}


# I Helper Functions
def _as_candidates(value):
    """A ``list``/``tuple`` value is a swept axis; anything else is a single fixed value.

    To sweep a value that is *itself* list-valued (e.g. ``steps_pattern=[3, 4]`` or
    ``list_parts=["tmd", "jmd_n"]``), wrap it in an outer list
    (``steps_pattern=[[3, 4], [2, 5]]``); a bare list is then one fixed value.
    """
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _is_scalar_axis(candidates):
    """True if every candidate is a scalar recordable verbatim in df_params."""
    return all((c is None or isinstance(c, (int, float, str, bool))) for c in candidates)


def _expand_with_records(params):
    """Expand a ``{key: value | [candidates]}`` dict into a Cartesian product.

    Returns a list of ``(kwargs, record)`` pairs: ``kwargs`` carries the concrete
    values to call the underlying function with; ``record`` carries the lightweight
    df_params cell (the literal value for scalar axes, else the position index into
    that key's candidate list).
    """
    params = params or {}
    if not params:
        return [({}, {})]
    keys = list(params)
    cand = {k: _as_candidates(params[k]) for k in keys}
    scalar = {k: _is_scalar_axis(cand[k]) for k in keys}
    out = []
    for idxs in itertools.product(*[range(len(cand[k])) for k in keys]):
        kwargs = {k: cand[k][i] for k, i in zip(keys, idxs)}
        record = {k: (cand[k][i] if scalar[k] else i) for k, i in zip(keys, idxs)}
        out.append((kwargs, record))
    return out


def _scales_candidates(params_scales):
    """``params_scales`` is a single df_scales, a list of df_scales, or None."""
    if params_scales is None:
        return [None]
    if isinstance(params_scales, (list, tuple)):
        return list(params_scales)
    return [params_scales]


def _resolve_df_cat(df_scales):
    """Resolve df_cat from df_scales alone ("df_scales is enough").

    AAontology-subset scales → ``None`` so ``CPP`` filters its default category
    table to them; custom / numeric dims (e.g. embedding ``d0..dN``) → a minimal
    one-row-per-dimension df_cat in a single ``"Numeric"`` category.
    """
    if df_scales is None:
        return None
    default_cat = ut.load_default_scales(scale_cat=True)
    cols = [str(c) for c in df_scales.columns]
    if set(cols).issubset(set(default_cat[ut.COL_SCALE_ID].astype(str))):
        return None
    return pd.DataFrame({ut.COL_SCALE_ID: cols, ut.COL_CAT: "Numeric",
                         ut.COL_SUBCAT: cols, ut.COL_SCALE_NAME: cols,
                         ut.COL_SCALE_DES: cols})


def _n_warnings_member(stats=None, n_filter=100):
    """Thread-safe per-combo warning count derived from ``last_filter_stats`` (D5b + D7).

    Live ``warnings.catch_warnings`` capture is not thread-safe under the default
    ``prefer="threads"`` backend, so the sparse-config (``n_candidates < n_filter``)
    and filter-shortfall conditions are re-derived from the deterministic filter-funnel
    counts. ``n_after_redundancy`` (not the max-run's ``n_final``) is the shortfall basis,
    so a member produced by slicing a larger run warns exactly as an independent run at
    its own ``n_filter`` would.
    """
    if not stats:
        return 0
    if stats.get("n_candidates", n_filter) < n_filter:
        return 1
    if stats.get("n_after_redundancy", n_filter) < n_filter:
        return 1
    return 0


def _combo_key(kw):
    """Stable hashable key for a kwargs dict (values may be lists/objects)."""
    return tuple(sorted((k, repr(v)) for k, v in kw.items()))


def _err_record(rec):
    """A df_params record for a configuration that could not run."""
    return {**rec, "n_warnings": 0, "n_errors": 1}


def _check_params_dict(name=None, params=None):
    """params_parts / params_split / params_cpp must be a dict of knobs or None."""
    if params is not None and not isinstance(params, dict):
        raise ValueError(f"'{name}' ({type(params).__name__}) should be a dict of "
                         f"knob -> value | [candidates], or None.")


# II Main Functions
class CPPGrid:
    """
    Grid-style sweep over CPP configurations (Tool).

    Runs the full **parts → splits → scales → run** pipeline across a Cartesian grid of
    configurations so a sweep needs one call instead of many manual ``get_df_parts`` /
    ``get_split_kws`` / ``CPP`` constructions. The dataset (``df_seq`` + ``labels``, plus
    ``dict_num`` for the numerical arm) is bound at construction; :meth:`run` takes four
    stage-grouped parameter dictionaries whose list-valued entries are swept.

    Notes
    -----
    * Inside each configuration ``CPP.run`` / ``run_num`` runs serially (``n_jobs=1``);
      the grid is parallelized **across** configurations to avoid nested oversubscription.
    * The default ``backend="threads"`` shares ``df_seq`` / ``df_scales`` in-process (no
      dataframe serialization, and it sidesteps the Python 3.14 / macOS ``__main__``-guard
      spawn footgun). Pass ``backend="loky"`` for process-based parallelism.

    See Also
    --------
    * :class:`CPP` : the per-configuration engine this class orchestrates.
    """

    def __init__(self,
                 df_seq: pd.DataFrame = None,
                 labels: ut.ArrayLike1D = None,
                 dict_num: Optional[Dict[str, np.ndarray]] = None,
                 accept_gaps: bool = False,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[int] = -1,
                 backend: str = "threads",
                 ):
        """
        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers and a
            ``sequence`` column with full protein sequences (any format accepted by
            :meth:`SequenceFeature.get_df_parts`).
        labels : array-like, shape (n_samples,)
            Class labels aligned to the resulting ``df_parts`` rows (test vs reference).
        dict_num : dict[str, np.ndarray], optional
            Mapping ``entry -> (L, D)`` per-residue tensor. If given, the grid runs the
            numerical arm (``NumericalFeature.get_parts`` → ``CPP.run_num``).
        accept_gaps : bool, default=False
            Whether to accept gaps when assigning scale values.
        verbose : bool, default=True
            If ``True``, enable verbose output.
        random_state : int, optional
            Seed forwarded to each ``CPP`` for reproducibility.
        n_jobs : int, default=-1
            Number of workers used **across** configurations (``-1`` = all cores).
        backend : {'threads', 'loky'}, default='threads'
            Joblib backend used across configurations.
        """
        # Check input
        ut.check_df_seq(df_seq=df_seq)
        if labels is None:
            raise ValueError("'labels' should not be None.")
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        ut.check_str_options(name="backend", val=backend, list_str_options=["threads", "loky"])
        if dict_num is not None and not isinstance(dict_num, dict):
            raise ValueError(f"'dict_num' ({type(dict_num).__name__}) should be a "
                             f"Dict[str, np.ndarray] or None.")
        # Internal attributes
        self.df_seq = df_seq.copy()
        self.labels = labels
        self.dict_num = dict_num
        self._accept_gaps = accept_gaps
        self._verbose = verbose
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._backend = backend

    # Workers
    def _build_parts(self, parts_kw):
        """Build (df_parts, dict_num_parts) for one parts-config (dict_num_parts is None in seq mode)."""
        if self.dict_num is None:
            df_parts = SequenceFeature(verbose=False).get_df_parts(df_seq=self.df_seq, **parts_kw)
            return df_parts, None
        return NumericalFeature.get_parts(df_seq=self.df_seq, dict_num=self.dict_num, **parts_kw)

    def _run_base(self, df_parts=None, dict_num_parts=None, split_kws=None, df_scales=None, cpp_kw=None):
        """Run CPP once for a base config (parts/splits precomputed); return (df_feat, stats)."""
        df_cat = _resolve_df_cat(df_scales)
        cpp = CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, df_cat=df_cat,
                  accept_gaps=self._accept_gaps, verbose=False, random_state=self._random_state)
        if self.dict_num is None:
            return cpp.run(labels=self.labels, n_jobs=1, return_stats=True, **cpp_kw)
        return cpp.run_num(dict_num_parts=dict_num_parts, labels=self.labels,
                           n_jobs=1, return_stats=True, **cpp_kw)

    def run(self,
            params_parts: Optional[dict] = None,
            params_split: Optional[dict] = None,
            params_scales: Union[pd.DataFrame, List[pd.DataFrame], None] = None,
            params_cpp: Optional[dict] = None,
            ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """
        Run the configuration grid and return per-combo feature tables plus a sweep summary.

        Parameters
        ----------
        params_parts : dict, optional
            ``get_df_parts`` / ``get_parts`` kwargs (``tmd_len``, ``jmd_n_len``,
            ``jmd_c_len``, ``list_parts``, ...). List-valued entries are swept.
        params_split : dict, optional
            ``SequenceFeature.get_split_kws`` kwargs (``split_types``, ``n_split_max``,
            ``len_max``, ``steps_pattern``, ...). List-valued entries are swept.
        params_scales : pd.DataFrame or list of pd.DataFrame, optional
            A single ``df_scales`` or a list of ``df_scales`` to sweep. ``df_cat`` is
            resolved internally per scale set.
        params_cpp : dict, optional
            ``CPP.run`` / ``run_num`` kwargs (``n_filter``, ``max_std_test``,
            ``max_overlap``, ``max_cor``, ...). List-valued entries are swept.

        Returns
        -------
        list_df_feat : list of pd.DataFrame or None
            One feature table per configuration (``None`` where that configuration raised
            at run time), aligned to ``df_params`` rows in ``itertools.product`` order.
        df_params : pd.DataFrame
            One row per configuration describing it: scalar axes hold the literal value,
            object axes (``df_scales`` and any list-valued knob) hold the position index
            into their candidate list, plus ``n_warnings`` and ``n_errors`` counts.

        Notes
        -----
        * To sweep a list-valued knob (``steps_pattern``, ``list_parts``) wrap it in a
          list; a bare list is one fixed value.
        * ``n_warnings`` is derived from each run's filter-funnel counts (sparse-config and
          filter-shortfall conditions); ``n_errors`` counts configurations that raised.
        * **Smart sweeping (no redundant CPP runs).** Sweeping ``n_filter`` does **not**
          re-run CPP per value: configurations that differ only in ``n_filter`` run CPP
          **once at the largest** value, and the smaller ones are exact ``head(n)`` slices
          (the redundancy filter is a greedy top-down pass, so the top-``n`` is invariant).
          ``df_parts`` are built once per parts-config and ``split_kws`` once per
          split-config, then reused across the grid; the D3 scale-lookup LRU is reused
          across configs sharing a ``df_scales``.
        """
        # Check input
        _check_params_dict(name="params_parts", params=params_parts)
        _check_params_dict(name="params_split", params=params_split)
        _check_params_dict(name="params_cpp", params=params_cpp)
        n_jobs = ut.check_n_jobs(n_jobs=self._n_jobs)
        scales_list = _scales_candidates(params_scales)
        for i, ds in enumerate(scales_list):
            if ds is not None and not isinstance(ds, pd.DataFrame):
                raise ValueError(f"'params_scales' entry {i} ({type(ds).__name__}) should "
                                 f"be a pd.DataFrame.")
        # Fail-fast: the numerical arm needs D == len(df_scales.columns) for every scale set
        if self.dict_num is not None:
            d_dims = {arr.shape[1] for arr in self.dict_num.values()}
            for i, ds in enumerate(scales_list):
                if ds is not None and len(ds.columns) not in d_dims:
                    raise ValueError(f"'params_scales' entry {i} has {len(ds.columns)} "
                                     f"columns but 'dict_num' tensors have D in {sorted(d_dims)}; "
                                     f"D must equal the number of df_scales columns.")
        parts_pairs = _expand_with_records(params_parts)
        split_pairs = _expand_with_records(params_split)
        cpp_pairs = _expand_with_records(params_cpp)
        has_nfilter = bool(params_cpp) and "n_filter" in params_cpp

        # Shortcut caches (built once, reused across all combos that share the sub-config).
        # df_parts depend ONLY on the parts-config; split_kws ONLY on the split-config.
        parts_cache = {}
        for pkw, _ in parts_pairs:
            key = _combo_key(pkw)
            if key not in parts_cache:
                try:
                    parts_cache[key] = self._build_parts(pkw)
                except Exception as exc:           # parts build failure -> members error
                    parts_cache[key] = exc
        split_cache = {}
        for skw, _ in split_pairs:
            key = _combo_key(skw)
            if key not in split_cache:
                try:
                    split_cache[key] = SequenceFeature.get_split_kws(**skw) if skw else None
                except Exception as exc:
                    split_cache[key] = exc

        # Build the full combo list in product order; group combos that differ ONLY in
        # n_filter so each group runs CPP ONCE at its max n_filter (the rest are head(n)
        # slices — exact, since the redundancy filter is a greedy top-down pass).
        combos, groups = [], {}
        for pi, (pkw, prec) in enumerate(parts_pairs):
            for si_, (skw, srec) in enumerate(split_pairs):
                for sci in range(len(scales_list)):
                    for ckw, crec in cpp_pairs:
                        idx = len(combos)
                        nf = ckw.get("n_filter") if has_nfilter else None
                        combos.append(dict(pkw=pkw, skw=skw, sci=sci, ckw=ckw, nf=nf,
                                           rec={**prec, **srec, **crec, "df_scales": sci}))
                        bkey = (pi, si_, sci, tuple(sorted(
                            (k, repr(v)) for k, v in ckw.items() if k != "n_filter")))
                        groups.setdefault(bkey, []).append(idx)

        def _run_group(member_indices):
            out = {}
            first = combos[member_indices[0]]
            parts = parts_cache[_combo_key(first["pkw"])]
            split = split_cache[_combo_key(first["skw"])]
            if isinstance(parts, Exception) or isinstance(split, Exception):
                return {i: (None, _err_record(combos[i]["rec"])) for i in member_indices}
            df_parts, dict_num_parts = parts
            # Resolve the n_filter members + the single run at their max valid value.
            if has_nfilter:
                valid = {i: combos[i]["nf"] for i in member_indices
                         if isinstance(combos[i]["nf"], int) and not isinstance(combos[i]["nf"], bool)
                         and combos[i]["nf"] >= 1}
                for i in member_indices:                      # invalid n_filter -> soft error
                    if i not in valid:
                        out[i] = (None, _err_record(combos[i]["rec"]))
                if not valid:
                    return out
                run_nf = max(valid.values())
                cpp_kw = {k: v for k, v in first["ckw"].items() if k != "n_filter"}
                cpp_kw["n_filter"] = run_nf
            else:
                valid = {member_indices[0]: None}             # single default-n_filter run
                cpp_kw = dict(first["ckw"])
            try:
                df_feat_max, stats = self._run_base(
                    df_parts=df_parts, dict_num_parts=dict_num_parts, split_kws=split,
                    df_scales=scales_list[first["sci"]], cpp_kw=cpp_kw)
            except Exception:
                for i in valid:
                    out[i] = (None, _err_record(combos[i]["rec"]))
                return out
            for i, nf in valid.items():
                df_feat = df_feat_max if nf is None else df_feat_max.head(nf).copy()
                rec = dict(combos[i]["rec"])
                rec["n_warnings"] = _n_warnings_member(stats=stats, n_filter=nf if nf is not None else 100)
                rec["n_errors"] = 0
                out[i] = (df_feat, rec)
            return out

        group_results = Parallel(n_jobs=n_jobs, backend=_JOBLIB_BACKEND[self._backend])(
            delayed(_run_group)(idxs) for idxs in groups.values())
        merged = {}
        for gr in group_results:
            merged.update(gr)
        list_df_feat = [merged[i][0] for i in range(len(combos))]
        df_params = pd.DataFrame([merged[i][1] for i in range(len(combos))])
        return list_df_feat, df_params

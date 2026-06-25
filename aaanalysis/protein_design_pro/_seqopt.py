"""
This is a script for the frontend of the SeqOpt class (**[pro]**): SHAP-guided, fuzzy-labeled
multi-objective directed-evolution optimization over sequence variants of one wild-type. SeqOpt
reuses a model-bound SeqMut as its fitness engine and ShapModel for per-generation residue
guidance, and is therefore gated behind the ``pro`` extra.
"""
from typing import Optional, List, Any, Callable, Tuple, Dict
import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from aaanalysis.template_classes import Tool
from aaanalysis.feature_engineering._sequence_feature import SequenceFeature
from aaanalysis.protein_design import SeqMut
from aaanalysis.explainable_ai_pro import ShapModel
from ._backend.seqopt.genome import canonical, apply_genome, variant_label
from ._backend.seqopt.run import evolve_nsga2, evolve_greedy
from ._backend.seqopt.nsga2 import normalize_objectives_
from ._backend.seqopt.metrics import hypervolume, spread, convergence
from ._backend.seqopt.penalty import apply_penalty


# I Helper Functions
def check_mode(mode=None):
    """Check the guidance mode."""
    ut.check_str_options(name="mode", val=mode, list_str_options=ut.LIST_SEQOPT_MODES)
    return mode


def check_match_df_seq_single(df_seq=None):
    """Check that df_seq is a single-entry, position-based wild-type frame."""
    ut.check_df_seq(df_seq=df_seq)
    missing = [c for c in ut.COLS_SEQ_POS if c not in df_seq.columns]
    if len(missing) > 0:
        raise ValueError(f"'df_seq' ({missing}) should be in the position-based format with "
                         f"columns {ut.COLS_SEQ_POS}; SeqOpt mutates residues and recomputes parts.")
    if len(df_seq) != 1:
        raise ValueError(f"'df_seq' (n={len(df_seq)}) should contain exactly one wild-type "
                         f"sequence; SeqOpt optimizes variants of a single sequence per run.")


def check_objectives(objectives=None, model=None):
    """Check the objectives spec and split it into names, goals and sources.

    ``objectives`` is a list of ``(name, goal, source)`` with ``goal`` in {'max','min'} and
    ``source`` a built-in column name (delta_pred / delta_cpp / shift_score / n_mut) or a
    callable. At least two objectives are required for a Pareto run.
    """
    objectives = ut.check_list_like(name="objectives", val=objectives, accept_none=False)
    if len(objectives) < 2:
        raise ValueError(f"'objectives' (n={len(objectives)}) should list at least two "
                         f"(name, goal, source) objectives for a Pareto run.")
    names, goals, sources = [], [], []
    for i, obj in enumerate(objectives):
        if not (isinstance(obj, (tuple, list)) and len(obj) == 3):
            raise ValueError(f"'objectives[{i}]' ({obj}) should be a (name, goal, source) triple.")
        name, goal, source = obj
        ut.check_str(name="objective name", val=name, accept_none=False)
        if goal not in ut.LIST_OBJECTIVE_GOALS:
            raise ValueError(f"'objectives[{i}]' goal ({goal}) should be one of "
                             f"{ut.LIST_OBJECTIVE_GOALS}.")
        if not (callable(source) or source in ut.LIST_OBJECTIVE_SOURCES):
            raise ValueError(f"'objectives[{i}]' source ({source}) should be a callable or one "
                             f"of {ut.LIST_OBJECTIVE_SOURCES}.")
        if source == ut.COL_DELTA_PRED and model is None:
            raise ValueError(f"'objectives[{i}]' uses '{ut.COL_DELTA_PRED}' but no 'model' was "
                             f"bound to SeqMut; pass a fitted model to the SeqOpt constructor.")
        names.append(name)
        goals.append(goal)
        sources.append(source)
    if len(set(names)) != len(names):
        raise ValueError(f"'objectives' names ({names}) should be unique (one df_pareto column each).")
    return names, goals, sources


def residue_weights_(df_feat, col, base):
    """Aggregate |df_feat[col]| per residue (window->full via base) into a position-weight dict.

    Parameters
    ----------
    df_feat : pd.DataFrame
        Feature set carrying ``positions`` (1-based window positions) and ``col``.
    col : str
        Attribution column to aggregate (``feat_importance`` or a ``feat_impact*`` column).
    base : int
        Full-sequence position of window index 1 (``tmd_start - jmd_n_len``).

    Returns
    -------
    weights : dict or None
        ``{full_position: weight}``; None when the column is absent or sums to zero.
    """
    if col is None or col not in df_feat.columns or ut.COL_POSITION not in df_feat.columns:
        return None
    weights: Dict[int, float] = {}
    for pos_str, val in zip(df_feat[ut.COL_POSITION], df_feat[col]):
        v = abs(float(val))
        if v == 0:
            continue
        for tok in str(pos_str).split(","):
            tok = tok.strip()
            if tok:
                full = base + int(tok) - 1
                weights[full] = weights.get(full, 0.0) + v
    return weights or None


# II Main Functions
class SeqOpt(Tool):
    """
    Sequence Optimizer (SeqOpt) class (**[pro]**, requires ``aaanalysis[pro]``) for SHAP-guided,
    multi-objective directed evolution over sequence variants [Breimann24a]_.

    ``SeqOpt`` is the **search/optimization** counterpart of :class:`SeqMut`: where ``SeqMut``
    *scores* mutations, ``SeqOpt`` *searches* the space of multi-mutation variants of a single
    wild-type for the trade-off (Pareto) front that best satisfies several objectives at once.
    It runs a re-implementation of NSGA-II [Deb02]_ using a model-bound :class:`SeqMut` as the
    fitness engine, guided each generation by residue-level model attribution: ``mode="impact"``
    refits :class:`ShapModel` under fuzzy labeling, ``mode="importance"`` uses the static
    ``feat_importance`` ranking.

    .. versionadded:: 1.0.0

    """
    def __init__(self,
                 mode: str = "impact",
                 model: Optional[Any] = None,
                 target_class: Optional[Any] = None,
                 df_seq_ref: Optional[pd.DataFrame] = None,
                 labels: Optional[ut.ArrayLike1D] = None,
                 df_scales: Optional[pd.DataFrame] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = False,
                 ):
        """
        Parameters
        ----------
        mode : str, default='impact'
            Residue-guidance mode. ``'impact'`` refits :class:`ShapModel` every generation under
            fuzzy labeling (the new variant's prediction score as a soft label vs. the balanced
            reference) and mutates the strongest-``feat_impact`` residues. ``'importance'`` uses
            the static ``feat_importance`` ranking from ``df_feat`` (no SHAP, no refit) and walks
            positions highest-first.
        model : object, optional
            A fitted classifier exposing ``predict_proba`` used as the fitness engine (the
            ``delta_pred`` objective) and, in ``mode='impact'``, as the model whose attribution
            guides the search. Required when an objective uses ``delta_pred`` or ``mode='impact'``.
        target_class : int or str, optional
            Class whose predicted probability the fitness tracks. ``None`` selects the positive class.
        df_seq_ref : pd.DataFrame, optional
            Labeled, balanced **reference** sequences (position-based format) the per-generation
            ``ShapModel`` refit is anchored on. Required for ``mode='impact'``.
        labels : array-like, shape (n_ref,), optional
            Binary class labels (1=positive, 0=negative) aligned to ``df_seq_ref``. Required for
            ``mode='impact'``.
        df_scales : pd.DataFrame, optional
            Amino acid scales. Default from :func:`load_scales`.
        random_state : int, optional
            Seed threaded through the evolutionary RNG and :class:`ShapModel`.
        verbose : bool, default=False
            If ``True``, verbose outputs are enabled.

        See Also
        --------
        * :class:`SeqMut` whose ``combine`` scores the variants (the fitness engine).
        * :class:`ShapModel` whose fuzzy-labeled SHAP values drive ``mode='impact'`` guidance.
        """
        self._verbose = ut.check_verbose(verbose)
        self._mode = check_mode(mode)
        if df_scales is None:
            df_scales = ut.load_default_scales()
        self.df_scales = df_scales
        if random_state is not None:
            ut.check_number_range(name="random_state", val=random_state, min_val=0, just_int=True)
        self._random_state = random_state
        self._sf = SequenceFeature(verbose=False)
        # Fitness engine (model-bound SeqMut); validation of model/target_class is reused there.
        self._seqmut = SeqMut(verbose=False, df_scales=df_scales, model=model,
                              target_class=target_class)
        self._model = model
        self._target_class = target_class
        # mode='impact' needs the labeled reference set for the per-generation ShapModel refit.
        if self._mode == "impact":
            if model is None or df_seq_ref is None or labels is None:
                raise ValueError("mode='impact' requires 'model', 'df_seq_ref' and 'labels' "
                                 "(the balanced reference the per-generation ShapModel refit is "
                                 "anchored on); use mode='importance' for a SHAP-free run.")
            ut.check_df_seq(df_seq=df_seq_ref)
            labels = ut.check_labels(labels=labels, len_required=len(df_seq_ref))
        self._df_seq_ref = df_seq_ref
        self._labels = labels

    # Helper methods
    def _scannable(self, df_seq, df_feat, region, to_aa, jmd_n_len, jmd_c_len):
        """Return (wt_entry, wt_seq, positions, alphabet, base) via a SeqMut scan."""
        df_scan = self._seqmut.scan(df_seq=df_seq, df_feat=df_feat, region=region, to_aa=to_aa,
                                    jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        positions = sorted(int(p) for p in df_scan[ut.COL_POS].unique())
        alphabet = sorted(str(a) for a in df_scan[ut.COL_TO_AA].unique())
        wt_entry = df_seq[ut.COL_ENTRY].iloc[0]
        wt_seq = df_seq[ut.COL_SEQ].iloc[0]
        base = int(df_seq[ut.COL_TMD_START].iloc[0]) - jmd_n_len
        return wt_entry, wt_seq, positions, alphabet, base

    def _build_fitness(self, df_seq, df_feat, names, sources, goals, constraints, penalty,
                       jmd_n_len, jmd_c_len):
        """Build a cached fitness_fn(genomes)->objective matrix backed by SeqMut.combine."""
        wt_entry = df_seq[ut.COL_ENTRY].iloc[0]
        wt_seq = df_seq[ut.COL_SEQ].iloc[0]
        combine_cols = {ut.COL_DELTA_PRED, ut.COL_DELTA_CPP, ut.COL_SHIFT_SCORE}
        need_combine = any(s in combine_cols for s in sources)
        cache: Dict[Tuple, Dict[str, float]] = {}
        # Per-(objective, variant) cache for callable objectives so a slow external predictor /
        # API is queried once per distinct variant sequence, not once per generation.
        call_cache: Dict[Tuple, float] = {}

        def _score_uniq(genomes):
            uniq = {}
            for g in genomes:
                key = canonical(g)
                if key not in cache and len(g) > 0:
                    uniq[key] = g
            if not (need_combine and uniq):
                return
            rows = []
            for g in uniq.values():
                label = variant_label(wt_seq, g)
                for pos, to_aa in sorted(g.items()):
                    rows.append((wt_entry, label, int(pos), to_aa))
            variants = pd.DataFrame(rows, columns=[ut.COL_ENTRY, ut.COL_VARIANT,
                                                   ut.COL_POS, ut.COL_TO_AA])
            df_var = self._seqmut.combine(df_seq=df_seq, variants=variants, df_feat=df_feat,
                                          jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
            by_label = df_var.set_index(ut.COL_VARIANT)
            for g in uniq.values():
                row = by_label.loc[variant_label(wt_seq, g)]
                sc = {ut.COL_DELTA_CPP: float(row[ut.COL_DELTA_CPP]),
                      ut.COL_SHIFT_SCORE: float(row[ut.COL_SHIFT_SCORE])}
                if ut.COL_DELTA_PRED in by_label.columns:
                    sc[ut.COL_DELTA_PRED] = float(row[ut.COL_DELTA_PRED])
                cache[canonical(g)] = sc

        def fitness_fn(genomes):
            _score_uniq(genomes)
            F = []
            for g in genomes:
                sc = cache.get(canonical(g), {})
                vec = []
                for src in sources:
                    if src == ut.COL_N_MUT:
                        vec.append(float(len(g)))
                    elif callable(src):
                        key = (id(src), canonical(g))
                        if key not in call_cache:
                            call_cache[key] = float(src(apply_genome(wt_seq, g)))
                        vec.append(call_cache[key])
                    elif len(g) == 0:
                        vec.append(0.0)
                    else:
                        vec.append(float(sc[src]))
                F.append(vec)
            F = np.asarray(F, dtype=float)
            if constraints:
                F = apply_penalty(F, genomes, constraints, goals, penalty=penalty)
            return F

        return fitness_fn, wt_entry, wt_seq

    def _impact_weights(self, df_seq, df_feat, best_genome, wt_seq, base, jmd_n_len, jmd_c_len):
        """Refit ShapModel under fuzzy labeling on the best variant; aggregate |feat_impact|."""
        mut_seq = apply_genome(wt_seq, best_genome)
        ts = int(df_seq[ut.COL_TMD_START].iloc[0])
        te = int(df_seq[ut.COL_TMD_STOP].iloc[0])
        df_var = self._df_seq_ref[list(ut.COLS_SEQ_POS) + [ut.COL_ENTRY]].copy() \
            if ut.COL_ENTRY in self._df_seq_ref.columns else self._df_seq_ref.copy()
        var_row = {ut.COL_ENTRY: "__variant__", ut.COL_SEQ: mut_seq,
                   ut.COL_TMD_START: ts, ut.COL_TMD_STOP: te}
        df_all = pd.concat([self._df_seq_ref, pd.DataFrame([var_row])], ignore_index=True)
        df_parts = self._sf.get_df_parts(df_seq=df_all, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        features = list(df_feat[ut.COL_FEATURE])
        X = np.asarray(self._sf.feature_matrix(features=features, df_parts=df_parts,
                                               df_scales=self.df_scales), dtype=float)
        proba = self._model.predict_proba(X[-1:].astype(float))
        proba = np.asarray(proba, dtype=float).ravel()
        p_var = float(proba[-1]) if proba.ndim == 1 and len(proba) else float(np.ravel(proba)[-1])
        p_var = min(max(p_var, 0.0), 1.0)
        labels_fuzzy = list(np.asarray(self._labels, dtype=float)) + [p_var]
        sm = ShapModel(random_state=self._random_state, verbose=False)
        sm.fit(X, labels_fuzzy, fuzzy_labeling=True)
        df_imp = sm.add_feat_impact(df_feat.copy(), samples=int(len(X) - 1), names="var", drop=True)
        impact_cols = [c for c in df_imp.columns if c.startswith(ut.COL_FEAT_IMPACT)
                       and c not in (ut.COL_FEAT_IMPACT, ut.COL_FEAT_IMPACT_STD)]
        col = impact_cols[0] if impact_cols else ut.COL_FEAT_IMPACT
        return residue_weights_(df_imp, col, base)

    def _build_guide(self, df_seq, df_feat, fitness_fn, goals, wt_seq, base, jmd_n_len, jmd_c_len):
        """Build guide_fn(population)->position-weight dict for the configured mode."""
        if self._mode == "importance":
            static = residue_weights_(df_feat, ut.COL_FEAT_IMPORT, base)
            return lambda pop: static
        # mode == "impact": initial prior from df_feat, then per-generation ShapModel refit.
        initial = (residue_weights_(df_feat, ut.COL_FEAT_IMPACT, base)
                   or residue_weights_(df_feat, ut.COL_FEAT_IMPORT, base))

        def guide_fn(pop):
            if pop is None or len(pop) == 0:
                return initial
            F = fitness_fn(pop)
            W = normalize_objectives_(F, goals)
            best = pop[int(np.argmax(W[:, 0]))]
            if len(best) == 0:
                return initial
            return self._impact_weights(df_seq, df_feat, best, wt_seq, base,
                                        jmd_n_len, jmd_c_len) or initial

        return guide_fn

    # Main methods
    def run(self,
            df_seq: pd.DataFrame,
            df_feat: pd.DataFrame,
            objectives: List[Tuple[str, str, Any]],
            algorithm: str = "nsga2",
            pop_size: int = 50,
            n_gen: int = 20,
            crossover: str = "uniform",
            mutation: str = "substitution",
            cx_prob: float = 0.5,
            mut_prob: float = 0.2,
            survival: str = "mu_plus_lambda",
            variation: str = "and",
            engine: str = "exact",
            constraints: Optional[List[Callable]] = None,
            penalty: str = "delta",
            hof_size: int = 10,
            n_mut_max: int = 5,
            region: Optional[Any] = None,
            to_aa: Optional[List[str]] = None,
            init: str = "random",
            seed: Optional[int] = None,
            jmd_n_len: int = 10,
            jmd_c_len: int = 10,
            ) -> pd.DataFrame:
        """
        Run multi-objective directed evolution and return the Pareto front of variants.

        A population of multi-mutation variants of the single wild-type ``df_seq`` is evolved by
        NSGA-II (``algorithm='nsga2'``) or an importance-ordered greedy walk
        (``algorithm='greedy'``), scored on every objective, and reduced to the non-dominated
        trade-off front.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (1, n_seq_info)
            The single wild-type, in the **position-based** format (``sequence``, ``tmd_start``,
            ``tmd_stop``). See :meth:`SequenceFeature.get_df_parts` for the full specification.
        df_feat : pd.DataFrame
            CPP feature set (output of :meth:`CPP.run`) defining the features and the residue
            attribution (``feat_importance`` / ``feat_impact``, ``positions``) the search reads.
        objectives : list of (str, str, object)
            ``(name, goal, source)`` per objective; ``goal`` in ``{'max','min'}`` and ``source``
            in ``{'delta_pred','delta_cpp','shift_score','n_mut'}`` or a ``callable(sequence) ->
            float``. The callable receives the **variant sequence** and returns a scalar, so any
            external predictor (scikit / torch model, or a sequence-level tool / web API such as
            a topology or signal-peptide predictor) can be optimized as an objective; its result
            is cached per distinct variant. At least two objectives.
        algorithm : str, default='nsga2'
            ``'nsga2'`` (population) or ``'greedy'`` (importance-ordered single path).
        pop_size : int, default=50
            Population size (NSGA-II only).
        n_gen : int, default=20
            Number of generations (NSGA-II only).
        crossover : str, default='uniform'
            Crossover operator: ``'uniform'`` / ``'one_point'`` / ``'two_point'``.
        mutation : str, default='substitution'
            Mutation operator: ``'substitution'`` or ``'shift'``.
        cx_prob : float, default=0.5
            Per-pair crossover probability.
        mut_prob : float, default=0.2
            Per-individual mutation probability.
        survival : str, default='mu_plus_lambda'
            Survival scheme: ``'mu_plus_lambda'`` (elitist), ``'mu_comma_lambda'`` or
            ``'ea_simple'`` (generational replacement).
        variation : str, default='and'
            Variation scheme: ``'and'`` (varAnd — crossover *and* mutation) or ``'or'`` (varOr —
            each offspring is crossover *or* mutation *or* reproduction; needs cx_prob+mut_prob<=1).
        engine : str, default='exact'
            ``'exact'`` (pure-Python, RNG-matched to the DEAP reference) or ``'fast'`` (numpy-
            vectorized non-dominated sort; numerically identical fronts, faster).
        constraints : list of callable, optional
            Feasibility predicates ``genome -> bool`` (``True`` = feasible). Infeasible variants
            are penalized so the search avoids them.
        penalty : str, default='delta'
            Penalty applied to infeasible variants: ``'delta'`` (fixed worst objective) or
            ``'closest_valid'`` (penalty scaled by the number of violated constraints).
        hof_size : int, default=10
            Size of the single-objective Hall of Fame (``SeqOpt.hall_of_fame_``) accumulated
            across generations.
        n_mut_max : int, default=5
            Maximum number of point mutations per variant.
        region : str or list of int, optional
            Restrict the mutable span (see :meth:`SeqMut.scan`).
        to_aa : list of str, optional
            Substitution alphabet (see :meth:`SeqMut.scan`).
        init : str, default='random'
            Population initialization: ``'random'`` or ``'suggest'`` (warm start from the top
            single mutations).
        seed : int, optional
            Per-call seed; overrides the constructor ``random_state``.
        jmd_n_len : int, default=10
            Length of JMD-N in number of amino acids.
        jmd_c_len : int, default=10
            Length of JMD-C in number of amino acids.

        Returns
        -------
        df_pareto : pd.DataFrame
            One row per final-population variant with ``entry``, ``variant``, ``n_mut``,
            ``sequence_mut``, one column per objective (named by ``objectives``), the
            non-dominated ``rank`` (0 = best front) and the ``crowding`` distance, sorted by
            ``rank`` then descending ``crowding``.

        Examples
        --------
        .. include:: examples/seqopt_run.rst
        """
        # Validate
        check_match_df_seq_single(df_seq=df_seq)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        names, goals, sources = check_objectives(objectives=objectives, model=self._model)
        # Remember the objective directions so eval() can maximization-normalize df_pareto.
        self._obj_meta_ = dict(zip(names, goals))
        ut.check_str_options(name="algorithm", val=algorithm,
                             list_str_options=ut.LIST_SEQOPT_ALGORITHMS)
        ut.check_str_options(name="crossover", val=crossover,
                             list_str_options=ut.LIST_SEQOPT_CROSSOVER)
        ut.check_str_options(name="mutation", val=mutation,
                             list_str_options=ut.LIST_SEQOPT_MUTATION)
        ut.check_str_options(name="survival", val=survival,
                             list_str_options=ut.LIST_SEQOPT_SURVIVAL)
        ut.check_str_options(name="variation", val=variation,
                             list_str_options=ut.LIST_SEQOPT_VARIATION)
        ut.check_str_options(name="engine", val=engine, list_str_options=ut.LIST_SEQOPT_ENGINE)
        ut.check_str_options(name="penalty", val=penalty, list_str_options=ut.LIST_SEQOPT_PENALTY)
        ut.check_str_options(name="init", val=init, list_str_options=ut.LIST_SEQOPT_INIT)
        ut.check_number_range(name="pop_size", val=pop_size, min_val=2, just_int=True)
        ut.check_number_range(name="n_gen", val=n_gen, min_val=1, just_int=True)
        ut.check_number_range(name="n_mut_max", val=n_mut_max, min_val=1, just_int=True)
        ut.check_number_range(name="hof_size", val=hof_size, min_val=1, just_int=True)
        ut.check_number_range(name="cx_prob", val=cx_prob, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="mut_prob", val=mut_prob, min_val=0, max_val=1, just_int=False)
        if constraints is not None:
            constraints = ut.check_list_like(name="constraints", val=constraints)
            for i, c in enumerate(constraints):
                if not callable(c):
                    raise ValueError(f"'constraints[{i}]' ({c}) should be a callable "
                                     f"genome->bool feasibility predicate.")
        if variation == ut.LIST_SEQOPT_VARIATION[1] and cx_prob + mut_prob > 1:
            raise ValueError(f"variation='or' requires cx_prob + mut_prob <= 1 "
                             f"(got {cx_prob} + {mut_prob}).")
        if seed is not None:
            ut.check_number_range(name="seed", val=seed, min_val=0, just_int=True)
        # Resolve RNG + scannable space
        import random as _random
        resolved_seed = seed if seed is not None else self._random_state
        rng = _random.Random(resolved_seed)
        wt_entry, wt_seq, positions, alphabet, base = self._scannable(
            df_seq, df_feat, region, to_aa, jmd_n_len, jmd_c_len)
        if len(positions) == 0:
            raise ValueError("No scannable positions for the given 'region'.")
        # Fitness + guidance
        fitness_fn, _, _ = self._build_fitness(df_seq, df_feat, names, sources, goals,
                                               constraints, penalty, jmd_n_len, jmd_c_len)
        guide_fn = self._build_guide(df_seq, df_feat, fitness_fn, goals, wt_seq, base,
                                     jmd_n_len, jmd_c_len)
        suggest_seeds = None
        if init == "suggest":
            df_sug = self._seqmut.suggest(df_seq=df_seq, df_feat=df_feat, n=pop_size,
                                          region=region, to_aa=to_aa,
                                          jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
            suggest_seeds = [{int(p): a} for p, a in zip(df_sug[ut.COL_POS], df_sug[ut.COL_TO_AA])]
        # Evolve
        if algorithm == ut.LIST_SEQOPT_ALGORITHMS[1]:       # "greedy"
            res = evolve_greedy(wt_seq, positions, alphabet, goals, fitness_fn, guide_fn,
                                n_mut_max=n_mut_max)
        else:                                               # "nsga2"
            res = evolve_nsga2(wt_seq, positions, alphabet, goals, fitness_fn, guide_fn, rng,
                               pop_size=pop_size, n_gen=n_gen, n_mut_max=n_mut_max,
                               crossover=crossover, mutation=mutation, cx_prob=cx_prob,
                               mut_prob=mut_prob, survival=survival, variation=variation,
                               engine=engine, hof_size=hof_size, suggest_seeds=suggest_seeds)
        self.trajectory_ = list(res["trajectory"])
        # Hall of Fame: best-k single-objective variants (labels) across all generations.
        self.hall_of_fame_ = [variant_label(wt_seq, g) for g in res.get("hall_of_fame", [])]
        # Per-generation history (hypervolume + spread + per-objective best front value).
        self.history_ = self._build_history(res, names)
        df_pareto = self._build_output(res, wt_entry, wt_seq, names)
        if self._verbose:
            n_front = int((df_pareto[ut.COL_RANK] == 0).sum())
            ut.print_out(f"SeqOpt ({self._mode}/{algorithm}) returned {len(df_pareto)} variants "
                         f"({n_front} on the Pareto front).")
        return df_pareto

    def _build_output(self, res, wt_entry, wt_seq, names):
        """Assemble df_pareto from the evolve result (genomes + objective matrix + rank/crowd)."""
        genomes, F = res["genomes"], np.asarray(res["F"], dtype=float)
        rank, crowding = res["rank"], res["crowding"]
        data = {ut.COL_ENTRY: [wt_entry] * len(genomes),
                ut.COL_VARIANT: [variant_label(wt_seq, g) for g in genomes],
                ut.COL_N_MUT: [len(g) for g in genomes],
                ut.COL_SEQ_MUT: [apply_genome(wt_seq, g) for g in genomes]}
        for j, name in enumerate(names):
            data[name] = F[:, j]
        data[ut.COL_RANK] = [int(r) for r in rank]
        data[ut.COL_CROWDING] = [float(c) for c in crowding]
        df_pareto = pd.DataFrame(data)
        df_pareto = df_pareto.sort_values([ut.COL_RANK, ut.COL_CROWDING],
                                          ascending=[True, False])
        # One row per distinct variant (the population may carry duplicate genomes); the sort
        # keeps the best-crowding copy of each.
        df_pareto = df_pareto.drop_duplicates(subset=[ut.COL_VARIANT]).reset_index(drop=True)
        return df_pareto

    def _build_history(self, res, names):
        """Per-generation convergence history (one row per generation)."""
        hv = list(res.get("trajectory", []))
        sp = list(res.get("spread_trajectory", [np.nan] * len(hv)))
        best = res.get("best_trajectory")
        data = {ut.COL_GENERATION: list(range(len(hv))),
                ut.COL_HYPERVOLUME: hv, ut.COL_SPREAD: sp}
        if best is not None and len(best):
            best = np.asarray(best, dtype=float)
            for j, name in enumerate(names):
                data[f"best_{name}"] = best[:, j]
        return pd.DataFrame(data)

    def eval(self,
             df_pareto: pd.DataFrame,
             ref_point: Optional[ut.ArrayLike1D] = None,
             ref_front: Optional[ut.ArrayLike2D] = None,
             ) -> pd.DataFrame:
        """
        Evaluate a Pareto front: hypervolume, front size, spread and (optionally) convergence.

        Parameters
        ----------
        df_pareto : pd.DataFrame
            Output of :meth:`SeqOpt.run`.
        ref_point : array-like, shape (n_objectives,), optional
            Reference (nadir) point for the hypervolume. If ``None``, the per-objective minimum
            (minus a small margin) of the front is used.
        ref_front : array-like, shape (n_ref, n_objectives), optional
            A reference (target) front in raw objective space. When given, a ``convergence``
            column (generational distance to this front; lower = closer) is added.

        Returns
        -------
        df_eval : pd.DataFrame, shape (1, 3 or 4)
            One row with ``hypervolume``, ``n_front`` (rank-0 size) and ``spread`` (plus
            ``convergence`` when ``ref_front`` is given).

        Examples
        --------
        .. include:: examples/seqopt_eval.rst
        """
        # Validate
        ut.check_df(df=df_pareto, name="df_pareto",
                    cols_required=[ut.COL_RANK, ut.COL_CROWDING])
        obj_cols = [c for c in df_pareto.columns if c not in
                    (ut.COLS_PARETO_BASE + [ut.COL_RANK, ut.COL_CROWDING])]
        if len(obj_cols) < 2:
            raise ValueError(f"'df_pareto' should carry at least two objective columns; found "
                             f"{obj_cols}.")
        # Evaluate on the first (rank=0) front
        front = df_pareto[df_pareto[ut.COL_RANK] == 0]
        F = front[obj_cols].to_numpy(dtype=float)
        # Maximization-normalize using the goals remembered from the last run() (every objective
        # flipped to max-is-better); fall back to all-max for a front from an unknown run.
        obj_meta = getattr(self, "_obj_meta_", {})
        eval_goals = [obj_meta.get(c, ut.LIST_OBJECTIVE_GOALS[0]) for c in obj_cols]
        W = normalize_objectives_(F, eval_goals)
        ref = None if ref_point is None else np.asarray(ref_point, dtype=float)
        record = {ut.COL_HYPERVOLUME: hypervolume(W, ref=ref),
                  ut.COL_N_FRONT: int(len(front)), ut.COL_SPREAD: spread(W)}
        if ref_front is not None:
            ref_W = normalize_objectives_(np.asarray(ref_front, dtype=float), eval_goals)
            record[ut.COL_CONVERGENCE] = convergence(W, ref_W)
        df_eval = pd.DataFrame([record])
        return df_eval

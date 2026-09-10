# Pre-v1.1.0 close-out audit

Audited at `origin/master` = `ed1e7efe`. Five issues whose subject becomes
permanently public at the v1.1.0 tag were read against the shipped code.
Nothing on the GitHub tracker was modified — every action below is proposed.

**Release state**: no `v1.1.0` tag anywhere, PyPI still serves `1.0.3`, no GitHub
Release for `v1.1.0` *or* `v1.0.3`. So none of the names below have shipped, and
every rename listed under "API freeze" is still free. That window closes at the tag.

---

## 1. Issue verdicts

| Issue | Prio | Milestone | Verdict |
|---|---|---|---|
| #261 SeqOpt | 1 | v1.2 | **Close** + narrow follow-up |
| #59 Multi-objective optimization | 1 | v1.2 | **Close as superseded by #261** — *maintainer decision, see below* |
| #91 ModelEvaluator | 1 | v1.3 | **Close** + narrow follow-up |
| #241 pipe + transformer | 2 | v1.2 | **Close** + one narrow follow-up |
| #473 Applicability domain / OOD | 2 | v1.2 | **Rewrite, keep open** (~60–65% ships) |
| #480 Calibration + reliability | 2 | v1.3 | **Rewrite, keep open** (~70% ships) |

### #261 / #59 — SeqOpt

Every requirement met or exceeded: 6 plot methods vs 2 promised; `variation`,
`penalty`, `hof_size`, `ref_front` beyond spec; 8 example notebooks with zero
param-coverage gaps; DEAP parity tests for rank, crowding values and survivor
profile. The byte-exact-DEAP-parity and `exact`/`fast` mode KPIs were deliberately
amended away with the maintainer in the loop — decisions, not debt.

Follow-up should contain exactly two items:
1. **Hypervolume monotonicity test.** `history_[hypervolume]` is recorded per
   generation but nothing asserts it is non-decreasing on a fixed seed. This was an
   explicit KPI and is the one acceptance criterion with no test. Needs a
   per-survival-scheme expectation (likely true for `mu_plus_lambda`, possibly false
   for `mu_comma_lambda` / `ea_simple`), which is why it warrants a ticket.
2. **Objective-callable contract drift**: the decision record says
   `callable(df_variant) -> array`; the shipped contract is
   `callable(sequence) -> float`. The shipped form is better — amend the record,
   not the code.

> **#59 is explicitly reserved to you.** #261's body states verbatim: *"Do not
> auto-close #59/#60 — maintainer to decide whether this supersedes #59."* The
> recommendation is close-as-superseded: every #59 criterion is met, and its one
> partial (a numeric per-objective weight vector) is not a gap but a paradigm
> change — weighted scalarization is the thing a Pareto front replaces. If you want
> that path preserved, spin out a small "optional `weights=` for a scalarized
> single-objective run" issue rather than keeping #59 open.

### #91 — ModelEvaluator

All requirements and KPIs met: repeated stratified CV, bootstrap CIs via the same
engine `comp_bootstrap_ci` wraps, paired per-fold deltas, Wilcoxon signed-rank
p-values, 39 + 12 tests, 5 notebooks at zero param gaps. Shipped via PR #436, both
commits tagged `(#91)`.

Residuals for the follow-up:
- `mcc` is accepted by `ModelEvaluator` but rejected by `AAPred.eval`
  (`LIST_METRICS_MODELEVAL = LIST_METRICS_PRED + ["mcc"]`). Two sibling evaluation
  classes disagree on the package's own headline metric. Adding `"mcc"` to
  `LIST_METRICS_PRED` is additive and non-breaking.
- `comp_auc_adjusted` — AAanalysis's own AUC metric — is not selectable; metrics
  come from `sklearn.metrics` directly.
- Each round re-seeds only the CV split, not the estimator, so model-seed variance
  is never sampled.

### #241 — pipe + SequenceFeatureTransformer

Root cause of the stale-open state confirmed: the acceptance checkboxes name an API
that was deliberately superseded before release, so no checkbox ever matched.
`cpp_feature_map` → `find_features`, `predict` → `predict_samples`, `explain` →
`explain_features`, the `dpulearn` flag relocated onto
`obtain_samples(reliable_negatives=True)`, plus two extra pipelines. Byte-identical
parity tests exist for 3 of 5 Ends; the leak-free `cross_val_score` KPI is directly
asserted.

Follow-up: **`pipe: parity anchor for predict_samples, the ≤10-line KPI, and
param-coverage gating for ap notebooks`**
- `ap.predict_samples` has no byte-identical parity test against the explicit chain.
- `find_features` parity is anchored only at `search="fast"`; rule `balanced` /
  `exhaustive` in or explicitly N/A.
- The "≤10 lines" KPI is asserted nowhere.
- `test_notebook_param_coverage.py` iterates `aaanalysis.__all__`, which by design
  excludes `pipe` — so `examples/pipe/*.ipynb` are the only example notebooks with
  no coverage gate, and `obtain_samples.pos_col` / `plot_eval.score_col` are already
  undemonstrated.
- No narrative "Golden pipelines" page under `usage_principles/`.

### #473 — Applicability domain / OOD

~60–65% ships, including two AD methods it never asked for (Mahalanobis, leverage).
Rewrite down to the genuine remainder:
1. ~~Degenerate reference silently reports every sample in-domain~~ — **fixed on
   branch `fix/pre-tag-defects`**.
2. `in_domain` is a bare bool; no `borderline` / `unknown` banding, and the fitted
   threshold lives in private state (no public `ad_threshold_`).
3. `ad_nearest_train` — the neighbour indices are already computed inside
   `apply_applicability_domain` and discarded.
4. **No candidate-set entry point.** `grep` over `aaanalysis/protein_engineering/`
   for the AD columns returns zero hits, yet `SeqMut` / `SeqOpt` push candidates away
   from the training cloud by construction. This is the issue's stated motivation and
   has no implementation.

Scope corrections to record as non-goals: the sequence-identity AD method is
self-contradicting (`comp_seq_sim` is `[pro]`/biopython, but the issue demands "core,
no new required dependency"); no second `ApplicabilityDomain` class; no `method=`
selector.

### #480 — Calibration + reliability

~70% ships: Platt (`"sigmoid"`) and isotonic via `CalibratedClassifierCV`, binned
calibration curve with configurable `n_bins`, `reliability_diagram`, raw/calibrated
separation (`score` vs `score_calibrated`).

**The remainder is not just Brier + ECE — there is a third item that blocks the
headline KPI.** `eval()` bins the **raw** `score`, never `score_calibrated`
(`_reliability_model.py:423`). So switching `calibration_method` cannot change a
single number `eval()` returns, and adding Brier/ECE on the raw score alone would
inherit that and produce a metric constant in the calibration setting. The eval
surface must be able to score the calibrated column first.

Non-goals to record: no `ProbabilityCalibrator` class (duplicate surface over the
same sklearn wrapper); no multiclass (the class is binary-only by hard contract); no
nested-CV plumbing (`CalibratedClassifierCV` already cross-fits internally).

---

## 2. API-freeze decisions — free now, a deprecation cycle after the tag

These are **yours to call**; none was applied.

| # | Item | Why it matters |
|---|---|---|
| F1 | **`SeqOpt.run` lost its keyword-only `*`** | 19 optional params are positional-or-keyword, so their **order freezes on publish**. Nobody will call `run(df, feat, objs, "nsga2", 50, 20, ...)` positionally — this locks the signature for zero user benefit. The repo already uses `*` in this exact situation in `_seqmut_plot.py`, `_cpp_plot.py`, `_aaclust_plot.py`, `_shap_model.py`, `_scan_motif.py`. |
| F2 | **`aa.SeqOpt()` raises; `aa.SeqMut()` and `aa.AAMut()` do not** | Default `mode="impact"` requires a fitted model, labels **and** `[pro]`. So the default config of a class documented as core is unusable in a base install, and every discovery path (`aa.SeqOpt?`, tab-completion, an agent probing the API) hits a `ValueError`. `mode="importance"` is core-only and works bare. Changing a default post-tag is a behaviour break. |
| F3 | **`verbose` is 8th in `SeqOpt.__init__`, 1st in both siblings** | `SeqMut(True)` sets verbose; `SeqOpt(True)` sets `mode=True` and raises. Making `__init__` keyword-only dissolves this permanently (pairs with F1). |
| F4 | **`ci` units clash inside `prediction/`** | `ModelEvaluator.run(ci=0.95)` is a **fraction**; `ReliabilityModel.fit(ci=90.0)` is a **percent**. Both new in 1.1.0, both emit `ci_low`/`ci_high`. `ModelEvaluator` matches `comp_bootstrap_ci(ci=0.95)` and is the one to keep. |
| F5 | **`ad_knn_dist` breaks its own pattern** | Siblings are `ad_mahalanobis` and `ad_leverage`, so `ad_<method>` is the convention. Rename to `ad_knn`. |
| F6 | **`n` as a returned column** in `ReliabilityModel.eval` | Every other count column is spelled out (`n_scores`). Suggest `n_samples`. Breaking after the tag. |
| F7 | **`in_domain` is a bool that cannot carry `unknown`** | If `ad_status` (#473) lands later it sits forever beside a bool that had to be redefined. Adding it pre-tag is free. |
| F8 | **"Experimental" banners contradict `__all__` membership** | `ModelEvaluator`, `AAPred`, `SequenceFeatureTransformer` carry *"API may change between minor releases without the usual deprecation cycle"* while being in `__all__`, i.e. the semver contract. `aaanalysis.pipe` resolved this correctly by staying **out** of `__all__`. Decide which way each goes — this one softens or hardens every row above. |
| F9 | **`score_std` means two things** | Std across CV folds in `AAPred`/`ModelEvaluator`; std across ensemble members in `ReliabilityModel`. Same public name, same shared constant, different statistic. |
| F10 | **`eval()` frame columns are string literals**, not `ut.COL_*` | `bin` / `mean_score` / `empirical_pos` / `n` are the only public output frame without a `COLS_*` bundle, and the summary row overloads `mean_score` and `empirical_pos` with unrelated quantities. |

---

## 3. Applied locally (two branches, nothing pushed)

**`doc/changelog-public-api-truth`** — `e8054b80`
- Adds the 11 public symbols the terse CHANGELOG omitted (prediction tier, design
  tier, `SequenceFeatureTransformer`, `aaanalysis.pipe`, `COLOR_SAMPLES_*`). The
  narrative release notes already covered these; only the developer-facing index lagged.
- Replaces `Deprecated: None` with the real `AAlogo` / `AAlogoPlot` entry, in both
  files. The shim is correctly implemented; the docs denied it existed.
- Corrects the 1.0.3 date in both files: 2026-04-06 → 2026-04-28 (tag commit and
  PyPI upload agree on the latter).

**`fix/pre-tag-defects`** — `95b1b04c`
- `apply_applicability_domain` no longer reports `in_domain=True` for every sample
  when the training kNN threshold collapses to 0. Now yields `ood_score=NaN`,
  `in_domain=False`, so the composite `reliable` flag stays False. Reachable on
  realistic input: a few distinct rows each heavily duplicated clears
  `check_X_unique_samples` while still zeroing every training distance.
- Regression test added; it fails on the previous code.
- `.. versionadded:: 1.0.0` → `1.1.0` on `SeqOpt` and `SeqOptPlot` (neither existed
  at v1.0.3 — that tag exports only `AAMut`/`AAMutPlot`/`SeqMut`/`SeqMutPlot`).
- 655 tests pass across `reliability_model_tests`, `prediction_tests`, `seqopt_tests`.

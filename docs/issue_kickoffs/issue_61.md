# Session kickoff — Issue #61: Multi-class & regression CPP via random reference generation

**Verdict:** ✅ Ready (prio:1, `type:feature`, topic:core) — **smaller than it looks; ship an MVP slice.**
**Run as:** `/github-issue-handoff` (context) → `/grill-with-docs` (resolve decisions) → implement.

## Scope / standards (core vs pro)
Three deliverables; the hard part (reference generation with the seed contract) already exists in `sample_synthetic`.
- **(a) reference generator** → CORE (reuse `sample_synthetic` backend: `scrambled`=shuffle, `global_freq`=composition-matched, `sample_different_protein` against background=length-matched).
- **(b) multi-class CPP** → CORE orchestration (loop over the existing binary `CPP.run`).
- **(c) regression CPP (quantile-split)** → CORE (reduces to binary `run`).
- **multi-class SHAP + SHAP regression** → **PRO** (`explainable_ai_pro/_shap_model.py`); the only genuinely new complexity — **defer to a separate PR.**

## Contract facts (cite)
- `CPP.run(labels, label_test=1, label_ref=0)` is strictly binary — `check_labels(..., allow_other_vals=False)` (`_cpp.py:468`); same in `run_num` (`:681`), `eval` (`:824`). **Do not loosen this shared guard** — build binary label vectors in the new methods and delegate.
- `CPP.__init__` binds `df_parts`/`df_scales` at construction (`_cpp.py:310-313`); `run` varies only `labels`/thresholds.
- `ShapModel.fit(... label_target_class=1)` binary-only (`_shap_model.py:80,127`); SHAP values 2D (`:253`).
- Seed contract per `reproducibility.md` (`_aa_window_sampler.py:356-360`).

## Files
- Reuse: `seq_analysis/_backend/aa_window_sampler/sample_synthetic.py` (generation engine), `feature_engineering/_cpp.py` (`run`, `_finalize_run_output`), `_sequence_feature.py` (`feature_matrix:510`, `get_df_parts:178`), `_utils/check_data.py:137` (`check_labels`).
- Create: reference generator (see D1); thin `CPP.fit_multiclass`/`fit_regression` (+ backend merge helper if needed); tests under `aa_window_sampler_tests/`, the cpp test dir, `shap_model_tests/`.

## Decisions for grilling (recommended answers)
- **D1 Generator location** → add as a method `AAWindowSampler.generate_reference` reusing `sample_synthetic` — **avoids the CONFIRM-FIRST `__init__.py` edit**, reuses seed contract + schema. (Literal top-level `generate_reference_sequences` = CONFIRM-FIRST; only if maintainer insists.)
- **D2 Multi-class** → **reuse `CPP.run` per class** (one-vs-rest); aggregate K `df_feat` into a multi-indexed frame. Acceptance "per-class == binary-per-class" holds by construction.
- **D3 `reference=None`** → one-vs-rest (class c vs merged rest); `reference=<set>` → each class vs shared generated reference. Ship one-vs-rest default.
- **D4 Regression** → **quantile-split first** (`fit_regression(sequences, targets, quantile=...)` → binary `run`). Defer reference-baseline + SHAP-regression correlation (needs benchmark).
- **D5 Methods vs functions** → `CPP.fit_multiclass` / `CPP.fit_regression` as methods (no `__init__.py` edit). Resolve the `df_parts`-binding wrinkle: MVP **requires reference rows pre-included** in `df_parts` (keeps the constructor invariant); internal extension via `get_df_parts` is future.
- **D6 SHAP** → confine all multiclass/regression SHAP to `_shap_model.py` (pro); 3D/dict SHAP storage + regressor defaults. **Last slice, separate PR.**
- **D7 Length matching** → references must match CPP part lengths (fixed TMD/JMD), not arbitrary length.

## MVP slice (ship first, independently valuable)
1. `AAWindowSampler.generate_reference` (3 methods, reuse backend) — closes acceptance #1, zero new algorithm.
2. `CPP.fit_multiclass` one-vs-rest delegating to `run` + merge helper — closes acceptance #2.
3. `CPP.fit_regression` quantile-split delegating to `run`.
4. *(later, pro)* multiclass SHAP + SHAP regression + benchmark (acceptance #3).

## Test plan
Reference composition similarity within tolerance (reuse `metrics.comp_kld`); shuffle = exact composition; **multiclass per-class `df_feat` == `run(binary_for_c)` row-for-row**; regression == `run` on quantile labels; seed determinism; SHAP shape tests skipped when `shap` absent.

## Launch command (paste in a fresh session)
```
/grill-with-docs Implement issue #61 per docs/issue_kickoffs/issue_61.md. First load github-issue-handoff
context. Resolve D1–D7 and ship the MVP slice (generate_reference + fit_multiclass one-vs-rest +
fit_regression quantile-split, all reusing CPP.run); keep all SHAP work pro and in a separate later PR.
```

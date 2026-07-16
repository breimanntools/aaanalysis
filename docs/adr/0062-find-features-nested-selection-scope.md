# ADR-0062 — find_features nested feature selection (selection_scope="global"|"fold")

Status: Accepted — 2026-07-16

## Context

`ap.find_features` selects CPP features on the **full** labeled set (`CPP(...).run(labels=...)`) and
then cross-validates the model on that fixed feature matrix `X`. Feature selection therefore sits
**outside** the cross-validation, so every score it reports — the whole returned `df_eval` sweep —
is post-selection / in-sample optimistic: the selection has already seen the held-out fold. It is an
adequate *relative* ranking signal but a misleading *absolute* generalization estimate, and
`find_features` is a shipped, agent-facing "golden pipeline" whose number a downstream agent reads
as held-out and over-trusts (issue #411). The class docstring already carried this caveat as an
Experimental warning.

## Decision

Add `selection_scope: Literal["global", "fold"] = "global"` to `find_features`.

- **`"global"` (default):** unchanged — CPP selects on all data, the model is cross-validated on the
  fixed `X`. **Byte-identical to the prior behaviour** (regression-guarded), so nothing about the
  shipped default changes.
- **`"fold"` (honest nested, B-uniform):** every place a CPP *configuration* is scored re-runs the
  selection inside the cross-validation. A new `_nested_cv_scores` scorer, given a per-fold
  `_config_selector`, uses the same `StratifiedKFold(shuffle=False)` fold geometry as the global
  `cross_validate` path and, per fold, re-runs `CPP.run` (and `CPP.simplify` where applicable) on the
  **train split only**, builds the feature matrix for the train and held-out rows from those
  train-selected features, fits the model on train, and scores the held-out fold — then averages over
  folds and models (the same `{metric: (mean, std)}` shape as the global scorer).

`df_eval` gains a `selection_scope` column naming the regime that produced each row. The returned
`df_feat` is **always** the winning configuration refit on all data (outer-CV semantics) in both
scopes — `"fold"` changes only how configurations are *scored/ranked during selection*, not the
final artifact. A cost `UserWarning` fires for `selection_scope="fold"` with `search="exhaustive"`
(the widest grid), and the docstring pairs `"fold"` with `search="fast"`/`"balanced"`.

### Scope of the nesting (and what runs on all data)

Fold nesting is applied at every **configuration-selection** score: Stage 1 grid and Stage 2 axis
refine (both via `_grid_stage`) and the `search="fast"` single-config score. That is where the
selection leakage the issue describes lives — which configuration wins, and what score is reported
for it.

The winner's **Stage-3 refinement** (`CPP.simplify` + recursive feature elimination) runs on **all
data in both scopes**, producing the final returned `df_feat` (the winning configuration refit on
all data — outer-CV semantics). So `"fold"` does **not** drop any refinement capability: simplify
and RFE behave exactly as in `"global"`; only the *config-selection* scores that precede them are
nested. (An earlier iteration nested the simplify keep-guard and skipped RFE in fold mode; that was
reverted to avoid silently dropping the second-step RFE capability and to keep the `"global"` path
byte-identical.) Truly nesting the whole selection+refinement *wrapper* per fold is the separate
paper-fidelity nested-CV Monte-Carlo engine tracked in **#276**.

## Consequences

- Default output (`df_feat` and the per-config scores) is byte-identical; only the additive
  `selection_scope` column is new on `df_eval`. `find_features` is not on the perf A/B benchmark
  path, and the underlying `CPP.run`/`feature_matrix` hot paths are unchanged.
- `"fold"` is much more expensive (it re-runs CPP `cv` times per configuration), hence the cost
  warning and the guidance to pair it with lighter `search` grades.
- No new dependency; core only.

## Alternatives considered

- **Making `"fold"` the default** — rejected: it changes every shipped score and is far costlier;
  the honest regime is opt-in, the fast ranking signal stays the default.
- **Nesting RFE per fold too** — that is the full wrapper nested-CV (#276); folding it in here would
  blur the two issues and multiply cost again. Deferred with an explicit pointer.

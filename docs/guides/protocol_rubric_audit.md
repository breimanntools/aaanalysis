# Protocol rubric audit: P2, P3, P6, P9, P10

The recorded audit of protocols **P6 (compositional vs positional)**, **P9
(interpretability)** and **P10 (validation)** against
[`protocol_quality_rubric.md`](protocol_quality_rubric.md), with Protocol 1 open
beside each one.

These three protocols are the ones the rubric singles out as *audited-and-tightened*
rather than *assumed to need a figure*: P6 and P9 already carried several figures,
and P10 had gained the package learning curve shortly before this audit. So the
deliverable here is the audit itself, plus whatever it turned up, not a figure quota.

Audited at `origin/master` = `48ad6e9a`. Every finding below was read off the
committed notebook, and every one marked **closed** was fixed in the same change.

---

## 1. Verdict at a glance

| Protocol | §1 Concept contrast | §2 Public surface | §3 Skeleton | §4 Mechanics | Entry verdict |
|---|---|---|---|---|---|
| **P6** Compositional vs positional | partial | **fail** | partial | **fail** | **FAIL** |
| **P9** Interpretability | partial | **fail** | partial | **fail** | **FAIL** |
| **P10** Validation | partial | partial | partial | partial | **FAIL** |

P2 and P3 were audited later and are recorded separately in [§7](#7-p2-and-p3-audited-after-the-fact),
because they entered that audit already passing.

None of the three passed on entry. The most serious single finding is in P6 and is
not a teaching-depth issue at all: the notebook asserted that a CPP strategy switch
does not exist, which stopped being true when `SequenceFeature.get_split_kws` gained
its `strategy` presets. After the changes recorded in §5, all three pass §1 to §4.

---

## 2. P6: Compositional vs positional

### §1 Concept-contrast visualizations: partial

Concepts the prose names, and whether a figure shows each:

| Concept | Shown? | Where |
|---|---|---|
| compositional vs positional signature | yes | split-type bar chart, then two feature maps |
| `Segment` vs `Pattern` vs `PeriodicPattern` | yes | split-type bar chart |
| **locality of the `positions` span** | **no** | asserted in *Output*: "a single broad span for compositional; tight, specific positions for positional" |
| compositional ~ protein, positional ~ residue, domain uses both | no | prose table only |

The missing one matters, because span locality *is* the protocol's thesis. The
*Output* section told the reader that compositional features average over one broad
span while positional features resolve to tight position sets, and then showed no
picture of it. The two feature maps do not close this: both are drawn over the same
part axis, so they contrast *which* properties win, not *how wide* each feature looks.

**Gap:** one concept-contrasting figure of the `positions` span.

### §2 Full public-surface coverage: fail

- **`SequenceFeature.get_split_kws`**, the protocol's focal method, passed
  `split_types`, `n_split_min`, `n_split_max`, `steps_pattern`,
  `steps_periodicpattern`. It never passed `n_min`, `n_max`, `len_max`, or
  **`strategy`**.
- **`CPP.run`** passed `labels` and `n_filter` only, so none of the redundancy knobs
  (`max_cor`, `max_overlap`, `check_cat`, `max_std_test`, `parametric`,
  `label_test` / `label_ref`, `n_jobs`) were visible.
- `load_dataset`, `get_df_parts`, `feature_matrix`, `TreeModel.fit` and
  `add_feat_importance` were each called with the minimum.

### §2b Source-verified API: fail (the serious one)

Two statements in the notebook were **false against the shipped signature**:

> "You never set a `strategy=` switch: the strategy *emerges* from the `split_kws`
> recipe you hand to `CPP`."

> "There is no `strategy=` parameter; the strategy *emerges* from `split_kws`."
> (*Common mistakes*)

`SequenceFeature.get_split_kws` takes
`strategy : {'compositional', 'positional'} or None`, added in 1.2.0. It is a
documented preset that sets `split_types`, `n_split_min` and `n_split_max`, and it
raises if any of those three is also given. The rubric requires "real,
source-verified API only", so this is a hard fail, and it is the finding most likely
to actively mislead a reader: the protocol talked them out of using a parameter that
exists and that does exactly what the protocol is about.

The underlying teaching point still holds and is worth keeping: a strategy is a
*shape of `split_kws`*, not a mode inside `CPP`. `strategy` is a naming convenience
for the two canonical shapes, and the returned dictionary is equal to the explicit
call. That is the corrected framing.

### §3 Narrative skeleton: partial

All seven bold lead-in fields are present and in order, and the protocol opens with
intuition before any API. One defect: *When not to use it* pointed at
**"P4: Engineer features"**. P4 is *Prediction levels*; feature engineering is
**P5**. A broken cross-reference in the one sentence that routes a lost reader
elsewhere.

### §4 Presentation mechanics: fail

- Every `display_df` call passed `n_rows=5` or `n_rows=10` with **no
  `show_shape=True`**, against the rubric's explicit
  `aa.display_df(df, n_rows=10, show_shape=True)`.
- The two `feature_map` cells ended with `plt.show()` and **no
  `plt.tight_layout()`**.
- Executed outputs committed, fixtures small and seeded, no em dashes, no issue
  references: all fine.

On `tight_layout` for composed CPP figures, the audit checked rather than assumed,
and the first check was not good enough. `feature_map` and `heatmap` freeze their own
composed layout, so adding a trailing `plt.tight_layout()` could plausibly have
disturbed them. A first A/B rendered both ways through `savefig` and came out
byte-identical, but that is not how a notebook renders a figure: the inline backend
saves with `bbox_inches="tight"`, which `savefig` was not doing, so the test did not
cover the case it was meant to. The check that settles it is a control execution:
the unmodified notebook from `HEAD` was executed in the same environment as the
edited one, and their figures compared. **Identical.** Adding the call satisfies the
rubric at zero visual cost, confirmed under the rendering path the notebook actually
uses.

---

## 3. P9: Interpretability

### §1 Concept-contrast visualizations: partial

| Concept | Shown? | Where |
|---|---|---|
| signed SHAP impact anchored to residues | yes | CPP-SHAP heatmap (impact mode) |
| feature *value* difference vs impact | yes | the mean-difference heatmap next to it |
| coherent blocks vs isolated cells | yes | readable off the heatmap |
| ranking and cumulative per-position impact | yes | ranking plus profile |
| **feature importance vs feature impact** | **no** | asserted twice in prose |

The importance / impact distinction is the *first* thing the protocol teaches, in its
opening paragraph, and it is repeated in *Common mistakes* as the trap that makes
`shap_plot=True` produce nonsense. Every figure in the protocol was drawn from the
sample-level impact column, so the group-level quantity it is being contrasted
against never appeared. The reader was asked to take the protocol's central
distinction on faith.

It is also a contrast with real content on this fixture, not a formality: the rank
correlation between group importance and APP's absolute impact is only **0.58**, and
several features carry a non-zero impact for APP at exactly `feat_importance = 0.0`.

**Gap:** one figure putting unsigned group importance beside signed per-sample impact
for the same features.

### §2 Full public-surface coverage: fail

`ShapModel` is demonstrated across four calls, and each was near-minimal:

- `ShapModel(...)`: `verbose`, `random_state`. Missing `explainer_class`,
  `explainer_kwargs`, `list_model_classes`, `list_model_kwargs`.
- `fit(...)`: `labels`, `n_rounds`. Missing `label_target_class`, `is_selected`,
  `fuzzy_labeling`, `fuzzy_aggregation`, `n_background_data`, `df_seq`,
  `fuzzy_labels`.
- `add_feat_impact(...)`: `df_feat`, `samples`, `names`. Missing `drop`, `normalize`,
  `group_average`, `shap_feat_importance`, `df_seq`, `sample_positions`.
- `add_sample_mean_dif(...)`: missing `label_ref`, `drop`, `group_average`, `df_seq`,
  `sample_positions`, `X_ref`.

### §3 Narrative skeleton: partial

Seven fields present and ordered. One defect, the same class as P6's: *Input* cited
**"P7: Build an interpretable classifier"**. P7 is *Select & reduce features*; the
classifier is **P8: Prediction**.

### §4 Presentation mechanics: fail

- `display_df` without `show_shape=True`.
- The two heatmap cells ended with `plt.show()` and no `plt.tight_layout()` (same
  byte-identical finding as P6).
- Everything else fine: executed outputs, `verbose=False`, seeded, small fixture.

---

## 4. P10: Validation

P10 was migrated to `ModelEvaluator.learning_curve` /
`ModelEvaluatorPlot.learning_curve` shortly before this audit, and it is audited in
that state.

### §1 Concept-contrast visualizations: partial

Six checks are named; three had a figure.

| Check | Shown? |
|---|---|
| 1 repeated stratified CV | as fold scores inside the real-vs-shuffled figure |
| 2 bootstrap CI of the mean | scalar dict |
| 3 shuffled-label control | yes, the control figure |
| 4 **feature stability under resampling** | **no**, a single number |
| 5 biological sense | table of top features (appropriate, a table is the right form) |
| 6 generalization headroom | yes, the learning curve |

The migration closed the rubric's "P10 gains at least one figure" requirement. What
it left behind is Check 4: an entire named check whose whole output is
`0.992`. A reader cannot tell from that number whether the effect sizes are stable
because the signal is real, or because the metric is insensitive. The contrast that
makes it legible is the one the protocol already teaches two checks earlier: the same
scatter under shuffled labels.

**Gap:** one figure for feature stability, contrasted against its shuffled-label null.

### §2 Full public-surface coverage: partial

Better than P6 and P9. Complete already:

- `ModelEvaluator.learning_curve`: `X`, `labels`, `train_sizes`, `n_cv`, `n_rounds`,
  `metrics`, `ci`, `random_state`. All of them.
- `comp_bootstrap_ci`: `values`, `n_rounds`, `ci`, `seed`. All of them.

Incomplete:

- `AAPredPlot.predict_group` was called with its data frame **positionally**, against
  the rubric's "prefer keyword calls throughout", and passed 4 of the parameters that
  apply to `kind="rank_scatter"`.
- `ModelEvaluator(...)`: missing `list_metrics`.
- `ModelEvaluatorPlot.learning_curve(...)`: missing `colors`, `show_ci`.
- `comp_auc_adjusted(...)`: missing `label_test`, `label_ref`, `n_jobs`.
- `CPP.run` and `load_dataset`: minimal, as in P6.

### §3 Narrative skeleton: partial

Seven fields present and ordered, and the *When to use it* check table is a model of
the form. One defect, again a stale protocol number: *Run* cited
**"P7: Build a classifier"** for the native-evaluator path. That is **P8: Prediction**.

### §4 Presentation mechanics: partial

- One `display_df` without `show_shape=True` (the `top` features table); the
  learning-curve table already had it.
- All three plot cells already ended `plt.tight_layout()` then `plt.show()`. Pass.
- The `stdout` stream output in the checks cell is `display_df(show_shape=True)`
  printing the frame shape, which is expected, not a leak.
- Executed outputs committed, seeded, fixtures small.

---

## 5. What changed

Applied in one change, with all three notebooks re-executed and their outputs
committed.

**Common to all three**

- `display_df(df=..., n_rows=10, show_shape=True)` everywhere.
- `plt.tight_layout()` before `plt.show()` in the composed CPP figure cells,
  verified against a control execution of the unmodified notebook (see §2 §4) so that
  no figure moves as a result of the change.
- The stale protocol cross-references corrected: P6 "P4: Engineer features" to
  **P5**, P9 "P7: Build an interpretable classifier" and P10 "P7: Build a
  classifier" to **P8: Prediction**.
- A *further parameters* cell per protocol that exercises the remaining public
  surface by name, deliberately placed so it does not feed any narrative figure.
  That keeps §2 satisfied without perturbing the outputs the prose reads.

**P6**

- The false "there is no `strategy=` switch" claim is gone from both the lede and
  *Common mistakes*, replaced by the accurate framing plus a cell that calls
  `get_split_kws(strategy="compositional")` and `strategy="positional"` and asserts
  the preset equals the explicit recipe it names.
- New figure: **where each feature looks**, one row per selected feature, marking the
  residue positions it averages over, compositional above and positional below on a
  shared axis. One whole-part span of 20 residues against a median span of 4.
- Full `get_split_kws` surface, and the `CPP.run` filter knobs.

**P9**

- New figure: **importance vs impact**, the same features shown as unsigned
  group-level importance beside signed per-sample impact for APP, with both impact
  directions represented so the "signed" half is actually signed.
- Full `ShapModel` constructor, `fit`, `add_feat_impact` and `add_sample_mean_dif`
  surface.

**P10**

- New figure: **feature stability under resampling**, per-feature full-data effect
  size against its bootstrap mean, with the shuffled-label null overlaid.
  `r = 0.99` on the diagonal against `r = -0.23` scattered off it.
- `predict_group` called with `data=` by keyword and its rank-scatter parameters
  named; `list_metrics`, `colors`, `show_ci`, and the `comp_auc_adjusted` label and
  `n_jobs` parameters passed.

The narrative skeleton and the order of the seven fields are unchanged in all three,
and in each case the first output figure, which is the gallery image, is untouched.

### Re-execution drift, and why the thumbnails were left alone

Re-executing a notebook at all turned out to move two of the three first figures
against their committed PNGs, before any edit was applied. The control executions
separate the two causes:

| First figure | committed vs `HEAD` re-executed | `HEAD` re-executed vs edited |
|---|---|---|
| P6 split-type bars | 12.4% of pixels differ | **identical** |
| P9 CPP-SHAP heatmap | 897x856 becomes 903x860 | **identical** |
| P10 rank plot | identical | identical |

So none of it is caused by this change: it is environment drift between the machine
that last executed these notebooks and the one executing them now. Comparing the P6
figure by eye confirms the nature of it. Every bar is the same height (Segment 20
against 16, Pattern 12, PeriodicPattern 2), the feature selection is unchanged, and
what differs is glyph rasterization, a slightly different font rendering of the same
chart.

The three gallery thumbnails were therefore **not** regenerated. They are built by
their own scripts under `docs/source/_artwork/thumb_scripts/`, not extracted from the
notebooks, and they remain accurate: no thumbnail shows anything the protocol page no
longer shows. Re-rendering them here would push this machine's font rasterization into
3 of the 10 gallery tiles and leave the other 7 as they are, buying a visible
inconsistency in exchange for nothing. If the thumbnails are ever refreshed, all ten
should be refreshed together on one machine.

## 6. Exit verdict

| Protocol | §1 | §2 | §3 | §4 | Verdict |
|---|---|---|---|---|---|
| **P6** | pass | pass | pass | pass | **PASS** |
| **P9** | pass | pass | pass | pass | **PASS** |
| **P10** | pass | pass | pass | pass | **PASS** |

## 7. P2 and P3: audited after the fact

P2 and P3 never received epic children, so they were audited last, at
`origin/master` = `64a0cdef`. They are the only two protocols that **entered the audit
already passing**, and the reason is on the record: the gap-closing work the other
protocols received in their audit had already shipped for these two, in

- `bd5a87cc` docs(protocols): bring P2 sequence analysis and P3 sampling to the protocol rubric
- `675657e5` docs(protocols): address review findings on the P2/P3 rubric pass
- `a416a53a` docs(protocols): contrast motif-matched sampling with a figure
- `27b9a387` docs(protocols): correct the claims about sampling anchors and loader behaviour

What was missing was the audit **record**, not the work. This section is that record.
**No notebook was changed by it**, which is the correct outcome: adding a further figure
to a protocol that already shows every concept it names would be decoration, and the
rubric asks for concept contrast, not a figure quota.

### Verdict

| Protocol | §1 Concept contrast | §2 Public surface | §3 Skeleton | §4 Mechanics | Entry verdict | Exit verdict |
|---|---|---|---|---|---|---|
| **P2** Exploratory sequence analysis | pass | pass | pass | pass | **PASS** | **PASS** (unchanged) |
| **P3** Sampling | pass | pass | pass | pass | **PASS** | **PASS** (unchanged) |

### P2: Exploratory sequence analysis

**§1 Concept-contrast visualizations: pass.** Six figures, and every concept the prose
names is one of them, each drawn against its alternative:

| Concept | Shown? | Contrast |
|---|---|---|
| composition per position | yes, the pooled probability logo (gallery image) | reference view |
| composition vs conservation | yes, the information logo | same set, letters scaled to bits |
| signal vs chance | yes, real vs shuffled conservation bars | same residues, order destroyed |
| the chance floor depends on set size | yes, bits vs number of sequences | real against its shuffled twin, `n` = 6 to 126 |
| one pooled set vs two groups | yes, substrate vs non-substrate logos | same parts, split by `label` |
| a first two-group look by sampling | yes, the window logos | two sampling strategies |

The *Key mental model* makes three claims (letter height is composition, the gray bar is
conservation; conservation needs a baseline; a pooled logo describes rather than
compares) and each has its own figure. There is no concept left asserted in prose only,
so no figure was added.

**§2 Full public-surface coverage: pass.** Twelve demonstrated call sites, every public
parameter passed by name: `load_dataset` 8/8, `SequenceFeature.get_df_parts` 8/8,
`AALogo` 1/1, `get_df_logo` 7/7, `get_df_logo_info` 7/7, `get_conservation` 2/2,
`AALogoPlot` 4/4, `single_logo` 33/33, `multi_logo` 29/29, `AAWindowSampler` 7/7,
`sample_same_protein` 17/17, `sample_different_protein` 16/16.

**§3 Narrative skeleton: pass.** Seven bold lead-in fields, in order, with the *Key
mental model* ahead of any API. Both cross-references (*P1: CPP signature*,
*P3: Sampling*) match the target protocols' actual H1 titles, so the stale-reference
defect found in P6, P9 and P10 does not occur here.

**§4 Presentation mechanics: pass.** All 6 `display_df` calls pass `n_rows` **and**
`show_shape=True`; every plot cell ends `plt.tight_layout()` then `plt.show()`; 6
committed `image/png` outputs; no cell with `output_type == "error"`; no code cell with
a null `execution_count`; kernelspec `python3`; no em dashes.

**§2b Source-verified API: pass.** Each checkable claim was read off the shipped
signature or behaviour rather than trusted: the default parts of `get_df_parts`
(`tmd` / `jmd_n_tmd_n` / `tmd_c_jmd_c`), `tmd_len` applying to the anchor-based `df_seq`
format only, `aa_window_size` applying to the residue-level `AA_*` datasets only, and
`get_conservation` accepting `min` / `mean` / `median` / `max`. The quantitative claim in
*How to interpret* ("the TMD rises well above its shuffled twin, mean 1.37 vs 0.62 bits")
matches the committed output of the region table exactly. `4.32` bits for a
fully-conserved residue is `log2(20)`.

### P3: Sampling

**§1 Concept-contrast visualizations: pass.** Five figures, one per sampling choice, and
each holds the proteins and `n` fixed so the only difference is the choice being taught:

| Concept | Shown? | Contrast |
|---|---|---|
| the assembled reference set | yes, windows on substrate proteins + role composition (gallery image) | test windows against sampled negatives |
| the distance band | yes, distance to nearest positive | no band vs non-overlapping vs neighbourhood |
| where the reference comes from | yes, composition heatmap + distance table | test composition against five candidate references |
| motif-matched lookalikes | yes, score distribution + per-position PWM contribution | test vs `motif_matched` vs the pool it is drawn from |
| anti-leakage and redundancy filters | yes, identity strips | same seed, filters off vs on |

This is the protocol's thesis (an unfairly drawn reference measures your sampling rather
than the biology) shown five separate ways, so again nothing is left asserted in prose
alone.

**§2 Full public-surface coverage: pass.** Nine demonstrated call sites, complete:
`AAWindowSampler` 7/7, `sample_same_protein` 17/17, `sample_different_protein` 16/16,
`sample_motif_matched` 14/14, `sample_synthetic` 8/8, `sample_benchmark_set` 3/3,
`SequencePreprocessor.get_aa_window` 7/7, `AALogo` 1/1, `get_df_logo` 7/7.

**§3 Narrative skeleton: pass.** Seven fields, in order, *Key mental model* first. Both
cross-references (*P1: CPP signature*, *P4: Prediction levels*) resolve to the right
protocols.

**§4 Presentation mechanics: pass.** All 13 `display_df` calls carry `n_rows` and
`show_shape=True`; every plot cell ends `plt.tight_layout()` then `plt.show()`; 5
committed figures; zero error outputs; zero null execution counts; kernelspec `python3`.

**§2b Source-verified API: pass, and this is the part worth recording.** P6's audit found
a protocol asserting that a shipped parameter did not exist, so every checkable claim in
P3 was re-verified against the running package rather than read for plausibility. All of
them hold:

- **Window geometry.** The protocol says a window starts `(window_size - 1) // 2`
  residues upstream of its P1 anchor, and both the `source_position` column and the
  gallery figure depend on it. Re-deriving each sampled window from its
  `source_position` with `get_aa_window` reproduces the sampler's own `window` string
  exactly, so the figure places its bars where the residues actually are.
- **`sample_motif_matched` scans only rows with no positives**, as claimed: the backend
  builds its candidate pool from the non-positive rows.
- **Synthetic rows** carry `entry=""`, `source_position=-1` and a per-call
  `entry_win="synth_{i}"`, which is what makes the notebook's dedupe warning correct.
- **Per-arm sub-seeds** of `sample_benchmark_set` are derived deterministically from the
  master seed via `numpy.random.SeedSequence`.
- **`aa.options["random_state"]`** is a real option key, not an invented one.
- **`AA_CASPASE3` is already windowed** (9-mers), the claim the *Input* section uses to
  justify building a toy full-sequence `df_seq` instead.
- The five sampler methods share one 8-column `segments` schema, and the pool arithmetic
  in *Common mistakes* (3 non-substrate proteins x 42 admissible 8-mers = 126) is right.

**One cosmetic inconsistency, left alone.** The same convention is written
`Schechter–Berger` (en dash) in *When to use it* and `Schechter-Berger` (hyphen) in the
motif figure's lead-in. Neither is an em dash, so no house rule is broken, and changing a
markdown cell would mean re-executing a notebook that has no defect. Recorded here rather
than fixed.

### Thumbnails, again

Not regenerated, for the reason established earlier in this document: re-execution alone
shifts tiles for environmental reasons, and refreshing 2 of 10 gallery tiles would buy a
visible inconsistency across the gallery in exchange for nothing. Neither notebook was
re-executed for content here at all. Both were run under `nbmake` to confirm they still
execute green, which validates them without touching the committed outputs.

## See also

- [`protocol_quality_rubric.md`](protocol_quality_rubric.md): the rubric applied here.
- [`protocol_style_guide.md`](protocol_style_guide.md): structure, voice, hard rules.
- `.claude/rules/notebooks.md`: the shared notebook presentation discipline.

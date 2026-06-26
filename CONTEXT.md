# aaanalysis

Python framework for interpretable, sequence-based protein prediction. This glossary captures domain terms used across the package, sharpened over time as conversations resolve ambiguities. Definitions are short and opinionated; aliases are explicitly listed so the canonical term wins on touch.

## Language

### Core sequence vocabulary

**df_seq**:
A pandas DataFrame whose rows describe protein sequences; must contain an `entry` column (unique protein identifier) and a `sequence` column (full amino-acid sequence). Method-specific columns (`pos column`, `aa_context_col`) layer on top of this base schema.
_Avoid_: sequence frame, protein df, input df.

**entry**:
A unique protein identifier; conventionally a UniProt accession.
_Avoid_: id, accession, protein_id.

**canonical amino acids**:
The 20 standard amino acids in alphabetical order, `ACDEFGHIKLMNPQRSTVWY`, exposed as `ut.LIST_CANONICAL_AA`.
_Avoid_: standard AAs, "the 20 AAs" without specifying order.

**pos column**:
A `df_seq` column whose cells hold per-row 1-based integer positions of interest (typically labeled positives); each cell is a `list[int]`, a single `int`, or empty (`None` / `NaN` / empty list / empty string). Default column name `"pos"`, referenced as `ut.COL_POS`. Two consumers share this column as **P1 anchors**: `AAWindowSampler` explodes it into sampled windows, and `SequenceFeature.get_df_parts` / `NumericalFeature.get_parts` (the **anchor input mode**) explode it into one 3-part row per anchor (TMD placed by `ut.get_window_offsets`, then the `jmd_n_len`/`jmd_c_len` extension), ided by **entry_win**.
_Avoid_: positions column (collides with `df_feat`'s `COL_POSITION`, which is a CSV string of feature positions).

**TMD coordinate convention**:
The single rule governing `tmd_start` / `tmd_stop` (`ut.COL_TMD_START` / `ut.COL_TMD_STOP`) in the **position-based** `df_seq` format: **1-based, start-inclusive, stop-inclusive** — matching standard biological annotation (e.g. UniProt), so an annotated TMD spanning residues `s..e` is stored verbatim as `tmd_start=s, tmd_stop=e` and its length is `tmd_stop - tmd_start + 1`. This convention is the *single source of truth*: every construction site that derives these columns (`_get_tmd_positions` from a `tmd` substring, the `seq_based` branch from `jmd_n_len`/`jmd_c_len`, `expand_pos_anchors_` from a P1 anchor) and every consumer that reads them (`Parts.get_tmd` slices `seq[tmd_start-1 : tmd_stop]`, `_slice_dict_num_to_basic_parts`) expresses the *same* convention, even though each uses different arithmetic. The convention is documented, not factored into one shared function, because the sites share a meaning, not a formula.
_Avoid_: 0-based, half-open / exclusive-stop, `len()`-style stop (a stop equal to `start + length` reads as exclusive — it is not).

**part**:
A named region of a protein over which a **split** operates and a scale is averaged; the `PART` field of a feature id (`PART-SPLIT-SCALE`). Parts are the columns of `df_parts`, produced by `SequenceFeature.get_df_parts`. The default vocabulary is **TMD-centric** — `jmd_n` / `tmd` / `jmd_c` (plus composites like `jmd_n_tmd_n`) — which fits **domain-level** tasks but is *semantically wrong* for other levels, so part naming should follow the **prediction level**:
- **Domain level:** replace the generic `tmd` with the **specific domain name** where known (e.g. the Pfam / InterPro domain), rather than the placeholder "tmd".
- **Residue level (cleavage / between-residues):** name positions by the **Schechter–Berger** convention — `… P2 · P1 │ P1′ · P2′ …` around the scissile bond (`│` = cleavage site; see **P1 anchor / source position**), not "tmd".
- **Protein level:** the whole chain is a single part; use a neutral name (e.g. `seq`, or N-term / core / C-term thirds), not "tmd".
First-class user-defined / renamed regions are tracked by **#27** (region abstraction); today a part is chosen from the predefined family.
_Avoid_: region (reserved for the #27 abstraction), domain (a part may be a window or sub-region, not a whole domain), segment (a split type).

**part label**:
The fixed, human-readable display string for a `part` token (e.g. `tmd` → "TMD", `jmd_n_tmd_n` → "JMD-N+TMD-N"), defined once in `ut.DICT_PART_LABEL` and used when rendering a feature id as prose. Deliberately *not* called a "region" (that noun is reserved for #27); a part label is purely cosmetic and changes no behavior.

**feature description**:
One standardized, human-readable sentence built deterministically from a `PART-SPLIT-SCALE` feature id by `SequenceFeature.get_feature_descriptions`: it joins the **part label**, the **split** rendered as a phrase (e.g. "segment 2 of 4"), the residue positions, and the scale's AAontology name / category / subcategory from `df_cat`. Additive only — the `feature` id string is unchanged — and optionally carried as the `feature_description` (`ut.COL_FEAT_DES`) column of `df_feat`. Distinct from the compact **feature name** (`get_feature_names`, `'subcategory [positions]'`), which drops the part and category.
_Avoid_: feature name (that is the shorter label form), scale description (that is the per-scale `scale_description` field, only one ingredient).

**assembled reference df_parts**:
A `df_parts` whose rows are **chimeras** of independent per-part windows — each part column is filled from its own window set (e.g. a separate `AAWindowSampler.sample_synthetic` call with its own generator and window size), rather than sliced from one contiguous protein. Built by `SequenceFeature.get_df_parts_from_windows` (windows aligned by position: the i-th window of every part forms the i-th row) and used as the **reference** class for `CPP`. Because a row stitches windows from different sources it has no single source `entry_win`; rows are ided by a per-row `REF{i}` index instead.
_Avoid_: synthetic df_parts (the windows need not be synthetic), reference window (that is the per-window term, not the assembled table).

**seq_kws**:
One protein's three part sequences (`jmd_n_seq`, `tmd_seq`, `jmd_c_seq`) bundled as a dict, ready to splat (`**seq_kws`) into the sample-level plot methods (`CPPPlot.profile` / `CPPPlot.feature_map`, e.g. SHAP explanations). Produced by `SequenceFeature.get_seq_kws(df_seq, df_parts, sample)`: the parts are taken from [[df_parts]] (the same parts that produced [[df_feat]] via `CPP.run`), so the displayed residues are **bound to the feature geometry** — there is no JMD-length argument (the lengths are read off `df_parts`; a JMD not encoded there comes back as an empty string). [[df_seq]] is cross-checked for consistency and raises on mismatch. **sample** selects the protein by **entry** (accession) or row position — never by **name**, since only `entry` is guaranteed unique in [[df_seq]] (`name` is a *display label*, not a selection key). Named `_kws` (not `args_seq`) to match the `split_kws` / `cbar_kws` forwarded-keyword-dict family.
_Avoid_: args_seq, seq_kwargs, part_seqs.

### Prediction-task taxonomy vocabulary

**prediction level**:
The biological unit a task predicts over, and the organizing backbone of the user-facing docs. Three levels, encoded in the `load_dataset` name-prefix scheme: **residue level** (`AA_*`), **domain level** (`DOM_*`), **protein level** (`SEQ_*`). The level is a convenient label for two deeper axes — the **unit of comparison** and **reference construction** — which actually determine the CPP setup. See ADR-0022.
_Avoid_: scale (reserved for AA physicochemical scales), granularity, task type (too generic).

**residue level**:
Per-residue / windowed prediction; datasets `AA_*`; the **unit of comparison** is a fixed-length **window** (`AAWindowSampler`). Two **sub-modes**: **single-residue** (odd `aa_window_size` — a site *on* a residue, e.g. a PTM) and **between-residues** (even window — a scissile bond P1│P1′, e.g. cleavage). Sub-modes, not separate levels: they differ only by window parity.
_Avoid_: position level, site level (ambiguous across the two sub-modes), residue-pair level (the "between" case is a sub-mode, not a level).

**domain level**:
Prediction over a defined sub-region of a protein; datasets `DOM_*` (e.g. `DOM_GSEC`); the **unit of comparison** is the **part** set derived from `tmd_start`/`tmd_stop` (`jmd_n` / `tmd` / `jmd_c`). CPP is native here.
_Avoid_: region level, segment level (segment is a split type).

**protein level**:
Whole-chain prediction; datasets `SEQ_*`; the whole sequence is the part. "Protein-level" is the **user-facing alias of the `SEQ_` prefix** (`SEQ_` = "sequence", not a third concept). Short peptides are the clean sub-case — the chain *is* the window.
_Avoid_: sequence level (use only when naming the `SEQ_` prefix spelling itself), global level.

**unit of comparison**:
The part CPP profiles for a task — a **window** (residue level), a **part** set (domain level), or the **whole chain** (protein level). One of the two axes that genuinely define a use-case class. See ADR-0022 (D3).
_Avoid_: granularity, scope.

**reference construction**:
How the contrasting set is built for a CPP / PU workflow — labeled A-vs-B groups, non-site / non-cleaved windows, an unlabeled pool, or a composition-matched shuffled background (#61 / #66). The second class-defining axis.
_Avoid_: negative set (too narrow — only one of the options).

**test group / reference group**:
The two sets CPP contrasts: the **test group** is profiled against the **reference group**, and a feature's mean difference (`mean_dif`) is computed as *test − reference* (`abs_auc` measures the separation magnitude, unsigned). The A-vs-B comparison at the literal heart of CPP — what **reference construction** produces and what **determinant discovery** interprets. At multi-class, each class is the test group in turn versus the rest as the reference group (**one-vs-rest (OvR)**).
_Avoid_: positive / negative set (conflates the comparison roles with PU label values 1/0); test/reference *set* (prefer "group" — "set" collides with reference construction's output).

**determinant discovery**:
A cross-cutting use-case class with **no prediction target**: CPP contrasts two groups to surface *what physicochemically distinguishes them*, interpreted via AAontology. CPP's purest, most interpretable use; applies at any **prediction level**.
_Avoid_: feature discovery (collides with feature engineering), profiling (too generic).

**design / engineering**:
A cross-cutting use-case class that inverts prediction: instead of asking *what distinguishes two groups*, it asks *how a mutation moves a sequence's CPP feature profile*, and uses that to move a sequence toward a target outcome ([[AAMut]] / [[SeqMut]]). Same prediction levels, opposite direction. The line is **scoring/measurement vs search/optimization**: [[SeqMut]] *scores* (apply mutations, measure [[ΔCPP]], rank by magnitude or — model-aware — by the [[prediction shift]] `delta_pred`, suggest the top target-shift, score explicit [[combined variant]]s); the *search/optimization* layer (greedy and multi-objective/Pareto directed evolution, active-learning selection) lives in the forthcoming **`SeqOpt`** class and is deferred (#59/#60). #57 made [[SeqMut]] optionally model-aware (opt-in `model`); the model-free [[ΔCPP]] is always available. [[AAMut]] stays deterministic and model-free.
_Avoid_: optimization (overloaded — reserved for the deferred chain), generation.

**relational / interaction (scope boundary)**:
Tasks about relationships *between* residues or chains (PPI interfaces, residue–residue contacts). AAanalysis profiles interface **segments** only; long-range pairwise contacts are **out of scope** and hand off to structure / PLM tooling. A boundary, **not** a fourth prediction level.
_Avoid_: pair level, contact level (implies first-class support that does not exist).

### Protein-design (mutation / ΔCPP) vocabulary

**AAMut**:
The **residue-level, CPP-agnostic** mutator (`aaanalysis/protein_design/`). `AAMut(df_scales).run(from_aa, to_aa, scales)` returns a tidy per-scale substitution-impact table — the signed `delta = scale[to_aa] - scale[from_aa]` for every requested substitution pair — independent of any sequence or task; `eval` ranks scales by mean `abs_delta` (substitution sensitivity). It is the physicochemical building block [[SeqMut]] uses per position. Distinct from a sequence-level mutator: AAMut never sees a [[df_seq]] or a [[df_feat]].
_Avoid_: substitution matrix (that is one *view* of AAMut's output, the `AAMutPlot.substitution_matrix` heatmap), BLOSUM (AAMut is property-scale-based, not log-odds).

**SeqMut**:
The **sequence-level, CPP-aware** mutator. Requires the **position-based** [[df_seq]] (`sequence` + `tmd_start`/`tmd_stop`) and a [[df_feat]]; applies point mutations and measures the [[ΔCPP]] they induce. Five verbs: `mutate` (apply a tidy `df_mut(entry, pos, to_aa)` table), `scan` (exhaustive per-position × substitution landscape, ranked by `delta_cpp`), `suggest` (top **target-shift** mutations), `eval` (**stable-vs-disruptive** per region), and `combine` (score a [[combined variant]] — several point mutations applied together). **Model-free by default; optionally model-aware**: constructing `SeqMut(model=...)` binds a fitted classifier so each mutation also carries its [[prediction shift]] (`delta_pred`) and `suggest` is guided by it. It *scores* mutations; search/optimization over them (greedy + multi-objective directed evolution) is the forthcoming `SeqOpt`. The returned table *is* the mutation history (no hidden state). It is the sole CPP-coupled class in the design module.
_Avoid_: sequence mutator (use SeqMut), perturbation (the legacy module name; the package is `protein_design`).

**ΔCPP** (`delta_cpp`):
The **feature-space** change a mutation induces: `ΔX = X_mut − X_wt` over the [[df_feat]] features (each `X` from `SequenceFeature.feature_matrix`), aggregated to the scalar L1 magnitude `delta_cpp = Σ|ΔX|`. The **model-free** measurement layer of [[design / engineering]], always available. Distinct from the [[prediction shift]] (`delta_pred`), the model-based score added when a `model` is bound. Only positions inside the parts referenced by `df_feat` can change `X`, so SeqMut's default scan region is the JMD-N + TMD + JMD-C span.
_Avoid_: mutation score, fitness (implies an optimization objective); do not conflate with the model-based `delta_pred` ([[prediction shift]]).

**target-shift** (`shift_score`):
The signed score by which [[SeqMut]]`.suggest` ranks a mutation toward the **test-class profile**: `shift_score = Σ sign(mean_dif) · ΔX`, optionally weighted per feature by `feat_importance`/`abs_auc`. The target direction is **implicit in `df_feat`'s `mean_dif`** (the way the test class differs from the reference), so no separate target vector is needed — though an explicit `target` feature-vector override is accepted. Positive = moves toward the test class.
_Avoid_: objective (reserved for the deferred multi-objective #59), gradient (no model is differentiated).

**stable vs disruptive**:
The binary tag [[SeqMut]]`.eval` assigns each scanned mutation by thresholding `|ΔCPP|` (default: the upper-tertile quantile of the observed distribution; user-overridable), summarized as a per-`entry`×`region` disruptive **rate**. "Disruptive" = large feature-space displacement; "stable" = small.
_Avoid_: deleterious / tolerated (phenotype claims AAanalysis does not make — the tag is feature-space displacement, not function).

**prediction shift** (`delta_pred`):
The **model-based** score a mutation induces, present only when a fitted classifier is bound to [[SeqMut]] (`SeqMut(model=...)`): `delta_pred = (P_target(mut) − P_target(wt)) · 100` — the change, in percentage points, of the model's predicted probability for the `target_class`. The ML-guided counterpart of the model-free [[ΔCPP]]; it is what `suggest` ranks by when a model is given and what the mutation-scan heatmap colors (and the fitness the forthcoming `SeqOpt` optimizes). The model is duck-typed on `predict_proba` ([[TreeModel]] returns the positive-class score *and* its std → the heatmap's "score ± std" title; a scikit-learn classifier returns the 2-D probability matrix, no std). "the change of prediction score per mutation and position."
_Avoid_: fitness, activity (phenotype claims — it is the model's score, not a measured function); ΔCPP (that is the model-free feature-space score).

**combined variant** ([[SeqMut]]`.combine`):
A set of point mutations applied **together** to one sequence and scored once (one [[ΔCPP]] / [[prediction shift]] per variant), labelled by the `'+'`-joined single mutations (e.g. `R20K+K27P`). The explicit-design move (the user names the variants); automated *search* over variants (greedy / multi-objective directed evolution) is the forthcoming `SeqOpt`. Contrast [[SeqMut]]`.mutate`, which scores each point mutation independently against the wild-type.
_Avoid_: library (a combined variant is one design, not an enumerated mutation library).

**epistasis**:
The pairwise **non-additivity** of two mutations, `ΔP(i+j) − (ΔP(i) + ΔP(j))`, visualized by `SeqMutPlot.epistasis` from a [[combined variant]] table of singles + pairs: positive = synergy (the pair beats the sum of singles), negative = antagonism.
_Avoid_: interaction (overloaded with the relational/PPI [[relational / interaction (scope boundary)]] sense); coupling.

### SeqOpt directed-evolution vocabulary

**SeqOpt**:
The **search/optimization** layer of [[design / engineering]] (**[pro]**, `aaanalysis/protein_design_pro/`), the counterpart to [[SeqMut]]'s scoring: it *searches* over sequence variants of **one wild-type** for those that best satisfy several objectives at once. This is **protein engineering** — machine-learning-guided **directed evolution** of an *existing* sequence (Yang et al. 2019; Wittmann et al. 2021) — explicitly **not** *de novo protein design* (building new proteins from the ground up, e.g. RFdiffusion→ProteinMPNN→AlphaFold; Yang et al. 2026), which is out of scope. A `Tool` (`run` → [[Pareto front]] `df_pareto`, `eval` → Pareto-quality metrics), paired with `SeqOptPlot`. Reuses model-bound [[SeqMut]] as the fitness engine and `ShapModel` for residue guidance, so it is `pro` (imports SHAP) even though [[SeqMut]] stays core. Two modes — `"impact"` (SHAP-guided, adaptive) and `"importance"` (feature-importance-guided, greedy) — see [[guidance mode (impact / importance)]]. It realizes the search that [[SeqMut]] and [[combined variant]] defer to.
_Avoid_: optimizer (overloaded with numerical/perf optimization — this is evolutionary *search*); generator, sampler; **de novo design / protein design** (SeqOpt does protein *engineering* / directed evolution, not generation of new proteins).

**population** / **generation**:
A **population** is the set of candidate **variants** (each a mutation-set on the one wild-type, carrying its [[prediction shift]] fitness and a per-residue importance map) that [[SeqOpt]] evolves; a **generation** (`generation`) is one evolve-score-select round. Standard NSGA-II vocabulary, claimed here from the reservation the [[design / engineering]] entry held for it.
_Avoid_: library (an enumerated set, not an evolving population); cohort, batch.

**Pareto front** (`df_pareto`):
The **non-dominated** set [[SeqOpt]]`.run` returns: variants where no other returned variant is at least as good on every objective and strictly better on one. The trade-off surface (e.g. high [[prediction shift]] vs. few mutations), not a single greedy path. Rows carry one column per objective plus the [[non-dominated rank]] and [[crowding distance]].
_Avoid_: best variants (the front is a *set* of trade-offs, not a ranked winner list); optimum.

**non-dominated rank** (`rank`) / **crowding distance** (`crowding`):
NSGA-II's two-level ordering. **Non-dominated rank** is the front index from fast non-dominated sorting (`rank=0` is the first/best front); **crowding distance** is the objective-space spread around a variant within its front (larger = more isolated = preferred), the diversity tie-breaker. Together they define survival and mating selection. Reimplemented pure-Python; DEAP is the dev-only parity oracle (identical front membership + ordering on a seed).
_Avoid_: dominance score (rank is a front *index*, not a score); density.

**hypervolume**:
The objective-space volume dominated by the [[Pareto front]] relative to a reference point — the scalar [[SeqOpt]]`.eval` reports for front quality and the per-generation convergence trace (`SeqOptPlot.hypervolume`). Larger = a better-spread, further-advanced front; non-decreasing across generations on a fixed seed.
_Avoid_: coverage, spread (a distinct diversity metric `eval` also reports).

**guidance mode (impact / importance)**:
How [[SeqOpt]] decides **which residues to mutate**, named after the `df_feat` attribution column it reads. `mode="impact"` refits `ShapModel` **every generation** under [[fuzzy labeling]] to get fresh per-residue `|feat_impact|` (adaptive, the headline). `mode="importance"` reads static `feat_importance` (TreeModel, no SHAP, no refit) and walks positions highest-first (deterministic, cheap). The guidance prunes the otherwise-combinatorial mutation-set space.
_Avoid_: strategy, policy.

**fuzzy labeling** (in [[SeqOpt]]):
How generated variants — which have no ground-truth label — enter the per-generation `ShapModel` refit: each variant's own model **prediction score** in `[0, 1]` becomes its **soft label** (`ShapModel.fit(fuzzy_labeling=True)`), explained against the original balanced 0/1 reference set. This is what lets SHAP attribution track the moving [[population]] (the `mode="impact"` engine). The shipped `ShapModel` fuzzy-labeling path, applied to directed evolution.
_Avoid_: pseudo-labeling (no hard threshold is assigned — the label *is* the continuous score), self-training.

**evolutionary operators** (in [[SeqOpt]]):
The DEAP-mapped algorithm families [[SeqOpt]] re-implements in **pure Python** (DEAP is a dev/test-only parity oracle, never a runtime dependency): **crossover** (uniform / one-point / two-point over mutation-sets), **mutation** (substitution / shift), **variation** (`varAnd` = crossover *and* mutation, `varOr` = one of crossover / mutation / reproduction), **survival** (`mu_plus_lambda` elitist / `mu_comma_lambda` / `ea_simple` generational), **constraints** (feasibility callables penalized DeltaPenalty- or ClosestValidPenalty-style), and the single-objective **Hall of Fame** (`SeqOpt.hall_of_fame_`) alongside the [[Pareto front]] archive. Counterpart to the multi-objective [[non-dominated rank]] / [[crowding distance]] selection core.
_Avoid_: GA primitive (these are the EA layer); DEAP operator (ours is an independent re-implementation).

**NSGA-II kernel / DEAP parity** (in [[SeqOpt]]):
The non-dominated sort + crowding formula (`nobj·span` normalization, matching DEAP's `assignCrowdingDist`) is a single pure-Python implementation with a **memory-bounded chunked-vectorized** dominance scan (an earlier `engine="exact"/"fast"` split was collapsed once both were shown to give byte-identical fronts). On a fixed seed it reproduces the **DEAP reference**'s [[non-dominated rank]] + [[crowding distance]] ordering (DEAP is a dev/test-only oracle; runtime stays DEAP-free), and the comparison benchmark shows it is faster than DEAP.
_Avoid_: byte-exact (the bar is equivalence — identical front membership + ordering, values within tolerance — not byte-identical serialization); engine (the `exact`/`fast` knob was removed).

**convergence** (`SeqOpt.eval`):
The optional generational-distance metric: the mean range-normalized distance from each [[Pareto front]] point to its nearest point on a user-supplied **reference front** (`ref_front`); lower = closer to the target. Joins `hypervolume` and `spread` in `df_eval` only when a reference is given.
_Avoid_: accuracy (this is a set-to-set distance, not a classification score).

### Multi-class & regression labeling vocabulary

Helpers on `SequenceFeature` that turn a multi-class or continuous target into the
binary `labels` CPP consumes — the operational layer of **reference construction**
for non-binary targets. The row-dropping members (OvO, tiered) take the value source
(`df_parts` and/or `dict_num_parts`) and return per group the **row-matched copy** plus
the binary labels, ready to drop into `CPP.run` / `CPP.run_num`; the selection is applied
internally (no mask is exposed).

**one-vs-rest (OvR)**:
A multi-class decomposition where each of the K classes in turn is the **test group**
and all remaining classes form the **reference group**. No samples are dropped, so the
K binary arrays share one `df_parts` and loop through a single `CPP` instance.
`SequenceFeature.get_labels_ovr`.
_Avoid_: one-vs-all (use OvR consistently), binarization (too generic).

**one-vs-one (OvO)**:
A multi-class decomposition over each unordered class **pair** `(a, b)`: the other
classes are discarded, so each pair returns its own row-matched `df_parts` / `dict_num_parts`
subset plus a binary array (`a`=test, `b`=ref). `SequenceFeature.get_labels_ovo`.
_Avoid_: pairwise comparison (collides with the relational scope boundary), one-vs-each.

**quantile labels**:
A continuous target discretized into binary labels by a single **quantile cut** — at or
above the q-quantile is the test group, below is the reference group.
`SequenceFeature.get_labels_quantile`.
_Avoid_: threshold labels (ambiguous about which threshold), binning (implies >2 bins).

**tiered labels**:
A continuous target turned into several binary **tiers** sharing one fixed positive set
(≥ the `q_pos` quantile) paired with progressively lower negative cuts (≤ each `q_neg`),
the middle band dropped; returns the row-matched `df_parts` / `dict_num_parts` subset plus
labels per tier. Lets a regression target be profiled at increasing positive-vs-reference
separation. `SequenceFeature.get_labels_tiered`.
_Avoid_: banded regression (reserved for the deferred banded-regression task), graded labels.

**label parameter names (canonical)**:
Four *distinct* labeling concepts, each with one canonical parameter name — similar
spellings name different things, so they are kept separate rather than unified:
- **`label_test` / `label_ref`** — the two groups of a **contrast** (positive/test vs
  reference): `CPP.run`/`eval`, `AAlogo.get_df_logo`.
- **`labels`** — a **single** 1D per-sample class-label vector `(n_samples,)`:
  `CPP.run`, `TreeModel.fit`/`eval`, `ShapModel.fit`, `dPULearn.fit`.
- **`list_labels`** — a **2D list of labelings** `(n_datasets, n_samples)` with
  `names_datasets`: `AAclust.eval`, `dPULearn.eval`. Plural on purpose.
- **`label_target_class`** — a **single target class to attribute** in SHAP
  (`ShapModel.fit`); can be *any* class (positive, negative, or a multi-class index),
  so it is **not** `label_test` and keeps its own name.
_Avoid_: renaming `label_target_class` to `label_test` (conflates a general target-class
selector with the test-vs-reference contrast); collapsing `list_labels` into `labels`.

### Window sampling vocabulary

**window**:
A fixed-length contiguous slice of a protein sequence, anchored at a 1-based P1 position.
_Avoid_: subsequence, fragment, region.

**P1 anchor / source position**:
The 1-based residue immediately N-terminal to a notional scissile bond (Schechter–Berger cleavage convention); the anchor coordinate for a window. For window length L, the window covers `(L-1)//2` residues upstream of the anchor, the anchor itself, and `L//2` residues downstream — right-heavy for even L.
_Avoid_: center, midpoint (ambiguous for even window sizes).

**test window**:
A window extracted at a position listed in the **pos column**; treated as a known-positive reference for the identity filter.
_Avoid_: positive window, target window.

**reference window**:
The umbrella term for any sampled window that is not a test window — covers `Negative`, `Unlabeled`, and `Control` rows.
_Avoid_: negative window (too narrow — only one of the three roles), drawn window.

**control window**:
A reference window produced by `sample_synthetic`; carries `entry=""` and has no source protein coordinates.
_Avoid_: simulated window, fake window.

**role**:
A categorical tag stored in the output's `role` column describing what a sampled row represents in a workflow. Standard values: `Test`, `Negative`, `Unlabeled`, `Control`. Users may pass custom strings. Defaults are opinionated and assume PU-learning / hard-negative-mining workflows.
_Avoid_: class, label_kind, category.

**strategy**:
A tag stored in the output's `strategy` column identifying which sampling method produced the row. Values: `same_protein`, `different_protein`, `motif_matched`, `synthetic:<generator>`. `strategy` (which method) plus `arm` (which named benchmark configuration) together carry full row provenance, so no separate `provenance` column is introduced.
_Avoid_: method, source, origin, provenance (subsumed by `strategy` + `arm`).

**entry_win**:
A row's window identifier, formatted `<entry>_<start_pos>-<end_pos>` with 1-based inclusive coordinates for protein-sourced windows, or `synth_{i}` (per-call counter) for synthetic windows. Identical biological windows across calls share the same `entry_win`, making `drop_duplicates(subset="entry_win")` the natural dedupe primitive — except for synthetic outputs, where the per-call counter is not call-stable.
_Avoid_: window_id, row_id.

**candidate pool**:
The set of eligible windows from which a sampling method draws, defined per strategy. `same_protein`: positions whose distance to the nearest positive on the same protein lies in the `(min_distance_to_pos, max_distance_to_pos)` **distance band**. `different_protein` / `motif_matched`: any window on a protein with no listed positives. `synthetic`: drawn fresh from the generator distribution.
_Avoid_: candidates, eligible set.

**distance band** (`min_distance_to_pos`, `max_distance_to_pos`):
A pair of optional residue-distance bounds used by `sample_same_protein` to filter candidate P1 anchors by their L1 distance to the *nearest* positive on the same protein. `min_distance_to_pos` is the lower bound (or `None` for no lower bound); `max_distance_to_pos` is the upper bound (or `None` for no upper bound). Both default to `None`, in which case every fully-fitting window on a positive-containing protein is admissible — so sampled "Negative" windows may overlap positive windows. For non-overlapping hard-negatives, set `min_distance_to_pos=window_size`; for windows targeted near positives, pair with a finite `max_distance_to_pos`.
_Avoid_: distance-to-positive (singular — misses the band).

**custom filter**:
A user-supplied predicate `(window: str, entry: str, source_position: int) -> bool` (keep the window when it returns `True`) set on `AAWindowSampler.__init__` alongside the similarity filters, so it composes in the per-window filter pipeline of every `sample_*` method automatically. It is the sanctioned escape hatch for structure-, domain-, or dataset-specific decoy rules that the built-in filters do not cover (the deferred `match_structure` filter is implemented as a custom-filter recipe). For synthetic windows there is no source protein, so it is called with `entry=""` and `source_position=-1` (composition-only context).
_Avoid_: user filter, callback (too generic), post-filter (it runs inside the iterative re-draw, not after).

**benchmark set**:
A single concatenated `segments`-mode DataFrame produced by `AAWindowSampler.sample_benchmark_set(df_seq, arms, seed)`, stacking one or more named **arms** with an extra `arm` column. Multi-arm orchestration only — it adds no new sampling behavior; each arm is one ordinary `sample_*` call. No automatic cross-arm dedupe (every row preserved): dedupe protein-sourced rows on `entry_win` and synthetic rows on `window` if needed.
_Avoid_: benchmark, sample set, arm set.

**arm**:
One named sampling configuration inside a **benchmark set**: a dict `{"method": <strategy-tag>, **kwargs}` where the method is one of the four strategy tags (`same_protein`, `different_protein`, `synthetic`, `motif_matched`) and the remaining keys forward to that `sample_*` method. The arm's name is written into the output's `arm` column; reproducibility comes from a per-arm sub-seed derived deterministically from the call's `seed` (`np.random.SeedSequence`).
_Avoid_: branch, group, configuration (overloaded).

### Synthetic generation vocabulary

**generator**:
The recipe by which `sample_synthetic` produces a window — a string (built-in mode like `"uniform"` / `"global_freq"` / `"position_specific"` / `"scrambled"`, or an AAontology preset name), a list of preset names (multiplicative mix), or a `dict[str, float]` (custom-alphabet frequency table).
_Avoid_: mode (historical name; renamed for clarity), prior (inaccurate for `scrambled` and `position_specific`).

**AAontology preset**:
A named generator backed by a curated AAontology scale loaded via `aa.load_scales` and sum-normalized into a probability distribution over the 20 canonical AAs. Composition presets are true AA-frequency distributions; conformation presets are normalized propensities used as physicochemically-biased priors.
_Avoid_: scale-based mode.

**custom-alphabet generator**:
A `dict[str, float]` generator mapping single-character symbols to non-negative probabilities summing to 1; the only sampling path that produces non-amino-acid windows.
_Avoid_: custom freq, custom dist (too generic).

### Scoring vocabulary

**PWM (position-weight matrix)**:
A `pd.DataFrame` of shape `(window_size, 20)` representing per-position residue scores over the 20 canonical AAs. Columns are the canonical AA letters in any order and are reindexed internally to `ut.LIST_CANONICAL_AA` (alphabetical, `ACDEFGHIKLMNPQRSTVWY`). `np.ndarray` PWMs are rejected — wrap with `pd.DataFrame(arr, columns=ut.LIST_CANONICAL_AA)` if you only have an array.
_Avoid_: scoring matrix (too generic), motif matrix.

**motif filter**:
The pair `(motif_pwm, motif_score_threshold)` used by `sample_same_protein` and `sample_different_protein` to keep (`motif_match="in"`) or drop (`motif_match="out"`) windows whose PWM score crosses the threshold. Optional on these two methods; required and `"in"`-only on `sample_motif_matched` (where it defines the candidate pool, not an overlay filter).
_Avoid_: motif gate, PWM filter.

**identity filter**:
A pair of filters based on per-position residue identity between fixed-length, aligned windows. `max_similarity_to_test` drops sampled windows too similar to any test window (anti-leakage); `max_similarity_within_ref` drops sampled windows too similar to a previously kept sampled window (redundancy reduction).
_Avoid_: similarity filter (overloaded), redundancy filter (too narrow — covers only the second).

### Embedding-based feature engineering vocabulary

**EmbeddingPreprocessor**:
Public core class in `aaanalysis/data_handling/` for protein-language-model (PLM) embeddings, instance-based (`ep = EmbeddingPreprocessor()`). Its primary method `encode` turns raw per-residue embeddings into a `[0, 1]`-normalized [[dict_num]] for `CPP.run_num`; the secondary `build_scales` / `build_cat` pair derives [[pseudo-scale]]s / [[pseudo-category]]s for the scale-based `CPP.run` path. Mirrors the family shape of `StructurePreprocessor` / `AnnotationPreprocessor` (all three: `encode*` → dict_num, then `build_scales` / `build_cat`).
_Avoid_: PLMPreprocessor, ESMPreprocessor (too narrow).

**encode** (embedding):
`EmbeddingPreprocessor.encode(df_seq, embeddings, method='minmax'|'quantile'|'sigmoid', return_df=False) → dict_num`. Fits one per-embedding-dimension normalizer over the whole corpus (all residues of all proteins) and maps every entry's `(L, D)` tensor into `[0, 1]`, the range `CPP.run_num` expects. AAanalysis does **not** run the PLM — the user supplies `embeddings` as `{entry: (L, D)}`. The fitted parameters are kept on the instance as `norm_params_`. This is the sanctioned raw-embeddings → dict_num step (raw PLM values are unbounded and not directly usable by `run_num`).
_Avoid_: normalize_embeddings, to_dict_num.

**pseudo-scale**:
A (20,)-shaped vector representing one PLM embedding dimension's per-AA average, computed by context-free averaging of the dimension's per-residue values over occurrences of each canonical AA in a reference corpus (typically the user's `df_seq`). Dataset-dependent — pseudo-scales for the same PLM differ across input corpora. Used only to derive pseudo-categories and to name dimensions in `df_scales`; never used as a residue-value source for feature aggregation in `CPP.run_num` (the per-residue [[dict_num]] tensor is consumed directly when supplied).
_Avoid_: dimension scale, AA average, embedding scale.

**pseudo-category**:
A cluster label assigned to a pseudo-scale by AAclust correlation-based clustering. Carried in `df_cat`'s `cat` (coarser threshold) and `subcat` (finer threshold) columns, mirroring the AAontology two-level hierarchy. Cluster IDs are deterministic given `(pseudo_scales, thresholds, random_state)` but inherit the dataset-dependence of pseudo-scales.
_Avoid_: PLM cluster, embedding group.

### Structure-based feature engineering vocabulary

**dict_dssp**:
A `Dict[entry, np.ndarray (L, D_dssp)]` of per-residue DSSP-derived numerical features (secondary structure one-hot, ASA, dihedrals). Produced by `StructurePreprocessor().encode_dssp(df_seq, pdb_folder, features=[...])`.
_Avoid_: ss_dict, dssp_tensor.

**dict_pdb**:
A `Dict[entry, np.ndarray (L, D_pdb)]` of per-residue features extracted directly from PDB ATOM records (mean B-factor, residue depth). Produced by `StructurePreprocessor().encode_pdb(df_seq, pdb_folder, features=[...])`.
_Avoid_: pdb_tensor, raw_pdb_dict.

**feature key**:
A canonical string identifier in the `StructurePreprocessor` registry that maps to a fixed `(num_dims, dim_names, category, subcategory, normalization recipe)` tuple. Used in the `features=[...]` parameter of `encode_dssp` / `encode_pdb` / `encode_pae` and in `build_scales(features=[...])` / `build_cat(features=[...])`. Feature-set **rev 1.1** keys (the StructurePreprocessor's own feature-registry revision — *not* the package version; the class as a whole debuts in package v1.1.0): `ss3`, `ss8`, `rasa`, `phi_psi_sincos`, `bfactor`, `depth`, `hse` (plus the rev-1.1 AlphaFold additions `plddt`, `plddt_disorder`, `plddt_tier`, `chi1_sincos`, `chi2_sincos`, `ca_centroid_dist`, `ca_centroid_dist_norm`, `contact_count_8A`, `contact_count_12A`, `pae_row_mean`, `pae_row_min`, `pae_row_max`, `pae_local_mean`, `pae_distal_mean`, `pae_asymmetry`, `pae_band_means`). Rev 1's `asa` (absolute) and `phi_psi` (raw degrees) are **removed** in rev 1.1.
_Avoid_: feature_id (collides with the `df_feat.feature` column), dim_key.

**StructurePreprocessor**:
Public class in `aaanalysis/data_handling_pro/` that converts PDB / CIF / AlphaFold files (and AF PAE sidecars) into [[dict_num]]-shape per-residue numerical tensors for `CPP.run_num`. Mirrors `EmbeddingPreprocessor`'s instance-based pattern (`stp = StructurePreprocessor()`). Nine public methods: `fetch_alphafold` (bulk download of AF-DB model + PAE files into a folder — see [[fetch_alphafold]]), `get_dssp` (raw DSSP list output), `encode_dssp` (DSSP → dict_num), `encode_pdb` (raw PDB → dict_num, includes AF model-file features), `encode_pae` (AF PAE sidecar → dict_num), `get_domains` (raw domain segmentation via afragmenter/chainsaw), `encode_domains` (domain files → dict_num), `build_scales` (corpus-derived per-AA-mean df_scales), `build_cat` (corpus-free df_cat metadata). The four `encode_*` methods return a **bare `dict_num` by default**; pass `return_df=True` for the `(dict_num, df_seq_out)` form, where `df_seq_out` echoes `df_seq` with a per-row `*_ok` status column. All encoder outputs are normalized to `[0, 1]` per the registry's `NORMALIZATION_RECIPES`; the inverse formulas are documented in the class docstring. `verbose` is constructor-only (no per-call override — see [[preprocessor verb taxonomy]]). Pro-extra gated (biopython; `requests` for `fetch_alphafold`); `msms` is a runtime check inside `encode_pdb(features=['depth'])`.
_Avoid_: PDBPreprocessor, DSSPPreprocessor (too narrow).

**fetch_alphafold**:
`StructurePreprocessor().fetch_alphafold(df_seq, out_folder, file_format='pdb'|'cif', timeout=30.0, skip_existing=True, on_failure='nan', return_df=False)`. Bulk-downloads each `entry`'s AlphaFold-DB model file (`AF-<entry>-F1-model_v4.{pdb,cif}`) and PAE sidecar (`AF-<entry>-F1-predicted_aligned_error_v4.json`) from `https://alphafold.ebi.ac.uk`, saving them under the filenames the `resolve_structure_path` / `resolve_pae_path` resolvers already find — so one call populates the `pdb_folder` / `pae_folder` that `encode_pdb` / `encode_pae` / `get_dssp` consume, with no glue. Returns a per-entry status DataFrame (`entry, model_ok, pae_ok, alphafold_ok, skipped, model_path, pae_path`); `return_df=True` also appends an `alphafold_ok` column to `df_seq`. A 404 (accession not in AF-DB, or fragmented `F2+`-only proteins) is the soft failure governed by `on_failure`; other network errors (timeout, 5xx) raise `RuntimeError` and abort. It is the `fetch_` (web) acquisition verb — the structure-side analog of `AnnotationPreprocessor.fetch_uniprot`. See ADR-0017.
_Avoid_: download_alphafold, get_alphafold (web acquisition is `fetch_`, not `get_`), fetch_pae (PAE ships with the model, not separately).

**preprocessor verb taxonomy**:
The shared method-naming rule across the Family-B preprocessors (`Embedding` / `Structure` / `Annotation`), recorded in ADR-0017. **Acquisition** verbs do I/O and yield a raw artifact: `fetch_*` pulls from a **web** resource (`fetch_uniprot`, `fetch_alphafold`), `get_*` runs a **local** tool / reads local files (`get_dssp`, `get_domains`). An acquisition getter exists **only where the raw output is an independently-useful, inspectable / curatable artifact** (a DSSP list, a domain segmentation, a `df_annot`) — not for a thin numeric read (`encode_pdb`'s ATOM-field extraction and `encode_pae`'s matrix collapse have no `get_` twin by design, not omission). **Transform** verbs are pure (no I/O): `encode` / `encode_*` map a source into a `[0, 1]` `dict_num`, with **one `encode_*` per distinct raw source** (`Structure` has four file sources → `encode_dssp` / `encode_pdb` / `encode_pae` / `encode_domains`) and a **bare `encode`** when the class has a single input (`Embedding`'s `embeddings` dict) or a single canonical intermediate (`Annotation`'s `df_annot`); `build_scales` / `build_cat` derive the secondary AA-scale-path metadata. `ingest` (user table → `df_annot`), `register_feature` (open-vocabulary registration), and `to_df_seq` (annotation → `AAWindowSampler` anchors) are `Annotation`-only by design. `verbose` is **constructor-only** family-wide (no per-call override on any method).
_Avoid_: acquire/load/read as method prefixes (use `get_`/`fetch_`); collapsing the four `encode_*` into one `encode(source=...)` dispatcher.

**combine_dict_nums**:
Top-level `aa.combine_dict_nums(dict_nums: List[Dict[entry, ndarray]]) → Dict[entry, ndarray]` that concatenates multiple per-residue tensors along the D axis. Source-agnostic — works with `dict_dssp`, `dict_pdb`, [[dict_pae]], `dict_embeddings`, or any user-supplied dict matching the shape contract. Validates same entry set + same L per entry across all inputs.
_Avoid_: merge_dict_num, stack_dict_nums.

**dict_pae**:
A `Dict[entry, np.ndarray (L, D_pae)]` of per-residue summaries derived from an AlphaFold PAE sidecar (`AF-{uniprot}-F1-predicted_aligned_error_v4.json`). Produced by `StructurePreprocessor().encode_pae(df_seq, pae_folder, features=[...])`. The L×L matrix is collapsed to per-residue summaries — row-mean / row-min / row-max / local-mean (±`local_window`) / distal-mean / asymmetry / band-means. All values normalized to `[0, 1]` by dividing by AlphaFold's PAE saturation cap (31.75 Å).
_Avoid_: pae_dict, dict_alignment_error.

**Feature-category colors** (locked v1.1 palette; source in `ut.DICT_COLOR_CAT`):
- `Structure` → `#2E6E5E` (deep teal-green) — all `StructurePreprocessor` outputs (DSSP / PDB / PAE / AF features).
- `Embeddings` → `#6B4FB5` (indigo-violet) — all `EmbeddingPreprocessor` outputs.
- `PTMs` → `#B36BCB` (lilac-magenta) — closed-vocabulary UniProt PTM/Processing outputs of `AnnotationPreprocessor`.
- `Functional sites` → `#2C6E9E` (deep ocean-blue) — open-vocabulary functional-site outputs of `AnnotationPreprocessor` (BINDING/ACT_SITE/DNA_BIND seeds + user/predictor keys). Deliberately not `#6B4FB5` (that is `Embeddings`).
- Plus the 8 AAontology categories (`ASA/Volume`, `Composition`, `Conformation`, `Energy`, `Others`, `Polarity`, `Shape`, `Structure-Activity`).
The redundancy filter's `check_cat=True` arm groups features by these top-level buckets; fine-grained semantic splits (e.g. `'Secondary structure (3-state)'` vs `'B-factor (CA mean)'` vs `'AlphaFold pLDDT (raw)'`) live in `subcategory` — these follow the AAontology naming convention (descriptive name with source / detail in parentheses) so the `CPPPlot.feature_map` y-axis reads cleanly.

**Color palette kind** (`plot_get_clist(kind=...)`):
The palette family a color list is drawn from, following Matplotlib's colormap taxonomy. `categorical` = *qualitative*, maximally distinct colors for discrete classes (curated for 2–9, `'husl'` for 10–20, capped at 20). `continuous` = `n_colors` sampled from any named palette via `seaborn.color_palette` — an ordered colormap (`'viridis'`) gives a perceptual ramp encoding magnitude, a qualitative one (`'husl'`, the default) gives distinct hues. `diverging` = two hues from a neutral center for signed/centered data (the house `'CPP'`/`'SHAP'` maps or any Matplotlib diverging map). The companion `plot_get_cmap` returns the pre-sized (101-point) diverging `CPP`/`SHAP` map. `husl` is a qualitative large-N hue generator, **not** an ordered/sequential ramp.
_Avoid_: calling the `'husl'` rainbow "continuous", calling `'viridis'` "diverging".

### Annotation-based feature engineering vocabulary

**AnnotationPreprocessor**:
Public pro-extra class in `aaanalysis/data_handling_pro/` that fetches per-residue PTM / functional-site annotations from UniProt (or ingests user/predictor labels), maps them into the canonical [[df_annot]] schema, and encodes them into `[0, 1]`-normalized per-residue [[dict_num]] tensors for `CPP.run_num`. Mirrors `StructurePreprocessor`'s instance-based pattern (`ap = AnnotationPreprocessor()`): `fetch_uniprot` (UniProt JSON → df_annot), `ingest` (user table → df_annot, open-vocabulary auto-register), `register_feature` (explicit open-vocabulary registration / override), `encode` (df_annot → dict_num; bare `dict_num` by default, `return_df=True` adds a `(dict_num, df_seq_out)` status echo with an `encode_ok` column — mirroring `StructurePreprocessor.encode_*`), `build_scales` (corpus per-AA-mean df_scales), `build_cat` (corpus-free df_cat), and `to_df_seq` (df_annot → df_seq with a `pos` column + an `aa_context` eligibility mask for residue-type-matched `AAWindowSampler` negatives — the seq-mode window-split path). Pro-gated (`requests`).
_Avoid_: PTMPreprocessor (too narrow — also handles functional sites), UniProtPreprocessor (also ingests non-UniProt user labels).

**df_annot**:
The canonical per-residue annotation schema (one row per annotated residue): `protein_id, start, end, aa, feature_type, category, source, evidence, score, bond_id` (columns in `ut.COLS_ANNOT`). Positions are 1-based in the **UniProt-canonical frame**; every mapped row is single-residue (`start == end`). `aa` is the expected residue identity, checked against the target `df_seq[sequence]` at encode time (`on_mismatch='raise'` by default — the off-by-isoform guard). `score` is a nullable float in `[0, 1]` (presence = `1.0`; predictor confidence otherwise). `bond_id` pairs the two endpoints of a disulfide / cross-link.
_Avoid_: annotation table, df_ptm (too narrow), df_sites.

**feature_type**:
The registry key naming one annotation channel (one `dict_num` dimension), e.g. `phospho`, `glyco_n`, `disulfide`, `signal_cleavage` (category `PTMs`) or `binding`, `act_site`, `dna_bind`, or a user key like `hotspot` (category `Functional sites`). Closed PTM vocabulary + functional seeds are built-in; user `feature_type`s auto-register as `Functional sites` at `ingest`.
_Avoid_: ptm_type, label, annotation_type.

**PTM (broad)**:
The `PTMs` category in the broad UniProt "PTM/Processing" sense — modified residues, glycosylation, lipidation, disulfide, cross-link, **and** the signal/propeptide/transit/SITE cleavage sources. A closed vocabulary, one source (UniProt). Disulfide stays in PTMs.
_Avoid_: post-translational-modification-only reading (here it includes cleavage/processing).

**Functional sites**:
The `Functional sites` category — an **open** vocabulary. UniProt BINDING/ACT_SITE/DNA_BIND ship as built-in seeds; user/predictor per-residue labels (RFdiffusion hotspots, BindCraft interface residues, custom) plug in as the extensibility point.
_Avoid_: active sites (too narrow — one seed of many), binding sites.

**evidence allow-set**:
The ECO-code filter applied by `fetch_uniprot(evidence=...)`. `'manual'` (default) keeps `ECO:0000269` (experimental) **and** `ECO:0007744` (combinatorial, manual); `'experimental'` keeps only `ECO:0000269`; `'all'` disables filtering. By-similarity (`ECO:0000250`) is never in the default positives. The raw ECO code is retained in the `evidence` column regardless.
_Avoid_: experimental-only (the default is broader), confidence (overloaded with `score`).

### CPP split vocabulary

**split**:
A rule that selects a subset of residue positions within a sequence **part**
(`jmd_n` / `tmd` / `jmd_c`), over which a scale's per-residue values are
averaged to produce one feature value. A feature ID is `PART-SPLIT-SCALE`
(e.g. `TMD-Segment(2,4)-ANDN920101`). Splits are residue-content-agnostic —
they map a part length to position indices, independent of the actual amino
acids.
_Avoid_: window (reserved for `AAWindowSampler`), segment (only one split type).

**df_feat (canonical column schema)**:
The standardized output of `CPP.run` / `CPP.run_num` (and `SequenceFeature.get_df_feat`):
one row per feature, columns in a **deterministic canonical order** (`ut.LIST_COLS_FEAT`):
`feature` (the `PART-SPLIT-SCALE` id), the four scale columns (`category`, `subcategory`,
`scale_name`, `scale_description`), the five stat columns (`abs_auc`, `abs_mean_dif`,
`mean_dif`, `std_test`, `std_ref`), the **dynamically-named** p-value column
(`p_val_mann_whitney` by default, `p_val_ttest_indep` when `parametric=True`),
`p_val_fdr_bh`, and `positions`. The order is a **lower bound, not a restriction**:
`ut.sort_cols_feat` appends any other column after `positions` in stable order and never
drops one, so the post-hoc explainable-AI columns (`feat_importance`, `feat_impact`) and the
per-substrate SHAP columns (`feat_impact_'name'`, `mean_dif_'name'`) survive unchanged. The
`feature` id stays an opaque string — parse it only with `ut.split_feat_id` (format with
`ut.join_feat_id`), never an ad-hoc `str.split("-")`.
_Avoid_: feature_id (the column is `feature`), region (the `PART` field is a **part**).

**split type**:
One of three split families, configured per-type via **split_kws**:
`Segment(i_th, n_split)` (the i-th of `n_split` contiguous chunks),
`Pattern(terminus, positions)` (fixed offsets from a terminus, bounded by
`len_max`), and `PeriodicPattern(terminus, step1/step2, start)` (alternating
periodic offsets). Exposed as `ut.LIST_SPLIT_TYPES`. The label generators
(`SplitRange.labels_*`) are **part-length independent** — they depend only on
`split_kws`, not on any part's length.
_Avoid_: split mode, split kind.

**compositional vs positional (CPP strategy)**:
The two ways `split_kws` resolves a feature's locality. **Compositional** uses a single whole-part average — `Segment(1,1)`, obtained via `get_split_kws(split_types="Segment", n_split_min=1, n_split_max=1)` — so the feature is an amino-acid-composition-like mean over the *entire* part (position-agnostic). **Positional** uses `n_split_max>1` (sub-segments) and/or **Pattern** / **PeriodicPattern**, resolving the feature to specific sub-regions/positions. There is no `strategy=` parameter today — the distinction *emerges* from `split_kws`; a named preset is a proposed enhancement. Maps onto **prediction level**: compositional ≈ protein, positional ≈ residue, domain uses both.
_Avoid_: global vs local, whole vs windowed.

**empty split bucket**:
A `(split type, part)` pairing that produces zero splits. Because label
generation is part-length independent, this happens only when the split-type
*config itself* yields no labels — and in practice only for **Pattern**, when
`n_min * steps[0] > len_max` (the shortest pattern already overflows `len_max`;
Segment always yields ≥1, PeriodicPattern always yields splits once its two
steps validate). Such buckets are **silently dropped** from feature generation
(legacy CPP behavior, preserved for parity), so the run proceeds with the
remaining split types rather than erroring. `check_split_kws` emits a
`UserWarning` at validation time naming the offending Pattern config, since a
whole-type drop is almost always a user misconfiguration of `len_max` / `steps`.
_Avoid_: empty split, dropped feature.

### Numerical-mode CPP vocabulary

**dict_num**:
A `Dict[str, np.ndarray]` mapping `entry` to a per-residue numerical tensor of shape `(L, D)`. Generic value source for `CPP.run_num`: covers PLM embeddings, DSSP one-hots, PTM dummies, or any other per-residue numerical representation. Same shape contract as the `embeddings` argument of `EmbeddingPreprocessor.encode`; the rename to `dict_num` signals that the contents need not be PLM embeddings. When `dict_num` is supplied, the AA→scale lookup in `_filters/_assign.py:101` is bypassed; the per-protein tensor is sliced into parts and consumed directly. The accompanying `df_scales`/`df_cat` then *name* the D dimensions (e.g. `dim_0`, `DSSP_H`, `phospho_S`) for the redundancy filter and output columns. **All three preprocessor families emit `[0, 1]`-normalized values** — `StructurePreprocessor` / `AnnotationPreprocessor` encoders per the per-key `NORMALIZATION_RECIPES` (NaN for unresolved positions), and `EmbeddingPreprocessor.encode` per its `method=` normalizer. `CPP.run_num`'s default `max_std_test=0.2` pre-filter is calibrated for that `[0, 1]` convention; raw (unbounded) embeddings must be passed through `encode` first.
_Avoid_: embeddings (too narrow — covers only one source), num_tensor, per_residue_dict.

**CPP.run_num**:
A method of `CPP` whose per-residue value source is a pre-sliced numerical tensor (`dict_num_parts`) rather than an AA→scale lookup. **The whole `CPP` is constructor-bound** to `df_parts` + `df_scales` + `df_cat`; `run_num(dict_num_parts=, labels=, ...)` consumes the per-part NaN-padded tensors produced by `NumericalFeature.get_parts(df_seq, dict_num) → (df_parts, dict_num_parts)`. Hard invariant: `D == len(df_scales.columns) == len(df_cat)`; `df_scales`/`df_cat` *name* the D dims (the per-AA values are unused as a value source). `check_cat=True` (default) groups by `df_cat.category` in the redundancy filter. Same pipeline + output schema as `CPP.run`. _(Earlier drafts described run_num's value source as "per-call `df_seq` plus optional `dict_num`" — that is outdated; the value source is the constructor `df_parts` plus `dict_num_parts`.)_
_Avoid_: run_embed (misleading — also handles non-embedding inputs and pure sequences), run_v2.

**_filters/**:
Backend folder holding the canonical CPP pipeline (seq-mode AND numerical-mode). Per-residue values flow between stages as `dict[part] = (n_samples, L_part_max, D)` float32 tensors with NaN padding for short parts; downstream aggregation uses `np.nanmean`. Performance: split-position computation reused across D via numpy broadcasting (collapses the n_dims loop), and a *streaming pre-filter* keeps only the survivors of the `std_test` mask in memory so `add_stat` no longer recomputes feature values from scratch. Batching (`n_batches`) partitions over D, not scales/parts. The Cython acceleration lives in the sibling `_filters_c/` folder. Originally named `_filters_num/` during PR4-PR5 when a parallel legacy `_filters/` still existed; renamed to `_filters/` in PR6 after the legacy was removed.
_Avoid_: _embed_filters/ (too narrow), _filters_num/ (legacy PR5 name), _filters_v2/ (versioning is ephemeral).

### CPP sweep & diagnostics vocabulary

**CPPGrid**:
Public class that runs a **grid sweep** of `CPP` configurations in one call, orchestrating the full `get_df_parts` → `get_split_kws` → scales → `run`/`run_num` pipeline so the user skips the manual wiring. The dataset (`df_seq` + `labels`, plus `dict_num` for the numerical arm) is bound at construction; `run` takes four **stage-grouped param dicts**. Parallelizes **across** configurations (inner `run`/`run_num` stays serial); the default `backend="threads"` shares dataframes in-process and sidesteps the Py3.14/macOS spawn footgun. **Smart-sweeps:** configs differing only in `n_filter` run CPP **once at the max** and the rest are exact `head(n)` slices (greedy top-down redundancy filter ⇒ top-`n` invariant); `df_parts`/`split_kws` are built once per sub-config and reused.
_Avoid_: run_many (the rejected classmethod name), sweep (too generic), grid search (implies hyperparameter tuning with a model).

**stage-grouped param dicts** (`params_parts`, `params_split`, `params_scales`, `params_cpp`):
The four `CPPGrid.run` arguments, each feeding exactly one pipeline stage (`get_df_parts`/`get_parts`, `get_split_kws`, the list of `df_scales`, and `CPP.run`/`run_num`). Within a dict, a **`list` value is a swept axis** and a scalar is fixed; a list-valued knob (`steps_pattern`, `list_parts`) is swept by wrapping it in an outer list. The grid is the Cartesian product across all swept entries; `df_cat` is resolved internally from each `df_scales` ("df_scales is enough").
_Avoid_: param_ranges, param_grid, config list (earlier rejected shapes).

**df_params**:
The lightweight sweep-summary DataFrame returned by `CPPGrid.run` alongside `list_df_feat` (one row per configuration, `itertools.product` order). Scalar axes hold the literal value; object axes (`df_scales` and any list-valued knob) hold the **position index** into their candidate list — reconstructable from the inputs without storing heavy objects. Carries `n_warnings` and `n_errors` count columns. Naming parallel: `params_*` = what was *asked*; `df_params` = what was *run*.
_Avoid_: df_combos (earlier name), df_results (collides with `list_df_feat`), df_grid.

**last_filter_stats_**:
A plain dict set on a `CPP` instance after every `run`/`run_num` (also stashed in `df_feat.attrs["last_filter_stats"]`), recording the filter funnel: `n_candidates` (features generated before filtering), `n_after_prefilter`, `n_after_redundancy`, `n_final`. Exposed programmatically via `return_stats=True`. Two shortfalls are surfaced as mutually-exclusive warnings: a **sparse-config `UserWarning`** when `n_candidates < n_filter` (the config can't generate enough; small `n_jmd`/narrow `split_kws`), and a **filter-shortfall `RuntimeWarning`** when filtering removed too many.
_Avoid_: FilterStats (rejected typed record), filter_diagnostics.

**n_jobs contract**:
The unified parallelism convention: `1` = serial, `-1` = all cores, `N>1` = exactly N, `None` = optimized via `ut.resolve_n_jobs(n_jobs, n_work)`. The `options['n_jobs']` global (default `"off"`) overrides the per-call value when set. Inside `CPPGrid`, each configuration runs at `n_jobs=1` while the grid parallelizes across configurations.
_Avoid_: n_processes, n_workers (use `n_jobs` everywhere).

### Site-localization metrics vocabulary

**per-protein AP**:
Average precision computed per protein over its per-residue scores against that protein's positive positions, then aggregated across proteins (NaN-aware). The canonical site-localization ranking metric. A **tolerant** variant (`tolerance=±k`) counts a hit within `k` residues of a true positive as correct, for positional jitter. `aa.comp_per_protein_ap`.
_Avoid_: mAP (averaging convention differs), accuracy.

**detection metrics**:
Recall / precision / F1 / MCC at a fixed score threshold, pooled across all per-residue predictions — the "is the true site actually *called*?" question, distinct from ranking. `aa.comp_detection_metrics`.
_Avoid_: classification metrics (too generic), accuracy@threshold.

**bootstrap CI**:
A seeded percentile confidence interval over a per-protein metric vector, for honest small-N uncertainty reporting. `aa.comp_bootstrap_ci`.
_Avoid_: confidence band (plotting term), error bar.

**peak-preserving smoothing**:
NaN-aware triangular/gaussian smoothing of a per-residue score track that takes `max(smoothed, raw)` so true peaks are never attenuated — for windowed protease/PTM prediction where positional jitter is universal. Pure-numpy, seeded where stochastic. `aa.smooth_scores`.
_Avoid_: denoising, blurring (attenuates peaks).

**rank plot**:
A per-protein **max-score-vs-rank** scatter colored by group (substrate / hold-out / non-substrate) with optional threshold lines — the standard deployed-predictor sanity check. A standalone `aa.plot_rank` (deliberately not a `*Plot` method; pairs with the standalone `aa.metrics` functions).
_Avoid_: ranking plot (collides with `CPPPlot.ranking`, which ranks *features* from `df_feat`).

### Feature selection vocabulary

**feature selection**:
The *post-fit* reduction of a `df_feat` to a chosen subset of features, performed by `TreeModel.select_features(df_feat, strategy, param)` after `TreeModel.fit`. It consumes the signals `fit` already produced — the Monte Carlo `feat_importance` and the per-round `is_selected_` masks — and returns a row-filtered `df_feat`. Distinct from **RFE prefiltering**, which happens *inside* `fit(use_rfe=True)` (an iterative re-fit loop), and from CPP **feature engineering**, which *creates* features rather than selecting among them.
_Avoid_: feature filtering (overloaded with CPP's split/scale filtering), feature extraction, dimensionality reduction.

**selection strategy**:
One of the tree-native rules `select_features` dispatches on, named by a `STRATEGY_*` constant: `top_k` (keep the `param` highest-`feat_importance` features), `threshold` (keep features with `feat_importance ≥ param`), `frequency` (keep features chosen in `≥ param` fraction of the per-round `is_selected_` masks — meaningful only when `fit` ran with `use_rfe=True`). The single `param` knob is a numeric scalar whose admissible type is fixed by the strategy (or a `dict` for a future multi-knob strategy), mirroring `sample_synthetic`'s polymorphic **generation strategy**. RFE itself is *not* a selection strategy — it is the `fit`-time engine that produces the masks `frequency` aggregates.
_Avoid_: selection mode (collides with `output_mode`), selection method.

**is_preselected**:
A constructor-level boolean mask on `TreeModel` marking features to keep *before* RFE runs in `fit` — an upstream gate, not a **selection strategy**. Orthogonal to `select_features`, which acts after fitting.
_Avoid_: preselection strategy (it carries no strategy), prefilter (collides with RFE prefiltering).

**feature pruning**:
The *model-free*, *post-hoc* reduction of a [[df_feat]] by dropping features that are near-constant
or empirically redundant **across the user's own samples**, performed by
`SequenceFeature.prune_by_variance` and `SequenceFeature.prune_by_correlation`. Pruning runs on a
fitted `df_feat` (e.g. from `CPP.run`) and composes **before** model-based **feature selection** —
the recommended order is **variance pruning → correlation pruning → `TreeModel.select_features`**.
Both methods are df_feat-in / df_feat-out (row-filtered, reset index) and build the feature matrix
via `SequenceFeature.feature_matrix` (or accept a pre-computed `X`). Distinct from **feature
selection** (model-based, TreeModel), CPP **feature filtering** (the in-run split/scale screening),
CPP **redundancy reduction** (scale-vector correlation + position overlap, see below), and **feature
simplification** (`CPP.simplify`, which relabels scales). See ADR-0026.
_Avoid_: feature filtering (the CPP in-run term), feature reduction (overloaded with selection),
data cleaning (too generic).

**variance pruning**:
`SequenceFeature.prune_by_variance(df_feat, df_parts, threshold=0.0)` — drops every feature whose
**feature-matrix column variance over all samples** is at or below `threshold`. The default `0.0`
removes only strictly constant features (zero peak-to-peak range, robust to float epsilon); raise it
to prune low-variance features. Distinct from CPP's in-run pre-filter, which screens *candidate*
features by the **test-group** standard deviation (`max_std_test`) rather than the spread of the
already-selected features over all samples.
_Avoid_: low-variance filter (use the method's pruning verb), constant-feature filter (too narrow).

**empirical correlation pruning**:
`SequenceFeature.prune_by_correlation(df_feat, df_parts, max_cor=0.7)` — among features whose
**realized feature values are pairwise correlated** beyond `max_cor` (absolute Pearson over the
actual samples), keeps the higher-`abs_auc` feature and drops the others; deterministic because
features are ordered by `[abs_auc, abs_mean_dif]` before pruning, and constant columns (undefined
correlation) are always retained. **Deliberately different** from CPP's in-run **redundancy
reduction**, which compares the underlying *scale vectors* (`df_scales.corr()`) plus positional
overlap — empirical pruning catches features that are redundant on a *specific dataset* even when
their scales are not. Reuses the `NumericalFeature.filter_correlation` matrix primitive.
_Avoid_: redundancy reduction (reserved for CPP's in-run scale-correlation step), correlation
filtering (ambiguous about scale-vector vs sample-level correlation).

**protein-level redundancy reduction**:
`AAclust.select_proteins(df_seq, X, ...)` — core (no `pro` dep) clustering of a **pre-pooled
per-protein feature matrix** `X` (CPP features, pooled embeddings, DSSP/structural features),
keeping one medoid (representative) per cluster and annotating `df_seq` with
`cluster` / `is_representative` / `dist_to_rep`. The **numerical** counterpart of `filter_seq`
(pro, sequence-identity clustering via CD-HIT/MMseqs2) and orthogonal to CPP's in-run
scale-correlation redundancy reduction. Pooling per-residue inputs to one vector per protein is
the caller's job (the method takes a single matrix, not `dict_num`).

### Explainability (CPP-SHAP) vocabulary

**feature importance**:
A **non-negative, group-level** ranking signal in `df_feat`, column `feat_importance` (`ut.COL_FEAT_IMPORT`), normalized to percent. Produced by `TreeModel.add_feat_importance` (Monte-Carlo tree importances) or by `ShapModel.add_feat_impact(shap_feat_importance=True)` (mean absolute SHAP). It answers *"how important is this feature across all samples?"* — it carries **no direction**. In plots it is the gray cumulative-bar / black-square channel.
_Avoid_: feature impact (signed and sample-level — the opposite axis), weight, attribution (too generic).

**feature impact**:
A **signed, sample- or subgroup-level** SHAP attribution in `df_feat`, columns `feat_impact_'name'` (and `feat_impact_std_'name'` for group averages; base `ut.COL_FEAT_IMPACT`). Produced by `ShapModel.add_feat_impact(names=…, sample_positions=…)`. It answers *"how much, and in which direction, did this feature push the prediction for this sample/group?"* — **positive = red (`ut.COLOR_SHAP_POS`), negative = blue (`ut.COLOR_SHAP_NEG`)**. Its magnitude `abs(feat_impact)` is the sample-level analogue of feature importance.
_Avoid_: feature importance (unsigned, group-level), SHAP value (the raw per-feature attribution before normalization into a `feat_impact_'name'` column).

**shap_plot**:
The uniform boolean toggle on the `CPPPlot` family (`profile`, `heatmap`, `ranking`, `feature_map`) selecting **CPP analysis** (`False`, group-level **feature importance**, `feat_importance` / `mean_dif`) versus **CPP-SHAP analysis** (`True`, sample-level **feature impact**, `feat_impact_'name'` / `mean_dif_'name'`). `True` switches color encoding to signed red/blue and the colorbar to the diverging SHAP colormap. It selects the *interpretation level*; it does not itself run SHAP (that is `ShapModel`). In `feature_map(shap_plot=True)` the cumulative bars stack the per-feature impact in one direction colored by sign; a `mean_dif_'name'` `col_val` keeps the mean-difference heatmap with those bars, while a `feat_impact_'name'` `col_val` moves the impact into the heatmap cells and switches the bars off.
_Avoid_: shap_mode, use_shap, sample_plot.

**fuzzy aggregation** (`fuzzy_aggregation`):
The strategy `ShapModel.fit` selects to turn a soft label `p` ∈ (0, 1) into a SHAP estimate when **fuzzy labeling** is active. `"interpolate"` (default, new in v1.1) fits the model twice (fuzzy sample at 0 → `S0`, at 1 → `S1`) and blends `p·S1 + (1−p)·S0` — the **unbiased** exact-`p` estimate. `"threshold"` (the `Breimann25` sweep) hard-labels the fuzzy sample `1` across a non-uniform `n_rounds × n_selection` grid and averages — a **biased** approximation whose effective positive-fraction is the grid's `frac1`, not `p`; kept as a first-class option. Each fuzzy protein is explained independently against the fixed balanced 0/1 **core**, with the other fuzzy proteins excluded from that run's training data. `n_rounds` (default `5`) is interpolate's speed/stability dial: `1` = fast exact two-fit estimate, `5` = light averaging, `≈15–20` = converged Monte-Carlo mean (run-to-run spread <5% on `DOM_GSEC`).
_Avoid_: fuzzy mode, blend mode, soft-label aggregation.

**CPPStructurePlot**:
Public **pro** plotting class in `aaanalysis/feature_engineering_pro/` (abbr `csp`) that paints per-residue CPP / CPP-SHAP **feature impact** onto a 3D protein structure. Its single method `map_structure(df_feat, pdb=…|uniprot=…)` maps each feature to the residues it spans (`get_positions_`, shifted to absolute residue numbers by `start`) and aggregates `col_imp` per residue with the **same normalized-sum** `CPPPlot.profile` uses — never a re-implemented per-position loop. It **reuses** the shared CPP position backend and the `StructurePreprocessor` structure parser (no duplication; a thin chain-by-id Cα/pLDDT extractor is the only new structure code). Modes: `"impact"` (white→`COLOR_SHAP_POS`/`COLOR_SHAP_NEG` ramp with a `sign·sqrt` perceptual transform) and `"plddt"` (AlphaFold confidence palette); focus `"whole"`/`"fade"`/`"zoom"`. Returns a [[StructureView]]. The structure-side companion to `CPPPlot` for the **CPP-SHAP analysis** level.
_Avoid_: structure_plot, plot_structure (the verb-noun method is `map_structure`), CPPStructure (it is a plot class, suffix `Plot`).

**StructureView**:
The thin return wrapper of [[CPPStructurePlot]]`.map_structure`, exposing a **uniform** `show()` / `write_html(path)` / `savefig(path)` / `_repr_html_` surface over its two render backends (interactive `py3Dmol` and static matplotlib `mplot3d`) whose native objects (`py3Dmol.view` vs `Figure`) are otherwise incompatible. A **pure delegator** — no rendering logic, no state beyond the backend object and the mapped `dict_impact` / `max_abs`. The package's first non-`Axes` plotting return type, a deliberate, documented exception to the "return fig/ax" rule (`savefig` is matplotlib-only; `write_html` is the py3Dmol shareable-interactive output).
_Avoid_: view wrapper, plot handle (it is specifically the structure-render delegator), figure (it is not a matplotlib Figure).

### Scale-set vocabulary

**explainable scale set** (`top_explain_n`):
A simplified amino-acid scale set restricted to the **n most interpretable AAontology subcategories** (`top_explain_n` ∈ {5,10,…,60}), loaded via `aa.load_scales`. Curated by ranking subcategories on **interpretability** from unsupervised clustering combined with expert domain knowledge of AAontology (no publication). The interpretability axis is **orthogonal to performance**: it is the explainability-first sibling of **top60** (which is performance-ranked and already AAclust redundancy-reduced). By default it returns **all member scales** of the selected subcategories (no redundancy reduction); pass **`top_explain_min_th`** to AAclust-reduce. Mutually exclusive with `top60_n`.
_Avoid_: simplified scales, interpretable subset, top_subcat (the xlsx-era name — the column and selector are `top_explain`).

**interpretability tier** (`top_explain` column):
The cumulative inclusion threshold (5,10,…,60) assigned to each classified subcategory; selecting `top_explain_n=n` keeps every scale whose subcategory has `top_explain <= n`. The 7 `Unclassified (...)` subcategories have `top_explain = NaN` and are always excluded by a tier selection (so `unclassified_out` is moot there). Lives on [[subcategory overview]] (`df_subcat`) — its single source; the per-scale `df_cat` no longer carries it, and tier selection joins it on by subcategory.
_Avoid_: interpretability level, rank, top_subcat.

**interpretability rating** (`interpret_grade` column; surfaced as the [[interpretability grade]]):
A per-subcategory 1–10 score (1 = most interpretable) underlying the tiering. Lives on [[subcategory overview]] (`df_subcat`) in the `interpret_grade` column, not on the per-scale `df_cat`. Distinct from a *tier*: the rating is the raw judgement, the tier is the cumulative cut. Column and the `CPP.simplify` parameter share the `interpret_grade` / `max_interpret_grade` naming.
_Avoid_: `interpretability` column (renamed to `interpret_grade`), interpretability score (overloaded).

**subcategory overview** (`df_subcat`, `aa.load_scales(name="subcat")`):
One row per AAontology subcategory (74) — the single home for per-subcategory [[interpretability rating]] and [[interpretability tier]], plus `cluster`, two scale counts, and `subcategory_description` / `key_references`. The two counts are AAindex-aware: `n_scales` (all member scales) and `n_scales_aaindex` (excluding the non-AAindex `LINS`/`KOEH` scales), computed live — interpretability and tier are AAindex-independent (subcategory-intrinsic). `just_aaindex=True` drops subcategories with no AAindex scales; `unclassified_out=True` drops the `Unclassified (...)` rows. Companion to `scales_cat` (the per-scale classification), which no longer carries the grade/tier columns.
_Avoid_: df_cat_int, subcat table (the object is `df_subcat`).

**top_explain_min_th**:
The Pearson-correlation threshold (∈ {0.3,…,0.9} or `None`) for an optional `AAclust` redundancy reduction layered on a tier, served from **pre-computed** per-tier selections (AAclust default settings, fixed seed). `None` = no reduction. Reduction is computed **per tier** (medoids are not nested across tiers) and on **dual grids** (with / without AAindex) so `just_aaindex` stays correct. May leave a subcategory with no representative — the reduced set need not cover every tier subcategory. See ADR-0025.
_Avoid_: min_corr, redundancy threshold (use the AAclust term `min_th`).

**feature simplification** (`CPP.simplify`):
The post-hoc rewriting of a fitted [[df_feat]] into a **more interpretable, and ideally
smaller** one: each feature's scale is swapped for a *correlated* scale drawn from a
**strictly better-graded subcategory** (a lower **interpretability grade** — see below; keeping
`PART-SPLIT`), the feature stats are recomputed, and the swap is accepted only if it keeps
passing CPP filtering (`max_std_test`) and does not reduce a cross-validation score (the CV-gate
model `ml_model` / `ml_metric` / `ml_th` / `ml_cv`, seeded from the CPP instance's
`random_state`). The swapped set is then redundancy-reduced — but this **protects original
features**: it only removes a *swapped* feature that became redundant with a kept feature
(signed correlation, matching `run`), never a feature the user already had. `max_interpret_grade`
caps the worst grade kept; `strategy` is `greedy` / `consolidate` / `swap_all`. The candidate
pool is the full rated AAontology scale set, loaded internally. Distinct from **feature
selection** (which *subsets* features by importance) and CPP **feature engineering** (which
*creates* features) — simplification *relabels* a feature onto a more interpretable scale while
preserving its signal.
_Avoid_: feature reduction (overloaded with selection), scale substitution (the unit is a feature).

**interpretability grade**:
The user-facing name for the [[interpretability rating]] (a 1-10 per-subcategory value, **1 =
best / most interpretable, so lower is better**) when it is used as a *threshold* on the output —
`CPP.simplify(max_interpret_grade=g)` keeps features graded `<= g` and replaces worse ones. Same
number as the `interpret_grade` column; "grade" is chosen so the parameter name signals that
lower is better.
_Avoid_: interpretability score (a high score usually reads as good; the grade is inverted).

### Optimization-equivalence vocabulary

These name the tiers of the **numerical-equivalence tolerance policy** (ADR-0032)
that governs when a performance optimization is allowed to change a numerical
output. An optimization lands at the **strictest tier it can satisfy**; each
carries the evidence + regression anchor its tier requires (extending the
ADR-0015 anchor pattern).

**byte-identical** (tier **T1**, the default):
Output is bit-for-bit identical to the implementation it replaces. The bar for
the vast majority of optimizations (vectorizing a loop with the same formula,
hoisting an invariant, caching a deterministic result). No extra evidence beyond
the change's own unit / parity tests.
_Avoid_: same-output (overloaded — historically meant T1+T2 together), exact (ambiguous).

**numerically-equivalent** (tier **T2**):
Numerical outputs agree to `np.allclose(atol=1e-10, rtol=0)` **and** every
**discrete decision** (labels, selected features / medoids, kept / dropped /
ranked sets) is identical. Covers ULP-level reorderings (einsum/BLAS reductions,
rolling-mean aggregation) and `allclose` distance/correlation reformulations.
_Avoid_: approximately equal, close-enough (the tolerance is exact: `atol=1e-10, rtol=0`).

**statistically-equivalent** (tier **T3**):
Outputs differ and discrete decisions *may* differ, but documented quality
metrics (clustering quality, downstream AUC, kept-feature overlap) stay within an
**agreed, numerically stated band** on named canonical datasets. Reserved for
genuinely algorithmic changes (e.g. AAclust binary-search `k`) — never a fallback
for a change that could meet T2.
_Avoid_: equivalent (unqualified — say which tier), good-enough.

**discrete decision**:
A non-numerical output an optimization must not silently change under T2 — a
label assignment, a selected feature / medoid, a kept / dropped / ranked set.
Distinct from a numerical *value*: a value may drift within tolerance, a decision
may not (under T2) without escalating to T3.
_Avoid_: result (overloaded), output (overloaded).

### Agentic-readiness & package-boundary vocabulary

These name the scope of the **agentic-readiness** program and the line between
AAanalysis and **ProtXplain** (the downstream agent-integration package). See
ADR-0035 (the original boundary) and ADR-0038 (the refinement below).

**agentic readiness**:
Making the OSS primitives maximally **legible, typed, contracted, and
improvable** — for human users *and* for the coding agents that *improve* the
package (consistent class templates, documented data contracts like `df_feat` /
`DICT_DF_SCHEMAS`, honest type hints, tests). It is **not** an in-package
agent-tool framework: the layer where external agents call AAanalysis *as a
tool* is withheld to ProtXplain. Science/product work (structure-XAI, XAI-eval,
the design bridge) is a separate track, **not** part of agentic readiness.
_Avoid_: "agent support" (overloaded — conflates the two audiences below).

**two agent audiences**:
The distinction that resolves the word "agent". Agents that *improve* AAanalysis
are served **in** AAanalysis (types, contracts, tests, templates). Agents that
*use* AAanalysis as a tool are served by **ProtXplain** (the MCP /
machine-readable tool contract). Usability + improvability stay here;
tool-integration goes downstream.

**machine-readable tool contract (boundary)**:
The border between the two packages. The MCP server, JSON/tool schemas, verb
orchestration, and selection/ranking/decision/OOD logic are **ProtXplain**;
human- and sklearn-idiomatic convenience is **AAanalysis**. A boundary, like the
**relational / interaction (scope boundary)** — not a feature AAanalysis is
missing. See ADR-0038.

**golden pipeline**:
A user-facing one-call function in the stateless `aaanalysis.pipe` (`aap`)
namespace that chains primitives into a common workflow. Thin (no own algorithm),
opt-in, defaults **byte-identical** to the explicit primitive path; emits plain
numpy/pandas (plus a Matplotlib `Axes` when it plots) that feed sklearn and torch
equally (torch stays the `[embed]` extra). It is **AAanalysis** convenience, not
ProtXplain. Named `verb_noun` (the only schema), the verb signalling **End** vs
**Means**. The mpl analogy: `aap` is to the primitives as `pyplot` is to the
`Axes`/`Figure` API. In the user-facing docs the two tiers are named the
**implicit interface** (`aap`, the golden pipelines) and the **explicit
interface** (`aa`, the building blocks) — the same pyplot-vs-objects split, kept
verbatim in `docs/source/api.rst` and `getting_started.rst`. See ADR-0040.
_Avoid_: "verb" / "tool" (those name the ProtXplain agent-integration layer);
calling a **Means** a golden pipeline (only an **End** is one).

**End** (golden pipeline):
A golden pipeline that returns a **deliverable** — the thing the user wanted:
`obtain_samples` (→ training set + validation report), `find_features`
(→ `df_feat` + map), `predict_samples` (→ a `{(feature_set, model): predictor}` dict + a
cross-validated comparison `df_eval` over the feature-set × model grid), `explain_features`
(→ SHAP + map, *pro*), `design_mutations` (→ ΔCPP table), `evaluate_models`
(→ repeated-CV + CIs). Returns the uniform triple `(results, figs, evals)` — each
slot a bare object or `None` (a plotting End sets `figs=None` when `plot=False`;
a non-evaluating one sets `evals=None`). "Golden pipeline" means an End.
_Avoid_: treating a producer step as an End.

**Means** (producer):
A step that prepares an **input** for an End (reliable negatives, a reduced scale
set, sampled windows, embeddings). Exposed as a **flag** on the End it feeds by
default (`find_features(dpulearn=True, subcategories=…)`); promoted to a standalone
`aap` producer (`sample_windows`, `embed_sequences`) only on genuine standalone
need — never both a flag and a function for the same canonical path.
_Avoid_: "helper" (overloaded); making every Means a standalone function.

**search effort** (`search`, the `find_features` grade):
How wide the `find_features` CPP AutoML feature search ranges — `"fast"` (a single
default configuration, no search), `"balanced"`, `"exhaustive"` (progressively wider
sweeps of the Split / Part / Scale / `n_filter` levers, the winner chosen by
cross-validated model performance). The parameter is named **`search`**, deliberately
**not `optimization`**: "optimization" is reserved for the [[SeqOpt]] sequence
directed-evolution chain, and the search-effort grade is a different idea (how hard to
look for discriminating features, not how to evolve a sequence toward a target).
Distinct from the per-winner **refinement** (`simplify` + RFE) that prunes the selected
feature set afterwards.
_Avoid_: optimization (reserved — SeqOpt); mode/depth (overloaded).

**axis impact** (`find_features` sensitivity staging):
The main-effect sensitivity of one feature-space axis (Part, Split, or Scale) in the
`find_features` search: the `max − min` spread of that axis's **marginal-mean**
cross-validated score across its levels (averaging over the other axes), normalized per
metric to `[0, 1]` and averaged across metrics. The axis with the largest impact is the
**dominant axis**, refined against `n_filter` while the others stay fixed at the
stage's Pareto-then-simplest optimum.
_Avoid_: importance (reserved for per-feature `feat_importance`).

### Plotting return-contract vocabulary

**FigAxResult / `(fig, ax)` contract**:
The single return shape of every public `*Plot` method (`AAclustPlot`, `CPPPlot`,
`dPULearnPlot`, `AAMutPlot`, `SeqMutPlot`, `AAlogoPlot`). `FigAxResult` is a thin
`tuple` subclass (`ut.FigAxResult`) that unpacks as `fig, ax = ...` and indexes
like a 2-tuple, and **also forwards attribute access to `ax`** so legacy
`ax = ...; ax.set_title(...)` still works — the proxy that lets the unification
land without breaking the previously Axes-only methods. The second element is a
single `Axes` or an array of `Axes` (multi-panel `eval` / `multi_logo`). Methods
that also yield a DataFrame (`AAclustPlot.centers` / `medoids`) put it on the
trailing-underscore attribute **`df_components_`**, not a third tuple element.
Replaces the historical three-shape mess (`(fig, ax)` / bare `Axes` /
`(ax, df)`). The `centers` / `medoids` unpacking change is the one genuinely
breaking part, scheduled for the next major. The lone exception to the contract
is `CPPStructurePlot`, which returns a `StructureView` (ADR-0028). See ADR-0039.
_Avoid_: "returns an Axes" / "returns a figure" (it returns the `(fig, ax)` pair);
"third return value" for `centers`/`medoids` (the DataFrame is an attribute).

## Relationships

- A **df_seq** row contains one **entry** and one sequence; optionally a **pos column** cell of 1-based positions.
- An **explainable scale set** (`top_explain_n`) ranks AAontology subcategories by **interpretability tier**; **top60** ranks scale *sets* by performance. The two selectors are mutually exclusive, and only the explain path adds `interpretability` / `top_explain` columns to `df_cat`. See ADR-0025.
- A **test window** is extracted at a **P1 anchor** listed in the **pos column**.
- A **reference window** is sampled from a **candidate pool** and tagged with a **role** (workflow meaning) and a **strategy** (sampling provenance).
- A **control window** is a **reference window** produced by the **generator** in `sample_synthetic` (no source entry).
- Every row in a `segments`-mode output has an **entry_win**; for non-synthetic rows it is globally unique by construction across calls; for **control windows** it is unique per call only.
- The **identity filter** uses **test windows** as the anti-leakage reference; the **motif filter** uses a **PWM** as the gate.
- A CPP feature is one **split** applied to one **part** scored by one scale; an **empty split bucket** contributes no features and is silently dropped (with a validation-time warning for the Pattern case).
- A **prediction level** fixes the **unit of comparison** (window → residue, part set → domain, whole chain → protein) and constrains **reference construction**; it maps 1:1 to the `load_dataset` prefix (`AA_` / `DOM_` / `SEQ_`). See ADR-0022.
- **Compositional** CPP strategy is a single `Segment(1,1)` whole-part split; **positional** is `n_split_max>1` / **Pattern** / **PeriodicPattern**. Compositional suits **protein level**, positional suits **residue level**, **domain level** uses both.
- **Determinant discovery** and **design / engineering** are cross-cutting use-case classes (any level); **relational / interaction** is a documented scope boundary, not a level.
- **Feature selection** happens *after* `TreeModel.fit`: `top_k` / `threshold` read the Monte Carlo `feat_importance`; `frequency` aggregates the per-round `is_selected_` masks that **RFE prefiltering** (`fit(use_rfe=True)`) produced. **is_preselected** gates features *before* RFE. So the order is: `is_preselected` (pre-RFE) → RFE in `fit` → `select_features` (post-fit).
- **Feature pruning** sits *before* feature selection and is model-free: **variance pruning** → **empirical correlation pruning** (`SequenceFeature.prune_by_*`) shrink a `df_feat` on the user's own samples, then `TreeModel.select_features` applies the model-based cut. Pruning's empirical correlation is distinct from CPP's in-run **redundancy reduction** (scale-vector correlation + position overlap). See ADR-0026.

## Example dialogue

> **Dev:** "If I call `sample_different_protein`, do I get a row for every entry in `df_seq`?"
> **Domain expert:** "No — only entries with no **pos column** entries form the **candidate pool**. Entries with positives are excluded from sampling but still contribute their **test windows** to the **identity filter**, so they affect *which* candidates survive."
>
> **Dev:** "So a `Negative` row from `sample_same_protein` and an `Unlabeled` row from `sample_different_protein` carry different **role** tags, but I can tell which method produced them from the **strategy** column?"
> **Domain expert:** "Right. **Role** is workflow-level — what the row *means* in your downstream task; override the defaults if your workflow differs. **Strategy** is provenance — which sampling method produced it. Two rows can share a strategy but carry different roles, or vice versa."
>
> **Dev:** "What's a **control window** for?"
> **Domain expert:** "A synthetic window with no source protein. Useful as a null baseline, for composition-bias controls, or for benchmarking. It shares the segments-mode schema with the other outputs but its `entry_win` is only unique within one `sample_synthetic` call — dedupe across calls on the `window` string instead."
>
> **Dev:** "Why is `motif_pwm` DataFrame-only — can't I just pass an ndarray?"
> **Domain expert:** "Ndarray columns are implicitly alphabetical, and a wrong-order array silently gives wrong scores. We rejected that path. Pass `pd.DataFrame(arr, columns=ut.LIST_CANONICAL_AA)` if you only have an array; the column order then can't be wrong because pandas reindexes internally."

## Flagged ambiguities

- `mode` was used for both `sample_synthetic`'s polymorphic generator parameter AND for `output_mode`. **Resolved**: the synthetic parameter is now **generator**; `output_mode` retains its name (different axis — schema vs. recipe).
- `negative` was used informally for both labeled-negative rows and any non-test sampled row. **Resolved**: **reference window** is the umbrella term; **role** holds the workflow-specific meaning (`Negative`, `Unlabeled`, `Control`, …).
- `NegativeSampler` (the class proposed by issue #66) is **not** a separate class. `AAWindowSampler` already provides same-/different-protein/synthetic sampling, the N/U/Control **role** taxonomy, the unified `segments` schema, composable filters, and per-call seeds — i.e. the substance of #66. **Resolved**: the only genuine gaps (`sample_benchmark_set` multi-arm orchestration and a `custom filter` hook) are added to `AAWindowSampler`; `NegativeSampler` as a name/class is dropped. See ADR-0020.
- `center` is used in backend code for the 0-based window-center index; in user-facing language and outputs we use **P1 anchor** / **source position** (1-based). **Resolved**: backend stays 0-based internally; frontend / output is 1-based throughout.
- `positions` was a column constant for `df_feat` (CSV string of feature positions); `pos` is the column constant for `df_seq`'s positive positions (list of ints). **Resolved**: distinct constants — `COL_POSITION` (`df_feat`, `str`) and `COL_POS` (`df_seq`, `list[int]`).
- `label-neutral` was a claim in the `AAWindowSampler` class docstring; in practice the API ships opinionated **role** and `label_ref` defaults. **Resolved**: framing is dropped; class docstring now states defaults explicitly assume PU-learning / hard-negative-mining workflows.
- `protein level` (user-facing) vs the `SEQ_` dataset prefix ("sequence") named the same thing two ways, risking a spurious third "sequence level". **Resolved**: they are one **prediction level**; "protein-level" is the canonical user-facing term, `SEQ_` is the dataset-prefix spelling, and "sequence" stays reserved for the amino-acid string (`df_seq`, the `sequence` column). The cleavage "between two residues" case is a **sub-mode** of **residue level**, not a separate level. See ADR-0022.
- `v1.1` / `v1.2` were used for **three different axes**: (a) the **package release version** (`pyproject.toml` / PyPI), (b) the **StructurePreprocessor feature-set revision** (rev 1 = DSSP+PDB; rev 1.1 = +AlphaFold — see ADR-0002 and the **feature key** entry), and (c) **git branch names** (`feat/structure-preprocessor-v1.2`). **Resolved**: the *only* authoritative version line is the **package version**, which follows semver — the line is `1.0.1 → 1.0.2 → 1.0.3 → 1.1.0` (the next, unreleased, minor; it ships the whole preprocessor family + CPPGrid + the site-localization metrics). The StructurePreprocessor's internal "rev 1 / rev 1.1" and any branch-name version are **not** package versions; never write a bare "v1.1" to mean a feature-set revision. See ADR-0010.

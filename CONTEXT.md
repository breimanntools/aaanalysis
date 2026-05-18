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
A `df_seq` column whose cells hold per-row 1-based integer positions of interest (typically labeled positives); each cell is a `list[int]`, a single `int`, or empty (`None` / `NaN` / empty list / empty string). Default column name `"pos"`, referenced as `ut.COL_POS`.
_Avoid_: positions column (collides with `df_feat`'s `COL_POSITION`, which is a CSV string of feature positions).

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
A tag stored in the output's `strategy` column identifying which sampling method produced the row. Values: `same_protein`, `different_protein`, `motif_matched`, `synthetic:<generator>`.
_Avoid_: method, source, origin.

**entry_win**:
A row's window identifier, formatted `<entry>_<start_pos>-<end_pos>` with 1-based inclusive coordinates for protein-sourced windows, or `synth_{i}` (per-call counter) for synthetic windows. Identical biological windows across calls share the same `entry_win`, making `drop_duplicates(subset="entry_win")` the natural dedupe primitive — except for synthetic outputs, where the per-call counter is not call-stable.
_Avoid_: window_id, row_id.

**candidate pool**:
The set of eligible windows from which a sampling method draws, defined per strategy. `same_protein`: positions ≥ `min_distance_to_positive` from any positive on a positive-containing protein. `different_protein` / `motif_matched`: any window on a protein with no listed positives. `synthetic`: drawn fresh from the generator distribution.
_Avoid_: candidates, eligible set.

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
An array of shape `(window_size, 20)` representing per-position residue scores over the 20 canonical AAs. Accepted by `motif_pwm` as either an `np.ndarray` (columns implicitly in alphabetical order `ACDEFGHIKLMNPQRSTVWY`) or a `pd.DataFrame` (columns by AA name, any order, reindexed internally).
_Avoid_: scoring matrix (too generic), motif matrix.

**motif filter**:
The pair `(motif_pwm, motif_score_threshold)` used by `sample_same_protein` and `sample_different_protein` to keep (`motif_match="in"`) or drop (`motif_match="out"`) windows whose PWM score crosses the threshold. Optional on these two methods; required and `"in"`-only on `sample_motif_matched` (where it defines the candidate pool, not an overlay filter).
_Avoid_: motif gate, PWM filter.

**identity filter**:
A pair of filters based on per-position residue identity between fixed-length, aligned windows. `max_similarity_to_test` drops sampled windows too similar to any test window (anti-leakage); `max_similarity_within_ref` drops sampled windows too similar to a previously kept sampled window (redundancy reduction).
_Avoid_: similarity filter (overloaded), redundancy filter (too narrow — covers only the second).

## Relationships

- A **df_seq** row contains one **entry** and one sequence; optionally a **pos column** cell of 1-based positions.
- A **test window** is extracted at a **P1 anchor** listed in the **pos column**.
- A **reference window** is sampled from a **candidate pool** and tagged with a **role** (workflow meaning) and a **strategy** (sampling provenance).
- A **control window** is a **reference window** produced by the **generator** in `sample_synthetic` (no source entry).
- Every row in a `segments`-mode output has an **entry_win**; for non-synthetic rows it is globally unique by construction across calls; for **control windows** it is unique per call only.
- The **identity filter** uses **test windows** as the anti-leakage reference; the **motif filter** uses a **PWM** as the gate.

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
> **Dev:** "Why does the **PWM** for `motif_pwm` accept both ndarray and DataFrame?"
> **Domain expert:** "Foot-gun avoidance. Ndarray columns are implicitly alphabetical — silently wrong if you got the order wrong. DataFrame columns are by AA name and reindexed internally, so the order can't be wrong. Use the DataFrame form unless you've already paid the cost to be careful with the array layout."

## Flagged ambiguities

- `mode` was used for both `sample_synthetic`'s polymorphic generator parameter AND for `output_mode`. **Resolved**: the synthetic parameter is now **generator**; `output_mode` retains its name (different axis — schema vs. recipe).
- `negative` was used informally for both labeled-negative rows and any non-test sampled row. **Resolved**: **reference window** is the umbrella term; **role** holds the workflow-specific meaning (`Negative`, `Unlabeled`, `Control`, …).
- `center` is used in backend code for the 0-based window-center index; in user-facing language and outputs we use **P1 anchor** / **source position** (1-based). **Resolved**: backend stays 0-based internally; frontend / output is 1-based throughout.
- `positions` was a column constant for `df_feat` (CSV string of feature positions); `pos` is the column constant for `df_seq`'s positive positions (list of ints). **Resolved**: distinct constants — `COL_POSITION` (`df_feat`, `str`) and `COL_POS` (`df_seq`, `list[int]`).
- `label-neutral` was a claim in the `AAWindowSampler` class docstring; in practice the API ships opinionated **role** and `label_ref` defaults. **Resolved**: framing is dropped; class docstring now states defaults explicitly assume PU-learning / hard-negative-mining workflows.

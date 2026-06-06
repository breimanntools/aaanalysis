# ADR-0022 — Prediction-task taxonomy: residue / domain / protein, by unit-of-comparison

Status: Accepted — 2026-06-06

## Context

AAanalysis ships benchmark datasets under three name prefixes — `AA_*`, `DOM_*`,
`SEQ_*` — and `load_dataset` already branches on them (`AA_*` get sliding-window
extraction, `DOM_*` carry `tmd_start`/`tmd_stop`, `SEQ_*` are whole-sequence).
But this was an internal *data* convention, never surfaced as a user-facing
mental model. New onboarding docs — a concept-overview page (#21) and
task-oriented protocols (#35) — need one canonical taxonomy to organize around,
and the glossary defined no term for "what level does this task predict at."
Without one, the concept table and each protocol would invent their own
vocabulary.

## Decision

**D1 — Three prediction levels, mapped 1:1 to the dataset prefixes.** Residue
(`AA_*`), Domain (`DOM_*`), Protein (`SEQ_*`). "Protein-level" is the
user-facing alias of the `SEQ_` prefix; `SEQ_` is read as "sequence", not a
distinct third concept.

**D2 — Residue level has two sub-modes, not two levels.** *single-residue* (odd
`aa_window_size`, a site *on* a residue — PTM/modification) and
*between-residues* (even window, a scissile bond P1│P1′ — cleavage). They share
the windowing machinery and differ only by window parity, so they are sub-modes
of one level.

**D3 — The defining axes are unit of comparison + reference construction, not
biological scale.** The level is a proxy; what actually determines the CPP setup
is (a) the part profiled (window / TMD part-set / whole chain) and (b) how the
contrasting set is built (labeled A/B, non-site windows, unlabeled pool,
composition-matched shuffled background). The concept table and protocols lead
with these two axes, with the level as the convenient label.

**D4 — Two cross-cutting use-case classes sit alongside the levels.**
*Determinant discovery* (no prediction target — contrast two groups to surface
the distinguishing physicochemistry, interpreted via AAontology) and
*design/engineering* (move a sequence toward a target CPP profile, AAMut/SeqMut).
Both apply at any level and showcase the interpretability edge, so they are
first-class rows in the concept table, not buried as examples.

**D5 — Relational/interaction is a scope boundary, not a level.** Interface
*segments* are in scope; long-range residue–residue contacts and PPI pair
prediction hand off to structure/PLM tooling. Stated explicitly so the
taxonomy's edge is honest rather than implied-complete.

## Rejected alternatives

- **Four flat levels** (single-residue · residue-pair · domain · protein).
  Promotes the cleavage "between" case to a peer level. Rejected: it shares the
  residue windowing path entirely (differs only by window parity), so a peer
  level overstates the structural difference and breaks the clean 1:1 with the
  `AA_/DOM_/SEQ_` prefixes.
- **Name the top level "sequence level"** to match the `SEQ_` prefix verbatim.
  Rejected: "sequence" already means the amino-acid string everywhere in the
  glossary (`df_seq`, the `sequence` column); "protein level" names the
  biological unit without colliding, and the alias is recorded so the prefix
  spelling stays unambiguous.
- **Glossary-only, no ADR.** Rejected per the ADR test: the protocols, concept
  table, dataset docs, and future datasets all build on these names; reversing
  the level count or the protein/sequence naming later is costly, and the choice
  is non-obvious to a future reader.
- **Add relational/interaction as a 4th level now.** Rejected: AAanalysis cannot
  do long-range contacts; listing it as a level promises capability it lacks. It
  is documented as a boundary instead.

## Consequences

- The concept-overview page (#21) and the three level-based protocols (#35) cite
  this taxonomy; `CONTEXT.md` gains the matching glossary terms.
- The compositional-vs-positional CPP strategy maps onto the levels
  (compositional ≈ protein, positional ≈ residue, domain uses both) — documented
  in the CPP strategies guide.
- Per repo convention, shipped code must not reference this ADR;
  `load_dataset`'s prefix branching keeps its rationale inline.

## Out of scope

- A `strategy=` preset for compositional/positional CPP (separate enhancement
  issue).
- First-class residue-pair / bond-centered feature ergonomics (separate issue).
- Any new dataset under these prefixes.

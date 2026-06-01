# AAanalysis docstring house style — reference

Distilled from `CPP`, `dPULearn`, `AAclust` (and `CPPPlot`). These are the
canonical patterns; mirror them. Pairs with `.claude/rules/docstrings.md`.

---

## 1. Canonical class-docstring template

```python
class AAclust(Wrapper):
    """
    Amino Acid clustering (**AAclust**) class: a k-optimized clustering wrapper
    for selecting redundancy-reduced sets of numerical scales [Breimann24a]_.

    <One expanded paragraph of purpose; optionally a ``*``-bulleted breakdown
    of strategies/steps.>

    .. versionadded:: 0.1.0

    Attributes
    ----------
    labels_ : array-like, shape (n_samples)
        Cluster labels in the order of samples in ``X``.
    centers_ : array-like, shape (n_clusters, n_features)
        Average scale values corresponding to each cluster.
    """
    def __init__(self, model_class=KMeans, model_kwargs=None, verbose=True,
                 random_state=None):
        """
        Parameters
        ----------
        model_class : Type[ClusterMixin], default=KMeans
            A clustering model class with ``n_clusters`` parameter.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer,
            results of stochastic processes are consistent, enabling
            reproducibility. If ``None``, stochastic processes are truly random.

        Notes
        -----
        * <cross-cutting caveats>

        See Also
        --------
        * :class:`AAclustPlot`: the respective plotting class.

        Examples
        --------
        .. include:: examples/aaclust.rst
        """
```

Key invariants:
- Summary is a **noun phrase** (`<Full Name> (**ACRONYM**) class ...`) on the
  line *after* a blank first line, present tense, ending in a `[Key]_` citation.
- `.. versionadded::` follows the prose, before any section.
- Class docstring carries **only** `Attributes` (sklearn `_`-suffixed fit-state),
  documented as `name_ : type, shape (...)`. Stateless classes may omit it.
- **`Parameters` belong in `__init__`** — never in the class docstring. `__init__`
  always has a docstring, ordered `Parameters → Notes → See Also → Examples`.

## 2. Canonical method-docstring template

```python
    def run(self, labels=None, ...):
        """
        Perform Comparative Physicochemical Profiling (CPP) algorithm: creation
        and two-step filtering of interpretable sequence-based features.

        Parameters
        ----------
        labels : array-like, shape (n_samples,)
            Class labels for samples in sequence DataFrame (typically, test=1,
            reference=0).

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with ... <column schema as ``*`` bullets in Notes>.

        Notes
        -----
        * <behavioral elaboration; output-column schema as ``*`` bullets>

        See Also
        --------
        * :meth:`CPP.eval`: evaluate the resulting feature set.

        Examples
        --------
        .. include:: examples/cpp_run.rst
        """
```

Key invariants:
- Summary is a **verb phrase** (imperative/present); no `→`/`+`/arrow shorthand.
- Section order: `Parameters → Returns → Raises → Notes → Warnings → See Also →
  Examples` (subset allowed, order preserved). Use **`Warnings`**, not `Warns`.
- `Returns` is **named** (`df_feat`, `df_eval`, `dict_num`, or the fitted
  instance) and matches the returned variable.
- **Every public method ends with `Examples` → one `.. include:: examples/<name>.rst`.**

## 3. Recurring-parameter baselines (verbatim, method specifics as a suffix)

- `df_seq` → `DataFrame containing an ``entry`` column with unique protein
  identifiers and a ``sequence`` column with full protein sequences.`
- `labels` → `Class labels for samples in sequence DataFrame (typically, test=1,
  reference=0).`
- `n_jobs` → `Number of CPU cores (>=1) used for multiprocessing. If ``None``,
  the number is optimized automatically. If ``-1``, all cores are used.
  Overridden by ``options['n_jobs']`` when set.`
- `random_state` → see template above.

## 4. Cross-reference conventions

- Roles: `:class:` (classes), `:meth:` (methods), `:func:` (top-level functions /
  sklearn fns), `:ref:` (usage-principles pages).
- `See Also` is a `* `-bulleted list; each entry `* :role:`Target`: gloss.`
  (single colon, gloss after `: `). No bare `name : desc` numpydoc entries, no
  ` : ` (space-colon-space).
- **Plot pair is reciprocal:** logic class lists `* :class:`<X>Plot`: the
  respective plotting class.`; the Plot class summary reads `Plotting class for
  :class:`<X>` ... [Key]_.` and links back via `See Also`.

## 5. Citations

- Cite via `[Key]_` inline only. Never inline a full `.. [Key] Author, Year, ...`
  reference, a raw URL, or `(Author et al. Year)` free text inside a docstring.
- New bibliography entries go in `docs/source/index/references.rst` (see
  `docstrings.md` for the key format), then are reused by key.

## 6. Full consistency checklist

A docstring is house-style if it passes:

1. Class summary is a noun phrase `<Full Name> (**ACRONYM**) class ...`, on the
   line after a blank first line, present tense — not an imperative verb.
2. Class summary ends with a project citation `[Breimann2x]_` (Plot classes also
   carry a `:class:` backlink).
3. `.. versionadded::` follows the class prose, before any section.
4. `Attributes` (sklearn `_`-state) documented in the class docstring as
   `name_ : type, shape (...)`.
5. `Parameters` documented in `__init__`, never the class docstring; `__init__`
   always has a docstring.
6. `__init__` section order = `Parameters → Notes → See Also → Examples`.
7. Method summary is a verb phrase; no `→`/`+` shorthand.
8. Method section order = `Parameters → Returns → Raises → Notes → Warnings →
   See Also → Examples`; header is `Warnings`, not `Warns`.
9. `Returns` value is named and matches the returned variable.
10. Every public method ends with `Examples` → one `.. include:: examples/<name>.rst`.
11. Recurring params reuse the canonical baseline sentence verbatim.
12. Citations are `[Key]_` only — no full reference, raw URL, `(Author Year)`, or
    inline `.. [Key]` `References` block.
13. `See Also` is `* :role:`Target`: gloss.` bullets — no bare names, no ` : `.
14. Logic/Plot cross-links are reciprocal.
15. No imperative verb lists or arrows in any summary; elaboration and column
    schemas go under `Notes` as `*` bullets.
16. `Returns` is named (`name : type`). Two type-only idioms are allowed: a bare
    class name (sklearn self-return, e.g. `AAclust`) and a polymorphic `X or Y`.
17. `Notes` (and other) list items use `*` bullets, not `-` (`CPP` is the model;
    `dPULearn.eval` is a known `-` deviation).
18. Examples includes target `examples/<name>.rst` (no other path/extension).

The checker's `--fix` applies only the mechanical subset (`Warns` → `Warnings`,
See-Also ` : ` → `: `); the rest are author-side fixes against §1–§5 above.

## 7. Worked example — why the Preprocessors "don't feel integrated"

The newer `EmbeddingPreprocessor` / `StructurePreprocessor` /
`AnnotationPreprocessor` drift from the above. Use these as a teaching set of
what to fix (and to verify the checker catches them):

- **Summary voice** — imperative ("Preprocess ...", "Fetch, ingest, and
  encode ...") instead of the canonical noun phrase, and missing the `[Key]_`
  citation (checks 1–2).
- **Misplaced sections** — `See Also` / `Notes` / `Parameters` live in the *class*
  docstring rather than `__init__`; some `__init__`s have no docstring (checks 5–6).
- **Missing Examples** — most Preprocessor methods omit the `Examples`
  `.. include::` that every canonical method carries (check 10).
- **See Also drift** — bare method names without `:meth:`, and ` : ` (space-colon-
  space) glosses; the three siblings aren't even mutually consistent (check 13).
- **Citation violations** — inline `.. [MilliganCooper88]` / `.. [Eisen98]`
  reference blocks and free-text `(Wells et al. 2024, ...; https://...)` instead
  of `[Key]_` (check 12).
- **Summary shorthand** — `... → ``dict_dssp``` arrows in method summaries (check 7).
- **`Warns` vs `Warnings`** — `EmbeddingPreprocessor` uses `Warns` (check 8).
- **Baseline drift** — `df_seq` re-described as `DataFrame with ``entry`` +
  ``sequence`` columns.` instead of the canonical baseline (check 11).

.. _docstring_guide:

Docstring Style Guide
=====================

This is the **single source of truth** for how docstrings are written in
AAanalysis. The mature classes :class:`~aaanalysis.CPP`,
:class:`~aaanalysis.AAclust`, and :class:`~aaanalysis.dPULearn` are the gold
standard — every public symbol should read as if written by the same hand.

The guide is *enforced* by an internal checker
(``.claude/skills/docstrings/scripts/check_docstrings.py``); run it
before and after touching any docstring. All other docstring notes in the
repository (the always-on ``.claude/rules/docstrings.md``, the
``docstrings`` skill, and the *Docstring Style* section of
``CONTRIBUTING``) are thin pointers to this page and define no rules of their
own.

Style basis: `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`__
+ `PEP 257 <https://peps.python.org/pep-0257/>`__. Sections appear in the
numpydoc order ``Parameters → Returns → Raises → Notes → Warnings → See Also →
References → Examples``.

Class docstrings
----------------

.. code-block:: python

    class AAclust(Wrapper):
        """
        Amino Acid clustering (**AAclust**) class: a k-optimized clustering wrapper
        for selecting redundancy-reduced sets of numerical scales [Breimann24a]_.

        <One expanded paragraph of purpose; optionally a ``*``-bulleted breakdown.>

        .. versionadded:: 0.1.0

        Attributes
        ----------
        labels_ : array-like, shape (n_samples,)
            Cluster labels in the order of samples in ``X``.
        """
        def __init__(self, model_class=KMeans, verbose=True, random_state=None):
            """
            Parameters
            ----------
            random_state : int, optional
                The seed used by the random number generator. If a positive
                integer, results of stochastic processes are consistent, enabling
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

Invariants:

* Summary is a **noun phrase** (``<Full Name> (**ACRONYM**) class ...``) on the
  line *after* a blank first line, present tense; not an imperative verb.
* **Citations are the exception, not the default.** A class earns a ``[Key]_``
  citation only when it is an **important class** that a specific reference
  describes — its own paper, or the project paper ``[Breimann25]_`` for the core
  γ-secretase CPP / dPULearn / TreeModel algorithms it covers. **Most classes —
  every data-prep / utility / helper class (preprocessors, loaders,
  :class:`~aaanalysis.NumericalFeature`, :class:`~aaanalysis.AALogo`, …) — carry no citation, and that is correct,
  not a gap.** Never add one to satisfy a checker note. **Verify before adding,
  and never invent one:** the key must be defined in ``references.rst`` (the
  checker's ``CITATION-UNDEFINED`` flags typo'd / fabricated keys) *and* the cited
  work must actually describe this class. ``CLASS-NO-CITATION`` is **advisory** — a
  reminder to confirm an *important* class isn't missing its citation, **not** a
  prompt to cite utilities; a wrong citation is worse than none.
* ``.. versionadded::`` follows the prose, before any section.
* The class docstring carries **only** ``Attributes`` (scikit-learn ``_``-suffixed
  fit-state), documented as ``name_ : type, shape (...)``. Stateless classes omit it.
* **Parameters belong in** ``__init__`` — never in the class docstring. ``__init__``
  always has a docstring, ordered ``Parameters → Notes → See Also → Examples``.
* **Plot pairs are reciprocal:** a Plot class summary reads
  ``Plotting class for :class:`<X>` ... [Key]_.`` and the logic/plot classes link
  each other under ``See Also``.

Method & function docstrings
----------------------------

.. code-block:: python

    def run(self, labels=None, ...):
        """
        Perform Comparative Physicochemical Profiling (CPP): creation and two-step
        filtering of interpretable sequence-based features.

        Parameters
        ----------
        labels : array-like, shape (n_samples,)
            Class labels for samples in sequence DataFrame (typically, test=1,
            reference=0).

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame; output-column schema as ``*`` bullets in Notes.

        See Also
        --------
        * :meth:`CPP.eval`: evaluate the resulting feature set.

        Examples
        --------
        .. include:: examples/cpp_run.rst
        """

Invariants:

* Summary is a **verb phrase** (imperative/present); no ``→`` / ``+`` / arrow
  shorthand.
* **Summary + description.** The one-line summary is followed (after a blank
  line) by a short **plain-language description** — *what it does* in simple
  words, the cited tool/method ``[Key]_`` if it integrates one, and the key
  ``:role:`` cross-references — before the first section. The same holds for
  classes (the expanded paragraph after the noun-phrase summary). Trivial
  one-line accessors may keep just the summary; the checker's ``SUMMARY-ONLY``
  is **advisory**, a prompt to add the description where it helps.
  Write the description as **natural flowing prose**; do not prefix it with a
  bold rhetorical label that names the meta-idea (``**Mental model.**``,
  ``**When to use.**``, ``**What it returns.**``) — state the content directly.
  (Bold for genuine emphasis or a structural ``*``-bullet label in ``Notes`` is
  fine; this rule is only about replacing the explanatory paragraph with a
  bold heading.)
* **Expand abbreviations on first use.** The first time an abbreviation or
  acronym appears in a docstring, spell it out with the short form in
  parentheses — *Command Line Interface (CLI)*, *Position Weight Matrix (PWM)*,
  *Find Individual Motif Occurrences (FIMO)* — and use the short form
  afterwards. Each docstring is **self-contained**, so re-introduce the term in
  every docstring that uses it. The bold ``(**ACRONYM**)`` in a class summary is
  the class-level form of this rule. Domain terms that have a ``.. glossary::``
  entry may instead be linked with ``:term:``. Universally standard forms
  (e.g. DNA, 3D, ID, CPU, PDB) need no expansion.
* ``Returns`` is **named** (``name : type``) and matches the returned variable.
  Two type-only idioms are allowed: a bare class name (scikit-learn self-return,
  e.g. :class:`~aaanalysis.AAclust`) and a polymorphic ``X or Y``.
* **Fixed-option parameters use** ``Literal``, **not** ``str``. When a parameter
  accepts a closed, finite set of string options (the values a ``check_str_options``
  /membership check validates against), type-hint it in the *signature* as
  ``typing.Literal["a", "b", ...]`` rather than a bare ``str`` — so the allowed set
  is self-documenting and IDE-checkable. Spell the members as **inline string
  literals**: ``Literal`` cannot reference the ``ut.X`` constants, even though the
  runtime validator still uses them. If the parameter also accepts ``None`` (default
  ``None`` or ``accept_none=True``), wrap as ``Optional[Literal[...]]`` (never put
  ``None`` inside ``Literal``); if it also accepts non-string values, use
  ``Union[Literal[...], <other>]``. In the *docstring*, type the parameter with
  numpydoc set notation matching the members —
  ``name : {'remove', 'keep', 'gap'}, default='remove'`` — not ``name : str``.
  Open or large sets (e.g. system font names) stay ``str``.
* **Explain each option as a nested bullet.** When the options are not
  self-evident from their names, follow the set-notation type line with a short
  lead-in and a nested ``-`` bullet list — **one bullet per option**, each naming
  the option value (in double backticks) followed by ``:`` and a concise gloss,
  e.g. an option ``'remove'`` documented as *"drop sequences with non-canonical
  amino acids"*. Keep the gloss to what the option *is* / when to pick it; defer
  fuller behaviour to ``Notes`` and point there with ``(see Notes)``. These
  per-option enumerations are the **exception to the ``*``-bullet rule below**:
  option sub-lists under a parameter use ``-`` (matching ``name`` /
  ``non_canonical_aa`` / ``logo_type``), while ``Notes`` and ``See Also`` section
  bullets use ``*``.
* The section header is ``Warnings``, never ``Warns``.
* List items use ``*`` bullets, not ``-``.
* **Every public method ends with** ``Examples`` → exactly one
  ``.. include:: examples/<name>.rst`` (no other path or extension).

Recurring parameters (DRY)
--------------------------

Parameters that appear in many signatures share **one baseline sentence**;
method-specific behaviour is a *suffix*, never a replacement. Describe the
structure first, the use second.

.. code-block:: text

    df_seq        DataFrame containing an ``entry`` column with unique protein
                  identifiers and a ``sequence`` column with full protein sequences.
    labels        Class labels for samples in sequence DataFrame (typically,
                  test=1, reference=0).
    n_jobs        Number of CPU cores (>=1) used for multiprocessing. If ``None``,
                  the number is optimized automatically. If ``-1``, all cores are
                  used. Overridden by ``options['n_jobs']`` when set.
    random_state  The seed used by the random number generator. If a positive
                  integer, results are reproducible; if ``None``, truly random.

To keep these from drifting, define each baseline **once** and inject it with a
pandas/seaborn-style mechanism (a ``_shared_docs`` dict + a ``@doc(...)`` /
``Substitution`` decorator) rather than re-typing it. This is the adopted target;
where injection is not yet wired, copy the baseline **verbatim**.

**A docstring is self-contained — document every parameter as its own entry.**
DRY means reusing the same *sentence*, not collapsing parameters into a single
cross-method reference. Never write a lumped entry like
``labels, n_filter, n_jobs, ... : See :meth:`run`. Same semantics.`` — repeat each
parameter (with its baseline sentence) so the reader never has to open another
method to learn what an argument does. The checker's ``PARAM-UNDOCUMENTED`` flags
the signature parameters such a lump leaves effectively undocumented.

Cross-references (``See Also``)
-------------------------------

* Roles: ``:class:`` (classes), ``:meth:`` (methods), ``:func:`` (top-level /
  scikit-learn functions), ``:ref:`` (usage-principles pages), ``:term:``
  (glossary terms, see below).
* ``See Also`` is a ``*``-bulleted list; each entry is
  ``* :role:`Target`: gloss.`` — single colon, gloss after ``: ``. No bare
  ``name : desc`` numpydoc entries and no ``  :  `` (space-colon-space).
* **Every cross-reference must resolve.** A ``:class:``/``:meth:``/``:func:``
  target (in ``See Also`` *or* inline prose) must name a real public symbol —
  :class:`~aaanalysis.CPP`, :meth:`~aaanalysis.CPP.run_num`, ``aaanalysis.combine_dict_nums``. Watch
  capitalization (:class:`~aaanalysis.AALogo`, not ``AALogo``) and method names on the right
  class. The checker's ``XREF-UNRESOLVED`` flags an internal target that does
  not resolve; external refs (``pandas.DataFrame``) are left alone.
* **Order multi-layer links by documentation layer.** When a ``See Also`` (or
  inline prose) links out to other documentation layers, reference them in the
  order Usage Principles → Tables → Tutorials. Add an **external-library**
  reference only when absolutely necessary.

Citations
---------

* Cite inline with ``[Key]_`` **only**. Never inline a full ``.. [Key] Author,
  Year, ...`` reference, a raw URL, or ``(Author et al. Year)`` free text.
* All bibliography entries live in ``docs/source/index/references.rst``, grouped
  by topic. Key format: single first-author ``[AuthorYY]`` (``[Song12]``),
  two-author ``[FirstSecondYY]`` (``[ElkanNoto08]``), same author/year gets a
  trailing letter (``[Breimann24a]``).
* Pick the **few most relevant** references per symbol — 1–2 per major method,
  plus the project paper (``[Breimann25]_``) at the class level **only for the
  classes that paper actually describes** (the core CPP / dPULearn / TreeModel
  pipeline). It is not a default stamp: a class the cited work does not cover
  carries no class-level citation (see the class-summary rule above).
* **Every external tool / method AAanalysis integrates must be cited and
  explained.** When a method wraps or runs an external tool (DSSP, Chainsaw,
  Merizo, AFragmenter, MEME / FIMO, cd-hit, mmseqs2, logomaker, SHAP, …), name it
  with a ``[Key]_`` citation (its paper, defined in ``references.rst`` — verify it
  exists, never a bare repo URL) **and** describe what the tool does in **one
  plain sentence**. Example: ``'chainsaw' ([Wells24]_): a fully-convolutional
  neural network that predicts domain boundaries from a PDB / CIF structure.``
* **Never reference internal decision records (ADRs) in docstrings.** ADRs
  (``docs/adr/``) are internal to the development process; docstrings are
  user-facing. If an ADR documents a user-visible choice (why a parameter exists,
  how an algorithm was selected), extract the *why* as plain language in the
  docstring — never cite the ADR number. (Developer-facing notes such as
  ``CONTEXT.md`` may cite ADRs; user docstrings may not.)

Versioning & deprecation
------------------------

* ``.. versionadded:: X.Y.Z`` (true first-release version) on every public class
  and function; ``.. versionchanged:: X.Y.Z`` when behaviour changes.
* **Parameter-level directives** — when a *parameter or option* is added/changed
  after its class, annotate it inside the parameter description:

  .. code-block:: text

      return_stats : bool, default=False
          If ``True``, also return the filter-funnel statistics.

          .. versionadded:: 1.1.0

* **Deprecation** uses ``.. deprecated:: X.Y.Z`` in the docstring plus a
  ``DeprecationWarning`` shim (see :ref:`api-stability <usage_principles>`); a
  renamed/removed public symbol keeps a one-minor-release shim before removal.

Class abbreviations
-------------------

Every public class has **one canonical abbreviation**, used identically as the
example-notebook instance variable (``aac = aa.AAclust()``) and the
example-notebook filename stem (``examples/feature_engineering/aac_fit.ipynb``).
This keeps the API, the example notebooks, and the rendered docs in lock-step.
The registry below is the single source of truth and is enforced by
``tests/unit/api_tests/test_class_abbreviation_registry.py`` (every public class
is registered; every ``<abbr> = aa.<Class>()`` and notebook filename matches).

How an abbreviation is formed (in priority order):

* **Keep the recognizable name prefix.** ``AA*`` classes keep ``aa``; ``Seq*``
  classes keep ``seq``; a genuine acronym stays whole (:class:`~aaanalysis.CPP` →
  ``cpp``); an established public spelling is kept (:class:`~aaanalysis.dPULearn` →
  ``dpul``).
* **Then add the first letter of each remaining CamelCase word.** So
  :class:`~aaanalysis.AAWindowSampler` → ``aaws``, :class:`~aaanalysis.SequenceFeature`
  → ``sf``, :class:`~aaanalysis.NumericalFeature` → ``nf``,
  :class:`~aaanalysis.TreeModel` → ``tm``, :class:`~aaanalysis.ShapModel` → ``sm``,
  :class:`~aaanalysis.AAclust` → ``aac``, :class:`~aaanalysis.AALogo` → ``aal``,
  :class:`~aaanalysis.AAMut` → ``aam``, :class:`~aaanalysis.AAPred` → ``aap``,
  :class:`~aaanalysis.SeqMut` → ``seqm``, :class:`~aaanalysis.SeqOpt` → ``seqo``.
* **Extend a component when the minimal initial is too generic or would collide.**
  The ``*Preprocessor`` family therefore uses a three-letter first-word stem plus
  ``p`` — :class:`~aaanalysis.SequencePreprocessor` → ``seqp``,
  :class:`~aaanalysis.EmbeddingPreprocessor` → ``embp``,
  :class:`~aaanalysis.StructurePreprocessor` → ``strp``,
  :class:`~aaanalysis.AnnotationPreprocessor` → ``annp`` — because two-letter
  initials (``sp``/``ep``) are too generic and clash. On a direct collision the
  *newer* class takes the longer form: :class:`~aaanalysis.SeqMut` is ``seqm``,
  leaving ``sm`` free for :class:`~aaanalysis.ShapModel`.
* **Reserved:** ``ap`` is the import alias for :mod:`aaanalysis.pipe`
  (``import aaanalysis.pipe as ap``) and is never a class abbreviation.
* A plot pair is the base abbreviation plus ``_plot`` (:class:`~aaanalysis.CPPPlot` → ``cpp_plot``).
* **A class instance is named the bare abbreviation, always** — ``cpp = aa.CPP(...)``,
  never ``cpp_res``/``cpp_dom``. If you build the same class repeatedly (e.g. one
  CPP per prediction level), **reassign the bare name** and let the *outputs* carry
  the qualifier (``df_feat_res``, ``X_res``). A ``<abbr>_<qualifier>`` *instance*
  name is allowed **only** for a genuinely concurrent second instance that cannot
  be restructured (``aaws_strict`` beside ``aaws``) — never an unrelated word.

**Type-suffix mnemonic.** Four class families encode their *kind* in the last
letter of the abbreviation, so a reader can tell what a variable holds at a
glance. The remaining tool / engine classes keep the prefix + concept-initial
form above (``cpp``, ``aac``, ``aam``, ``seqm``, ``seqo``, ``aap``, ``aaws``,
``dpul``).

.. list-table:: Abbreviation type-suffix map
   :header-rows: 1
   :widths: 12 40 48

   * - Suffix
     - Family
     - Members
   * - ``*p``
     - preprocessors
     - ``seqp``, ``embp``, ``strp``, ``annp``
   * - ``*f``
     - feature containers
     - ``sf``, ``nf``
   * - ``*m``
     - models (fit / produce a model or estimate)
     - ``tm``, ``rm``, ``sm``
   * - ``*x``
     - XAI explainers (post-hoc; *planned* — see *Reserved future abbreviations*)
     - ``fx``, ``cx``, ``ex``, ``rx``, ``nx``, ``sx``
   * - ``*_plot``
     - plot companions
     - ``cpp_plot``, ``aap_plot``, ``seqo_plot``

.. list-table:: Canonical class abbreviations
   :header-rows: 1
   :widths: 35 20 25

   * - Class
     - Abbr.
     - Extra
   * - :class:`~aaanalysis.SequencePreprocessor`
     - ``seqp``
     -
   * - :class:`~aaanalysis.EmbeddingPreprocessor`
     - ``embp``
     -
   * - :class:`~aaanalysis.AALogo` / :class:`~aaanalysis.AALogoPlot`
     - ``aal`` / ``aal_plot``
     -
   * - :class:`~aaanalysis.AAWindowSampler`
     - ``aaws``
     -
   * - :class:`~aaanalysis.AAclust` / :class:`~aaanalysis.AAclustPlot`
     - ``aac`` / ``aac_plot``
     -
   * - :class:`~aaanalysis.SequenceFeature`
     - ``sf``
     -
   * - :class:`~aaanalysis.NumericalFeature`
     - ``nf``
     -
   * - :class:`~aaanalysis.CPP` / :class:`~aaanalysis.CPPPlot`
     - ``cpp`` / ``cpp_plot``
     -
   * - :class:`~aaanalysis.CPPGrid`
     - ``cppg``
     -
   * - :class:`~aaanalysis.CPPStructurePlot`
     - ``cpps_plot``
     - ``pro``
   * - :class:`~aaanalysis.dPULearn` / :class:`~aaanalysis.dPULearnPlot`
     - ``dpul`` / ``dpul_plot``
     -
   * - :class:`~aaanalysis.AAMut` / :class:`~aaanalysis.AAMutPlot`
     - ``aam`` / ``aam_plot``
     -
   * - :class:`~aaanalysis.SeqMut` / :class:`~aaanalysis.SeqMutPlot`
     - ``seqm`` / ``seqm_plot``
     -
   * - :class:`~aaanalysis.SeqOpt` / :class:`~aaanalysis.SeqOptPlot`
     - ``seqo`` / ``seqo_plot``
     - core (``mode="impact"`` needs ``pro``)
   * - :class:`~aaanalysis.TreeModel`
     - ``tm``
     -
   * - :class:`~aaanalysis.AAPred` / :class:`~aaanalysis.AAPredPlot`
     - ``aap`` / ``aap_plot``
     -
   * - :class:`~aaanalysis.ReliabilityModel` / :class:`~aaanalysis.ReliabilityModelPlot`
     - ``rm`` / ``rm_plot``
     -
   * - :class:`~aaanalysis.ShapModel`
     - ``sm``
     - ``pro``
   * - :class:`~aaanalysis.StructurePreprocessor`
     - ``strp``
     - ``pro``
   * - :class:`~aaanalysis.AnnotationPreprocessor`
     - ``annp``
     - ``pro``

Reserved future abbreviations
-----------------------------

The v2 Explainability (XAI) layer will add one ``*Explainer`` class per method
category, each wrapping several library approaches as methods (one method per
approach). Their abbreviations follow the ``*x`` (e\ **x**\ plainer) suffix and
are **reserved now** so they are not reused before the classes land. The shipped
``*Model`` wrappers — :class:`~aaanalysis.TreeModel` (``tm``),
:class:`~aaanalysis.ReliabilityModel` (``rm``), :class:`~aaanalysis.ShapModel`
(``sm``, kept as the legacy SHAP wrapper) — stay unchanged.

.. list-table:: Reserved XAI ``*Explainer`` abbreviations (planned, not yet implemented)
   :header-rows: 1
   :widths: 26 10 16 48

   * - Class (planned)
     - Abbr.
     - Scope
     - Wraps (one method per approach)
   * - ``FeatureExplainer``
     - ``fx``
     - ``pro``
     - permutation importance, PDP, ICE, ALE, LIME
   * - ``ExampleExplainer``
     - ``ex``
     - core / ``pro``
     - prototypes (AAclust medoids, MMD-CRITIC), counterfactuals (Wachter, DiCE, CEM)
   * - ``RuleExplainer``
     - ``rx``
     - ``pro``
     - ANCHOR, RuleFit, LORE, TREPAN
   * - ``NeuralExplainer``
     - ``nx``
     - ``pro``
     - integrated gradients, LRP, Grad-CAM, TCAV (Captum)
   * - ``SurrogateExplainer``
     - ``sx``
     - core / ``pro``
     - tree / linear / symbolic distillation of a fitted model
   * - ``CausalExplainer``
     - ``cx``
     - ``pro``
     - DoWhy, EconML

Each planned class gets a plot companion (``fx_plot``, ``ex_plot``, …). The
PascalCase pass for :class:`~aaanalysis.AALogo` / :class:`~aaanalysis.AALogoPlot`
(abbreviations ``aal`` / ``aal_plot`` unchanged) is **done**; the former
``AAlogo`` / ``AAlogoPlot`` names remain importable as deprecated back-compatible
aliases (resolved via ``aaanalysis.__getattr__`` with a ``DeprecationWarning``).

Tutorial notebook titles
------------------------

A tutorial notebook's first ``#`` heading **is** its rendered page title and the
sidebar link text, so the titles must stay consistent across the tutorial set. A
tutorial centred on a single public class is titled

   ``<ClassName>: <capability phrase>``

— the **exact public class name** (as in the API), a colon, then a short phrase
for what the reader learns. Examples already in the set:
:doc:`AAclust: Selecting redundancy-reduced scale sets </generated/tutorial3a_aaclust>`,
:doc:`SequenceFeature: Creation of CPP feature components </generated/tutorial3b_sequence_feature>`,
:doc:`CPP: Identification of physicochemical signatures </generated/tutorial3c_cpp>`,
:doc:`dPULearn: Learning from unbalanced data </generated/tutorial4a_dpulearn>`,
:doc:`ShapModel: Explaining with single-residue resolution </generated/tutorial5a_shap_model>`,
:doc:`CPPGrid: Sweeping, evaluating, and ranking configurations </generated/tutorial6_comparison_harness>`,
:doc:`SeqOpt: Optimizing sequences by directed evolution </generated/tutorial7_protein_engineering>`.

Rules:

* **No ``Tutorial`` word or number in the title** (never ``Tutorial 6 — …`` or
  ``Tutorial: …``). Ordering lives in the filename (``tutorial6_*``) and the
  ``tutorials.rst`` toctree, not the heading.
* When a section of ``tutorials.rst`` introduces the notebook in prose, name it by
  the **same class** (the :doc:`CPPGrid tutorial </generated/tutorial6_comparison_harness>`,
  the :doc:`SeqOpt tutorial </generated/tutorial7_protein_engineering>`) so
  the prose, the title, and the API all read the same.
* A tutorial that teaches a **function** or a **cross-cutting topic** rather than
  one class uses a plain descriptive title — :doc:`Data loading </generated/tutorial2a_data_loader>`,
  :doc:`Scale loading </generated/tutorial2b_scales_loader>`,
  :doc:`CPP across data representations </generated/tutorial3d_data_representations>`.
* The onboarding notebooks under :ref:`Getting Started <getting_started>`
  (:doc:`A minimal CPP analysis </generated/tutorial0_minimal>`,
  :doc:`Quick start with AAanalysis </generated/tutorial1_quick_start>`,
  :doc:`Slow start with AAanalysis </generated/tutorial1_slow_start>`) are titled by what they deliver, not a class.

Output / data-object names
--------------------------

The objects passed *between* steps have canonical names too — most are defined in
the project glossary (``CONTEXT.md``). Use these consistently so a snippet reads
the same everywhere; this table is a reference, **not** a test-enforced gate (only
the *class-instance* names above are checked).

.. list-table:: Canonical data-object variable names
   :header-rows: 1
   :widths: 22 45

   * - Variable
     - Object (producer)
   * - ``df_seq``
     - sequence frame (:func:`~aaanalysis.load_dataset`, :func:`~aaanalysis.read_fasta`)
   * - ``df_scales`` / ``df_cat``
     - AAontology scales / scale categories (:func:`~aaanalysis.load_scales`, ``build_cat``)
   * - ``df_parts``
     - assembled parts (``sf.get_df_parts``)
   * - ``split_kws``
     - split specification (``sf.get_split_kws``) — matches the ``split_kws=`` parameter
   * - ``df_feat``
     - feature frame, canonical schema (``cpp.run``, :func:`~aaanalysis.load_features`, ``sm.add_*``)
   * - ``X`` / ``labels``
     - feature matrix (``sf.feature_matrix``) / class labels (``df_seq["label"].to_list()``)
   * - ``df_eval``
     - evaluation results (``cpp/tm/dpul/aam/seqm .eval(...)``)
   * - ``df_pos`` / ``feat_importance``
     - feature positions (``sf.get_df_pos``) / importance column (``tm.add_feat_importance``)
   * - ``df_logo`` / ``df_logo_info``
     - sequence-logo frames (``aal.get_df_logo`` / ``get_df_logo_info``)
   * - ``df_impact`` / ``df_scan``
     - mutation impact (``aam.run``) / scan (``seqm.scan``)
   * - ``df_pu`` / ``dict_num`` / ``df_annot`` / ``df_params``
     - PU frame (``dpul``) / numerical parts (``nf``) / annotations (``annp``) / grid params (``cppg``)

**Qualifiers belong on the data level.** A variant of a data object takes a
``<name>_<qualifier>`` suffix (``df_feat_res``, ``X_res``, ``df_cat_selected``,
``df_top15``) — used **only when you actually have a variant**, not stamped onto
every example. Class instances stay the bare abbreviation (see above).

.. _kws-overview:

Keyword-dict (``_kws``) parameters
----------------------------------

A ``*_kws`` parameter groups several related keyword arguments into one dict.
There are two **kinds**, and the *forwarded* kind splits into two **subtypes**:

* **forwarded** — passed through (mostly) verbatim to another callable, so it
  accepts whatever that callable accepts (open-ended keys):

  * *external* — the target is a third-party library (e.g. Matplotlib);
  * *internal* — the target is another AAanalysis function or method.

* **structured bundle** — a **fixed, documented key set** consumed *inside*
  AAanalysis (validate unknown keys, default the unset ones).

Name any new one ``<noun>_kws`` and add it to the table below.

.. list-table:: Keyword-dict (``_kws``) parameters
   :header-rows: 1
   :widths: 22 22 56

   * - Parameter
     - Kind
     - Contents / target
   * - ``cbar_kws``
     - forwarded, external
     - Arbitrary keys for :meth:`matplotlib.figure.Figure.colorbar`.
   * - ``legend_kws``
     - forwarded, internal
     - Arbitrary keys for :func:`~aaanalysis.plot_legend`.
   * - ``feature_map_kws``
     - forwarded, internal
     - Arbitrary keys for :meth:`~aaanalysis.CPPPlot.feature_map` (from ``CPPStructurePlot``).
   * - ``aal_kws`` / ``list_aal_kws``
     - forwarded, internal
     - Keys for :meth:`~aaanalysis.AALogo.get_df_logo` / ``get_df_logo_info`` (``list_`` = one dict per logo).
   * - ``df_parts_kws``
     - forwarded, internal
     - Keys for ``sf.get_df_parts`` (e.g. ``list_parts``, ``jmd_n_len``) when *building* ``df_parts``.
   * - ``split_kws``
     - structured bundle
     - One param-dict per split type: ``Segment`` (``n_split_min``, ``n_split_max``), ``Pattern``
       (``steps``, ``n_min``, ``n_max``, ``len_max``), ``PeriodicPattern`` (``steps``). From ``sf.get_split_kws``.
   * - ``bootstrap_kws``
     - structured bundle
     - Fixed keys ``rounds`` / ``resample`` / ``frac`` (any subset; unset keys keep their default) —
       :meth:`~aaanalysis.CPP` bootstrap configuration.
   * - ``sample_kws``
     - structured bundle
     - Fixed keys ``sample`` (an entry name / id / accession ``str`` or a row-position ``int``),
       ``df_seq`` and ``df_parts`` — selects one sample for :meth:`~aaanalysis.CPPPlot.feature_map` /
       :meth:`~aaanalysis.CPPPlot.heatmap` / :meth:`~aaanalysis.CPPPlot.profile`, resolving its TMD-JMD
       sequence (and per-sample ``feat_impact`` column) from ``df_parts``. The bundled alternative to
       passing ``tmd_seq`` / ``jmd_n_seq`` / ``jmd_c_seq`` directly.

Label parameter names
---------------------

Each labeling **concept** has one canonical parameter name, used consistently
across the classes a user combines in one workflow. Several names look similar
but name *different* concepts — keep them distinct rather than collapsing them.

.. list-table:: Canonical label parameter names
   :header-rows: 1
   :widths: 28 20 52

   * - Concept
     - Canonical name
     - Notes
   * - Contrast markers (the two groups being compared)
     - ``label_test`` / ``label_ref``
     - The positive/test group vs the reference group of a *contrast* —
       :meth:`~aaanalysis.CPP.run` / :meth:`~aaanalysis.CPP.eval`, :meth:`~aaanalysis.AALogo.get_df_logo`.
   * - Single labeling (1D)
     - ``labels``
     - One per-sample class-label vector, shape ``(n_samples,)`` — e.g.
       :meth:`~aaanalysis.CPP.run`, :meth:`~aaanalysis.TreeModel.fit` / ``eval``, :meth:`~aaanalysis.ShapModel.fit`,
       :meth:`~aaanalysis.dPULearn.fit`.
   * - List of labelings (2D, multi-dataset)
     - ``list_labels``
     - Several labelings stacked, shape ``(n_datasets, n_samples)``, paired with
       ``names_datasets`` — :meth:`~aaanalysis.AAclust.eval`, :meth:`~aaanalysis.dPULearn.eval`. Distinct from
       ``labels`` **on purpose**: the plural marks a list of datasets, not one.
   * - Target-class selector (a single class to attribute)
     - ``label_target_class``
     - :meth:`~aaanalysis.ShapModel.fit` — the one class whose model prediction the SHAP values
       explain. It can be **any** class (the positive, the negative/reference,
       or any index in a multi-class model), so it is **not** the same concept as
       ``label_test`` and deliberately keeps its own name.

Examples & verification
-----------------------

* Examples are authored as notebooks under ``examples/<subpackage>/`` and pulled
  in with ``.. include:: examples/<name>.rst`` (converted at docs-build time).
* Keep them small, seeded, and deterministic. Example notebooks **and** tutorials
  are executed in CI (``pytest --nbmake examples/ tutorials/``) so they cannot rot;
  tiny self-contained snippets may additionally use ``>>>`` doctests run with
  ``--doctest-modules``.
* **Commit notebooks with their executed outputs.** The docs render the *stored*
  cell outputs (``nbsphinx_execute = 'never'`` in ``conf.py``), and
  ``create_notebooks_docs.py`` converts each notebook to ``.rst`` from those saved
  outputs. A cell with no saved output renders **no figure or table** on Read the
  Docs — even though the blocking CI (which only checks that cells *run*) stays
  green, so the gap is invisible until you look at the built page. After editing
  any example/tutorial cell (including programmatic ``NotebookEdit``, which clears
  the cell's outputs), re-run the whole notebook and save it with outputs, e.g.
  ``jupyter nbconvert --to notebook --execute --inplace
  examples/<subpackage>/<name>.ipynb``, then confirm the figures are embedded
  before committing.

Notebook content & structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An example notebook *teaches* the symbol, it does not merely call it. Use this order:

1. **Concept first** (opening markdown cell). In natural prose *before* any
   code, explain what the method/class does, when and why to use it, and what
   it returns. Write it as flowing text — do **not** prefix the paragraphs with
   bold rhetorical labels (``**Mental model.**``, ``**When to use.**``,
   ``**What it returns.**``); state the content directly.
2. **Minimal example** (code cell). The smallest seeded, runnable call, with output.
3. **Parameter walkthrough** (markdown + code). Introduce **every** public parameter,
   with **one cell per parameter group**: parameters that belong together share a
   cell (e.g. ``jmd_n_len`` / ``jmd_c_len``, or a family of ``max_*`` thresholds).
   Each group gets a short markdown note (what it controls) and a code cell showing
   its effect on the result. **No parameter may be left uncovered.**
4. Show output so the docs render it; keep every cell small, seeded, deterministic.
   The notebook must be committed **with** its executed outputs (figures + tables) —
   see *Examples & verification* above.

Glossary cross-links
--------------------

Domain terms (``df_seq``, ``dict_num``, ``pseudo-scale``, ``entry``, ...) are
defined once in a Sphinx ``.. glossary::`` (sourced from the project glossary) and
referenced from docstrings via ``:term:`dict_num``` so a reader can click through
to the canonical definition.

Math
----

Render formulas with ``.. math::`` (or inline ``:math:`...```) rather than ASCII,
e.g. in the ``metrics`` functions (AUC*, BIC, KLD).

Prose & punctuation
-------------------

Keep punctuation plain across all documentation (docstrings *and* ``.rst``
pages), not only this guide.

* **Avoid em dashes (``—``).** They read as a tic when overused. Prefer a
  **colon** to introduce a label, list, or elaboration
  (``**Overview**: New to AAanalysis ...``, ``four sections: ...``) and a
  **comma** for an aside or appositive
  (``... signatures, the features that distinguish ...``).
* Reserve the em dash for the rare genuine break in thought; never use it as a
  default connector or as bullet-label punctuation.

Conformance checklist
---------------------

A docstring is house-style if every applicable item holds. The right column is
the code emitted by the internal checker. The checker separates **defects**
(hard violations — the run fails only on these) from **advisory** notes
(``CLASS-NO-CITATION`` — never fails, since utility classes legitimately omit a
citation), and **skips UNDER CONSTRUCTION stubs** entirely (a class whose summary
starts ``UNDER CONSTRUCTION``, or a method whose body is just
``raise NotImplementedError``). ``0 defect(s)`` therefore means the convention is
satisfied for every implemented public symbol.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Rule
     - Checker code
   * - Class summary is a noun phrase (not a verb)
     - ``CLASS-SUMMARY-VERB``
   * - Class summary ends with a ``[Key]_`` citation *when a matching reference exists* (advisory)
     - ``CLASS-NO-CITATION``
   * - Every ``[Key]_`` is defined in ``references.rst`` (no typo'd / invented keys)
     - ``CITATION-UNDEFINED``
   * - ``.. versionadded::`` present
     - ``NO-VERSIONADDED``
   * - ``Parameters`` in ``__init__``, not the class docstring
     - ``CLASS-HAS-PARAMETERS``
   * - ``__init__`` has a docstring
     - ``INIT-NO-DOCSTRING``
   * - Public symbol has a docstring
     - ``MISSING-DOCSTRING``
   * - Method summary has no arrow shorthand
     - ``SUMMARY-ARROW``
   * - ``Returns`` is named (``name : type``)
     - ``RETURNS-UNNAMED``
   * - Public method ends with an ``Examples`` include
     - ``METHOD-NO-EXAMPLES``
   * - Recurring params reuse the baseline sentence
     - ``DFSEQ-BASELINE``
   * - Citations use ``[Key]_`` only
     - ``INLINE-CITATION`` / ``FREETEXT-CITATION``
   * - ``See Also`` entries are role bullets
     - ``SEEALSO-NO-BULLET``
   * - ``See Also`` gloss uses colon-space, not space-colon-space
     - ``SEEALSO-SPACE-COLON``
   * - ``Warnings`` header (not ``Warns``)
     - ``WARNS-SECTION``
   * - ``Notes`` list items use ``*`` not ``-``
     - ``NOTES-DASH-BULLET``
   * - Every ``[Key]_`` is defined in ``references.rst``
     - ``CITATION-UNDEFINED``
   * - ``:class:``/``:meth:``/``:func:`` targets resolve to a real public symbol
     - ``XREF-UNRESOLVED``
   * - A body that raises documents a ``Raises`` section *(advisory)*
     - ``RAISES-UNDOCUMENTED``
   * - Summary is followed by a plain-language description *(advisory)*
     - ``SUMMARY-ONLY``

The build itself is the final gate: ``cd docs && make html`` (ideally with
``SPHINXOPTS="-W"``) must finish without warnings — broken section underlines,
inline-literal RST errors, and unresolved ``.. include::`` targets surface only
there, not in the structural checker.

Tooling
-------

.. code-block:: bash

    # audit the whole package (or a single file)
    python .claude/skills/docstrings/scripts/check_docstrings.py aaanalysis/
    # auto-fix the safe mechanical drift (Warns→Warnings, See-Also colon spacing)
    python .claude/skills/docstrings/scripts/check_docstrings.py --fix aaanalysis/

``--fix`` applies only the mechanical subset; every other finding is an
author-side fix against the templates above.

API reference order
-------------------

The API reference (``docs/source/api.rst``) is ordered to read as the analysis
**pipeline**, not alphabetically. A newly integrated public symbol must be slotted
in by these rules:

#. **Sections follow the workflow:** Data Handling → Sequence Analysis → Feature
   Engineering → PU Learning → Explainable AI → Protein Engineering → Utility Functions
   (data in → analyse → engineer features → model → design → helpers).
#. **Within a section, follow data flow:** inputs/loaders first, then the
   processing classes, then combiners/outputs (Data Handling = loaders →
   preprocessors → :func:`~aaanalysis.combine_dict_nums`).
#. **Parallel-modality families go sequence → structure → embedding → annotation**
   (the sequence-to-structure-to-function logic), e.g. the preprocessors:
   :class:`~aaanalysis.SequencePreprocessor`, :class:`~aaanalysis.StructurePreprocessor`, :class:`~aaanalysis.EmbeddingPreprocessor`,
   :class:`~aaanalysis.AnnotationPreprocessor`.
#. **A logic class is immediately followed by its ``*Plot`` pair**, and close
   variants form one contiguous family: :class:`~aaanalysis.AAclust` → :class:`~aaanalysis.AAclustPlot`; the CPP
   family :class:`~aaanalysis.CPP` → :class:`~aaanalysis.CPPGrid` → :class:`~aaanalysis.CPPPlot`; :class:`~aaanalysis.dPULearn` → :class:`~aaanalysis.dPULearnPlot`.
#. **Core before pro** where modality does not dictate otherwise (:class:`~aaanalysis.TreeModel`
   before :class:`~aaanalysis.ShapModel`).
#. **Group functions by kind** in Utility Functions: the ``comp_*`` metrics
   together, then display/options, then the ``plot_*`` helpers.
#. **Every public symbol appears in ``api.rst`` exactly once** — ``__all__`` (incl.
   the commented pro entries) and ``api.rst`` must match. The checker's
   ``API-INDEX-MISSING`` / ``API-INDEX-STALE`` enforces this coverage; placement
   within the section (rules 1–6) is a human call.

.. _docstring_guide:

Docstring Style Guide
=====================

This is the **single source of truth** for how docstrings are written in
AAanalysis. The mature classes :class:`~aaanalysis.CPP`,
:class:`~aaanalysis.AAclust`, and :class:`~aaanalysis.dPULearn` are the gold
standard — every public symbol should read as if written by the same hand.

The guide is *enforced* by an internal checker
(``.claude/skills/docstring-consistency/scripts/check_docstrings.py``); run it
before and after touching any docstring. All other docstring notes in the
repository (the always-on ``.claude/rules/docstrings.md``, the
``docstring-consistency`` skill, and the *Docstring Style* section of
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
  line *after* a blank first line, present tense, ending in a ``[Key]_`` project
  citation. Not an imperative verb.
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
* ``Returns`` is **named** (``name : type``) and matches the returned variable.
  Two type-only idioms are allowed: a bare class name (scikit-learn self-return,
  e.g. ``AAclust``) and a polymorphic ``X or Y``.
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

Cross-references (``See Also``)
-------------------------------

* Roles: ``:class:`` (classes), ``:meth:`` (methods), ``:func:`` (top-level /
  scikit-learn functions), ``:ref:`` (usage-principles pages), ``:term:``
  (glossary terms, see below).
* ``See Also`` is a ``*``-bulleted list; each entry is
  ``* :role:`Target`: gloss.`` — single colon, gloss after ``: ``. No bare
  ``name : desc`` numpydoc entries and no ``  :  `` (space-colon-space).

Citations
---------

* Cite inline with ``[Key]_`` **only**. Never inline a full ``.. [Key] Author,
  Year, ...`` reference, a raw URL, or ``(Author et al. Year)`` free text.
* All bibliography entries live in ``docs/source/index/references.rst``, grouped
  by topic. Key format: single first-author ``[AuthorYY]`` (``[Song12]``),
  two-author ``[FirstSecondYY]`` (``[ElkanNoto08]``), same author/year gets a
  trailing letter (``[Breimann24a]``).
* Pick the **few most relevant** references per symbol — 1–2 per major method,
  plus the project paper (``[Breimann25a]_``) at the class level.

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

Examples & verification
-----------------------

* Examples are authored as notebooks under ``examples/<subpackage>/`` and pulled
  in with ``.. include:: examples/<name>.rst`` (converted at docs-build time).
* Keep them small, seeded, and deterministic. Example notebooks **and** tutorials
  are executed in CI (``pytest --nbmake examples/ tutorials/``) so they cannot rot;
  tiny self-contained snippets may additionally use ``>>>`` doctests run with
  ``--doctest-modules``.

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

Conformance checklist
---------------------

A docstring is house-style if every applicable item holds. The right column is
the code emitted by the internal checker.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Rule
     - Checker code
   * - Class summary is a noun phrase (not a verb)
     - ``CLASS-SUMMARY-VERB``
   * - Class summary ends with a ``[Key]_`` citation
     - ``CLASS-NO-CITATION``
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

Tooling
-------

.. code-block:: bash

    # audit the whole package (or a single file)
    python .claude/skills/docstring-consistency/scripts/check_docstrings.py aaanalysis/
    # auto-fix the safe mechanical drift (Warns→Warnings, See-Also colon spacing)
    python .claude/skills/docstring-consistency/scripts/check_docstrings.py --fix aaanalysis/

``--fix`` applies only the mechanical subset; every other finding is an
author-side fix against the templates above.

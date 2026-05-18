.. _adr_0002:

AAWindowSampler PWM scoring is canonical-amino-acid-only
========================================================

``AAWindowSampler`` accepts ``motif_pwm`` as either an ``np.ndarray`` of shape
``(window_size, 20)`` (columns in alphabetical order =
``ut.LIST_CANONICAL_AA``) or a ``pd.DataFrame`` indexed by canonical AA letter
(reindexed internally). The DataFrame path rejects any non-canonical column
(``X``, ``B``, ``Z``, ``U``, ``O``, …) and any missing canonical column. We
deliberately do **not** support ambiguity codes or non-protein alphabets in
PWM scoring, even though the dual-input refactor could technically be
extended that way. The reason is in-core / in-pro parity (``CLAUDE.md`` §6)
with :func:`aaanalysis.scan_motif`, which delegates to the FIMO CLI: FIMO's
default protein alphabet is the canonical 20, and supporting ambiguity codes
in the in-memory path would break byte-identical hit-set parity with the
FIMO path.

Considered Options
------------------

- **Canonical + standard ambiguity codes** (``X B Z U O J``). Rejected:
  ambiguity codes would have to be either translated to FIMO's expanded
  alphabet at the pro boundary (added complexity) or silently rejected by
  the pro wrapper (asymmetric behavior between core and pro paths).
- **Fully arbitrary single-character alphabet** (matching
  ``sample_synthetic``'s custom-dict generator). Rejected: would break the
  FIMO parity story entirely and would also force a rethink of the
  ``AAWindowSampler`` class name (which currently is honest precisely
  *because* PWM scoring is AA-only).
- **Drop the DataFrame input and stay ndarray-only.** Rejected: the
  column-order foot-gun (silently wrong scores when the user uses
  non-alphabetical order) is a real defect class worth solving even within
  the canonical-only scope.

Consequences
------------

- ``sample_synthetic``'s custom-dict generator remains the *only* path in
  the class that produces non-amino-acid output. This asymmetry is called
  out in the class docstring's *Alphabets* note and in ``CONTEXT.md`` under
  the ``custom-alphabet generator`` entry.
- If a future feature needs PWM scoring over ambiguity codes or non-AA
  alphabets, this decision must be revisited together with the FIMO parity
  story — not in isolation.

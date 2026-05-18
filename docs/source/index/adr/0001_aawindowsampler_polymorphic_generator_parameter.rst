.. _adr_0001:

AAWindowSampler ``sample_synthetic`` keeps a polymorphic ``generator`` parameter
================================================================================

``AAWindowSampler.sample_synthetic`` accepts a single ``generator`` parameter
whose runtime shape — ``str``, ``list[str]``, or ``dict[str, float]`` — selects
one of three different operations (built-in / preset, multiplicative mix of
presets, custom-alphabet frequency table). The alternative was to split into
three named parameters (``generator``, ``mix_presets``, ``custom_freq``) with
mutual exclusion; we kept polymorphism for call-site brevity and because the
three shapes all answer the same conceptual question ("what's the recipe for
one window?"). The parameter was renamed from ``mode`` to ``generator`` to
make the dispatch-on-shape role legible at the call site, because ``mode`` is
reused elsewhere (``output_mode``) and gave the false impression of a flat
enum.

Considered Options
------------------

- **Three named parameters** (``generator: str | None``,
  ``mix_presets: list[str] | None``, ``custom_freq: dict | None``) with an
  "exactly one must be set" guard. Rejected: more friction at call sites for
  the common case; the three shapes are variants of the same concept.
- **Three methods** (``sample_synthetic``, ``sample_synthetic_mixed``,
  ``sample_synthetic_custom``). Rejected: doubles the API surface for what
  is one conceptual operation parameterized differently.

Consequences
------------

- The ``check_synth_generator`` validator has to dispatch on ``isinstance``
  and is non-trivial (~50 lines). The trade is intentional: validation
  complexity moves out of every call site into one place.
- A reader scanning the signature sees only ``generator=...`` and must
  consult the docstring to learn the three accepted shapes. The renamed
  parameter and the ``CONTEXT.md`` ``generator`` entry are the load-bearing
  discoverability mitigations.

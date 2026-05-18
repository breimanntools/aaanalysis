---
paths:
  - "aaanalysis/**/*.py"
---

# Reproducibility (`random_state` contract)

- Every new function or method that uses randomness MUST accept either a
  `random_state` (constructor) or `seed` (per-call) parameter.
- The parameter is threaded through to backend code and any
  sklearn / numpy RNG primitives.
- Per-call `seed` overrides constructor `random_state`. Constructor
  `random_state` overrides defaults. `aaanalysis.options["random_state"]`
  overrides everything when not `"off"`.
- Validate seeds with `ut.check_number_range(name="seed", val=seed,
  min_val=0, just_int=True)`.
- PRs that introduce untracked nondeterminism (e.g. a bare
  `np.random.default_rng()` without a plumbed seed) fail review.

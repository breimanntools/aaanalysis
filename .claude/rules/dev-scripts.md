---
paths:
  - "dev_scripts/**"
---

# `dev_scripts/`

- Add-only zone. Claude may **add** new `dev_<feature>.py` scripts.
  Claude must **not** modify or delete existing ones unless explicitly asked.
- When introducing a new public class or method, Claude **automatically
  creates** a `dev_<feature>.py` smoke-test script following the pattern of
  `dev_aa_window_sampler.py` (an inspection harness: print the API call,
  input shape / columns / head, output schema and key distributions, so the
  user can eyeball whether the result is what they wanted).
- Dev scripts are not packaged and not tested in CI; they are for local
  prototyping and reproducible bug-bisection only.

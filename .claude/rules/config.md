---
paths:
  - "aaanalysis/config.py"
---

# Configuration (`aaanalysis.options`)

- `aaanalysis.options` (`Settings` instance, in `config.py`) is the only
  sanctioned global state. Editing `config.py` is **CONFIRM-FIRST** (see
  CLAUDE.md §10).
- Adding a new option requires:
  1. A new key in `_dict_options`.
  2. A matching branch in `_check_option`.
  3. A unit test exercising both valid and invalid values.
- Per-call args override `options` only when the corresponding option is
  `"off"`. When an option is set to a concrete value, it wins. Maintain that
  contract for any new option.
- **Do not use `options[...]` as an internal cache.** Caching belongs in a
  private `@lru_cache` function on a pure loader, not in `Settings`.
- Do not add new global state outside `Settings`.

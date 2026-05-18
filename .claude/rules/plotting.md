---
paths:
  - "aaanalysis/plotting/**/*.py"
  - "aaanalysis/**/_*plot*.py"
  - "aaanalysis/**/_*Plot*.py"
  - "aaanalysis/**/*_plot*.py"
---

# Plotting

- Library code **never** calls `plt.show()`. Return `fig` / `ax` or mutate the
  passed `ax`.
- Colors come from `ut.COLOR_*` and `ut.DICT_COLOR*`. Do not hardcode hex
  values.
- `plot_settings()` mutates rcParams globally — call only from user-facing
  entry points, never from inside library logic.

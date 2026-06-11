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
- **In notebooks (the user side of the boundary), the opposite holds:** every
  example/tutorial plot cell MUST end with `plt.tight_layout()` then
  `plt.show()` so the figure renders as an image — a bare `Plot().method(...)`
  leaves only a `<Axes …>` text repr. See `notebooks.md`.

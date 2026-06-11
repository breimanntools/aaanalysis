# Example & tutorial notebooks

Path-scoped to `examples/**/*.ipynb` and `tutorials/**/*.ipynb`. These notebooks
are the executed source the docs build converts to RST (`conf.py` →
`create_notebooks_docs.export_example_notebooks_to_rst`) and are run in CI
(`pytest --nbmake tutorials/ examples/`). Author them with embedded outputs.

## Tables

Show every DataFrame with:

```python
aa.display_df(df, n_rows=10, show_shape=True)
```

Never a bare `df`, `df.head()`, or `print(df)` — those render as plain text and
lose the styled, shape-annotated presentation. `display_df` clamps `n_rows`/
`n_cols` to the frame size, so `n_rows=10` is safe on a table of any length
(a 3-row frame just shows all 3).

## Plots

A plot method returns an `Axes`/`Figure`; if it is the cell's last expression
the notebook captures only a useless `<Axes: …>` text repr, **not the image**.
End every plot cell explicitly:

```python
import matplotlib.pyplot as plt
aa.plot_settings()                      # once; nice rcParams (user-facing entry point)
aa.SomePlot().some_method(...)          # draws onto the current figure
plt.tight_layout()
plt.show()                              # flushes the figure as an inline PNG; returns None
```

- Run/execute the notebook with the **inline backend** (do NOT force
  `MPLBACKEND=Agg` when building) so `plt.show()` embeds an `image/png` output.
- This is **notebook/user code**, the sanctioned place for `plt.show()` /
  `plt.tight_layout()` / `plot_settings()`. Library plot code still never calls
  them (see `plotting.md`).

## General

- Cover every public parameter of the demonstrated method (grouped sensibly).
- Set `aa.options["verbose"] = False` at the top to keep outputs clean.
- Commit notebooks **with executed outputs** (tables + images), and re-run
  before pushing — a stale/unexecuted notebook is a recurring miss.
- One notebook per public method, named `<abbrev>_<method>.ipynb`, matching the
  method docstring's `.. include:: examples/<name>.rst`.

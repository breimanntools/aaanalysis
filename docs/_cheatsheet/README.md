# AAanalysis cheat sheet — build tooling

Single-source generator for the cheat sheet shipped at
`docs/source/_static/cheat_sheet.{html,pdf}` and embedded by
`docs/source/index/cheat_sheet.rst`.

```
content.py            ← SINGLE SOURCE: panels, snippets, glossary, citations, gallery
template.html.jinja   ← layout (Jinja2)
cheat_sheet.css       ← screen styles + @page print rules (landscape A4, 3 pages)
build_cheat_sheet.py  ← renders HTML (Jinja2) AND PDF (WeasyPrint) from the above
gen_plots.py          ← regenerates the example figures (needs the full aaanalysis stack)
fonts/                ← vendored DejaVu TTFs (embedded into the PDF; OFL, see LICENSE_DEJAVU)
```

The "Example Outputs" gallery embeds real figures from
`docs/source/_static/cs_plots/*.png`. Those are produced by `gen_plots.py`
(it runs CPP / TreeModel / dPULearn / AAlogo on a small `DOM_GSEC` sample) and
committed. Regenerate them only when the figures should change:

```bash
COVERAGE_CORE=sysmon .venv/bin/python docs/_cheatsheet/gen_plots.py
```

One build produces **both** outputs, so the HTML and PDF can never drift. Edit
`content.py` (text/snippets) or `cheat_sheet.css` (layout), then regenerate.

## Regenerate

The build needs WeasyPrint + Jinja2 in a throwaway venv (kept out of git):

```bash
# one-time: create the build venv
uv venv docs/_cheatsheet/.buildenv --python 3.13
uv pip install --python docs/_cheatsheet/.buildenv/bin/python weasyprint jinja2

# build HTML + PDF
docs/_cheatsheet/.buildenv/bin/python docs/_cheatsheet/build_cheat_sheet.py

# HTML only (no WeasyPrint / system libs needed)
docs/_cheatsheet/.buildenv/bin/python docs/_cheatsheet/build_cheat_sheet.py --html-only
```

Then commit the regenerated `docs/source/_static/cheat_sheet.{html,pdf}`.

## Notes

- **macOS / WeasyPrint:** WeasyPrint needs the Pango/GLib system libraries
  (`brew install pango`). The build script re-execs itself with
  `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` so those libs are found; no
  manual env setup is required.
- **Fonts:** the PDF embeds the vendored DejaVu fonts (the same family the
  package's matplotlib figures use) via `@font-face`, injected only into the PDF
  pass. macOS system fonts (`Helvetica.ttc`) are TrueType *collections*
  WeasyPrint cannot load — it silently falls back to a serif — hence the vendored
  TTFs, which also make the PDF reproducible on any machine (incl. Linux CI).
- **Not packaged:** this directory is docs tooling; it is not part of the
  installed `aaanalysis` wheel.

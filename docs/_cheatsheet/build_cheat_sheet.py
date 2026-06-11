"""This is a script for rendering the AAanalysis cheat sheet to HTML and PDF.

Single-source pipeline: ``content.py`` (data) -> ``template.html.jinja`` (layout)
+ ``cheat_sheet.css`` (style) -> ``docs/source/_static/cheat_sheet.{html,pdf}``.
Building both from one source guarantees the HTML and PDF never drift.

Usage (see README.md for environment setup)::

    docs/_cheatsheet/.buildenv/bin/python docs/_cheatsheet/build_cheat_sheet.py
    docs/_cheatsheet/.buildenv/bin/python docs/_cheatsheet/build_cheat_sheet.py --html-only

On macOS the script re-execs itself with ``DYLD_FALLBACK_LIBRARY_PATH`` pointed at
the Homebrew lib dir so WeasyPrint can load the Pango/GLib system libraries.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "source" / "_static"
HTML_OUT = OUT_DIR / "cheat_sheet.html"
PDF_OUT = OUT_DIR / "cheat_sheet.pdf"


def _ensure_macos_libs():
    """Re-exec once on macOS with the Homebrew lib dir on the dyld fallback path."""
    if sys.platform != "darwin" or os.environ.get("_AA_CS_REEXEC"):
        return
    for libdir in ("/opt/homebrew/lib", "/usr/local/lib"):
        if (Path(libdir) / "libgobject-2.0.dylib").exists():
            existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            if libdir not in existing.split(":"):
                env = dict(os.environ)
                env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{libdir}:{existing}".rstrip(":")
                env["_AA_CS_REEXEC"] = "1"
                os.execve(sys.executable, [sys.executable, *sys.argv], env)
            return


def _font_face_css() -> str:
    """@font-face block binding 'AA Sans'/'AA Mono' to the bundled DejaVu TTFs.

    Injected only into the PDF pass so WeasyPrint embeds a real .ttf (Helvetica
    on macOS is a .ttc collection it cannot use, and silently falls back to a
    serif). The DejaVu fonts are vendored under ``fonts/`` (same family the
    package's matplotlib figures use), so the PDF renders identically on any
    machine. URLs are relative to ``base_url`` (= HERE).
    """
    faces = [
        ("AA Sans", "DejaVuSans.ttf", "normal", "normal"),
        ("AA Sans", "DejaVuSans-Bold.ttf", "bold", "normal"),
        ("AA Sans", "DejaVuSans-Oblique.ttf", "normal", "italic"),
        ("AA Sans", "DejaVuSans-BoldOblique.ttf", "bold", "italic"),
        ("AA Mono", "DejaVuSansMono.ttf", "normal", "normal"),
        ("AA Mono", "DejaVuSansMono-Bold.ttf", "bold", "normal"),
    ]
    fonts_dir = HERE / "fonts"
    out = []
    for family, fname, weight, style in faces:
        uri = (fonts_dir / fname).as_uri()  # absolute file:// (PDF pass only)
        out.append(
            f"@font-face {{ font-family: '{family}'; font-weight: {weight}; "
            f"font-style: {style}; src: url('{uri}'); }}"
        )
    return "\n".join(out)


import html as _html
import re as _re

_TOKEN = _re.compile(r"(#[^\n]*)|('[^']*'|\"[^\"]*\")|(\baa\.[A-Za-z_][A-Za-z0-9_]*)")


def _hl(code: str):
    """Light syntax highlighting: comments, strings, and aa.<name> -> spans."""
    from markupsafe import Markup
    out, i = [], 0
    for mtok in _TOKEN.finditer(code):
        out.append(_html.escape(code[i:mtok.start()]))
        cls = "c-cm" if mtok.group(1) else "c-st" if mtok.group(2) else "c-nm"
        out.append(f'<span class="{cls}">{_html.escape(mtok.group())}</span>')
        i = mtok.end()
    out.append(_html.escape(code[i:]))
    return Markup("".join(out))


def render_html(*, pdf: bool = False) -> str:
    import jinja2
    from content import CONTENT  # local module (HERE is on sys.path)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(HERE)),
        autoescape=jinja2.select_autoescape(["html", "jinja"]),
        trim_blocks=True, lstrip_blocks=True,
    )
    env.filters["hl"] = _hl
    css = (HERE / "cheat_sheet.css").read_text(encoding="utf-8")
    if pdf:
        css = _font_face_css() + "\n" + css
        # WeasyPrint resolves against base_url=HERE; use absolute file:// for images
        img_base = (OUT_DIR / "cs_plots").as_uri() + "/"
        logo_base = (OUT_DIR / "cs_img").as_uri() + "/"
    else:
        # committed HTML lives in _static/; images in _static/cs_plots|cs_img/
        img_base = "cs_plots/"
        logo_base = "cs_img/"
    template = env.get_template("template.html.jinja")
    return template.render(css=css, c=CONTENT, m=CONTENT["meta"],
                           img_base=img_base, logo_base=logo_base)


def main() -> int:
    html_only = "--html-only" in sys.argv
    sys.path.insert(0, str(HERE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    html = render_html(pdf=False)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"[cheat-sheet] HTML  -> {HTML_OUT}  ({len(html):,} bytes)")

    if html_only:
        print("[cheat-sheet] --html-only: skipped PDF.")
        return 0

    from weasyprint import HTML  # imported late so --html-only needs no libs
    HTML(string=render_html(pdf=True), base_url=str(HERE)).write_pdf(str(PDF_OUT))
    print(f"[cheat-sheet] PDF   -> {PDF_OUT}")
    return 0


if __name__ == "__main__":
    _ensure_macos_libs()
    raise SystemExit(main())

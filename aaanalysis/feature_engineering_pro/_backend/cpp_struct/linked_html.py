"""
This is a script for the backend of the CPPStructurePlot linked view: it builds a
self-contained HTML page where the CPP feature map (an image with per-column hover
strips) and a 3Dmol.js cartoon are linked — hovering a feature-map column highlights
the corresponding residue in the structure, reproducing the deployed app. The 3Dmol
viewer is hand-built (not via py3Dmol's ``_make_html``) so its JS handle is reachable
from the overlay's hover events.
"""
import json

import aaanalysis.utils as ut

# 3Dmol.js CDN (the deployed app loads the same build). The viewer script polls for
# ``$3Dmol`` before initialising, so it works whether the CDN <script> loaded
# synchronously (standalone page) or asynchronously (notebook HTML injection).
_3DMOL_CDN = "https://3Dmol.org/build/3Dmol-min.js"
# Highlight colour for the hovered residue (the app's selected-site magenta).
_HL_COLOR = ut.COLOR_STRUCT_HIGHLIGHT
_HL_STICK_RADIUS = 0.45
_HL_SPHERE_RADIUS = 1.9


# I Helper Functions
def _columns(heatmap_ax, n_pos, start):
    """Per-column overlay geometry in figure fractions, mapped to absolute residues.

    Returns a list of ``{resi, left, width}`` (figure-fraction x of each heatmap column)
    plus the heatmap band's ``top`` / ``height`` (figure fractions from the top), computed
    the way the deployed app maps a hovered column to a residue.
    """
    pos = heatmap_ax.get_position()
    x0, x1, y0, y1 = pos.x0, pos.x1, pos.y0, pos.y1
    width = (x1 - x0) / n_pos
    cols = [{"resi": start + i, "left": x0 + i * width, "width": width} for i in range(n_pos)]
    return cols, (1.0 - y1), (y1 - y0)   # top (from image top), height


# II Main Functions
def build_linked_html(*, uid, pdb_text, fmt, chain_id, residue_styles, base_color,
                      faded, fade_opacity, zoom_resis, fmap_png_b64, columns,
                      band_top, band_height, width=520, height=460):
    """Build the self-contained, feature-map-linked structure HTML.

    ``residue_styles`` is ``[[resi, color_hex, stick_radius], ...]`` for the impact residues
    (precomputed in Python so the JS mirrors the py3Dmol render exactly); ``columns`` is the
    overlay geometry from :func:`_columns`. Hovering a column calls the viewer's highlight
    function for ``resi``; leaving restores the base styling.
    """
    view_id = f"cppstruct_view_{uid}"
    chain_js = json.dumps(chain_id)            # "A" or null
    impact_js = json.dumps(residue_styles)     # [[resi, "#hex", radius], ...]
    # Embed the PDB text as a JS string; escape "</" so a stray "</script>" in the
    # structure text cannot break out of the <script> element (HTML parses before JS).
    pdb_js = json.dumps(pdb_text).replace("</", "<\\/")
    zoom_js = json.dumps([str(r) for r in zoom_resis]) if zoom_resis else "null"
    base_cartoon = (f'{{color:"{base_color}",opacity:{fade_opacity}}}' if faded
                    else f'{{color:"{base_color}"}}')

    # Per-column hover strips over the feature-map image (percent positions).
    strips = []
    for c in columns:
        strips.append(
            f'<div class="cpp-col" data-resi="{c["resi"]}" '
            f'style="position:absolute;left:{c["left"] * 100:.4f}%;'
            f'width:{c["width"] * 100:.4f}%;top:{band_top * 100:.4f}%;'
            f'height:{band_height * 100:.4f}%;cursor:crosshair;"></div>')
    strips_html = "".join(strips)

    wrap_id = f"cppstruct_wrap_{uid}"
    return f"""\
<div id="{wrap_id}" style="display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap;">
  <div id="{view_id}" style="width:{width}px;height:{height}px;position:relative;"></div>
  <div style="position:relative;display:inline-block;">
    <img src="data:image/png;base64,{fmap_png_b64}" style="display:block;max-width:100%;height:auto;"/>
    {strips_html}
  </div>
</div>
<script src="{_3DMOL_CDN}"></script>
<script>
(function() {{
  function init() {{
    var wrap = document.getElementById("{wrap_id}");
    if (!wrap || wrap.dataset.cppInit) return;
    wrap.dataset.cppInit = "1";
    var v = $3Dmol.createViewer(document.getElementById("{view_id}"), {{backgroundColor: "white"}});
    v.addModel({pdb_js}, "{fmt}");
    var CH = {chain_js};
    function sel(r) {{ return CH ? {{resi: String(r), chain: CH}} : {{resi: String(r)}}; }}
    var IMPACT = {impact_js};
    function applyBase() {{
      v.setStyle({{}}, {{cartoon: {base_cartoon}}});
      IMPACT.forEach(function(x) {{
        v.setStyle(sel(x[0]), {{cartoon: {{color: x[1]}}}});
        if (x[2] > 0) v.addStyle(sel(x[0]), {{stick: {{color: x[1], radius: x[2]}}}});
      }});
    }}
    applyBase();
    var ZOOM = {zoom_js};
    if (ZOOM) {{ v.zoomTo(CH ? {{resi: ZOOM, chain: CH}} : {{resi: ZOOM}}); }} else {{ v.zoomTo(); }}
    v.render();
    function highlight(r) {{
      applyBase();
      v.setStyle(sel(r), {{cartoon: {{color: "{_HL_COLOR}"}}}});
      v.addStyle(sel(r), {{stick: {{color: "{_HL_COLOR}", radius: {_HL_STICK_RADIUS}}}}});
      v.addStyle(sel(r), {{sphere: {{color: "{_HL_COLOR}", radius: {_HL_SPHERE_RADIUS}}}}});
      v.render();
    }}
    wrap.querySelectorAll('.cpp-col').forEach(function(el) {{
      el.addEventListener('mouseenter', function() {{ highlight(parseInt(el.dataset.resi)); }});
      el.addEventListener('mouseleave', function() {{ applyBase(); v.render(); }});
    }});
  }}
  // Run once 3Dmol.js is available (the CDN <script> may load asynchronously).
  if (window.$3Dmol) {{ init(); }}
  else {{ var __t = setInterval(function() {{ if (window.$3Dmol) {{ clearInterval(__t); init(); }} }}, 50); }}
}})();
</script>"""


def page(body):
    """Wrap a linked-view body in a minimal standalone HTML page."""
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>CPPStructurePlot linked view</title></head><body>{body}</body></html>")

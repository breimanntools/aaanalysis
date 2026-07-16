"""Shared helper for gallery thumbnail scripts.

Saves the current matplotlib figure as a uniform, white, square tile: flush
tight (so nothing — titles, importance bars, legends — is clipped), then pad
the natural bounding box onto a centered white square. Keeping every tile the
same square shape is what makes the card grid line up.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def save_square(out_path, size=900, pad_frac=0.06, dpi=150):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name("_tmp_" + out_path.name)
    fig = plt.gcf()
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close("all")
    im = Image.open(tmp).convert("RGB")
    side = int(max(im.size) * (1 + pad_frac))
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    canvas = canvas.resize((size, size), Image.LANCZOS)
    canvas.save(out_path)
    tmp.unlink()
    print("saved", out_path)

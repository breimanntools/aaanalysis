"""
This is a script for the backend of EmbeddingPreprocessor.fetch_embeddings:
the curated protein-language-model (PLM) registry, hardware detection, the
size-aware model recommendation, and residue→protein pooling. The heavy
``torch`` / ``transformers`` compute path lives behind ``_load_model_and_tokenizer``
and is imported lazily so this module stays importable on a base install.
"""
import os
from typing import Dict, List, Optional

import numpy as np

import aaanalysis.utils as ut


# I Helper Functions
# Curated model registry. A structured registry with citations/footprints is the
# sanctioned exception to "domain-bundle constants live in utils.py" (see
# code-conventions.md, like the AAWindowSampler PRESETS dict). Footprints are
# order-of-magnitude inference floors: weights at fp32 (CPU) / fp16 (GPU) plus a
# modest activation margin; real peak scales with batch_size x sequence length.
REGISTRY: Dict[str, Dict] = {
    "esm2_t6_8M": dict(
        repo_id="facebook/esm2_t6_8M_UR50D", params_m=7.8, dim=320,
        ram_gb=0.3, vram_gb=0.2, max_len=None, has_cls=True, tokenizer="bert",
        license="MIT", note="laptop / CI smoke / huge corpora; low-RAM fallback"),
    "esm2_t12_35M": dict(
        repo_id="facebook/esm2_t12_35M_UR50D", params_m=34.0, dim=480,
        ram_gb=0.5, vram_gb=0.3, max_len=None, has_cls=True, tokenizer="bert",
        license="MIT", note="default: best size/quality on a typical 16 GB CPU box"),
    "esm2_t30_150M": dict(
        repo_id="facebook/esm2_t30_150M_UR50D", params_m=148.8, dim=640,
        ram_gb=1.5, vram_gb=0.8, max_len=None, has_cls=True, tokenizer="bert",
        license="MIT", note="mid-tier; richer residue features, still CPU-tolerable"),
    "esm2_t33_650M": dict(
        repo_id="facebook/esm2_t33_650M_UR50D", params_m=652.4, dim=1280,
        ram_gb=3.0, vram_gb=2.0, max_len=None, has_cls=True, tokenizer="bert",
        license="MIT", note="strong default when a GPU is present; common in the literature"),
    "esm2_t36_3B": dict(
        repo_id="facebook/esm2_t36_3B_UR50D", params_m=2800.0, dim=2560,
        ram_gb=12.0, vram_gb=7.0, max_len=None, has_cls=True, tokenizer="bert",
        license="MIT", note="only with a >=12 GB-VRAM GPU; behind the crash guard"),
    "esm1b": dict(
        repo_id="facebook/esm1b_t33_650M_UR50S", params_m=650.0, dim=1280,
        ram_gb=3.0, vram_gb=2.0, max_len=1022, has_cls=True, tokenizer="bert",
        license="MIT", note="legacy ESM-1b comparability; hard 1022-residue cap"),
    "prott5_xl_u50": dict(
        repo_id="Rostlab/prot_t5_xl_half_uniref50-enc", params_m=1200.0, dim=1024,
        ram_gb=5.0, vram_gb=3.0, max_len=None, has_cls=False, tokenizer="t5",
        license="academic", note="matches UniProt's precomputed embeddings; encoder-only half"),
    "prostt5": dict(
        repo_id="Rostlab/ProstT5", params_m=1200.0, dim=1024,
        ram_gb=5.0, vram_gb=3.0, max_len=None, has_cls=False, tokenizer="t5",
        license="MIT", note="structure-aware (3Di-trained) sequence embeddings; no PDB needed"),
}

LIST_MODELS: List[str] = list(REGISTRY)


def detect_hardware() -> Dict:
    """Detect device, total RAM, and free VRAM with no extra dependency.

    RAM via POSIX ``os.sysconf`` (skipped on Windows, where it is absent); GPU
    via ``torch`` only if it is importable. Returns a dict with keys ``device``
    ('cuda' | 'mps' | 'cpu'), ``has_cuda``, ``has_mps``, ``total_ram_gb``,
    ``free_vram_gb`` (``None`` where unavailable)."""
    info = dict(device="cpu", has_cuda=False, has_mps=False,
                total_ram_gb=None, free_vram_gb=None)
    # Total RAM (POSIX only): page size x number of physical pages.
    names = getattr(os, "sysconf_names", {})
    if "SC_PAGE_SIZE" in names and "SC_PHYS_PAGES" in names:
        try:
            info["total_ram_gb"] = (os.sysconf("SC_PAGE_SIZE")
                                    * os.sysconf("SC_PHYS_PAGES")) / 1e9
        except (ValueError, OSError):
            pass
    # GPU: only probe if torch is present (it is, on the compute path).
    try:
        import torch
    except ImportError:
        return info
    if torch.cuda.is_available():
        info["has_cuda"] = True
        info["device"] = "cuda"
        free, _total = torch.cuda.mem_get_info()
        info["free_vram_gb"] = free / 1e9
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        info["has_mps"] = True
        info["device"] = "mps"
    return info


def estimate_footprint_gb(model: str, device: str = "cpu", batch_size: int = 8) -> float:
    """Estimate the inference memory floor (GB) for ``model`` on ``device``.

    Uses the registry's per-device floor plus a crude activation margin that
    grows with ``batch_size``. A floor, not a ceiling."""
    meta = REGISTRY[model]
    base = meta["vram_gb"] if device in ("cuda", "mps") else meta["ram_gb"]
    return float(base) * (1.0 + 0.1 * batch_size)


def recommend_model(mem_gb: Optional[float] = None, device: str = "cpu") -> str:
    """Recommend the largest registry model whose footprint fits ``mem_gb``.

    Falls back to the smallest model when nothing fits or ``mem_gb`` is unknown
    (``None`` keeps the smallest model, the always-safe choice)."""
    by_size = sorted(REGISTRY, key=lambda k: REGISTRY[k]["params_m"])
    if mem_gb is None:
        return by_size[0]
    fitting = [k for k in by_size if estimate_footprint_gb(k, device=device) <= mem_gb]
    return fitting[-1] if fitting else by_size[0]


def pool_residue_(arr: np.ndarray, pooling: str = "mean") -> np.ndarray:
    """Pool a per-residue ``(L, D)`` array into one ``(D,)`` protein vector.

    Supports ``'mean'`` / ``'max'`` over residues. ``'cls'`` is rejected here
    because residue arrays carry no leading special token — it is available only
    via ``fetch_embeddings(mode='protein', pooling='cls')``, which reads it from
    the raw model output before special tokens are stripped."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"'embedding array' (ndim={arr.ndim}) should be 2-D (L, D)")
    if pooling == "mean":
        return arr.mean(axis=0)
    if pooling == "max":
        return arr.max(axis=0)
    raise ValueError(
        f"'pooling' ('{pooling}') should be one of 'mean', 'max' for residue arrays "
        f"('cls' is only available via fetch_embeddings(mode='protein'))")

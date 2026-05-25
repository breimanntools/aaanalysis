"""
This is a script for the backend of the StructurePreprocessor: a thin
wrapper around AFragmenter (Verwimp et al. 2025,
https://github.com/sverwimp/AFragmenter, PyPI package name
``protein-domain-segmentation``).

AFragmenter clusters AlphaFold PAE matrices into domain segments via Leiden
community detection. The wrapper lazy-imports the optional dependency and
returns a chopping string compatible with the v1.2 ``encode_domains`` reader
format: domains separated by commas, discontinuous segments within a domain
separated by underscores, segments are ``start-end`` 1-based inclusive.

Install hint: ``pip install aaanalysis[pro-domains]``.
"""
from pathlib import Path
from typing import List, Tuple


# I Helper Functions
def _try_import_afragmenter():
    """Lazy-import AFragmenter; return the module or None.

    PyPI distributes the package as ``protein-domain-segmentation`` but
    the import name is ``afragmenter`` (per the upstream repo). We try
    the documented name first and fall back to the package-name underscore
    variant to be defensive.
    """
    for mod_name in ("afragmenter", "protein_domain_segmentation"):
        try:
            import importlib
            return importlib.import_module(mod_name)
        except ImportError:
            continue
    return None


# II Main Functions
def check_afragmenter_available() -> None:
    """Raise ``RuntimeError`` with a friendly install hint if AFragmenter
    is not importable."""
    if _try_import_afragmenter() is None:
        raise RuntimeError(
            "'afragmenter' (PyPI 'protein-domain-segmentation') is not "
            "installed. Install via:\n"
            "  pip install aaanalysis[pro-domains]\n"
            "or directly:\n"
            "  pip install protein-domain-segmentation\n"
            "See https://github.com/sverwimp/AFragmenter for details.")


def _domains_to_chopping(domains) -> str:
    """Format a list-of-list-of-(start, end) into the Merizo/ChainSaw chopping
    string convention: ``1-10,15-25_30-40``.

    ``domains`` shape: ``[[(s, e), (s, e), ...], [(s, e), ...], ...]``
    """
    parts: List[str] = []
    for segments in domains:
        if not segments:
            continue
        seg_strs = [f"{s}-{e}" for s, e in segments]
        parts.append("_".join(seg_strs))
    return ",".join(parts)


def run_afragmenter_on_pae(pae_path: Path,
                           resolution: float = 0.7,
                           threshold: float = 2.0) -> str:
    """Run AFragmenter on a single PAE JSON and return a chopping string.

    Parameters
    ----------
    pae_path : pathlib.Path
        Path to the AlphaFold ``*_predicted_aligned_error_v4.json`` file
        (already decompressed by the caller if originally ``.gz``).
    resolution : float, default=0.7
        Leiden-community-detection resolution. Higher = more, smaller
        domains. AFragmenter docs recommend 0.7 as the default.
    threshold : float, default=2.0
        PAE upper threshold (Å) for the graph-edge cutoff. Pairs with
        PAE ≤ threshold are connected; the algorithm clusters the
        resulting graph.

    Returns
    -------
    chopping : str
        ``encode_domains``-compatible chopping string, e.g.
        ``"1-50,55-120,125-200"``. Empty string when AFragmenter assigns
        zero domains.

    Raises
    ------
    RuntimeError
        If AFragmenter is not installed (re-raised from
        ``check_afragmenter_available``) or if the call fails on the
        given PAE matrix.
    """
    check_afragmenter_available()
    mod = _try_import_afragmenter()
    # AFragmenter exposes a top-level ``AFragmenter`` class (per the docs)
    # that takes a PAE matrix or path and exposes a ``run_clustering`` /
    # ``cluster`` method returning the domain ranges. We adapt to either
    # entry point that the upstream version exposes.
    af_class = getattr(mod, "AFragmenter", None) or getattr(
        mod, "afragmenter", None)
    if af_class is None:
        raise RuntimeError(
            f"'afragmenter' module imported but no AFragmenter class found; "
            f"unexpected upstream version. Check `dir({mod.__name__})`.")
    try:
        # AFragmenter loads PAE from JSON path directly.
        seg = af_class(str(pae_path), threshold=threshold)
        # Newer versions use .cluster(resolution=...); older may use
        # .run_clustering(resolution=...).
        if hasattr(seg, "cluster"):
            domains = seg.cluster(resolution=resolution)
        elif hasattr(seg, "run_clustering"):
            domains = seg.run_clustering(resolution=resolution)
        else:
            raise RuntimeError(
                "AFragmenter object exposes neither .cluster nor "
                ".run_clustering — unexpected upstream version.")
    except Exception as e:
        raise RuntimeError(
            f"AFragmenter failed on PAE '{pae_path}': {e}") from e
    # Normalize the upstream output to list-of-list-of-(start, end)
    # 1-based inclusive tuples. AFragmenter typically returns a list of
    # lists of (start, end) tuples (one outer per domain, one inner per
    # contiguous segment). Some versions return a flat list of (s, e)
    # pairs (one per domain); the helper below handles both shapes.
    if not isinstance(domains, list):
        raise RuntimeError(
            f"AFragmenter returned {type(domains).__name__}, expected list")
    norm: List[List[Tuple[int, int]]] = []
    for dom in domains:
        if isinstance(dom, tuple) and len(dom) == 2:
            norm.append([(int(dom[0]), int(dom[1]))])
        elif isinstance(dom, list):
            norm.append([(int(s), int(e)) for s, e in dom])
        else:
            raise RuntimeError(
                f"AFragmenter domain item {dom!r} not understood; expected "
                f"(start, end) tuple or list of such tuples")
    return _domains_to_chopping(norm)

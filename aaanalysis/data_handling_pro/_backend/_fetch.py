"""
This is a script for the shared HTTP-fetch backend of the data_handling_pro
web acquisition tools (AlphaFold model/PAE downloads, UniProt annotation
records). It owns the single transport seam (``http_get_``) and the
order-preserving, opt-in-concurrent runner (``run_in_order_``) that both the
struct_preproc and annot_preproc fetch backends share.

``http_get_`` routes every request through a per-thread :class:`requests.Session`
so connection pooling survives across a bulk loop (and across an entry's
multiple files) without sharing a Session across threads — ``requests.Session``
is not documented thread-safe, so the opt-in :class:`ThreadPoolExecutor` path in
``run_in_order_`` hands each worker its own pooled session. Each fetch backend
imports ``http_get_`` into its own namespace, so the test suite patches a
module-local binding (``patch("<backend>.http_get_")``) per backend.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence
import threading

import requests


# I Helper Functions
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Return this thread's pooled :class:`requests.Session` (lazily created)."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = _thread_local.session = requests.Session()
    return session


# II Main Functions
def http_get_(url: str, timeout: float = 30.0):
    """GET ``url`` through the calling thread's pooled session.

    The single transport seam for the pro web-fetch backends; returns the
    :class:`requests.Response` exactly as ``requests.get(url, timeout=timeout)``
    would, so callers and their tests are unchanged apart from the connection
    reuse.
    """
    return _get_session().get(url, timeout=timeout)


def run_in_order_(func: Callable, items: Sequence,
                  max_workers: Optional[int] = None) -> List:
    """Map ``func`` over ``items``, returning results in **input order**.

    Serial (a plain list comprehension) when ``max_workers`` is ``None`` or
    ``<= 1`` — byte-identical to the original sequential loop. Otherwise runs on
    a :class:`ThreadPoolExecutor`; ``executor.map`` preserves input order, so the
    reassembled result list is identical regardless of worker count. Concurrency
    is opt-in because parallel requests to AlphaFold-DB / UniProt risk HTTP-429
    throttling that would turn successes into failures (an output change).

    The first exception raised by any item propagates (consumed in order),
    matching the original loop's abort-on-failure semantics.
    """
    if max_workers is None or max_workers <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))

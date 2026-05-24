"""
setup.py — minimal companion to pyproject.toml so the setuptools build
backend compiles the Cython kernel into the wheel.

All package metadata lives in ``pyproject.toml`` under ``[project]`` (PEP
621). This file exists solely to declare the Cython extension; without
it, setuptools doesn't know there's a ``.pyx`` to compile.

End users `pip install aaanalysis` (or `uv add aaanalysis`) get the
precompiled `.so` inside the wheel — no manual build step. See
docs/adr/0001-cpp-backend-architecture.md.
"""
import numpy as np
from Cython.Build import cythonize
from setuptools import setup


setup(
    ext_modules=cythonize(
        [
            "aaanalysis/feature_engineering/_backend/cpp/_filters_c/_inner.pyx",
        ],
        language_level=3,
        compiler_directives={
            # Match the directives in _inner.pyx (boundscheck=False,
            # wraparound=False, cdivision=True, initializedcheck=False).
            # Re-stated here so they're applied even if a future contributor
            # removes the header line.
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
    include_dirs=[np.get_include()],
)

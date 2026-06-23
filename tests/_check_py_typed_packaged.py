"""Guard: the PEP 561 ``py.typed`` marker must ship in the INSTALLED package.

This is **not** a pytest test — it is invoked as a standalone script by the
cibuildwheel ``test-command`` (see ``[tool.cibuildwheel]`` in ``pyproject.toml``)
so it runs against the freshly built wheel installed in a clean env, on every
shipped platform / Python. Run as ``python <this-file>``, Python prepends only
this file's directory (``tests/``, which holds no ``aaanalysis`` package) to
``sys.path`` — so ``import aaanalysis`` resolves to the installed wheel, never the
source tree. That is the whole point: a pytest test under the editable dev matrix
resolves to source (where the marker always exists) and would pass even if a
``[tool.setuptools.package-data]`` misconfiguration dropped the marker from the
artifact. This script catches that before the wheel reaches PyPI.
"""
import sys
from importlib.resources import files

import aaanalysis  # noqa: F401  (resolves to the installed wheel; see module docstring)

marker = files("aaanalysis").joinpath("py.typed")
if not marker.is_file():
    sys.exit(f"FAIL: PEP 561 py.typed missing from installed aaanalysis ({marker})")
print(f"OK: py.typed present in installed artifact -> {marker}")

"""This is a script to test that consecutive methods are separated by a blank line.

House style (and PEP 8 E301): two methods in a class body must have at least one
blank line between them. A signature rewrite once removed the blank line before
``CPP.simplify``, leaving ``return ...`` glued to the next ``def`` — readable code
review missed it, so this guards it programmatically across the whole package.

The check is intentionally narrow: it flags only a method ``def`` whose immediately
preceding sibling in the class body is *also* a method (method-to-method). It does
NOT require a blank line between a class docstring (or the class header) and the
first method, which is a separate, conventionally-accepted spacing.
"""
import ast
import pathlib

import aaanalysis  # noqa: F401  (anchor the installed/source package root)

_PKG_ROOT = pathlib.Path(aaanalysis.__file__).parent


def _method_spacing_violations():
    violations = []
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            body = node.body
            for prev, cur in zip(body, body[1:]):
                func_types = (ast.FunctionDef, ast.AsyncFunctionDef)
                if not (isinstance(cur, func_types) and isinstance(prev, func_types)):
                    continue
                start = cur.decorator_list[0].lineno if cur.decorator_list else cur.lineno
                blank_lines = start - prev.end_lineno - 1
                if blank_lines < 1:
                    rel = path.relative_to(_PKG_ROOT.parent)
                    violations.append(f"{rel}:{start}: '{cur.name}' has no blank line "
                                      f"after method '{prev.name}'")
    return violations


class TestMethodSpacing:
    """Every method must be preceded by at least one blank line after the prior method."""

    def test_no_glued_methods(self):
        violations = _method_spacing_violations()
        assert not violations, "Methods must be separated by a blank line:\n" + "\n".join(violations)

import os
import sys
import importlib.util
import inspect
from typing import Any
from datetime import datetime
import platform
from types import WrapperDescriptorType

sys.path.append(os.path.abspath('.'))
from create_tables_doc import generate_table_rst

# -- Path and Platform setup --------------------------------------------------
SEP = "\\" if platform.system() == "Windows" else "/"
path_source = os.path.join(os.path.dirname(__file__))

# Root path
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../aaanalysis'))
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
project = 'AAanalysis'
copyright = f'{datetime.now():%Y}, Stephan Breimann'
author = 'Stephan Breimann'
version = "latest"
release = '2023'
repository_url = "https://github.com/breimanntools/aaanalysis"
pygments_style = "sphinx"
todo_include_todos = False

# -- General configuration ---------------------------------------------------
templates_path = ['_templates']
exclude_patterns = []
source_suffix = '.rst'
master_doc = 'index'

nitpicky = True
needs_sphinx = '4.0'
suppress_warnings = ['ref.citation', 'myst.header']

extensions = [
    'sphinx.ext.autodoc',  # Autogenerate documentation from docstrings
    'numpydoc',  # Support for Numpy-style docstrings
    'sphinx.ext.autosummary',  # Generate summary tables for API reference
    'sphinx.ext.viewcode',  # Link from docs to source code
    'sphinx_rtd_theme',  # Theme emulating "Read the Docs" style
    'sphinx_copybutton',  # Adds a "copy" button to code blocks
    'sphinx.ext.intersphinx',  # Links to documentation of objects in other Sphinx projects
    'sphinx.ext.doctest',  # Test code examples in documentation
    'myst_nb',  # Integrate Jupyter notebooks using MyST parser
    'sphinx.ext.mathjax',  # Render math equations using MathJax
    'sphinx.ext.coverage',  # Checks documentation coverage for modules
    'sphinx.ext.linkcode',  # Link from docs to specific lines in source code
    'matplotlib.sphinxext.plot_directive',  # Integrate Matplotlib plots in docs
    'sphinx_design',  # Advanced design elements for Sphinx docs
    'sphinxext.opengraph',  # OpenGraph meta tags for rich link previews
    # 'sphinx.ext.napoleon',  # Support for Numpy-style and Google-style docstrings
    # 'sphinx_autodoc_typehints',  # Display Python type hints in documentation (needs to be after napoleon)
    # 'sphinx_book_theme',  # Theme optimized for book-style content presentation
    # 'pydata_sphinx_theme',  # Theme with a focus on long-form content and optimized for data-focused libraries
]

# -- Autodoc & Numpydoc settings ----------------------------------------------
autodoc_default_options = {
    "members": True,
    "inherited-members": True
}
autosummary_generate = True
numpydoc_show_class_members = False
"""
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]
"""

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'shap': ('https://shap.readthedocs.io/en/latest/', None),
}

# -- Options for HTML output -------------------------------------------------
html_title = "AAanalysis"
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#343131",
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = [os.path.join(path_source, '_static')]
html_css_files = ['css/style.css']
html_show_sphinx = False
html_logo = "_artwork/logo_big_trans.png"
html_favicon = "_artwork/logo_small.png"
"""
html_favicon = "path_to_your_favicon.ico"
htmlhelp_basename = "YOUR_PROJECT_NAMEdoc"
"""

# -- Options for manual page output ---------------------------------------
man_pages = [
    (master_doc, "aaanalysis", "AAanalysis Documentation", [author], 1)
]

# -- Options for Texinfo output -------------------------------------------
texinfo_documents = [
    (master_doc, "AAanalysis", "AAanalysis Documentation", author, "AAanalysis",
     "Advanced analysis tools for researchers.", "Miscellaneous"),
]

# Create table.rst
generate_table_rst()

# -- Linkcode configuration ---------------------------------------------------
_module_path = os.path.dirname(importlib.util.find_spec("aaanalysis").origin)  # type: ignore


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object. This will link
    directly to the correct version of the file on GitHub.
    """
    if domain != 'py':
        return None

    module_name = info.get('module')
    if not module_name:
        return None

    try:
        obj: Any = sys.modules[module_name]
        for part in info["fullname"].split("."):
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)

        if isinstance(obj, property):
            obj = inspect.unwrap(obj.fget)  # type: ignore
        if isinstance(obj, WrapperDescriptorType):
            # Skip wrapper descriptors
            return None

        path = inspect.getsourcefile(obj)  # type: ignore
        src, lineno = inspect.getsourcelines(obj)

        # Convert module path to GitHub format
        relative_path = path.replace(_module_path, '').strip('/')
        url_path = f"{repository_url}/tree/master/aaanalysis/{relative_path}#L{lineno}-L{lineno + len(src) - 1}"
        return url_path
    except Exception as e:
        print(f"Error generating linkcode for {module_name}: {e}")
        return None


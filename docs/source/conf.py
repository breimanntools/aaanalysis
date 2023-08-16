import os
import sys
import importlib.util
import inspect
from typing import Any
from datetime import datetime
path_source = os.path.join(os.path.dirname(__file__))
# Root path
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'AAanalysis'
copyright = f'{datetime.now():%Y}, Stephan Breimann'
author = 'Stephan Breimann'
release = '2023'
repository_url = "https://github.com/breimanntools/aaanalysis"

# -- General configuration ---------------------------------------------------

nitpicky = True
needs_sphinx = '4.0'
suppress_warnings = ['ref.citation', 'myst.header']

# Default settings
templates_path = ['_templates']
exclude_patterns = []
source_suffix = '.rst'
master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx.ext.extlinks',
    #'sphinx_autodoc_typehints', # needs to be after napoleon
    #'numpydoc',
    'sphinx_rtd_theme',
    'myst_nb',
    #'sphinx_copybutton',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'sphinxext.opengraph'
]

# Generate the API documentation when building
autosummary_generate = True
autosummary_generate_overwrite = True  # Overwrites existing stubs
autodoc_member_order = 'bysource'
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'shap': ('https://shap.readthedocs.io/en/latest/', None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    #"repository_url": repository_url,
    #"use_repository_button": True,
}
html_static_path = [os.path.join(path_source, '_static')]
html_css_files = [os.path.join(path_source, '_static/css/style.css'),
                  os.path.join(path_source, "_static/css/override.css")]
html_show_sphinx = False
html_title = "AAanalysis"

# Add any extra configuration from Scanpy that you might need,
# such as the `intersphinx_mapping` for links to other projects' documentation.

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
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)

        if isinstance(obj, property):
            obj = inspect.unwrap(obj.fget)  # type: ignore

        path = inspect.getsourcefile(obj)  # type: ignore
        src, lineno = inspect.getsourcelines(obj)

        # Convert module path to GitHub format
        relative_path = path.replace(_module_path, '').strip('/')
        url_path = f"{repository_url}/tree/master/aaanalysis/{relative_path}#L{lineno}-L{lineno + len(src) - 1}"
        return url_path
    except Exception as e:
        print(f"Error generating linkcode for {module_name}: {e}")
        return None

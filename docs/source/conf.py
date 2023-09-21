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
path_source = os.path.join(os.path.dirname(__file__))
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
exclude_patterns = ['index/tables_template.rst', '**.ipynb_checkpoints']
source_suffix = ['.rst', '.md', '.ipynb']
master_doc = 'index'

nitpicky = False #True
needs_sphinx = '4.0'
suppress_warnings = ['ref.citation', 'myst.header']

extensions = [
    'sphinx.ext.autodoc',  # Autogenerate documentation from docstrings
    #'numpydoc',  # Support for Numpy-style docstrings
    'sphinx.ext.autosummary',  # Generate summary tables for API reference
    'sphinx_rtd_theme',  # Theme emulating "Read the Docs" style # "sphinx_book_theme"
    'sphinx_copybutton',  # Adds a "copy" button to code blocks
    'sphinx.ext.intersphinx',  # Links to documentation of objects in other Sphinx projects
    'sphinx.ext.doctest',  # Test code examples in documentation
    'nbsphinx',  # Integrate Jupyter notebooks (myst-nb alternative)
    'sphinx.ext.mathjax',  # Render math equations using MathJax
    'sphinx.ext.coverage',  # Checks documentation coverage for modules
    'sphinx.ext.linkcode',  # Link from docs to specific lines in source code
    'matplotlib.sphinxext.plot_directive',  # Integrate Matplotlib plots in docs
    'sphinx_design',  # Advanced design elements for Sphinx docs
    'sphinxext.opengraph',  # OpenGraph meta tags for rich link previews
    'sphinx.ext.napoleon',  # Support for Numpy-style and Google-style docstrings
    'sphinx_autodoc_typehints',  # Display Python type hints in documentation (needs to be after napoleon)
    # 'pydata_sphinx_theme',  # Theme with a focus on long-form content and optimized for _data-focused libraries
]

# -- Autodoc & Numpydoc settings ----------------------------------------------
# Autodoc settings
# See https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
autodoc_default_options = {
    "members": True,  # Document members (methods, attributes, etc.) of documented classes and modules
    "undoc-members": False,  # Document members without docstrings
    "private-members": False,  # Document members with a name prefixed by an underscore (_)
    "special-members": "",  # Document special members like __call__, __getitem__
    "inherited-members": False,  # Document members that are inherited from the base class
    "show-inheritance": False,  # Show the base classes in the documentation for a class
    "ignore-module-all": False,  # Ignore __all__ when looking for members to document
    "exclude-members": "__init__",  # List of members to be excluded from documentation
    "member-order": "bysource",  # Sort the documented members. Options: 'alphabetical', 'bysource', 'groupwise'
    "autodoc_typehints": "description",  # How to display type hints. Options: 'none', 'signature', 'description'
    "imported-members": False,  # Document members imported into the documented module from other modules
}


# Auto summary settings
# See https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html#configuration
autosummary_generate = True
autosummary_ignore_module_all = False

"""
# Numpydoc settings
# See https://numpydoc.readthedocs.io/en/latest/install.html#sphinx-extensions-configuration
numpydoc_use_plots = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False # Test True
# numpydoc_xref_aliases = { create aliases which can be reference in doc}
# numpydoc_validation_checks = {"all"}    # Strict checking for Sphinx build
"""

# Napoleon settings
# See https://sphinxcontrib-napoleon.readthedocs.io/en/latest/sphinxcontrib.napoleon.html#sphinxcontrib.napoleon.Config
napoleon_google_docstring = False   # Not default
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None


# -- Juypter notebooks integration -------------------------------------------
nbsphinx_execute = 'auto'


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

html_context = {
    'display_github': True,  # Add the 'Edit on GitHub' link
    'github_user': 'breimanntools',
    'github_repo': 'aaanalysis',
    'github_version': 'master/',
}


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
    #Determine the URL corresponding to Python object. This will link
    #directly to the correct version of the file on GitHub.
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


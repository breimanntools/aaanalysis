import os
import sys
import importlib.util
import inspect
from typing import Any
from datetime import datetime
import platform
from types import WrapperDescriptorType
import warnings

sys.path.append(os.path.abspath('.'))

# Create notebooks rst and table rst first
from create_tables_doc import generate_table_rst
from create_notebooks_docs import export_example_notebooks_to_rst, export_tutorial_notebooks_to_rst

export_tutorial_notebooks_to_rst()
export_example_notebooks_to_rst()
generate_table_rst()

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
exclude_patterns = ['index/tables_template.rst', '**.ipynb_checkpoints', 'generated/examples/*']
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
    'nbsphinx',  # Integrate Jupyter tutorials (myst-nb alternative)
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
    "undoc-members": False,
    "private-members": False,
    "special-members": "",
    "inherited-members": False,  # Document members that are inherited from the base class
    "show-inheritance": False,  # Show the base classes in the documentation for a class
    "ignore-module-all": False,  # Ignore __all__ when looking for members to document
    "exclude-members": "",
    "autodoc_typehints": "description",
    "imported-members": False,  # Document members imported into the documented module from other modules
}

autodoc_member_order = 'bysource'

# Type hint settings
typehints_fully_qualified = False
set_type_checking_flag = False
always_document_param_types = True
typehints_document_rtype = False    # Check


# Auto summary settings
# See https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html#configuration
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_ignore_module_all = False

# Napoleon settings
# See https://sphinxcontrib-napoleon.readthedocs.io/en/latest/sphinxcontrib.napoleon.html#sphinxcontrib.napoleon.Config
napoleon_google_docstring = False   # Not default
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True  # If True, list __init__ docstring separately from class docstring
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

# -- Juypter tutorials integration --------------------------------------------
nbsphinx_execute = 'never'


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
    'upsetplot': ('https://upsetplot.readthedocs.io/en/stable/', None),
    'biopython': ('https://biopython.org/docs/latest/api/', None),
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
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = [os.path.join(path_source, '_static')]
html_css_files = ['css/style.css', 'css/notebook.css']
html_show_sphinx = False
html_logo = "_artwork/logos/logo_black_large.svg"
html_favicon = "_artwork/logos/favicon_white.png"

html_context = {
    'display_github': True,  # Add the 'Edit on GitHub' link
    'github_user': 'breimanntools',
    'github_repo': 'aaanalysis',
    'github_version': 'master/',
}

html_meta = {
    'google-site-verification': 'Rk3T0-H7cpFf5UxXiL4-LMS0WN7FIyU_3NiomozORV0'
}

"""
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


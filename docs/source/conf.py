# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'AAanalysis'
copyright = '2023, Stephan Breimann'
author = 'Stephan Breimann'
release = '2023'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Generate documentation from docstrings
    'sphinx.ext.viewcode',      # Add a link to the source code for classes and functions
    'sphinx_rtd_theme'          # Use the theme provided by Read the Docs
]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = '.rst'        # Use '.md' if you're writing your docs in Markdown
master_doc = 'index'          # This is the main document, usually named "index.rst"

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

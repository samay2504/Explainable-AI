# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../python'))


# -- Project information -----------------------------------------------------

project = 'napkinXC'
copyright = '2018-2024 Marek Wydmuch'
author = 'Marek Wydmuch'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Autodoc configuration ---------------------------------------------------
# Mock required packages
autodoc_mock_imports = [
    "gdown",
    "napkinxc._napkinxc",
    "numpy",
    "scipy",
    "scipy.sparse",
    "requests"
    "warnings"
    "zipfile"
]

# Autodoc options
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# Generate autosummary pages. Output should be set with: `:toctree: pythonapi/`
autosummary_generate = ['python_api.rst']

# Only the class' docstring is inserted.
autoclass_content = 'class'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# The master toctree document.
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

import sphinx_rtd_theme

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
#html_css_files = ['custom.css']
html_theme = "sphinx_rtd_theme"
html_logo = "_static/napkinXC_logo_white.png"
html_favicon = "_static/favicon.png"
html_show_sourcelink = True
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_context = {}
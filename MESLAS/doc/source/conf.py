# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'MESLAS'
copyright = '2020, Trygve Olav Fossum, Jo Eidsvik, David Ginsbourger, Kanna Rajan, Cedric Travelletti'
author = 'Trygve Olav Fossum, Jo Eidsvik, David Ginsbourger, Kanna Rajan, Cedric Travelletti'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'numpydoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinxcontrib.bibtex',
]

# Disable docstring inheritance
autodoc_inherit_docstrings = False

# Otherwise numpydoc automatically includes methods from parent in summary.
numpydoc_show_inherited_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for Latex output --------------
_PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{bm}
"""
latex_engine = 'xelatex'
latex_elements = {
            'preamble': _PREAMBLE,
                # other settings go here
                }

imgmath_latex_preamble=r"""
\usepackage{amsmath}
\usepackage{bm}
"""

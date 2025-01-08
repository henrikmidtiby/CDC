# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

import OCDC  # noqa: F401 #

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "OCDC"
copyright = "2024, Henrik Dyrberg Egemose"  # noqa: A001
author = "Henrik Dyrberg Egemose"
release = "v0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "sphinxarg.ext",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Automatically generate stub pages when using the .. autosummary directive
autosummary_generate = True

autodoc_typehints = "description"
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    # "source_repository": "todo",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
html_title = "OCDC"

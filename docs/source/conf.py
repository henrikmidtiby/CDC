# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

import CDC  # noqa: F401 #

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "CDC"
copyright = f"2023-{datetime.now().year}, SDU UAS Center"  # noqa: A001
author = "SDU UAS Center"
version = CDC.__version__
release = CDC.__version__

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
autodoc_type_aliases = {"NDArray": "NDArray"}
autoclass_content = "both"
autodoc_member_order = "groupwise"

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    # "source_repository": "todo",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
html_title = "CDC"

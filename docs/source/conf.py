import os
import sys

sys.path.insert(
    0,
    os.path.abspath("../../src")
)

project = "pyINSYDE"
copyright = "2026, pyINSYDE Research Team"
author = "pyINSYDE Research Team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"

autodoc_typehints = "description"
autodoc_member_order = "bysource"

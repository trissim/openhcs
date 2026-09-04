# SMELL-IGNORE-FILE: This file is a documentation build script that runs outside the application context

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

from openhcs import __version__ as release
from openhcs.resources.brand import (
    BRAND_PRODUCT_NAME,
    BrandAsset,
    brand_asset_path,
)

# -- Project information -----------------------------------------------------

project = BRAND_PRODUCT_NAME
copyright = "2025, trissim"
author = "trissim"

# The short X.Y version
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx_rtd_theme",
    "sphinx_toolbox.collapse",
    "sphinx_design",
    "openhcs_config_reference",
    "openhcs_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# Archived pages are migration history, not current documentation.
exclude_patterns = ["archive/**", "**/archive/**"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx settings
local_intersphinx_root = os.environ.get("OPENHCS_LOCAL_INTERSPHINX_ROOT")


def first_party_inventory(owner: str):
    """Use the current CI-built inventory when available, else fetch remotely."""
    if not local_intersphinx_root:
        return None
    inventory = Path(local_intersphinx_root) / f"owner-docs-{owner}" / "objects.inv"
    return str(inventory) if inventory.is_file() else None


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scikit-image": ("https://scikit-image.org/docs/stable/", None),
    "objectstate": (
        "https://objectstate.readthedocs.io/en/latest/",
        first_party_inventory("objectstate"),
    ),
    "arraybridge": (
        "https://arraybridge.readthedocs.io/en/latest/",
        first_party_inventory("arraybridge"),
    ),
    "metaclass-registry": (
        "https://metaclass-registry.readthedocs.io/en/latest/",
        first_party_inventory("metaclass-registry"),
    ),
    "polystore": (
        "https://polystore.readthedocs.io/en/latest/",
        first_party_inventory("polystore"),
    ),
    "pyqt-reactive": (
        "https://pyqt-reactive.readthedocs.io/en/latest/",
        first_party_inventory("pyqt-reactive"),
    ),
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_rtd_theme"
html_logo = str(brand_asset_path(BrandAsset.LOCKUP_HORIZONTAL))
html_favicon = str(brand_asset_path(BrandAsset.FAVICON))

# Theme options
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "logo_only": True,
}

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    "**": [
        "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
    ]
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "OpenHCSdoc"

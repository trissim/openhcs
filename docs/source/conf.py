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
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openhcs import __version__ as release

# Mock imports for Read the Docs
import sphinx.ext.autodoc

class Mock(sphinx.ext.autodoc.ClassDocumenter):
    @classmethod
    def can_document_class(cls, *args, **kwargs):
        return False

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    @classmethod
    def __getattr__(cls, name):
        if name in ('__file__', '__path__'):
            return '/dev/null'
        elif name == '__all__':
            return []
        return Mock()

SYSTEM_LIBS = [
    'cv2', 'numpy', 'scipy', 'matplotlib', 'pandas', 'PIL', 'tifffile',
]

# Check if we're on Read the Docs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# If we're on Read the Docs, mock the system libraries
if on_rtd:
    sys.modules.update((mod_name, Mock()) for mod_name in SYSTEM_LIBS)

# -- Project information -----------------------------------------------------

project = 'OpenHCS'
copyright = '2025, trissim'
author = 'trissim'

# The short X.Y version
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx_rtd_theme',
    'sphinx_toolbox.collapse',
    'sphinx_design',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# Archived pages are migration history, not current documentation.
exclude_patterns = ['archive/**', '**/archive/**']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

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
local_intersphinx_root = os.environ.get('OPENHCS_LOCAL_INTERSPHINX_ROOT')


def first_party_inventory(owner: str):
    """Use the current CI-built inventory when available, else fetch remotely."""
    if not local_intersphinx_root:
        return None
    inventory = Path(local_intersphinx_root) / f'owner-docs-{owner}' / 'objects.inv'
    return str(inventory) if inventory.is_file() else None


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scikit-image': ('https://scikit-image.org/docs/stable/', None),
    'objectstate': (
        'https://objectstate.readthedocs.io/en/latest/',
        first_party_inventory('objectstate'),
    ),
    'arraybridge': (
        'https://arraybridge.readthedocs.io/en/latest/',
        first_party_inventory('arraybridge'),
    ),
    'metaclass-registry': (
        'https://metaclass-registry.readthedocs.io/en/latest/',
        first_party_inventory('metaclass-registry'),
    ),
    'polystore': (
        'https://polystore.readthedocs.io/en/latest/',
        first_party_inventory('polystore'),
    ),
    'pyqt-reactive': (
        'https://pyqt-reactive.readthedocs.io/en/latest/',
        first_party_inventory('pyqt-reactive'),
    ),
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'logo_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'OpenHCSdoc'

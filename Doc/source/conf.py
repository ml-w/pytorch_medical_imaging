# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
import datetime

sys.path.insert(0, os.path.abspath('../../'))
import pytorch_med_imaging


# -- Project information -----------------------------------------------------

project = 'PyTorch Medical Imaging'
copyright = f'2026, Lun M Wong. Last Update {datetime.datetime.now().strftime("%B %d, %Y")}'
author = 'Lun M Wong'


# -- General configuration ---------------------------------------------------

mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxcontrib.mermaid',
    'm2r2',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

templates_path = ['_templates']
exclude_patterns = []

# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = False       # render Args as a definition list, not :param: directives
napoleon_use_rtype = False       # keep return type inside Returns section, not a separate :rtype:
napoleon_use_ivar = True
napoleon_use_keyword = True
napoleon_attr_annotations = True
napoleon_preprocess_types = True  # hyperlink type names in Args/Returns to their API docs

# prefix
modindex_common_prefix = ['pytorch_med_imaging']
add_module_names = False


# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_logo = "_static/cuhk_logo.gif"
html_theme_options = {
    'navbar_align': 'left',
    'navigation_depth': 4,
    'show_toc_level': 2,
}
html_sidebars = {
    '**': ['sidebar-nav-bs'],
}

html_static_path = ['_static']
html_css_files = ['layout.css']


# -- Extensions to the Napoleon GoogleDocstring class -----------------------

from sphinx.ext.napoleon.docstring import GoogleDocstring


def parse_keys_section(self, section):
    return self._format_fields('Keys', self._consume_fields())
GoogleDocstring._parse_keys_section = parse_keys_section


def parse_attributes_section(self, section):
    return self._format_fields('Attributes', self._consume_fields())
GoogleDocstring._parse_attributes_section = parse_attributes_section


def parse_class_attributes_section(self, section):
    return self._format_fields('Class Attributes', self._consume_fields())
GoogleDocstring._parse_class_attributes_section = parse_class_attributes_section


def patched_parse(self):
    self._sections['keys'] = self._parse_keys_section
    self._sections['class attributes'] = self._parse_class_attributes_section
    self._unpatched_parse()
GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse


# -- Custom directives -------------------------------------------------------

from docutils.parsers.rst import Directive
from docutils import nodes


class HintDirective(Directive):
    has_content = True

    def run(self):
        self.assert_has_content()
        text = '\n'.join(self.content)
        node = nodes.admonition(text, classes=["hint"])
        node += nodes.title(text="Tips")
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def setup(app):
    app.add_directive("tips", HintDirective)

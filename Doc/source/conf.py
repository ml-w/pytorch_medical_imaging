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
copyright = f'2025, Lun M Wong. Last Update {datetime.datetime.now().strftime("%B %d, %Y")}'
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
napolean_use_keyword = True

# prefix
modindex_common_prefix = ['pytorch_med_imaging']
add_module_names = False


# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_logo = "_static/cuhk_logo.gif"
html_theme_options = {
    'sidebar_hide_name': False,
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

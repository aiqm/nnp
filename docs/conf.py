import nnp  # noqa: F401
import sphinx_rtd_theme

project = 'NNP'
copyright = '2019, Xiang Gao'
author = 'Xiang Gao'

version = nnp.__version__
release = nnp.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
]

templates_path = ['_templates']
html_static_path = []

source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
htmlhelp_basename = 'NNPdoc'
# Temporary fix for bug in HTML5 support in the RTD theme
html4_writer = True

sphinx_gallery_conf = {
    'examples_dirs': ['../tests/', '../nnp/'],
    'gallery_dirs': ['examples', 'code'],
    'filename_pattern': r'.*\.py',
    'ignore_pattern': r'__init__\.py|test_.*',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'torch': ('https://pytorch.org/docs/master/', None),
    'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
}

latex_documents = [
    (master_doc, 'NNP.tex', 'NNP Documentation',
     'Xiang Gao', 'manual'),
]

man_pages = [
    (master_doc, 'torchani', 'TorchANI Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'TorchANI', 'TorchANI Documentation',
     author, 'TorchANI', 'One line description of project.',
     'Miscellaneous'),
]

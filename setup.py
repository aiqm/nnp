from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='nnp',
    description='Common tools for PyTorch based of neural network potentials',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/aiqm/nnp',
    author='Xiang Gao',
    author_email='qasdfgtyuiop@gmail.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=['torch'],
)

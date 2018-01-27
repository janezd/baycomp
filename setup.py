from setuptools import setup, find_packages

setup(
    name = 'baycomp',
    version = '0.7.0',
    url = 'https://github.com/janezd/baycomp.git',
    author = 'J. Demsar, A. Benavoli, G. Corani',
    author_email = 'janez.demsar@fri.uni-lj.si',
    description = 'Bayesian tests for classifier comparison',
    packages = find_packages(),
    install_requires = [
        'matplotlib >= 2.1.2',
        'numpy >= 1.13.1',
        'scipy >= 0.19.1',
        'seaborn >= 0.8.1'
        ],
)

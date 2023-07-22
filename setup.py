from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name='baycomp',
    version='1.0.3',
    url='https://github.com/janezd/baycomp.git',
    author='J. Demsar, A. Benavoli, G. Corani',
    author_email='janez.demsar@fri.uni-lj.si',
    description='Bayesian tests for comparison of classifiers',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(),
    install_requires=[
        'matplotlib >= 2.1.2',
        'numpy >= 1.13.1',
        'scipy >= 0.19.1'],
    extra_requires=[
        'pystan >= 3.4.0'
    ],
    python_requires='>=3',
    package_data={
        'baycomp': ['hierarchical-t-test.stan']}
)

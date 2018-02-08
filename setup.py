from setuptools import setup, find_packages

setup(
    name='baycomp',
    version='0.7.1',
    url='https://github.com/janezd/baycomp.git',
    author='J. Demsar, A. Benavoli, G. Corani',
    author_email='janez.demsar@fri.uni-lj.si',
    description='Bayesian tests for comparison of classifiers',
    classifiers=[
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
],
    packages=find_packages(),
    install_requires=[
        'matplotlib >= 2.1.2',
        'numpy >= 1.13.1',
        'scipy >= 0.19.1',
        'seaborn >= 0.8.1'
        ],
    python_requires='>=3',
    setup_requires=['pytest-runner'],
    tests_require=['pytest']

)

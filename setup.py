#!/usr/bin/env python
# -*- coding: utf-8 -*-

# {# pkglts, pysetup.kwds
# format setup arguments

from setuptools import setup, find_packages


short_descr = "An OpenAlea library for the detection and quantitative data extraction from microscopy images of cell nuclei"
readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')


# find version number in src/openalea/nuclei_quantification/version.py
version = {}
with open("src/openalea/nuclei_quantification/version.py") as fp:
    exec(fp.read(), version)


setup_kwds = dict(
    name='openalea.nuclei_quantification',
    version=version["__version__"],
    description=short_descr,
    long_description=readme + '\n\n' + history,
    author="Guillaume Cerutti, ",
    author_email="guillaume.cerutti@inria.fr, ",
    url='https://github.com/Guillaume Cerutti/nuclei_quantification',
    license='cecill-c',
    zip_safe=False,

    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        ],
    tests_require=[
        "mock",
        "nose",
        ],
    entry_points={},
    keywords='',
    test_suite='nose.collector',
)
# #}
# change setup_kwds below before the next pkglts tag

# do not change things below
# {# pkglts, pysetup.call
setup(**setup_kwds)
# #}

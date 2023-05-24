#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Daniel Gilman",
    version='0.2.1',
    author_email='daniel.gilman@utoronto.ca',
    classifiers=[
        'Development Status :: 5 - Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="A tool for rendering full mass distributions for gravitational lensing simulations",
    entry_points={
        'console_scripts': [
            'pyHalo=pyHalo.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyHalo',
    name='pyhalo',
    packages=find_packages(),
    package_dir={'lenstronomy': 'lenstronomy'},
    #packages=find_packages(include=['pyHalo']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/dangilman/pyHalo',
    zip_safe=False,
)

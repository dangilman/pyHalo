======
pyHalo
======

.. image:: https://travis-ci.com/dangilman/pyHalo.svg?branch=master
        :target: https://travis-ci.com/dangilman/pyHalo

.. image:: https://coveralls.io/repos/github/dangilman/pyHalo/badge.svg?branch=master
        :target: https://coveralls.io/github/dangilman/pyHalo?branch=master
        
.. image:: https://badge.fury.io/py/pyhalo.svg
    :target: https://badge.fury.io/py/pyhalo
        
.. image:: https://github.com/dangilman/pyHalo/blob/master/readme_fig.jpg
        :target: https://github.com/dangilman/pyHalo/blob/master/readme_fig

pyHalo renders full mass distributions for substructure lensing simulations with the open source gravitational lensing software package lenstronomy (https://github.com/sibirrer/lenstronomy). The example notebooks illustrate the core functionality of this package. 

If you would like to use this package and have questions, please get in touch with me at daniel.gilman@utoronto.ca - I am happy to help! 

Installation
------------
"python3 -m pip install pyhalo"

Or clone the repository, navigate into the directory that contains the setup.py file and run "python setup.py install"

The first code release, before majoring refactoring, is 8302393. 

In order to use this package when not installing via pip, you'll need to install colossus http://www.benediktdiemer.com/code/colossus/ 

Features
--------

- Quickly render full populations of dark matter subhalos and line of sight halos for gravitational lensing simulations. Implemented models currently include cold and warm dark matter, custom mass-concentration relations, self-interacting dark matter, and more.
- Translte halo properties (mass, concentration, redshift, etc) into angular units for lensing computations with lenstronomy


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

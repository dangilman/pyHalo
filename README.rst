======
pyHalo
======


.. image:: https://img.shields.io/pypi/v/pyhalo.svg
        :target: https://pypi.python.org/pypi/pyhalo

.. image:: https://travis-ci.org/dangilman/pyHalo.png?branch=main
        :target: https://travis-ci.org/dangilman/pyHalo

.. image:: https://coveralls.io/repos/github/dangilman/pyHalo/badge.svg?branch=main
        :target: https://coveralls.io/github/dangilman/pyHalo?branch=main





pyHalo is a tool for rendering full substructure mass distributions for gravitational lensing simulations. It is intended for use with the open source gravitational lensing software package lenstronomy (https://github.com/sibirrer/lenstronomy). 

The core functionality illustrated in the example notebook is stable and well tested.  

If you would like to use this package and have questions, please get in touch with me at gilman@astro.utoronto.ca - I am happy to help! 

Installation
------------
Clone the repository, navigate into the main pyHalo directory and run python3 setup.py develop --user. (in the process of making this pip-installable) 

In order to use this package you'll need to install colossus http://www.benediktdiemer.com/code/colossus/ 


Features
--------

- Quickly render full populations of dark matter subhalos and line of sight halos for gravitational lensing simulations. Implemented models currently include Cold and Warm Dark Matter, custom mass-concentration relations, and more.
- Translte halo properties (mass, concentration, redshift, etc) into angular units for lensing computations with lenstronomy

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

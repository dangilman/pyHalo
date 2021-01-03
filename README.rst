======
pyHalo
======


.. image:: https://img.shields.io/pypi/v/pyhalo.svg
        :target: https://pypi.python.org/pypi/pyhalo

.. image:: https://travis-ci.org/dangilman/pyHalo.png?branch=master
        :target: https://travis-ci.org/dangilman/pyHalo

.. image:: https://coveralls.io/repos/github/dangilman/pyHalo/badge.svg?branch=master
        :target: https://coveralls.io/github/dangilman/pyHalo?branch=master





pyHalo is a tool for rendering full substructure mass distributions for gravitational lensing simulations. It is intended for use with the open source gravitational lensing software package lenstronomy (https://github.com/sibirrer/lenstronomy). 

This code is under active development, but the core functionality expressed in the example notebook is stable and well tested. 

If you would like to use this package and have questions, please get in touch with me at gilman@astro.utoronto.ca - I am happy to help! 

* Free software: MIT license
Installation
------------
.. code-block:: bash

    $ pip install pyHaloLens --user


Features
--------

- Quickly render full populations of dark matter subhalos and line of sight halos for gravitational lensing simulations. Implemented models currently include Cold and Warm Dark Matter, custom mass-concentration relations, and more.
- Translte halo properties (mass, concentration, redshift, etc) into angular units for lensing computations with lenstronomy

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

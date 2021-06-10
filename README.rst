======
pyHalo
======

.. image:: https://travis-ci.com/dangilman/pyHalo.svg?branch=master
        :target: https://travis-ci.com/dangilman/pyHalo

.. image:: https://coveralls.io/repos/github/dangilman/pyHalo/badge.svg?branch=master
        :target: https://coveralls.io/github/dangilman/pyHalo?branch=master
        
.. image:: https://github.com/dangilman/pyHalo/blob/master/readme_fig.jpg
        :target: https://github.com/dangilman/pyHalo/blob/master/readme_fig

pyHalo renders full mass distributions for substructure lensing simulations with the open source gravitational lensing software package lenstronomy (https://github.com/sibirrer/lenstronomy). The example notebook illustrates the core functionality of this package, which is stable and well tested. 

If you would like to use this package and have questions, please get in touch with me at gilman@astro.utoronto.ca - I am happy to help! 

Installation
------------
Clone the repository, navigate into the main pyHalo directory and run python3 setup.py develop --user. (in the process of making this pip-installable) 

In order to use this package you'll need to install colossus http://www.benediktdiemer.com/code/colossus/ 


Features
--------

- Quickly render full populations of dark matter subhalos and line of sight halos for gravitational lensing simulations. Implemented models currently include cold and warm dark matter, custom mass-concentration relations, self-interacting dark matter, and more.
- Translte halo properties (mass, concentration, redshift, etc) into angular units for lensing computations with lenstronomy

Citation
-------
If you use pyHalo, please cite `Gilman et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210505259G/abstract>`_.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

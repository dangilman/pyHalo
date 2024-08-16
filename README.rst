======
pyHalo
======

.. image:: https://codecov.io/github/dangilman/pyHalo/graph/badge.svg?token=3H5QT3S6OO 
 :target: https://codecov.io/github/dangilman/pyHalo

.. image:: https://badge.fury.io/py/pyhalo.svg
    :target: https://badge.fury.io/py/pyhalo

.. image:: https://github.com/dangilman/pyHalo/blob/master/readme_fig.jpg
        :target: https://github.com/dangilman/pyHalo/blob/master/readme_fig

pyHalo renders full mass distributions for substructure lensing simulations with the open source gravitational lensing software package lenstronomy (https://github.com/lenstronomy/lenstronomy). The example notebooks illustrate the core functionality of this package.

If you would like to use this package and have questions, please get in touch with me at gilmanda@uchicago.edu - I am happy to help!

Installation
------------
Install via pypi: "python3 -m pip install pyhalo".
The installation via pip of versions after 1.2.0 includes some changes that may not be backwards compatible with the previous stable version. To use the previous stable version clone the repository from commit 4dc87c8.

Install from github: "git clone https://github.com/dangilman/pyHalo.git; cd pyhalo; python -m setup.py install". Make sure to check the requirements listed in requirements.txt

Features
--------
The purpose of this code is to quickly render full populations of dark matter subhalos and line of sight halos for gravitational lensing simulations. pyHalo also transltes halo properties (mass, concentration, redshift, etc) into angular units for lensing computations with lenstronomy. Implemented dark matter models currently include:

1) cold dark matter
    - https://arxiv.org/abs/1909.02573
    - https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.3163G/abstract
    - https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.5843D/abstract

2) warm dark matter
    - https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.6077G/abstract
    - https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.6159K/abstract
    - https://ui.adsabs.harvard.edu/abs/2024arXiv240303253G/abstract
    - https://ui.adsabs.harvard.edu/abs/2024arXiv240501620K/abstract

3) self-interacting dark matter
    - https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.2432G/abstract
    - https://ui.adsabs.harvard.edu/abs/2023PhRvD.107j3008G/abstract
    - https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.5455D/abstract

4) fuzzy dark matter
    - https://ui.adsabs.harvard.edu/abs/2022MNRAS.517.1867L/abstract

5) black holes
    - https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.5434D/abstract

(if you have used pyHalo in a paper but your publication is not listed here, feel free to submit a pull request to edit the README)

Other useful features:

1) customizeable mass-concentration relations and flexible parameterization of the halo mass function with variable normalizations and logarithmic slopes (see, for example,  https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.3163G/abstract)

2) a variety of tidal stripping modules designed to predict the bound subhalo mass function given the infall subhalo mass function

3) automatic calculation of negative sheets of convergence along the line of sight to keep the mean density of the Universe equal to the critical density after adding halos

4) different geometries for rendering line-of-sight halos, ranging from a cylindrical volume to a double-cone configuration that opens towards the lens and closes towards the source.

5) correlated structure around the main deflector arising from the two-halo term (https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.5721G/abstract, https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.6064L/abstract)

pyHalo currently supports a variety of halo mass profiles, including Navarro-Frenk-White (NFW) profiles, truncated (NFW) profiles, cored power-law (PL) profiles, double PL profiles with variable inner and outer logarithmic slopes, cored NFW profiles, and point masses.

Attribution
-------

When using pyHalo, please link to this repository in a footnote and cite https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.6077G/abstract

Contribution
-------

Please feel free to make a pull request implementing any changes you feel would improve the functionality of the code! 

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

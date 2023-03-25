import numpy as numpy
from colossus.halo.concentration import concentration, peaks
from pyHalo.defaults import halo_default
import warnings
warnings.filterwarnings("ignore")

class Concentration(object):

    def __init__(self, lens_cosmo):

        """
        This class handles concentrations of the mass-concentration relation for NFW profiles
        :param lens_cosmo: an instance of LensCosmo
        """

        self._lens_cosmo = lens_cosmo

    def nfw_concentration(self, M, z, model, mdef, logmhm,
                          scatter, scatter_amplitude, kwargs_suppresion, suppression_model):

        """
        :param M: mass in units M_solar (no little h)
        :param z: redshift
        :param model: the model for the concentration-mass relation
        if type dict, will assume a custom MC relation parameterized by c0, beta, zeta (see _NFW_concentration_custom)
        if string, will use the corresponding concentration model in colossus (see http://www.benediktdiemer.com/code/colossus/)

        :param mdef: mass defintion for use in colossus modules. Default is '200c', or 200 times rho_crit(z)
        :param logmhm: log10 of the half-mode mass in units M_sun, specific to warm dark matter models.
        This parameter defaults to 0. in the code, which leaves observables unaffected for the mass scales of interest ~10^7
        :param scatter: bool; if True will induce lognormal scatter in the MC relation with amplitude
        scatter_amplitude in dex
        :param scatter_amplitude: the amplitude of the scatter in the mass-concentration relation in dex
        :param kwargs_suppresion: keyword arguments for the suppression function
        :param suppression_model: the type of suppression, either 'polynomial' or 'hyperbolic'
        :return: the concentration of the NFW halo
        """

        h = self._lens_cosmo.cosmo.h

        if isinstance(model, dict):

            assert 'custom' in model.keys()

            if isinstance(M, float) or isinstance(M, int):
                M_h = M * h
                _c = self.NFW_concentration_custom(M_h, z, model)
            else:

                if isinstance(z, numpy.ndarray) or isinstance(z, list):
                    assert len(z) == len(M)
                    _c = [self.NFW_concentration_custom(float(mi * h), z[i], model)
                         for i, mi in enumerate(M)]
                else:
                    _c = [self.NFW_concentration_custom(float(mi * h), z, model)
                         for i, mi in enumerate(M)]

        else:

            if isinstance(M, float) or isinstance(M, int):
                M_h = M * self._lens_cosmo.cosmo.h
                _c = concentration(M_h, mdef=mdef, model=model, z=z)

            else:

                if isinstance(z, numpy.ndarray) or isinstance(z, list):
                    assert len(z) == len(M)
                    _c = []
                    for (mi, zi) in zip(M, z):
                        M_h = mi * self._lens_cosmo.cosmo.h
                        _c.append(concentration(M_h, mdef=mdef, model=model, z=zi))

                else:
                    M_h = M * self._lens_cosmo.cosmo.h
                    _c = concentration(M_h, mdef=mdef, model=model, z=z)

        c = numpy.array(_c)

        if scatter:
            _log_c = numpy.log(c)
            c = numpy.random.lognormal(_log_c, scatter_amplitude)

        if logmhm is not None:
            rescale = WDM_concentration_suppresion_factor(M, z, logmhm, suppression_model, kwargs_suppresion)
        else:
            rescale = 1.

        return c * rescale

    def NFW_concentration_custom(self, M_h, z, kwargs_model):

        """

        :param M_h: halo mass in units M_sun / h
        :param z: redshift
        :param kwargs_model: keyword arguments for the mass-concentration relation. Should include c0, beta, and zeta
        The relation is parameterized as a power law in peak height with a normalization c0 at 10^8, a logarithmic
        slope -beta, and a redshift evolution (1+z)^zeta.
        Note: the peak height evolves with redshift on its own, so the (1+z)^zeta term modifies this evolution.
        :return: The concentration of the halo
        """

        Mref_h = 10 ** 8 * self._lens_cosmo.cosmo.h
        nu = peaks.peakHeight(M_h, z)
        nu_ref = peaks.peakHeight(Mref_h, 0)

        if 'c0' in kwargs_model.keys():
            assert kwargs_model['c0'] > 0, 'negative normalizations are unphysical.'
        assert kwargs_model['zeta'] <= 0, 'positive values of zeta are unphysical'

        if 'log10c0' in kwargs_model.keys():
            c0 = 10 ** kwargs_model['log10c0']
        else:
            c0 = kwargs_model['c0']

        c = c0 * (1 + z) ** (kwargs_model['zeta']) * (nu / nu_ref) ** (-kwargs_model['beta'])

        return c

def WDM_concentration_suppresion_factor(halo_mass, z, log_half_mode_mass, suppression_model, kwargs_supression):

    """

    :param halo_mass: the mass of the halo in units M
    (note that the exact units don't mass as long as the half mode mass is in the same units)
    :param z: redshift
    :param log_half_mode_mass: log10 of the half mode mass in a WDM model
    :param suppression_model: the type of suppression, either 'polynomial' or 'hyperbolic'
    :param kwargs_supression: keyword arguments for the suppression function
    :return: the ratio c_wdm over c_cdm
    """

    if 'mc_suppression_redshift_evolution' not in kwargs_supression.keys():
        kwargs_supression['mc_suppression_redshift_evolution'] = \
            halo_default.kwargs_suppression['mc_suppression_redshift_evolution']

    if suppression_model == 'polynomial':
        return _suppression_polynomial(halo_mass, z, log_half_mode_mass, kwargs_supression['c_scale'],
                                       kwargs_supression['c_power'], kwargs_supression['c_power_inner'],
                                       kwargs_supression['mc_suppression_redshift_evolution'])
    elif suppression_model == 'hyperbolic':
        return _suppression_hyperbolic(halo_mass, z, log_half_mode_mass, kwargs_supression['a_mc'],
                                       kwargs_supression['b_mc'])
    else:
        raise Exception('suppression model '+str(suppression_model)+' not recognized; allowed models '
                                                                    'are polynomial and hyperbolic')

def _suppression_hyperbolic(halo_mass, z, log_half_mode_mass, a, b):

    """
    :param halo_mass: halo mass
    :param z: halo redshift
    :param log_half_mode_mass: log10 of half-mode mass
    :param a: the scale where the relation turns over
    :param b: the steepness of the turnover

    The functional form is:
    c_wdm / c_cdm = 0.5 * (1 + tanh((log10(u) - a) / 2b))
    where u = halo_mass / half_mode_mass

    :return: the ratio c_wdm over c_cdm
    """
    if b < 0:
        raise Exception('b parameters < 0 are unphysical')

    mhm = 10 ** log_half_mode_mass

    log10u = numpy.log10(halo_mass / mhm)

    argument = (log10u - a) / (2 * b)

    return 0.5 * (1 + numpy.tanh(argument))

def _suppression_polynomial(halo_mass, z, log_half_mode_mass, c_scale, c_power, c_power_inner,
                            mc_suppression_redshift_evolution=True):

    """
    :param halo_mass: halo mass
    :param z: halo redshift
    :param log_half_mode_mass: log10 of half-mode mass
    :param c_scale: the scale where the relation turns over
    :param c_power: the steepness of the turnover

    The functional form is:
    c_wdm / c_cdm = (1 + c_scale * mhm / m)^c_power * redshift_factor
    where
    redshift_factor = (1+z)^(0.026 * z - 0.04)
    (Bose et al. 2016)

    :return: the ratio c_wdm over c_cdm
    """
    if c_power > 0:
        raise Exception('c_power parameters > 0 are unphysical')
    if c_scale < 0:
        raise Exception('c_scale parameters < 0 are unphysical')

    mhm = 10 ** log_half_mode_mass

    mass_ratio = mhm / halo_mass

    concentration_factor = (1 + c_scale * mass_ratio ** c_power_inner) ** c_power

    if mc_suppression_redshift_evolution:
        redshift_factor = (1 + z) ** (0.026 * z - 0.04)
    else:
        redshift_factor = 1.0

    rescale = redshift_factor * concentration_factor

    return rescale

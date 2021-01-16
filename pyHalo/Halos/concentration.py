import numpy as numpy
from colossus.halo.concentration import concentration, peaks

import warnings
warnings.filterwarnings("ignore")

class Concentration(object):

    def __init__(self, lens_cosmo):

        """
        This class handles concentrations of the mass-concentration relation for NFW profiles
        :param lens_cosmo: an instance of LensCosmo
        """

        self._lens_cosmo = lens_cosmo

    def NFW_concentration(self, M, z, model, mdef, logmhm,
                          scatter, c_scale, c_power, scatter_amplitude):

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
        :param c_scale: modifies the mass-concentration relation in warm dark matter models (see below)
        :param c_power: modifies the mass-concentration relation in warm dark matter models (see below)
        :param scatter_amplitude: the amplitude of the scatter in the mass-concentration relation in dex
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
            rescale = WDM_concentration_suppresion_factor(M, z, logmhm, c_scale, c_power)
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

        assert kwargs_model['beta'] >= 0, 'beta values < 0 are unphysical.'
        assert kwargs_model['c0'] > 0, 'negative normalizations are unphysical.'
        assert kwargs_model['zeta']<=0, 'positive values of zeta are unphysical'

        c = kwargs_model['c0'] * (1 + z) ** (kwargs_model['zeta']) * (nu / nu_ref) ** (-kwargs_model['beta'])

        return c

def WDM_concentration_suppresion_factor(halo_mass, z, log_half_mode_mass, c_scale, c_power):

    """

    :param halo_mass: the mass of the halo in units M
    (note that the exact units don't mass as long as the half mode mass is in the same units)
    :param z: redshift
    :param log_half_mode_mass: log10 of the half mode mass in a WDM model
    :param c_scale: coefficient that multiplies the mass ratio M_hm / M
    :param c_power: exponent of the suppression factor

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

    concentration_factor = (1 + c_scale * mass_ratio) ** c_power

    redshift_factor = (1 + z) ** (0.026 * z - 0.04)

    rescale = redshift_factor * concentration_factor

    return rescale

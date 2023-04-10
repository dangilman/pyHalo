import numpy as numpy
from colossus.halo.concentration import concentration, peaks
import warnings
warnings.filterwarnings("ignore")

class ConcentrationWDMPolynomial(object):

    def __init__(self, lens_cosmo, concentration_cdm_class, log_mc, c_scale=60.0,
                 c_power=-0.17, c_power_inner=1.0, mc_suppression_redshift_evolution=True, kwargs_cdm={}):
        """

        :param lens_cosmo: an instance of LensCosmo
        :param concentration_cdm_class: a concentration class for CDM
        :param c_scale: the leading coefficient of the suppression term (see below)
        :param c_power: the exponent outside the parenthesis of the suppression term (see equation below)
        :param c_power_inner: the exponent inside the parenthesis of the suppression term (see equation below)
        :param mc_suppression_redshift_evolution: bool; adds the (mild) redshift evolution from Bose et al. (2016)
        :param kwargs_cdm: keyword arguments for the CDM concentration class
        """
        self._cdm_concentration_cdm = concentration_cdm_class(lens_cosmo, **kwargs_cdm)
        if c_power > 0:
            raise Exception('c_power parameters > 0 are unphysical')
        if c_scale < 0:
            raise Exception('c_scale parameters < 0 are unphysical')

        self._log_mc = log_mc
        self._c_scale = c_scale
        self._c_power = c_power
        self._c_power_inner = c_power_inner
        self._mc_suppression_redshift_evolution = mc_suppression_redshift_evolution

    def suppression(self, m, z):
        """

        :param m: halo mass [same units as log_mc]
        :param z: halo redshift
        :param log_mc: mass scale where suppresion kicks in [same units as m]
        :return:
        """

        mhm = 10 ** self._log_mc
        rescale_factor = (1 + self._c_scale * (mhm / m) ** self._c_power_inner) ** self._c_power
        if self._mc_suppression_redshift_evolution:
            redshift_factor = (1 + z) ** (0.026 * z - 0.04)
        else:
            redshift_factor = 1.0
        rescale = redshift_factor * rescale_factor
        return rescale

class ConcentrationWDMHyperbolic(object):

    def __init__(self, lens_cosmo, concentration_cdm_class, log_mc, a,  b, kwargs_cdm={}):
        """

        :param lens_cosmo:
        :param concentration_cdm_class:
        :param kwargs_cdm:
        """
        self._cdm_concentration_cdm = concentration_cdm_class(lens_cosmo, **kwargs_cdm)
        self._a = a
        self._b = b
        self._log_mc = log_mc

    def suppression(self, m, z):
        """

        :param m:
        :param z:
        :param log_mc:
        :param a:
        :param b:
        :return:
        """
        if b < 0:
            raise Exception('b parameters < 0 are unphysical')
        mhm = 10 ** self._log_mc
        log10u = numpy.log10(m / mhm)
        argument = (log10u - self._a) / (2 * self._b)
        return 0.5 * (1 + numpy.tanh(argument))

class _ConcentrationCDM(object):

    def nfw_concentration(self, M, z, scatter=True, scatter_amplitude=0.2):
        """

        :param M: halo mass
        :param z: halo redshift
        :param scatter: bool; add log-normal scatter to concentration
        :param scatter_amplitude: the amount of scatter in dex, assumes log-normal distribution
        :return:
        """

        c = self.evaluate_concentration(M, z)
        if scatter:
            log_c = numpy.log(c)
            c = numpy.random.lognormal(log_c, scatter_amplitude)
        return c * self.suppression_fuction()

    def evaluate_concentration(self, *args, **kargs):
        raise Exception('Custom concentration class must have a method evaluate_concentration with inputs mass, redshift')

    def suppression_fuction(self, *args, **kwargs):
        return 1.0

class ConcentrationDiemerJoyce(_ConcentrationCDM):

    def __init__(self, lens_cosmo, *args, **kwargs):
        """
        This class handles concentrations of the mass-concentration relation for NFW profiles
        :param lens_cosmo: an instance of LensCosmo
        """
        self._lens_cosmo = lens_cosmo

    def evaluate_concentration(self, M, z):

        """
        Evaluates the concentration of an NFW profile

        :param M: halo mass; m200 with respect to critical density of the Universe at redshift z
        :param z: redshift
        :param model:
        :param mdef:
        :param logmhm:
        :param scatter:
        :param scatter_amplitude:
        :param kwargs_suppresion:
        :param suppression_model:
        :return:
        """

        model = 'diemer19'
        mdef = '200c'
        if isinstance(M, float) or isinstance(M, int):
            M_h = M * self._lens_cosmo.cosmo.h
            c = concentration(M_h, mdef=mdef, model=model, z=z)
        else:
            if isinstance(z, numpy.ndarray) or isinstance(z, list):
                assert len(z) == len(M)
                c = []
                for (mi, zi) in zip(M, z):
                    M_h = mi * self._lens_cosmo.cosmo.h
                    c.append(concentration(M_h, mdef=mdef, model=model, z=zi))
            else:
                M_h = M * self._lens_cosmo.cosmo.h
                c = concentration(M_h, mdef=mdef, model=model, z=z)
        return c

class ConcentrationPeakHeight(_ConcentrationCDM):

    def __init__(self, lens_cosmo, c0, zeta, beta):
        """
        This class handles concentrations of the mass-concentration relation for NFW profiles
        :param lens_cosmo: an instance of LensCosmo
        """
        self._lens_cosmo = lens_cosmo
        self._c0 = c0
        if zeta > 0:
            raise Exception('positive values of zeta are unphysical')
        self._zeta = zeta
        self._beta = beta

    def evaluate_concentration(self, M, z):

        """

        :param M: halo mass
        :param z: redshift
        :return: concentration
        """
        M_h = M * self._lens_cosmo.h
        Mref_h = 10 ** 8 * self._lens_cosmo.cosmo.h
        nu = peaks.peakHeight(M_h, z)
        nu_ref = peaks.peakHeight(Mref_h, 0)
        c = self._c0 * (1 + z) ** self._zeta * (nu / nu_ref) ** (-self._beta)
        return c

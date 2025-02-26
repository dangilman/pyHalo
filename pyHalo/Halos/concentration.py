import numpy as numpy
from colossus.halo.concentration import concentration, peaks
import warnings
import numpy
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore")

__all__ = ['ConcentrationDiemerJoyce',
           'ConcentrationWDMHyperbolic',
           'ConcentrationWDMPolynomial',
           'ConcentrationPeakHeight']

class ConcentrationConstant(object):
    name = 'CONSTANT'
    def __init__(self, cosmo, c):
        self._c = c
    def nfw_concentration(self, *args, **kwargs):
        return self._c

class _ConcentrationCDM(object):
    _universal_minimum = 1.2 # no concentrations less than this
    def __init__(self, cosmo, scatter=True, scatter_dex=0.2, scatter_dex_z_dep=0.0, *args, **kwargs):
        """
        This class handles concentrations of the mass-concentration relation for NFW profiles
        :param lens_cosmo: an instance of LensCosmo
        :param scatter: include scatter the concentration
        :param scatter_dex: scatter the concentration
        :param scatter_dex_z_dep: redshift dependence of the scatter
        """
        self._cosmo = cosmo
        self._scatter = scatter
        self._scatter_dex = scatter_dex
        self._scatter_dex_z_dep = scatter_dex_z_dep

    def nfw_concentration(self, m, z, force_no_scatter=False):
        """
        Evaluates the concentration of a halo of mass 'm' at redshift z
        :param M: halo mass [M_sun]
        :param z: halo redshift
        :param force_no_scatter: bool; if True, will return the median concentration
        :return:
        """
        if isinstance(m, float) or isinstance(m, int):
            c = float(self._evaluate_concentration(m, z))
        else:
            c = numpy.array([self._evaluate_concentration(mi, z) for mi in m])
        if self._scatter:
            if force_no_scatter:
                pass
            else:
                log_c = numpy.log(c)
                sigma = self._scatter_dex + self._scatter_dex_z_dep * z
                c = numpy.random.lognormal(log_c, sigma)
        if isinstance(c, float):
            c = max(c, self._universal_minimum)
        else:
            c[numpy.where(c < self._universal_minimum)] = self._universal_minimum
        return c

    def _evaluate_concentration(self, *args, **kargs):
        raise Exception(
            'Custom concentration class must have a method evaluate_concentration with inputs mass, redshift')

class _ConcentrationTurnover(object):
    _universal_minimum = 1.2
    def __init__(self, cdm_concentration):
        """

        :param cdm_concentration: an instantiated CDM concentration-mass relation class
        """
        self._cdm_concentration = cdm_concentration

    def nfw_concentration(self, m, z):
        """
        Evaluates the concentration of a halo of mass 'm' at redshift z
        :param M: halo mass [M_sun]
        :param z: halo redshift
        :return:
        """
        c_cdm = self._cdm_concentration.nfw_concentration(m, z)
        c_wdm = c_cdm * self.suppression(m, z)
        if isinstance(c_wdm, float):
            c_wdm = max(c_wdm, self._universal_minimum)
        else:
            c_wdm[numpy.where(c_wdm < self._universal_minimum)] = self._universal_minimum
        return c_wdm

    def suppression(self, *args, **kwargs):
        raise Exception('a WDM model with a turnover must have a suppression function')

class ConcentrationDiemerJoyce(_ConcentrationCDM):

    name = 'DIEMERJOYCE19'

    def __init__(self, cosmo, scatter=True, scatter_dex=0.2, mdef='200c', *args, **kwargs):
        """
        This class handles concentrations of the mass-concentration relation for NFW profiles
        :param cosmo: an instance of astropy cosmology
        """
        self._mdef = mdef
        super(ConcentrationDiemerJoyce, self).__init__(cosmo, scatter, scatter_dex)

    def _evaluate_concentration(self, M, z):

        """
        Evaluates the concentration of an NFW profile

        :param M: halo mass; m200 with respect to critical density of the Universe at redshift z
        :param z: redshift
        :return: halo concentratioon
        """
        model = 'diemer19'
        if isinstance(M, float) or isinstance(M, int):
            M_h = M * self._cosmo.h
            c = concentration(M_h, mdef=self._mdef, model=model, z=z)
        else:
            if isinstance(z, numpy.ndarray) or isinstance(z, list):
                assert len(z) == len(M)
                c = []
                for (mi, zi) in zip(M, z):
                    M_h = mi * self._cosmo.h
                    c.append(concentration(M_h, mdef=self._mdef, model=model, z=zi))
            else:
                M_h = M * self._cosmo.h
                c = concentration(M_h, mdef=self._mdef, model=model, z=z)
        return c

class ConcentrationLudlow(_ConcentrationCDM):

    name = 'LUDLOW2016'

    def __init__(self, cosmo, scatter=True, scatter_dex=0.2, mdef='200c', *args, **kwargs):
        """
        This class handles concentrations of the mass-concentration relation for NFW profiles
        :param cosmo: an instance of astropy cosmology
        """
        self._cosmo = cosmo
        self._mdef = mdef
        super(ConcentrationLudlow, self).__init__(cosmo, scatter, scatter_dex)

    def _evaluate_concentration(self, M, z):

        """
        Evaluates the concentration of an NFW profile
        :param M: halo mass; m200 with respect to critical density of the Universe at redshift z
        :param z: redshift
        :return: halo concentratioon
        """
        M_h = M * self._cosmo.h
        z = min(15.0, z)
        c = concentration(M_h, mdef=self._mdef, model='ludlow16', z=z)
        return c

class ConcentrationPeakHeight(_ConcentrationCDM):

    name = 'PEAK_HEIGHT_POWERLAW'

    def __init__(self, cosmo, c0, zeta, beta, scatter=True, scatter_dex=0.2):
        """
        This class handles concentrations of the mass-concentration relation for NFW profiles
        :param cosmo: an instance of astropy cosmology
        :param c0: the amplitude of the concentration-mass relation at 10^8 M_sun at z=0
        :param zeta: modifies the logarithmic slope of the concentration-mass relation (1+z)^zeta
        :param beta: the logarithmic slope of the concentration-mass relation in peak height
        :param scatter: bool; whether to include scatter in concentration-mass relation
        :param scatter_dex: scatter in concentration in dex
        """
        self._c0 = c0
        self._zeta = zeta
        self._beta = beta
        self._redshift_evolution = _zEvolutionPeakHeight(cosmo)
        super(ConcentrationPeakHeight, self).__init__(cosmo, scatter, scatter_dex)

    def _evaluate_concentration(self, M, z):

        """
        Evaluates the concentration of an NFW profile

        :param M: halo mass; m200 with respect to critical density of the Universe at redshift z
        :param z: redshift
        :return: halo concentratioon
        """
        M_h = M * self._cosmo.h
        Mref_h = 10 ** 8 * self._cosmo.h
        nu = peaks.peakHeight(M_h, z)
        nu_ref = peaks.peakHeight(Mref_h, z)
        redshift_factor = self._redshift_evolution(M, z)
        c = self._c0 * (nu / nu_ref) ** -self._beta * redshift_factor ** self._zeta
        return c

class ConcentrationWDMPolynomial(_ConcentrationTurnover):

    name = 'WDM_POLYNOMIAL'

    def __init__(self, cosmo, concentration_cdm_class, log_mc, c_scale=60.0,
                 c_power=-0.17, c_power_inner=1.0, mc_suppression_redshift_evolution=True, scatter=True,
                 scatter_dex=0.2, kwargs_cdm={}):
        """

        :param cosmo: an instance of astropy cosmology
        :param concentration_cdm_class: a concentration class for CDM
        :param c_scale: the leading coefficient of the suppression term (see below)
        :param c_power: the exponent outside the parenthesis of the suppression term (see equation below)
        :param c_power_inner: the exponent inside the parenthesis of the suppression term (see equation below)
        :param mc_suppression_redshift_evolution: bool; adds the (mild) redshift evolution from Bose et al. (2016)
        :param scatter: bool; whether to include scatter in concentration-mass relation
        :param scatter_dex: scatter in concentration in dex
        :param kwargs_cdm: keyword arguments for the CDM concentration class
        """
        if 'scatter' not in kwargs_cdm.keys():
            kwargs_cdm['scatter'] = scatter
        if 'scatter_dex' not in kwargs_cdm.keys():
            kwargs_cdm['scatter_dex'] = scatter_dex
        cdm_concentration = concentration_cdm_class(cosmo, **kwargs_cdm)
        if c_power > 0:
            raise Exception('c_power parameters > 0 are unphysical')
        if c_scale < 0:
            raise Exception('c_scale parameters < 0 are unphysical')
        self._log_mc = log_mc
        self._c_scale = c_scale
        self._c_power = c_power
        self._c_power_inner = c_power_inner
        self._mc_suppression_redshift_evolution = mc_suppression_redshift_evolution
        self._redshift_evolution = _zEvolutionBose2016()
        super(ConcentrationWDMPolynomial, self).__init__(cdm_concentration)

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
            redshift_factor = self._redshift_evolution(m, z)
        else:
            redshift_factor = 1.0
        rescale = redshift_factor * rescale_factor
        return rescale

class ConcentrationWDMHyperbolic(_ConcentrationTurnover):
    name = 'WDM_HYPERBOLIC'
    def __init__(self, cosmo, concentration_cdm_class, log_mc, a,  b, scatter=True,
                 scatter_dex=0.2, kwargs_cdm={}):
        """

        :param cosmo:
        :param concentration_cdm_class:
        :param kwargs_cdm:
        """
        if 'scatter' not in kwargs_cdm.keys():
            kwargs_cdm['scatter'] = scatter
        if 'scatter_dex' not in kwargs_cdm.keys():
            kwargs_cdm['scatter_dex'] = scatter_dex
        cdm_concentration = concentration_cdm_class(cosmo, **kwargs_cdm)
        self._a = a
        self._b = b
        self._log_mc = log_mc
        super(ConcentrationWDMHyperbolic, self).__init__(cdm_concentration)

    def suppression(self, m, z):
        """

        :param m:
        :param z:
        :param log_mc:
        :param a:
        :param b:
        :return:
        """
        mhm = 10 ** self._log_mc
        log10u = numpy.log10(m / mhm)
        argument = (log10u - self._a) / (2 * self._b)
        return 0.5 * (1 + numpy.tanh(argument))

class ConcentrationLudlowWDM(_ConcentrationTurnover):
    name = 'LUDLOW_WDM'
    _universal_minimum = 1.2  # no concentrations less than this
    points = (numpy.array([1.0, 2.0, 2.7, 3.2, 4.0]),
              numpy.array([6.0, 7.0, 8.0]),
              numpy.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    values_a = numpy.array([ 0.554,  0.484,  0.432,  0.276,  0.116,  0.474,  0.46 ,  0.376,
        0.206,  0.053,  0.486,  0.429,  0.319,  0.12 , -0.048,  0.497,
        0.457,  0.37 ,  0.288,  0.131,  0.472,  0.446,  0.35 ,  0.216,
        0.057,  0.477,  0.393,  0.329,  0.134, -0.021,  0.49 ,  0.412,
        0.367,  0.268,  0.121,  0.454,  0.406,  0.343,  0.24 ,  0.074,
        0.455,  0.388,  0.287,  0.123, -0.017,  0.453,  0.428,  0.363,
        0.271,  0.128,  0.442,  0.412,  0.337,  0.225,  0.072,  0.409,
        0.392,  0.282,  0.146,  0.004,  0.432,  0.414,  0.348,  0.255,
        0.103,  0.425,  0.419,  0.318,  0.166,  0.065,  0.412,  0.379,
        0.283,  0.119,  0.01 ])
    values_b = numpy.array([0.805, 0.766, 0.85 , 0.927, 0.938, 0.739, 0.782, 0.868, 0.952,
       1.015, 0.782, 0.778, 0.892, 0.946, 1.045, 0.545, 0.513, 0.552,
       0.617, 0.715, 0.533, 0.574, 0.606, 0.711, 0.721, 0.549, 0.56 ,
       0.638, 0.702, 0.749, 0.499, 0.46 , 0.495, 0.543, 0.646, 0.468,
       0.485, 0.553, 0.57 , 0.686, 0.498, 0.509, 0.56 , 0.64 , 0.684,
       0.457, 0.464, 0.499, 0.492, 0.631, 0.445, 0.471, 0.499, 0.555,
       0.657, 0.45 , 0.504, 0.518, 0.641, 0.685, 0.42 , 0.447, 0.477,
       0.483, 0.584, 0.42 , 0.476, 0.501, 0.565, 0.622, 0.424, 0.473,
       0.514, 0.59 , 0.665])
    interp_a = RegularGridInterpolator(points, values_a.reshape(5, 3, 5))
    interp_b = RegularGridInterpolator(points, values_b.reshape(5, 3, 5))

    def __init__(self, cosmo, log_mc, dlogT_dlogk, scatter=True, scatter_dex=0.2, mdef='200c'):
        """

        :param cosmo:
        :param log_mc:
        :param dlogT_dlogk:
        :param scatter:
        :param scatter_dex:
        :param mdef:
        """
        if dlogT_dlogk > 0:
            raise Exception('positive logarithmic derivatives are unphysical')
        self._dlogT_dlogk = dlogT_dlogk
        self._log_mc = log_mc
        cdm_concentration = ConcentrationLudlow(cosmo, scatter, scatter_dex, mdef)
        super(ConcentrationLudlowWDM, self).__init__(cdm_concentration)

    @staticmethod
    def _make_in_bounds(log10_mhm, dlogT_dlogk, z):
        """
        Forces the values of log10_mhm, dlogT_dlogk, and z to be inside the range of interpolation
        :return: the values of these parameters
        """
        log10_mhm_eval = max(6.0, log10_mhm)
        log10_mhm_eval = min(8.0, log10_mhm_eval)
        dlogT_dlogk_eval = -1.0 * dlogT_dlogk
        dlogT_dlogk_eval = max(1.0, dlogT_dlogk_eval)
        dlogT_dlogk_eval = min(4.0, dlogT_dlogk_eval)
        z_eval = max(z, 0.0)
        z_eval = min(z_eval, 4.0)
        return log10_mhm_eval, dlogT_dlogk_eval, z_eval

    def suppression_fit(self, log10_mhm, dlogT_dlogk, z):
        """
        Evaluates the coefficients of the hyperbolic suppression term
        :param log10_mhm: log10 half-mode mass
        :param dlogT_dlogk: logarithmic derivative at the half-mode scale k_1/2
        :param z: redshift
        :return: suppression of the WDM relation relative to CDM
        """
        # we will only evaluate this model around the scales where it was calibrated; i.e. no extrapolation
        log10_mhm_eval, dlogT_dlogk_eval, z_eval = self._make_in_bounds(log10_mhm, dlogT_dlogk, z)
        x = (dlogT_dlogk_eval, log10_mhm_eval, z_eval)
        a = self.interp_a(x)
        b = self.interp_b(x)
        return numpy.squeeze(a), numpy.squeeze(b)

    def nfw_concentration(self, m, z):
        """
        Evaluates the concentration of a halo of mass 'm' at redshift z
        :param M: halo mass [M_sun]
        :param z: halo redshift
        :return: halo concentration
        """
        c_cdm = self._cdm_concentration.nfw_concentration(m, z)
        c_wdm = c_cdm * self.suppression(m, z)
        if isinstance(c_wdm, float):
            c_wdm = max(c_wdm, self._universal_minimum)
        else:
            c_wdm[numpy.where(c_wdm < self._universal_minimum)] = self._universal_minimum
        return c_wdm

    def suppression(self, m, z):
        """
        Evaluates the suppression of the concentration mass relation such that c_wdm = c_cdm * suppression
        :param m: halo mass [solar mass]
        :param z: redshift
        :return: the suppression factor of the WDM concentration-mass relation
        """
        mhm = 10 ** self._log_mc
        a, b = self.suppression_fit(self._log_mc, self._dlogT_dlogk, z)
        a = numpy.squeeze(a)
        b = numpy.squeeze(b)
        log10u = numpy.log10(m / mhm)
        arg = (log10u - a) / (2 * b)
        return 0.5 * (1 + numpy.tanh(arg))

class _zEvolutionPeakHeight(object):

    def __init__(self, cosmo):
        self._cosmo = cosmo

    def __call__(self, m, z):
        """
        This method evaluates the redshift evolution according to the redshift evolution of the peak height
        :param m: halo mass in units 200c
        :param z: redshift
        :return: the relative evolution of the peak height between z=0 and z=z
        Note that the choice to divide by peaks.peakHeight(M_h, z) causes a decrease in amplitude with increasing
        redshift at fixed mass
        """
        M_h = m * self._cosmo.h
        redshift_factor = peaks.peakHeight(M_h, 0.0) / peaks.peakHeight(M_h, z)
        return redshift_factor

class _zEvolutionBose2016(object):

    def __call__(self, m, z):
        """
        This method evaluates the redshift evolution according to the redshift evolution for WDM halos presented by
        Bose 2016
        :param m: halo mass in units 200c
        :param z: redshift
        :return: the scaling with redshift of the Bose et al. (2016) model
        """
        redshift_factor = (1 + z) ** (0.026 * z - 0.04)
        return redshift_factor


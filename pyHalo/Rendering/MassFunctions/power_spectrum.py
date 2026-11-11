"""
Halo mass functions computed from a linear matter power spectrum.

The multiplicity function is Equation 8 of Benson, Nadler, Du & Gluscevic (2026), arXiv:2606.12137,

dn/dM = sqrt(2/pi) A (rho_m / M^2) |dln(sigma)/dln(M)| (1 + (a nu^2)^-p) (a nu^2)^(q/2) exp(-c (a nu^2)^b / 2)

with nu = delta_c / sigma(M, z). By default it is used with their parameters and their window function (the extended
ETHOS window; see pyHalo.Cosmology.power_spectrum), which is the "intrinsic" halo mass function of Benson et al.: it
excludes subhalos and backsplash halos, and in LCDM it sits 10-20% below Sheth-Tormen between 10^8 and 10^12 M_sun/h.
With b = c = q = 1 the same expression is the Sheth & Tormen (1999) mass function; with the top-hat window this
reproduces the Sheth-Tormen mass function pyHalo evaluates through colossus.

The calibration of these parameters and the conventions they assume are documented in
notebooks/calibrate_benson_hmf.ipynb. The paper does not give the sign of p explicitly; it is taken as negative there.
"""
import numpy as np
from pyHalo.Cosmology.power_spectrum import DELTA_COLLAPSE
from pyHalo.Rendering.MassFunctions.tabulated import TabulatedMassFunction

__all__ = ['PowerSpectrumMassFunction', 'PowerSpectrumSubhaloMassFunction', 'MULTIPLICITY_FUNCTION_PARAMS',
           'multiplicity_function']

# Benson et al. (2026) Equation 8 parameters from their Figure 6 posteriors; Sheth & Tormen (1999) in the same form
MULTIPLICITY_FUNCTION_PARAMS = {'BENSON2026': {'A': 0.291, 'a': 0.938, 'b': 1.066, 'c': 0.991, 'p': -0.399,
                                               'q': 0.361},
                                'SHETH_TORMEN': {'A': 0.3222, 'a': 0.707, 'b': 1.0, 'c': 1.0, 'p': 0.3, 'q': 1.0}}
# the window function each multiplicity function was calibrated with
_DEFAULT_FILTER = {'BENSON2026': 'BENSON2026', 'SHETH_TORMEN': 'TOPHAT'}


def multiplicity_function(sigma, A, a, b, c, p, q, delta_c=DELTA_COLLAPSE):
    """
    Equation 8 of Benson et al. (2026), written as f(sigma) such that dn/dM = f (rho_m / M^2) |dln(sigma)/dln(M)|

    :param sigma: the rms variance of the linear density field at the halo mass and redshift
    :param A, a, b, c, p, q: parameters of the multiplicity function
    :param delta_c: the linear collapse threshold
    :return: f(sigma)
    """
    a_nu2 = a * (delta_c / sigma) ** 2
    return np.sqrt(2 / np.pi) * A * (1 + a_nu2 ** -p) * a_nu2 ** (q / 2) * np.exp(-c * a_nu2 ** b / 2)


class PowerSpectrumMassFunction(TabulatedMassFunction):
    """
    The halo mass function computed from a linear power spectrum at each lens plane, and sampled as a
    TabulatedMassFunction with nodes evenly spaced in log10(m)

    kwargs_model for this class include:
    1) log_mlow: minimum halo mass to render (log base 10)
    2) log_mhigh: maximum halo mass to render (log base 10)
    3) power_spectrum: an instance of LinearPowerSpectrum (pyHalo.Cosmology.power_spectrum), created with the same
    cosmology as the realization
    4) multiplicity_function: BENSON2026 (default) or SHETH_TORMEN
    5) kwargs_multiplicity: keyword arguments that replace the default parameters of the multiplicity function
    (A, a, b, c, p, q)
    6) filter: the window function used to compute sigma; defaults to the window the multiplicity function was
    calibrated with (BENSON2026 for BENSON2026, TOPHAT for SHETH_TORMEN)
    7) LOS_normalization: rescales the amplitude of the mass function (default 1)
    8) m_pivot: the pivot mass for delta_power_law_index (default 10^8)
    9) delta_power_law_index: tilts the logarithmic slope of the mass function around m_pivot (default 0)
    10) nodes_per_dex: number of nodes per decade in halo mass (default 20)
    11) draw_poisson: bool; draw the number of halos from a Poisson distribution
    """
    name = 'POWER_SPECTRUM'

    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, kwargs_model):
        """

        :param z: redshift of the lens plane
        :param delta_z: thickness of the lens plane
        :param geometry_class: an instance of Geometry
        :param kwargs_model: keyword arguments for the mass function (see the class docstring)
        :return: an instance of PowerSpectrumMassFunction
        """
        log_mlow, log_mhigh = kwargs_model['log_mlow'], kwargs_model['log_mhigh']
        draw_poisson = kwargs_model['draw_poisson']
        if log_mlow >= log_mhigh:
            return PowerSpectrumMassFunction(log_mlow, log_mhigh, np.array([1.0, 2.0]), np.zeros(2), draw_poisson)
        power_spectrum = kwargs_model['power_spectrum']
        power_spectrum.check_cosmology(geometry_class.cosmo)
        n_nodes = max(2, int(np.ceil((log_mhigh - log_mlow) * kwargs_model.get('nodes_per_dex', 20))) + 1)
        m = np.logspace(log_mlow, log_mhigh, n_nodes)
        dN_dm = cls.dndm_comoving(m, z, kwargs_model) * \
            cls._volume_normalization(m, z, delta_z, geometry_class, kwargs_model)
        return PowerSpectrumMassFunction(log_mlow, log_mhigh, m, dN_dm, draw_poisson)

    @staticmethod
    def dndm_comoving(m, z, kwargs_model):
        """
        The comoving differential halo mass function

        :param m: halo mass [M_sun]
        :param z: redshift
        :param kwargs_model: keyword arguments for the mass function (see the class docstring)
        :return: dn/dm [Mpc^-3 M_sun^-1]
        """
        power_spectrum = kwargs_model['power_spectrum']
        model = kwargs_model.get('multiplicity_function', 'BENSON2026')
        if model not in MULTIPLICITY_FUNCTION_PARAMS:
            raise ValueError('multiplicity_function must be one of ' + str(list(MULTIPLICITY_FUNCTION_PARAMS.keys())))
        params = dict(MULTIPLICITY_FUNCTION_PARAMS[model])
        kwargs_multiplicity = kwargs_model.get('kwargs_multiplicity', None)
        if kwargs_multiplicity is not None:
            for key in kwargs_multiplicity.keys():
                if key not in params:
                    raise ValueError('unrecognized multiplicity function parameter ' + str(key))
            params.update(kwargs_multiplicity)
        filter = kwargs_model.get('filter', _DEFAULT_FILTER[model])
        sigma = power_spectrum.sigma(m, z, filter)
        alpha = np.abs(power_spectrum.dlnsigma_dlnm(m, filter))
        return multiplicity_function(sigma, **params) * power_spectrum.rho_m0 / m ** 2 * alpha


class PowerSpectrumSubhaloMassFunction(TabulatedMassFunction):
    """
    A subhalo mass function at infall computed from a linear power spectrum: the power law of CDMPowerLaw, times the
    ratio R(m) of the field halo mass function computed from the power spectrum to the one computed from a reference
    (LCDM) power spectrum, evaluated at a fixed redshift z_eval that stands for a typical infall redshift, and normalized
    to one at m_pivot:

    dN/dm = normalization * m^power_law_index * R(m, z_eval) / R(m_pivot, z_eval)

    The subhalo renderer sets normalization from sigma_sub (or dndA) at m_pivot, as for CDMPowerLaw; the
    BinnedPowerSpectrum preset rescales sigma_sub by R(m_pivot, z_eval), so the subhalo mass function there is
    sigma_sub * (power law) * R(m, z_eval). The ratio is evaluated at nodes evenly spaced in log10(m) and the mass
    function is sampled as a TabulatedMassFunction (a power law between nodes); with the reference power spectrum it is
    exactly the power law of CDMPowerLaw.

    Keyword arguments for this class, besides those the subhalo renderer passes, include:
    1) power_spectrum: an instance of LinearPowerSpectrum
    2) power_spectrum_reference: the LinearPowerSpectrum of the reference (LCDM) model
    3) z_eval: the redshift at which the ratio of mass functions is evaluated
    4) multiplicity_function, kwargs_multiplicity, filter: as for PowerSpectrumMassFunction
    5) m_pivot: the mass at which the ratio is normalized (default 10^8)
    6) nodes_per_dex: number of nodes per decade in halo mass (default 20)
    """
    name = 'POWER_SPECTRUM_SHMF'

    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson, normalization, power_spectrum,
                 power_spectrum_reference, z_eval, m_pivot=10 ** 8, nodes_per_dex=20,
                 multiplicity_function='BENSON2026', kwargs_multiplicity=None, filter=None, **kwargs):
        """

        :param log_mlow: log10(minimum halo mass)
        :param log_mhigh: log10(maximum halo mass)
        :param power_law_index: logarithmic slope of the power law
        :param draw_poisson: bool; whether to draw the number of halos from a Poisson distribution
        :param normalization: the amplitude of the power law, dN/dm = normalization * m^power_law_index
        :param power_spectrum: an instance of LinearPowerSpectrum
        :param power_spectrum_reference: the LinearPowerSpectrum of the reference (LCDM) model
        :param z_eval: the redshift at which the ratio of mass functions is evaluated
        :param m_pivot: the mass at which the ratio is normalized to one
        :param nodes_per_dex: number of nodes per decade in halo mass
        :param multiplicity_function: BENSON2026 (default) or SHETH_TORMEN
        :param kwargs_multiplicity: keyword arguments that replace the default parameters of the multiplicity function
        :param filter: the window function used to compute sigma; see PowerSpectrumMassFunction
        :param kwargs: other keyword arguments from the subhalo renderer, which are not used
        """
        if log_mlow >= log_mhigh:
            super(PowerSpectrumSubhaloMassFunction, self).__init__(log_mlow, log_mhigh, np.array([1.0, 2.0]),
                                                                   np.zeros(2), draw_poisson)
            return
        n_nodes = max(2, int(np.ceil((log_mhigh - log_mlow) * nodes_per_dex)) + 1)
        m = np.logspace(log_mlow, log_mhigh, n_nodes)
        ratio = self.mass_function_ratio(np.append(m, m_pivot), z_eval, power_spectrum, power_spectrum_reference,
                                         multiplicity_function, kwargs_multiplicity, filter)
        dN_dm = normalization * m ** power_law_index * ratio[:-1] / ratio[-1]
        super(PowerSpectrumSubhaloMassFunction, self).__init__(log_mlow, log_mhigh, m, dN_dm, draw_poisson)

    @staticmethod
    def mass_function_ratio(m, z, power_spectrum, power_spectrum_reference, multiplicity_function='BENSON2026',
                            kwargs_multiplicity=None, filter=None):
        """
        The ratio R(m, z) of the field halo mass function computed from power_spectrum to the one computed from
        power_spectrum_reference

        :param m: halo mass [M_sun]; array
        :param z: redshift
        :param power_spectrum: an instance of LinearPowerSpectrum
        :param power_spectrum_reference: the LinearPowerSpectrum of the reference model
        :param multiplicity_function: BENSON2026 (default) or SHETH_TORMEN
        :param kwargs_multiplicity: keyword arguments that replace the default parameters of the multiplicity function
        :param filter: the window function used to compute sigma; see PowerSpectrumMassFunction
        :return: R(m, z)
        """
        kwargs_model = {'multiplicity_function': multiplicity_function}
        if kwargs_multiplicity is not None:
            kwargs_model['kwargs_multiplicity'] = kwargs_multiplicity
        if filter is not None:
            kwargs_model['filter'] = filter
        dndm = PowerSpectrumMassFunction.dndm_comoving(m, z, dict(kwargs_model, power_spectrum=power_spectrum))
        dndm_reference = PowerSpectrumMassFunction.dndm_comoving(m, z, dict(kwargs_model,
                                                                            power_spectrum=power_spectrum_reference))
        return dndm / dndm_reference

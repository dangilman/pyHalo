"""
Mass function suppression calibrated to the halo mass function of Benson, Nadler, Du &
Gluscevic (2026), arXiv:2606.12137.

The suppression is expressed through the mass scales M20, M50 and M80 at which the ratio
n_wdm/n_cdm reaches 0.2, 0.5 and 0.8, in units of the half-mode mass. These are tabulated on a
grid of (|dlogT/dlogk|, z) and converted to the (a, b, c) of the standard turnover
(1 + a (m_hm/m)^b)^c using the same inversion as the Stucker et al. (2022) calibration.

Mass scales are tabulated rather than (a, b, c) because the latter are strongly degenerate: a
direct least-squares fit reproduces each suppression curve to better than 0.005 while c wanders
over three orders of magnitude across the grid, which makes interpolation meaningless. The mass
scales are smooth and monotonic in redshift everywhere.

Unlike the Stucker calibration, this one is redshift dependent. The suppression at m = m_hm
strengthens by about 25 per cent between z = 0 and z = 10; below z ~ 2 the dependence is
negligible, which is why a redshift-independent fit is adequate at low redshift.

Note this calibrates the HALO mass function. Benson et al. exclude subhalos and backsplash
halos by construction, so there is no subhalo equivalent and the subhalo mass function should
continue to use the Stucker calibration.
"""
import numpy
from scipy.interpolate import RegularGridInterpolator
from pyHalo.Rendering.MassFunctions.stucker import mscales_to_abc
from pyHalo.Rendering.MassFunctions.mass_function_base import WDMPowerLaw
from pyHalo.Rendering.MassFunctions.density_peaks import evaluate_mass_function

__all__ = ['benson_suppression_params', 'ShethTormenTurnoverBenson']

# (|dlogT/dlogk|, z)
points = (numpy.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
          numpy.arange(0.0, 10.01, 1.0))

# log10(M20 / m_hm), log10(M50 / m_hm), log10(M80 / m_hm)
log10_m20 = numpy.array([
    [-0.472, -0.469, -0.455, -0.430, -0.396, -0.351, -0.298, -0.237, -0.169, -0.094, -0.014],
    [-0.356, -0.354, -0.346, -0.332, -0.313, -0.288, -0.258, -0.223, -0.185, -0.143, -0.099],
    [-0.327, -0.326, -0.320, -0.310, -0.296, -0.278, -0.256, -0.231, -0.203, -0.173, -0.141],
    [-0.322, -0.321, -0.316, -0.308, -0.296, -0.281, -0.263, -0.243, -0.220, -0.196, -0.169],
    [-0.324, -0.323, -0.319, -0.312, -0.302, -0.289, -0.273, -0.255, -0.235, -0.214, -0.190],
    [-0.329, -0.328, -0.325, -0.318, -0.309, -0.297, -0.283, -0.266, -0.248, -0.228, -0.207],
    [-0.335, -0.334, -0.331, -0.325, -0.316, -0.305, -0.292, -0.276, -0.259, -0.241, -0.221]])
log10_m50 = numpy.array([
    [ 0.310,  0.315,  0.333,  0.363,  0.406,  0.460,  0.525,  0.600,  0.683,  0.775,  0.873],
    [ 0.257,  0.260,  0.268,  0.282,  0.302,  0.327,  0.357,  0.391,  0.429,  0.470,  0.514],
    [ 0.209,  0.210,  0.215,  0.224,  0.237,  0.253,  0.272,  0.294,  0.318,  0.344,  0.372],
    [ 0.172,  0.173,  0.177,  0.183,  0.193,  0.205,  0.219,  0.235,  0.253,  0.273,  0.294],
    [ 0.143,  0.144,  0.147,  0.153,  0.160,  0.170,  0.182,  0.195,  0.210,  0.226,  0.244],
    [ 0.121,  0.122,  0.124,  0.129,  0.136,  0.144,  0.155,  0.166,  0.179,  0.193,  0.208],
    [ 0.103,  0.104,  0.106,  0.110,  0.116,  0.124,  0.133,  0.144,  0.156,  0.168,  0.182]])
log10_m80 = numpy.array([
    [ 1.222,  1.233,  1.262,  1.310,  1.376,  1.459,  1.559,  1.674,  1.803,  1.943,  2.093],
    [ 0.919,  0.922,  0.932,  0.949,  0.972,  1.002,  1.036,  1.076,  1.119,  1.167,  1.217],
    [ 0.755,  0.756,  0.762,  0.770,  0.782,  0.798,  0.816,  0.837,  0.860,  0.885,  0.912],
    [ 0.654,  0.655,  0.658,  0.664,  0.672,  0.682,  0.694,  0.707,  0.723,  0.739,  0.756],
    [ 0.587,  0.587,  0.590,  0.594,  0.600,  0.607,  0.616,  0.626,  0.637,  0.650,  0.663],
    [ 0.539,  0.539,  0.541,  0.545,  0.549,  0.555,  0.562,  0.570,  0.580,  0.589,  0.600],
    [ 0.503,  0.503,  0.505,  0.508,  0.512,  0.517,  0.523,  0.530,  0.538,  0.546,  0.555]])

_interp_20 = RegularGridInterpolator(points, log10_m20)
_interp_50 = RegularGridInterpolator(points, log10_m50)
_interp_80 = RegularGridInterpolator(points, log10_m80)


def _make_params_in_bounds(dlogT_dlogk, z):
    """
    Force the arguments inside the range of the calibration; no extrapolation.

    :param dlogT_dlogk: logarithmic derivative of the transfer function at k_1/2; negative
    :param z: redshift
    :return: (|dlogT/dlogk|, z) clipped to the grid
    """
    dlogT_dlogk_eval = min(max(1.0, -1.0 * dlogT_dlogk), 4.0)
    z_eval = min(max(z, 0.0), 10.0)
    return dlogT_dlogk_eval, z_eval


def benson_suppression_params(dlogT_dlogk, z):
    """
    Maps the logarithmic derivative of the transfer function at the half-mode scale and the
    redshift onto the a, b, c parameters that describe the suppression of the halo mass
    function, calibrated to Benson et al. (2026).

    :param dlogT_dlogk: the logarithmic derivative of the transfer function at k_1/2; negative
    :param z: redshift
    :return: the a, b, c parameters of the turnover (1 + a (m_hm/m)^b)^c
    """
    if dlogT_dlogk > 0:
        raise Exception('positive logarithmic derivatives are unphysical')
    x = _make_params_in_bounds(dlogT_dlogk, z)
    a_stucker, b, c = mscales_to_abc(10 ** float(_interp_20(x)),
                                     10 ** float(_interp_50(x)),
                                     10 ** float(_interp_80(x)))
    return a_stucker ** b, b, c

class ShethTormenTurnoverBenson(WDMPowerLaw):
    """
    Sheth-Tormen halo mass function with the Benson et al. (2026) turnover at a scale
    10^log_mc. Unlike ShethTormenTurnover, the turnover parameters are not fixed at model
    setup: they are evaluated per redshift slice from dlogT_dlogk inside from_redshift,
    because the calibration is redshift dependent.

    kwargs_model for this class include:
    1) log_mlow: minimum halo mass to render (log base 10)
    2) log_mhigh: maximum halo mass to render (log base 10)
    3) m_pivot: the pivot mass; the logarithmic slope is defined around the pivot
    4) LOS_normalization: rescales the amplitude of the mass function at the pivot scale
    5) delta_power_law_index: adjusts the logarithmic slope around the pivot scale
    6) log_mc: log 10 of the break scale
    7) dlogT_dlogk: the logarithmic derivative of the transfer function at k_1/2; negative
    8) z_eval_suppression: redshift at which to evaluate the suppression; None evaluates it
    at the redshift of each lens plane

    cutoff has the functional form (1 + a_wdm * (m_c/m) ^ b_wdm) ^ c_wdm, with a_wdm, b_wdm
    and c_wdm calibrated to Benson et al. (2026) at the redshift of each slice
    """
    name = 'SHETH_TORMEN_BENSON'

    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, kwargs_model):
        """
        :param z: redshift of the slice
        :param delta_z: width of the slice
        :param geometry_class: an instance of Geometry
        :param kwargs_model: keyword arguments for the mass function
        :return: an instance of ShethTormenTurnoverBenson
        """
        _ = geometry_class.cosmo.colossus
        m_pivot = kwargs_model['m_pivot']
        h = geometry_class.cosmo.h
        m = numpy.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 10)
        dndM_comoving = evaluate_mass_function(m, h, z, 'sheth99')
        coeffs = numpy.polyfit(numpy.log10(m / m_pivot), numpy.log10(dndM_comoving), 1)
        plaw_index = coeffs[0] + kwargs_model['delta_power_law_index']
        norm_dv = 10 ** coeffs[1] / (m_pivot ** plaw_index)
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = kwargs_model['LOS_normalization'] * norm_dv * volume_element_comoving
        # z_eval_suppression = None evaluates the suppression at the redshift of this lens
        # plane; a number holds it fixed at that redshift for every plane
        z_suppression = kwargs_model.get('z_eval_suppression', None)
        if z_suppression is None:
            z_suppression = z
        a_wdm, b_wdm, c_wdm = benson_suppression_params(kwargs_model['dlogT_dlogk'],
                                                        z_suppression)
        return ShethTormenTurnoverBenson(kwargs_model['log_mlow'], kwargs_model['log_mhigh'],
                                         plaw_index, kwargs_model['draw_poisson'],
                                         normalization, kwargs_model['log_mc'],
                                         a_wdm, b_wdm, c_wdm)

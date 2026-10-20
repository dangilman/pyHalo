"""
Mass function suppression calibrated to the halo mass function of Benson, Nadler, Du &
Gluscevic (2026), arXiv:2606.12137.

The suppression is expressed through the mass scales M20, M50 and M80 at which the ratio
n_wdm/n_cdm reaches 0.2, 0.5 and 0.8, in units of the half-mode mass. These are tabulated on a
grid of (|dlogT/dlogk|, z) and converted to the (a, b, c) of the standard turnover
(1 + a (m_hm/m)^b)^c using the same inversion as the Stucker et al. (2022) calibration.
Mass scales are tabulated rather than (a, b, c) because the latter are strongly degenerate.

This model is redshift-dependent. The suppression at m = m_hm
strengthens by about 25 per cent between z = 0 and z = 10; below z ~ 2 the dependence is
negligible.

Note tht Benson et al. exclude subhalos and backsplash halos by construction.
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
    [-0.470, -0.468, -0.455, -0.433, -0.401, -0.360, -0.310, -0.252, -0.186, -0.113, -0.034],
    [-0.355, -0.353, -0.347, -0.334, -0.316, -0.292, -0.264, -0.232, -0.195, -0.155, -0.112],
    [-0.326, -0.325, -0.320, -0.311, -0.298, -0.281, -0.261, -0.237, -0.211, -0.182, -0.151],
    [-0.321, -0.320, -0.316, -0.309, -0.298, -0.284, -0.267, -0.248, -0.227, -0.203, -0.177],
    [-0.324, -0.323, -0.319, -0.313, -0.303, -0.291, -0.277, -0.260, -0.241, -0.220, -0.198],
    [-0.329, -0.328, -0.325, -0.319, -0.310, -0.299, -0.286, -0.270, -0.253, -0.234, -0.214],
    [-0.334, -0.334, -0.331, -0.325, -0.317, -0.307, -0.294, -0.280, -0.264, -0.246, -0.227]])
log10_m50 = numpy.array([
    [ 0.312,  0.316,  0.332,  0.360,  0.399,  0.449,  0.509,  0.579,  0.658,  0.745,  0.839],
    [ 0.259,  0.261,  0.268,  0.281,  0.299,  0.322,  0.350,  0.382,  0.417,  0.456,  0.498],
    [ 0.210,  0.211,  0.216,  0.224,  0.235,  0.250,  0.268,  0.288,  0.311,  0.336,  0.362],
    [ 0.173,  0.174,  0.177,  0.183,  0.192,  0.203,  0.216,  0.231,  0.248,  0.267,  0.287],
    [ 0.144,  0.145,  0.148,  0.153,  0.160,  0.169,  0.180,  0.192,  0.206,  0.222,  0.238],
    [ 0.122,  0.123,  0.125,  0.129,  0.135,  0.143,  0.153,  0.164,  0.176,  0.189,  0.204],
    [ 0.104,  0.105,  0.107,  0.111,  0.116,  0.123,  0.132,  0.142,  0.153,  0.165,  0.178]])
log10_m80 = numpy.array([
    [ 1.225,  1.233,  1.259,  1.302,  1.362,  1.438,  1.529,  1.635,  1.753,  1.883,  2.023],
    [ 0.921,  0.924,  0.933,  0.948,  0.969,  0.996,  1.028,  1.064,  1.105,  1.149,  1.197],
    [ 0.756,  0.758,  0.762,  0.770,  0.781,  0.796,  0.812,  0.832,  0.853,  0.877,  0.902],
    [ 0.655,  0.656,  0.659,  0.665,  0.672,  0.681,  0.692,  0.705,  0.719,  0.735,  0.751],
    [ 0.588,  0.589,  0.591,  0.595,  0.600,  0.607,  0.615,  0.625,  0.635,  0.647,  0.659],
    [ 0.540,  0.541,  0.542,  0.545,  0.550,  0.555,  0.562,  0.570,  0.578,  0.588,  0.598],
    [ 0.504,  0.505,  0.506,  0.509,  0.512,  0.517,  0.523,  0.529,  0.537,  0.545,  0.553]])

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

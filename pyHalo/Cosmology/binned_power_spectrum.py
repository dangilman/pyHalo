"""
A linear matter power spectrum that deviates from LCDM in logarithmically spaced bins of wavenumber.

Each bin i carries an amplitude a_i, with log10(P / P_LCDM) = a_i - 1 at the bin center: a_i = 1 is LCDM and a_i = 1.3
is 10^0.3 = 1.995 times LCDM. The ratio P / P_LCDM is interpolated in log10(k) between the bin centers with a monotone
cubic spline (PCHIP slopes at the bin centers), and pinned to one with zero slope at k_min and k_max, so the power
spectrum is exactly LCDM outside [k_min, k_max], has no kinks, and never overshoots the amplitudes of neighboring bins.

Wavenumbers are in 1/Mpc and power spectra in Mpc^3 (no little h), as in pyHalo.Cosmology.power_spectrum.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
from pyHalo.Cosmology.power_spectrum import LinearPowerSpectrum

__all__ = ['binned_power_spectrum_nodes', 'binned_power_spectrum_ratio', 'lcdm_power_spectrum',
           'binned_linear_power_spectrum']


def binned_power_spectrum_nodes(n_bins, k_min=5.0, k_max=200.0):
    """
    The edges and centers of n_bins bins spaced evenly in log10(k) between k_min and k_max

    :param n_bins: the number of bins
    :param k_min: the smallest wavenumber of the first bin [1/Mpc]
    :param k_max: the largest wavenumber of the last bin [1/Mpc]
    :return: the bin edges (length n_bins + 1) and bin centers (length n_bins) [1/Mpc]
    """
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError('the number of bins must be at least one')
    if not 0 < k_min < k_max:
        raise ValueError('the bins must satisfy 0 < k_min < k_max')
    log10k_edges = np.linspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    log10k_centers = 0.5 * (log10k_edges[1:] + log10k_edges[:-1])
    return 10 ** log10k_edges, 10 ** log10k_centers


def binned_power_spectrum_ratio(k, amplitudes, k_min=5.0, k_max=200.0):
    """
    P(k) / P_LCDM(k) for the binned model (see the module docstring)

    :param k: wavenumber [1/Mpc]; float or array
    :param amplitudes: the amplitude of each bin, with log10(P / P_LCDM) = amplitude - 1 at the bin center; the number
    of bins is len(amplitudes)
    :param k_min: the smallest wavenumber of the first bin [1/Mpc]
    :param k_max: the largest wavenumber of the last bin [1/Mpc]
    :return: P(k) / P_LCDM(k), with the shape of k
    """
    amplitudes = np.atleast_1d(np.asarray(amplitudes, dtype=float))
    if amplitudes.ndim != 1 or not np.all(np.isfinite(amplitudes)):
        raise ValueError('the bin amplitudes must be a one-dimensional sequence of finite numbers')
    k_edges, k_centers = binned_power_spectrum_nodes(len(amplitudes), k_min, k_max)
    x_nodes = np.log10(np.concatenate(([k_edges[0]], k_centers, [k_edges[-1]])))
    y_nodes = np.concatenate(([0.0], amplitudes - 1.0, [0.0]))
    # PCHIP slopes at the bin centers, zero slope at k_min and k_max so the ratio joins LCDM smoothly
    slopes = PchipInterpolator(x_nodes, y_nodes).derivative()(x_nodes)
    slopes[[0, -1]] = 0.0
    spline = CubicHermiteSpline(x_nodes, y_nodes, slopes)
    x = np.log10(k)
    log10_ratio = np.where((x <= x_nodes[0]) | (x >= x_nodes[-1]), 0.0,
                           spline(np.clip(x, x_nodes[0], x_nodes[-1])))
    return 10 ** log10_ratio


def lcdm_power_spectrum(cosmology, k=None):
    """
    The LCDM linear matter power spectrum at z = 0 that colossus computes for the cosmology (the model pyHalo uses for
    the Sheth-Tormen mass function and the Ludlow et al. 2016 concentration-mass relation)

    :param cosmology: an instance of Cosmology (pyHalo.Cosmology.cosmology)
    :param k: wavenumbers [1/Mpc]; defaults to 2000 points evenly spaced in log10(k) between 10^-5 and 10^5
    :return: k [1/Mpc] and P(k) [Mpc^3]
    """
    if k is None:
        k = np.logspace(-5, 5, 2000)
    h = cosmology.h
    return k, cosmology.colossus.matterPowerSpectrum(k / h, 0.0) / h ** 3


def binned_linear_power_spectrum(cosmology, amplitudes, k_min=5.0, k_max=200.0, k=None,
                                 **kwargs_linear_power_spectrum):
    """
    The binned power spectrum as an instance of LinearPowerSpectrum, for the mass function and concentration-mass
    relation classes that take a power spectrum (BENSON_PK, SHETH_TORMEN_PK, LUDLOW2016_PK)

    :param cosmology: an instance of Cosmology (pyHalo.Cosmology.cosmology)
    :param amplitudes: the amplitude of each bin (see binned_power_spectrum_ratio)
    :param k_min: the smallest wavenumber of the first bin [1/Mpc]
    :param k_max: the largest wavenumber of the last bin [1/Mpc]
    :param k: wavenumbers at which to tabulate the power spectrum [1/Mpc]; see lcdm_power_spectrum for the default
    :param kwargs_linear_power_spectrum: keyword arguments passed to LinearPowerSpectrum
    :return: an instance of LinearPowerSpectrum
    """
    k, pk = lcdm_power_spectrum(cosmology, k)
    pk = pk * binned_power_spectrum_ratio(k, amplitudes, k_min, k_max)
    return LinearPowerSpectrum(k, pk, cosmology, **kwargs_linear_power_spectrum)

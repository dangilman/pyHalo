import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from lenstronomy.LensModel.Profiles.tnfw import TNFW


def tnfw_density_profile(r, rhos, rs, rt):
    """

    :param r:
    :param rhos:
    :param rs:
    :param r_t:
    :return:
    """
    x = r/rs
    tau = rt/rs
    nfw_prof = rhos / (x * (1+x)**2)
    truncation = tau ** 2 / (x**2 + tau**2)
    return nfw_prof * truncation

def tnfw_mass_fraction(tau, c):
    """
    This returns the ratio of the final mass to the initial mass of a truncated NFW profile given the truncation radius
    in units of rs (tau) and the concentration (c)
    :param tau: halo truncation radius in units of the scale radius
    :param c: the halo concentration
    :return: the ratio m_bound / m_virial
    """
    prof = TNFW()
    rs = 1.0
    rho0 = 1 / (4 * np.pi * rs ** 3)
    m = prof.mass_3d(c, rs, rho0, tau)
    nfw_func = np.log(1 + c) - c/(1+c)
    return m/nfw_func

def tau_mf_interpolation():

    """
    This function interpolates solutions for the truncation radius of a truncated NFW profile given the concentration
    and final bound mass
    :return: an instance of RegularGridInterpolator that returns (r_t / r_s) given a final mass and concentration
    """
    N = 100
    tau = np.logspace(-4.1, 2.3, N)
    log10_c = np.linspace(0, 2.7, N)
    # mass_fraction_1d = np.logspace(-1.45, -0.02, N)
    mass_fraction_1d = np.logspace(-6.0, np.log10(0.9999), N)
    log10tau_2d = np.zeros((N, N))

    # This computes the value of tau that correponds to each pair of (concentration, mass_loss)
    for i, log10con_i in enumerate(log10_c):
        log10final_mass = np.log10(tnfw_mass_fraction(tau, 10**log10con_i))
        mfinterp = interp1d(log10final_mass, np.log10(tau), bounds_error=False, fill_value='extrapolate')
        for j, mass_j in enumerate(mass_fraction_1d):
            log10tau_2d[i, j] = mfinterp(np.log10(mass_j))

    interp_points = (log10_c, np.log10(mass_fraction_1d))
    interpolator = RegularGridInterpolator(interp_points, log10tau_2d, bounds_error=False, fill_value=None)
    return interpolator


def cubic_real_roots(a, b, c, d):
    """Real roots of a x^3 + b x^2 + c x + d (analytic; avoids np.roots' eigvals).
    Used to solve for the radius where the logarithmic profile slope equals some value (see TNFW class)
    """
    if abs(a) < 1e-14:  # degenerate -> quadratic / linear
        if abs(b) < 1e-14:
            return [] if abs(c) < 1e-14 else [-d / c]
        disc = c * c - 4 * b * d
        if disc < 0:
            return []
        s = disc ** 0.5
        return [(-c + s) / (2 * b), (-c - s) / (2 * b)]
    b /= a;
    c /= a;
    d /= a
    p = c - b * b / 3.0
    q = 2 * b ** 3 / 27.0 - b * c / 3.0 + d
    off = -b / 3.0
    disc = (q / 2) ** 2 + (p / 3) ** 3
    if disc > 0:  # one real root
        s = disc ** 0.5
        return [np.cbrt(-q / 2 + s) + np.cbrt(-q / 2 - s) + off]
    r = (-(p / 3)) ** 0.5 if p < 0 else 0.0  # three real roots (trig form)
    if r == 0:
        return [off] * 3
    phi = np.arccos(max(-1.0, min(1.0, (-q / 2) / r ** 3)))
    return [2 * r * np.cos((phi + 2 * np.pi * k) / 3.0) + off for k in range(3)]

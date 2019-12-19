import numpy as np
from scipy.integrate import quad

##################################################################################
"""
Functions for computing properties of spherical truncated NFW profiles
"""
##################################################################################

def tnfw_bound_mass_from_rhors(rho_s, rs, X, tau):
    
    denom = ((1 + X)*(1 + tau**2)**2)
    numerator = (2 * rho_s * np.pi * rs ** 3 *tau ** 2 * (-2 * X * (1 + tau ** 2) + 4 * (1 + X) * tau *
       np.arctan(X / tau) + 2*(1 + X)*(-1 + tau**2)*np.log((1 + X)*tau) - (1 + X)*(-1 + tau**2)*np.log(X**2 + tau**2)))

    return numerator/denom

def rho_nfw(r, r_s, rho_s):

    softening = 0.001*r_s
    x = (r+softening) / r_s

    return rho_s * (x * (1 + x) ** 2) ** -1

def rho_tnfw(r, r_s, rho_s, r_t):

    tau = r_t/r_s
    softening = 0.001 * r_s
    x = (r + softening) / r_s

    trunc_term = tau ** 2 / (x ** 2 + tau ** 2)
    return trunc_term * rho_s * (x * (1 + x) ** 2) ** -1

def nfw_profile_mass(r, r_s, rho_s):

    x = r/r_s

    return 4*np.pi*r_s**3 * rho_s * (np.log(1+x) - x/(1+x))

def nfw_profile_mean_density(r, r_s, rho_s):

    volume = 4*np.pi*r**3/3
    return nfw_profile_mass(r, r_s, rho_s)/volume

##################################################################################
"""
Functions for computing properties of a composite isothermal+NFW profile
"""
##################################################################################

def density_norm(r_s_host, r_ein_kpc, sigma_crit):

    """

    :param r_ein_kpc: Einstein radius in physical kpc
    :param r_s_host: host scale radius in physical kpc
    :param sigma_crit: critical density for lensing [M_sun / kpc^2]
    :return: normalization of a composite profile
    rho \propto 1/(r^2 * (1+r))

    Something that goes like r^-2 inside r_s and r^-3 outside
    """

    x = r_ein_kpc / r_s_host

    if x>=1:
        raise Exception('Only implemented for Rein < rs_host. '
                        '(Physically, this should always be the case... )')

    rho_crit = sigma_crit/r_s_host
    numerator = 0.5 * rho_crit * x ** 2 # units of density

    sqrt = np.sqrt(1-x**2)
    denom = np.pi * x + np.log(0.25 * x ** 2) + 2 * sqrt * np.arctanh(sqrt) # dimensionless

    return numerator/denom

def rho_composite(r, r_s_host, rho0):

    x = r/r_s_host
    return rho0 / (x**2 * (1+x))

def composite_profile_mass(r, r_s_host, rho0):

    return 4*np.pi*r_s_host**3 * rho0 * np.log(1 + r/r_s_host)

def composite_profile_mean_density(r, r_s_host, rho0):

    volume = 4*np.pi*r**3/3
    return composite_profile_mass(r, r_s_host, rho0)/volume
    
##################################################################################
"""
Functions for computing properties of a cored NFW profile
rho \propto 1./((rc^10 + r^10)^(1/10)*(1+r)^2)
"""
##################################################################################

def cored_rho_nfw(r, r_s, rho_s, r_c):

    softening = 0.001 * r_s
    x = (r + softening) / r_s
    beta = r_c / r_s
    core_factor = (x ** 10 + beta ** 10) ** (0.1)

    return rho_nfw(r, r_s, rho_s) * x / core_factor


##################################################################################
"""
Functions for computing numerical integrals of profiles
"""
##################################################################################

def mass_integral_numerical(r, density_function, args):

    return 4 * np.pi * r ** 2 * density_function(r, *args)

def mean_density_numerical(rmax, profile, args):
    integral_args = (profile, args)
    mass_int = quad(mass_integral_numerical, 0, rmax, integral_args)[0]

    volume = (4. / 3) * np.pi * rmax ** 3

    return mass_int / volume

def _kappa_integrand(z, r2d, func, func_args):
    r = np.sqrt(r2d ** 2 + z ** 2)
    return func(r, *func_args)


def kappa(r2d, func, func_args, zmax):
    profileargs = (r2d, func, func_args)
    return 2 * quad(_kappa_integrand, 0, zmax, args=profileargs)[0]


def _mean_kappa_integrand(r2d, func, func_args, zmax):
    return r2d * kappa(r2d, func, func_args, zmax)

def mean_kappa(R, func, func_args, zmax):
    return (2 / R ** 2) * quad(_mean_kappa_integrand, 0, R, args=(func, func_args, zmax))[0]

def deflection_angle(R, func, func_args, zmax):
    return R * mean_kappa(R, func, func_args, zmax)

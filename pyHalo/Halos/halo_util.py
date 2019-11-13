import numpy as np
from scipy.integrate import quad

def rho_nfw(r, r_s, rho_s):

    softening = 0.001*r_s
    x = (r+softening) / r_s

    return rho_s * (x * (1 + x) ** 2) ** -1

def cored_rho_nfw(r, r_s, rho_s, r_c):

    softening = 0.001 * r_s
    x = (r + softening) / r_s
    beta = r_c / r_s
    core_factor = (x ** 10 + beta ** 10) ** (0.1)

    return rho_nfw(r, r_s, rho_s) * x / core_factor

def rho_composite(r, r_s_host_kpc, r_ein_kpc, sigma_crit):
    geometric_factor = r_ein_kpc ** 2 * (2 * np.pi * r_s_host_kpc ** 2) ** -1 * (
            r_ein_kpc + r_s_host_kpc - np.sqrt(r_ein_kpc ** 2 + r_s_host_kpc ** 2))
    x = r / r_s_host_kpc
    rho0 = sigma_crit * geometric_factor
    return rho0 * (x ** 2 * (1 + x) ** 2) ** -1

def mass_integral(r, density_function, args):

    return 4 * np.pi * r ** 2 * density_function(r, *args)

def mean_density(rmax, profile, args):
    integral_args = (profile, args)
    mass_int = quad(mass_integral, 0, rmax, integral_args)[0]

    volume = (4. / 3) * np.pi * rmax ** 3

    return mass_int / volume

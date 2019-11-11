import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

def integrand_mtotal(r, rhofunc, args):
    try:
        return 4 * np.pi * r ** 2 * rhofunc(r, *args)
    except:
        return 4 * np.pi * r ** 2 * rhofunc(r, args)


def mtotal(Rmax, rho_function, function_args):
    return quad(integrand_mtotal, 0, Rmax, args=(rho_function, function_args))[0]


def norm_m200(m200, r200, rho_function, function_args):
    mtot = mtotal(r200, rho_function, function_args)
    norm = m200 * mtot ** -1
    return norm


def r3d(r2d, z):
    return np.sqrt(r2d ** 2 + z ** 2)

def integrand_mproj(z, r2d, rhofunc, args):
    try:
        return 2 * rhofunc(r3d(r2d, z), *args)
    except:
        return 2 * rhofunc(r3d(r2d, z), args)

def integrand_deflection(r, rhofunc, args):
    return r * projected_mass(r, rhofunc, args)

def projected_mass(R2D, rho_function, function_args):
    return quad(integrand_mproj, 0, 1000, args=(R2D, rho_function, function_args))[0]

def deflection_point(R, rho_function, function_args):
    return (2 * R ** -1) * quad(integrand_deflection, 0, R, args=(rho_function, function_args))[0]

def deflection(Rvalues, rho_function, function_args, m200=False, r200=False, print_progress = False):

    """

    :param Rvalues: r coordinates in 3d
    :param rho_function: a function that outputs the 3d density. Must be of the form

    def rho_function(r3d, arg1, arg2, ...):
        return density_at_r3d

    :param function_args: a tuple (arg1, arg2, ...)
    :param m200:
    :param r200:
    :return:
    """

    if m200 is False or r200 is False:
        norm = 1
    else:
        norm = norm_m200(m200, r200, rho_function, function_args)

    defangle = []
    for k, ri in enumerate(Rvalues):
        if print_progress and k%(0.5*len(Rvalues)) == 0:
            print(str(10*(k%len(Rvalues)))+ '% ...')

        defangle.append(deflection_point(ri, rho_function, function_args))
    return norm * np.array(defangle)

def deflection_from_array(Rvalues, rho_values, function_args, m200=False, r200=False, print_progress = False):

    """

    :param Rvalues: r coordinates in 3d
    :param rho_values: an array continaing pre-computed density in 3d
    :param function_args: a tuple (arg1, arg2, ...)
    :param m200:
    :param r200:
    :return:
    """

    rho_function = interp1d(Rvalues, rho_values)

    if m200 is False or r200 is False:
        norm = 1
    else:
        norm = norm_m200(m200, r200, rho_function, function_args)

    defangle = []
    for k, ri in enumerate(Rvalues):
        if print_progress and k%(0.5*len(Rvalues)) == 0:
            print(str(10*(k%len(Rvalues)))+ '% ...')

        defangle.append(deflection_point(ri, rho_function, function_args))
    return norm * np.array(defangle)

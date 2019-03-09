import numpy as np
from scipy.integrate import quad
from pyHalo.Scattering.cross_section_classical import classical_semi, classical_cored_1, classical_cored_2
import matplotlib.pyplot as plt

def kernel(v, v0, g = 1000):

    x = (v*v0**-1) ** 2
    return 4 * np.pi * v ** 3 * np.exp(-x) * (np.pi * v0 ** 2)**-1.5 * np.log(1+x*g)

def kernel_simple(v, v0):

    x = (v*v0**-1) ** 2
    return 4 * np.pi * v ** 3 * np.exp(-x) * (np.pi * v0 ** 2)**-1.5

def sigma_v_simple(sigma0, v_rms, alpha):
    """

    :param sigma0: cross section in cm^2/gram
    :param v_0: pivot velocity in km/sec
    :param alpha: slope for v>v_0
    :param nu: turnover rate
    :return: sigma*v/m in km^3 / solar mass / gigayear
    """

    def _integrand_simple(velocity):

        _cross = classical_semi(sigma0, alpha=alpha, v=velocity)

        return kernel_simple(velocity, v_rms) * _cross

    return quad(_integrand_simple, 0, 50*v_rms)[0]

def sigma_v_cored_1(sigma0, v_rms, alpha, v_crit=0.1):

    def _integrand(velocity):

        _cross = classical_cored_1(sigma0, alpha, velocity, v_crit = v_crit)
        return kernel_simple(velocity, v_rms) * _cross

    return quad(_integrand, 0, 50*v_rms)[0]

def sigma_v_cored_2(sigma0, v_rms, alpha, v_crit=0.1):

    def _integrand(velocity):

        _cross = classical_cored_2(sigma0, alpha, velocity, v_crit = v_crit)
        return kernel_simple(velocity, v_rms) * _cross

    return quad(_integrand, 0, 50*v_rms)[0]


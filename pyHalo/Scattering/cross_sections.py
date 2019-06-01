import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def maxwell_boltzmann_average(v_rms, cross_section_function):
    #most_probable_v = v_rms
    most_probable_v = v_rms * np.sqrt(2./3)
    # 2 * np.sqrt(2./3) / np.sqrt(np.pi)
    # cross section times v_avg = cross0 * above factor

    def _integrand(v):
        x = v * most_probable_v ** -1
        kernel = 4 * np.pi * v ** 3 * np.exp(-x**2)
        norm = (np.pi * most_probable_v**2) ** -1.5

        return norm * kernel * cross_section_function(v)

    integral = quad(_integrand, 0, 10*most_probable_v)[0]

    return integral

class ClassicalCrossSection(object):

    def __init__(self, norm, m_phi = 17e-3, m_DM = 100, coupling=1/137):

        self._beta_const = 2 * m_phi * coupling * m_DM ** -1 * norm
        self._mphi = m_phi
        self._mchi = m_DM
        self._prefactor = np.pi * self._mphi ** -2

    def _eval_single(self, beta):

        if beta < 0.1:
            return 4 * self._prefactor * beta ** 2 * np.log(1 + beta ** -1)
        elif beta < 10**3:
            return 8 * self._prefactor * beta ** 2 / (1 + 1.5*beta**1.65)
        else:
            return self._prefactor * (np.log(beta) + 1 - 0.5 * np.log(beta)**-1)**2

    def __call__(self,v):

        beta = self._beta_const * v ** -2
        if isinstance(beta, float) or isinstance(beta, int):
            return self._eval_single(beta)
        else:
            cross = np.array([self._eval_single(beta_i) for beta_i in beta])
            return cross

class VelocityDependentCross(object):

    def __init__(self, cross0, v_ref=30, v_pow=0):

        self._cross = cross0
        self.v_pow = v_pow
        self.v_ref = v_ref
        if v_pow != 0:
            self.has_v_dep = True
        else:
            self.has_v_dep = False

    def cross_v(self, v):

        x = self.v_ref * v ** -1

        return self._cross * x ** self.v_pow

    def __call__(self, v_dis):

        sigma_times_velocity = maxwell_boltzmann_average(v_dis, self.cross_v)

        return sigma_times_velocity

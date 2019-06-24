import numpy as np
from scipy.special import jv, yv, jvp, yvp
from scipy.integrate import odeint

class NoInteraction():

    def __init__(self, xmax):

        self._xmax = xmax

    def __call__(self, vec, x, l):

        first_derivative = vec[1]
        second_derivative = vec[0] * (-self.a * self.a + l * (l + 1) / x ** 2 - self.potential(x))

        return [first_derivative, second_derivative]

    @property
    def x_min_init(self):
        return 0.001

    @property
    def x_max_init(self):
        return self._xmax



class YukawaInteraction(object):

    name = 'Yukawa'

    def __init__(self, m_phi, m_chi, coupling, v):

        self.mchi = m_chi
        self.mphi = m_phi
        self.coupling = abs(coupling)
        self.mratio = m_chi / m_phi
        self.v = v

        if coupling > 0:
            self._sign = 1
        else:
            self._sign = -1

    @property
    def properties(self):

        return {'a': self.a, 'b': self.b}

    def potential(self, x):
        xoverb = x / self.b
        return self._sign * np.exp(-xoverb) / x

    def __call__(self, vec, x, l):

        first_derivative = vec[1]
        second_derivative = vec[0]*(-self.a*self.a + l*(l+1) / x**2 - self.potential(x))

        return [first_derivative, second_derivative]

    @property
    def b(self):
        return self.coupling * self.mratio

    @property
    def a(self):
        return 0.5 * self.v / self.coupling

    @property
    def x_max_init(self, scale=1):

        xi = 1
        ratio = abs(self.potential(xi)) * self.a ** -2

        while ratio > 0.00005 / scale:
            xi += 0.25
            ratio = abs(self.potential(xi)) * self.a ** -2

        return xi

    @property
    def x_min_init(self):
        return 0.0025

    def phase_shift(self, l, xm, chi_prime_xm, chi_xm):

        beta_l = xm * chi_prime_xm * chi_xm ** -1 - 1
        X = self.a*xm
        num = X * jvp(l, X) - beta_l * jv(l, X)
        denom = X * yvp(l, X) - beta_l * yv(l, X)
        return np.arctan(num / denom)



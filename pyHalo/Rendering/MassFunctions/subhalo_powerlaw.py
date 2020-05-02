import numpy as np
from scipy.integrate import quad

class SubhaloPowerLaw(object):

    """
    Mass function for subhalos of LOS halos (not the subhalo mass function of main deflector)

    Functional form from Giocoli et al. 2008

    dN/dm = (N_0 / (alpha * M)) * x^-(1+alpha) * exp(-k * x^3)
    where x = m / (alpha*M)

    N_0 = 0.21
    alpha = 0.8
    k = 6.283
    """

    def __init__(self, m_low, M_parent, N_0 = 0.21, alpha = 0.8, k = 6.283):

        # don't render all the way up to parent mass
        self._max_M_scale = 0.8
        self._M = M_parent
        self._mlow = m_low

        self._norm = N_0
        self._index = -(alpha + 1)
        self._k = k

        norm = self._norm * (alpha*M_parent) ** -1

        # compute n_sub by integrating dN/dm
        if np.log10(M_parent * m_low**-1) < 1.5:
            self.nsub_mean = 0
        else:
            def _integrand(m):

                x = m * (alpha * M_parent) ** -1
                return norm * x ** -(alpha+1) * np.exp(-k * x ** 3)

            self.nsub_mean = quad(_integrand, m_low, M_parent)[0]

    def draw(self):

        if self.nsub_mean > 0:
            nsub = np.random.poisson(self.nsub_mean)
        else:
            return []

        halos = None

        # don't render all the way up to parent mass
        max_M = self._M * self._max_M_scale

        while True:
            x = np.random.rand()
            X = (x * (max_M ** (1 + self._index) - self._mlow ** (1 + self._index)) + self._mlow ** (1 + self._index)) ** (
                (1 + self._index) ** -1)

            reject = np.exp(-self._k * (X * self._M ** -1)**3)

            u = np.random.rand()
            if u < reject:

                if halos is None:
                    halos = X
                else:
                    halos = np.append(halos, X)
                    if len(halos) >= nsub:
                        break
            else:
                continue
        return halos

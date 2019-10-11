from scipy.integrate import quad
import numpy as np

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

class Gaussian(object):

    """
    set up for black holes right now
    """

    def __init__(self, log_mass_mean, standard_dev, n_draw):

        # log_mass_mean is base 10
        self.log_mass_mean_base_e = np.log(10**log_mass_mean)

        # standard deviation is in dex
        self.std_base_e = np.log(10**standard_dev)
        self.N_draw = n_draw

    def draw(self):

        log_masses_base_e = np.random.normal(self.log_mass_mean_base_e, self.std_base_e, self.N_draw)

        log_masses = np.log10(np.exp(log_masses_base_e))

        return 10 ** log_masses

class PowerLaw(object):

    draw_poisson = True

    def __init__(self, power_law_index = None, log_mlow = None, log_mhigh = None, normalization = None):

        if power_law_index > 0:
            raise ValueError('you have specified a power law index which is greater than zero, this is unphysical.')

        self._index = power_law_index

        self._mL, self._mH = 10 ** log_mlow, 10 ** log_mhigh

        if normalization < 0:
            raise Exception('normalization cannot be < 0.')
        else:
            Nhalos_mean = self._moment(normalization, self._mL, self._mH, 0)
          
            self.norm = normalization

            self.Nhalos_mean = Nhalos_mean

    def draw(self):

        if self.draw_poisson:
            N = np.random.poisson(self.Nhalos_mean)
        else:
            N = int(np.round(self.Nhalos_mean))

        x = np.random.rand(N)
        #x = np.random.rand(int(np.round(self.Nhalos_mean)))
        X = (x * (self._mH ** (1 + self._index) - self._mL ** (1 + self._index)) + self._mL ** (1 + self._index)) ** (
                (1 + self._index) ** -1)

        return np.array(X)

    def _moment(self, norm, m1, m2, n):

        return norm * self._integral(n, m1, m2)

    def _integral(self, n, m1, m2):

        if self._index == 1 and n == 0:
            return np.log(m2 / m1)
        else:
            return (n + 1 + self._index) ** -1 * (m2 ** (n + 1 + self._index) - m1 ** (n + 1 + self._index))

class BrokenPowerLaw(object):

    def __init__(self, power_law_index = None, log_mlow = None, log_mhigh = None, normalization = None,
                 log_m_break = None, break_index = None, break_scale = None, draw_poisson = True, **kwargs):

        self._plaw = PowerLaw(power_law_index, log_mlow, log_mhigh, normalization)
        self._plaw.draw_poisson = draw_poisson

        self.log_m_break = log_m_break

        self.break_index = break_index

        self.break_scale = break_scale

        if break_index > 0:
            raise ValueError('Break index should be a negative number (otherwise mass function gets steeper (unphysical)')

        self._unbroken_masses = self._plaw.draw()

    def draw(self):

        if self.log_m_break == 0:
            return self._unbroken_masses

        if len(self._unbroken_masses) == 0:
            return np.array([])

        mbreak = 10**self.log_m_break
        ratio = self.break_scale * mbreak * self._unbroken_masses**-1
        u = np.random.rand(int(len(self._unbroken_masses)))

        func = (1 + ratio) ** self.break_index

        return self._unbroken_masses[np.where(u < func)]


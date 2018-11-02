from scipy.special import erf
import numpy as np

class LogNormal(object):

    def __init__(self, mean = None, standard_dev = None, log_mlow = None,
                 log_mhigh = None, normalization = None):

        self._mean = mean
        self._sigma = standard_dev

        if normalization < 0:
            raise Exception('normalization cannot be <0')
        else:
            Nhalos_mean = self._integrate(normalization, 10**log_mlow, 10**log_mhigh)
            self.Nhalos_mean = Nhalos_mean

    def _integrate(self, norm, mlow, mhigh):

        # norm is fpbh * rho_matter(z) * delta_V
        I1 = self._I1(self._mean, self._sigma, mlow, mhigh)
        I0 = self._I0(self._mean, self._sigma, mlow, mhigh)
        print(I1, I0)
        I = np.exp(self._mean + 0.5*self._sigma**2) * I1 * I0 ** -1

        return norm * I

    def _I0(self, mu, sig, ml, mh):
        x1 = (mu + sig ** 2 - np.log(mh)) / (2 ** 0.5 * sig)
        x2 = (mu + sig ** 2 - np.log(ml)) / (2 ** 0.5 * sig)

        return erf(x2) - erf(x1)

    def _I1(self, mu, sig, ml, mh):

        x1 = (mu - np.log10(mh)) / (2 ** 0.5 * sig)
        x2 = (mu - np.log10(ml)) / (2 ** 0.5 * sig)

        return erf(x2) - erf(x1)

    def draw(self):

        N = np.random.poisson(self.Nhalos_mean)

        samples = np.random.lognormal(self._mean, self._standard_dev, size=N)

        return np.array(samples)

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
                 log_m_break = None, break_index = None, draw_poisson = True, **kwargs):

        self._plaw = PowerLaw(power_law_index, log_mlow, log_mhigh, normalization)
        self._plaw.draw_poisson = draw_poisson

        self.log_m_break = log_m_break

        self.break_index = break_index

        if break_index > 0:
            raise ValueError('Break index should be a negative number (otherwise mass function gets steeper (unphysical)')

        self._unbroken_masses = self._plaw.draw()

    def draw(self):

        if self.log_m_break == 0:
            return self._unbroken_masses

        if len(self._unbroken_masses) == 0:
            return np.array([])

        mbreak = 10**self.log_m_break
        ratio = mbreak * self._unbroken_masses**-1
        u = np.random.rand(int(len(self._unbroken_masses)))
        func = (1 + ratio) ** self.break_index

        return self._unbroken_masses[np.where(u < func)]

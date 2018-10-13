import numpy as np

class PowerLaw(object):

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

        x = np.random.rand(np.random.poisson(self.Nhalos_mean))
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
                 log_m_break = None, break_index = None, **kwargs):

        self._plaw = PowerLaw(power_law_index, log_mlow, log_mhigh, normalization)

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

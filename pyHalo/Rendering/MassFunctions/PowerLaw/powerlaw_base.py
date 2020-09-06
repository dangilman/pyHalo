import numpy as np

class PowerLawBase(object):

    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson=True):

        if power_law_index > 0:
            raise ValueError('you have specified a power law index which is greater than zero, this is unphysical.')

        self._mL = 10**log_mlow
        self._mH = 10**log_mhigh
        self._index = power_law_index

        self.draw_poisson = draw_poisson

    def sample_from_power_law(self, n_draw):

        if self.draw_poisson:
            N = np.random.poisson(n_draw)
        else:
            N = int(round(np.round(n_draw)))

        x = np.random.rand(N)
        if self._index == -1:
            norm = np.log(self._mH/self._mL)
            X = self._mL * np.exp(norm * x)
        else:
            X = (x * (self._mH ** (1 + self._index) - self._mL ** (1 + self._index)) + self._mL ** (
                1 + self._index)) ** (
                (1 + self._index) ** -1)

        return np.array(X)

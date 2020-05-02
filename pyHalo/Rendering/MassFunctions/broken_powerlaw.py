import numpy as np
from pyHalo.Rendering.MassFunctions.powerlaw import PowerLaw

class BrokenPowerLaw(object):

    def __init__(self, power_law_index=None, log_mlow = None, log_mhigh = None, normalization = None,
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
        ratio = (mbreak * self._unbroken_masses**-1)**self.break_scale
        u = np.random.rand(int(len(self._unbroken_masses)))

        func = (1 + ratio) ** self.break_index

        return self._unbroken_masses[np.where(u < func)]

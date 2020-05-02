import numpy as np
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_analytic

class PowerLaw(object):

    def __init__(self, power_law_index=None, log_mlow=None, log_mhigh=None, normalization=None,
                 draw_poission=True):

        if power_law_index > 0:
            raise ValueError('you have specified a power law index which is greater than zero, this is unphysical.')

        self._index = power_law_index

        self._mL, self._mH = 10 ** log_mlow, 10 ** log_mhigh

        self.draw_poission = draw_poission

        if normalization < 0:
            raise Exception('normalization cannot be < 0.')
        else:

            Nhalos_mean = integrate_power_law_analytic(normalization, self._mL, self._mH, 0, power_law_index)

            self.norm = normalization

            self.Nhalos_mean = Nhalos_mean

    def draw(self):

        if self.draw_poission:
            N = np.random.poisson(self.Nhalos_mean)
        else:
            N = int(round(np.round(self.Nhalos_mean)))

        x = np.random.rand(N)
        #x = np.random.rand(int(np.round(self.Nhalos_mean)))
        X = (x * (self._mH ** (1 + self._index) - self._mL ** (1 + self._index)) + self._mL ** (1 + self._index)) ** (
                (1 + self._index) ** -1)

        return np.array(X)

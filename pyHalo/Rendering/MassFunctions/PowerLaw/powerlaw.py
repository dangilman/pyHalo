import numpy as np
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_analytic
from pyHalo.Rendering.MassFunctions.PowerLaw.powerlaw_base import PowerLawBase

class PowerLaw(PowerLawBase):

    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson, normalization, **kwargs):

        if normalization < 0:
            raise Exception('normalization cannot be < 0.')

        self.Nhalos_mean = integrate_power_law_analytic(normalization, self._mL, self._mH, 0, power_law_index)

        super(PowerLaw, self).__init__(log_mlow, log_mhigh, power_law_index, draw_poisson)

    def draw(self):

        return self.sample_from_power_law(self.Nhalos_mean)

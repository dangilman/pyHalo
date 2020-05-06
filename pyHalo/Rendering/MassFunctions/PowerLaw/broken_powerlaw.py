import numpy as np
from pyHalo.Rendering.MassFunctions.PowerLaw.powerlaw_base import PowerLawBase
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_analytic
from pyHalo.Rendering.MassFunctions.mass_function_utilities import WDM_suppression

class BrokenPowerLaw(PowerLawBase):

    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson, normalization,
                 log_m_break, break_index, break_scale, **kwargs):

        if normalization < 0:
            raise Exception('normalization cannot be < 0.')
        if break_index > 0:
            raise ValueError('Break index should be a negative number (otherwise mass function gets steeper (unphysical)')
        if break_scale < 0:
            raise ValueError('Break scale should be a positive number for suppression factor: '
                             '( 1 + (m/m_hm)^(break_scale )^break_index')

        self.log_m_break = log_m_break
        self.break_index = break_index
        self.break_scale = break_scale

        self.Nhalos_mean = integrate_power_law_analytic(normalization, 10**log_mlow, 10**log_mhigh, 0,
                                                        power_law_index)

        super(BrokenPowerLaw, self).__init__(log_mlow, log_mhigh, power_law_index, draw_poisson)

    def draw(self):

        m = self.sample_from_power_law(self.Nhalos_mean)

        if self.log_m_break == 0 or len(m) == 0:
            return m

        factor = WDM_suppression(m, 10**self.log_m_break, self.break_index, self.break_scale)

        u = np.random.rand(int(len(m)))

        return m[np.where(u < factor)]

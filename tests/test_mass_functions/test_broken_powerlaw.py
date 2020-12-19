import numpy.testing as npt
import pytest
from pyHalo.Rendering.MassFunctions.PowerLaw.broken_powerlaw import BrokenPowerLaw
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad
import numpy as np

class TestBrokenPowerLaw(object):

    def setup(self):

        self.log_mlow = 6.
        self.log_mhigh = 8.7
        self.plaw_index = -1.9
        self.norm = 10 ** 12
        self.func_cdm = BrokenPowerLaw(self.log_mlow, self.log_mhigh, self.plaw_index,
                                   draw_poisson=False, normalization=self.norm, log_m_break=None,
                                       break_index=None, break_scale=None)

        self.func_wdm = BrokenPowerLaw(self.log_mlow, self.log_mhigh, self.plaw_index,
                                       draw_poisson=False, normalization=self.norm,
                                       log_m_break=7.5, break_index=-1.3, break_scale=0.5)

    def test_draw_cdm(self):

        logmhm = 0
        n = 1
        mtheory = integrate_power_law_quad(self.norm, 10**self.log_mlow, 10**self.log_mhigh, logmhm, n,
                                           self.plaw_index)
        m = self.func_cdm.draw()
        npt.assert_almost_equal(np.sum(m)/mtheory, 1, 2)

    def test_draw_wdm(self):

        logmhm = 7.5
        n = 1
        mtheory = integrate_power_law_quad(self.norm, 10**self.log_mlow, 10**self.log_mhigh, logmhm, n,
                                           self.plaw_index, break_index=-1.3, break_scale=0.5)
        m = self.func_wdm.draw()
        npt.assert_almost_equal(np.sum(m)/mtheory, 1, 2)

if __name__ == '__main__':
    pytest.main()

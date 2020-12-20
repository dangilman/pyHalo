import numpy.testing as npt
import pytest
from pyHalo.Rendering.MassFunctions.PowerLaw.broken_powerlaw import BrokenPowerLaw
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, integrate_power_law_analytic
import numpy as np

class TestBrokenPowerLaw(object):

    def setup(self):

        self.log_mlow = 6.
        self.log_mhigh = 8.7
        self.plaw_index = -1.9
        self.norm = 10 ** 12
        self.func_cdm = BrokenPowerLaw(self.log_mlow, self.log_mhigh, self.plaw_index,
                                       draw_poisson=False, normalization=self.norm, log_mc=None,
                                       a_wdm=None, b_wdm=None, c_wdm=None)

        self.func_wdm = BrokenPowerLaw(self.log_mlow, self.log_mhigh, self.plaw_index,
                                       draw_poisson=False, normalization=self.norm,
                                       log_mc=7.5, a_wdm=2., b_wdm=0.5, c_wdm=-1.3)

    def test_draw_cdm(self):

        n = 1
        mtheory = integrate_power_law_analytic(self.norm, 10**self.log_mlow, 10**self.log_mhigh, n,
                                           self.plaw_index)
        m = self.func_cdm.draw()
        npt.assert_almost_equal(np.sum(m)/mtheory, 1, 2)

    def test_draw_wdm(self):

        n = 1
        mtheory = integrate_power_law_quad(self.norm, 10**self.log_mlow, 10**self.log_mhigh, 7.5, n,
                                           self.plaw_index, a_wdm=2., b_wdm=0.5, c_wdm=-1.3)
        m = self.func_wdm.draw()
        npt.assert_almost_equal(np.sum(m)/mtheory, 1, 2)

if __name__ == '__main__':
    pytest.main()

import numpy.testing as npt
import pytest
from pyHalo.Rendering.MassFunctions.power_law import GeneralPowerLaw
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, integrate_power_law_analytic
import numpy as np

class TestGeneralPowerLaw(object):

    def setup(self):

        self.log_mlow = 6.
        self.log_mhigh = 8.7
        self.plaw_index = -1.9
        self.norm = 10 ** 12
        self.func_cdm = GeneralPowerLaw(self.log_mlow, self.log_mhigh, self.plaw_index,
                                        draw_poisson=False, normalization=self.norm, log_mc=None,
                                        a_wdm=None, b_wdm=None, c_wdm=None)

        self.func_wdm = GeneralPowerLaw(self.log_mlow, self.log_mhigh, self.plaw_index,
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

    def test_number_of_halos(self):

        n_model = self.func_cdm._nhalos_mean_unbroken
        ntheory = self.norm * 4.407e-6
        npt.assert_almost_equal(n_model/ntheory, 1, 5)


if __name__ == '__main__':
    pytest.main()

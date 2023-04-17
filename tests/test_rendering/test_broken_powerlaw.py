import numpy.testing as npt
import pytest
from pyHalo.Rendering.MassFunctions.mass_function_base import GeneralPowerLaw
import numpy as np
from pyHalo.Rendering.MassFunctions.delta_function import ScaleFree, PolynomialSuppression, MixedDMSuppression

class TestGeneralPowerLaw(object):

    def setup(self):

        self.log_mlow = 6.
        self.log_mhigh = 8.7
        self.plaw_index = -1.9
        self.norm = 10 ** 12
        self.model_scalefree = ScaleFree()
        self.kwargs_model_scalefree = {}
        self.func_cdm = GeneralPowerLaw(self.log_mlow, self.log_mhigh, self.plaw_index,
                                        False, self.norm, self.model_scalefree, self.kwargs_model_scalefree)

        self.model_polynomial = PolynomialSuppression()
        self.kwargs_model_polynomial = {'a_wdm': 1.0, 'b_wdm': 1.2, 'c_wdm': -1.5, 'log_mc': 7.8}
        self.func_wdm = GeneralPowerLaw(self.log_mlow, self.log_mhigh, self.plaw_index,
                                        False, self.norm,
                                        self.model_polynomial, self.kwargs_model_polynomial)

        self.model_mixed = MixedDMSuppression()
        self.kwargs_model_mixed = {'a_wdm': 1.0, 'b_wdm': 1.2, 'c_wdm': -1.5, 'log_mc': 7.8, 'mixed_DM_frac': 0.6}
        self.func_mixedDM = GeneralPowerLaw(self.log_mlow, self.log_mhigh, self.plaw_index,
                                        False, self.norm,
                                        self.model_mixed, self.kwargs_model_mixed)

    def test_draw_cdm(self):

        n = 1
        mtheory = self.model_scalefree.integrate_power_law_analytic(self.norm, 10**self.log_mlow, 10**self.log_mhigh, n,
                                           self.plaw_index)
        m = self.func_cdm.draw()
        npt.assert_almost_equal(np.sum(m)/mtheory, 1, 2)

    def test_draw_wdm(self):

        n = 1
        mtheory = self.model_polynomial.integrate_power_law_quad(self.norm, 10**self.log_mlow, 10**self.log_mhigh, n,
                                           self.plaw_index, **self.kwargs_model_polynomial)
        m = self.func_wdm.draw()
        npt.assert_almost_equal(np.sum(m)/mtheory, 1, 2)

    def test_draw_mixedDM(self):

        n = 1
        mtheory = self.model_mixed.integrate_power_law_quad(self.norm, 10**self.log_mlow, 10**self.log_mhigh, n,
                                           self.plaw_index, **self.kwargs_model_mixed)
        m = self.func_mixedDM.draw()
        npt.assert_almost_equal(np.sum(m)/mtheory, 1, 2)

    def test_number_of_halos(self):

        n_model = self.func_cdm._nhalos_mean_unbroken
        ntheory = self.norm * 4.407e-6
        npt.assert_almost_equal(n_model/ntheory, 1, 5)


if __name__ == '__main__':
    pytest.main()

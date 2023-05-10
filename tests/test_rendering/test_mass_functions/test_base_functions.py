import pytest
import numpy as np
import numpy.testing as npt
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw, WDMPowerLaw, MixedWDMPowerLaw


class TestCDMPowerLaw(object):

    def setup_method(self):

        self.log_mlow = 5.7
        self.log_mhigh = 8.7
        self.power_law_index = -1.9
        self.draw_poisson = False
        self.normalization = 1e11
        self.mass_function = CDMPowerLaw(self.log_mlow, self.log_mhigh, self.power_law_index,
                                self.draw_poisson, self.normalization)

    def test_zeroth_moment(self):
        n_expected = self.mass_function.n_mean
        n_rendered = len(self.mass_function.draw())
        npt.assert_almost_equal(n_rendered / n_expected, 1, 3)

    def test_first_moment(self):
        expected = self.mass_function.first_moment
        rendered = np.sum(self.mass_function.draw())
        npt.assert_almost_equal(expected / rendered, 1, 2)

class TestWDMPowerLaw(object):

    def setup_method(self):

        self.log_mlow = 5.7
        self.log_mhigh = 8.7
        self.power_law_index = -1.9
        self.draw_poisson = False
        self.normalization = 2e11
        self.log_mc = 7.
        self.a_wdm = 0.5
        self.b_wdm = 1.2
        self.c_wdm = -2.1
        self.mass_function = WDMPowerLaw(self.log_mlow, self.log_mhigh, self.power_law_index,
                                self.draw_poisson, self.normalization, self.log_mc, self.a_wdm, self.b_wdm, self.c_wdm)
    def test_zeroth_moment(self):
        n_expected = self.mass_function.n_mean
        n_rendered = len(self.mass_function.draw())
        npt.assert_almost_equal(n_rendered / n_expected, 1, 2)

    def test_first_moment(self):
        expected = self.mass_function.first_moment
        rendered = np.sum(self.mass_function.draw())
        npt.assert_almost_equal(expected / rendered, 1, 2)

class TestMixedWDMPowerLaw(object):

    def setup_method(self):

        self.log_mlow = 5.7
        self.log_mhigh = 8.7
        self.power_law_index = -1.9
        self.draw_poisson = False
        self.normalization = 2e11
        self.log_mc = 7.
        self.a_wdm = 0.5
        self.b_wdm = 1.2
        self.c_wdm = -1.6
        self.f_mixed = 0.6
        self.mass_function = MixedWDMPowerLaw(self.log_mlow, self.log_mhigh, self.power_law_index,
                                self.draw_poisson, self.normalization, self.log_mc, self.a_wdm, self.b_wdm,
                                         self.c_wdm, self.f_mixed)

        self.mass_function_mixed_pure_wdm = MixedWDMPowerLaw(self.log_mlow, self.log_mhigh, self.power_law_index,
                                         self.draw_poisson, self.normalization, self.log_mc, self.a_wdm, self.b_wdm,
                                         self.c_wdm, 0.0)

        self.mass_function_mixed_pure_cdm = MixedWDMPowerLaw(self.log_mlow, self.log_mhigh, self.power_law_index,
                                                            self.draw_poisson, self.normalization, self.log_mc,
                                                            self.a_wdm, self.b_wdm,
                                                            self.c_wdm, 1.0)

        self.mass_function_pure_wdm = WDMPowerLaw(self.log_mlow, self.log_mhigh, self.power_law_index,
                                                        self.draw_poisson, self.normalization, self.log_mc, self.a_wdm,
                                                        self.b_wdm,
                                                        self.c_wdm)

        self.mass_function_pure_cdm = CDMPowerLaw(self.log_mlow, self.log_mhigh, self.power_law_index,
                                                  self.draw_poisson, self.normalization)

    def test_zeroth_moment(self):
        n_expected = self.mass_function.n_mean
        n_rendered = len(self.mass_function.draw())
        npt.assert_almost_equal(n_rendered / n_expected, 1, 2)

    def test_first_moment(self):
        expected = self.mass_function.first_moment
        rendered = np.sum(self.mass_function.draw())
        npt.assert_almost_equal(expected / rendered, 1, 2)

    def test_pure_wdm(self):

        first_moment_wdm = self.mass_function_mixed_pure_wdm.first_moment
        first_moment_mixed_pure_wdm = self.mass_function_pure_wdm.first_moment
        npt.assert_almost_equal(first_moment_mixed_pure_wdm/first_moment_wdm, 1.0, 8)

    def test_pure_cdm(self):

        first_moment_cdm = self.mass_function_mixed_pure_cdm.first_moment
        first_moment_mixed_pure_cdm = self.mass_function_pure_cdm.first_moment
        npt.assert_almost_equal(first_moment_cdm/first_moment_mixed_pure_cdm, 1.0, 8)

# class TestTabulated(object):
#
#     def setup_method(self):
#
#         self.log_mlow = 6.0
#         self.log_mhigh = 8.0
#         self.power_law_index = -1.9
#         self.draw_poisson = False
#         self.normalization = 1e10
#         self.mass_function_powerlaw = CDMPowerLaw(self.log_mlow, self.log_mhigh, self.power_law_index,
#                                          self.draw_poisson, self.normalization)
#
#         m = np.logspace(6, 8, 1000)
#         dndm = self.normalization * m ** self.power_law_index
#         self.mass_function_tabulated = Tabulated(self.log_mlow, self.log_mhigh, self.draw_poisson,
#                                                  m, dndm)
#
#     def test_zeroth_moment(self):
#
#         npt.assert_almost_equal(self.mass_function_powerlaw.n_mean/self.mass_function_tabulated.n_mean, 1.0, 3)
#
#     def test_first_moment(self):
#
#         npt.assert_almost_equal(self.mass_function_powerlaw.first_moment/self.mass_function_tabulated.first_moment, 1.0, 3)
#         rendered = np.sum(self.mass_function_tabulated.draw())
#         expected = np.sum(self.mass_function_powerlaw.draw())
#         npt.assert_almost_equal(rendered/expected,1.0)


if __name__ == '__main__':
   pytest.main()

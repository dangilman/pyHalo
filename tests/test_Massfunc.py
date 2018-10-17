import numpy.testing as npt
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw, PowerLaw
import pytest
import numpy as np
from scipy.integrate import quad

class TestMassfunc(object):
    """
    test angular to mass unit conversions
    """
    def setup(self):

        self.plaw_idx = -1.9
        self.norm = 10**12
        self.log_m_break = 7
        self.break_index = -1

        self.power_law_broken = BrokenPowerLaw(power_law_index=self.plaw_idx, log_mlow=6, log_mhigh=11,
                                               normalization=self.norm, log_m_break=self.log_m_break, break_index=self.break_index)
        self.power_law = PowerLaw(power_law_index=self.plaw_idx, log_mlow=6, log_mhigh=11,
                                               normalization=self.norm)
        self.power_law_unbroken = BrokenPowerLaw(power_law_index=self.plaw_idx, log_mlow=6, log_mhigh=11,
                                               normalization=self.norm, log_m_break=0, break_index=-1)
        self.bins = 10 ** np.arange(6, 11, 1)

    def test_power_law(self):

        halos_cdm = self.power_law.draw()
        h, b = np.histogram(halos_cdm, bins=self.bins)
        x = b[0:-1]
        h_cdm = h
        idx_cdm, norm_cdm = np.polyfit(np.log10(x),np.log10(h), 1)

        halos_cdm_broken = self.power_law_unbroken.draw()
        h, b = np.histogram(halos_cdm_broken, bins=self.bins)
        idx_cdm_2, norm_cdm_2 = np.polyfit(np.log10(x), np.log10(h), 1)

        npt.assert_almost_equal(norm_cdm, np.log10(self.norm), decimal=1)
        npt.assert_almost_equal(norm_cdm, norm_cdm_2, decimal=1)

        npt.assert_almost_equal(idx_cdm, self.plaw_idx+1, decimal=1)
        npt.assert_almost_equal(idx_cdm, idx_cdm_2, decimal=1)

        halos_wdm = self.power_law_broken.draw()
        h, b = np.histogram(halos_wdm, bins=self.bins)
        npt.assert_almost_equal(h[-1] / h_cdm[-1], 1, decimal=1)

    def test_power_law_integrals(self):

        def _integral(m, m_break, break_index, n):
            return self.norm * m ** (n + self.plaw_idx) * (1 + m_break / m) ** break_index

        halos_cdm = self.power_law.draw()
        halos_wdm = self.power_law_broken.draw()

        analytic = quad(_integral, 10**6, 10**11, args=(0, self.break_index, 1))[0]
        analytic_wdm = quad(_integral, 10**6, 10**11, args=(10**self.log_m_break, self.break_index, 1))[0]

        npt.assert_almost_equal(np.sum(halos_cdm) / analytic, 1, decimal = 2)
        npt.assert_almost_equal(np.sum(halos_wdm) / analytic_wdm, 1, decimal=2)

if __name__ == '__main__':
    pytest.main()

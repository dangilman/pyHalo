import pytest
import numpy.testing as npt
from pyHalo.Halos.util import *
from scipy.integrate import quad

class TestUtil(object):

    def test_tnfw_mass_fraction(self):

        def _integrand(ri, rhos, rs, rt):
            return 4 * np.pi * ri ** 2 * tnfw_density_profile(ri, rhos, rs, rt)
        tau = 8.0
        c = 15.0
        mfrac = tnfw_mass_fraction(tau, c)
        rs = 0.3
        rt = tau * rs
        r200 = rs * c
        rhos = 10 ** 7.5
        m200 = 4 * np.pi * rs ** 3 * rhos * (np.log(1+c) - c/(1+c))
        m = quad(_integrand, 0, r200, args=(rhos, rs, rt))[0]
        npt.assert_almost_equal(m/m200, mfrac)

    def test_interp(self):

        interpolator = tau_mf_interpolation()
        log10_c = 1.9
        m_final_over_m = 0.5
        log10_mfrac = np.log10(m_final_over_m)
        x = (log10_c, log10_mfrac)
        log10tau = interpolator(x)
        tau = 10 ** log10tau
        mfrac = tnfw_mass_fraction(tau, 10 ** log10_c)
        npt.assert_almost_equal(mfrac, m_final_over_m, 3)

if __name__ == '__main__':
     pytest.main()

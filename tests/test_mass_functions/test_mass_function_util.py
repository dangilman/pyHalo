import numpy.testing as npt
from scipy.special import hyp2f1
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, integrate_power_law_analytic
import pytest


class TestMassFunctionUtil(object):

    def test_integrate_mass_function(self):

        def _analytic_integral_bound(x, a, b, c, n):

            term1 = x ** (a + n + 1) * ((c + x) / c) ** (-b) * ((c + x) / x) ** b
            term2 = hyp2f1(-b, a - b + n + 1, a - b + n + 2, -x / c) / (a - b + n + 1)
            return term1 * term2

        norm = 1.
        m_low, m_high = 10 ** 6, 10 ** 10
        log_mc = 0.

        a_wdm, b_wdm, c_wdm = 1., 1., -1.3
        plaw_index = -1.8

        for n in [0, 1]:
            integral = integrate_power_law_quad(norm, m_low, m_high, log_mc, n, plaw_index,
                                                a_wdm, b_wdm, c_wdm)
            integral_analytic = integrate_power_law_analytic(norm, m_low, m_high, n, plaw_index)

            analytic_integral = norm * (m_high ** (1 + plaw_index + n) - m_low ** (1 + plaw_index + n)) / (
                    n + 1 + plaw_index)
            npt.assert_almost_equal(integral/analytic_integral, 1, 4)
            npt.assert_almost_equal(integral_analytic/analytic_integral, 1, 5)

        log_mc = 8.
        a_wdm, b_wdm, c_wdm = 1., 1, -1.3
        for n in [0, 1]:
            integral = integrate_power_law_quad(norm, m_low, m_high, log_mc, n, plaw_index,
                                           a_wdm, b_wdm, c_wdm)

            analytic_integral = _analytic_integral_bound(m_high, plaw_index, c_wdm, 10 ** log_mc, n) - \
                                _analytic_integral_bound(m_low, plaw_index, c_wdm, 10 ** log_mc, n)

            npt.assert_almost_equal(integral/analytic_integral, 1, 4)

if __name__ == '__main__':
    pytest.main()

from scipy.integrate import quad
import numpy as np

class ScaleFree(object):

    @staticmethod
    def integrate_power_law_analytic(norm, m_low, m_high, n, plaw_index):
        """
        Numerically integrates a power law profile
        :param norm:
        :param m_low:
        :param m_high:
        :param n:
        :param plaw_index:
        :return:
        """

        factor = n + 1 + plaw_index

        if factor == 0:

            integral = np.log(m_high / m_low)

        else:

            integral = (m_high ** factor - m_low ** factor) / factor

        return norm * integral

    def integrate_power_law_quad(self, norm, m_low, m_high, n, plaw_index, **kwargs):

        return self.integrate_power_law_analytic(norm, m_low, m_high, n, plaw_index)

    def suppression(self, *args, **kwargs):

        return 1.

class PolynomialSuppression(ScaleFree):

    def integrate_power_law_quad(self, norm, m_low, m_high, n, plaw_index, log_mc, a_wdm, b_wdm, c_wdm, **kwargs):

        """
        Numerically integrates a double power law profile

        """

        def _integrand_wdm(m):
            return norm * m ** (n + plaw_index) * self.suppression(m, log_mc, a_wdm, b_wdm, c_wdm)

        moment = quad(_integrand_wdm, m_low, m_high)[0]

        return moment

    def suppression(self, m, log_mc, a_wdm, b_wdm, c_wdm):
        """
        Suppression function from Lovell et al. 2020
        :return: the factor that multiplies the CDM halo mass function to give the WDM halo mass function

        dN/dm (WDM) = dN/dm (CDM) * WDM_suppression

        where WDM suppression is (1 + a_wdm * (m_c / m)^b_wdm)^c_wdm
        """
        m_c = 10 ** log_mc
        r = a_wdm * (m_c / m) ** b_wdm
        factor = 1 + r

        return factor ** c_wdm

class MixedDMSuppression(ScaleFree):

    def integrate_power_law_quad(self, norm, m_low, m_high, n, plaw_index,
                                 log_mc, a_wdm, b_wdm, c_wdm, mixed_DM_frac, **kwargs):

        """
        Numerically integrates a double power law profile

        """

        def _integrand_wdm(m):
            return norm * m ** (n + plaw_index) * self.suppression(m, log_mc, a_wdm, b_wdm, c_wdm, mixed_DM_frac)

        moment = quad(_integrand_wdm, m_low, m_high)[0]

        return moment

    def suppression(self, m, log_mc, a_wdm, b_wdm, c_wdm, mixed_DM_frac):
        """

        """

        m_c = 10 ** log_mc
        r = a_wdm * (m_c / m) ** b_wdm
        factor = 1 + r
        wdm_comp = factor ** c_wdm
        suppression_factor = (mixed_DM_frac + (1 - mixed_DM_frac) * np.sqrt(wdm_comp)) ** 2
        return suppression_factor

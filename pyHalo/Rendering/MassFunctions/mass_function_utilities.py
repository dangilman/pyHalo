from scipy.integrate import quad
import numpy as np

def integrate_power_law_quad(norm, m_low, m_high, log_mc, n, plaw_index, a_wdm, b_wdm, c_wdm):

    """
    Numerically integrates a double power law profile

    """
    print('wdm suppression, integral')

    def _integrand_wdm(m, m_break, plaw_index, n):
        print('wdm suppression, integral')# this doesn't seem to be used...
        return norm * m ** (n + plaw_index) * WDM_suppression(m, m_break, a_wdm, b_wdm, c_wdm)
    def _integrand_cdm(m, plaw_index, n):
        return norm * m ** (n + plaw_index)

    if log_mc is not None:
        moment = quad(_integrand_wdm, m_low, m_high, args=(10 ** log_mc, plaw_index, n))[0]
    else:
        moment = quad(_integrand_cdm, m_low, m_high, args=(plaw_index, n))[0]

    return moment

def integrate_power_law_quad_MixDM(norm, m_low, m_high, log_mc, n, plaw_index, a_wdm, b_wdm, c_wdm, frac):

    """
    Numerically integrates a MixDM profile

    """
    print('using MixDM suppression, integrator')

    def _integrand_wdm(m, m_break, plaw_index, n):
        return norm * m ** (n + plaw_index) * MixDM_suppression(m, m_break, a_wdm, b_wdm, c_wdm, frac)
    def _integrand_cdm(m, plaw_index, n):
        return norm * m ** (n + plaw_index)

    if log_mc is not None:
        moment = quad(_integrand_wdm, m_low, m_high, args=(10 ** log_mc, plaw_index, n))[0]
    else:
        moment = quad(_integrand_cdm, m_low, m_high, args=(plaw_index, n))[0]

    return moment

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

        integral = np.log(m_high/m_low)

    else:

        integral = (m_high ** factor - m_low ** factor)/factor

    return norm * integral

def WDM_suppression(m, m_c, a_wdm, b_wdm, c_wdm):

    """
    Suppression function from Lovell et al. 2020
    :return: the factor that multiplies the CDM halo mass function to give the WDM halo mass function

    dN/dm (WDM) = dN/dm (CDM) * WDM_suppression

    where WDM suppression is (1 + a_wdm * (m_c / m)^b_wdm)^c_wdm
    """
    print('wdm suppression, definition')
    r = a_wdm * (m_c / m) ** b_wdm
    factor = 1 + r

    return factor ** c_wdm


def MixDM_suppression(m, m_c, a_wdm, b_wdm, c_wdm, frac):

    """
    Suppression function from Lovell et al. 2020
    :return: the factor that multiplies the CDM halo mass function to give the WDM halo mass function

    dN/dm (WDM) = dN/dm (CDM) * WDM_suppression

    where WDM suppression is (1 + a_wdm * (m_c / m)^b_wdm)^c_wdm
    """
    print('using MixDM suppression, definition')
    r = a_wdm * (m_c / m) ** b_wdm
    factor = 1 + r
    wdm_comp = factor ** c_wdm
    tot_supp = (frac + (1-frac)*np.sqrt(wdm_comp))**2
    return tot_supp

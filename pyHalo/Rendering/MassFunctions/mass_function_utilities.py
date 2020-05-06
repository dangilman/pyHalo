from scipy.integrate import quad
import numpy as np

def integrate_power_law_quad(norm, m_low, m_high, log_m_break, n, plaw_index, break_index=0, break_scale=1):

    def _integrand(m, m_break, plaw_index, n):
        return norm * m ** (n + plaw_index) * (1 + (m_break / m) ** break_scale) ** break_index

    moment = quad(_integrand, m_low, m_high, args=(10 ** log_m_break, plaw_index, n))[0]

    return moment

def integrate_power_law_analytic(norm, m_low, m_high, n, plaw_index):

    factor = n + 1 + plaw_index

    integral = (m_high ** factor - m_low ** factor)/factor

    return norm * integral

def WDM_suppression(m, m_half_mode, break_index, break_scale, break_coeff=1.):

    """
    Suppression function from Lovell et al.
    :return: the factor that multiplies the CDM halo mass function to give the WDM halo mass function

    dN/dm (WDM) = dN/dm (CDM) * WDM_suppression
    """
    ratio = m_half_mode/m
    factor = 1 + break_coeff * ratio ** break_scale
    return factor ** break_index

def subhalo_of_field_halo_suppression(m, m_host, N_0=0.21, alpha=0.8, k=6.283):

    """

    :param m: subhalo mass
    :param m_host: host halo mass
    :param k: factor from Giocoli et al. 2008
    :return: the factor that multiplies a power law to give the subhalo mass function from Giocoli et al. 2008
    """

    x = m * (alpha * m_host) ** -1
    return np.exp(-k * x ** 3)


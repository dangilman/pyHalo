import numpy as np
from scipy.integrate import quad


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

def integrate_power_law_quad(norm, m_low, m_high, n, plaw_index, turnover_function, kwargs_turnover):

    """
    Numerically integrates a power law profile with an arbitrary turnover or shape function
    """

    def _integrand_wdm(m):
        return norm * m ** (n + plaw_index) * turnover_function(m, **kwargs_turnover)
    moment = quad(_integrand_wdm, m_low, m_high)[0]
    return moment

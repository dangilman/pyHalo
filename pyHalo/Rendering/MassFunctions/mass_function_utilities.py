from scipy.integrate import quad

def integrate_power_law_quad(norm, m_low, m_high, log_m_break, n, plaw_index, break_index=0, break_scale=1):

    def _integrand(m, m_break, plaw_index, n):
        return norm * m ** (n + plaw_index) * (1 + (m_break / m) ** break_scale) ** break_index

    moment = quad(_integrand, m_low, m_high, args=(10 ** log_m_break, plaw_index, n))[0]

    return moment

def integrate_power_law_analytic(norm, m_low, m_high, n, plaw_index):

    factor = n + 1 + plaw_index

    integral = (m_high ** factor - m_low ** factor)/factor

    return norm * integral

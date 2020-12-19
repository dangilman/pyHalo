import numpy as np
from scipy.integrate import quad
from pyHalo.Rendering.MassFunctions.PowerLaw.powerlaw_base import PowerLawBase
from pyHalo.Rendering.MassFunctions.mass_function_utilities import WDM_suppression, subhalo_of_field_halo_suppression

class SubhaloPowerLaw(PowerLawBase):

    """
    Mass function for subhalos of LOS halos (not the subhalo mass function of main deflector)

    Functional form from Giocoli et al. 2008

    dN/dm = (N_0 / (alpha * M)) * x^-(1+alpha) * exp(-k * x^3)
    where x = m / (alpha*M)

    N_0 = 0.21
    alpha = 0.8
    k = 6.283
    """

    def __init__(self, log_mlow, M_parent, draw_poisson,
                 log_m_break, break_index, break_scale, N_0=0.21, alpha=0.8, k=6.283, **kwargs):

        raise Exception('this class is still under development')

        if break_index > 0:
            raise ValueError('Break index should be a negative number (otherwise mass function gets steeper (unphysical)')
        if break_scale < 0:
            raise ValueError('Break scale should be a positive number for suppression factor: '
                             '( 1 + (m/m_hm)^(break_scale )^break_index')

        self._N0, self._alpha, self._k = N_0, alpha, k

        self.log_m_break = log_m_break
        self.break_index = break_index
        self.break_scale = break_scale
        self.M_parent = M_parent

        max_M_scale = 0.8
        power_law_index = -(alpha + 1)

        m_high = max_M_scale * M_parent
        m_low = 10**log_mlow

        log_mhigh = 10**m_high

        if m_low/m_high > 0.1:
            self.Nhalos_mean = 0.

        else:

            norm = N_0 * (alpha * M_parent) ** -1

            x = lambda m: m * (alpha * M_parent) ** -1

            integrand = lambda m: norm * (m * x(m) ** power_law_index) * np.exp(-k * x(m) ** 3)

            self.Nhalos_mean = quad(integrand, m_low, m_high)[0]

        super(SubhaloPowerLaw, self).__init__(log_mlow, log_mhigh, power_law_index, draw_poisson)

    def draw(self):

        m = self.sample_from_power_law(self.Nhalos_mean)

        if self.log_m_break == 0 or len(m) == 0:
            return m

        factor = WDM_suppression(m, 10 ** self.log_m_break, self.break_index, self.break_scale)

        factor *= subhalo_of_field_halo_suppression(m, self.M_parent, self._N0, self._alpha, self._k)

        u = np.random.rand(int(len(m)))

        return m[np.where(u < factor)]



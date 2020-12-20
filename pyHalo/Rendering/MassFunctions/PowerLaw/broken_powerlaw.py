import numpy as np
from pyHalo.Rendering.MassFunctions.PowerLaw.powerlaw_base import PowerLawBase
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_analytic
from pyHalo.Rendering.MassFunctions.mass_function_utilities import WDM_suppression

class BrokenPowerLaw(PowerLawBase):

    """
    This class handles computations of a double power law mass function of the form
    dn/dm = m^x * (1 + (a * m_c / m)^b)^c
    where a, b, and c are constants, and m_c is a characteristic mass scale.

    The keywords for a, b, c are a_wdm, b_wdm, and c_wdm, respectively

    Lovell 2020 fit this mass function to simulations of Warm Dark Matter cosmologies and find
    (a, b, c) = (2.3, 0.8, -1) for central halos and (4.2, 2.5, -0.2) for subhalos
    """

    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson, normalization,
                 log_mc, a_wdm, b_wdm, c_wdm):

        if a_wdm is None:
            assert b_wdm is None, 'If one of a_wdm, b_wdm, or c_wdm is not specified (None), all parameters must be None'
            assert c_wdm is None, 'If one of a_wdm, b_wdm, or c_wdm is not specified (None), all parameters must be None'
        else:
            assert b_wdm is not None, 'Must specify values for all three of a_wdm, b_wdm, c_wdm'
            assert c_wdm is not None, 'Must specify values for all three of a_wdm, b_wdm, c_wdm'
        if b_wdm is None:
            assert a_wdm is None, 'If one of a_wdm, b_wdm, or c_wdm is not specified (None), all parameters must be None'
            assert c_wdm is None, 'If one of a_wdm, b_wdm, or c_wdm is not specified (None), all parameters must be None'
        else:
            assert a_wdm is not None, 'Must specify values for all three of a_wdm, b_wdm, c_wdm'
            assert c_wdm is not None, 'Must specify values for all three of a_wdm, b_wdm, c_wdm'
        if c_wdm is None:
            assert a_wdm is None, 'If one of a_wdm, b_wdm, or c_wdm is not specified (None), all parameters must be None'
            assert b_wdm is None, 'If one of a_wdm, b_wdm, or c_wdm is not specified (None), all parameters must be None'
        else:
            assert a_wdm is not None, 'Must specify values for all three of a_wdm, b_wdm, c_wdm'
            assert b_wdm is not None, 'Must specify values for all three of a_wdm, b_wdm, c_wdm'

        if normalization < 0:
            raise Exception('normalization cannot be < 0.')
        if c_wdm is not None and c_wdm > 0:
            raise ValueError('c_wdm should be a negative number (otherwise mass function gets steeper (unphysical)')
        if a_wdm is not None and a_wdm < 0:
            raise ValueError('a_wdm should be a positive number for suppression factor: '
                             '( 1 + (a_wdm * m/m_c)^b_wdm)^c_wdm')

        if np.any([a_wdm is None, b_wdm is None, c_wdm is None]):
            assert log_mc is None, 'If log_mc is specified, must also specify kwargs for a_wdm, b_wdm, c_wdm.' \
                                   '(See documentation in pyHalo/Rendering/MassFunctions/Powerlaw/broken_powerlaw'

        self._log_mc = log_mc
        self._a_wdm = a_wdm
        self._b_wdm = b_wdm
        self._c_wdm = c_wdm

        self._nhalos_mean_unbroken = integrate_power_law_analytic(normalization, 10 ** log_mlow, 10 ** log_mhigh, 0,
                                                                  power_law_index)

        super(BrokenPowerLaw, self).__init__(log_mlow, log_mhigh, power_law_index, draw_poisson)

    def draw(self):

        m = self.sample_from_power_law(self._nhalos_mean_unbroken)

        if len(m) == 0 or self._log_mc is None:
            return m

        factor = WDM_suppression(m, 10 ** self._log_mc, self._a_wdm, self._b_wdm, self._c_wdm)
        u = np.random.rand(int(len(m)))
        return m[np.where(u < factor)]

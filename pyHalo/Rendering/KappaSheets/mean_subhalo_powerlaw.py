from copy import deepcopy
from pyHalo.Rendering.KappaSheets.base import KappaSheetBase
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, integrate_power_law_analytic
import numpy as np

class MeanSubhaloPowerLaw(KappaSheetBase):

    def __call__(self, log_mlow=None, log_mhigh=None, realization=None):

        if 'subtract_exact_mass_sheets' in self.kwargs_rendering:

            return self.convergence_in_planes_exact(realization)

        else:

            return self.convergence_in_planes_theory(log_mlow, log_mhigh)

    def convergence_in_planes_theory(self, log_mlow, log_mhigh):

        mass_in_planes = self.mass_in_planes_theory(log_mlow, log_mhigh)

        mass_in_planes *= self.kwargs_rendering['subhalo_mass_sheet_scale']

        return self.negative_kappa_sheets(mass_in_planes)

    def mass_in_planes_theory(self, log_mlow, log_mhigh):

        kwargs = deepcopy(self.kwargs_rendering)

        kwargs['log_mlow'], kwargs['log_mhigh'] = log_mlow, log_mhigh

        norm = self.normalization_function(self.kwargs_rendering)

        m_low, m_high = 10 ** log_mlow, 10 ** log_mhigh

        plaw_index = kwargs['power_law_index']
        log_m_break = self.kwargs_rendering['log_m_break']
        break_index = self.kwargs_rendering['break_index']
        break_scale = self.kwargs_rendering['break_scale']

        moment = 1

        if log_m_break == 0 or log_m_break / log_mlow < 0.01:
            mass = integrate_power_law_analytic(norm, m_low, m_high, moment, plaw_index)
        else:
            mass = integrate_power_law_quad(norm, m_low, m_high, log_m_break, moment,
                                            plaw_index, break_index, break_scale)

        return np.array(mass)

from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as npt
import pytest


class TestRender(object):

    def setup(self):

        mass_definition = 'TNFW'
        zlens, zsource = 0.6, 1.8
        kwargs_halo_mass_function = {'geometry_type': 'DOUBLE_CONE'}
        pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_halo_mass_function)

        cone_opening_angle = 8.
        log_mlow = 6.
        log_mhigh = 9.
        power_law_index = -1.9
        delta_power_law_index = 0.
        sigma_sub = 0.0
        self._LOS_norm = 10.
        c0, beta, zeta = 17.5, 0.8, -0.25
        realization_kwargs = {'mass_func_type': 'POWER_LAW',
                              'log_mlow': log_mlow,
                              'log_mhigh': log_mhigh,
                              'log_mass_sheet_min': log_mlow,
                              'log_mass_sheet_max': 10,
                              'log_m_parent': 13.2,
                              'mdef_main': mass_definition,
                              'mdef_los': mass_definition, 'sigma_sub': sigma_sub,
                              'cone_opening_angle': cone_opening_angle, 'r_tidal': '0.25Rs',
                              'power_law_index': power_law_index,
                              'mc_model': {'custom': True, 'c0': c0, 'beta': beta, 'zeta': zeta},
                              'delta_power_law_index': delta_power_law_index,
                              'LOS_normalization': self._LOS_norm, 'subhalo_convergence_correction_profile': 'UNIFORM',
                              'subtract_exact_mass_sheets': False}

        realization_type = 'composite_powerlaw'

        self.realization_cdm = pyhalo.render(realization_type, realization_kwargs, nrealizations=1)[0]

        self.geometry = self.realization_cdm.geometry

        self.lensing_mass_function = LensingMassFunction(self.realization_cdm.lens_cosmo.cosmo,
                  10**6, 10**9, zlens, zsource, cone_opening_angle,
                 m_pivot=10**8)

        mass_definition = 'TNFW'  # black holes are by definition point masses
        zlens, zsource = 0.6, 1.8
        kwargs_halo_mass_function = {'geometry_type': 'DOUBLE_CONE'}
        pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_halo_mass_function)

        cone_opening_angle = 8.
        log_mlow = 6.
        log_mhigh = 9.
        power_law_index = -1.9
        self._delta_power_law_index = 0.2
        sigma_sub = 0.0
        self._LOS_norm = 10.
        c0, beta, zeta = 17.5, 0.8, -0.25
        realization_kwargs = {'mass_func_type': 'POWER_LAW',
                              'log_mlow': log_mlow,
                              'log_mhigh': log_mhigh,
                              'log_mass_sheet_min': log_mlow,
                              'log_mass_sheet_max': 10,
                              'log_m_parent': 13.2,
                              'mdef_main': mass_definition,
                              'mdef_los': mass_definition, 'sigma_sub': sigma_sub,
                              'cone_opening_angle': cone_opening_angle, 'r_tidal': '0.25Rs',
                              'power_law_index': power_law_index,
                              'mc_model': {'custom': True, 'c0': c0, 'beta': beta, 'zeta': zeta},
                              'delta_power_law_index': self._delta_power_law_index,
                              'LOS_normalization': self._LOS_norm, 'subhalo_convergence_correction_profile': 'UNIFORM',
                              'subtract_exact_mass_sheets': False}

        realization_type = 'composite_powerlaw'

        self.realization_cdm_delta_plaw_index = pyhalo.render(realization_type, realization_kwargs, nrealizations=1)[0]

        self.geometry_delta_plaw_index = self.realization_cdm_delta_plaw_index.geometry

        self.lensing_mass_function_delta_plaw_index = LensingMassFunction(
            self.realization_cdm_delta_plaw_index.lens_cosmo.cosmo,
                  10**6, 10**9, zlens, zsource, cone_opening_angle)

    def test_mass_rendered(self):

        m = np.logspace(6, 9, 20)
        delta_z = 0.02
        m_pivot = 10**8

        ratios = []
        for zi in self.realization_cdm.unique_redshifts:
            mass_at_redshift = self.realization_cdm.mass_at_z_exact(zi)
            volume_at_redshift = self.geometry.volume_element_comoving(zi, delta_z)

            norm_dV, plaw_index = self.lensing_mass_function._mass_function_params(m, zi)

            m_pivot_scale = 1/(m_pivot**plaw_index)
            norm = self._LOS_norm * m_pivot_scale * norm_dV * volume_at_redshift
            mass_theory = self.lensing_mass_function.integrate_power_law(norm, 10**6, 10**9, 0., 1, plaw_index)
            ratios.append(mass_theory/mass_at_redshift)

        npt.assert_almost_equal(np.median(ratios), 1, 1)

        ratios = []
        for zi in self.realization_cdm_delta_plaw_index.unique_redshifts:
            mass_at_redshift = self.realization_cdm_delta_plaw_index.mass_at_z_exact(zi)
            volume_at_redshift = self.geometry_delta_plaw_index.volume_element_comoving(zi, delta_z)

            norm_dV, plaw_index = self.lensing_mass_function_delta_plaw_index._mass_function_params(m, zi)
            plaw_index += self._delta_power_law_index
            m_pivot_scale = 1 / (m_pivot ** plaw_index)
            norm = self._LOS_norm * m_pivot_scale * norm_dV * volume_at_redshift
            mass_theory = self.lensing_mass_function_delta_plaw_index.integrate_power_law(norm, 10 ** 6, 10 ** 9, 0., 1, plaw_index)
            ratios.append(mass_theory / mass_at_redshift)

        npt.assert_almost_equal(np.median(ratios), 1, 1)

if __name__ == '__main__':
    pytest.main()

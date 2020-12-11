from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as npt
import pytest


class TestWDM(object):

    def setup(self):

        mass_definition = 'TNFW'
        zlens, zsource = 0.6, 2.
        kwargs_halo_mass_function = {'geometry_type': 'DOUBLE_CONE'}
        pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_halo_mass_function)

        cone_opening_angle = 10.
        log_mlow = 6.
        log_mhigh = 9
        power_law_index = -1.9
        delta_power_law_index = 0.
        sigma_sub = 0.9
        self._LOS_norm = 10.
        self._logmhm = 8.
        self._break_index = -1.2
        self._break_scale = 1.

        realization_kwargs = {'mass_func_type': 'POWER_LAW',
                              'log_mlow': log_mlow,
                              'log_mhigh': log_mhigh,
                              'log_mass_sheet_min': log_mlow,
                              'log_mass_sheet_max': 10,
                              'log_m_parent': 13.2,
                              'log_m_break': self._logmhm,
                              'break_index': self._break_index,
                              'break_scale': self._break_scale,
                              'mdef_main': mass_definition,
                              'mdef_los': mass_definition, 'sigma_sub': sigma_sub,
                              'cone_opening_angle': cone_opening_angle, 'r_tidal': '0.25Rs',
                              'power_law_index': power_law_index,
                              'delta_power_law_index': delta_power_law_index,
                              'LOS_normalization': self._LOS_norm, 'subhalo_convergence_correction_profile': 'UNIFORM',
                              'subtract_exact_mass_sheets': False}

        realization_type = 'main_lens'

        self.realization_wdm_1 = pyhalo.render(realization_type, realization_kwargs, nrealizations=1)[0]

        realization_kwargs = {'mass_func_type': 'POWER_LAW',
                              'log_mlow': log_mlow,
                              'log_mhigh': log_mhigh,
                              'log_mass_sheet_min': log_mlow,
                              'log_mass_sheet_max': 10,
                              'log_m_parent': 13.2,
                              'log_m_break': 1.,
                              'break_index': self._break_index,
                              'break_scale': self._break_scale,
                              'mdef_main': mass_definition,
                              'mdef_los': mass_definition, 'sigma_sub': sigma_sub,
                              'cone_opening_angle': cone_opening_angle, 'r_tidal': '0.25Rs',
                              'power_law_index': power_law_index,
                              'delta_power_law_index': delta_power_law_index,
                              'LOS_normalization': self._LOS_norm, 'subhalo_convergence_correction_profile': 'UNIFORM',
                              'subtract_exact_mass_sheets': False}
        self.realization_cdm = pyhalo.render(realization_type, realization_kwargs, nrealizations=1)[0]

        self.lensing_mass_function = LensingMassFunction(self.realization_wdm_1.lens_cosmo.cosmo,
                  10**6, 10**10, zlens, zsource, cone_opening_angle,
                 m_pivot=10**8)

    def test_wdm_mass_function(self):

        mass_bins = np.linspace(6, 9, 20)
        halo_masses = [halo.mass for halo in self.realization_wdm_1.halos]
        log_halo_mass = np.log10(halo_masses)
        h_1, logM = np.histogram(log_halo_mass, bins=mass_bins)
        logmstep = (logM[1] - logM[0]) / 2
        logM = logM[0:-1] + logmstep

        mass_bins = np.linspace(6, 9, 20)
        halo_masses = [halo.mass for halo in self.realization_cdm.halos]
        log_halo_mass = np.log10(halo_masses)
        h_2, _ = np.histogram(log_halo_mass, bins=mass_bins)

        m_ratio = 10 ** (self._logmhm - logM)

        suppression_factor = (1 + m_ratio ** self._break_scale) ** self._break_index

        slope_modeled = np.polyfit(logM[0:4], np.log10(h_1)[0:4], 1)[0]
        slope_predicted = -0.9 - self._break_index

        diff_slope = np.absolute(slope_modeled - slope_predicted)

        diff = h_2 * suppression_factor / h_1
        npt.assert_almost_equal(np.mean(diff), 1, 1)
        npt.assert_almost_equal(diff_slope, 0., 1)

        dndm = np.gradient(h_1[0:5], 10 ** logM[0:5])
        logarithmic_slope_model = np.polyfit(logM[0:5], np.log10(dndm), 1)[0]
        logarithmic_slope_predicted = -1.9 - self._break_index
        npt.assert_almost_equal(logarithmic_slope_model, logarithmic_slope_predicted, 0.1)

    def test_cdm_mass_function(self):

        mass_bins = np.linspace(6, 9, 20)
        halo_masses = [halo.mass for halo in self.realization_cdm.halos]
        log_halo_mass = np.log10(halo_masses)
        h_1, logM = np.histogram(log_halo_mass, bins=mass_bins)
        logmstep = (logM[1] - logM[0]) / 2
        logM = logM[0:-1] + logmstep

        dndm = np.gradient(h_1, 10**logM)
        logarithmic_slope_model = np.polyfit(logM, np.log10(-dndm), 1)[0]
        npt.assert_almost_equal(logarithmic_slope_model, -1.9, 1)


if __name__ == '__main__':
    pytest.main()

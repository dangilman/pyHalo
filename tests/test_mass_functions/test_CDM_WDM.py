from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Rendering.MassFunctions.mass_function_utilities import WDM_suppression
import numpy as np
import numpy.testing as npt
import pytest

class TestMassFunctionSlopes(object):

    def setup(self):

        mass_definition = 'TNFW'
        zlens, zsource = 0.6, 2.
        kwargs_halo_mass_function = {'geometry_type': 'DOUBLE_CONE'}
        pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_halo_mass_function)

        cone_opening_angle = 10.
        log_mlow = 6.
        log_mhigh = 9
        power_law_index = -1.9
        delta_power_law_index = -0.1
        self.delta_plaw_index = delta_power_law_index
        sigma_sub = 0.9
        self._LOS_norm = 10.
        self._logmc = 7.5

        self._awdm = 2.3
        self._bwdm = 0.6
        self._cwdm = -1.5

        realization_kwargs = {'mass_func_type': 'POWER_LAW',
                              'log_mlow': log_mlow,
                              'log_mhigh': log_mhigh,
                              'log_mass_sheet_min': log_mlow,
                              'log_mass_sheet_max': 10,
                              'log_m_host': 13.2,
                              'log_mc': self._logmc,
                              'a_wdm': self._awdm,
                              'b_wdm': self._bwdm,
                              'c_wdm': self._cwdm,
                              'mdef_main': mass_definition,
                              'mdef_los': mass_definition, 'sigma_sub': sigma_sub,
                              'cone_opening_angle': cone_opening_angle, 'r_tidal': '0.25Rs',
                              'power_law_index': power_law_index,
                              'delta_power_law_index': delta_power_law_index,
                              'LOS_normalization': self._LOS_norm, 'subhalo_convergence_correction_profile': 'UNIFORM',
                              'subtract_exact_mass_sheets': False}
        self.realization_cdm_kwargs = realization_kwargs
        realization_type = 'main_lens'

        self.realization_wdm_1 = pyhalo.render(realization_type, realization_kwargs, nrealizations=1)[0]

        realization_kwargs = {'mass_func_type': 'POWER_LAW',
                              'log_mlow': log_mlow,
                              'log_mhigh': log_mhigh,
                              'log_mass_sheet_min': log_mlow,
                              'log_mass_sheet_max': 10,
                              'log_m_host': 13.2,
                              'log_mc': None,
                              'a_wdm': None,
                              'b_wdm': None,
                              'c_wdm': None,
                              'mdef_main': mass_definition,
                              'mdef_los': mass_definition, 'sigma_sub': sigma_sub,
                              'cone_opening_angle': cone_opening_angle, 'r_tidal': '0.25Rs',
                              'power_law_index': power_law_index,
                              'delta_power_law_index': delta_power_law_index,
                              'LOS_normalization': self._LOS_norm, 'subhalo_convergence_correction_profile': 'UNIFORM',
                              'subtract_exact_mass_sheets': False}
        self.realization_cdm = pyhalo.render(realization_type, realization_kwargs, nrealizations=1)[0]
        self.realization_wdm_kwargs = realization_kwargs
        self.lensing_mass_function = LensingMassFunction(self.realization_wdm_1.lens_cosmo.cosmo,
                  10**6, 10**10, zlens, zsource, cone_opening_angle,
                 m_pivot=10**8)

    def test_wdm_mass_function(self):

        mass_bins = np.linspace(6, 9, 20)
        halo_masses_wdm = [halo.mass for halo in self.realization_wdm_1.halos]
        log_halo_mass = np.log10(halo_masses_wdm)
        h_wdm, logM = np.histogram(log_halo_mass, bins=mass_bins)
        logmstep = (logM[1] - logM[0]) / 2
        logM = logM[0:-1] + logmstep

        mass_bins = np.linspace(6, 9, 20)
        halo_masses_cdm = [halo.mass for halo in self.realization_cdm.halos]
        log_halo_mass = np.log10(halo_masses_cdm)
        h_cdm, _ = np.histogram(log_halo_mass, bins=mass_bins)

        suppression_factor = WDM_suppression(10**logM, 10**self._logmc, self._awdm, self._bwdm, self._cwdm)

        for i in range(0, 10):
            ratio = h_wdm[i]/h_cdm[i]
            npt.assert_almost_equal(ratio, suppression_factor[i], 1)

        halo_masses_cdm = np.array(halo_masses_cdm)
        halo_masses_wdm = np.array(halo_masses_wdm)

        step = 0.2
        msteps = 10**np.arange(6, 6.6 + step, step=0.2)
        logmsteps = np.log10(msteps)[0:-1]
        dn_dM_cdm, dn_dM_wdm = [], []
        for i in range(0, len(msteps)-1):
            cond1 = halo_masses_cdm >= msteps[i]
            cond2 = halo_masses_cdm < msteps[i+1]
            condition = np.logical_and(cond1, cond2)
            cond1 = halo_masses_wdm >= msteps[i]
            cond2 = halo_masses_wdm < msteps[i + 1]
            condition_wdm = np.logical_and(cond1, cond2)
            ncdm = np.sum(condition)
            nwdm = np.sum(condition_wdm)
            dn_dM_cdm.append(ncdm / msteps[i])
            dn_dM_wdm.append(nwdm / msteps[i])

        dn_dM_cdm = np.array(dn_dM_cdm)
        dn_dM_wdm = np.array(dn_dM_wdm)
        differential_slope_cdm = np.polyfit(logmsteps, np.log10(dn_dM_cdm), 1)[0]
        differential_slope_wdm = np.polyfit(logmsteps, np.log10(dn_dM_wdm), 1)[0]

        log_slope_predictd_cdm = -1.9 + self.delta_plaw_index
        log_slope_predictd_wdm = -1.9 + self.delta_plaw_index - (self._bwdm * self._cwdm)

        npt.assert_almost_equal(abs(1-differential_slope_cdm/log_slope_predictd_cdm), 0, 1)
        npt.assert_almost_equal(abs(1-differential_slope_wdm/log_slope_predictd_wdm), 0, 1)

    def test_cdm_mass_function(self):

        mass_bins = np.linspace(6, 8, 20)
        halo_masses = [halo.mass for halo in self.realization_cdm.halos]
        log_halo_mass = np.log10(halo_masses)
        h_1, logM = np.histogram(log_halo_mass, bins=mass_bins)

        logN = np.log10(h_1)

        n_per_unit_mass_slope = -0.9 + self.delta_plaw_index
        slope = np.polyfit(logM[0:-1], logN, 1)[0]
        npt.assert_almost_equal(slope, n_per_unit_mass_slope, 2)


if __name__ == '__main__':
    pytest.main()

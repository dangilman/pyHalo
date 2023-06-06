from pyHalo.preset_models import *
import pytest
import numpy as np
import numpy.testing as npt


class TestPresetModels(object):

    def test_CDM(self):

        cdm = CDM(0.5, 1.5)
        _ = cdm.lensing_quantities()
        _ = preset_model_from_name('CDM')

    def test_WDM(self):

        wdm = WDM(0.5, 1.5, 8.0)
        _ = wdm.lensing_quantities()
        _ = preset_model_from_name('WDM')

    def test_ULDM(self):

        flucs_shape = 'ring'
        flucs_args = {'angle': 0.0, 'rmin': 0.9, 'rmax': 1.1}
        uldm = ULDM(0.5, 1.5, -21, flucs_shape=flucs_shape, flucs_args=flucs_args)
        _ = uldm.lensing_quantities()
        _ = preset_model_from_name('ULDM')

    def test_SIDM_core_collapse(self):
        mass_ranges_subhalos = [[6, 8], [8, 10]]
        mass_ranges_field_halos = [[6, 8], [8, 10]]
        probabilities_subhalos = [1, 1]
        probabilities_field_halos = [1, 1]
        sidm_cc = SIDM_core_collapse(0.5, 1.5, mass_ranges_subhalos, mass_ranges_field_halos,
        probabilities_subhalos, probabilities_field_halos)
        _ = sidm_cc.lensing_quantities()
        _ = preset_model_from_name('SIDM_core_collapse')

    def test_WDM_mixed(self):
        wdm_mixed = WDM_mixed(0.5, 1.5, 8.0, 0.5)
        _ = wdm_mixed.lensing_quantities()
        _ = preset_model_from_name('WDM_mixed')

    def test_CDM_emulator(self):

        def emulator_input_callable(*args, **kwargs):
            subhalo_infall_masses = np.array([10**7,10**8])
            subhalo_x_kpc = np.array([1.0, 1.0])
            subhalo_y_kpc = np.array([1.0, 1.0])
            subhalo_final_bound_masses = subhalo_infall_masses / 2
            subhalo_infall_concentrations = np.array([16.0, 20.0])
            return subhalo_infall_masses, subhalo_x_kpc, subhalo_y_kpc, subhalo_final_bound_masses, subhalo_infall_concentrations

        concentrations = np.array([16.0, 20.0])
        mass_array = np.array([10 ** 7, 10 ** 8])
        kwargs_cdm = {'LOS_normalization': 0.0}
        cdm_subhalo_emulator = CDMFromEmulator(0.5, 1.5, emulator_input_callable, kwargs_cdm)
        _ = cdm_subhalo_emulator.lensing_quantities()
        for i, halo in enumerate(cdm_subhalo_emulator.halos):
            npt.assert_equal(halo.mass, mass_array[i])
            npt.assert_almost_equal(halo.x, 0.1584666, 4)
            npt.assert_almost_equal(halo.y, 0.1584666, 4)
            npt.assert_equal(halo.c, concentrations[i])

        emulator_input_array = np.empty((2, 5))
        emulator_input_array[:, 0] = mass_array
        emulator_input_array[:, 1] = np.array([1.0, 1.0])
        emulator_input_array[:, 2] = np.array([1.0, 1.0])
        emulator_input_array[:, 3] = mass_array / 2
        emulator_input_array[:, 4] = concentrations
        cdm_subhalo_emulator = CDMFromEmulator(0.5, 1.5, emulator_input_array, kwargs_cdm)
        _ = cdm_subhalo_emulator.lensing_quantities()
        for i, halo in enumerate(cdm_subhalo_emulator.halos):
            npt.assert_equal(halo.mass, mass_array[i])
            npt.assert_almost_equal(halo.x, 0.1584666, 4)
            npt.assert_almost_equal(halo.y, 0.1584666, 4)
            npt.assert_equal(halo.c, concentrations[i])

if __name__ == '__main__':
     pytest.main()

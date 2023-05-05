from pyHalo.preset_models import *
import pytest


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

if __name__ == '__main__':
     pytest.main()

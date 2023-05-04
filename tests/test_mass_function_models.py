import numpy.testing as npt
import pytest
from pyHalo.mass_function_models import preset_mass_function_models

class TestMassFunctionModels(object):

    def setup_method(self):
        self.model_list = ['SHMF_LOVELL2020', 'LOVELL2020', 'SHMF_LOVELL2014',
                           'LOVELL2014', 'SCHIVE2016', 'SHMF_SCHIVE2016', 'POWER_LAW',
                           'POWER_LAW_TURNOVER']
        self.model_names = ['WDM_POWER_LAW', 'SHETH_TORMEN', 'WDM_POWER_LAW',
                            'SHETH_TORMEN', 'SHETH_TORMEN', 'WDM_POWER_LAW', 'CDM_POWER_LAW',
                            'WDM_POWER_LAW']

    def test_models(self):

        for model, model_name in zip(self.model_list, self.model_names):
            mod, kw = preset_mass_function_models(model)
            npt.assert_string_equal(model_name, mod.name)


if __name__ == '__main__':
    pytest.main()

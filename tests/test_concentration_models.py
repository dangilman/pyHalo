import numpy.testing as npt
import pytest
from pyHalo.concentration_models import preset_concentration_models

class TestConcentrationModels(object):

    def setup_method(self):

        self.model_list = ['DIEMERJOYCE19', 'PEAK_HEIGHT_POWERLAW', 'WDM_HYPERBOLIC',
                           'WDM_POLYNOMIAL', 'BOSE2016', 'LAROCHE2022']
        self.model_names = ['DIEMERJOYCE19', 'PEAK_HEIGHT_POWERLAW', 'WDM_HYPERBOLIC',
                            'WDM_POLYNOMIAL', 'WDM_POLYNOMIAL', 'WDM_POLYNOMIAL']

    def test_models(self):

        for model, model_name in zip(self.model_list, self.model_names):
            mod, kw = preset_concentration_models(model)
            npt.assert_string_equal(model_name, mod.name)

if __name__ == '__main__':
     pytest.main()

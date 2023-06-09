import numpy.testing as npt
import pytest
from pyHalo.concentration_models import preset_concentration_models
import numpy as np

class _CustomMC(object):

    def __init__(self, a, b):

        self._a = a
        self._b = b

    def nfw_concentration(self, m, z):
        return self._a * (m/10**8) ** self._b * (1 + z) ** -1

class _CustomMCInvalid(object):

    def return_concentration(self, m, z):
        return 16.0 * (m / 10 ** 8) ** -0.2 * (1 + z) ** -0.5

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

    def test_custom_class(self):

        kwargs_model = {'custom_class': _CustomMC, 'a': 10.0, 'b': -1.0}
        mod, kw = preset_concentration_models('CUSTOM', kwargs_model)
        concentration_model = mod(**kw)
        c = concentration_model.nfw_concentration(10 ** 7.5, 0.5)
        npt.assert_almost_equal(c, kwargs_model['a'] * (10 ** 7.5 / 10 ** 8) ** kwargs_model['b'] / 1.5)

        kwargs_model = {'custom_class': _CustomMCInvalid}
        args = ('CUSTOM', kwargs_model)
        npt.assert_raises(Exception, preset_concentration_models, args)

    def test_formation_history(self):

        kwargs_model = {'dlogT_dlogk': 2.0}
        mod, kw = preset_concentration_models('FROM_FORMATION_HISTORY', kwargs_model)
        concentration_model = mod(**kw)

if __name__ == '__main__':
     pytest.main()

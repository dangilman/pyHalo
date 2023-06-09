import numpy as np
import pytest
import numpy.testing as npt
from pyHalo.Rendering.MassFunctions.stucker import stucker_suppression_params, _supp_mscale
from pyHalo.mass_function_models import preset_mass_function_models


class TestStucker(object):

    def test_load_model(self):

        kwargs_model = {'dlogT_dlogk': 2.0}
        model, kwargs_stucker = preset_mass_function_models('STUCKER', kwargs_model)
        a_wdm, b_wdm, c_wdm = kwargs_stucker['a_wdm'], kwargs_stucker['b_wdm'], kwargs_stucker['c_wdm']
        a, b, c = stucker_suppression_params(2.0)
        npt.assert_almost_equal(a, a_wdm)
        npt.assert_almost_equal(b, b_wdm)
        npt.assert_almost_equal(c, c_wdm)

        kwargs_model = {}
        args = ('STUCKER', kwargs_model)
        npt.assert_raises(Exception, preset_mass_function_models, *args)
        kwargs_model = {'dlogT_dlogk': 2.0, 'a_wdm': 1.0}
        args = ('STUCKER', kwargs_model)
        npt.assert_raises(Exception, preset_mass_function_models, *args)
        kwargs_model = {'dlogT_dlogk': 2.0, 'b_wdm': 1.0}
        args = ('STUCKER', kwargs_model)
        npt.assert_raises(Exception, preset_mass_function_models, *args)
        kwargs_model = {'dlogT_dlogk': 2.0, 'c_wdm': 1.0}
        args = ('STUCKER', kwargs_model)
        npt.assert_raises(Exception, preset_mass_function_models, *args)

    def test_abc_params(self):
        dlogT_dlogk = 2.0
        _a_wdm, b_wdm, c_wdm = stucker_suppression_params(dlogT_dlogk)
        a_wdm = _a_wdm ** (1/b_wdm)

        theory_mass_scale_m20 = _supp_mscale(a_wdm, b_wdm, c_wdm, frac=0.2)
        theory_mass_scale_m50 = _supp_mscale(a_wdm, b_wdm, c_wdm, frac=0.5)
        theory_mass_scale_m80 = _supp_mscale(a_wdm, b_wdm, c_wdm, frac=0.8)

        suppression_model = (1 + (a_wdm/theory_mass_scale_m20) ** b_wdm) ** c_wdm
        npt.assert_almost_equal(suppression_model, 0.2)
        suppression_model = (1 + (a_wdm/theory_mass_scale_m50) ** b_wdm) ** c_wdm
        npt.assert_almost_equal(suppression_model, 0.5)
        suppression_model = (1 + (a_wdm/theory_mass_scale_m80) ** b_wdm) ** c_wdm
        npt.assert_almost_equal(suppression_model, 0.8)

        dlogT_dlogk = 4.0
        _a_wdm, b_wdm, c_wdm = stucker_suppression_params(dlogT_dlogk)
        a_wdm = _a_wdm ** (1 / b_wdm)

        theory_mass_scale_m20 = _supp_mscale(a_wdm, b_wdm, c_wdm, frac=0.2)
        theory_mass_scale_m50 = _supp_mscale(a_wdm, b_wdm, c_wdm, frac=0.5)
        theory_mass_scale_m80 = _supp_mscale(a_wdm, b_wdm, c_wdm, frac=0.8)

        suppression_model = (1 + (a_wdm / theory_mass_scale_m20) ** b_wdm) ** c_wdm
        npt.assert_almost_equal(suppression_model, 0.2)
        suppression_model = (1 + (a_wdm / theory_mass_scale_m50) ** b_wdm) ** c_wdm
        npt.assert_almost_equal(suppression_model, 0.5)
        suppression_model = (1 + (a_wdm / theory_mass_scale_m80) ** b_wdm) ** c_wdm
        npt.assert_almost_equal(suppression_model, 0.8)

        dlogT_dlogk = 6.0
        _a_wdm, b_wdm, c_wdm = stucker_suppression_params(dlogT_dlogk)
        a_wdm = _a_wdm ** (1 / b_wdm)

        theory_mass_scale_m20 = _supp_mscale(a_wdm, b_wdm, c_wdm, frac=0.2)
        theory_mass_scale_m50 = _supp_mscale(a_wdm, b_wdm, c_wdm, frac=0.5)
        theory_mass_scale_m80 = _supp_mscale(a_wdm, b_wdm, c_wdm, frac=0.8)

        suppression_model = (1 + (a_wdm / theory_mass_scale_m20) ** b_wdm) ** c_wdm
        npt.assert_almost_equal(suppression_model, 0.2)
        suppression_model = (1 + (a_wdm / theory_mass_scale_m50) ** b_wdm) ** c_wdm
        npt.assert_almost_equal(suppression_model, 0.5)
        suppression_model = (1 + (a_wdm / theory_mass_scale_m80) ** b_wdm) ** c_wdm
        npt.assert_almost_equal(suppression_model, 0.8)


if __name__ == '__main__':
   pytest.main()

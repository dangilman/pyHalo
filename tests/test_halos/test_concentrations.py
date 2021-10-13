import pytest
import numpy as np
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import numpy.testing as npt
from pyHalo.Halos.concentration import Concentration, WDM_concentration_suppresion_factor

class TestConcentration(object):

    def setup(self):

        cosmo = Cosmology()
        self.cosmo = cosmo
        self.lenscosmo = LensCosmo(0.5, 1.5, cosmo)
        self.concentration = Concentration(self.lenscosmo)

    def test_concentration(self):


        m = 10 ** 8
        z = 0.5
        kwargs_suppresion, suppression_model = {'c_scale': 60., 'c_power': -0.17}, 'polynomial'
        c = self.concentration.nfw_concentration(m, z, 'diemer19', '200c', None, False, 0.1,
                                                 kwargs_suppresion, suppression_model)
        npt.assert_array_less(0., c)

        m_array = np.logspace(6, 6, 10)
        z = 0.5
        c_array = self.concentration.nfw_concentration(m_array, z, 'diemer19', '200c', None, False,
                                                 0.1, kwargs_suppresion, suppression_model)
        npt.assert_equal(True, len(c_array)==10)

        z_array = np.linspace(0.5, 0.5, 10)
        c_array = self.concentration.nfw_concentration(m, z_array, 'diemer19', '200c', None, False,
                                                       0.1, kwargs_suppresion, suppression_model)
        npt.assert_equal(True, len(c_array) == 10)
        for ci in c_array:
            npt.assert_equal(ci, c)

        m_array_2, z_array_2 = np.logspace(6, 8, 10), np.linspace(0.2, 1., 10)
        c_array = self.concentration.nfw_concentration(m_array_2, z_array_2, 'diemer19', '200c', None, False,
                                                       0.1, kwargs_suppresion, suppression_model)
        for i, ci in enumerate(c_array):
            _c = self.concentration.nfw_concentration(m_array_2[i], z_array_2[i], 'diemer19', '200c', None, False,
                                                       0.1, kwargs_suppresion, suppression_model)
            npt.assert_equal(_c, ci)

        kwargs_suppresion, suppression_model = {'a_mc': 0.5, 'b_mc': 0.8}, 'hyperbolic'
        cwdm = self.concentration.nfw_concentration(m, z, 'diemer19', '200c', 8., False, 0.1, kwargs_suppresion, suppression_model)
        fac = WDM_concentration_suppresion_factor(m, z, 8., suppression_model, kwargs_suppresion)
        npt.assert_equal(cwdm, c * fac)

        model1 = {'custom': True, 'log10c0': 1., 'beta': 0.9, 'zeta': -0.1}
        model2 = {'custom': True, 'c0': 10., 'beta': 0.9, 'zeta': -0.1}
        ccdm1 = self.concentration.nfw_concentration(m, z, model1, None, None, False, None, kwargs_suppresion, suppression_model)
        ccdm2 = self.concentration.nfw_concentration(m, z, model2, None, None, False, None, kwargs_suppresion, suppression_model)
        npt.assert_almost_equal(ccdm1, ccdm2)

        model = {'custom': True, 'c0': 2., 'beta': 0.9, 'zeta': -0.1}
        kwargs_suppresion, suppression_model = {'c_scale': 60., 'c_power': -0.26}, 'polynomial'
        ccdm = self.concentration.nfw_concentration(m, z, model, None, None, False, 0.1, kwargs_suppresion, suppression_model)
        cwdm = self.concentration.nfw_concentration(m, z, model, None, 8., False, 0.1, kwargs_suppresion, suppression_model)
        fac = WDM_concentration_suppresion_factor(m, z, 8.,suppression_model, kwargs_suppresion)
        npt.assert_equal(cwdm, ccdm * fac)

        ccdm = self.concentration.nfw_concentration(m_array_2, z_array_2, model, None, None, False, 0.1, kwargs_suppresion, suppression_model)
        npt.assert_equal(len(ccdm), len(m_array_2))

        kwargs_suppresion, suppression_model = {'c_scale': 60., 'c_power': 0.26}, 'polynomial'
        npt.assert_raises(Exception, WDM_concentration_suppresion_factor, 10**8, 0.1, 7., suppression_model, kwargs_suppresion)
        kwargs_suppresion, suppression_model = {'c_scale': -60., 'c_power': -0.26}, 'polynomial'
        npt.assert_raises(Exception, WDM_concentration_suppresion_factor, 10 ** 8, 0.1, 7., suppression_model,
                          kwargs_suppresion)
        kwargs_suppresion, suppression_model = {'a_mc': 1., 'b_mc': -0.26}, 'hyperbolic'
        npt.assert_raises(Exception, WDM_concentration_suppresion_factor, 10 ** 8, 0.1, 7., suppression_model,
                          kwargs_suppresion)

if __name__ == '__main__':
    pytest.main()

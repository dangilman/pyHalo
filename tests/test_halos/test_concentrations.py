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
        c = self.concentration.NFW_concentration(m, z, 'diemer19', '200c', None, False,
                                                 None, None, 0.1)
        npt.assert_array_less(0., c)

        m_array = np.logspace(6, 6, 10)
        z = 0.5
        c_array = self.concentration.NFW_concentration(m_array, z, 'diemer19', '200c', None, False,
                                                 None, None, 0.1)
        npt.assert_equal(True, len(c_array)==10)

        z_array = np.linspace(0.5, 0.5, 10)
        c_array = self.concentration.NFW_concentration(m, z_array, 'diemer19', '200c', None, False,
                                                       None, None, 0.1)
        npt.assert_equal(True, len(c_array) == 10)
        for ci in c_array:
            npt.assert_equal(ci, c)

        m_array_2, z_array_2 = np.logspace(6, 8, 10), np.linspace(0.2, 1., 10)
        c_array = self.concentration.NFW_concentration(m_array_2, z_array_2, 'diemer19', '200c', None, False,
                                                       None, None, 0.1)
        for i, ci in enumerate(c_array):
            _c = self.concentration.NFW_concentration(m_array_2[i], z_array_2[i], 'diemer19', '200c', None, False,
                                                       None, None, 0.1)
            npt.assert_equal(_c, ci)

        cwdm = self.concentration.NFW_concentration(m, z, 'diemer19', '200c', 8., False, 60., -0.26, 0.1)
        fac = WDM_concentration_suppresion_factor(m, z, 8., 60, -0.26)
        npt.assert_equal(cwdm, c * fac)

        model = {'custom': True, 'c0': 2., 'beta': 0.9, 'zeta': -0.1}
        ccdm = self.concentration.NFW_concentration(m, z, model, None, None, False, None, None, None)
        cwdm = self.concentration.NFW_concentration(m, z, model, None, 8., False, 60., -0.26, 0.1)
        fac = WDM_concentration_suppresion_factor(m, z, 8., 60, -0.26)
        npt.assert_equal(cwdm, ccdm * fac)

if __name__ == '__main__':
    pytest.main()

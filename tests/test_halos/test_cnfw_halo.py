import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo as LensCosmoLenstronomy
from pyHalo.Halos.HaloModels.NFW_core import CoreNFWHalo
from pyHalo.Halos.concentration import ConcentrationConstant
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import pytest
import numpy as np

class TestNFWHalos(object):

    def setup_method(self):

        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.zhalo = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, cosmo)
        self.truncation_class = None
        self.c = 8
        self.concentration_class = ConcentrationConstant(self.lens_cosmo, self.c)
        self.lclenstronomy = LensCosmoLenstronomy(self.zhalo, self.zsource, astropy)

    def test_lenstronomy_params(self):

        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        unique_tag = 1.0
        kwargs_profile = {'beta': 0.99999}
        is_subhalo = False
        cnfw_halo = CoreNFWHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                      self.truncation_class, self.concentration_class, unique_tag)
        lenstronomy_params = cnfw_halo.lenstronomy_params[0][0]
        npt.assert_almost_equal(lenstronomy_params['r_core'], lenstronomy_params['Rs'], 3)

        profile_args = cnfw_halo.profile_args
        npt.assert_almost_equal(profile_args[0], self.c, 7)
        npt.assert_almost_equal(profile_args[1], kwargs_profile['beta'], 7)

if __name__ == '__main__':
    pytest.main()


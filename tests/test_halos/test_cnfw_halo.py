import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo as LensCosmoLenstronomy
from pyHalo.Halos.HaloModels.NFW_core import CoreNFWHalo
from pyHalo.Halos.concentration import ConcentrationConstant
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo
import pytest
import numpy as np

class TestCNFWHalos(object):

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

    def test_mass3d(self):

        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        unique_tag = 1.0
        kwargs_profile = {'beta': 0.000001}
        is_subhalo = False
        zlens, zsource = 0.5, 2.0
        lens_cosmo = LensCosmo(zlens, zsource)
        cdm_halo = TNFWFieldHalo.simple_setup(m, 0.0, 0.0, zlens, tau=1000, lens_cosmo=lens_cosmo)
        cnfw_halo = CoreNFWHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                self.truncation_class, self.concentration_class, unique_tag)
        cnfw_halo._c = cdm_halo.c
        r = np.logspace(-0.5, 0.0, 2000) * cnfw_halo.nfw_params[1]
        rho_sidm = cnfw_halo.density_profile_3d(r)
        rho_cdm = cdm_halo.density_profile_3d(r)
        npt.assert_almost_equal(np.sum(rho_sidm / rho_cdm)/len(rho_sidm), 1, 1)


if __name__ == '__main__':
    pytest.main()


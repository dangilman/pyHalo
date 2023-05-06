import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo as LensCosmoLenstronomy
from pyHalo.Halos.HaloModels.NFW import NFWSubhhalo, NFWFieldHalo
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import pytest

class TestNFWHalos(object):

    def setup_method(self):

        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.zhalo = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, cosmo)
        self.truncation_class = None
        self.concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo, scatter=False)
        self.lclenstronomy = LensCosmoLenstronomy(self.zhalo, self.zsource, astropy)

    def test_lenstronomy_params(self):

        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        unique_tag = 1.0
        kwargs_profile = {'evaluate_mc_at_zlens': False, 'c_scatter': False, 'c_scatter_dex': 0.2}
        is_subhalo = False
        nfw_field_halo = NFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                      self.truncation_class, self.concentration_class, unique_tag)

        kwargs_halo, _ = nfw_field_halo.lenstronomy_params
        id = nfw_field_halo.lenstronomy_ID
        npt.assert_string_equal('NFW', id[0])

        rho0, Rs, c, r200, M200 = self.lclenstronomy.nfw_angle2physical(kwargs_halo[0]['Rs'],
                                                                        kwargs_halo[0]['alpha_Rs'])
        npt.assert_almost_equal(M200/m, 1.0, 3)

        is_subhalo = True
        nfw_subhalo = NFWSubhhalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                  self.truncation_class, self.concentration_class, unique_tag)
        c_subhalo = nfw_subhalo.c
        c_field = nfw_field_halo.c
        npt.assert_equal(True, c_subhalo <= c_field)

    def test_z_infall(self):
        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        unique_tag = 1.0
        kwargs_profile = {'evaluate_mc_at_zlens': False, 'c_scatter': False, 'c_scatter_dex': 0.2}
        is_subhalo = True
        nfw_subhalo = NFWSubhhalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                      self.truncation_class, self.concentration_class, unique_tag)
        z_infall = nfw_subhalo.z_infall
        npt.assert_equal(True, self.zhalo <= z_infall)

if __name__ == '__main__':
    pytest.main()


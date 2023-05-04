import numpy.testing as npt
import numpy as np
from lenstronomy.Cosmo.lens_cosmo import LensCosmo as LensCosmoLenstronomy
from pyHalo.Halos.HaloModels.PsuedoJaffe import PJaffeSubhalo, PJaffeFieldhalo
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from pyHalo.Halos.tidal_truncation import TruncationRoche
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import pytest
from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe as PJaffeLenstronomy

class TestPjaffeHalo(object):

    def setup_method(self):
        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.zhalo = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, cosmo)
        self.truncation_class = TruncationRoche(None, 100000000.0)
        self.concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo, scatter=False)
        self.lclenstronomy = LensCosmoLenstronomy(self.zhalo, self.zsource, astropy)

        m = 10 ** 8
        x = 0.0
        y = 0.0
        r3d = 100.0
        is_subhalo = True
        kwargs_profile = {'evaluate_mc_at_zlens': False}
        unique_tag = 1.0
        self.subhalo = PJaffeSubhalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                      self.truncation_class, self.concentration_class, unique_tag)
        is_subhalo = False
        unique_tag = 2.0
        self.field_halo = PJaffeFieldhalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                     self.truncation_class, self.concentration_class, unique_tag)

    def test_lenstronomy_ID(self):
        id = self.subhalo.lenstronomy_ID
        npt.assert_string_equal(id[0], 'PJAFFE')
        id = self.field_halo.lenstronomy_ID
        npt.assert_string_equal(id[0], 'PJAFFE')

    def test_z_infall(self):

        z_infall = self.subhalo.z_infall
        npt.assert_equal(True, self.zhalo <= z_infall)

    def test_total_mass(self):

        c = float(self.subhalo.profile_args)
        rhos, rs, r200 = self.lens_cosmo.NFW_params_physical(self.subhalo.mass, c, self.zhalo)
        fc = np.log(1 + c) - c / (1 + c)
        m_nfw = 4 * np.pi * rs ** 3 * rhos * fc

        lenstronomy_kwargs, _ = self.subhalo.lenstronomy_params
        sigma0, ra, rs = lenstronomy_kwargs[0]['sigma0'], lenstronomy_kwargs[0]['Ra'], lenstronomy_kwargs[0]['Rs']

        arcsec_to_kpc = self.lens_cosmo.cosmo.kpc_proper_per_asec(self.zhalo)
        ra *= arcsec_to_kpc ** -1
        rs *= arcsec_to_kpc ** -1
        rho = self.subhalo._rho(m_nfw, rs, ra, c*rs)

        pjaffe_lenstronomy = PJaffeLenstronomy()
        m3d = pjaffe_lenstronomy.mass_3d(c*rs, rho, ra, rs)
        npt.assert_almost_equal(np.log10(m3d), np.log10(m_nfw))

    def test_concentration(self):

        profile_args = self.subhalo.profile_args
        (c_subhalo) = profile_args
        profile_args = self.field_halo.profile_args
        (c_fieldhalo) = profile_args
        npt.assert_equal(c_subhalo/c_fieldhalo<1, True)

    def test_params_physical(self):

        params_physical = self.subhalo.params_physical
        npt.assert_equal(len(params_physical), 4)

if __name__ == '__main__':
    pytest.main()


import numpy.testing as npt
import numpy as np
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Halos.HaloModels.generalized_nfw import GeneralNFWFieldHalo, GeneralNFWSubhalo, GeneralNFWHaloFromMass
from pyHalo.Halos.HaloModels.NFW import NFWFieldHalo
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import PseudoDoublePowerlaw
from lenstronomy.LensModel.Profiles.nfw import NFW
import pytest
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Cosmology.cosmology import Cosmology

class TestGeneralNFW(object):

    def setup_method(self):

        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.zhalo = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, cosmo)
        self.nfw_profile_lenstronomy = NFW()
        self.gnfw_profile_lenstronomy = PseudoDoublePowerlaw()
        self.truncation_class = None
        self.concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo, scatter=False)

    def test_lenstronomy_params(self):

        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        is_subhalo = False
        gamma_inner = 1.0001
        gamma_outer = 3.0001
        x_match = 2.9
        unique_tag = 1.0
        kwargs_profile = {'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'x_match': x_match,
                          'evaluate_mc_at_zlens': True}
        gnfw = GeneralNFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)
        nfw = NFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)
        kwargs_nfw_profile = nfw.lenstronomy_params[0][0]
        kwargs_gnfw_profile = gnfw.lenstronomy_params[0][0]
        alpha_Rs = kwargs_gnfw_profile['alpha_Rs']
        npt.assert_almost_equal(alpha_Rs/kwargs_nfw_profile['alpha_Rs'], 1.0, 2)
        rs = kwargs_nfw_profile['Rs']
        npt.assert_almost_equal(rs/kwargs_gnfw_profile['Rs'], 1.0, 4)

        id = gnfw.lenstronomy_ID
        npt.assert_string_equal('PSEUDO_DPL', id[0])

    def test_enclosed_mass(self):

        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        is_subhalo = False
        gamma_inner = 2.2
        gamma_outer = 3.2
        x_match = 3.5
        unique_tag = 1.0
        kwargs_profile = {'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'x_match': x_match,
                          'evaluate_mc_at_zlens': True}
        gnfw = GeneralNFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)
        nfw = NFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)

        kwargs_gnfw = gnfw.lenstronomy_params[0][0]
        kwargs_nfw = nfw.lenstronomy_params[0][0]
        rmatch = x_match * kwargs_nfw['Rs']
        m3d_nfw = self.nfw_profile_lenstronomy.mass_3d_lens(rmatch, kwargs_nfw['Rs'],
                                                            kwargs_nfw['alpha_Rs'])
        m3d = self.gnfw_profile_lenstronomy.mass_3d_lens(rmatch, kwargs_gnfw['Rs'],
                                                         kwargs_gnfw['alpha_Rs'],
                                                         kwargs_gnfw['gamma_inner'], kwargs_gnfw['gamma_outer'])
        npt.assert_almost_equal(m3d_nfw/m3d, 1.0, 4)
        m = 10 ** 7.3
        x = 0.5
        y = 1.0
        r3d = 100
        is_subhalo = False
        gamma_inner = 1.5
        gamma_outer = 2.89
        x_match = 4.0
        unique_tag = 1.0
        kwargs_profile = {'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'x_match': x_match}
        gnfw = GeneralNFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)
        nfw = NFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                           self.truncation_class, self.concentration_class, unique_tag)
        kwargs_gnfw = gnfw.lenstronomy_params[0][0]
        kwargs_nfw = nfw.lenstronomy_params[0][0]
        rmatch = x_match * kwargs_nfw['Rs']
        m3d_nfw = self.nfw_profile_lenstronomy.mass_3d_lens(rmatch, kwargs_nfw['Rs'],
                                                            kwargs_nfw['alpha_Rs'])
        m3d = self.gnfw_profile_lenstronomy.mass_3d_lens(rmatch, kwargs_gnfw['Rs'],
                                                         kwargs_gnfw['alpha_Rs'],
                                                         kwargs_gnfw['gamma_inner'], kwargs_gnfw['gamma_outer'])
        npt.assert_almost_equal(m3d_nfw / m3d, 1.0, 4)

    def test_concentration_redshift_eval(self):
        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        gamma_inner = 1.25
        gamma_outer = 3.3
        x_match = 2.5
        unique_tag = 1.0
        kwargs_profile = {'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'x_match': x_match,
                          'evaluate_mc_at_zlens': False}
        is_subhalo = False
        gnfw = GeneralNFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)
        is_subhalo = True
        gnfw_subhalo = GeneralNFWSubhalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)
        c = gnfw.profile_args[0]
        c_sub = gnfw_subhalo.profile_args[0]
        npt.assert_equal(False, c==c_sub)

    def test_from_mass(self):

        m = 10 ** 8
        c = 10
        z = 0.5

        rhos, rs, r200 = self.lens_cosmo.NFW_params_physical(m, c, z)

        x = 0.5
        y = 1.0
        r3d = 100
        gamma_inner = 1.5
        gamma_outer = 3.01
        is_subhalo = False
        kwargs_profile = {'M': m, 'R': r200, 'r': rs, 'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer}
        unique_tag = 1.0
        gnfw = GeneralNFWHaloFromMass(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                 self.truncation_class, self.concentration_class, unique_tag)
        rkpc = np.linspace(0.01, c, 1000) * rs
        density_profile_gnfw = gnfw.density_profile_3d_lenstronomy(rkpc)
        m_numerical = np.trapz(4*np.pi*density_profile_gnfw * rkpc**2, rkpc)
        npt.assert_almost_equal(m_numerical / m, 1, decimal=3)

        kwargs_profile = {'M': m, 'R': r200, 'r': 2*rs, 'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer}
        unique_tag = 1.0
        gnfw = GeneralNFWHaloFromMass(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                      self.truncation_class, self.concentration_class, unique_tag)
        rkpc = np.linspace(0.01, c, 1000) * rs
        density_profile_gnfw = gnfw.density_profile_3d_lenstronomy(rkpc)
        m_numerical = np.trapz(4 * np.pi * density_profile_gnfw * rkpc ** 2, rkpc)
        npt.assert_almost_equal(m_numerical / m, 1, decimal=3)

        factor_nfw = (np.log(2) - 0.5) / (np.log(1+c) - c/(1+c))
        kwargs_profile = {'M': m * factor_nfw, 'R': rs, 'r': rs, 'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer}
        unique_tag = 1.0
        gnfw = GeneralNFWHaloFromMass(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                      self.truncation_class, self.concentration_class, unique_tag)
        rkpc = np.linspace(0.001, rs, 10000)
        density_profile_gnfw = gnfw.density_profile_3d_lenstronomy(rkpc)
        m_numerical = np.trapz(4 * np.pi * density_profile_gnfw * rkpc ** 2, rkpc)
        m_target = m * factor_nfw
        npt.assert_almost_equal(m_numerical/m_target, 1, decimal=3)


if __name__ == '__main__':
    pytest.main()

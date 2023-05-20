import numpy.testing as npt
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Halos.HaloModels.powerlaw import PowerLawSubhalo, PowerLawFieldHalo
from pyHalo.Halos.HaloModels.NFW import NFWFieldHalo
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from lenstronomy.LensModel.Profiles.splcore import SPLCORE
from lenstronomy.LensModel.Profiles.nfw import NFW
import pytest
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Cosmology.cosmology import Cosmology

class TestSPLCORE(object):

    def setup_method(self):

        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.zhalo = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, cosmo)
        self.nfw_profile_lenstronomy = NFW()
        self.splcore = SPLCORE()
        self.truncation_class = None
        self.concentration_class = ConcentrationDiemerJoyce(self.lens_cosmo, scatter=False)

    def test_lenstronomy_params(self):

        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        is_subhalo = False
        gamma = 2.7
        x_match = 2.9
        unique_tag = 1.0
        kwargs_profile = {'log_slope_halo': gamma, 'x_match': x_match, 'x_core_halo': 0.075,
                          'evaluate_mc_at_zlens': True}
        splcore = PowerLawFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)
        nfw = NFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)
        kwargs_nfw_profile = nfw.lenstronomy_params[0][0]
        kwargs_splcore_profile = splcore.lenstronomy_params[0][0]
        npt.assert_almost_equal(kwargs_splcore_profile['r_core'],
                                kwargs_nfw_profile['Rs'] * kwargs_profile['x_core_halo'], 5)
        id = splcore.lenstronomy_ID
        npt.assert_string_equal('SPL_CORE', id[0])

    def test_enclosed_mass(self):

        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        is_subhalo = False
        gamma = 2.8
        x_match = 3.5
        unique_tag = 1.0
        kwargs_profile = {'log_slope_halo': gamma, 'x_match': x_match, 'x_core_halo': 0.075,
                          'evaluate_mc_at_zlens': True}
        splcore = PowerLawFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                    self.truncation_class, self.concentration_class, unique_tag)
        nfw = NFWFieldHalo(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)

        kwargs_splcore = splcore.lenstronomy_params[0][0]
        kwargs_nfw = nfw.lenstronomy_params[0][0]
        rmatch = x_match * kwargs_nfw['Rs']
        m3d_nfw = self.nfw_profile_lenstronomy.mass_3d_lens(rmatch, kwargs_nfw['Rs'],
                                                            kwargs_nfw['alpha_Rs'])
        m3d = self.splcore.mass_3d_lens(rmatch, kwargs_splcore['sigma0'],
                                                         kwargs_splcore['r_core'], kwargs_splcore['gamma'])
        npt.assert_almost_equal(m3d_nfw/m3d, 1.0, 4)

    def test_concentration_redshift_eval(self):
        m = 10 ** 8
        x = 0.5
        y = 1.0
        r3d = 100
        gamma = 2.6
        x_match = 2.5
        unique_tag = 1.0
        kwargs_profile = {'log_slope_halo': gamma, 'x_match': x_match, 'x_core_halo': 0.075,
                          'evaluate_mc_at_zlens': False}
        splcore = PowerLawFieldHalo(m, x, y, r3d, self.zhalo, False, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)
        splcore_subhalo = PowerLawSubhalo(m, x, y, r3d, self.zhalo, True, self.lens_cosmo, kwargs_profile,
                                   self.truncation_class, self.concentration_class, unique_tag)
        c = splcore.profile_args[0]
        c_sub = splcore_subhalo.profile_args[0]
        npt.assert_equal(False, c==c_sub)

if __name__ == '__main__':
    pytest.main()

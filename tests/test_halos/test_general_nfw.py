import numpy.testing as npt
import numpy as np
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.defaults import set_default_kwargs
from pyHalo.Halos.HaloModels.generalized_nfw import GeneralNFWFieldHalo, GeneralNFWSubhalo
from pyHalo.Halos.HaloModels.NFW import NFWFieldHalo
from lenstronomy.LensModel.Profiles.general_nfw import GNFW
from lenstronomy.LensModel.Profiles.nfw import NFW
import pytest

class TestGeneralNFW(object):

    def setup(self):

        mass = 10 ** 8
        x = 0.0
        y = 0.0
        z = 0.5
        zsource = 2.0
        gamma_inner = 1.0
        gamma_outer = 3.0
        lens_cosmo = LensCosmo(z, zsource)
        args = set_default_kwargs({'log_mlow': 6.0, 'log_mhigh': 10., 'x_match': 3.0,
                                   'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer,'c_scatter':False}, zsource)
        self.args = args
        self.gnfw = GeneralNFWFieldHalo(mass, x, y, 100, 'GNFW', z, False, lens_cosmo, args, 1.0)
        self.gnfw_sub = GeneralNFWSubhalo(mass, x, y, 100, 'GNFW', z, False, lens_cosmo, args, 1.0)
        self.nfw = NFWFieldHalo(mass, x, y, 100, 'NFW', z, False, lens_cosmo, args, 2.0)
        self.c = self.gnfw.profile_args[0]
        self.nfw._c = self.c

        self.nfw_profile_lenstronomy = NFW()
        self.gnfw_profile_lenstronomy = GNFW()

    def test_lenstronomy_params(self):

        kwargs_nfw = self.nfw.lenstronomy_params[0][0]
        lenstronomy_kwargs = self.gnfw.lenstronomy_params[0][0]
        alpha_Rs = lenstronomy_kwargs['alpha_Rs']
        npt.assert_almost_equal(alpha_Rs/kwargs_nfw['alpha_Rs'], 1.0, 2)
        rs = lenstronomy_kwargs['Rs']
        npt.assert_almost_equal(rs/kwargs_nfw['Rs'],1.0,4)

    def test_enclosed_mass(self):
        # defines the normalization of the profile
        kwargs_gnfw = self.gnfw.lenstronomy_params[0][0]
        kwargs_nfw = self.nfw.lenstronomy_params[0][0]
        rmatch = self.args['x_match'] * kwargs_nfw['Rs']

        m3d_nfw = self.nfw_profile_lenstronomy.mass_3d_lens(rmatch, kwargs_nfw['Rs'],
                                                            kwargs_nfw['alpha_Rs'])
        m3d = self.gnfw_profile_lenstronomy.mass_3d_lens(rmatch, kwargs_gnfw['Rs'],
                                                         kwargs_gnfw['alpha_Rs'],
                                                         kwargs_gnfw['gamma_inner'], kwargs_gnfw['gamma_outer'])

        npt.assert_almost_equal(m3d_nfw/m3d, 1.0, 4)

    def test_lenstronomy_ID(self):

        id = self.gnfw.lenstronomy_ID
        npt.assert_string_equal('GNFW', id[0])

    def test_concentration_redshift_eval(self):
        c = self.gnfw.profile_args[0]
        c_sub = self.gnfw_sub.profile_args[0]
        npt.assert_equal(False, c==c_sub)

if __name__ == '__main__':
    pytest.main()

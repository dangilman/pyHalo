import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.PTMass import PTMass
from astropy.cosmology import FlatLambdaCDM
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import pytest

class TestPTMass(object):

    def setup_method(self):
        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy_instance=astropy)
        self.zhalo = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, cosmo)
        self.truncation_class = None
        self.concentration_class = None
        kwargs_profile = {}
        self.ptmass = PTMass(10**9, 1.0, 1.0, None, self.zhalo, False, self.lens_cosmo, kwargs_profile, None, None, 1.0)

    def test_lenstronomy_ID(self):

        id = self.ptmass.lenstronomy_ID
        npt.assert_string_equal(id[0], 'POINT_MASS')

    def test_profile_args(self):

        profile_args = self.ptmass.profile_args
        npt.assert_almost_equal(len(profile_args), 0)

    def test_thetaE(self):

        m = 10**8
        x = 0.0
        y = 0.0
        r3d = None
        is_subhalo = False
        kwargs_profile = {}
        unique_tag = 0.1
        ptmass = PTMass(m, x, y, r3d, self.zhalo, is_subhalo, self.lens_cosmo, kwargs_profile,
                                      self.truncation_class, self.concentration_class, unique_tag)
        kwargs, _ = ptmass.lenstronomy_params
        theta_E = kwargs[0]['theta_E']
        theta_E_theory = self.lens_cosmo.point_mass_factor_z(ptmass.z) * np.sqrt(10**8)
        npt.assert_almost_equal(theta_E_theory, theta_E)


if __name__ == '__main__':
    pytest.main()


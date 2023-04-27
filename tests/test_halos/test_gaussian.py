import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.gaussian import Gaussian
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from astropy.cosmology import FlatLambdaCDM
import pytest


class TestGaussianHalo(object):

    def setup_method(self):

        mass = 10 ** 8.
        x = 0.5
        y = 1.
        r3d = np.sqrt(1 + 0.5 ** 2 + 70 ** 2)
        astropy = FlatLambdaCDM(70.0, 0.3)
        cosmo = Cosmology(astropy)
        z = 0.5
        self.z = z
        lens_cosmo = LensCosmo(z, 2., cosmo)
        profile_args = {'amp':1,'sigma':1,'center_x':1.0,'center_y':1.0}
        sub_flag = False
        self.halo = Gaussian(mass, x, y, r3d, z,
                               sub_flag, lens_cosmo,
                               profile_args, None, None, unique_tag=np.random.rand())

    def test_lenstronomy_params(self):

        kwargs, _ = self.halo.lenstronomy_params
        for param in kwargs[0].keys():
            npt.assert_equal(kwargs[0][param], 1.0)

    def test_lenstronomy_ID(self):

        id = self.halo.lenstronomy_ID
        npt.assert_string_equal(id[0], 'GAUSSIAN_KAPPA')

    def test_redshift_eval(self):

        z_halo = self.halo.z_eval
        npt.assert_equal(z_halo, self.z)

if __name__ == '__main__':
   pytest.main()



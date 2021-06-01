import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.PTMass import PTMass
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from colossus.halo.concentration import concentration, peaks
import pytest

class TestPTMass(object):

    def setup(self):

        mass = 10 ** 8.
        x = 0.5
        y = 1.
        z = 0.9
        r3d = np.sqrt(1 + 0.5 ** 2 + 70 ** 2)
        self.r3d = r3d
        mdef = 'TNFW'
        self.z = z
        sub_flag = True

        self.H0 = 70
        self.omega_baryon = 0.03
        self.omega_DM = 0.25
        self.sigma8 = 0.82
        curvature = 'flat'
        self.ns = 0.9608
        cosmo_params = {'H0': self.H0, 'Om0': self.omega_baryon + self.omega_DM, 'Ob0': self.omega_baryon,
                        'sigma8': self.sigma8, 'ns': self.ns, 'curvature': curvature}

        cosmo = Cosmology(cosmo_kwargs=cosmo_params)
        self.lens_cosmo = LensCosmo(self.z, 2., cosmo)


        self.ptmass = PTMass(mass, x, y, r3d, mdef, z, sub_flag, self.lens_cosmo, args={},
                             unique_tag=np.random.rand())

    def test_lenstronomy_ID(self):

        id = self.ptmass.lenstronomy_ID
        npt.assert_string_equal(id[0], 'POINT_MASS')

    def test_profile_args(self):

        profile_args = self.ptmass.profile_args

        npt.assert_almost_equal(len(profile_args), 0)

if __name__ == '__main__':
    pytest.main()


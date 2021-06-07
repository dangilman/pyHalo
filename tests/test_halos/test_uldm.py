import numpy as np
import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.ULDM import ULDMFieldHalo, ULDMSubhalo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
import pytest

class TestTNFWHalos(object):

    def setup(self):

        mass = 10**8.
        x = 0.5
        y = 1.
        r3d = np.sqrt(1 + 0.5 ** 2 + 70**2)
        self.r3d = r3d
        mdef = 'TNFW'
        self.z = 0.25
        sub_flag = True

        self.H0 = 70
        self.omega_baryon = 0.03
        self.omega_DM = 0.25
        self.sigma8 = 0.82
        curvature = 'flat'
        self.ns = 0.9608
        cosmo_params = {'H0': self.H0, 'Om0': self.omega_baryon + self.omega_DM, 'Ob0': self.omega_baryon,
                        'sigma8': self.sigma8, 'ns': self.ns, 'curvature': curvature}
        self._dm, self._bar = self.omega_DM, self.omega_baryon
        cosmo = Cosmology(cosmo_kwargs=cosmo_params)
        self.lens_cosmo = LensCosmo(self.z, 2., cosmo)

        profile_args = {'RocheNorm': 1.2, 'RocheNu': 2/3,
                        'evaluate_mc_at_zlens': True,
                        'log_mc': None, 'c_scale': 60.,
                        'c_power': -0.17, 'c_scatter': False,
                        'mc_model': 'diemer19', 'LOS_truncation_factor': 40,
                        'c_scatter_dex': 0.1, 'mc_mdef': '200c'}

        self.subhalo = ULDMSubhalo(mass, x, y, r3d, mdef, self.z,
                                   sub_flag, self.lens_cosmo,
                                   profile_args, unique_tag=np.random.rand())
        self.fieldhalo = ULDMFieldHalo(mass, x, y, r3d, mdef, self.z,
                                       sub_flag, self.lens_cosmo,
                                       profile_args, unique_tag=np.random.rand())

    def test_lenstronomy_ID(self):

        ID = self.fieldhalo.lenstronomy_ID
        npt.assert_string_equal(ID[0], 'TNFW')
        npt.assert_string_equal(ID[1], 'ULDM')
        ID = self.subhalo.lenstronomy_ID
        npt.assert_string_equal(ID[0], 'TNFW')
        npt.assert_string_equal(ID[1], 'ULDM')

    def test_redshift_eval(self):

        z_subhalo = self.subhalo.z_eval
        z_field = self.fieldhalo.z_eval
        npt.assert_equal(z_field, self.z)
        # because the concentration is evaluated at infall, and z_infall > z
        npt.assert_equal(True, z_subhalo > z_field)

    def test_profile_normalization(self):

        """
        Test that the mass enclosed within r200 (or whatever radius) of the composite profile is what it should be
        """
        


import numpy.testing as npt
from pyHalo.Halos.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Profiles.splcore import SPLCORE
from pyHalo.Halos.HaloModels.powerlaw import GlobularCluster
import pytest
import numpy as np


class TestSPLCORE(object):

    def setup_method(self):

        self.zhalo = 0.5
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, None)
        self.splcore = SPLCORE()

    def test_lenstronomy_ID(self):

        mass = 10 ** 5
        args = {'gamma': 2.5,
                'r_core_fraction': 0.05,
                'gc_size_lightyear': 100}
        profile = GlobularCluster(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                  args, 1)
        lenstronomy_ID = profile.lenstronomy_ID
        npt.assert_string_equal(lenstronomy_ID[0], 'SPL_CORE')

    def test_lenstronomy_args(self):

        logM = 5.0
        mass = 10 ** logM
        args = {'gamma': 2.5,
                'r_core_fraction': 0.05,
                'gc_size_lightyear': 100}
        profile = GlobularCluster(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                  args, 1)
        lenstronomy_args, _ = profile.lenstronomy_params

    def test_mass(self):

        logM = 5.0
        mass = 10 ** logM
        args = {'gamma': 2.5,
                'r_core_fraction': 0.05,
                'gc_size_lightyear': 100}
        profile = GlobularCluster(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                  args, 1)
        profile_args = profile.profile_args
        rho0 = profile_args['rho0']
        r_core = profile_args['r_core_kpc']
        r_max = profile_args['gc_size']
        gamma = profile_args['gamma']
        total_mass = profile._prof.mass_3d(r_max, rho0, r_core, gamma)
        npt.assert_almost_equal(total_mass, mass)

        kpc_per_arcsec = profile.lens_cosmo.cosmo.kpc_proper_per_asec(profile.z)
        lenstronomy_args = profile.lenstronomy_params[0]
        sigma0 = lenstronomy_args[0]['sigma0']
        rcore = lenstronomy_args[0]['r_core']
        gamma = lenstronomy_args[0]['gamma']
        r_max = profile_args['gc_size'] / kpc_per_arcsec
        sigma_crit_mpc = profile.lens_cosmo.get_sigma_crit_lensing(profile.z, profile.lens_cosmo.z_source)
        sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
        total_mass = profile._prof.mass_3d(r_max, sigma0/rcore, rcore, gamma) * sigma_crit_arcsec
        npt.assert_almost_equal(total_mass, mass)

if __name__ == '__main__':
    pytest.main()

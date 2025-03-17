import numpy.testing as npt
from pyHalo.Halos.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Profiles.splcore import SPLCORE
from pyHalo.Halos.HaloModels.powerlaw import GlobularCluster
import pytest
import numpy as np


class TestGlobularClusters(object):

    def setup_method(self):

        self.zhalo = 0.4
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, None)
        self.splcore = SPLCORE()

    def test_lenstronomy_ID(self):

        mass = 10 ** 5
        args = {'gamma': 2.5,
                'gc_size_lightyear': 100,
                'gc_concentration': 50}
        profile = GlobularCluster(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                  args, 1)
        lenstronomy_ID = profile.lenstronomy_ID
        npt.assert_string_equal(lenstronomy_ID[0], 'SPL_CORE')

    def test_lenstronomy_args(self):

        logM = 5.0
        mass = 10 ** logM
        args = {'gamma': 2.5,
                'gc_size_lightyear': 100,
                'gc_concentration': 50}
        profile = GlobularCluster(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                  args, 1)
        lenstronomy_args, _ = profile.lenstronomy_params

    def test_mass(self):

        logM = 6.0
        mass = 10 ** logM
        args = {'gamma': 3.2,
                'gc_size_lightyear': 100,
                'gc_concentration': 15.0}
        profile = GlobularCluster(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                  args, 1)
        profile_args = profile.profile_args
        rho0 = profile_args[0]
        r_core = profile_args[3]
        r_max = profile_args[1]
        gamma = profile_args[2]
        total_mass = profile._prof.mass_3d(r_max, rho0, r_core, gamma)
        npt.assert_almost_equal(total_mass, mass)

        kpc_per_arcsec = profile.lens_cosmo.cosmo.kpc_proper_per_asec(profile.z)
        lenstronomy_args = profile.lenstronomy_params[0]
        sigma0 = lenstronomy_args[0]['sigma0']
        rcore = lenstronomy_args[0]['r_core']
        gamma = lenstronomy_args[0]['gamma']
        r_max = profile_args[1] / kpc_per_arcsec
        sigma_crit_mpc = profile.lens_cosmo.get_sigma_crit_lensing(profile.z, profile.lens_cosmo.z_source)
        sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
        total_mass = profile._prof.mass_3d(r_max, sigma0/rcore, rcore, gamma) * sigma_crit_arcsec
        npt.assert_almost_equal(total_mass / mass, 1, 1)

        # test using the lenstronomy density method
        kpc_per_lightyear = 0.000306
        r = np.linspace(0.00001, 1.0, 100000) * args['gc_size_lightyear'] * kpc_per_lightyear
        rho = profile.density_profile_3d_lenstronomy(r)
        m = np.trapz(4 * np.pi * r ** 2 * rho, r)
        npt.assert_almost_equal(m / 10**logM, 1, 4)

        # the total mass will be slightly larger
        r = np.linspace(0.00001, 10.0, 2000000) * args['gc_size_lightyear'] * kpc_per_lightyear
        rho = profile.density_profile_3d_lenstronomy(r)
        m_total = np.trapz(4 * np.pi * r ** 2 * rho, r)
        npt.assert_almost_equal(m_total / 10**logM, 1.5918, 4)

        # test the average mass, rather than central density
        r = np.linspace(0.00001, 1.0, 100000) * args['gc_size_lightyear'] * kpc_per_lightyear
        rho = profile.density_profile_3d_lenstronomy(r)
        m_total = np.trapz(4 * np.pi * r ** 2 * rho, r)
        volume = 4*np.pi/3 * (args['gc_size_lightyear'])**3
        average_density = m_total / volume
        npt.assert_almost_equal(average_density, 0.2387, 4)


if __name__ == '__main__':
    pytest.main()

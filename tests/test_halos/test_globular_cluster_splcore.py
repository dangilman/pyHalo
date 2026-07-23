import numpy.testing as npt
from pyHalo.Halos.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Profiles.splcore import SPLCORE
from pyHalo.Halos.HaloModels.globular_cluster import GlobularCluster
from scipy.optimize import brentq
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
        args = {'gamma': 6.0,
                'gc_size_pc': 100,
                'gc_concentration': 15}
        profile = GlobularCluster(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                  args, 1)
        lenstronomy_ID = profile.lenstronomy_ID
        npt.assert_string_equal(lenstronomy_ID[0], 'SPL_CORE')

    def test_lenstronomy_args(self):

        logM = 5.0
        mass = 10 ** logM
        args = {'gamma': 6.0,
                'gc_size_pc': 100,
                'gc_concentration': 15}
        profile = GlobularCluster(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                  args, 1)
        lenstronomy_args, _ = profile.lenstronomy_params
        kw = lenstronomy_args[0]
        npt.assert_equal(set(kw.keys()), {'sigma0', 'gamma', 'r_core', 'center_x', 'center_y'})
        # r_core = gc_size (pc -> kpc -> arcsec) / concentration
        kpc_per_arcsec = profile.lens_cosmo.cosmo.kpc_proper_per_asec(profile.z)
        gc_size_arcsec = (args['gc_size_pc'] * 1e-3) / kpc_per_arcsec
        npt.assert_almost_equal(kw['r_core'], gc_size_arcsec / args['gc_concentration'])
        npt.assert_equal(kw['gamma'], args['gamma'])

    def test_mass(self):

        logM = 6.0
        mass = 10 ** logM
        args = {'gamma': 6.0,
                'gc_size_pc': 100,
                'gc_concentration': 15.0}
        profile = GlobularCluster(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                  args, 1)
        # profile_args = (rho0, gc_size_kpc, gamma, r_core_kpc)
        profile_args = profile.profile_args
        rho0 = profile_args[0]
        r_max = profile_args[1]
        gamma = profile_args[2]
        r_core = profile_args[3]

        # normalization: mass within gc_size == input mass
        total_mass = profile._prof.mass_3d(r_max, rho0, r_core, gamma)
        npt.assert_almost_equal(total_mass / mass, 1, 5)

        # same recovered from the lensing (sigma0) parameterization
        kpc_per_arcsec = profile.lens_cosmo.cosmo.kpc_proper_per_asec(profile.z)
        lenstronomy_args = profile.lenstronomy_params[0]
        sigma0 = lenstronomy_args[0]['sigma0']
        rcore = lenstronomy_args[0]['r_core']
        gamma = lenstronomy_args[0]['gamma']
        r_max_arcsec = profile_args[1] / kpc_per_arcsec
        sigma_crit_mpc = profile.lens_cosmo.get_sigma_crit_lensing(profile.z, profile.lens_cosmo.z_source)
        sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
        total_mass = profile._prof.mass_3d(r_max_arcsec, sigma0 / rcore, rcore, gamma) * sigma_crit_arcsec
        npt.assert_almost_equal(total_mass / mass, 1, 2)

        # lenstronomy density method: mass within gc_size recovers the input mass
        r = np.linspace(0.00001, 1.0, 100000) * args['gc_size_pc'] * 1e-3
        rho = profile.density_profile_3d_lenstronomy(r)
        m = np.trapezoid(4 * np.pi * r ** 2 * rho, r)
        npt.assert_almost_equal(m / mass, 1.0, 3)

        # with a steep slope (gamma=6) almost no mass lies beyond gc_size, so integrating
        # to 10 x gc_size adds only ~0.05%
        r = np.linspace(0.00001, 10.0, 2000000) * args['gc_size_pc'] * 1e-3
        rho = profile.density_profile_3d_lenstronomy(r)
        m_10 = np.trapezoid(4 * np.pi * r ** 2 * rho, r)
        npt.assert_almost_equal(m_10 / mass, 1.0005, 3)

    def test_half_mass_radius(self):

        mass = 10 ** 6
        args = {'gamma': 6.0,
                'gc_size_pc': 100,
                'gc_concentration': 15.0}
        profile = GlobularCluster(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                  args, 1)
        rho0, gc_size_kpc, gamma, r_core_kpc = profile.profile_args
        rh = brentq(lambda r: profile._prof.mass_3d(r, rho0, r_core_kpc, gamma) - 0.5 * mass,
                    1e-4 * gc_size_kpc, gc_size_kpc) * 1000
        npt.assert_almost_equal(profile.half_mass_radius / rh, 1, 3)

    def test_mass_linearity(self):

        args = {'gamma': 6.0, 'gc_size_pc': 100, 'gc_concentration': 15.0}
        p1 = GlobularCluster(10 ** 6, 0.0, 0.0, self.zhalo, self.lens_cosmo, args, 1)
        p2 = GlobularCluster(2 * 10 ** 6, 0.0, 0.0, self.zhalo, self.lens_cosmo, args, 2)
        s1 = p1.lenstronomy_params[0][0]['sigma0']
        s2 = p2.lenstronomy_params[0][0]['sigma0']
        npt.assert_almost_equal(s2 / s1, 2.0)


if __name__ == '__main__':
    pytest.main()

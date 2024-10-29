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
        profile_lenstronomy = SPLCORE()
        profile_args = profile.profile_args
        gc_size = profile_args['gc_size']
        rho0 = profile_args['rho0']
        r_core = profile_args['r_core_arcsec']
        gamma = profile_args['gamma']
        mass = profile_lenstronomy.mass_3d(gc_size, rho0, r_core, gamma)
        sigma_crit_mpc = self.lens_cosmo.get_sigma_crit_lensing(profile.z, self.lens_cosmo.z_source)
        kpc_per_arcsec = self.lens_cosmo.cosmo.kpc_proper_per_asec(profile.z)
        sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
        npt.assert_almost_equal(logM, np.log10(mass * sigma_crit_arcsec))

if __name__ == '__main__':
    pytest.main()

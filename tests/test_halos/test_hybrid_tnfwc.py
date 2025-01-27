import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.NFW_core_trunc import Hybrid, HybridSubhalo, TNFWFieldHalo, TNFWSubhalo, \
    TNFWCFieldHaloSIDM, TNFWCSubhaloSIDM
from pyHalo.truncation_models import ConstantTruncationArcsec
from pyHalo.concentration_models import ConcentrationDiemerJoyce
from pyHalo.Halos.lens_cosmo import LensCosmo
import matplotlib.pyplot as plt
import pytest

class TestTNFWC(object):

    def setup_method(self):

        mass = 10 ** 8
        x = 0.5
        y = 1.0
        z = 0.5
        r3d = 100
        unique_tag = 1.0
        lens_cosmo = LensCosmo(0.5, 2.5)
        kwargs_cdm_profile = {}
        truncation_class = ConstantTruncationArcsec(lens_cosmo, 1000.0)
        concentration_class = ConcentrationDiemerJoyce(lens_cosmo.cosmo, scatter=False)
        kwargs_sidm_profile = {'sidm_timescale': 12.0, 'lambda_t': 1.0}
        self._rescaling_factor = 0.7

        self.tnfw_halo = TNFWFieldHalo(mass, x, y, r3d, z,
                                       False, lens_cosmo, kwargs_cdm_profile,
                                       truncation_class, concentration_class, unique_tag)
        self.tnfwc_halo = TNFWCFieldHaloSIDM(mass, x, y, r3d, z,
                                             False, lens_cosmo, kwargs_sidm_profile,
                                             truncation_class, concentration_class, unique_tag)

        self.tnfw_subhalo = TNFWSubhalo(mass, x, y, r3d, z,
                                       True, lens_cosmo, kwargs_cdm_profile,
                                       truncation_class, concentration_class, unique_tag)
        self.tnfwc_subhalo = TNFWCSubhaloSIDM(mass, x, y, r3d, z,
                                             True, lens_cosmo, kwargs_sidm_profile,
                                             truncation_class, concentration_class, unique_tag)

        self.hybrid_fieldhalo = Hybrid(self.tnfw_halo, self.tnfwc_halo, self._rescaling_factor)
        self.hybrid_subhalo = HybridSubhalo(self.tnfw_subhalo, self.tnfwc_subhalo, self._rescaling_factor)

    def test_joint_parameters(self):

        npt.assert_equal(self.hybrid_fieldhalo.tnfwc_halo.c, self.hybrid_fieldhalo.tnfw_halo.c)
        npt.assert_equal(self.hybrid_subhalo.tnfwc_halo.c, self.hybrid_subhalo.tnfw_halo.c)
        npt.assert_equal(self.hybrid_subhalo.tnfwc_halo.z_infall, self.hybrid_subhalo.tnfw_halo.z_infall)
        rtrunc_tnfw = self.hybrid_fieldhalo.lenstronomy_params[0][0]['r_trunc']
        rtrunc_tnfwc = self.hybrid_fieldhalo.lenstronomy_params[0][1]['r_trunc']
        npt.assert_almost_equal(rtrunc_tnfwc / rtrunc_tnfw, 1, 4)

        rtrunc_tnfw = self.hybrid_subhalo.lenstronomy_params[0][0]['r_trunc']
        rtrunc_tnfwc = self.hybrid_subhalo.lenstronomy_params[0][1]['r_trunc']
        npt.assert_almost_equal(rtrunc_tnfwc / rtrunc_tnfw, 1, 4)

        npt.assert_almost_equal(self.hybrid_subhalo.tnfwc_halo.halo_effective_age,
                                self.hybrid_subhalo.tnfw_halo.halo_age)
        npt.assert_almost_equal(self.hybrid_fieldhalo.tnfwc_halo.halo_effective_age,
                                self.hybrid_fieldhalo.tnfw_halo.halo_age)

    def test_lenstronomy_ID(self):

        ID = self.hybrid_fieldhalo.lenstronomy_ID
        npt.assert_string_equal(ID[0],'TNFW')
        npt.assert_string_equal(ID[1], 'TNFWC')

    def test_lenstronomy_args(self):

        kwargs_lenstronomy = self.hybrid_fieldhalo.lenstronomy_params[0]
        npt.assert_equal(len(kwargs_lenstronomy),2)

    def test_profile_3d(self):

        rs = self.hybrid_fieldhalo.nfw_params[1][1]
        r = np.logspace(-1.5, 1.5, 1000) * rs
        profile_3d_tnfw = self.hybrid_fieldhalo.tnfw_halo.density_profile_3d(r)
        profile_3d_tnfwc = self.hybrid_fieldhalo.tnfwc_halo.density_profile_3d(r)
        plt.loglog(r, profile_3d_tnfw, color='k')
        plt.loglog(r, profile_3d_tnfwc, color='r')
        combined_true = profile_3d_tnfw * (1 - self._rescaling_factor) + profile_3d_tnfwc * self._rescaling_factor
        combined = self.hybrid_fieldhalo.density_profile_3d(r)
        npt.assert_equal(combined, combined_true)

        rs = self.hybrid_subhalo.nfw_params[1][1]
        r = np.logspace(-1.5, 1.5, 1000) * rs
        profile_3d_tnfw = self.hybrid_subhalo.tnfw_halo.density_profile_3d(r)
        profile_3d_tnfwc = self.hybrid_subhalo.tnfwc_halo.density_profile_3d(r)
        plt.loglog(r, profile_3d_tnfw, color='k')
        plt.loglog(r, profile_3d_tnfwc, color='r')
        combined_true = profile_3d_tnfw * (1 - self._rescaling_factor) + profile_3d_tnfwc * self._rescaling_factor
        combined = self.hybrid_subhalo.density_profile_3d(r)
        npt.assert_equal(combined, combined_true)


if __name__ == '__main__':
    pytest.main()

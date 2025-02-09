import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.NFW_core_trunc import Hybrid, TNFWCHalo
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo
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
        self._rescaling_factor = 0.7

        self.tnfw_halo = TNFWFieldHalo(mass, x, y, r3d, z,
                                       False, lens_cosmo, kwargs_cdm_profile,
                                       truncation_class, concentration_class, unique_tag)
        kwargs_profile = {'sidm_timescale': 20.,
                          'lambda_t': 1.0,
                          'mass_conservation': self.tnfw_halo.mass}
        self.tnfwc_halo = TNFWCHalo(mass, x, y, r3d, z,
                                             False, lens_cosmo, kwargs_profile,
                                             truncation_class, concentration_class, unique_tag)

        self.hybrid = Hybrid(self.tnfw_halo, self.tnfwc_halo, self._rescaling_factor)

    def test_lenstronomy_ID(self):

        ID = self.hybrid.lenstronomy_ID
        npt.assert_string_equal(ID[0],'TNFW')
        npt.assert_string_equal(ID[1], 'TNFWC')

    def test_lenstronomy_args(self):

        kwargs_lenstronomy = self.hybrid.lenstronomy_params[0]
        npt.assert_equal(len(kwargs_lenstronomy),2)

    def test_profile_3d(self):

        rs = self.hybrid.nfw_params[1][1]
        r = np.logspace(-1.5, 1.5, 1000) * rs
        profile_3d_tnfw = self.hybrid.tnfw_halo.density_profile_3d_lenstronomy(r)
        profile_3d_tnfwc = self.hybrid.tnfwc_halo.density_profile_3d_lenstronomy(r)
        plt.loglog(r, profile_3d_tnfw, color='k')
        plt.loglog(r, profile_3d_tnfwc, color='r')
        combined_true = profile_3d_tnfw * (1 - self._rescaling_factor) + profile_3d_tnfwc * self._rescaling_factor
        combined = self.hybrid.density_profile_3d_lenstronomy(r)
        npt.assert_equal(combined, combined_true)

if __name__ == '__main__':
    pytest.main()

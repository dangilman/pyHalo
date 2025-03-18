import numpy.testing as npt
import numpy as np
from lenstronomy.LensModel.Profiles.nfw_core_truncated import TNFWC as TNFWCLenstronomy
from lenstronomy.LensModel.Profiles.nfw import NFW as NFWLenstronomy
from pyHalo.Halos.HaloModels.NFW_core_trunc import TNFWCHalo
from pyHalo.truncation_models import ConstantTruncationArcsec
from pyHalo.concentration_models import ConcentrationConstant
from pyHalo.Halos.lens_cosmo import LensCosmo
import pytest


class TestTNFWC(object):

    def setup_method(self):

        self.lens_cosmo = LensCosmo(0.5, 2.5)
        self.tnfwc_lenstronomy = TNFWCLenstronomy()
        self.nfw_lenstronomy = NFWLenstronomy()
        self.truncation_class = ConstantTruncationArcsec(self.lens_cosmo, 1000.0)
        self.concentration_class = ConcentrationConstant(None, 10.0)

    def test_mass_conservation(self):

        mass0 = 10 ** 8
        kwargs_profile = {'sidm_timescale': 20.,
                          'lambda_t': 1.0,
                          'mass_conservation': mass0}
        tnfwc = TNFWCHalo(mass0, 0.0, 0.0, None, 0.5, False,
                               self.lens_cosmo, kwargs_profile, self.truncation_class,
                               self.concentration_class, 1.0)
        r = np.linspace(0.001, tnfwc.c, 10000) * tnfwc.nfw_params[1]
        mass = np.trapz(4*np.pi*r**2*tnfwc.density_profile_3d_lenstronomy(r),r)

        npt.assert_almost_equal(mass / mass0, 1, 2)

        kwargs_profile = {'sidm_timescale': 10.,
                          'lambda_t': 1.0,
                          'mass_conservation': mass0}
        tnfwc = TNFWCHalo(mass0, 0.0, 0.0, None, 0.5, False,
                          self.lens_cosmo, kwargs_profile, self.truncation_class,
                          self.concentration_class, 1.0)
        mass = np.trapz(4*np.pi*r**2*tnfwc.density_profile_3d_lenstronomy(r),r)
        npt.assert_almost_equal(mass / mass0, 1, 2)

        kwargs_profile = {'sidm_timescale': 2.5,
                          'lambda_t': 1.0,
                          'mass_conservation': mass0}
        tnfwc = TNFWCHalo(mass0, 0.0, 0.0, None, 0.5, False,
                          self.lens_cosmo, kwargs_profile, self.truncation_class,
                          self.concentration_class, 1.0)
        mass = np.trapz(4*np.pi*r**2*tnfwc.density_profile_3d_lenstronomy(r),r)
        npt.assert_array_less(abs(mass / mass0 - 1), 0.016)

        kwargs_profile = {'sidm_timescale': 0.5,
                          'lambda_t': 1.0,
                          'mass_conservation': mass0}
        tnfwc = TNFWCHalo(mass0, 0.0, 0.0, None, 0.5, False,
                          self.lens_cosmo, kwargs_profile, self.truncation_class,
                          self.concentration_class, 1.0)
        mass = np.trapz(4*np.pi*r**2*tnfwc.density_profile_3d_lenstronomy(r),r)
        npt.assert_array_less(abs(mass / mass0 - 1), 0.016)

    def test_vmax(self):
        """Here we test that these objects have the vmax of the reference TNFW/NFW halo profile"""

        mass0 = 10 ** 8
        kwargs_profile = {'sidm_timescale': 0.5,
                          'lambda_t': 1.0,
                          'mass_conservation': mass0}
        tnfwc = TNFWCHalo(mass0, 0.0, 0.0, None, 0.5, False,
                          self.lens_cosmo, kwargs_profile, self.truncation_class,
                          self.concentration_class, 1.0)
        vmax = tnfwc.vmax_nfw
        rhos, rs, _ = tnfwc.nfw_params
        vmax_calc = self.lens_cosmo.nfw_vmax(rhos, rs)
        npt.assert_almost_equal(vmax / vmax_calc, 1, 2)

if __name__ == '__main__':
    pytest.main()

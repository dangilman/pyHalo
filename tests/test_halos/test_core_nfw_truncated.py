import numpy.testing as npt
import numpy as np
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from lenstronomy.LensModel.Profiles.nfw_core_truncated import TNFWC as TNFWCLenstronomy
from pyHalo.Halos.HaloModels.TNFW import TNFWSubhalo
from lenstronomy.LensModel.Profiles.nfw import NFW as NFWLenstronomy
from pyHalo.Halos.HaloModels.NFW_core_trunc import TNFWCHaloEvolving, TNFWCHaloParametric
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
                          'mass_conservation': mass0,
                          'rt_kpc': 1000}
        tnfwc = TNFWCHaloEvolving(mass0, 0.0, 0.0, None, 0.5, False,
                               self.lens_cosmo, kwargs_profile, self.truncation_class,
                               self.concentration_class, 1.0)
        r = np.linspace(0.001, tnfwc.c, 10000) * tnfwc.nfw_params[1]
        mass = np.trapz(4*np.pi*r**2*tnfwc.density_profile_3d_lenstronomy(r),r)

        npt.assert_almost_equal(mass / mass0, 1, 2)

        kwargs_profile = {'sidm_timescale': 10.,
                          'lambda_t': 1.0,
                          'mass_conservation': mass0,
                          'rt_kpc': 1000}
        tnfwc = TNFWCHaloEvolving(mass0, 0.0, 0.0, None, 0.5, False,
                          self.lens_cosmo, kwargs_profile, self.truncation_class,
                          self.concentration_class, 1.0)
        mass = np.trapz(4*np.pi*r**2*tnfwc.density_profile_3d_lenstronomy(r),r)
        npt.assert_almost_equal(mass / mass0, 1, 2)

        kwargs_profile = {'sidm_timescale': 2.5,
                          'lambda_t': 1.0,
                          'mass_conservation': mass0,
                          'rt_kpc': 1000}
        tnfwc = TNFWCHaloEvolving(mass0, 0.0, 0.0, None, 0.5, False,
                          self.lens_cosmo, kwargs_profile, self.truncation_class,
                          self.concentration_class, 1.0)
        mass = np.trapz(4*np.pi*r**2*tnfwc.density_profile_3d_lenstronomy(r),r)
        npt.assert_array_less(abs(mass / mass0 - 1), 0.016)

        kwargs_profile = {'sidm_timescale': 0.5,
                          'lambda_t': 1.0,
                          'mass_conservation': mass0,
                          'rt_kpc': 1000}
        tnfwc = TNFWCHaloEvolving(mass0, 0.0, 0.0, None, 0.5, False,
                          self.lens_cosmo, kwargs_profile, self.truncation_class,
                          self.concentration_class, 1.0)
        mass = np.trapz(4*np.pi*r**2*tnfwc.density_profile_3d_lenstronomy(r),r)
        npt.assert_array_less(abs(mass / mass0 - 1), 0.016)

    def test_vmax(self):
        """Here we test that these objects have the vmax of the reference TNFW/NFW halo profile"""

        mass0 = 10 ** 8
        kwargs_profile = {'sidm_timescale': 0.5,
                          'lambda_t': 1.0,
                          'mass_conservation': mass0,
                          'rt_kpc': 1000}
        tnfwc = TNFWCHaloEvolving(mass0, 0.0, 0.0, None, 0.5, False,
                          self.lens_cosmo, kwargs_profile, self.truncation_class,
                          self.concentration_class, 1.0)
        vmax = tnfwc.vmax_nfw
        rhos, rs, _ = tnfwc.nfw_params
        vmax_calc = self.lens_cosmo.nfw_vmax(rhos, rs)
        npt.assert_almost_equal(vmax / vmax_calc, 1, 2)

    def test_from_tnfw_subhalo(self):

        x = 0.5
        y = 1.0
        z = 0.45
        z_infall = 1.0
        mass = 10 ** 9
        f_bound = 0.4
        tnfw_subhalo = TNFWSubhalo.simple_setup(mass,
                                                f_bound,
                                                z_infall,
                                                x, y, z,
                                                self.lens_cosmo)
        _, rt_kpc = tnfw_subhalo.profile_args

        x_core_halo = 0.05
        mass_conservation = tnfw_subhalo.mass_3d('r200')
        kwargs_profile = {'x_core_halo': x_core_halo,
                          'mass_conservation': mass_conservation,
                          'rt_kpc': rt_kpc}
        concentration_class = ConcentrationConstant(None, tnfw_subhalo.c)
        truncation_class = None
        sidm_halo = TNFWCHaloParametric(tnfw_subhalo.mass,
                                        tnfw_subhalo.x,
                                        tnfw_subhalo.y,
                                        tnfw_subhalo.r3d,
                                        tnfw_subhalo.z,
                                        tnfw_subhalo.is_subhalo,
                                        tnfw_subhalo.lens_cosmo,
                                        kwargs_profile,
                                        truncation_class,
                                        concentration_class,
                                        tnfw_subhalo.unique_tag)

        rhos, rs, r200 = tnfw_subhalo.nfw_params
        r = np.logspace(-4, np.log10(tnfw_subhalo.c), 1000) * rs
        density_profile_sidm = sidm_halo.density_profile_3d_lenstronomy(r)
        mass3d_tnfw = tnfw_subhalo.mass_3d('r200')
        mass3d_sidm = np.trapz(4 * np.pi * r ** 2 * density_profile_sidm, r)
        npt.assert_array_less(abs(mass3d_sidm / mass3d_tnfw - 1), 5)


if __name__ == '__main__':
    pytest.main()

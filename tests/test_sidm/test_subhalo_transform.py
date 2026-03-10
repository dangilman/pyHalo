import numpy.testing as npt
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Halos.HaloModels.TNFW import TNFWSubhalo
import numpy as np
from pyHalo.single_realization import Realization
from pyHalo.realization_extensions import RealizationExtensions
import pytest

class TestSubhaloTransform(object):

    def setup_method(self):

        self.lens_cosmo = LensCosmo(0.5, 2.5)

    def test_transform_heavy_stripping(self):

        x = 0.5
        y = 1.0
        z = 0.45
        z_infall = 4.0
        mass = 1*10 ** 7
        f_bound = 0.001
        tnfw_subhalo = TNFWSubhalo.simple_setup(mass,
                                                f_bound,
                                                z_infall,
                                                x,
                                                y,
                                                z,
                                                self.lens_cosmo)
        realization = Realization.from_halos([tnfw_subhalo],
                                             self.lens_cosmo,
                                             kwargs_halo_model=None,
                                             msheet_correction=None,
                                             rendering_classes=None)
        ext = RealizationExtensions(realization)
        indexes = [0]
        x_core_halo = 0.05
        sidm_subhalo = ext.add_core_collapsed_halos(indexes,
                                           t_c=None,
                                           x_core_halo=x_core_halo).halos[0]

        #print(tnfw_subhalo.rescale_norm, tnfw_subhalo.profile_args[1]/tnfw_subhalo.nfw_params[1])
        mass_tnfw_subhalo = tnfw_subhalo.mass_3d('r200')
        mass_sidm_subhalo = sidm_subhalo.mass_3d('r200')
        npt.assert_almost_equal(mass_tnfw_subhalo / mass_sidm_subhalo, 1, 2)
        rhos, rs, r200 = tnfw_subhalo.nfw_params
        r = np.linspace(0.001 * r200, r200, 1000)
        density_nfw = tnfw_subhalo.density_profile_3d_lenstronomy(r)
        density_sidm = sidm_subhalo.density_profile_3d_lenstronomy(r)
        mass_analytic_tnfw = np.trapz(4 * np.pi * r ** 2 * density_nfw, r)
        mass_analytic_sidm = np.trapz(4 * np.pi * r ** 2 * density_sidm, r)
        npt.assert_almost_equal(mass_analytic_tnfw / mass_analytic_sidm, 1, 2)

    def test_transform_light_stripping(self):

        x = 0.5
        y = 1.0
        z = 0.45
        z_infall = 1.0
        mass = 1*10 ** 7.7
        f_bound = 0.9
        tnfw_subhalo = TNFWSubhalo.simple_setup(mass,
                                                f_bound,
                                                z_infall,
                                                x,
                                                y,
                                                z,
                                                self.lens_cosmo)
        realization = Realization.from_halos([tnfw_subhalo],
                                             self.lens_cosmo,
                                             kwargs_halo_model=None,
                                             msheet_correction=None,
                                             rendering_classes=None)
        ext = RealizationExtensions(realization)
        indexes = [0]
        x_core_halo = 0.05
        sidm_subhalo = ext.add_core_collapsed_halos(indexes,
                                           t_c=None,
                                           x_core_halo=x_core_halo).halos[0]

        #print(tnfw_subhalo.rescale_norm, tnfw_subhalo.profile_args[1]/tnfw_subhalo.nfw_params[1])
        mass_tnfw_subhalo = tnfw_subhalo.mass_3d('r200')
        mass_sidm_subhalo = sidm_subhalo.mass_3d('r200')
        npt.assert_almost_equal(mass_tnfw_subhalo / mass_sidm_subhalo, 1, 2)
        rhos, rs, r200 = tnfw_subhalo.nfw_params
        r = np.linspace(0.001 * r200, r200, 1000)
        density_nfw = tnfw_subhalo.density_profile_3d_lenstronomy(r)
        density_sidm = sidm_subhalo.density_profile_3d_lenstronomy(r)
        mass_analytic_tnfw = np.trapz(4 * np.pi * r ** 2 * density_nfw, r)
        mass_analytic_sidm = np.trapz(4 * np.pi * r ** 2 * density_sidm, r)
        npt.assert_almost_equal(mass_analytic_tnfw / mass_analytic_sidm, 1, 2)

if __name__ == '__main__':
    pytest.main()

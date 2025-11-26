import numpy.testing as npt
import numpy as np
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo, TNFWSubhalo
from pyHalo.Halos.HaloModels.core_collapsed_halo import CoreCollapsedHalo
from pyHalo.truncation_models import ConstantTruncationArcsec
from pyHalo.concentration_models import ConcentrationConstant
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.single_realization import Realization
from pyHalo.realization_extensions import RealizationExtensions
import pytest


class TestCoreCollapsedHalo(object):

    def setup_method(self):

        self.lens_cosmo = LensCosmo(0.5, 2.5)
        self.truncation_class_field_halo = ConstantTruncationArcsec(self.lens_cosmo, 1000.0)

    def test_field_halo_highc(self):
        mass = 10 ** 8
        halo_concentration = 25.0
        x = 0.0
        y = 0.0
        r3d = None
        z = 0.5
        sub_flag = False
        lens_cosmo = LensCosmo(z, 2.0)
        unique_tag = 1.0
        tau = 2000
        tnfw_halo = TNFWFieldHalo.simple_setup(mass, x, y, z, tau, lens_cosmo,
                                               concentration_model='DIEMERJOYCE19')
        tnfw_halo._c = halo_concentration
        _, rs, r200 = tnfw_halo.nfw_params
        rt_kpc = tnfw_halo.profile_args[1]
        gamma_match = -2.0
        r_match_kpc = tnfw_halo.log_derivative_inverse(gamma_match, rs, rt_kpc)
        mass_R = tnfw_halo.mass_3d(r_match_kpc)
        Rs_inner_kpc = r_match_kpc
        gamma_inner = 2.5
        gamma_outer = 6.0

        truncation_class = None
        concentration_class = ConcentrationConstant(None, tnfw_halo.c)

        scale_match_m = 1.6
        m_target_R = scale_match_m * mass_R
        args = {'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'rt_kpc': rt_kpc,
                'm_target_r200': mass, 'm_target_R': m_target_R, 'Rs_inner_kpc': Rs_inner_kpc,
                'r_match_kpc': r_match_kpc}
        halo = CoreCollapsedHalo(mass, x, y, r3d, z,
                                 sub_flag, lens_cosmo, args, truncation_class, concentration_class, unique_tag)

        r = np.logspace(-3.5, np.log10(tnfw_halo.c), 2000) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        total_density = density_inner + density_outer
        density_nfw = tnfw_halo.density_profile_3d(r)
        mass_cc_r200 = np.trapz(4 * np.pi * r ** 2 * total_density, r)
        mass_nfw_r200 = np.trapz(4 * np.pi * r ** 2 * density_nfw, r)
        npt.assert_almost_equal(mass_nfw_r200 / mass, 1,2)
        npt.assert_almost_equal(mass_cc_r200 / mass, 1, 2)

        r = np.logspace(-4, np.log10(r_match_kpc/rs), 2500) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        total_density = density_inner + density_outer
        density_nfw = tnfw_halo.density_profile_3d(r)
        mass_cc_rmatch = np.trapz(4 * np.pi * r ** 2 * total_density, r)
        mass_nfw_rmatch = scale_match_m * np.trapz(4 * np.pi * r ** 2 * density_nfw, r)
        npt.assert_almost_equal(mass_cc_rmatch / mass_nfw_rmatch, 1, 2)

    def test_field_halo_lowc(self):
        mass = 10 ** 8
        halo_concentration = 3.0
        x = 0.0
        y = 0.0
        r3d = None
        z = 0.5
        sub_flag = False
        lens_cosmo = LensCosmo(z, 2.0)
        unique_tag = 1.0
        tau = 2000
        tnfw_halo = TNFWFieldHalo.simple_setup(mass, x, y, z, tau, lens_cosmo,
                                               concentration_model='DIEMERJOYCE19')
        tnfw_halo._c = halo_concentration
        _, rs, r200 = tnfw_halo.nfw_params
        rt_kpc = tnfw_halo.profile_args[1]
        gamma_match = -2.0
        r_match_kpc = tnfw_halo.log_derivative_inverse(gamma_match, rs, rt_kpc)
        mass_R = tnfw_halo.mass_3d(r_match_kpc)
        Rs_inner_kpc = r_match_kpc
        gamma_inner = 2.5
        gamma_outer = 6.0

        truncation_class = None
        concentration_class = ConcentrationConstant(None, tnfw_halo.c)

        scale_match_m = 1.6
        m_target_R = scale_match_m * mass_R
        args = {'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'rt_kpc': rt_kpc,
                'm_target_r200': mass, 'm_target_R': m_target_R, 'Rs_inner_kpc': Rs_inner_kpc,
                'r_match_kpc': r_match_kpc}
        halo = CoreCollapsedHalo(mass, x, y, r3d, z,
                                 sub_flag, lens_cosmo, args, truncation_class, concentration_class, unique_tag)

        r = np.logspace(-3.5, np.log10(tnfw_halo.c), 2000) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        total_density = density_inner + density_outer
        density_nfw = tnfw_halo.density_profile_3d(r)
        mass_cc_r200 = np.trapz(4 * np.pi * r ** 2 * total_density, r)
        mass_nfw_r200 = np.trapz(4 * np.pi * r ** 2 * density_nfw, r)
        npt.assert_almost_equal(mass_nfw_r200 / mass, 1,2)
        npt.assert_almost_equal(mass_cc_r200 / mass, 1, 2)

        r = np.logspace(-4, np.log10(r_match_kpc/rs), 2500) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        total_density = density_inner + density_outer
        density_nfw = tnfw_halo.density_profile_3d(r)
        mass_cc_rmatch = np.trapz(4 * np.pi * r ** 2 * total_density, r)
        mass_nfw_rmatch = scale_match_m * np.trapz(4 * np.pi * r ** 2 * density_nfw, r)
        npt.assert_almost_equal(mass_cc_rmatch / mass_nfw_rmatch, 1, 2)

    def test_subhalo_low_fbound(self):
        mass = 10 ** 8
        x = 0.0
        y = 0.0
        r3d = None
        z = 0.5
        sub_flag = False
        lens_cosmo = LensCosmo(z, 2.0)
        unique_tag = 1.0
        f_bound = 0.001
        z_infall = 2.0
        tnfw_subhalo = TNFWSubhalo.simple_setup(mass, f_bound, z_infall, x, y, z, lens_cosmo)
        _, rs, r200 = tnfw_subhalo.nfw_params
        rt_kpc = tnfw_subhalo.profile_args[1]
        r_match_kpc = tnfw_subhalo.log_derivative_inverse(-2.0, rs, rt_kpc)
        mass_R = tnfw_subhalo.mass_3d(r_match_kpc)
        m_target_r200 = tnfw_subhalo.mass_3d(r200)
        Rs_inner_kpc = r_match_kpc
        gamma_inner = 2.4
        gamma_outer = 6.0
        scale_match_mass = 2.0
        args = {'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'rt_kpc': rt_kpc,
                'm_target_r200': m_target_r200, 'm_target_R': scale_match_mass * mass_R, 'r_match_kpc': r_match_kpc,
                'Rs_inner_kpc': Rs_inner_kpc}

        truncation_class = None
        concentration_class = ConcentrationConstant(None, tnfw_subhalo.c)

        halo = CoreCollapsedHalo(mass, x, y, r3d, z,
                                 sub_flag, lens_cosmo, args, truncation_class, concentration_class, unique_tag)

        r = np.logspace(-4, np.log10(tnfw_subhalo.c), 100) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        total_density = density_inner + density_outer
        mass_numerical = np.trapz(4*np.pi*r**2*total_density, r)
        npt.assert_array_less(abs(-1+m_target_r200 / mass_numerical), 0.03)

        r = np.logspace(-4, np.log10(r_match_kpc/rs), 2000) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        total_density = density_inner + density_outer
        mass_numerical = np.trapz(4 * np.pi * r ** 2 * total_density, r)
        npt.assert_array_less(abs(-1+scale_match_mass * mass_R / mass_numerical), 0.03)

    def test_subhalo_high_fbound(self):

        mass = 10 ** 8
        x = 0.0
        y = 0.0
        r3d = None
        z = 0.5
        sub_flag = False
        lens_cosmo = LensCosmo(z, 2.0)
        unique_tag = 1.0
        f_bound = 0.6
        z_infall = 2.0
        tnfw_subhalo = TNFWSubhalo.simple_setup(mass, f_bound, z_infall, x, y, z, lens_cosmo)
        _, rs, r200 = tnfw_subhalo.nfw_params
        rt_kpc = tnfw_subhalo.profile_args[1]
        r_match_kpc = tnfw_subhalo.log_derivative_inverse(-2.0, rs, rt_kpc)
        mass_R = tnfw_subhalo.mass_3d(r_match_kpc)
        m_target_r200 = tnfw_subhalo.mass_3d(r200)
        Rs_inner_kpc = r_match_kpc
        gamma_inner = 2.4
        gamma_outer = 6.0
        scale_match_mass = 2.
        m_target_R = scale_match_mass * mass_R
        args = {'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'rt_kpc': rt_kpc,
                'm_target_r200': m_target_r200, 'm_target_R': m_target_R, 'r_match_kpc': r_match_kpc,
                'Rs_inner_kpc': Rs_inner_kpc}

        truncation_class = None
        concentration_class = ConcentrationConstant(None, tnfw_subhalo.c)

        halo = CoreCollapsedHalo(mass, x, y, r3d, z,
                                 sub_flag, lens_cosmo, args, truncation_class, concentration_class, unique_tag)

        r = np.logspace(-3.5, np.log10(tnfw_subhalo.c), 1000) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        m_num = np.trapz(4 * np.pi * r ** 2 * (density_inner + density_outer), r)
        npt.assert_array_less(abs(-1+m_num / m_target_r200), 0.07)

        r = np.logspace(-3.5, np.log10(r_match_kpc / rs), 100) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        m_num = np.trapz(4 * np.pi * r ** 2 * (density_inner + density_outer), r)
        npt.assert_array_less(abs(-1+m_num / m_target_R), 0.03)

    def test_subhalo_low_fbound(self):

        mass = 10 ** 8
        x = 0.0
        y = 0.0
        r3d = None
        z = 0.5
        sub_flag = False
        lens_cosmo = LensCosmo(z, 2.0)
        unique_tag = 1.0
        f_bound = 0.005
        z_infall = 2.0
        tnfw_subhalo = TNFWSubhalo.simple_setup(mass, f_bound, z_infall, x, y, z, lens_cosmo)
        _, rs, r200 = tnfw_subhalo.nfw_params
        rt_kpc = tnfw_subhalo.profile_args[1]
        r_match_kpc = tnfw_subhalo.log_derivative_inverse(-2.0, rs, rt_kpc)
        mass_R = tnfw_subhalo.mass_3d(r_match_kpc)
        m_target_r200 = tnfw_subhalo.mass_3d(r200)
        Rs_inner_kpc = r_match_kpc
        gamma_inner = 2.4
        gamma_outer = 6.0
        scale_match_mass = 2.
        m_target_R = scale_match_mass * mass_R
        args = {'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'rt_kpc': rt_kpc,
                'm_target_r200': m_target_r200, 'm_target_R': m_target_R, 'r_match_kpc': r_match_kpc,
                'Rs_inner_kpc': Rs_inner_kpc}

        truncation_class = None
        concentration_class = ConcentrationConstant(None, tnfw_subhalo.c)

        halo = CoreCollapsedHalo(mass, x, y, r3d, z,
                                 sub_flag, lens_cosmo, args, truncation_class, concentration_class, unique_tag)
        penalty1, penalty2 = halo.profile_matching
        npt.assert_equal(penalty1,None)
        npt.assert_equal(penalty2, None)
        r = np.logspace(-3.5, np.log10(tnfw_subhalo.c), 1000) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        m_num = np.trapz(4 * np.pi * r ** 2 * (density_inner + density_outer), r)
        npt.assert_array_less(abs(-1+m_num / m_target_r200), 0.03)

        r = np.logspace(-3.5, np.log10(r_match_kpc / rs), 1000) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        m_num = np.trapz(4 * np.pi * r ** 2 * (density_inner + density_outer), r)
        npt.assert_array_less(abs(-1+m_num / m_target_R), 0.04)

        penalty1, penalty2 = halo.profile_matching
        npt.assert_array_less(penalty1, 0.05)
        npt.assert_array_less(penalty2, 0.05)

    def test_through_realization_extensions(self):

        mass = 10 ** 9
        x = 0.0
        y = 0.0
        z = 0.5
        lens_cosmo = LensCosmo(z, 2.0)
        f_bound = 0.6
        z_infall = 2.0
        kwargs_halo_model = None
        msheet_correction = None
        rendering_classes = None
        rendering_center_x = 0.0
        rendering_center_y = 0.0
        geometry = None
        tnfw_subhalo = TNFWSubhalo.simple_setup(mass, f_bound, z_infall, x, y, z, lens_cosmo)
        _, rs, r200 = tnfw_subhalo.nfw_params
        rt_kpc = tnfw_subhalo.profile_args[1]
        log_slope_match = -2.1
        r_match_kpc = tnfw_subhalo.log_derivative_inverse(log_slope_match, rs, rt_kpc)
        mass_R = tnfw_subhalo.mass_3d(r_match_kpc)
        m_target_r200 = tnfw_subhalo.mass_3d(r200)
        scale_match_mass = 2.5
        gamma_inner = 2.5
        m_target_R = scale_match_mass * mass_R
        halo_realization = Realization.from_halos([tnfw_subhalo], lens_cosmo, kwargs_halo_model,
                                                  msheet_correction, rendering_classes, rendering_center_x,
                                                  rendering_center_y, geometry)
        ext = RealizationExtensions(halo_realization)
        kwargs_halo = {'gamma_inner': gamma_inner, 'scale_match_r': scale_match_mass, 'log_slope_match': log_slope_match}
        realization_sidm = ext.add_core_collapsed_halos([0], halo_profile='CC_COMPOSITE', **kwargs_halo)
        halo = realization_sidm.halos[0]

        r = np.logspace(-3.5, np.log10(tnfw_subhalo.c), 1000) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        m_num = np.trapz(4 * np.pi * r ** 2 * (density_inner + density_outer), r)
        npt.assert_array_less(abs(-1 + m_num / m_target_r200), 0.03)

        r = np.logspace(-3.5, np.log10(r_match_kpc / rs), 1000) * rs
        density_inner, density_outer = halo.component_density_profile_3d(r)
        m_num = np.trapz(4 * np.pi * r ** 2 * (density_inner + density_outer), r)
        npt.assert_array_less(abs(-1 + m_num / m_target_R), 0.04)

if __name__ == '__main__':
    pytest.main()

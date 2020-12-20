from pyHalo.Rendering.Main.mainlens import MainLensPowerLaw
from pyHalo.Rendering.Main.SHMF_normalizations import normalization_sigmasub
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Rendering.render import render_main
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, integrate_power_law_analytic

import numpy as np
import numpy.testing as npt
import pytest

class TestFieldPowerLaw(object):

    def setup(self):

        self.delta_power_law_index = -0.17
        kwargs = {'power_law_index': -1.95, 'log_mlow': 6., 'log_mhigh': 10., 'log_mc': None, 'sigma_sub': 0.2,
                         'a_wdm': None, 'b_wdm': None, 'c_wdm': None, 'log_mass_sheet_min': 6., 'log_mass_sheet_max': 10.,
                        'subhalo_mass_sheet_scale': 1., 'draw_poisson': False,
                         'subhalo_convergence_correction_profile': 'UNIFORM', 'host_m200': 10**13.1, 'r_tidal': '0.25Rs',
                        'subhalo_spatial_distribution': 'HOST_NFW','cone_opening_angle': 6.,
                         'delta_power_law_index': -0.2, 'm_pivot': 10**8}

        kwargs_nfw_kappa_sheet = {'power_law_index': -1.95, 'log_mlow': 6., 'log_mhigh': 10., 'log_mc': None, 'sigma_sub': 0.2,
                         'a_wdm': None, 'b_wdm': None, 'c_wdm': None, 'log_mass_sheet_min': 6., 'log_mass_sheet_max': 10.,
                        'subhalo_mass_sheet_scale': 1., 'draw_poisson': False,
                         'subhalo_convergence_correction_profile': 'NFW', 'host_m200': 10**13.1, 'r_tidal': '0.25Rs',
                        'subhalo_spatial_distribution': 'HOST_NFW','cone_opening_angle': 6.,
                         'delta_power_law_index': -0.2, 'm_pivot': 10**8}

        self.kwargs = kwargs
        self.kwargs_nfw_kappa_sheet = kwargs_nfw_kappa_sheet

        kwargs_wdm = {'power_law_index': -1.95, 'log_mlow': 6., 'log_mhigh': 10., 'log_mc': None, 'sigma_sub': 0.2,
                         'a_wdm': 1., 'b_wdm': 1., 'c_wdm': -1.2, 'log_mass_sheet_min': 6., 'log_mass_sheet_max': 10.,
                        'subhalo_mass_sheet_scale': 1., 'draw_poisson': False,
                         'subhalo_convergence_correction_profile': 'UNIFORM', 'host_m200': 10**13.1, 'r_tidal': '0.25Rs',
                        'subhalo_spatial_distribution': 'HOST_NFW','cone_opening_angle': 6.,
                         'delta_power_law_index': -0.2, 'm_pivot': 10**8}

        self.kwargs_wdm = kwargs_wdm

        self.opening_angle= 6.
        cosmo = Cosmology()

        geometry_class = Geometry(cosmo, 0.5, 1.5, 6., 'DOUBLE_CONE')
        self.geometry = geometry_class
        self.cosmo = geometry_class._cosmo
        self.lens_cosmo = LensCosmo(0.5, 1.5, self.cosmo)
        self.arcsec = self.cosmo.arcsec

        self.func = MainLensPowerLaw(self.kwargs, self.geometry)
        self.func_nfw = MainLensPowerLaw(self.kwargs_nfw_kappa_sheet, self.geometry)
        self.func_wdm = MainLensPowerLaw(self.kwargs_wdm, self.geometry)

    def test_render(self):

        plaw_index = self.kwargs['power_law_index'] + self.kwargs['delta_power_law_index']
        kpc_per_arcsec_zlens = self.geometry.kpc_per_arcsec(0.5)
        norm = normalization_sigmasub(self.kwargs['sigma_sub'], self.kwargs['host_m200'], 0.5,
                                      kpc_per_arcsec_zlens, self.geometry.cone_opening_angle, plaw_index, 10 ** 8)
        nhalos_expected = integrate_power_law_analytic(norm, 10**self.kwargs['log_mlow'], 10**self.kwargs['log_mhigh'],
                                                       n=0, plaw_index=plaw_index)

        m, x, y, r3, redshifts = self.func()
        npt.assert_almost_equal(nhalos_expected/len(x), 1, 3)
        r2d_max = max(np.hypot(x, y))
        rmax_arcsec = self.geometry.cone_opening_angle / 2
        npt.assert_array_less(r2d_max, rmax_arcsec)

        m, x, y, r3, redshifts = self.func_wdm()
        plaw_index = self.kwargs_wdm['power_law_index'] + self.kwargs_wdm['delta_power_law_index']
        kpc_per_arcsec_zlens = self.geometry.kpc_per_arcsec(0.5)
        norm = normalization_sigmasub(self.kwargs_wdm['sigma_sub'], self.kwargs_wdm['host_m200'], 0.5,
                                      kpc_per_arcsec_zlens, self.geometry.cone_opening_angle, plaw_index, 10 ** 8)

        nhalos_expected = integrate_power_law_quad(norm, 10 ** self.kwargs_wdm['log_mlow'],
                                                       10 ** self.kwargs_wdm['log_mhigh'],
                                                   self.kwargs_wdm['log_mc'],
                                                   0, plaw_index, self.kwargs_wdm['a_wdm'],
                                                   self.kwargs_wdm['b_wdm'], self.kwargs_wdm['c_wdm'])
        r2d_max = max(np.hypot(x, y))
        rmax_arcsec = self.geometry.cone_opening_angle / 2
        npt.assert_array_less(r2d_max, rmax_arcsec)
        npt.assert_almost_equal(nhalos_expected / len(x), 1, 1)

    def test_render_function(self):

        m, x, y, r3, redshifts = render_main(self.func)
        plawindex = self.kwargs['power_law_index'] + self.delta_power_law_index
        mtotal = np.sum(m)
        kpc_per_arcsec_zlens = self.geometry.kpc_per_arcsec(0.5)
        norm = normalization_sigmasub(self.kwargs['sigma_sub'], self.kwargs['host_m200'], 0.5,
                                      kpc_per_arcsec_zlens, self.geometry.cone_opening_angle, plawindex, 10 ** 8)

        mtheory = integrate_power_law_analytic(norm, 10**self.kwargs['log_mlow'], 10**self.kwargs['log_mhigh'],
                                                       n=1, plaw_index=plawindex)
        npt.assert_array_less(abs(mtheory/mtotal - 1), 0.2)


    def test_convergence_sheets(self):

        kwargs_out, profile_name_out, redshifts_out = self.func.negative_kappa_sheets_theory()
        total_kappa = 0
        for i in range(0, len(kwargs_out)):
            total_kappa += kwargs_out[i]['kappa_ext']
        mass_rendered = np.sum(self.func()[0])
        area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, 0.5)
        kappa_rendered = mass_rendered / self.lens_cosmo.sigma_crit_mass(0.5, area)
        npt.assert_almost_equal(-kappa_rendered / total_kappa, 1, 1)

        kwargs_out, profile_name_out, redshifts_out = self.func_wdm.negative_kappa_sheets_theory()
        total_kappa = 0
        for i in range(0, len(kwargs_out)):
            total_kappa += kwargs_out[i]['kappa_ext']
        mass_rendered = np.sum(self.func_wdm()[0])
        area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, 0.5)
        kappa_rendered = mass_rendered / self.lens_cosmo.sigma_crit_mass(0.5, area)
        npt.assert_almost_equal(-kappa_rendered / total_kappa, 1, 0)

        kwargs_out, profile_name_out, redshifts_out = self.func_nfw.negative_kappa_sheets_theory()
        alpha_rs = kwargs_out[0]['alpha_Rs']
        rs = kwargs_out[0]['Rs']
        rcore = kwargs_out[0]['r_core'] * self.geometry.kpc_per_arcsec(0.5)
        npt.assert_array_less(alpha_rs, 0.)
        rs_kpc = rs * self.geometry.kpc_per_arcsec(0.5)
        npt.assert_almost_equal(rcore/rs_kpc, 0.25)


if __name__ == '__main__':

    pytest.main()

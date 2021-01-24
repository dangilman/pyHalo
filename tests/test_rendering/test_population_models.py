from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Rendering.halo_population import HaloPopulation
import numpy as np
import numpy.testing as npt
import pytest
from copy import deepcopy


class TestPopulationModel(object):

    def setup(self):

        zlens, zsource = 0.5, 2.
        zmin = 0.01
        zmax = 1.98
        log_mlow = 6.
        log_mhigh = 9.
        host_m200 = 10**13
        LOS_normalization = 1.
        draw_poisson = False
        log_mass_sheet_min = 7.
        log_mass_sheet_max = 10.
        kappa_scale = 1.
        delta_power_law_index = -0.17
        delta_power_law_index_coupling = 0.5
        cone_opening_angle = 6.
        m_pivot = 10 ** 8
        sigma_sub = 0.1
        power_law_index = -1.9
        subhalo_spatial_distribution = 'HOST_NFW'

        kwargs_cdm = {'zmin': zmin,
                      'zmax': zmax,
                      'log_mc': None,
                      'log_mlow': log_mlow,
                      'sigma_sub': sigma_sub,
                      'c_scale': None, 'c_power': None,
                      'a_wdm': None, 'b_wdm': None, 'c_wdm': None,
                      'c_scatter_dex': 0.2,
                      'log_mhigh': log_mhigh,
                      'host_m200': host_m200,
                      'LOS_normalization': LOS_normalization,
                      'draw_poisson': draw_poisson,
                      'subhalo_spatial_distribution': subhalo_spatial_distribution,
                      'log_mass_sheet_min': log_mass_sheet_min, 'log_mass_sheet_max': log_mass_sheet_max,
                      'kappa_scale': kappa_scale,
                      'power_law_index': power_law_index,
                      'delta_power_law_index': delta_power_law_index,
                      'delta_power_law_index_coupling': delta_power_law_index_coupling,
                       'm_pivot': m_pivot,
                      'cone_opening_angle': cone_opening_angle,
                      'subhalo_mass_sheet_scale': 1.,
                      'subhalo_convergence_correction_profile': 'NFW',
                      'r_tidal': '0.5Rs'}

        kwargs_wdm = deepcopy(kwargs_cdm)

        log_mc = 7.3
        a_wdm = 0.8
        b_wdm = 1.1
        c_wdm = -1.2
        kwargs_wdm['log_mc'] = log_mc
        kwargs_wdm['a_wdm'] = a_wdm
        kwargs_wdm['b_wdm'] = b_wdm
        kwargs_wdm['c_wdm'] = c_wdm
        kwargs_wdm['c_scale'] = 60.
        kwargs_wdm['c_power'] = -0.17

        lens_plane_redshifts = np.append(np.arange(0.01, 0.5, 0.02), np.arange(0.5, 1.5, 0.02))
        delta_zs = []
        for i in range(0, len(lens_plane_redshifts) - 1):
            delta_zs.append(lens_plane_redshifts[i + 1] - lens_plane_redshifts[i])
        delta_zs.append(1.5 - lens_plane_redshifts[-1])

        cosmo = Cosmology()
        self.lens_plane_redshifts = lens_plane_redshifts
        self.delta_zs = delta_zs
        self.halo_mass_function = LensingMassFunction(cosmo, 0.5, 1.5, kwargs_cdm['log_mlow'], kwargs_cdm['log_mhigh'],
                                                      6., m_pivot=kwargs_cdm['m_pivot'],
                                                      geometry_type='DOUBLE_CONE')
        self.geometry = Geometry(cosmo, 0.5, 1.5, 6., 'DOUBLE_CONE')
        self.lens_cosmo = LensCosmo(zlens, zsource, cosmo)

        self.kwargs_cdm = kwargs_cdm
        self.kwargs_wdm = kwargs_wdm

    def test_line_of_sight(self):

        for i, kwargs in enumerate([self.kwargs_cdm, self.kwargs_wdm]):

            population_model = HaloPopulation(['LINE_OF_SIGHT'], kwargs, self.lens_cosmo, self.geometry,
                                              self.halo_mass_function, self.lens_plane_redshifts,
                                              self.delta_zs)

            m, x, y, r3, redshifts, sub_flag = population_model.render()
            prof, zprof, kw = population_model.convergence_sheet_correction()

            for flag in sub_flag:
                npt.assert_equal(True, flag is False)

            npt.assert_equal(len(m)==len(x), True)
            npt.assert_equal(len(x) == len(y), True)
            npt.assert_equal(len(y) == len(r3), True)
            npt.assert_equal(len(r3) == len(redshifts), True)
            npt.assert_equal(len(redshifts) == len(sub_flag), True)
            npt.assert_equal(len(prof) == len(zprof), True)
            npt.assert_equal(len(zprof) == len(kw), True)

    def test_subhalos(self):

        for i, kwargs in enumerate([self.kwargs_cdm, self.kwargs_wdm]):

            population_model = HaloPopulation(['SUBHALOS'], kwargs, self.lens_cosmo, self.geometry,
                                              self.halo_mass_function, self.lens_plane_redshifts,
                                              self.delta_zs)

            m, x, y, r3, redshifts, sub_flag = population_model.render()
            prof, zprof, kw = population_model.convergence_sheet_correction()

            for flag in sub_flag:
                npt.assert_equal(True, flag is True)

            npt.assert_equal(len(m) == len(x), True)
            npt.assert_equal(len(x) == len(y), True)
            npt.assert_equal(len(y) == len(r3), True)
            npt.assert_equal(len(r3) == len(redshifts), True)
            npt.assert_equal(len(redshifts) == len(sub_flag), True)
            npt.assert_equal(len(prof) == len(zprof), True)
            npt.assert_equal(len(zprof) == len(kw), True)

    def test_two_halo(self):

        for i, kwargs in enumerate([self.kwargs_cdm, self.kwargs_wdm]):

            population_model = HaloPopulation(['TWO_HALO'], kwargs, self.lens_cosmo, self.geometry,
                                              self.halo_mass_function, self.lens_plane_redshifts,
                                              self.delta_zs)

            m, x, y, r3, redshifts, sub_flag = population_model.render()
            prof, zprof, kw = population_model.convergence_sheet_correction()

            for flag in sub_flag:
                npt.assert_equal(True, flag is False)

            npt.assert_equal(len(m) == len(x), True)
            npt.assert_equal(len(x) == len(y), True)
            npt.assert_equal(len(y) == len(r3), True)
            npt.assert_equal(len(r3) == len(redshifts), True)
            npt.assert_equal(len(redshifts) == len(sub_flag), True)
            npt.assert_equal(len(prof) == len(zprof), True)
            npt.assert_equal(len(zprof) == len(kw), True)

    def test_combined(self):

        for i, kwargs in enumerate([self.kwargs_cdm, self.kwargs_wdm]):

            population_model = HaloPopulation(['LINE_OF_SIGHT', 'SUBHALOS', 'TWO_HALO'], kwargs, self.lens_cosmo, self.geometry,
                                              self.halo_mass_function, self.lens_plane_redshifts,
                                              self.delta_zs)

            m, x, y, r3, redshifts, sub_flag = population_model.render()
            prof, zprof, kw = population_model.convergence_sheet_correction()

            npt.assert_equal(len(m) == len(x), True)
            npt.assert_equal(len(x) == len(y), True)
            npt.assert_equal(len(y) == len(r3), True)
            npt.assert_equal(len(r3) == len(redshifts), True)
            npt.assert_equal(len(redshifts) == len(sub_flag), True)
            npt.assert_equal(len(prof) == len(zprof), True)
            npt.assert_equal(len(zprof) == len(kw), True)

    #
    # def test_mass_sheets(self):
    #
    #     kwargs_out, profile_names_out, zout = self.func.negative_kappa_sheets_theory()
    #     kwargs_out_wdm, profile_names_out, zout = self.func.negative_kappa_sheets_theory()
    #
    #     redshifts = self.func.lens_plane_redshifts[0::2]
    #     delta_z = 2 * self.func.delta_zs[0::2]
    #     kappa_scale = self.kwargs['kappa_scale']
    #     for i, (zi, dzi) in enumerate(zip(redshifts, delta_z)):
    #
    #         plaw_index = self.halo_mass_function.plaw_index_z(zi) + self.kwargs['delta_power_law_index']
    #         norm = self.halo_mass_function.norm_at_z(zi, plaw_index, dzi, 10 ** 8) * self.kwargs['LOS_normalization']
    #         mtheory = integrate_power_law_analytic(norm, 10 ** self.kwargs['log_mass_sheet_min'],
    #                                                10 ** self.kwargs['log_mass_sheet_max'],
    #                                                1, plaw_index)
    #
    #         kappa = self.func._convergence_at_z(zi, dzi, self.kwargs['log_mass_sheet_min'],
    #                                                             self.kwargs['log_mass_sheet_max'], self.kwargs['kappa_scale'])
    #
    #         area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, zi)
    #         sigma_crit_mass = self.func.lens_cosmo.sigma_crit_mass(zi, area)
    #
    #         npt.assert_almost_equal(kappa_scale * mtheory / sigma_crit_mass, kappa)
    #         npt.assert_almost_equal(-kwargs_out[i]['kappa_ext'], kappa)
    #         npt.assert_string_equal(profile_names_out[i], 'CONVERGENCE')
    #         npt.assert_almost_equal(zout[i], zi)
    #
    #     kwargs_out_wdm, profile_names_out, zout = self.func_wdm.negative_kappa_sheets_theory()
    #     redshifts = self.func_wdm.lens_plane_redshifts[0::2]
    #     delta_z = 2 * self.func_wdm.delta_zs[0::2]
    #     kappa_scale = self.kwargs_wdm['kappa_scale']
    #     for i, (zi, dzi) in enumerate(zip(redshifts, delta_z)):
    #
    #         plaw_index = self.halo_mass_function.plaw_index_z(zi) + self.kwargs_wdm['delta_power_law_index']
    #         norm = self.halo_mass_function.norm_at_z(zi, plaw_index, dzi, 10 ** 8) * self.kwargs_wdm['LOS_normalization']
    #         mtheory = integrate_power_law_quad(norm, 10 ** self.kwargs_wdm['log_mass_sheet_min'],
    #                                            10 ** self.kwargs_wdm['log_mass_sheet_max'],
    #                                            self.kwargs_wdm['log_mc'],
    #                                            1, plaw_index, self.kwargs_wdm['a_wdm'],
    #                                            self.kwargs_wdm['b_wdm'], self.kwargs_wdm['c_wdm'])
    #
    #         kappa = self.func_wdm._convergence_at_z(zi, dzi, self.kwargs_wdm['log_mass_sheet_min'],
    #                                             self.kwargs_wdm['log_mass_sheet_max'], self.kwargs_wdm['kappa_scale'])
    #
    #         area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, zi)
    #         sigma_crit_mass = self.func_wdm.lens_cosmo.sigma_crit_mass(zi, area)
    #
    #         npt.assert_almost_equal(kappa_scale * mtheory / sigma_crit_mass, kappa)
    #         npt.assert_almost_equal(-kwargs_out_wdm[i]['kappa_ext'], kappa)
    #         npt.assert_string_equal(profile_names_out[i], 'CONVERGENCE')
    #         npt.assert_almost_equal(zout[i], zi)


if __name__ == '__main__':

    pytest.main()

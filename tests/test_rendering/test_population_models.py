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
        kwargs_suppression = {'c_scale': 10.5, 'c_power': -0.2}
        suppression_model = 'polynomial'
        kwargs_cdm = {'zmin': zmin,
                      'zmax': zmax,
                      'log_mc': None,
                      'log_mlow': log_mlow,
                      'sigma_sub': sigma_sub,
                      'kwargs_suppression': kwargs_suppression, 'suppression_model': suppression_model,
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
                      'r_tidal': '0.5Rs',
                      'mass_function_LOS_type': 'POWER_LAW'}

        kwargs_wdm = deepcopy(kwargs_cdm)

        log_mc = 7.3
        a_wdm = 0.8
        b_wdm = 1.1
        c_wdm = -1.2
        kwargs_wdm['log_mc'] = log_mc
        kwargs_wdm['a_wdm'] = a_wdm
        kwargs_wdm['b_wdm'] = b_wdm
        kwargs_wdm['c_wdm'] = c_wdm
        kwargs_wdm['kwargs_suppression'] = kwargs_suppression
        kwargs_wdm['suppression_model'] = suppression_model

        kwargs_no_sheet = deepcopy(kwargs_cdm)
        kwargs_no_sheet['mass_function_LOS_type'] = 'DELTA'
        kwargs_no_sheet['mass_fraction'] = 0.1
        kwargs_no_sheet['logM'] = 6.

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
        self.kwargs_no_sheet = kwargs_no_sheet

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

    def test_line_of_sight_nosheet(self):

        population_model = HaloPopulation(['LINE_OF_SIGHT_NOSHEET'], self.kwargs_no_sheet, self.lens_cosmo, self.geometry,
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
        npt.assert_equal(len(zprof) == 0, True)

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

    def test_redshift_dependent_mass(self):

        def log_mlow_func(z):

            if z > 0.5:
                return 6.
            else:
                return 7.

        def log_mhigh_func(z):

            if z > 0.5:
                return 9.
            else:
                return 9.5

        pop_model = HaloPopulation(['LINE_OF_SIGHT'], self.kwargs_cdm, self.lens_cosmo, self.geometry,
                                              self.halo_mass_function, self.lens_plane_redshifts,
                                              self.delta_zs)
        ml, mh = pop_model.rendering_classes[0]._redshift_dependent_mass_range(1., log_mlow_func, log_mhigh_func)
        npt.assert_equal(ml, 6)
        npt.assert_equal(mh, 9)

        ml, mh = pop_model.rendering_classes[0]._redshift_dependent_mass_range(0.1, log_mlow_func, log_mhigh_func)
        npt.assert_equal(ml, 7)
        npt.assert_equal(mh, 9.5)

        ml, mh = pop_model.rendering_classes[0]._redshift_dependent_mass_range(0.1, 7.5, log_mhigh_func)
        npt.assert_equal(ml, 7.5)
        npt.assert_equal(mh, 9.5)

if __name__ == '__main__':

    pytest.main()

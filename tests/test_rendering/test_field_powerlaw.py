from pyHalo.Rendering.Field.PowerLaw.powerlaw import LOSPowerLaw
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Rendering.render import render_los
import numpy as np
import numpy.testing as npt
import pytest

class TestFieldPowerLaw(object):

    def setup(self):

        self.delta_power_law_index = -0.17
        kwargs = {'zmin': 0.01, 'zmax': 1.98, 'log_m_break': None, 'log_mlow': 6.,
         'log_mhigh': 9., 'host_m200': 10**13, 'LOS_normalization': 2000., 'LOS_normalization_mass_sheet': 1.,
         'draw_poisson': False, 'log_mass_sheet_min': 7., 'log_mass_sheet_max': 10., 'kappa_scale': 1.,
         'break_index': None, 'break_scale': None, 'delta_power_law_index': self.delta_power_law_index,
         'm_pivot': 10**8, 'cone_opening_angle': 6.}
        self.kwargs = kwargs

        kwargs_wdm = {'zmin': 0.01, 'zmax': 1.98, 'log_m_break': 7.5, 'log_mlow': 6.,
                  'log_mhigh': 9., 'host_m200': 10 ** 13, 'LOS_normalization': 2000.,
                  'draw_poisson': False, 'log_mass_sheet_min': 7., 'log_mass_sheet_max': 10., 'kappa_scale': 1.6,
                  'break_index': -1.3, 'break_scale': 1., 'delta_power_law_index': self.delta_power_law_index,
                  'm_pivot': 10 ** 8, 'cone_opening_angle': 6.}
        self.kwargs_wdm = kwargs_wdm

        self.opening_angle= 6.
        cosmo = Cosmology()
        halo_mass_function = LensingMassFunction(cosmo, kwargs['log_mlow'], kwargs['log_mhigh'],
                                                 0.5, 1.5, 6., m_pivot=kwargs['m_pivot'], two_halo_term=True,
                                                 geometry_type='DOUBLE_CONE')
        self.halo_mass_function = halo_mass_function
        geometry_class = Geometry(cosmo, 0.5, 1.5, 6., 'DOUBLE_CONE')
        self.geometry = geometry_class
        self.cosmo = geometry_class._cosmo
        self.arcsec = self.cosmo.arcsec

        lens_plane_redshifts = np.append(np.arange(0.01, 0.5, 0.02), np.arange(0.5, 1.5, 0.02))
        delta_zs = []
        for i in range(0, len(lens_plane_redshifts) - 1):
            delta_zs.append(lens_plane_redshifts[i + 1] - lens_plane_redshifts[i])
        delta_zs.append(1.5 - lens_plane_redshifts[-1])

        self.func = LOSPowerLaw(kwargs, halo_mass_function, geometry_class, lens_plane_redshifts, delta_zs)
        self.func_wdm = LOSPowerLaw(kwargs_wdm, halo_mass_function, geometry_class, lens_plane_redshifts, delta_zs)

    def test_render_function(self):

        m, x, y, r3, redshifts = render_los(self.func, self.func.lens_plane_redshifts, self.func.delta_zs,
                                    0.0, 1.5)
        mtotal = np.sum(m)
        mtheory = 0
        for zi, dzi in zip(self.func.lens_plane_redshifts, self.func.delta_zs):
            plawindex = self.halo_mass_function.plaw_index_z(zi) + self.delta_power_law_index
            mtheory += self.halo_mass_function.integrate_mass_function(zi, plawindex, dzi, 10 ** self.kwargs['log_mlow'],
                                                                      10 ** self.kwargs['log_mhigh'], None, None, None,
                                                                      n=1,
                                                                      norm_scale=self.kwargs['LOS_normalization'])
        npt.assert_array_less(abs(mtheory/mtotal - 1), 0.2)

        m, x, y, r3, redshifts = render_los(self.func_wdm, self.func_wdm.lens_plane_redshifts, self.func_wdm.delta_zs,
                                            0.0, 1.5)
        mtotal = np.sum(m)
        mtheory = 0
        for zi, dzi in zip(self.func_wdm.lens_plane_redshifts, self.func_wdm.delta_zs):
            plawindex = self.halo_mass_function.plaw_index_z(zi) + self.delta_power_law_index
            mtheory += self.halo_mass_function.integrate_mass_function(zi, plawindex, dzi,
                                                                       10 ** self.kwargs['log_mlow'],
                                                                       10 ** self.kwargs['log_mhigh'],
                                                                       self.kwargs_wdm['log_m_break'],
                                                                       self.kwargs_wdm['break_index'],
                                                                       self.kwargs_wdm['break_scale'],
                                                                       n=1, norm_scale=self.kwargs['LOS_normalization'])
        npt.assert_array_less(abs(mtheory / mtotal - 1), 0.2)

    def test_render_positions_at_z(self):

        z = 0.1
        x, y, r3 = self.func.render_positions_at_z(z, 100, 0, 0)
        npt.assert_almost_equal(len(x), 100)
        npt.assert_almost_equal(len(y), 100)
        npt.assert_almost_equal(len(r3), 100)
        r2max = max(np.hypot(x, y))
        rad = self.geometry.angle_to_comovingradius(self.opening_angle/2, z)
        tz = self.cosmo.D_C_transverse(z)
        npt.assert_array_less(r2max, rad/tz/self.arcsec)

        z = 0.5
        x, y, r3 = self.func.render_positions_at_z(z, 100, 0, 0)
        npt.assert_almost_equal(len(x), 100)
        npt.assert_almost_equal(len(y), 100)
        npt.assert_almost_equal(len(r3), 100)
        r2max = max(np.hypot(x, y))
        rad = self.geometry.angle_to_comovingradius(self.opening_angle / 2, z)
        tz = self.cosmo.D_C_transverse(z)
        npt.assert_array_less(r2max, rad / tz / self.arcsec)

        z = 1.3
        x, y, r3 = self.func.render_positions_at_z(z, 100, 0, 0)
        npt.assert_almost_equal(len(x), 100)
        npt.assert_almost_equal(len(y), 100)
        npt.assert_almost_equal(len(r3), 100)
        r2max = max(np.hypot(x, y))
        rad = self.geometry.angle_to_comovingradius(self.opening_angle / 2, z)
        tz = self.cosmo.D_C_transverse(z)
        rmax = rad / tz / self.arcsec
        npt.assert_array_less(r2max, rmax)

    def test_twohalo_boost(self):

        boost = self.func.two_halo_boost(0.3, 0.01, 10**13, 0.5, self.halo_mass_function)
        npt.assert_array_less(boost, 1.0000001)

        boost = self.func.two_halo_boost(0.5, 0.01, 10 ** 13, 0.5, self.halo_mass_function)
        npt.assert_array_less(1.000001, boost)

    def test_render_masses(self):

        z = 0.2
        dz = 0.03
        m = self.func.render_masses(z, dz, None)
        plaw_index = self.halo_mass_function.plaw_index_z(z) + self.delta_power_law_index
        mtheory = self.halo_mass_function.integrate_mass_function(z, plaw_index, dz, 10**self.kwargs['log_mlow'],
                                                        10**self.kwargs['log_mhigh'], None, None, None, n=1,
                                                                  norm_scale=self.kwargs['LOS_normalization'])
        diff = np.absolute(1 - mtheory/np.sum(m))
        npt.assert_array_less(diff, 0.1, 2)

        m = self.func_wdm.render_masses(z, dz, None)
        plaw_index = self.halo_mass_function.plaw_index_z(z) + self.delta_power_law_index
        mtheory = self.halo_mass_function.integrate_mass_function(z, plaw_index, dz, 10 ** self.kwargs['log_mlow'],
                                                                  10 ** self.kwargs['log_mhigh'], self.kwargs_wdm['log_m_break'],
                                                                  self.kwargs_wdm['break_index'], self.kwargs_wdm['break_scale'],
                                                                  n=1, norm_scale=self.kwargs_wdm['LOS_normalization'])
        diff = np.absolute(1 - mtheory / np.sum(m))
        npt.assert_array_less(diff, 0.1, 2)




        z = 0.9
        dz = 0.03
        m = self.func.render_masses(z, dz, None)

        plaw_index = self.halo_mass_function.plaw_index_z(z) + self.delta_power_law_index
        mtheory = self.halo_mass_function.integrate_mass_function(z, plaw_index, dz, 10 ** self.kwargs['log_mlow'],
                                                                  10 ** self.kwargs['log_mhigh'], None, None, None, n=1,
                                                                  norm_scale=self.kwargs['LOS_normalization'])
        diff = np.absolute(1 - mtheory / np.sum(m))
        npt.assert_array_less(diff, 0.1, 2)

        m = self.func_wdm.render_masses(z, dz, None)
        plaw_index = self.halo_mass_function.plaw_index_z(z) + self.delta_power_law_index
        mtheory = self.halo_mass_function.integrate_mass_function(z, plaw_index, dz, 10 ** self.kwargs['log_mlow'],
                                                                  10 ** self.kwargs['log_mhigh'],
                                                                  self.kwargs_wdm['log_m_break'],
                                                                  self.kwargs_wdm['break_index'],
                                                                  self.kwargs_wdm['break_scale'],
                                                                  n=1, norm_scale=self.kwargs['LOS_normalization'])
        diff = np.absolute(1 - mtheory / np.sum(m))
        npt.assert_array_less(diff, 0.1, 2)

    def test_normalization(self):

        z = 0.6
        dz = 0.01
        plaw_idx = self.halo_mass_function.plaw_index_z(z) + self.delta_power_law_index
        dv = self.halo_mass_function.geometry.volume_element_comoving(z, dz)
        rho_dv = self.halo_mass_function.norm_at_z_density(z, plaw_idx, 10**8)
        norm = rho_dv * dv

        scale = 1

        m_pivot = 10**8

        norm_code = self.func.normalization(z, dz, 0.5, self.halo_mass_function, self.kwargs['host_m200'],
                                            dv, scale, plaw_idx, m_pivot)

        npt.assert_almost_equal(norm_code, norm)

        args = (z, dz, 0.5, self.halo_mass_function, self.kwargs['host_m200'],
                                            dv, scale, plaw_idx, 2 * m_pivot)
        npt.assert_raises(AssertionError, self.func.normalization, *args)

    def test_mass_sheets(self):

        kwargs_out, profile_names_out, zout = self.func.negative_kappa_sheets_theory()
        kwargs_out_wdm, profile_names_out, zout = self.func.negative_kappa_sheets_theory()
        redshifts = self.func.lens_plane_redshifts[0::2]
        delta_z = 2 * self.func.delta_zs[0::2]
        kappa_scale = self.kwargs['kappa_scale']
        for i, (zi, dzi) in enumerate(zip(redshifts, delta_z)):

            plaw_index = self.halo_mass_function.plaw_index_z(zi) + self.delta_power_law_index
            mtheory = self.halo_mass_function.integrate_mass_function(zi, plaw_index, dzi,
                                      10 ** self.kwargs['log_mass_sheet_min'], 10 ** self.kwargs['log_mass_sheet_max'], None, None, None,
                                           n=1, norm_scale=self.kwargs['LOS_normalization'])

            kappa = self.func._convergence_at_z(zi, dzi, self.kwargs['log_mass_sheet_min'],
                                                                self.kwargs['log_mass_sheet_max'], self.kwargs['kappa_scale'])

            area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, zi)
            sigma_crit_mass = self.func.lens_cosmo.sigma_crit_mass(zi, area)

            npt.assert_almost_equal(kappa_scale * mtheory / sigma_crit_mass, kappa)
            npt.assert_almost_equal(-kwargs_out[i]['kappa_ext'], kappa)
            npt.assert_string_equal(profile_names_out[i], 'CONVERGENCE')
            npt.assert_almost_equal(zout[i], zi)

        kwargs_out_wdm, profile_names_out, zout = self.func_wdm.negative_kappa_sheets_theory()
        redshifts = self.func_wdm.lens_plane_redshifts[0::2]
        delta_z = 2 * self.func_wdm.delta_zs[0::2]
        kappa_scale = self.kwargs_wdm['kappa_scale']
        for i, (zi, dzi) in enumerate(zip(redshifts, delta_z)):

            plaw_index = self.halo_mass_function.plaw_index_z(zi) + self.delta_power_law_index
            mtheory = self.halo_mass_function.integrate_mass_function(zi, plaw_index, dzi,
                                                                      10 ** self.kwargs_wdm['log_mass_sheet_min'],
                                                                      10 ** self.kwargs_wdm['log_mass_sheet_max'],
                                                                      self.kwargs_wdm['log_m_break'],
                                                                      self.kwargs_wdm['break_index'], self.kwargs_wdm['break_scale'],
                                                                      1, self.kwargs_wdm['LOS_normalization'])

            kappa = self.func_wdm._convergence_at_z(zi, dzi, self.kwargs_wdm['log_mass_sheet_min'],
                                                self.kwargs_wdm['log_mass_sheet_max'], self.kwargs_wdm['kappa_scale'])

            area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, zi)
            sigma_crit_mass = self.func_wdm.lens_cosmo.sigma_crit_mass(zi, area)

            npt.assert_almost_equal(kappa_scale * mtheory / sigma_crit_mass, kappa)
            npt.assert_almost_equal(-kwargs_out_wdm[i]['kappa_ext'], kappa)
            npt.assert_string_equal(profile_names_out[i], 'CONVERGENCE')
            npt.assert_almost_equal(zout[i], zi)

if __name__ == '__main__':

    pytest.main()

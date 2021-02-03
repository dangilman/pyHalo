from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Rendering.two_halo import TwoHaloContribution
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from pyHalo.pyhalo import pyHalo
import pytest

class TestLOS(object):

    def setup(self):
        zlens, zsource = 0.5, 2.
        zmin = 0.01
        zmax = 1.98
        log_mlow = 6.
        log_mhigh = 9.
        host_m200 = 10 ** 13
        LOS_normalization = 1.
        draw_poisson = False
        log_mass_sheet_min = 7.
        log_mass_sheet_max = 10.
        kappa_scale = 1.
        delta_power_law_index = -0.1
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
                      'mdef_los': 'TNFW',
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

        pyhalo = pyHalo(zlens, zsource)
        self.realization_cdm = pyhalo.render(['TWO_HALO'], kwargs_cdm)[0]

        lens_plane_redshifts, delta_zs = pyhalo.lens_plane_redshifts(kwargs_cdm)

        cosmo = Cosmology()
        self.lens_plane_redshifts = lens_plane_redshifts
        self.delta_zs = delta_zs
        self.halo_mass_function = LensingMassFunction(cosmo, zlens, zsource, kwargs_cdm['log_mlow'], kwargs_cdm['log_mhigh'],
                                                      kwargs_cdm['cone_opening_angle'],
                                                      m_pivot=kwargs_cdm['m_pivot'],
                                                      geometry_type='DOUBLE_CONE')
        self.geometry = Geometry(cosmo, zlens, zsource, kwargs_cdm['cone_opening_angle'], 'DOUBLE_CONE')
        self.lens_cosmo = LensCosmo(zlens, zsource, cosmo)

        self.kwargs_cdm = kwargs_cdm

        self.rendering_class = TwoHaloContribution(kwargs_cdm, self.halo_mass_function, self.geometry, self.lens_cosmo,
                                            self.lens_plane_redshifts, self.delta_zs)

    def test_norm_slope(self):

        delta_z = self.delta_zs[1]
        norm, slope = self.rendering_class._norm_slope(self.lens_cosmo.z_lens, delta_z)
        slope_theory = self.halo_mass_function.plaw_index_z(self.lens_cosmo.z_lens) + self.kwargs_cdm['delta_power_law_index']
        npt.assert_almost_equal(slope, slope_theory)
        norm_theory_background = self.halo_mass_function.norm_at_z_density(self.lens_cosmo.z_lens,
                                                                           slope,
                                                                           self.kwargs_cdm['m_pivot'])
        norm_theory_background *= self.geometry.volume_element_comoving(self.lens_cosmo.z_lens, delta_z)
        rmax = self.lens_cosmo.cosmo.D_C_transverse(self.lens_cosmo.z_lens + delta_z) - \
               self.lens_cosmo.cosmo.D_C_transverse(self.lens_cosmo.z_lens)
        boost = self.halo_mass_function.two_halo_boost(self.kwargs_cdm['host_m200'], self.lens_cosmo.z_lens, rmax=rmax)
        norm_theory = (boost - 1) * norm_theory_background
        npt.assert_almost_equal(norm_theory, norm)

    def test_rendering(self):

        m = self.rendering_class.render_masses_at_z(self.lens_cosmo.z_lens, 0.02)
        npt.assert_equal(True, len(m) > 0)

        npt.assert_raises(Exception, self.rendering_class.render_masses_at_z, 0.9, 0.02)

        x, y = self.rendering_class.render_positions_at_z(self.lens_cosmo.z_lens, 10000)
        rmax = np.max(np.hypot(x, y))
        rmax_theory = 0.5 * self.kwargs_cdm['cone_opening_angle'] * self.geometry.rendering_scale(self.lens_cosmo.z_lens)
        npt.assert_array_less(rmax, rmax_theory)

        x, y = self.rendering_class.render_positions_at_z(self.lens_cosmo.z_lens, 0)
        npt.assert_equal(True, len(x) == 0)

        m, x, y, r3, redshifts, flag = self.rendering_class.render()
        npt.assert_equal(len(m), len(x))
        npt.assert_equal(len(y), len(r3))

        n = 0
        for z in np.unique(redshifts):
            n += np.sum(redshifts == z)
        npt.assert_equal(True, n==len(m))

        npt.assert_equal(True, len(self.realization_cdm.halos) == len(m))

    def test_convergence_correction(self):

        kwargs_out, profile_names_out, redshifts = self.rendering_class.convergence_sheet_correction()

        npt.assert_equal(0, len(kwargs_out))
        npt.assert_equal(0, len(profile_names_out))
        npt.assert_equal(0, len(redshifts))

    def test_keys_convergence_sheets(self):

        keywords_out = self.rendering_class.keys_convergence_sheets()
        npt.assert_equal(len(keywords_out), 0)

    def test_keys_rendering(self):

        keywords_out = self.rendering_class.keyword_parse_render(self.kwargs_cdm)
        required_keys = ['log_mlow', 'log_mhigh', 'host_m200', 'LOS_normalization',
                         'draw_poisson', 'delta_power_law_index', 'm_pivot', 'log_mc', 'a_wdm', 'b_wdm', 'c_wdm']

        for x in required_keys:
            npt.assert_equal(x in keywords_out.keys(), True)

        kw_cdm = deepcopy(self.kwargs_cdm)
        kw_cdm['log_mc'] = None
        keywords_out = self.rendering_class.keyword_parse_render(kw_cdm)
        npt.assert_equal(keywords_out['a_wdm'] is None, True)
        npt.assert_equal(keywords_out['b_wdm'] is None, True)
        npt.assert_equal(keywords_out['c_wdm'] is None, True)

if __name__ == '__main__':
    pytest.main()

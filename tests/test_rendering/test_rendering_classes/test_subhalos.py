from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Rendering.subhalos import Subhalos, normalization_sigmasub
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from pyHalo.pyhalo import pyHalo
import pytest
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_analytic

class TestSubhalos(object):

    def setup(self):
        zlens, zsource = 0.5, 2.
        zmin = 0.01
        zmax = 1.98
        log_mlow = 6.
        log_mhigh = 9.
        host_m200 = 10 ** 13.5
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

        kwargs_cdm_uniform = {'zmin': zmin,
                      'zmax': zmax,
                      'log_mc': None,
                      'log_mlow': log_mlow,
                      'sigma_sub': sigma_sub,
                      'c_scale': None, 'c_power': None,
                      'a_wdm': None, 'b_wdm': None, 'c_wdm': None,
                      'c_scatter_dex': 0.2,
                      'log_mhigh': log_mhigh,
                      'mdef_subs': 'TNFW',
                      'log_m_host': np.log10(host_m200),
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
                      'subhalo_convergence_correction_profile': 'UNIFORM',
                      'r_tidal': '0.5Rs'}
        kwargs_cdm_nfw = deepcopy(kwargs_cdm_uniform)
        kwargs_cdm_nfw['subhalo_convergence_correction_profile'] = 'NFW'

        pyhalo = pyHalo(zlens, zsource)
        self.realization_cdm = pyhalo.render(['SUBHALOS'], kwargs_cdm_uniform)[0]
        self.realization_cdm_nfw = pyhalo.render(['SUBHALOS'], kwargs_cdm_nfw)[0]

        lens_plane_redshifts, delta_zs = pyhalo.lens_plane_redshifts(kwargs_cdm_uniform)

        cosmo = Cosmology()
        self.lens_plane_redshifts = lens_plane_redshifts
        self.delta_zs = delta_zs
        self.halo_mass_function = LensingMassFunction(cosmo, zlens, zsource, kwargs_cdm_uniform['log_mlow'],
                                                      kwargs_cdm_uniform['log_mhigh'],
                                                      zlens, zsource, kwargs_cdm_uniform['cone_opening_angle'],
                                                      m_pivot=kwargs_cdm_uniform['m_pivot'],
                                                      geometry_type='DOUBLE_CONE')
        self.geometry = Geometry(cosmo, zlens, zsource, kwargs_cdm_uniform['cone_opening_angle'], 'DOUBLE_CONE')
        self.lens_cosmo = LensCosmo(zlens, zsource, cosmo)

        self.kwargs_cdm = kwargs_cdm_uniform

        self.rendering_class_uniform = Subhalos(kwargs_cdm_uniform, self.geometry, self.lens_cosmo)
        self.rendering_class_nfw = Subhalos(kwargs_cdm_nfw, self.geometry, self.lens_cosmo)

    def test_normalization(self):

        norm = normalization_sigmasub(self.kwargs_cdm['sigma_sub'], 10**self.kwargs_cdm['log_m_host'],
                                      self.lens_cosmo.z_lens, self.geometry.kpc_per_arcsec_zlens, self.kwargs_cdm['cone_opening_angle'],
                                      self.kwargs_cdm['power_law_index'] + self.kwargs_cdm['delta_power_law_index'] *
                                      self.kwargs_cdm['delta_power_law_index_coupling'], self.kwargs_cdm['m_pivot'])

        k1, k2 = 0.88, 1.7
        slope = self.kwargs_cdm['power_law_index'] + self.kwargs_cdm['delta_power_law_index'] * \
                     self.kwargs_cdm['delta_power_law_index_coupling']
        mhalo = 10 ** self.kwargs_cdm['log_m_host']
        scale = k1 * np.log10(mhalo / 10**13) + k2 * np.log10(self.lens_cosmo.z_lens + 0.5)
        host_scaling = 10 ** scale

        norm_theory = self.kwargs_cdm['sigma_sub'] * host_scaling

        kpc_per_arcsec_zlens = self.geometry.kpc_per_arcsec(self.lens_cosmo.z_lens)
        norm_theory *= np.pi * (0.5 * self.kwargs_cdm['cone_opening_angle'] * kpc_per_arcsec_zlens) ** 2
        norm_theory *= self.kwargs_cdm['m_pivot'] ** -(slope + 1)

        npt.assert_almost_equal(norm, norm_theory)

        _norm, _slope = self.rendering_class_uniform._norm_slope()
        npt.assert_almost_equal(_norm, norm)
        npt.assert_almost_equal(_slope, slope)

    def test_rendering(self):

        m = self.rendering_class_uniform.render_masses_at_z()
        npt.assert_equal(True, len(m) > 0)
        x, y, r3 = self.rendering_class_uniform.render_positions_at_z(10000)
        rmax = np.max(np.hypot(x, y))
        rmax_theory = 0.5 * self.kwargs_cdm['cone_opening_angle'] * self.geometry.rendering_scale(self.lens_cosmo.z_lens)
        npt.assert_array_less(rmax, rmax_theory)

        m, x, y, r3, redshifts, flag = self.rendering_class_uniform.render()
        npt.assert_equal(len(m), len(x))
        npt.assert_equal(len(y), len(r3))

        n = 0
        for z in np.unique(redshifts):
            n += np.sum(redshifts == z)
        npt.assert_equal(True, n==len(m))

        npt.assert_equal(True, len(self.realization_cdm.halos) == len(m))

    def test_convergence_correction(self):

        z = self.lens_cosmo.z_lens

        norm, slope = self.rendering_class_uniform._norm_slope()

        # factor of two because the kappa sheets are added at every other lens plane
        mtheory = integrate_power_law_analytic(norm, 10 ** self.kwargs_cdm['log_mass_sheet_min'],
                                               10 ** self.kwargs_cdm['log_mass_sheet_max'],
                                               1., slope)
        area = self.geometry.angle_to_physical_area(0.5 * self.kwargs_cdm['cone_opening_angle'], z)
        kappa_theory = mtheory / self.lens_cosmo.sigma_crit_mass(z, area)

        kwargs_out, profile_names_out, redshifts = self.rendering_class_uniform.convergence_sheet_correction()

        kw = kwargs_out[0]
        name = profile_names_out[0]

        npt.assert_equal(True, name=='CONVERGENCE')
        kappa_generated = -kw['kappa_ext']
        npt.assert_array_less(abs(kappa_theory/kappa_generated - 1), 0.05)

    def test_keys_convergence_sheets(self):

        keywords_out = self.rendering_class_uniform.keys_convergence_sheets(self.kwargs_cdm)
        required_keys = ['log_mass_sheet_min', 'log_mass_sheet_max', 'subhalo_mass_sheet_scale',
                         'subhalo_convergence_correction_profile',
                         'r_tidal', 'delta_power_law_index', 'delta_power_law_index_coupling']

        for x in required_keys:
            npt.assert_equal(x in keywords_out.keys(), True)

    def test_keys_rendering(self):

        keywords_out = self.rendering_class_uniform.keyword_parse_render(self.kwargs_cdm)
        required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_mc', 'sigma_sub',
                         'a_wdm', 'b_wdm', 'c_wdm', 'log_mass_sheet_min', 'log_mass_sheet_max',
                         'subhalo_mass_sheet_scale', 'draw_poisson', 'host_m200',
                         'subhalo_convergence_correction_profile', 'r_tidal',
                         'delta_power_law_index', 'm_pivot', 'delta_power_law_index_coupling',
                         'cone_opening_angle']

        for x in required_keys:
            npt.assert_equal(x in keywords_out.keys(), True)

        kw_cdm = deepcopy(self.kwargs_cdm)
        kw_cdm['log_mc'] = None

        keywords_out = self.rendering_class_uniform.keyword_parse_render(kw_cdm)
        npt.assert_equal(keywords_out['a_wdm'] is None, True)
        npt.assert_equal(keywords_out['b_wdm'] is None, True)
        npt.assert_equal(keywords_out['c_wdm'] is None, True)

if __name__ == '__main__':
    pytest.main()

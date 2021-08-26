from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Halos.lens_cosmo import LensCosmo
from copy import deepcopy
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Rendering.MassFunctions.mass_function_utilities import WDM_suppression
import numpy as np
import numpy.testing as npt
from pyHalo.pyhalo import pyHalo
import pytest
from pyHalo.Rendering.subhalos import normalization_sigmasub
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, integrate_power_law_analytic

class TestRenderedPopulations(object):

    def setup(self):

        zlens, zsource = 0.5, 2.
        zmin = 0.01
        zmax = 1.98
        log_mlow = 6.
        log_mhigh = 9.
        host_m200 = 10 ** 13
        LOS_normalization = 100.
        draw_poisson = False
        log_mass_sheet_min = 7.
        log_mass_sheet_max = 10.
        kappa_scale = 1.
        delta_power_law_index = -0.17
        delta_power_law_index_coupling = 0.7
        cone_opening_angle = 6.
        m_pivot = 10 ** 8
        sigma_sub = 0.6
        power_law_index = -1.8
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
                      'r_tidal': '0.5Rs',
                      'mdef_subs': 'TNFW',
                      'mdef_los': 'TNFW'}

        kwargs_wdm = deepcopy(kwargs_cdm)

        log_mc = 7.1
        a_wdm = 1.
        b_wdm = 0.8
        c_wdm = -1.3
        kwargs_wdm['log_mc'] = log_mc
        kwargs_wdm['a_wdm'] = a_wdm
        kwargs_wdm['b_wdm'] = b_wdm
        kwargs_wdm['c_wdm'] = c_wdm
        kwargs_wdm['c_scale'] = 60.
        kwargs_wdm['c_power'] = -0.17

        pyhalo = pyHalo(zlens, zsource)
        lens_plane_redshifts, delta_zs = pyhalo.lens_plane_redshifts(kwargs_cdm)
        cosmo = Cosmology()
        self.lens_plane_redshifts = lens_plane_redshifts
        self.delta_zs = delta_zs
        self.halo_mass_function = LensingMassFunction(cosmo, zlens, zsource, kwargs_cdm['log_mlow'], kwargs_cdm['log_mhigh'],
                                                      kwargs_cdm['cone_opening_angle'], m_pivot=kwargs_cdm['m_pivot'],
                                                      geometry_type='DOUBLE_CONE')
        self.geometry = Geometry(cosmo, zlens, zsource, kwargs_cdm['cone_opening_angle'], 'DOUBLE_CONE')
        self.lens_cosmo = LensCosmo(zlens, zsource, cosmo)

        self.kwargs_cdm = kwargs_cdm
        self.kwargs_wdm = kwargs_wdm

        pyhalo = pyHalo(zlens, zsource)
        model_list_field = ['LINE_OF_SIGHT']
        model_list_sub = ['SUBHALOS']

        self.realization_cdm = pyhalo.render(model_list_sub + model_list_field, kwargs_cdm)[0]
        self.realization_cdm_field = pyhalo.render(model_list_field, kwargs_cdm)[0]
        self.realization_wdm_field = pyhalo.render(model_list_field, kwargs_wdm)[0]
        self.realization_wdm_subhalos = pyhalo.render(model_list_sub, kwargs_wdm)[0]

    def test_mass_rendered_subhalos(self):

        plaw_index = self.kwargs_cdm['power_law_index'] + \
                     self.kwargs_cdm['delta_power_law_index'] * self.kwargs_cdm['delta_power_law_index_coupling']

        norm = normalization_sigmasub(self.kwargs_cdm['sigma_sub'],
                                      self.kwargs_cdm['host_m200'],
                                      self.lens_cosmo.z_lens,
                                      self.geometry.kpc_per_arcsec_zlens,
                                      self.kwargs_cdm['cone_opening_angle'],
                                      plaw_index, self.kwargs_cdm['m_pivot']
                                      )
        mtheory = integrate_power_law_analytic(norm, 10 ** self.kwargs_cdm['log_mlow'], 10 ** self.kwargs_cdm['log_mhigh'],
                                               1., plaw_index)
        m_subs = 0.
        for halo in self.realization_cdm.halos:
            if halo.is_subhalo:
                m_subs += halo.mass
        ratio = mtheory/m_subs
        npt.assert_array_less(1-ratio, 0.05)

        plaw_index = self.kwargs_wdm['power_law_index'] + \
                     self.kwargs_wdm['delta_power_law_index'] * self.kwargs_wdm['delta_power_law_index_coupling']

        norm = normalization_sigmasub(self.kwargs_wdm['sigma_sub'],
                                      self.kwargs_wdm['host_m200'],
                                      self.lens_cosmo.z_lens,
                                      self.geometry.kpc_per_arcsec_zlens,
                                      self.kwargs_cdm['cone_opening_angle'],
                                      plaw_index, self.kwargs_wdm['m_pivot']
                                      )
        mtheory = integrate_power_law_quad(norm, 10 ** self.kwargs_wdm['log_mlow'],
                                               10 ** self.kwargs_wdm['log_mhigh'],
                                                self.kwargs_wdm['log_mc'],
                                               1., plaw_index, self.kwargs_wdm['a_wdm'],
                                           self.kwargs_wdm['b_wdm'], self.kwargs_wdm['c_wdm'])
        m_subs = 0.
        for halo in self.realization_wdm_subhalos.halos:
            if halo.is_subhalo:
                m_subs += halo.mass
        ratio = mtheory / m_subs
        npt.assert_array_less(1 - ratio, 0.1)

    def test_mass_rendered_line_of_sight(self):

        m_theory = 0
        m_rendered = 0

        for z, dz in zip(self.lens_plane_redshifts, self.delta_zs):

            m_rendered += self.realization_cdm_field.mass_at_z_exact(z)

            slope = self.halo_mass_function.plaw_index_z(z) + self.kwargs_cdm['delta_power_law_index']
            norm = self.kwargs_cdm['LOS_normalization'] * \
                   self.halo_mass_function.norm_at_z_density(z, slope, self.kwargs_cdm['m_pivot']) * \
                   self.geometry.volume_element_comoving(z, dz)

            m_theory += integrate_power_law_analytic(norm, 10 ** self.kwargs_cdm['log_mlow'],
                                                            10 ** self.kwargs_cdm['log_mhigh'], 1,
                                                            slope)

        ratio = m_theory/m_rendered
        npt.assert_array_less(1 - ratio, 0.05)

        m_theory = 0
        m_rendered = 0

        for z, dz in zip(self.lens_plane_redshifts, self.delta_zs):

            m_rendered += self.realization_wdm_field.mass_at_z_exact(z)
            slope = self.halo_mass_function.plaw_index_z(z) + self.kwargs_wdm['delta_power_law_index']
            norm = self.kwargs_cdm['LOS_normalization'] * \
                   self.halo_mass_function.norm_at_z_density(z, slope, self.kwargs_wdm['m_pivot']) * \
                    self.geometry.volume_element_comoving(z, dz)

            m_theory += integrate_power_law_quad(norm, 10 ** self.kwargs_wdm['log_mlow'],
                                                            10 ** self.kwargs_wdm['log_mhigh'], self.kwargs_wdm['log_mc'],
                                                        1, slope, self.kwargs_wdm['a_wdm'], self.kwargs_wdm['b_wdm'],
                                                        self.kwargs_wdm['c_wdm'])

        ratio = m_theory / m_rendered
        npt.assert_array_less(1 - ratio, 0.05)

    def test_wdm_mass_function(self):

        mass_bins = np.linspace(6, 9, 20)
        halo_masses_wdm = [halo.mass for halo in self.realization_wdm_field.halos]
        log_halo_mass = np.log10(halo_masses_wdm)
        h_wdm, logM = np.histogram(log_halo_mass, bins=mass_bins)
        logmstep = (logM[1] - logM[0]) / 2
        logM = logM[0:-1] + logmstep

        mass_bins = np.linspace(6, 9, 20)
        halo_masses_cdm = [halo.mass for halo in self.realization_cdm_field.halos]
        log_halo_mass = np.log10(halo_masses_cdm)
        h_cdm, _ = np.histogram(log_halo_mass, bins=mass_bins)

        suppression_factor = WDM_suppression(10**logM,
                                             10**self.kwargs_wdm['log_mc'],
                                             self.kwargs_wdm['a_wdm'],
                                             self.kwargs_wdm['b_wdm'],
                                             self.kwargs_wdm['c_wdm'],
                                             )

        for i in range(0, 10):
            ratio = h_wdm[i]/h_cdm[i]
            npt.assert_almost_equal(ratio, suppression_factor[i], 1)

    def test_cdm_mass_function(self):

        mass_bins = np.linspace(6, 9., 20)
        halo_masses_sub = []
        halo_masses_field = []
        for halo in self.realization_cdm.halos:
            if halo.is_subhalo:
                halo_masses_sub.append(halo.mass)
            else:
                halo_masses_field.append(halo.mass)

        log_halo_mass_sub = np.log10(halo_masses_sub)
        log_halo_mass_field = np.log10(halo_masses_field)
        h_sub, logM_sub = np.histogram(log_halo_mass_sub, bins=mass_bins)
        h_field, logM_field = np.histogram(log_halo_mass_field, bins=mass_bins)

        logN_sub = np.log10(h_sub)
        logN_field = np.log10(h_field)

        slope_theory = -0.9 + self.kwargs_cdm['delta_power_law_index']
        slope = np.polyfit(logM_field[0:-1], logN_field, 1)[0]
        npt.assert_array_less(abs(1 - slope/slope_theory), 0.05, 2)

        slope_theory = (1+self.kwargs_cdm['power_law_index']) + self.kwargs_cdm['delta_power_law_index'] * \
                       self.kwargs_cdm['delta_power_law_index_coupling']
        slope = np.polyfit(logM_sub[0:-1], logN_sub, 1)[0]
        npt.assert_array_less(abs(1 - slope/slope_theory), 0.05, 2)

if __name__ == '__main__':
    pytest.main()

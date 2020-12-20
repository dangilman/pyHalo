from pyHalo.single_realization import Realization
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Rendering.Field.PowerLaw.powerlaw import LOSPowerLaw
import numpy as np
import numpy.testing as npt
from scipy.interpolate import interp1d
import pytest

class TestSingleRealization(object):

    def setup(self):

        zlens, zsource = 0.4, 2.

        masses = [10 ** 9.04, 10 ** 8.2, 10 ** 7.345, 10 ** 7, 10 ** 6.05, 10 ** 7, 10 ** 7]
        x = [0.5, 0.1, -0.9, -1.4, 1.2, 0., -1.]
        y = [0., 0.9, -1.4, -1., -0.4, 1., 0.9]
        redshifts = [0.1, 0.2, 0.3, zlens, zlens, 0.94, 0.94]
        subflags = [False, False, False, True, True, False, False]
        r3d = [100] * len(masses)
        mdefs_TNFW = ['TNFW'] * len(masses)
        #
        self.x = x
        self.y = y
        self.masses = masses
        self.redshifts = redshifts
        self.subflags = subflags
        self.r3d = r3d
        self.mdefs = mdefs_TNFW

        masses2 = [10 ** 6.34, 10 ** 9.2, 10 ** 8.36]
        x2 = [1.5, 0.15, -1.9]
        y2 = [0.2, 0.3, -0.44]
        redshifts2 = [zlens, 0.4, 1.9]
        subflags2 = [True, False, False]
        r3d2 = [100] * len(masses2)


        profile_args_TNFW = {'mc_model': 'diemer19', 'c_scatter': False, 'c_scatter_dex': 0.1}
        cosmo = Cosmology()
        halo_mass_function = LensingMassFunction(cosmo, 10 ** 6, 10 ** 9, zlens, zsource, 6.,
                                                         m_pivot=10 ** 8)
        self.halo_mass_function = halo_mass_function
        kwargs_cdm = {'zmin': 0.01, 'zmax': 1.98, 'log_mc': None, 'log_mlow': 6.,
                  'log_mhigh': 9., 'host_m200': 10 ** 13, 'LOS_normalization': 2000.,
                  'LOS_normalization_mass_sheet': 1.,
                  'draw_poisson': False, 'log_mass_sheet_min': 7., 'log_mass_sheet_max': 10., 'kappa_scale': 1.,
                  'delta_power_law_index': 0., 'a_wdm': None, 'b_wdm': None, 'c_wdm': None,
                  'm_pivot': 10 ** 8, 'cone_opening_angle': 6.}
        kwargs_cdm_2 = {'zmin': 0.01, 'zmax': 1.98, 'log_mc': None, 'log_mlow': 6.,
                      'log_mhigh': 9., 'host_m200': 10 ** 13, 'LOS_normalization': 2000.,
                      'LOS_normalization_mass_sheet': 1.,
                      'draw_poisson': False, 'log_mass_sheet_min': 7., 'log_mass_sheet_max': 10., 'kappa_scale': 1.,
                      'delta_power_law_index': 0., 'a_wdm': None, 'b_wdm': None, 'c_wdm': None,
                      'm_pivot': 10 ** 8, 'cone_opening_angle': 6., 'subtract_exact_mass_sheets': True}

        kwargs_cdm.update(profile_args_TNFW)
        self.kwargs_cdm = kwargs_cdm

        lens_plane_redshifts = np.append(np.arange(0.01, 0.5, 0.02), np.arange(0.5, 1.5, 0.02))
        delta_zs = []
        for i in range(0, len(lens_plane_redshifts) - 1):
            delta_zs.append(lens_plane_redshifts[i + 1] - lens_plane_redshifts[i])
        delta_zs.append(1.5 - lens_plane_redshifts[-1])

        rendering_classes = [LOSPowerLaw(kwargs_cdm,
                                        halo_mass_function,
                                        halo_mass_function.geometry,
                                        lens_plane_redshifts,
                                        delta_zs)]
        self.rendering_classes = rendering_classes

        mdef = 'NFW'
        self.realization_cdm = Realization(masses, x, y, r3d, mdef, redshifts, subflags,
                                  halo_mass_function, halos=None, halo_profile_args=self.kwargs_cdm_2,
                                  mass_sheet_correction=True, rendering_classes=rendering_classes)

        mdef = 'PT_MASS'
        self.realization_cdm2 = Realization(masses2, x2, y2, r3d2, mdef, redshifts2, subflags2,
                                           halo_mass_function, halos=None, halo_profile_args=self.kwargs_cdm,
                                           mass_sheet_correction=True, rendering_classes=rendering_classes)

        mdef = 'PJAFFE'
        self.realization_cdm3 = Realization(masses2, x2, y2, r3d2, mdef, redshifts2, subflags2,
                                            halo_mass_function, halos=None, halo_profile_args=self.kwargs_cdm,
                                            mass_sheet_correction=True, rendering_classes=rendering_classes)

        self.halos_cdm = self.realization_cdm.halos

        halo_tags = []
        for halo in self.realization_cdm.halos:
            halo_tags.append(halo.unique_tag)
        self.halo_tags = halo_tags
        halo_tags = []
        for halo in self.realization_cdm2.halos:
            halo_tags.append(halo.unique_tag)
        self.halo_tags2 = halo_tags
        self.real_1_data = [x, y, masses, redshifts, subflags, redshifts]
        self.real_2_data = [x2, y2, masses2, redshifts2, subflags2, redshifts2]

    def test_build_from_halos(self):

        realization_fromhalos = Realization.from_halos(self.halos_cdm, self.halo_mass_function,
                                   self.kwargs_cdm, self.realization_cdm.apply_mass_sheet_correction,
                                                       self.realization_cdm.rendering_classes)

        for halo_1, halo_2 in zip(realization_fromhalos.halos, self.realization_cdm.halos):
            npt.assert_equal(halo_1.x, halo_2.x)
            npt.assert_equal(halo_1.y, halo_2.y)
            npt.assert_equal(halo_1.mass, halo_2.mass)
            npt.assert_equal(halo_1.r3d, halo_2.r3d)
            npt.assert_string_equal(halo_1.mdef, halo_2.mdef)
            npt.assert_equal(halo_1.z, halo_2.z)
            npt.assert_equal(halo_1.is_subhalo, halo_2.is_subhalo)
            npt.assert_equal(halo_1.unique_tag, halo_2.unique_tag)

        npt.assert_equal(realization_fromhalos.apply_mass_sheet_correction, self.realization_cdm.apply_mass_sheet_correction)

    def test_join(self):

        new_realization = self.realization_cdm.join(self.realization_cdm2, join_rendering_classes=True)
        npt.assert_equal(len(new_realization.rendering_classes), 2)

        length = len(self.realization_cdm2.halos) + len(self.realization_cdm.halos)
        new_realization = self.realization_cdm.join(self.realization_cdm2)
        npt.assert_equal(len(new_realization.halos), length)
        npt.assert_equal(len(new_realization.rendering_classes), 1)
        # now change two of the tags to be the same as in realization_cdm, make sure they're gone

        idx2_1 = 0
        idx2_2 = 1
        idx1 = 0
        idx2 = 1

        self.realization_cdm.halos[idx1].unique_tag = self.realization_cdm2.halos[idx2_1].unique_tag
        self.realization_cdm.halos[idx2].unique_tag = self.realization_cdm2.halos[idx2_2].unique_tag

        new_realization = self.realization_cdm.join(self.realization_cdm2)
        npt.assert_equal(len(new_realization.halos), length-2)

        original_x_list = np.append(self.realization_cdm.x, self.realization_cdm2.x)
        original_m_list = np.append(self.realization_cdm.masses, self.realization_cdm2.masses)

        for (xi, mi) in zip(original_x_list, original_m_list):
            if xi not in new_realization.x:
                npt.assert_equal(True, mi not in new_realization.masses)
            if mi not in new_realization.masses:
                npt.assert_equal(True, xi not in new_realization.x)

    def test_shift_background_to_source(self):

        dmax = self.halo_mass_function.geometry._cosmo.D_C_transverse(2.)
        z = np.linspace(0, dmax, 100)
        ray_interp_x, ray_interp_y = interp1d(z, np.ones_like(z)), interp1d(z, -np.ones_like(z))
        realization_shifted = self.realization_cdm.shift_background_to_source(ray_interp_x, ray_interp_y)

        for halo, halo_0 in zip(realization_shifted.halos, self.realization_cdm.halos):
            npt.assert_equal(halo.x, halo_0.x + 1)
            npt.assert_equal(halo.y, halo_0.y - 1)

        # make sure you can only shift realizations once
        realization_shifted = realization_shifted.shift_background_to_source(ray_interp_x, ray_interp_y)
        for halo, halo_0 in zip(realization_shifted.halos, self.realization_cdm.halos):
            npt.assert_equal(halo.x, halo_0.x + 1)
            npt.assert_equal(halo.y, halo_0.y - 1)

    def test_filter(self):

        masses = [10 ** 9.04, 10 ** 8.2, 10 ** 7.345, 10 ** 7, 10 ** 6.05, 10 ** 7, 10 ** 7]
        x = [0.5, 0.1, -0.9, -1.4, 1.2, 0., -1.]
        y = [0., 0.9, -1.4, -1., -0.4, 1., 0.9]
        redshifts = [0.1, 0.2, 0.3, 0.4, 0.4, 0.94, 0.94]
        subflags = [False, False, False, True, True, False, False]
        r3d = [100] * len(masses)
        mdefs_TNFW = ['TNFW'] * len(masses)

        realization_cdm = Realization(masses, x, y, r3d, mdefs_TNFW, redshifts, subflags,
                                      self.halo_mass_function, halos=None, halo_profile_args=self.kwargs_cdm,
                                      mass_sheet_correction=True, rendering_classes=self.rendering_classes)

        r = [self.halo_mass_function.geometry._cosmo.D_C_transverse(zi) for zi in self.redshifts]

        x_intercepts_1 = [0.5, 0.1, -0.9, -1.4, 1.2, 0., -1.]
        y_intercepts_1 = [0., 0.9, -1.4, -1., -0.4, 1., 0.9]
        x_intercepts_2 = [1000] * len(x_intercepts_1)
        y_intercepts_2 = [1000] * len(y_intercepts_1)
        ray_interp_x1 = interp1d(r, x_intercepts_1)
        ray_interp_y1 = interp1d(r, y_intercepts_1)
        ray_interp_x2 = interp1d(r, x_intercepts_2)
        ray_interp_y2 = interp1d(r, y_intercepts_2)

        aperture_radius_front = 0.3
        aperture_radius_back = 0.3
        log_mass_allowed_in_aperture_front = 6
        log_mass_allowed_in_aperture_back = 6
        log_mass_allowed_global_front = 6
        log_mass_allowed_global_back = 6
        interpolated_x_angle = [ray_interp_x1, ray_interp_x2]
        interpolated_y_angle = [ray_interp_y1, ray_interp_y2]

        realization_filtered = realization_cdm.filter(aperture_radius_front,
                   aperture_radius_back,
                   log_mass_allowed_in_aperture_front,
                   log_mass_allowed_in_aperture_back,
                   log_mass_allowed_global_front,
                   log_mass_allowed_global_back,
                   interpolated_x_angle, interpolated_y_angle)
        npt.assert_equal(realization_filtered == realization_cdm, True)

        x_intercepts_1 = [1000] * len(x_intercepts_1)
        y_intercepts_1 = [1000] * len(x_intercepts_1)
        x_intercepts_2 = [1000] * len(x_intercepts_1)
        y_intercepts_2 = [1000] * len(y_intercepts_1)
        ray_interp_x1 = interp1d(r, x_intercepts_1)
        ray_interp_y1 = interp1d(r, y_intercepts_1)
        ray_interp_x2 = interp1d(r, x_intercepts_2)
        ray_interp_y2 = interp1d(r, y_intercepts_2)

        aperture_radius_front = 0.3
        aperture_radius_back = 0.3
        log_mass_allowed_in_aperture_front = 6
        log_mass_allowed_in_aperture_back = 6
        log_mass_allowed_global_front = 10
        log_mass_allowed_global_back = 10
        interpolated_x_angle = [ray_interp_x1, ray_interp_x2]
        interpolated_y_angle = [ray_interp_y1, ray_interp_y2]

        realization_filtered = realization_cdm.filter(aperture_radius_front,
                                                      aperture_radius_back,
                                                      log_mass_allowed_in_aperture_front,
                                                      log_mass_allowed_in_aperture_back,
                                                      log_mass_allowed_global_front,
                                                      log_mass_allowed_global_back,
                                                      interpolated_x_angle, interpolated_y_angle)
        npt.assert_equal(len(realization_filtered.halos), 0)

        x_intercepts_1 = [1000] * len(x_intercepts_1)
        y_intercepts_1 = [1000] * len(x_intercepts_1)
        x_intercepts_2 = [0.5, 0.1, -0.9, -1.4, 1.2, 0., -1.]
        y_intercepts_2 = [0., 0.9, -1.4, -1., -0.4, 1., 0.9]
        offset = 0.25
        idx_offset = 4
        tag = realization_cdm.halos[idx_offset].unique_tag
        x_intercepts_2[idx_offset] += offset

        offset = 0.35
        idx_offset = 2
        tag2 = realization_cdm.halos[idx_offset].unique_tag
        y_intercepts_2[idx_offset] += offset

        ray_interp_x1 = interp1d(r, x_intercepts_1)
        ray_interp_y1 = interp1d(r, y_intercepts_1)
        ray_interp_x2 = interp1d(r, x_intercepts_2)
        ray_interp_y2 = interp1d(r, y_intercepts_2)

        aperture_radius_front = 0.3
        aperture_radius_back = 0.3
        log_mass_allowed_in_aperture_front = 6.
        log_mass_allowed_in_aperture_back = 6
        log_mass_allowed_global_front = 10
        log_mass_allowed_global_back = 10
        interpolated_x_angle = [ray_interp_x1, ray_interp_x2]
        interpolated_y_angle = [ray_interp_y1, ray_interp_y2]

        realization_filtered = realization_cdm.filter(aperture_radius_front,
                                                      aperture_radius_back,
                                                      log_mass_allowed_in_aperture_front,
                                                      log_mass_allowed_in_aperture_back,
                                                      log_mass_allowed_global_front,
                                                      log_mass_allowed_global_back,
                                                      interpolated_x_angle, interpolated_y_angle)
        new_tags = realization_filtered._tags()
        npt.assert_equal(True, tag in new_tags)
        npt.assert_equal(True, tag2 not in new_tags)

    def test_lensing_quantities(self):

        lens_model_list, redshift_array, kwargs_lens, kwargs_lensmodel = \
            self.realization_cdm.lensing_quantities(add_mass_sheet_correction=True)
        npt.assert_equal(True, 'CONVERGENCE' in lens_model_list)

        lens_model_list, redshift_array, kwargs_lens, kwargs_lensmodel = \
            self.realization_cdm.lensing_quantities(add_mass_sheet_correction=False)
        npt.assert_equal(True, 'CONVERGENCE' not in lens_model_list)
        npt.assert_equal(len(lens_model_list), len(self.x))

        lens_model_list, redshift_array, kwargs_lens, kwargs_lensmodel = \
            self.realization_cdm2.lensing_quantities(add_mass_sheet_correction=True)
        npt.assert_equal(True, 'CONVERGENCE' not in lens_model_list)
        npt.assert_equal(len(lens_model_list), len(self.x))

    def test_split_at_z(self):

        real_before, real_after = self.realization_cdm.split_at_z(0.4)
        halos_before = real_before.halos
        halos_after = real_after.halos
        npt.assert_equal(len(halos_before), np.sum(np.array(self.redshifts) <= 0.4))
        npt.assert_equal(len(halos_after), np.sum(np.array(self.redshifts) > 0.4))

if __name__ == '__main__':
    pytest.main()




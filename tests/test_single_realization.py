from pyHalo.single_realization import Realization, SingleHalo, realization_at_z
from lenstronomy.LensModel.lens_model import LensModel
import numpy as np
import numpy.testing as npt
from scipy.interpolate import interp1d
from copy import deepcopy
import pytest
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce, ConcentrationPeakHeight
from pyHalo.Halos.tidal_truncation import TruncationRN
from pyHalo.preset_models import CDM
from pyHalo.Halos.lens_cosmo import LensCosmo
import matplotlib.pyplot as plt


class TestSingleRealization(object):

    def setup_method(self):
        z_lens = 0.5
        z_source = 1.5
        self._logmlow = 6.0
        self._logmhigh = 9.0
        self.realization = CDM(z_lens, z_source, log_mlow=self._logmlow, log_mhigh=self._logmhigh, sigma_sub=0.05,
                                   cone_opening_angle_arcsec=12.0)
        self.rendering_classes = self.realization.rendering_classes


    def test_mass_sheet_correction(self):

        kwargs_mass_sheets, profile_list, z_sheets = self.realization._mass_sheet_correction(self.rendering_classes,
                                                                                             subtract_exact_sheets=False,
                                                                                             kappa_scale=1.0,
                                                                                             log_mlow_sheets=self._logmlow,
                                                                                             log_mhigh_sheets=self._logmhigh)

        npt.assert_equal(len(kwargs_mass_sheets), len(profile_list))
        npt.assert_equal(len(kwargs_mass_sheets), len(z_sheets))

        profile_list, lens_redshift_list, kwargs_halos, _ = self.realization.lensing_quantities()
        lens_model = LensModel(profile_list, lens_redshift_list=list(lens_redshift_list),
                               z_source=self.realization.lens_cosmo.z_source,
                               multi_plane=True, cosmo=self.realization.lens_cosmo.cosmo.astropy)
        _r = np.linspace(-2.5, 2.5, 100)
        xx, yy = np.meshgrid(_r, _r)
        kappa = lens_model.kappa(xx.ravel(), yy.ravel(), kwargs_halos)
        mean_kappa = np.mean(kappa)
        npt.assert_array_less(abs(mean_kappa), 0.06)

        kwargs_mass_sheets, profile_list, z_sheets = self.realization._mass_sheet_correction(self.rendering_classes,
                                                                                             subtract_exact_sheets=True,
                                                                                             kappa_scale=1.0,
                                                                                             log_mlow_sheets=self._logmlow,
                                                                                             log_mhigh_sheets=self._logmhigh)

        npt.assert_equal(len(kwargs_mass_sheets), len(profile_list))
        npt.assert_equal(len(kwargs_mass_sheets), len(z_sheets))
        profile_list, lens_redshift_list, kwargs_halos, _ = self.realization.lensing_quantities()
        lens_model = LensModel(profile_list, lens_redshift_list=list(lens_redshift_list),
                               z_source=self.realization.lens_cosmo.z_source,
                               multi_plane=True, cosmo=self.realization.lens_cosmo.cosmo.astropy)
        _r = np.linspace(-2.5, 2.5, 100)
        xx, yy = np.meshgrid(_r, _r)
        kappa = lens_model.kappa(xx.ravel(), yy.ravel(), kwargs_halos)
        mean_kappa = np.mean(kappa)
        npt.assert_array_less(abs(mean_kappa), 0.04)

    def test_build_from_halos(self):

        realization_fromhalos = Realization.from_halos(self.realization.halos, self.realization.lens_cosmo,
                                   self.realization.kwargs_halo_model,
                                                       self.realization.apply_mass_sheet_correction,
                                                       self.realization.rendering_classes)

        for halo_1, halo_2 in zip(realization_fromhalos.halos, self.realization.halos):
            npt.assert_equal(halo_1.x, halo_2.x)
            npt.assert_equal(halo_1.y, halo_2.y)
            npt.assert_equal(halo_1.mass, halo_2.mass)
            npt.assert_equal(halo_1.r3d, halo_2.r3d)
            npt.assert_string_equal(halo_1.mdef, halo_2.mdef)
            npt.assert_equal(halo_1.z, halo_2.z)
            npt.assert_equal(halo_1.is_subhalo, halo_2.is_subhalo)
            npt.assert_equal(halo_1.unique_tag, halo_2.unique_tag)

        npt.assert_equal(True, realization_fromhalos == self.realization)
        npt.assert_equal(realization_fromhalos.apply_mass_sheet_correction,
                         self.realization.apply_mass_sheet_correction)

    def test_join(self):

        z_lens = 0.5
        z_source = 1.5
        realization_2 = CDM(z_lens, z_source, log_mlow=self._logmlow, log_mhigh=self._logmhigh, sigma_sub=0.05,
                                 cone_opening_angle_arcsec=12.0, LOS_normalization=0.0)
        new_realization = self.realization.join(realization_2, join_rendering_classes=True)
        npt.assert_equal(len(new_realization.rendering_classes), 6)

        length = len(realization_2.halos) + len(self.realization.halos)
        npt.assert_equal(len(new_realization.halos), length)
        # now change two of the tags to be the same as in realization_cdm, make sure they're gone

        idx2_1 = 0
        idx2_2 = 1
        idx1 = 0
        idx2 = 1
        self.realization.halos[idx1].unique_tag = realization_2.halos[idx2_1].unique_tag
        self.realization.halos[idx2].unique_tag = realization_2.halos[idx2_2].unique_tag
        new_realization = self.realization.join(realization_2)
        npt.assert_equal(len(new_realization.halos), length-2)

        original_x_list = np.append(self.realization.x, realization_2.x)
        original_m_list = np.append(self.realization.masses, realization_2.masses)

        for (xi, mi) in zip(original_x_list, original_m_list):
            if xi not in new_realization.x:
                npt.assert_equal(True, mi not in new_realization.masses)
            if mi not in new_realization.masses:
                npt.assert_equal(True, xi not in new_realization.x)

    def test_shift_background_to_source(self):

        dmax = self.realization.geometry.cosmo.D_C_transverse(self.realization.lens_cosmo.z_source)
        z = np.linspace(0, dmax, 100)
        ray_interp_x, ray_interp_y = interp1d(z, np.ones_like(z)), interp1d(z, -np.ones_like(z))

        centerx_interp, centery_interp = self.realization.rendering_center
        npt.assert_almost_equal(centerx_interp(1000), 0.)
        npt.assert_almost_equal(centery_interp(1000), 0.)

        model_name, _, kwargs_lens_init, _ = self.realization.lensing_quantities()
        for name, kw in zip(model_name, kwargs_lens_init):
            if name == 'CONVERGENCE':
                npt.assert_almost_equal(kw['ra_0'], 0.)
                npt.assert_almost_equal(kw['dec_0'], 0.)

        test_realization = deepcopy(self.realization)
        realization_shifted = test_realization.shift_background_to_source(ray_interp_x, ray_interp_y)

        for halo, halo_0 in zip(realization_shifted.halos, self.realization.halos):

            npt.assert_equal(halo.x, halo_0.x + 1)
            npt.assert_equal(halo.y, halo_0.y - 1)

        model_names, redshifts_lens, kwargs_lens_init, _ = self.realization.lensing_quantities()
        centerx, centerey = self.realization.rendering_center
        for kw, zi, name in zip(kwargs_lens_init, redshifts_lens, model_names):
            if name == 'CONVERGENCE':
                di = self.realization.lens_cosmo.cosmo.D_C_z(zi)
                npt.assert_almost_equal(kw['ra_0'], centerx(di))
                npt.assert_almost_equal(kw['dec_0'], centerey(di))

        # make sure you can only shift realizations once
        realization_shifted = realization_shifted.shift_background_to_source(ray_interp_x, ray_interp_y)
        for halo, halo_0 in zip(realization_shifted.halos, self.realization.halos):
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
                                      self.realization.lens_cosmo, halos=None, kwargs_halo_model=self.realization.kwargs_halo_model,
                                      mass_sheet_correction=True, rendering_classes=self.realization.rendering_classes)

        r = [self.realization.geometry.cosmo.D_C_transverse(zi) for zi in realization_cdm.redshifts]

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

        realization_filtered = realization_cdm.filter(aperture_radius_front,
                                                      aperture_radius_back,
                                                      log_mass_allowed_in_aperture_front,
                                                      log_mass_allowed_in_aperture_back,
                                                      log_mass_allowed_global_front,
                                                      log_mass_allowed_global_back,
                                                      interpolated_x_angle, interpolated_y_angle, aperture_units='MPC')

        npt.assert_equal(len(realization_filtered.halos) > 0, True)

        args = (aperture_radius_front,aperture_radius_back, log_mass_allowed_in_aperture_front,
                        log_mass_allowed_in_aperture_back, log_mass_allowed_global_front,
                        log_mass_allowed_global_back, interpolated_x_angle,
                interpolated_y_angle, None, None, 'nonsense')

        npt.assert_raises(Exception, realization_cdm.filter, *args)

    def test_lensing_quantities(self):

        lens_model_list, redshift_array, kwargs_lens, kwargs_lensmodel = \
            self.realization.lensing_quantities(add_mass_sheet_correction=True)
        npt.assert_equal(True, 'CONVERGENCE' in lens_model_list)

        lens_model_list, redshift_array, kwargs_lens, kwargs_lensmodel = \
            self.realization.lensing_quantities(add_mass_sheet_correction=False)
        npt.assert_equal(True, 'CONVERGENCE' not in lens_model_list)
        npt.assert_equal(len(lens_model_list), len(self.realization.x))

        lens_model_list, redshift_array, kwargs_lens, kwargs_lensmodel = \
            self.realization.lensing_quantities(add_mass_sheet_correction=True)
        npt.assert_equal(True, 'CONVERGENCE' in lens_model_list)

    def test_split_at_z(self):

        real_before, real_after = self.realization.split_at_z(0.4)
        halos_before = real_before.halos
        halos_after = real_after.halos
        npt.assert_equal(len(halos_before), np.sum(np.array(self.realization.redshifts) <= 0.4))
        npt.assert_equal(len(halos_after), np.sum(np.array(self.realization.redshifts) > 0.4))

    def test_single_halo(self):

        kwargs_halo_model = {'concentration_model': None, 'truncation_model': None, 'kwargs_density_profile': {}}
        single_halo = SingleHalo(10**8, 0.5, 1.0, 'PT_MASS', 0.5, 0.5, 1.5, r3d=None, subhalo_flag=False,
                                 kwargs_halo_model=kwargs_halo_model)
        lens_model_list, redshift_array, kwargs_lens, kwargs_lensmodel = single_halo.lensing_quantities()
        npt.assert_equal(len(lens_model_list), 1)
        npt.assert_string_equal(lens_model_list[0], 'POINT_MASS')

    def test_realization_at_z(self):

        realatz, inds = realization_at_z(self.realization, 0.5, 0., 0., 1)
        for halo in realatz.halos:
            npt.assert_equal(True, halo.z == 0.5)
            npt.assert_array_less(np.hypot(halo.x, halo.y), 1.00000000001)

    def test_comoving_coordinates(self):

        x, y, logm, z = self.realization.halo_comoving_coordinates()
        d1 = self.realization.lens_cosmo.cosmo.D_C_z(z[0])
        npt.assert_almost_equal(x[0], self.realization.x[0]*d1)
        npt.assert_almost_equal(y[0], self.realization.y[0] * d1)
        npt.assert_almost_equal(10**logm[0], self.realization.masses[0])

    def test_assign_concentration_models(self):

        m = 10 ** 8
        x = 0.5
        y = 0.1
        mdef = 'TNFW'
        z = 0.5
        zlens = 0.5
        zsource = 1.5
        r3d = None

        lens_cosmo = LensCosmo(zlens, zsource)
        kwargs_halo_model_1 = {'concentration_model': ConcentrationDiemerJoyce(lens_cosmo.cosmo.astropy),
                               'truncation_model': TruncationRN(lens_cosmo), 'kwargs_density_profile': {}}
        subhalo_flag = True
        single_subhalo_diemer_joyce = SingleHalo(10**7, x, y, mdef, z, zlens, zsource, r3d, subhalo_flag,
                                   kwargs_halo_model_1, lens_cosmo=lens_cosmo)
        subhalo_flag = False
        single_fieldhalo_diemer_joyce = SingleHalo(10**8, x, y, mdef, z, zlens, zsource, r3d, subhalo_flag,
                                                 kwargs_halo_model_1, lens_cosmo=lens_cosmo)

        kwargs_halo_model_2 = {'concentration_model': ConcentrationPeakHeight(lens_cosmo.cosmo.astropy, 15.0, -0.1, -0.8),
                             'truncation_model': TruncationRN(lens_cosmo), 'kwargs_density_profile': {}}
        subhalo_flag = True
        single_subhalo_peak_height = SingleHalo(10**9, x, y, mdef, z, zlens, zsource, r3d, subhalo_flag,
                                                 kwargs_halo_model_2, lens_cosmo=lens_cosmo)
        subhalo_flag = False
        single_fieldhalo_peak_height = SingleHalo(10**10, x, y, mdef, z, zlens, zsource, r3d, subhalo_flag,
                                                 kwargs_halo_model_2, lens_cosmo=lens_cosmo)

        realization = single_subhalo_diemer_joyce.join(single_fieldhalo_diemer_joyce).join(single_subhalo_peak_height).\
            join(single_fieldhalo_peak_height)

        subhalo_diemer_joyce = realization.halos[3]
        fieldhalo_diemer_joyce = realization.halos[2]
        subhalo_peak_height = realization.halos[1]
        fieldhalo_peak_height = realization.halos[0]

        npt.assert_string_equal('DIEMERJOYCE19', subhalo_diemer_joyce._concentration_class.name)
        npt.assert_string_equal('DIEMERJOYCE19', fieldhalo_diemer_joyce._concentration_class.name)
        npt.assert_string_equal('PEAK_HEIGHT_POWERLAW', subhalo_peak_height._concentration_class.name)
        npt.assert_string_equal('PEAK_HEIGHT_POWERLAW', fieldhalo_peak_height._concentration_class.name)
        npt.assert_string_equal('DIEMERJOYCE19', kwargs_halo_model_1['concentration_model'].name)
        npt.assert_string_equal('PEAK_HEIGHT_POWERLAW', kwargs_halo_model_2['concentration_model'].name)

        kwargs_halo_model = {'concentration_model_subhalos': ConcentrationDiemerJoyce(lens_cosmo.cosmo.astropy),
                               'concentration_model_field_halos': ConcentrationPeakHeight(lens_cosmo.cosmo.astropy, 15.0, -0.1, -0.8),
                               'truncation_model_subhalos': TruncationRN(lens_cosmo),
                             'truncation_model_field_halos': TruncationRN(lens_cosmo), 'kwargs_density_profile': {}}
        subhalo_flag = True
        single_subhalo_diemer_joyce = SingleHalo(10 ** 7, x, y, mdef, z, zlens, zsource, r3d, subhalo_flag,
                                                 kwargs_halo_model, lens_cosmo=lens_cosmo)
        subhalo_flag = False
        single_fieldhalo_peak_height = SingleHalo(10 ** 8, x, y, mdef, z, zlens, zsource, r3d, subhalo_flag,
                                                   kwargs_halo_model, lens_cosmo=lens_cosmo)



        realization = single_subhalo_diemer_joyce.join(single_fieldhalo_peak_height)

        subhalo_diemer_joyce = realization.halos[1]
        fieldhalo_peak_height = realization.halos[0]

        npt.assert_string_equal('DIEMERJOYCE19', subhalo_diemer_joyce._concentration_class.name)
        npt.assert_string_equal('PEAK_HEIGHT_POWERLAW', fieldhalo_peak_height._concentration_class.name)
        npt.assert_string_equal('DIEMERJOYCE19', kwargs_halo_model['concentration_model_subhalos'].name)
        npt.assert_string_equal('PEAK_HEIGHT_POWERLAW', kwargs_halo_model['concentration_model_field_halos'].name)

    def test_mass_at_z_exact(self):

        astropy = self.realization.lens_cosmo.cosmo.astropy
        concentration_model = ConcentrationDiemerJoyce(astropy)
        halo_z = 0.3
        z_lens = 0.5
        realization = SingleHalo(10**8, 0.5, 1.0, 'NFW', halo_z, z_lens, 1.5, r3d=None, subhalo_flag=False,
                 kwargs_halo_model={'concentration_model': concentration_model,
                                    'truncation_model': None,
                                    'kwargs_density_profile': {}})
        npt.assert_equal(realization.halos[0].mass, realization.mass_at_z_exact(halo_z))
        npt.assert_equal(0.0, realization.mass_at_z_exact(z_lens))

    def test_number_of_halos_before_after_z(self):

        astropy = self.realization.lens_cosmo.cosmo.astropy
        concentration_model = ConcentrationDiemerJoyce(astropy)
        halo_z = 0.3
        z_lens = 0.5
        halo_1 = SingleHalo(10 ** 8, 0.5, 1.0, 'NFW', halo_z, z_lens, 1.5, r3d=None, subhalo_flag=False,
                                 kwargs_halo_model={'concentration_model': concentration_model,
                                                    'truncation_model': None,
                                                    'kwargs_density_profile': {}})
        halo_z = 0.3
        halo_2 = SingleHalo(10 ** 8, 0.5, 1.0, 'NFW', halo_z, z_lens, 1.5, r3d=None, subhalo_flag=False,
                            kwargs_halo_model={'concentration_model': concentration_model,
                                               'truncation_model': None,
                                               'kwargs_density_profile': {}})
        halo_z = 0.7
        halo_3 = SingleHalo(10 ** 8, 0.5, 1.0, 'NFW', halo_z, z_lens, 1.5, r3d=None, subhalo_flag=False,
                            kwargs_halo_model={'concentration_model': concentration_model,
                                               'truncation_model': None,
                                               'kwargs_density_profile': {}})
        realization = halo_1.join(halo_2).join(halo_3)
        npt.assert_equal(realization.number_of_halos_after_redshift(0.1), 3)
        npt.assert_equal(realization.number_of_halos_after_redshift(z_lens), 1)
        npt.assert_equal(realization.number_of_halos_before_redshift(2.0), 3)
        npt.assert_equal(realization.number_of_halos_at_redshift(0.7), 1)

    def test_plot(self):

        fig = plt.figure(1)
        ax1 = plt.subplot(111, projection='3d')
        self.realization.plot(ax1)

if __name__ == '__main__':
    pytest.main()


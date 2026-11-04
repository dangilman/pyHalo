import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LensModel.lens_model import LensModel
from pyHalo.Rendering.convergence_sheet_rescaling import ConvergenceSheetRescaling, \
    ConvergenceSheetRescalingShethTormen
from pyHalo.PresetModels.cdm import CDM, CDMBinned


class TestConvergenceSheetRescalingShethTormen(object):

    def setup_method(self):

        self.zlens = 0.5
        self.zsource = 1.5
        self.rescaling = ConvergenceSheetRescalingShethTormen(self.zlens, self.zsource, log_mlow=6.0, log_mhigh=10.0,
                                                              cone_opening_angle_arcsec=6.0)

    def test_call(self):

        npt.assert_array_almost_equal(self.rescaling(self.rescaling.redshifts), self.rescaling.rescaling_factor)
        npt.assert_equal(len(self.rescaling.redshifts), len(self.rescaling.rescaling_factor))
        npt.assert_equal(np.all(self.rescaling.rescaling_factor > 0), True)

    def test_limits(self):

        # near the main deflector the rendering area contains most of each halo, and truncated halos contain more
        # mass than M_200; close to the observer and the source halos are large compared to the rendering area
        npt.assert_array_less(1.0, self.rescaling(self.zlens))
        npt.assert_array_less(self.rescaling(0.01), 0.5)
        npt.assert_array_less(self.rescaling(self.zsource - 0.02), 1.0)

    def test_normalization(self):

        # with one population the amplitude of the mass function cancels; with several it sets their relative weight
        rescaling = ConvergenceSheetRescalingShethTormen(self.zlens, self.zsource, log_mlow=6.0, log_mhigh=10.0,
                                                         LOS_normalization=2.0)
        npt.assert_array_almost_equal(rescaling.rescaling_factor, self.rescaling.rescaling_factor)
        rescaling_split = ConvergenceSheetRescalingShethTormen(self.zlens, self.zsource, log_mlow=[6.0, 8.0],
                                                               log_mhigh=[8.0, 10.0], LOS_normalization=[1.0, 1.0])
        npt.assert_array_almost_equal(rescaling_split.rescaling_factor, self.rescaling.rescaling_factor, 2)
        rescaling_weighted = ConvergenceSheetRescalingShethTormen(self.zlens, self.zsource, log_mlow=[6.0, 8.0],
                                                                  log_mhigh=[8.0, 10.0], LOS_normalization=[1.0, 5.0])
        # more weight on massive halos, which extend further outside the rendering area
        npt.assert_array_less(rescaling_weighted(0.1), rescaling_split(0.1))

    def test_realization_sheets(self):

        realization = CDM(self.zlens, self.zsource, sigma_sub=0.0, log_mlow=6.0, log_mhigh=10.0,
                          cone_opening_angle_arcsec=6.0, two_halo_contribution=False)
        kwargs_mass_sheet = {'log_mlow_sheets': 6.0, 'log_mhigh_sheets': 10.0}
        lens_model_list, redshifts, kwargs_lens, _ = realization.lensing_quantities(
            kwargs_mass_sheet=kwargs_mass_sheet)
        lens_model_list_scaled, redshifts_scaled, kwargs_lens_scaled, _ = realization.lensing_quantities(
            kwargs_mass_sheet=dict(kwargs_mass_sheet, kappa_scale=self.rescaling))
        for name, z, kw, kw_scaled in zip(lens_model_list, redshifts, kwargs_lens, kwargs_lens_scaled):
            if name == 'CONVERGENCE':
                npt.assert_almost_equal(kw_scaled['kappa'] / kw['kappa'], self.rescaling(z))
                npt.assert_equal(kw_scaled['ra_0'], kw['ra_0'])
                npt.assert_equal(kw_scaled['dec_0'], kw['dec_0'])

        rescaling = ConvergenceSheetRescaling.from_realization(realization)
        npt.assert_array_almost_equal(rescaling(self.rescaling.redshifts), self.rescaling.rescaling_factor)

        _, _, kwargs_lens_from_realization, _ = realization.lensing_quantities(
            kwargs_mass_sheet=dict(kwargs_mass_sheet, kappa_scale='FROM_REALIZATION'))
        for kw_scaled, kw_from_realization in zip(kwargs_lens_scaled, kwargs_lens_from_realization):
            if 'kappa' in kw_scaled:
                npt.assert_almost_equal(kw_from_realization['kappa'], kw_scaled['kappa'])
        # computed once for each mass range and stored with the realization
        cached = realization._kappa_scale_from_realization[(6.0, 10.0)]
        realization.lensing_quantities(kwargs_mass_sheet=dict(kwargs_mass_sheet, kappa_scale='FROM_REALIZATION'))
        npt.assert_equal(realization._kappa_scale_from_realization[(6.0, 10.0)] is cached, True)
        # the rescaling follows the mass range of the sheets, not the rendering range
        kwargs_mass_sheet_7 = {'log_mlow_sheets': 7.0, 'log_mhigh_sheets': 10.0}
        names_7, redshifts_7, kwargs_lens_7, _ = realization.lensing_quantities(kwargs_mass_sheet=kwargs_mass_sheet_7)
        _, _, kwargs_lens_7_scaled, _ = realization.lensing_quantities(
            kwargs_mass_sheet=dict(kwargs_mass_sheet_7, kappa_scale='FROM_REALIZATION'))
        rescaling_7 = ConvergenceSheetRescalingShethTormen(self.zlens, self.zsource, log_mlow=7.0, log_mhigh=10.0,
                                                           cone_opening_angle_arcsec=6.0)
        for name, z, kw, kw_scaled in zip(names_7, redshifts_7, kwargs_lens_7, kwargs_lens_7_scaled):
            if name == 'CONVERGENCE':
                npt.assert_almost_equal(kw_scaled['kappa'] / kw['kappa'], rescaling_7(z))
        npt.assert_equal(len(realization._kappa_scale_from_realization), 2)
        npt.assert_raises(Exception, realization.lensing_quantities,
                          kwargs_mass_sheet=dict(kwargs_mass_sheet, kappa_scale='NOT_AN_OPTION'))


class TestConvergenceSheetRescalingFromRealization(object):

    def setup_method(self):

        self.zlens = 0.5
        self.zsource = 1.5
        kwargs_binned = {'log10_sigma_sub': -1.0, 'log10_mc_low': 1.5, 'log10_mc_high': 1.0,
                         'scale_mass_function_low': 2.0, 'scale_mass_function_high': 0.5, 'beta': 0.8, 'zeta': -0.2}
        self.realization = CDMBinned(self.zlens, self.zsource, **kwargs_binned)
        self.rescaling = ConvergenceSheetRescaling.from_realization(self.realization)

    def test_binned_model(self):

        los_classes = [rc for rc in self.realization.rendering_classes if rc.name == 'LINE_OF_SIGHT']
        rescaling = ConvergenceSheetRescalingShethTormen(
            self.zlens, self.zsource, log_mlow=[6.0, 8.0], log_mhigh=[8.0, 10.7],
            LOS_normalization=[rc._kwargs_mass_function['LOS_normalization'] for rc in los_classes],
            concentration_model=self.realization.kwargs_halo_model['concentration_model_field_halos'])
        npt.assert_array_almost_equal(rescaling.rescaling_factor, self.rescaling.rescaling_factor)

    def test_realization_sheets(self):

        # each line-of-sight population subtracts its own mass range and is rescaled by the same factor
        lens_model_list, redshifts, kwargs_lens, _ = self.realization.lensing_quantities(use_class_mass_ranges=True)
        _, _, kwargs_lens_scaled, _ = self.realization.lensing_quantities(
            use_class_mass_ranges=True, kwargs_mass_sheet={'kappa_scale': self.rescaling})
        n_subhalo_sheets = 0
        for name, z, kw, kw_scaled in zip(lens_model_list, redshifts, kwargs_lens, kwargs_lens_scaled):
            if name != 'CONVERGENCE':
                continue
            ratio = kw_scaled['kappa'] / kw['kappa']
            if z == self.zlens and ratio == 1.0:
                n_subhalo_sheets += 1
            else:
                npt.assert_almost_equal(ratio, self.rescaling(z))
        npt.assert_equal(n_subhalo_sheets, 2)

        _, _, kwargs_lens_from_realization, _ = self.realization.lensing_quantities(
            use_class_mass_ranges=True, kwargs_mass_sheet={'kappa_scale': 'FROM_REALIZATION'})
        for kw_scaled, kw_from_realization in zip(kwargs_lens_scaled, kwargs_lens_from_realization):
            if 'kappa' in kw_scaled:
                npt.assert_almost_equal(kw_from_realization['kappa'], kw_scaled['kappa'])


class TestMassSheetSubtraction(object):
    """
    Generates many realizations of line-of-sight halos, computes the convergence of the halos plus the negative
    convergence sheets near the center of the rendering area, and checks that the rescaled sheets
    (kappa_scale='FROM_REALIZATION') remove the mean convergence of the halos, while the default sheets subtract
    too much mass.
    """

    def setup_method(self):

        self.zlens = 0.5
        self.zsource = 2.0
        self.kwargs_model = {'sigma_sub': 0.0, 'two_halo_contribution': False, 'log_mlow': 7.0, 'log_mhigh': 10.0,
                             'cone_opening_angle_arcsec': 6.0}
        self.kwargs_mass_sheet = {'log_mlow_sheets': 7.0, 'log_mhigh_sheets': 10.0}
        # points distributed uniformly inside a circle of radius 1 arcsec (the region where lensed images appear)
        r = np.sqrt((np.arange(6) + 0.5) / 6)
        theta = 2 * np.pi * (np.arange(12) + 0.5) / 12
        self.x_aperture = (r[:, None] * np.cos(theta)[None, :]).ravel()
        self.y_aperture = (r[:, None] * np.sin(theta)[None, :]).ravel()

    def _convergence(self, realization, kwargs_mass_sheet):
        """
        Total line-of-sight convergence of halos plus sheets, summed over lens planes (Born approximation), averaged
        inside the image region; at each plane the region scales with the rendering area as light rays converge
        towards the source
        """
        lens_model_list, redshifts, kwargs_lens, _ = realization.lensing_quantities(
            kwargs_mass_sheet=kwargs_mass_sheet)
        kappa = 0.0
        for z in np.unique(redshifts):
            index = np.where(redshifts == z)[0]
            scale = realization.geometry.rendering_scale(z)
            lens_model = LensModel([lens_model_list[i] for i in index])
            kappa += np.mean(lens_model.kappa(scale * self.x_aperture, scale * self.y_aperture,
                                              [kwargs_lens[i] for i in index]))
        return kappa

    def test_rescaled_sheets_subtract_mean_convergence(self):

        np.random.seed(3)
        n_realizations = 100
        kappa_default, kappa_rescaled = [], []
        for _ in range(n_realizations):
            realization = CDM(self.zlens, self.zsource, **self.kwargs_model)
            kappa_default.append(self._convergence(realization, self.kwargs_mass_sheet))
            kappa_rescaled.append(self._convergence(realization,
                                                    dict(self.kwargs_mass_sheet, kappa_scale='FROM_REALIZATION')))
        mean_default, mean_rescaled = np.mean(kappa_default), np.mean(kappa_rescaled)
        # the sheets are the same in every realization, so the scatter comes from the halos
        standard_error = np.std(kappa_default, ddof=1) / np.sqrt(n_realizations)

        # the default sheets subtract more mass than the halos deposit near the center of the rendering area
        npt.assert_array_less(mean_default, -3 * standard_error)
        # the rescaled sheets remove the mean convergence of the halos
        npt.assert_array_less(abs(mean_rescaled), 3 * standard_error)
        npt.assert_array_less(abs(mean_rescaled), abs(mean_default))


if __name__ == '__main__':
    pytest.main()

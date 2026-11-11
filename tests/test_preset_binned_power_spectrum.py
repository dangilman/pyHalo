import pytest
import numpy as np
import numpy.testing as npt
from unittest import mock
from scipy.optimize import least_squares
from pyHalo.preset_models import preset_model_from_name
from pyHalo.PresetModels.cdm import CDM
from pyHalo.PresetModels import power_spectrum as preset_module
from pyHalo.PresetModels.power_spectrum import BinnedPowerSpectrum, sigma_sub_rescaling_factor
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.binned_power_spectrum import binned_linear_power_spectrum, binned_power_spectrum_nodes, \
    binned_power_spectrum_ratio
from pyHalo.Halos.concentration import ConcentrationLudlowPowerSpectrum, ConcentrationLudlowWDM
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen
from pyHalo.Rendering.MassFunctions.power_spectrum import PowerSpectrumMassFunction, \
    PowerSpectrumSubhaloMassFunction
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw, WDMPowerLaw
from pyHalo.Rendering.MassFunctions.benson import benson_suppression_params
from pyHalo.Cosmology.binned_power_spectrum import lcdm_power_spectrum
from pyHalo.Cosmology.power_spectrum import LinearPowerSpectrum


class TestBinnedPowerSpectrumPreset(object):

    def setup_method(self):

        self.z_lens, self.z_source = 0.5, 1.5
        self.kwargs = {'log_mlow': 7.0, 'log_mhigh': 10.0, 'cone_opening_angle_arcsec': 5.0}

    @staticmethod
    def _masses(realization):
        return np.array([halo.mass for halo in realization.halos])

    def test_models(self):

        amplitudes = (1.2, 0.8, 1.4, 1.0)
        with mock.patch.object(PowerSpectrumMassFunction, 'from_redshift',
                               wraps=PowerSpectrumMassFunction.from_redshift) as from_redshift:
            np.random.seed(1)
            realization = BinnedPowerSpectrum(self.z_lens, self.z_source, pk_amplitudes=amplitudes, **self.kwargs)
        # the field halo and two-halo mass functions come from the binned power spectrum at every lens plane
        npt.assert_array_less(10, from_redshift.call_count)
        kwargs_model = from_redshift.call_args[0][3]
        npt.assert_equal(kwargs_model['multiplicity_function'], 'BENSON2026')
        power_spectrum = kwargs_model['power_spectrum']
        npt.assert_equal(any(halo.is_subhalo for halo in realization.halos), True)
        for halo in realization.halos:
            npt.assert_equal(isinstance(halo._concentration_class, ConcentrationLudlowPowerSpectrum), True)
            npt.assert_equal(halo._concentration_class._power_spectrum is power_spectrum, True)
        masses = self._masses(realization)
        npt.assert_equal(np.all((masses >= 10 ** 7) & (masses <= 10 ** 10)), True)
        # the power spectrum is the binned model
        expected = binned_linear_power_spectrum(Cosmology(), amplitudes)
        m = np.logspace(6, 10, 5)
        npt.assert_allclose(power_spectrum.sigma(m), expected.sigma(m), rtol=1e-12)
        names, _, _, _ = realization.lensing_quantities(kwargs_mass_sheet={'log_mlow_sheets': 7.0,
                                                                          'log_mhigh_sheets': 10.0})
        npt.assert_array_less(0, names.count('CONVERGENCE'))

    def test_per_bin_amplitudes(self):

        np.random.seed(4)
        r1 = BinnedPowerSpectrum(self.z_lens, self.z_source, pk_amplitudes=[1.3, 0.9, 1.1], **self.kwargs)
        np.random.seed(4)
        r2 = BinnedPowerSpectrum(self.z_lens, self.z_source, pk_amplitude_1=1.3, pk_amplitude_2=0.9,
                                 pk_amplitude_3=1.1, **self.kwargs)
        npt.assert_equal(self._masses(r1), self._masses(r2))
        # the concentrations draw their scatter when first accessed
        np.random.seed(5)
        c1 = [halo.c for halo in r1.halos]
        np.random.seed(5)
        c2 = [halo.c for halo in r2.halos]
        npt.assert_equal(c1, c2)

    def test_concentration_responds(self):

        # more small-scale power: halos form earlier and are more concentrated
        medians = []
        for amplitude in [1.0, 1.5]:
            np.random.seed(2)
            realization = BinnedPowerSpectrum(self.z_lens, self.z_source, pk_amplitudes=[amplitude] * 4,
                                              **self.kwargs)
            halo = [halo for halo in realization.halos if not halo.is_subhalo][0]
            medians.append(halo._concentration_class.nfw_concentration(np.array([1e7, 1e8, 1e9]), 0.5,
                                                                       force_no_scatter=True))
        npt.assert_array_less(1.2 * medians[0], medians[1])

    def test_lcdm_sheth_tormen(self):

        # with LCDM amplitudes and the Sheth-Tormen multiplicity function the field halo abundance matches the CDM
        # preset; without Poisson sampling the subhalos are identical
        kwargs = dict(self.kwargs, draw_poisson=False)
        np.random.seed(3)
        r_cdm = CDM(self.z_lens, self.z_source, **kwargs)
        np.random.seed(3)
        r_pk = BinnedPowerSpectrum(self.z_lens, self.z_source, pk_amplitudes=[1.0] * 4,
                                   multiplicity_function='SHETH_TORMEN', **kwargs)
        n_sub = [sum(halo.is_subhalo for halo in r.halos) for r in [r_cdm, r_pk]]
        n_field = [sum(not halo.is_subhalo for halo in r.halos) for r in [r_cdm, r_pk]]
        npt.assert_equal(n_sub[0], n_sub[1])
        npt.assert_allclose(n_field[1], n_field[0], rtol=0.02)
        # the Benson et al. (2026) mass function lies below Sheth-Tormen
        np.random.seed(3)
        r_benson = BinnedPowerSpectrum(self.z_lens, self.z_source, pk_amplitudes=[1.0] * 4, **kwargs)
        n_benson = sum(not halo.is_subhalo for halo in r_benson.halos)
        npt.assert_array_less(n_benson, 0.95 * n_field[0])

    def test_concentration_model_fixed(self):

        # the concentration-mass relation is always Ludlow et al. (2016) from the binned power spectrum
        args = (self.z_lens, self.z_source)
        for key, value in [('concentration_model_subhalos', 'DIEMERJOYCE19'),
                           ('concentration_model_fieldhalos', 'DIEMERJOYCE19'),
                           ('kwargs_concentration_model_subhalos', {}),
                           ('kwargs_concentration_model_fieldhalos', {})]:
            npt.assert_raises(TypeError, BinnedPowerSpectrum, *args, pk_amplitudes=[1.2] * 4,
                              **dict(self.kwargs, **{key: value}))

    def test_cdm_field_halo_mass_function(self):

        # the CDM preset is unchanged by default, and accepts another field halo mass function
        np.random.seed(6)
        r1 = CDM(self.z_lens, self.z_source, **self.kwargs)
        np.random.seed(6)
        r2 = CDM(self.z_lens, self.z_source, mass_function_model_fieldhalos=ShethTormen, **self.kwargs)
        npt.assert_equal(self._masses(r1), self._masses(r2))

    def test_sigma_sub_rescaling(self):

        # sigma_sub_effective = sigma_sub * A_los / A_los_cdm, with the amplitudes of the field halo mass function at
        # 10^8 M_sun evaluated at z_lens + shmf_z_eval_offset
        amplitudes = [1.3, 1.3, 1.3, 1.3]
        cosmo = Cosmology()
        power_spectrum = binned_linear_power_spectrum(cosmo, amplitudes)
        power_spectrum_lcdm = LinearPowerSpectrum(*lcdm_power_spectrum(cosmo), cosmo)
        z_eval = self.z_lens + 3.3
        a_los = PowerSpectrumMassFunction.dndm_comoving(np.array([1e8]), z_eval, {'power_spectrum': power_spectrum})
        a_los_cdm = PowerSpectrumMassFunction.dndm_comoving(np.array([1e8]), z_eval,
                                                            {'power_spectrum': power_spectrum_lcdm})
        factor = float(a_los[0] / a_los_cdm[0])
        npt.assert_allclose(sigma_sub_rescaling_factor(power_spectrum, power_spectrum_lcdm, z_eval), factor, rtol=1e-12)
        npt.assert_array_less(1.1, factor)
        npt.assert_equal(sigma_sub_rescaling_factor(power_spectrum_lcdm, power_spectrum_lcdm, z_eval), 1.0)
        with mock.patch.object(preset_module, 'CDM', wraps=CDM) as cdm:
            _ = BinnedPowerSpectrum(self.z_lens, self.z_source, pk_amplitudes=amplitudes, sigma_sub=0.04,
                                    **self.kwargs)
            kwargs_cdm = cdm.call_args[1]
            npt.assert_allclose(kwargs_cdm['sigma_sub'], 0.04 * factor, rtol=1e-6)
            kwargs_shmf = kwargs_cdm['kwargs_mass_function_subhalos']
            npt.assert_equal(kwargs_cdm['mass_function_model_subhalos'] is PowerSpectrumSubhaloMassFunction, True)
            npt.assert_allclose(kwargs_shmf['z_eval'], z_eval)
            _ = BinnedPowerSpectrum(self.z_lens, self.z_source, pk_amplitudes=amplitudes, log10_sigma_sub=-1.5,
                                    log10_dNdA=-1.2, shmf_z_eval_offset=2.0, **self.kwargs)
            kwargs_cdm = cdm.call_args[1]
            factor_2 = sigma_sub_rescaling_factor(power_spectrum, power_spectrum_lcdm, self.z_lens + 2.0)
            npt.assert_allclose(kwargs_cdm['log10_sigma_sub'], -1.5 + np.log10(factor_2), rtol=1e-6)
            npt.assert_allclose(kwargs_cdm['log10_dNdA'], -1.2 + np.log10(factor_2), rtol=1e-6)
            _ = BinnedPowerSpectrum(self.z_lens, self.z_source, pk_amplitudes=[1.0] * 4, sigma_sub=0.04,
                                    **self.kwargs)
            npt.assert_equal(cdm.call_args[1]['sigma_sub'], 0.04)

    def test_subhalos_respond(self):

        # the number of subhalos follows sigma_sub * m^-1.9 * R(m, z_eval) integrated over the rendered mass range
        amplitudes = [1.0, 1.0, 1.5, 0.8]
        kwargs = dict(self.kwargs, draw_poisson=False, sigma_sub=0.1)
        n_sub = []
        for amps in [[1.0] * 4, amplitudes]:
            np.random.seed(7)
            realization = BinnedPowerSpectrum(self.z_lens, self.z_source, pk_amplitudes=amps, **kwargs)
            n_sub.append(sum(halo.is_subhalo for halo in realization.halos))
        cosmo = Cosmology()
        m = np.logspace(7, 10, 3001)
        ratio = PowerSpectrumSubhaloMassFunction.mass_function_ratio(
            m, self.z_lens + 3.3, binned_linear_power_spectrum(cosmo, amplitudes),
            LinearPowerSpectrum(*lcdm_power_spectrum(cosmo), cosmo))
        expected = np.trapezoid(m ** -0.9 * ratio, np.log(m)) / np.trapezoid(m ** -0.9, np.log(m))
        npt.assert_allclose(n_sub[1] / n_sub[0], expected, rtol=0.02)
        npt.assert_array_less(0.05, abs(expected - 1))

    def test_preset_model_from_name(self):

        npt.assert_equal(preset_model_from_name('BinnedPowerSpectrum') is BinnedPowerSpectrum, True)

    def test_exceptions(self):

        args = (self.z_lens, self.z_source)
        npt.assert_raises(ValueError, BinnedPowerSpectrum, *args, **self.kwargs)
        npt.assert_raises(ValueError, BinnedPowerSpectrum, *args, pk_amplitudes=[1.0], pk_amplitude_1=1.0)
        npt.assert_raises(ValueError, BinnedPowerSpectrum, *args, pk_amplitude_1=1.0, pk_amplitude_3=1.0)
        npt.assert_raises(ValueError, BinnedPowerSpectrum, *args, pk_amplitudes=[1.0],
                          multiplicity_function='PRESS_SCHECHTER')
        npt.assert_raises(TypeError, BinnedPowerSpectrum, *args, pk_amplitudes=[1.0],
                          mass_function_model_fieldhalos=ShethTormen)
        npt.assert_raises(TypeError, BinnedPowerSpectrum, *args, pk_amplitudes=[1.0],
                          mass_function_model_subhalos=CDMPowerLaw)
        npt.assert_raises(TypeError, BinnedPowerSpectrum, *args, pk_amplitudes=[1.0], not_a_keyword=1.0)


class TestBinnedPowerSpectrumWDM(object):
    """
    A BinnedPowerSpectrum fit to a WDM power spectrum reproduces the field halo and subhalo mass functions and the
    concentration-mass relation of the WDMGeneral preset with the same half-mode mass and dlogT/dlogk, each relative
    to the LCDM model of its own preset
    """

    def setup_method(self):

        self.cosmo = Cosmology()
        self.power_spectrum_lcdm = LinearPowerSpectrum(*lcdm_power_spectrum(self.cosmo), self.cosmo)
        self.z_lens = 0.5
        self.z_eval = self.z_lens + 3.3  # the default shmf_z_eval_offset of both presets
        self.m = np.logspace(6, 10, 41)
        self.c_lcdm = ConcentrationLudlowPowerSpectrum(self.cosmo.astropy, self.power_spectrum_lcdm,
                                                       scatter=False).nfw_concentration(self.m, self.z_lens)
        # the bins have to extend to k ~ 2000 / Mpc, because the binned power spectrum returns to LCDM above k_max
        # and the concentrations of 10^6 M_sun halos depend on P(k) out to k ~ 1000 / Mpc; four bins over this
        # range miss sharp cutoffs (|dlogT/dlogk| > 2) by up to 25%, six bins reproduce every case to within 15%
        self.n_bins, self.k_min, self.k_max = 6, 5.0, 2000.0

    def _wdm_ratio(self, k, log_mc, dlogT_dlogk, gamma=5.0):
        """
        P_WDM / P_LCDM = T(k)^2, with T = [1 + f (k / k_hm)^beta]^-gamma, T(k_hm) = 1/2, the logarithmic slope
        dlogT_dlogk at k_hm, and the half-mode mass m_hm = (4 pi / 3) rho_m (pi / k_hm)^3
        """
        k_hm = np.pi * (3 * 10 ** log_mc / (4 * np.pi * self.power_spectrum_lcdm.rho_m0)) ** (-1. / 3)
        f = 0.5 ** (-1 / gamma) - 1
        beta = -dlogT_dlogk * (1 + f) / (f * gamma)
        return (1 + f * (k / k_hm) ** beta) ** (-2 * gamma)

    def _fit_amplitudes(self, log_mc, dlogT_dlogk):
        """
        Least-squares fit of the bin amplitudes to the transfer function sqrt(P / P_LCDM), with every amplitude
        <= 1 so P <= P_LCDM everywhere
        """
        k = np.logspace(np.log10(self.k_min), np.log10(self.k_max), 400)
        _, centers = binned_power_spectrum_nodes(self.n_bins, self.k_min, self.k_max)
        target = np.sqrt(self._wdm_ratio(k, log_mc, dlogT_dlogk))
        a0 = np.clip(1 + np.log10(np.maximum(self._wdm_ratio(centers, log_mc, dlogT_dlogk), 1e-4)), -20, 0.999)
        fit = least_squares(lambda a: np.sqrt(binned_power_spectrum_ratio(k, a, self.k_min, self.k_max)) - target,
                            a0, bounds=(-20, 1))
        return list(fit.x)

    @pytest.mark.parametrize('log_mc', [7.0, 8.0, 9.0])
    @pytest.mark.parametrize('dlogT_dlogk', [-1.44, -2.5, -4.0])
    def test_reproduces_wdm(self, log_mc, dlogT_dlogk):

        amplitudes = self._fit_amplitudes(log_mc, dlogT_dlogk)
        npt.assert_array_less(np.array(amplitudes), 1.0 + 1e-10)
        power_spectrum = binned_linear_power_spectrum(self.cosmo, amplitudes, self.k_min, self.k_max)
        # WDMGeneral: the Benson et al. (2026) turnover of the field halo mass function at each lens plane and of the
        # subhalo mass function at z_eval, and the LUDLOW_WDM concentrations
        wdm = {'field': WDMPowerLaw._turnover(self.m, log_mc, *benson_suppression_params(dlogT_dlogk, self.z_lens)),
               'subhalos': WDMPowerLaw._turnover(self.m, log_mc,
                                                 *benson_suppression_params(dlogT_dlogk, self.z_eval)),
               'concentration': ConcentrationLudlowWDM(self.cosmo.astropy, log_mc, dlogT_dlogk, scatter=False
                                                       ).nfw_concentration(self.m, self.z_lens) / self.c_lcdm}
        # BinnedPowerSpectrum: the Benson et al. (2026) mass function and Ludlow et al. (2016) concentrations
        # computed from P(k)
        binned = {'field': PowerSpectrumSubhaloMassFunction.mass_function_ratio(
                      self.m, self.z_lens, power_spectrum, self.power_spectrum_lcdm),
                  'subhalos': PowerSpectrumSubhaloMassFunction.mass_function_ratio(
                      self.m, self.z_eval, power_spectrum, self.power_spectrum_lcdm),
                  'concentration': ConcentrationLudlowPowerSpectrum(self.cosmo.astropy, power_spectrum, scatter=False
                                                                    ).nfw_concentration(self.m, self.z_lens) /
                                   self.c_lcdm}
        # compare where WDMGeneral suppresses the field halo mass function by less than a factor of 10; below that
        # its fitting functions extrapolate past the regime where they were calibrated
        ok = wdm['field'] > 0.1
        npt.assert_array_less(5, np.sum(ok))
        npt.assert_array_less(wdm['field'][ok].min(), 0.5)
        for key in wdm:
            npt.assert_allclose(binned[key][ok], wdm[key][ok], rtol=0.2, err_msg=key)


if __name__ == '__main__':
    pytest.main()

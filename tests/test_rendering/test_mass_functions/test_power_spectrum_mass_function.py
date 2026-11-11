import pytest
import numpy as np
import numpy.testing as npt
from colossus.lss import mass_function
from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.power_spectrum import LinearPowerSpectrum
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Halos.tidal_truncation import TruncationRN
from pyHalo.Halos.concentration import ConcentrationLudlowPowerSpectrum
from pyHalo.Rendering.MassFunctions.power_spectrum import PowerSpectrumMassFunction, multiplicity_function, \
    MULTIPLICITY_FUNCTION_PARAMS
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen, evaluate_mass_function
from pyHalo.Rendering.MassFunctions.benson import benson_suppression_params
from pyHalo.Rendering.SpatialDistributions.uniform import LensConeUniform
from pyHalo.Rendering.line_of_sight import LineOfSight
from pyHalo.Rendering.convergence_sheet_rescaling import ConvergenceSheetRescaling
from pyHalo.mass_function_models import preset_mass_function_models
from pyHalo.utilities import generate_lens_plane_redshifts


def lcdm_power_spectrum(cosmo, sigma8=None):
    """
    The power spectrum colossus uses for the Sheth-Tormen mass function (Eisenstein & Hu 1998), in pyHalo units
    """
    k = np.logspace(-5, 5, 2000)
    pk = cosmo.colossus.matterPowerSpectrum(k / cosmo.h, 0.0) / cosmo.h ** 3
    return k, pk, LinearPowerSpectrum(k, pk, cosmo, sigma8=sigma8)


def wdm_transfer_function(k, m_hm, dlogT_dlogk, power_spectrum, gamma=5.0):
    """
    T(k) = [1 + (alpha k)^beta]^-gamma with T(k_hm) = 1/2 and logarithmic slope dlogT_dlogk at k_hm, where
    m_hm = (4 pi / 3) rho_m (pi / k_hm)^3 (as in notebooks/calibrate_benson_hmf.ipynb)
    """
    k_half = np.pi / power_spectrum.lagrangian_radius(m_hm)
    f = -1 + 0.5 ** (-1 / gamma)
    beta = -dlogT_dlogk * (1 + f) / f / gamma
    alpha = (2 ** (1.0 / gamma) - 1) ** (1.0 / beta) / k_half
    return (1.0 + (alpha * k) ** beta) ** -gamma


class TestShethTormenFromPowerSpectrum(object):

    def setup_method(self):

        self.cosmo = Cosmology()
        self.h = self.cosmo.h
        self.geometry = Geometry(self.cosmo, 0.5, 2.0, 6.0, 'DOUBLE_CONE')
        _, _, self.power_spectrum = lcdm_power_spectrum(self.cosmo)
        _, self.kwargs_model = preset_mass_function_models('SHETH_TORMEN_PK',
                                                           {'m_pivot': 1e8, 'LOS_normalization': 1.0,
                                                            'delta_power_law_index': 0.0, 'draw_poisson': False,
                                                            'power_spectrum': self.power_spectrum})

    def test_exact_sheth_tormen(self):

        # the LCDM power spectrum with a top-hat window reproduces the Sheth-Tormen mass function from colossus
        for z in [0.2, 1.0, 1.9]:
            kwargs_model = dict(self.kwargs_model, log_mlow=6.0, log_mhigh=10.0)
            mfunc = PowerSpectrumMassFunction.from_redshift(z, 0.02, self.geometry, kwargs_model)
            volume = self.geometry.volume_element_comoving(z, 0.02)
            m = np.logspace(6.0, 10.0, 4001)
            dN_dm = evaluate_mass_function(m, self.h, z, 'sheth99') * volume
            npt.assert_allclose(mfunc.evaluate(m), dN_dm, rtol=5e-3)
            npt.assert_allclose(mfunc.n_mean, np.trapezoid(dN_dm * m, np.log(m)), rtol=2e-3)
            npt.assert_allclose(mfunc.first_moment, np.trapezoid(dN_dm * m ** 2, np.log(m)), rtol=2e-3)

    def test_sheth_tormen_class(self):

        # ShethTormen fits a power law to the same mass function in each lens plane
        for log_mlow, log_mhigh in [(6.0, 10.0), (6.0, 8.0), (8.0, 10.0)]:
            for z in [0.2, 1.0, 1.9]:
                kwargs_model = dict(self.kwargs_model, log_mlow=log_mlow, log_mhigh=log_mhigh)
                mfunc = PowerSpectrumMassFunction.from_redshift(z, 0.02, self.geometry, kwargs_model)
                mfunc_st = ShethTormen.from_redshift(z, 0.02, self.geometry, kwargs_model)
                npt.assert_allclose(mfunc.n_mean, mfunc_st.n_mean, rtol=1.5e-2)
                npt.assert_allclose(mfunc.first_moment, mfunc_st.first_moment, rtol=1.5e-2)
                m = np.logspace(log_mlow, log_mhigh, 20)
                npt.assert_allclose(mfunc.evaluate(m), mfunc_st.evaluate(m), rtol=3e-2)

    def test_normalization(self):

        z = 0.6
        kwargs_model = dict(self.kwargs_model, log_mlow=6.0, log_mhigh=10.0)
        mfunc = PowerSpectrumMassFunction.from_redshift(z, 0.02, self.geometry, kwargs_model)
        kwargs_model_scaled = dict(kwargs_model, LOS_normalization=1.5, delta_power_law_index=-0.1)
        mfunc_scaled = PowerSpectrumMassFunction.from_redshift(z, 0.02, self.geometry, kwargs_model_scaled)
        mfunc_st = ShethTormen.from_redshift(z, 0.02, self.geometry, kwargs_model)
        mfunc_st_scaled = ShethTormen.from_redshift(z, 0.02, self.geometry, kwargs_model_scaled)
        m = np.logspace(6, 10, 9)
        npt.assert_allclose(mfunc_scaled.evaluate(m) / mfunc.evaluate(m), 1.5 * (m / 1e8) ** -0.1, rtol=1e-10)
        npt.assert_allclose(mfunc_scaled.evaluate(m) / mfunc.evaluate(m),
                            mfunc_st_scaled.evaluate(m) / mfunc_st.evaluate(m), rtol=1e-10)

    def test_convergence_sheets(self):

        lens_cosmo = LensCosmo(0.5, 2.0, self.cosmo)
        lens_plane_redshifts, delta_z_list = generate_lens_plane_redshifts(0.5, 2.0)
        spatial_distribution = LensConeUniform(self.geometry.cone_opening_angle, self.geometry)
        kappa = []
        for model, kwargs_model in [(ShethTormen, self.kwargs_model),
                                    (PowerSpectrumMassFunction, self.kwargs_model)]:
            kwargs_model = dict(kwargs_model, log_mlow=6.0, log_mhigh=10.0)
            los = LineOfSight(model, kwargs_model, spatial_distribution, self.geometry, lens_cosmo,
                              lens_plane_redshifts, delta_z_list)
            kwargs_out, _, _ = los.convergence_sheet_correction(1.0, 6.0, 10.0)
            kappa.append(np.array([kw['kappa'] for kw in kwargs_out]))
        npt.assert_allclose(kappa[1], kappa[0], rtol=1e-2)

    def test_empty_range(self):

        kwargs_model = dict(self.kwargs_model, log_mlow=8.0, log_mhigh=8.0)
        mfunc = PowerSpectrumMassFunction.from_redshift(0.5, 0.02, self.geometry, kwargs_model)
        npt.assert_equal(mfunc.first_moment, 0)
        npt.assert_equal(len(mfunc.draw()), 0)

    def test_cosmology_mismatch(self):

        geometry = Geometry(Cosmology(cosmo_kwargs={'H0': 70.0}), 0.5, 2.0, 6.0, 'DOUBLE_CONE')
        kwargs_model = dict(self.kwargs_model, log_mlow=6.0, log_mhigh=10.0)
        npt.assert_raises(ValueError, PowerSpectrumMassFunction.from_redshift, 0.5, 0.02, geometry, kwargs_model)
        _ = Cosmology()

    def test_presets(self):

        model, kwargs_out = preset_mass_function_models('BENSON_PK', {'power_spectrum': self.power_spectrum})
        npt.assert_equal(model is PowerSpectrumMassFunction, True)
        npt.assert_equal(kwargs_out['multiplicity_function'], 'BENSON2026')
        npt.assert_equal(kwargs_out['power_spectrum'] is self.power_spectrum, True)
        npt.assert_equal(self.kwargs_model['multiplicity_function'], 'SHETH_TORMEN')
        npt.assert_raises(Exception, preset_mass_function_models, 'BENSON_PK', {})
        npt.assert_raises(Exception, preset_mass_function_models, 'BENSON_PK',
                          {'power_spectrum': self.power_spectrum, 'multiplicity_function': 'SHETH_TORMEN'})
        kwargs_model = dict(self.kwargs_model, log_mlow=6.0, log_mhigh=10.0, kwargs_multiplicity={'x': 1.0})
        npt.assert_raises(ValueError, PowerSpectrumMassFunction.from_redshift, 0.5, 0.02, self.geometry,
                          kwargs_model)

    def test_multiplicity_function(self):

        # with b = c = q = 1, Equation 8 of Benson et al. (2026) is the Sheth-Tormen multiplicity function
        sigma = np.linspace(0.5, 5.0, 10)
        nu_p = 0.707 * (1.68647 / sigma) ** 2
        f_st = 0.3222 * np.sqrt(2 * nu_p / np.pi) * np.exp(-0.5 * nu_p) * (1 + nu_p ** -0.3)
        npt.assert_allclose(multiplicity_function(sigma, **MULTIPLICITY_FUNCTION_PARAMS['SHETH_TORMEN']), f_st,
                            rtol=1e-12)


class TestBensonFromPowerSpectrum(object):

    def setup_method(self):

        # the cosmology of notebooks/calibrate_benson_hmf.ipynb (planck18 in colossus)
        self.cosmo = Cosmology(cosmo_kwargs={'H0': 67.66, 'Om0': 0.3111, 'Ob0': 0.049, 'sigma8': 0.8102,
                                             'ns': 0.9665})
        self.h = self.cosmo.h
        self.k, self.pk, self.power_spectrum = lcdm_power_spectrum(self.cosmo, sigma8=0.8102)
        self.kwargs_model = {'power_spectrum': self.power_spectrum}

    def teardown_method(self):
        # restore the default colossus cosmology for the tests that follow
        _ = Cosmology()

    def test_calibration_notebook(self):

        # Benson / Sheth-Tormen at z = 0 printed in section 5 of the calibration notebook, at 10^8 - 10^14 M_sun/h
        m_h = 10 ** np.arange(8.0, 14.1, 1.0)
        ratio_notebook = np.array([0.839, 0.833, 0.832, 0.838, 0.859, 0.908, 0.977])
        dndlnm = PowerSpectrumMassFunction.dndm_comoving(m_h / self.h, 0.0, self.kwargs_model) * m_h / self.h
        dndlnm_st = mass_function.massFunction(m_h, 0.0, mdef='fof', model='sheth99', q_out='dndlnM') * self.h ** 3
        npt.assert_allclose(dndlnm / dndlnm_st, ratio_notebook, atol=2e-3)

    def test_suppression(self):

        # with a suppressed power spectrum the ratio to CDM matches the turnover calibrated to the same model and
        # installed in pyHalo.Rendering.MassFunctions.benson, within the accuracy of that fit
        m_hm = 10 ** 7.5
        u = np.linspace(-1.0, 3.0, 41)
        m = m_hm * 10 ** u
        for dlogT_dlogk, z in [(-1.0, 0.5), (-2.0, 0.0), (-3.0, 2.0)]:
            transfer = wdm_transfer_function(self.k, m_hm, dlogT_dlogk, self.power_spectrum)
            power_spectrum_wdm = LinearPowerSpectrum(self.k, self.pk * transfer ** 2, self.cosmo, sigma8=0.8102)
            ratio = PowerSpectrumMassFunction.dndm_comoving(m, z, {'power_spectrum': power_spectrum_wdm}) / \
                PowerSpectrumMassFunction.dndm_comoving(m, z, self.kwargs_model)
            a, b, c = benson_suppression_params(dlogT_dlogk, z)
            npt.assert_allclose(ratio, (1 + a * (m_hm / m) ** b) ** c, atol=1.5e-2)

    def test_render(self):

        # an end-to-end realization of line-of-sight halos with the mass function and concentration-mass relation
        # computed from the power spectrum
        z_lens, z_source = 0.5, 1.5
        pyhalo = pyHalo(z_lens, z_source, {'H0': 67.66, 'Om0': 0.3111, 'Ob0': 0.049, 'sigma8': 0.8102,
                                          'ns': 0.9665})
        geometry = Geometry(pyhalo.cosmology, z_lens, z_source, 6.0, 'DOUBLE_CONE')
        model, kwargs_los = preset_mass_function_models('BENSON_PK', {'log_mlow': 7.0, 'log_mhigh': 10.0,
                                                                      'LOS_normalization': 1.0,
                                                                      'draw_poisson': True,
                                                                      'power_spectrum': self.power_spectrum})
        concentration = ConcentrationLudlowPowerSpectrum(pyhalo.astropy_cosmo, self.power_spectrum)
        truncation = TruncationRN(pyhalo.lens_cosmo, 50)
        kwargs_halo_model = {'truncation_model_subhalos': truncation, 'concentration_model_subhalos': concentration,
                             'truncation_model_field_halos': truncation,
                             'concentration_model_field_halos': concentration, 'kwargs_density_profile': {}}
        kwargs_spatial = {'cone_opening_angle': 6.0, 'geometry': geometry}
        np.random.seed(5)
        realization = pyhalo.render(['LINE_OF_SIGHT'], [model], [kwargs_los], [LensConeUniform], [kwargs_spatial],
                                    geometry, 'TNFW', 'TNFW', kwargs_halo_model, nrealizations=1)[0]
        npt.assert_array_less(0, len(realization.halos))
        masses = np.array([halo.mass for halo in realization.halos])
        npt.assert_equal(np.all((masses >= 1e7) & (masses <= 1e10)), True)
        _ = [halo.c for halo in realization.halos]
        names, redshifts, kwargs_lens, _ = realization.lensing_quantities(
            kwargs_mass_sheet={'log_mlow_sheets': 7.0, 'log_mhigh_sheets': 10.0})
        npt.assert_array_less(0, names.count('CONVERGENCE'))
        rescaling = ConvergenceSheetRescaling.from_realization(realization)
        npt.assert_equal(np.all(np.isfinite(rescaling.rescaling_factor)), True)
        npt.assert_equal(np.all(rescaling.rescaling_factor > 0), True)


if __name__ == '__main__':
    pytest.main()

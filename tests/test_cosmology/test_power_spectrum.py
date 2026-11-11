import pytest
import warnings
from copy import deepcopy
import numpy as np
import numpy.testing as npt
from colossus.lss import peaks
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.power_spectrum import LinearPowerSpectrum


class TestLinearPowerSpectrum(object):

    def setup_method(self):

        self.cosmo = Cosmology()
        self.colossus = self.cosmo.colossus
        self.h = self.cosmo.h
        # the LCDM power spectrum colossus uses for the Sheth-Tormen mass function, in pyHalo units
        self.k = np.logspace(-5, 5, 2000)
        self.pk = self.colossus.matterPowerSpectrum(self.k / self.h, 0.0) / self.h ** 3
        self.ps = LinearPowerSpectrum(self.k, self.pk, self.cosmo)
        self.m = np.logspace(4, 15, 23)
        self.radius_h = peaks.lagrangianR(self.m * self.h)

    def test_sigma_tophat(self):

        for z in [0.0, 1.0, 3.0]:
            npt.assert_allclose(self.ps.sigma(self.m, z), self.colossus.sigma(self.radius_h, z), rtol=1e-4)
        npt.assert_allclose(self.ps.sigma8_tophat, self.colossus.sigma8, rtol=1e-3)
        npt.assert_equal(isinstance(self.ps.sigma(1e8, 0.5), float), True)

    def test_dlnsigma_dlnm(self):

        # colossus differentiates a spline of its own sigma table, which is accurate to ~0.2%
        dlnsigma_colossus = self.colossus.sigma(self.radius_h, 0.0, derivative=True) / 3
        npt.assert_allclose(self.ps.dlnsigma_dlnm(self.m), dlnsigma_colossus, rtol=3e-3)
        for filter in ['TOPHAT', 'BENSON2026']:
            eps = 1e-3
            finite_difference = (np.log(self.ps.sigma(self.m * (1 + eps), 0.0, filter)) -
                                 np.log(self.ps.sigma(self.m * (1 - eps), 0.0, filter))) / \
                                (np.log1p(eps) - np.log1p(-eps))
            npt.assert_allclose(self.ps.dlnsigma_dlnm(self.m, filter), finite_difference, rtol=2e-4)

    def test_growth_factor(self):

        z = np.array([0.0, 0.5, 2.0, 7.0])
        npt.assert_allclose(self.ps.growth_factor(z), self.colossus.growthFactor(z), rtol=1e-5)
        npt.assert_raises(ValueError, self.ps.growth_factor, 500.0)

    def test_power_spectrum(self):

        # interpolation onto the internal grid (spacing 0.023 in ln k) smooths the baryon wiggles at the 1e-4 level
        k = np.logspace(-3, 3, 20)
        npt.assert_allclose(self.ps.power_spectrum(k), self.colossus.matterPowerSpectrum(k / self.h, 0.0) /
                            self.h ** 3, rtol=1e-3)
        # beyond the input table the power spectrum continues as a power law
        k_high = np.array([1e6, 1e7])
        slope = np.log(self.pk[-1] / self.pk[-2]) / np.log(self.k[-1] / self.k[-2])
        npt.assert_allclose(self.ps.power_spectrum(k_high), self.pk[-1] * (k_high / self.k[-1]) ** slope, rtol=1e-3)

    def test_sigma8_normalization(self):

        ps = LinearPowerSpectrum(self.k, 3.0 * self.pk, self.cosmo, sigma8=0.7)
        npt.assert_allclose(ps.sigma8_tophat, 0.7, rtol=1e-8)
        npt.assert_allclose(ps.sigma(self.m), self.ps.sigma(self.m) * 0.7 / self.ps.sigma8_tophat, rtol=1e-8)

    def test_little_h_units(self):

        ps = LinearPowerSpectrum(self.k / self.h, self.pk * self.h ** 3, self.cosmo, little_h_units=True)
        npt.assert_allclose(ps.sigma(self.m, 0.5), self.ps.sigma(self.m, 0.5), rtol=1e-10)
        npt.assert_allclose(ps.sigma(self.m, 0.5, 'BENSON2026'), self.ps.sigma(self.m, 0.5, 'BENSON2026'), rtol=1e-10)

    def test_benson_window(self):

        # the logarithmic slope of the smoothed power spectrum approaches n_s on large scales, and c_W = c_W0 where the
        # slope equals n0 (see notebooks/calibrate_benson_hmf.ipynb)
        n = self.ps.smoothed_logarithmic_slope(0.686)
        k = self.ps._k
        npt.assert_almost_equal(np.interp(1e-5, k, n), 0.965, 1)
        c_w, beta = self.ps._benson_window_parameters()
        i = np.argmin(np.abs(n[(k > 1e-2) & (k < 1e4)] - (-2.6)))
        npt.assert_almost_equal(c_w[(k > 1e-2) & (k < 1e4)][i], 3.547, 2)
        # a smaller c_W moves the edge of the window, at k R ~ c_W, to larger scales and removes power
        ps = LinearPowerSpectrum(self.k, self.pk, self.cosmo, kwargs_benson_window={'cW0': 2.0})
        npt.assert_array_less(ps.sigma(self.m, 0.0, 'BENSON2026'), self.ps.sigma(self.m, 0.0, 'BENSON2026'))
        npt.assert_raises(ValueError, LinearPowerSpectrum, self.k, self.pk, self.cosmo,
                          kwargs_benson_window={'not_a_parameter': 1.0})

    def test_deepcopy(self):

        npt.assert_equal(deepcopy(self.ps) is self.ps, True)
        kwargs = deepcopy({'power_spectrum': self.ps})
        npt.assert_equal(kwargs['power_spectrum'] is self.ps, True)

    def test_check_cosmology(self):

        self.ps.check_cosmology(Cosmology())
        npt.assert_raises(ValueError, self.ps.check_cosmology, Cosmology(cosmo_kwargs={'H0': 70.0}))
        # restore the default colossus cosmology for the tests that follow
        _ = Cosmology()

    def test_extrapolation_warning(self):

        k = np.logspace(-4, 2, 500)
        pk = self.colossus.matterPowerSpectrum(k / self.h, 0.0) / self.h ** 3
        ps = LinearPowerSpectrum(k, pk, self.cosmo)
        with pytest.warns(UserWarning):
            ps.sigma(1e6)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            ps.sigma(1e13)

    def test_exceptions(self):

        npt.assert_raises(ValueError, self.ps.sigma, 1e1)
        npt.assert_raises(ValueError, self.ps.sigma, 1e18)
        npt.assert_raises(ValueError, self.ps.sigma, 1e8, 0.0, 'NOT_A_FILTER')
        npt.assert_raises(ValueError, LinearPowerSpectrum, self.k, -self.pk, self.cosmo)
        npt.assert_raises(ValueError, LinearPowerSpectrum, self.k[::-1], self.pk, self.cosmo)
        npt.assert_raises(ValueError, LinearPowerSpectrum, self.k, self.pk[1:], self.cosmo)


if __name__ == '__main__':
    pytest.main()

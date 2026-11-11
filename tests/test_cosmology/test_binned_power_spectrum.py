import pytest
import numpy as np
import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.power_spectrum import LinearPowerSpectrum
from pyHalo.Cosmology.binned_power_spectrum import binned_power_spectrum_nodes, binned_power_spectrum_ratio, \
    lcdm_power_spectrum, binned_linear_power_spectrum


class TestBinnedPowerSpectrum(object):

    def setup_method(self):

        self.k = np.logspace(-1, 3, 20001)
        self.amplitudes = [1.2, 0.7, 1.15, 0.85]

    def test_nodes(self):

        edges, centers = binned_power_spectrum_nodes(3, 5.0, 200.0)
        npt.assert_allclose(edges, [5.0, 17.0998, 58.4804, 200.0], rtol=1e-5)
        npt.assert_allclose(centers, np.sqrt(edges[1:] * edges[:-1]), rtol=1e-12)
        npt.assert_allclose(centers, [9.24656, 31.62278, 108.14838], rtol=1e-5)
        # appending a bin of the same width keeps the first four bins
        edges_4, _ = binned_power_spectrum_nodes(4, 5.0, 200.0)
        edges_5, _ = binned_power_spectrum_nodes(5, 5.0, 5.0 * 40 ** 1.25)
        npt.assert_allclose(edges_5[:5], edges_4, rtol=1e-12)

    def test_lcdm(self):

        npt.assert_equal(binned_power_spectrum_ratio(self.k, [1.0, 1.0, 1.0]), np.ones_like(self.k))
        npt.assert_equal(binned_power_spectrum_ratio(self.k, [1.0]), np.ones_like(self.k))

    def test_bin_centers(self):

        _, centers = binned_power_spectrum_nodes(4)
        npt.assert_allclose(binned_power_spectrum_ratio(centers, self.amplitudes),
                            10 ** (np.array(self.amplitudes) - 1), rtol=1e-12)
        npt.assert_equal(np.ndim(binned_power_spectrum_ratio(30.0, self.amplitudes)), 0)

    def test_outside_bins(self):

        ratio = binned_power_spectrum_ratio(self.k, self.amplitudes)
        outside = (self.k <= 5.0) | (self.k >= 200.0)
        npt.assert_equal(ratio[outside], 1.0)
        ratio = binned_power_spectrum_ratio(self.k, self.amplitudes, 2.0, 500.0)
        npt.assert_equal(ratio[(self.k <= 2.0) | (self.k >= 500.0)], 1.0)
        npt.assert_array_less(1.0001, ratio[np.argmin(np.abs(self.k - 3.0))])

    def test_no_overshoot(self):

        # equal amplitudes give a flat ratio between the first and last bin centers, and the ratio never leaves the
        # range spanned by the amplitudes and LCDM
        _, centers = binned_power_spectrum_nodes(4)
        inside = (self.k >= centers[0]) & (self.k <= centers[-1])
        npt.assert_allclose(binned_power_spectrum_ratio(self.k, [1.3] * 4)[inside], 10 ** 0.3, rtol=1e-12)
        log10_ratio = np.log10(binned_power_spectrum_ratio(self.k, self.amplitudes))
        npt.assert_array_less(np.min(np.array(self.amplitudes) - 1) - 1e-12, log10_ratio)
        npt.assert_array_less(log10_ratio, np.max(np.array(self.amplitudes) - 1) + 1e-12)

    def test_smooth(self):

        # the slope of the ratio is continuous at the bin centers, and zero where the ratio joins LCDM at k_min and k_max
        edges, centers = binned_power_spectrum_nodes(4)
        log10_ratio = lambda x: np.log10(binned_power_spectrum_ratio(10 ** x, self.amplitudes))
        h = 1e-6
        for x0 in np.log10(centers):
            slope_left = (log10_ratio(x0) - log10_ratio(x0 - h)) / h
            slope_right = (log10_ratio(x0 + h) - log10_ratio(x0)) / h
            npt.assert_allclose(slope_left, slope_right, atol=1e-3)
        for x0 in np.log10([edges[0], edges[-1]]):
            npt.assert_allclose([(log10_ratio(x0 + h) - log10_ratio(x0)) / h,
                                 (log10_ratio(x0) - log10_ratio(x0 - h)) / h], 0.0, atol=1e-3)
        # a kink would show up as a jump in the slope
        x = np.log10(self.k)
        slope = np.gradient(log10_ratio(x), x)
        npt.assert_array_less(np.max(np.abs(np.diff(slope))), 2e-2)

    def test_exceptions(self):

        npt.assert_raises(ValueError, binned_power_spectrum_ratio, self.k, [])
        npt.assert_raises(ValueError, binned_power_spectrum_ratio, self.k, [1.0, np.nan])
        npt.assert_raises(ValueError, binned_power_spectrum_ratio, self.k, [[1.0, 1.1]])
        npt.assert_raises(ValueError, binned_power_spectrum_ratio, self.k, [1.0], 200.0, 5.0)
        npt.assert_raises(ValueError, binned_power_spectrum_nodes, 0)


class TestBinnedLinearPowerSpectrum(object):

    def setup_method(self):

        self.cosmo = Cosmology()
        self.k, self.pk = lcdm_power_spectrum(self.cosmo)

    def test_lcdm_power_spectrum(self):

        h = self.cosmo.h
        npt.assert_allclose(self.pk, self.cosmo.colossus.matterPowerSpectrum(self.k / h, 0.0) / h ** 3, rtol=1e-12)
        k = np.logspace(0, 2, 5)
        npt.assert_equal(lcdm_power_spectrum(self.cosmo, k)[0] is k, True)

    def test_binned_linear_power_spectrum(self):

        m = np.logspace(6, 10, 9)
        ps_lcdm = LinearPowerSpectrum(self.k, self.pk, self.cosmo)
        ps = binned_linear_power_spectrum(self.cosmo, [1.0, 1.0, 1.0, 1.0])
        npt.assert_allclose(ps.sigma(m, 0.0, 'BENSON2026'), ps_lcdm.sigma(m, 0.0, 'BENSON2026'), rtol=1e-12)
        amplitudes = [1.2, 0.7, 1.15, 0.85]
        ps = binned_linear_power_spectrum(self.cosmo, amplitudes)
        _, centers = binned_power_spectrum_nodes(4)
        npt.assert_allclose(ps.power_spectrum(centers) / ps_lcdm.power_spectrum(centers),
                            10 ** (np.array(amplitudes) - 1), rtol=1e-3)
        # large scales, and so sigma8, are unchanged
        npt.assert_allclose(ps.sigma8_tophat, ps_lcdm.sigma8_tophat, rtol=1e-6)
        # keyword arguments reach LinearPowerSpectrum
        ps = binned_linear_power_spectrum(self.cosmo, amplitudes, log10m_min=4.0, log10m_max=12.0)
        npt.assert_raises(ValueError, ps.sigma, 1e3)


if __name__ == '__main__':
    pytest.main()

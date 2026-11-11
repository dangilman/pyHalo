import pytest
import numpy as np
import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.power_spectrum import LinearPowerSpectrum
from pyHalo.Halos.concentration import ConcentrationLudlow, ConcentrationLudlowPowerSpectrum, ConcentrationTabulated
from pyHalo.Rendering.convergence_sheet_rescaling import _MedianConcentration
from pyHalo.concentration_models import preset_concentration_models


class TestConcentrationLudlowPowerSpectrum(object):

    def setup_method(self):

        self.cosmo = Cosmology()
        self.astropy = self.cosmo.astropy
        h = self.cosmo.h
        # the LCDM power spectrum colossus uses for the ludlow16 model (Eisenstein & Hu 1998), in pyHalo units
        self.k = np.logspace(-5, 5, 2000)
        self.pk = self.cosmo.colossus.matterPowerSpectrum(self.k / h, 0.0) / h ** 3
        self.power_spectrum = LinearPowerSpectrum(self.k, self.pk, self.cosmo)
        self.reference = ConcentrationLudlow(self.astropy, scatter=False, use_interpolation_table=False)
        self.model = ConcentrationLudlowPowerSpectrum(self.astropy, self.power_spectrum, scatter=False)
        self.model_exact = ConcentrationLudlowPowerSpectrum(self.astropy, self.power_spectrum, scatter=False,
                                                            use_interpolation_table=False)
        self.m = np.logspace(6, 10, 9)

    def test_lcdm(self):

        # a LCDM power spectrum reproduces the Ludlow et al. (2016) model as computed by colossus
        for z in [0.2, 1.0, 2.5]:
            c_reference = self.reference.nfw_concentration(self.m, z)
            npt.assert_allclose(self.model_exact.nfw_concentration(self.m, z), c_reference, rtol=5e-4)
            npt.assert_allclose(self.model.nfw_concentration(self.m, z), c_reference, rtol=2e-3)
        npt.assert_allclose(self.model.nfw_concentration(10 ** 13.3, 0.5),
                            self.reference.nfw_concentration(10 ** 13.3, 0.5), rtol=2e-3)
        npt.assert_equal(isinstance(self.model.nfw_concentration(1e8, 0.5), float), True)

    def test_array_redshift(self):

        m = np.array([1e7, 1e8, 1e9, 1e8])
        z = np.array([0.3, 0.7, 0.3, 1.5])
        c = self.model_exact.nfw_concentration(m, z)
        npt.assert_allclose(c, [self.model_exact.nfw_concentration(mi, zi) for mi, zi in zip(m, z)], rtol=1e-12)
        npt.assert_allclose(c, self.reference._evaluate_concentration_colossus(m, z), rtol=5e-4)

    def test_scatter(self):

        model = ConcentrationLudlowPowerSpectrum(self.astropy, self.power_spectrum, scatter=True, scatter_dex=0.2)
        median = self.model.nfw_concentration(1e8, 0.5)
        npt.assert_allclose(model.nfw_concentration(1e8, 0.5, force_no_scatter=True), median, rtol=1e-12)
        npt.assert_allclose(_MedianConcentration(model).nfw_concentration(1e8, 0.5), median, rtol=1e-12)
        np.random.seed(2)
        c = model.nfw_concentration(np.full(20000, 1e8), 0.5)
        # the scatter is lognormal around the median, with width scatter_dex in log10(c)
        npt.assert_allclose(np.median(c), median, rtol=1e-2)
        npt.assert_allclose(np.std(np.log10(c)), 0.2, rtol=3e-2)

    def test_suppressed_power_spectrum(self):

        # removing small-scale power (half-mode mass ~10^6.5 M_sun) delays formation and lowers the concentration,
        # by more at lower masses; well above the cutoff the concentration approaches CDM
        transfer = (1 + (self.k / 300.0) ** 2) ** -5
        power_spectrum = LinearPowerSpectrum(self.k, self.pk * transfer ** 2, self.cosmo)
        model = ConcentrationLudlowPowerSpectrum(self.astropy, power_spectrum, scatter=False)
        ratio = model.nfw_concentration(self.m, 0.5) / self.model.nfw_concentration(self.m, 0.5)
        npt.assert_array_less(ratio, 1.0)
        npt.assert_array_less(0, np.diff(ratio))
        npt.assert_allclose(ratio[-1], 1.0, atol=2e-2)

    def test_preset(self):

        model, kwargs = preset_concentration_models('LUDLOW2016_PK', {'cosmo': self.astropy,
                                                                      'power_spectrum': self.power_spectrum})
        npt.assert_equal(model is ConcentrationLudlowPowerSpectrum, True)
        npt.assert_equal(kwargs['power_spectrum'] is self.power_spectrum, True)
        _ = model(**kwargs)
        npt.assert_raises(Exception, preset_concentration_models, 'LUDLOW2016_PK', {'cosmo': self.astropy})


class TestConcentrationTabulated(object):

    def setup_method(self):

        self.cosmo = Cosmology()
        self.log10m_table = np.linspace(6.0, 10.0, 9)
        self.z_table = np.array([0.0, 0.5, 1.0, 2.0])
        self.c_table = 20.0 * (10 ** self.log10m_table[:, np.newaxis] / 1e8) ** -0.1 * \
            (1 + self.z_table[np.newaxis, :]) ** -0.7
        self.model = ConcentrationTabulated(self.cosmo.astropy, self.log10m_table, self.z_table, self.c_table,
                                            scatter=False)

    def test_nodes(self):

        for j, z in enumerate(self.z_table):
            npt.assert_allclose(self.model.nfw_concentration(10 ** self.log10m_table, z), self.c_table[:, j],
                                rtol=1e-12)
        npt.assert_allclose(self.model.nfw_concentration(1e8, 0.5), self.c_table[4, 1], rtol=1e-12)
        npt.assert_equal(isinstance(self.model.nfw_concentration(1e8, 0.5), float), True)

    def test_interpolation(self):

        # the table is a power law in m and (1 + z), which bilinear interpolation of log10(c) in
        # (log10(m), log10(1 + z)) reproduces exactly
        m = np.array([3e6, 2e8, 7e9])
        z = np.array([0.1, 0.8, 1.7])
        expected = 20.0 * (m / 1e8) ** -0.1 * (1 + z) ** -0.7
        npt.assert_allclose(self.model.nfw_concentration(m, z), expected, rtol=1e-10)
        npt.assert_allclose(self.model.nfw_concentration(m, 0.8), 20.0 * (m / 1e8) ** -0.1 * 1.8 ** -0.7,
                            rtol=1e-10)

    def test_redshift_independent(self):

        model = ConcentrationTabulated(self.cosmo.astropy, self.log10m_table, [0.0], self.c_table[:, 0],
                                       scatter=False)
        npt.assert_allclose(model.nfw_concentration(3e7, 0.2), model.nfw_concentration(3e7, 5.0), rtol=1e-12)
        npt.assert_allclose(model.nfw_concentration(3e7, 0.2), 20.0 * 0.3 ** -0.1, rtol=1e-10)

    def test_scatter(self):

        model = ConcentrationTabulated(self.cosmo.astropy, self.log10m_table, self.z_table, self.c_table,
                                       scatter=True, scatter_dex=0.2)
        median = self.model.nfw_concentration(1e8, 0.5)
        npt.assert_allclose(model.nfw_concentration(1e8, 0.5, force_no_scatter=True), median, rtol=1e-12)
        np.random.seed(4)
        c = model.nfw_concentration(np.full(20000, 1e8), 0.5)
        npt.assert_allclose(np.std(np.log10(c)), 0.2, rtol=3e-2)

    def test_exceptions(self):

        npt.assert_raises(ValueError, self.model.nfw_concentration, 1e5, 0.5)
        npt.assert_raises(ValueError, self.model.nfw_concentration, 1e8, 2.5)
        npt.assert_raises(ValueError, ConcentrationTabulated, self.cosmo.astropy, self.log10m_table, self.z_table,
                          self.c_table[:, :2])
        npt.assert_raises(ValueError, ConcentrationTabulated, self.cosmo.astropy, self.log10m_table, self.z_table,
                          -self.c_table)

    def test_preset(self):

        model, kwargs = preset_concentration_models('TABULATED', {'cosmo': self.cosmo.astropy,
                                                                  'log10m_table': self.log10m_table,
                                                                  'z_table': self.z_table, 'c_table': self.c_table})
        npt.assert_equal(model is ConcentrationTabulated, True)
        _ = model(**kwargs)


if __name__ == '__main__':
    pytest.main()

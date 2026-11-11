import pytest
import numpy as np
import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.power_spectrum import LinearPowerSpectrum
from pyHalo.Cosmology.binned_power_spectrum import binned_linear_power_spectrum, lcdm_power_spectrum
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw
from pyHalo.Rendering.MassFunctions.power_spectrum import PowerSpectrumMassFunction, \
    PowerSpectrumSubhaloMassFunction


class TestPowerSpectrumSubhaloMassFunction(object):

    def setup_method(self):

        cosmo = Cosmology()
        self.power_spectrum_lcdm = LinearPowerSpectrum(*lcdm_power_spectrum(cosmo), cosmo)
        self.power_spectrum = binned_linear_power_spectrum(cosmo, [1.0, 1.0, 1.5, 0.8])
        self.z_eval = 3.8
        self.args = (6.0, 10.0, -1.9, False, 2e6)
        # keyword arguments the subhalo renderer passes, which the class does not use
        self.kwargs_renderer = {'sigma_sub': 0.025, 'host_m200': 10 ** 13.3, 'host_scaling_factor': 0.55,
                                'redshift_scaling_factor': 0.37, 'delta_power_law_index': 0.0}

    def test_lcdm(self):

        # with the reference power spectrum the mass function is the power law of CDMPowerLaw
        mfunc = PowerSpectrumSubhaloMassFunction(*self.args, power_spectrum=self.power_spectrum_lcdm,
                                                 power_spectrum_reference=self.power_spectrum_lcdm,
                                                 z_eval=self.z_eval, **self.kwargs_renderer)
        mfunc_power_law = CDMPowerLaw(*self.args)
        npt.assert_allclose(mfunc.n_mean, mfunc_power_law.n_mean, rtol=1e-10)
        npt.assert_allclose(mfunc.first_moment, mfunc_power_law.first_moment, rtol=1e-10)

    def test_shape(self):

        mfunc = PowerSpectrumSubhaloMassFunction(*self.args, power_spectrum=self.power_spectrum,
                                                 power_spectrum_reference=self.power_spectrum_lcdm,
                                                 z_eval=self.z_eval, **self.kwargs_renderer)
        m = np.logspace(6, 10, 9)
        ratio = PowerSpectrumMassFunction.dndm_comoving(m, self.z_eval, {'power_spectrum': self.power_spectrum}) / \
            PowerSpectrumMassFunction.dndm_comoving(m, self.z_eval, {'power_spectrum': self.power_spectrum_lcdm})
        ratio_pivot = PowerSpectrumSubhaloMassFunction.mass_function_ratio(
            np.array([1e8]), self.z_eval, self.power_spectrum, self.power_spectrum_lcdm)[0]
        npt.assert_allclose(mfunc.evaluate(m), 2e6 * m ** -1.9 * ratio / ratio_pivot, rtol=1e-10)
        npt.assert_allclose(mfunc.evaluate(1e8), 2e6 * 1e8 ** -1.9, rtol=1e-10)
        # the power spectrum changes the shape of the subhalo mass function
        npt.assert_array_less(0.1, np.max(np.abs(ratio / ratio_pivot - 1)))
        np.random.seed(1)
        m_draw = mfunc.draw()
        npt.assert_equal(len(m_draw), int(round(mfunc.n_mean)))
        npt.assert_equal(np.all((m_draw >= 1e6) & (m_draw <= 1e10)), True)

    def test_mass_function_ratio(self):

        m = np.logspace(6, 10, 5)
        ratio = PowerSpectrumSubhaloMassFunction.mass_function_ratio(m, 0.5, self.power_spectrum,
                                                                     self.power_spectrum_lcdm, 'SHETH_TORMEN')
        kwargs = {'multiplicity_function': 'SHETH_TORMEN'}
        expected = PowerSpectrumMassFunction.dndm_comoving(m, 0.5, dict(kwargs, power_spectrum=self.power_spectrum)) / \
            PowerSpectrumMassFunction.dndm_comoving(m, 0.5, dict(kwargs, power_spectrum=self.power_spectrum_lcdm))
        npt.assert_allclose(ratio, expected, rtol=1e-12)
        npt.assert_allclose(PowerSpectrumSubhaloMassFunction.mass_function_ratio(
            m, 0.5, self.power_spectrum_lcdm, self.power_spectrum_lcdm), 1.0, rtol=1e-12)

    def test_empty(self):

        mfunc = PowerSpectrumSubhaloMassFunction(8.0, 8.0, -1.9, True, 2e6, power_spectrum=self.power_spectrum,
                                                 power_spectrum_reference=self.power_spectrum_lcdm, z_eval=3.8)
        npt.assert_equal(mfunc.n_mean, 0)
        npt.assert_equal(len(mfunc.draw()), 0)


if __name__ == '__main__':
    pytest.main()

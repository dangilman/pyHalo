import pytest
import numpy as np
import numpy.testing as npt
from scipy.stats import kstest
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Rendering.MassFunctions.tabulated import TabulatedMassFunction
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw
from pyHalo.Rendering.MassFunctions.density_peaks import evaluate_mass_function
from pyHalo.mass_function_models import preset_mass_function_models


class TestTabulatedMassFunction(object):

    def setup_method(self):

        # a curved mass function, so that the nodes matter
        self.m_nodes = np.logspace(5.5, 10.5, 26)
        self.dN_dm_nodes = 3e7 * self.m_nodes ** -1.9 * np.exp(-(self.m_nodes / 10 ** 9.5) ** 0.5)
        self.mfunc = TabulatedMassFunction(6.0, 10.0, self.m_nodes, self.dN_dm_nodes, draw_poisson=True)

    def _cdf(self, m):
        m_grid = np.logspace(6.0, 10.0, 20001)
        y = self.mfunc.evaluate(m_grid) * m_grid
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(np.log(m_grid)))))
        return np.interp(m, m_grid, cdf / cdf[-1])

    def test_power_law(self):

        # with two nodes the tabulated mass function is a power law
        norm, index = 4e6, -1.85
        m_nodes = np.array([1e6, 1e10])
        mfunc = TabulatedMassFunction(6.0, 10.0, m_nodes, norm * m_nodes ** index, False)
        mfunc_power_law = CDMPowerLaw(6.0, 10.0, index, False, norm)
        npt.assert_allclose(mfunc.n_mean, mfunc_power_law.n_mean, rtol=1e-10)
        npt.assert_allclose(mfunc.first_moment, mfunc_power_law.first_moment, rtol=1e-10)
        m = np.logspace(6, 10, 15)
        npt.assert_allclose(mfunc.evaluate(m), mfunc_power_law.evaluate(m), rtol=1e-10)
        # the special cases of the integrals and of the inverse cumulative distribution
        for index in [-1.0, -2.0]:
            mfunc = TabulatedMassFunction(6.0, 10.0, m_nodes, norm * m_nodes ** index, False)
            mfunc_power_law = CDMPowerLaw(6.0, 10.0, index, False, norm)
            npt.assert_allclose(mfunc.n_mean, mfunc_power_law.n_mean, rtol=1e-10)
            npt.assert_allclose(mfunc.first_moment, mfunc_power_law.first_moment, rtol=1e-10)
            np.random.seed(1)
            m_draw = mfunc.draw()
            npt.assert_equal(np.all((m_draw >= 1e6) & (m_draw <= 1e10)), True)

    def test_moments(self):

        m = np.logspace(6.0, 10.0, 200001)
        dN_dm = self.mfunc.evaluate(m)
        npt.assert_allclose(self.mfunc.n_mean, np.trapezoid(dN_dm * m, np.log(m)), rtol=1e-6)
        npt.assert_allclose(self.mfunc.first_moment, np.trapezoid(dN_dm * m ** 2, np.log(m)), rtol=1e-6)
        # the nodes are reproduced, and the range is cut at log_mlow and log_mhigh
        inside = (self.m_nodes > 1e6) & (self.m_nodes < 1e10)
        npt.assert_allclose(self.mfunc.evaluate(self.m_nodes[inside]), self.dN_dm_nodes[inside], rtol=1e-12)
        npt.assert_equal(isinstance(self.mfunc.evaluate(1e8), float), True)

    def test_draw(self):

        np.random.seed(10)
        counts = []
        masses = []
        for _ in range(400):
            m = self.mfunc.draw()
            counts.append(len(m))
            masses.append(m)
        masses = np.concatenate(masses)
        npt.assert_array_less(np.abs(np.mean(counts) - self.mfunc.n_mean),
                              4 * np.sqrt(self.mfunc.n_mean / len(counts)))
        npt.assert_equal(np.all((masses >= 1e6) & (masses <= 1e10)), True)
        npt.assert_array_less(1e-3, kstest(masses, self._cdf).pvalue)
        mfunc = TabulatedMassFunction(6.0, 10.0, self.m_nodes, self.dN_dm_nodes, draw_poisson=False)
        npt.assert_equal(len(mfunc.draw()), int(round(mfunc.n_mean)))

    def test_zero_segments(self):

        # a node with dN/dm = 0 at 10^7 M_sun empties both adjacent segments, so no halos below 10^8 M_sun
        m_nodes = np.logspace(6, 10, 5)
        dN_dm = 1e7 * m_nodes ** -1.9
        dN_dm[1] = 0.0
        mfunc = TabulatedMassFunction(6.0, 10.0, m_nodes, dN_dm, False)
        expected = CDMPowerLaw(8.0, 10.0, -1.9, False, 1e7).n_mean
        npt.assert_allclose(mfunc.n_mean, expected, rtol=1e-10)
        npt.assert_equal(mfunc.evaluate(np.array([3e6, 3e7])), np.zeros(2))
        np.random.seed(3)
        m = mfunc.draw()
        npt.assert_equal(np.all(m >= 1e8), True)

    def test_empty(self):

        mfunc = TabulatedMassFunction(8.0, 8.0, self.m_nodes, self.dN_dm_nodes, True)
        npt.assert_equal(mfunc.n_mean, 0)
        npt.assert_equal(mfunc.first_moment, 0)
        npt.assert_equal(len(mfunc.draw()), 0)

    def test_exceptions(self):

        npt.assert_raises(ValueError, TabulatedMassFunction, 5.0, 10.0, self.m_nodes, self.dN_dm_nodes, True)
        npt.assert_raises(ValueError, TabulatedMassFunction, 6.0, 10.0, self.m_nodes[::-1], self.dN_dm_nodes, True)
        npt.assert_raises(ValueError, TabulatedMassFunction, 6.0, 10.0, self.m_nodes, -self.dN_dm_nodes, True)


class TestTabulatedFromRedshift(object):

    def setup_method(self):

        cosmo = Cosmology()
        self.h = cosmo.h
        self.geometry = Geometry(cosmo, 0.5, 1.5, 6.0, 'DOUBLE_CONE')
        # a table of the Sheth-Tormen mass function
        self.log10m_table = np.linspace(5.0, 11.0, 61)
        self.z_table = np.linspace(0.0, 3.0, 31)
        dndm = np.array([evaluate_mass_function(10 ** self.log10m_table, self.h, zi, 'sheth99')
                         for zi in self.z_table]).T
        self.kwargs_model = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'LOS_normalization': 1.0, 'draw_poisson': False,
                             'log10m_table': self.log10m_table, 'z_table': self.z_table,
                             'dndm_comoving_table': dndm}

    def test_from_redshift(self):

        for z in [0.35, 1.27]:
            mfunc = TabulatedMassFunction.from_redshift(z, 0.02, self.geometry, self.kwargs_model)
            m = np.logspace(6, 10, 30)
            exact = evaluate_mass_function(m, self.h, z, 'sheth99') * self.geometry.volume_element_comoving(z, 0.02)
            npt.assert_allclose(mfunc.evaluate(m), exact, rtol=2e-3)

    def test_normalization(self):

        z = 0.7
        mfunc = TabulatedMassFunction.from_redshift(z, 0.02, self.geometry, self.kwargs_model)
        kwargs_model = dict(self.kwargs_model, LOS_normalization=2.0, delta_power_law_index=0.1, m_pivot=1e8)
        mfunc_scaled = TabulatedMassFunction.from_redshift(z, 0.02, self.geometry, kwargs_model)
        m = np.logspace(6, 10, 9)
        npt.assert_allclose(mfunc_scaled.evaluate(m), 2.0 * (m / 1e8) ** 0.1 * mfunc.evaluate(m), rtol=1e-10)

    def test_redshift_independent(self):

        kwargs_model = dict(self.kwargs_model, z_table=[0.5],
                            dndm_comoving_table=self.kwargs_model['dndm_comoving_table'][:, 5])
        mfunc_1 = TabulatedMassFunction.from_redshift(0.2, 0.02, self.geometry, kwargs_model)
        mfunc_2 = TabulatedMassFunction.from_redshift(5.0, 0.02, self.geometry, kwargs_model)
        npt.assert_allclose(mfunc_1.n_mean / self.geometry.volume_element_comoving(0.2, 0.02),
                            mfunc_2.n_mean / self.geometry.volume_element_comoving(5.0, 0.02), rtol=1e-10)

    def test_exceptions(self):

        npt.assert_raises(ValueError, TabulatedMassFunction.from_redshift, 3.5, 0.02, self.geometry,
                          self.kwargs_model)
        kwargs_model = dict(self.kwargs_model, log_mhigh=11.5)
        npt.assert_raises(ValueError, TabulatedMassFunction.from_redshift, 0.5, 0.02, self.geometry, kwargs_model)

    def test_preset(self):

        model, kwargs_out = preset_mass_function_models('TABULATED', self.kwargs_model)
        npt.assert_equal(model is TabulatedMassFunction, True)
        npt.assert_raises(Exception, preset_mass_function_models, 'TABULATED', {'log_mlow': 6.0})


if __name__ == '__main__':
    pytest.main()

import pytest
import numpy as np
import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.node_ratios import Log10RatioSpline
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw
from pyHalo.Rendering.MassFunctions.node_ratio import NodeRatioMassFunction, NodeRatioSubhaloMassFunction
from pyHalo.mass_function_models import preset_mass_function_models


class TestNodeRatioMassFunction(object):

    def setup_method(self):

        self.geometry = Geometry(Cosmology(), 0.5, 2.0, 6.0, 'DOUBLE_CONE')
        self.kwargs = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 10 ** 8, 'LOS_normalization': 1.0,
                       'delta_power_law_index': 0.0, 'draw_poisson': False}
        self.z, self.dz = 0.7, 0.02
        self.mfunc_cdm = ShethTormen.from_redshift(self.z, self.dz, self.geometry, self.kwargs)

    def test_cdm(self):

        # with every node at zero the mass function is the one of the CDM class
        mfunc = NodeRatioMassFunction.from_redshift(self.z, self.dz, self.geometry,
                                                    dict(self.kwargs, log10_n_ratio_nodes=[0.0, 0.0, 0.0]))
        npt.assert_allclose(mfunc.n_mean, self.mfunc_cdm.n_mean, rtol=1e-8)
        npt.assert_allclose(mfunc.first_moment, self.mfunc_cdm.first_moment, rtol=1e-8)
        m = np.logspace(6, 10, 7)
        npt.assert_allclose(mfunc.evaluate(m), self.mfunc_cdm.evaluate(m), rtol=1e-8)

    def test_ratio(self):

        nodes = [0.5, -0.4, 0.0]
        mfunc = NodeRatioMassFunction.from_redshift(self.z, self.dz, self.geometry,
                                                    dict(self.kwargs, log10_n_ratio_nodes=nodes))
        # the tabulation nodes are 20 per dex, so the mass function is exact there
        m = np.logspace(6, 10, 81)
        expected = self.mfunc_cdm.evaluate(m) * 10 ** Log10RatioSpline(nodes)(np.log10(m))
        npt.assert_allclose(mfunc.evaluate(m), expected, rtol=1e-8)
        npt.assert_array_less(mfunc.n_mean, 2 * self.mfunc_cdm.n_mean)
        np.random.seed(2)
        m_draw = mfunc.draw()
        npt.assert_equal(len(m_draw), int(round(mfunc.n_mean)))
        npt.assert_equal(np.all((m_draw >= 1e6) & (m_draw <= 1e10)), True)

    def test_floor(self):

        kwargs = dict(self.kwargs, log10_n_ratio_nodes=[-4.0, -1.0, 0.0])
        mfunc = NodeRatioMassFunction.from_redshift(self.z, self.dz, self.geometry, kwargs)
        npt.assert_allclose(mfunc.evaluate(1e6) / self.mfunc_cdm.evaluate(1e6), 1e-2, rtol=1e-8)
        mfunc = NodeRatioMassFunction.from_redshift(self.z, self.dz, self.geometry,
                                                    dict(kwargs, log10_n_ratio_floor=None))
        npt.assert_allclose(mfunc.evaluate(1e6) / self.mfunc_cdm.evaluate(1e6), 1e-4, rtol=1e-8)

    def test_empty_mass_range(self):

        mfunc = NodeRatioMassFunction.from_redshift(self.z, self.dz, self.geometry,
                                                    dict(self.kwargs, log_mlow=10.0, log10_n_ratio_nodes=[0.3, 0.0, 0.0]))
        npt.assert_equal(mfunc.n_mean, 0)
        npt.assert_equal(len(mfunc.draw()), 0)

    def test_preset_mass_function_models(self):

        model, kwargs = preset_mass_function_models('NODE_RATIO', {'log10_n_ratio_nodes': [0.1, 0.2, 0.0]})
        npt.assert_equal(model is NodeRatioMassFunction, True)
        model, _ = preset_mass_function_models('NODE_RATIO_SHMF', {'log10_n_ratio_nodes': [0.1, 0.2, 0.0]})
        npt.assert_equal(model is NodeRatioSubhaloMassFunction, True)
        npt.assert_raises(Exception, preset_mass_function_models, 'NODE_RATIO', {})


class TestNodeRatioSubhaloMassFunction(object):

    def setup_method(self):

        self.args = (6.0, 10.0, -1.9, False, 2e6)
        # keyword arguments the subhalo renderer passes, which the class does not use
        self.kwargs_renderer = {'sigma_sub': 0.025, 'host_m200': 10 ** 13.3, 'host_scaling_factor': 0.55,
                                'redshift_scaling_factor': 0.37, 'delta_power_law_index': 0.0, 'm_pivot': 10 ** 8}

    def test_cdm(self):

        mfunc = NodeRatioSubhaloMassFunction(*self.args, log10_n_ratio_nodes=[0.0, 0.0, 0.0], **self.kwargs_renderer)
        mfunc_cdm = CDMPowerLaw(*self.args)
        npt.assert_allclose(mfunc.n_mean, mfunc_cdm.n_mean, rtol=1e-8)
        npt.assert_allclose(mfunc.first_moment, mfunc_cdm.first_moment, rtol=1e-8)

    def test_ratio(self):

        nodes = [-0.6, 0.3, 0.0]
        mfunc = NodeRatioSubhaloMassFunction(*self.args, log10_n_ratio_nodes=nodes, **self.kwargs_renderer)
        m = np.logspace(6, 10, 81)
        npt.assert_allclose(mfunc.evaluate(m), 2e6 * m ** -1.9 * 10 ** Log10RatioSpline(nodes)(np.log10(m)),
                            rtol=1e-8)
        # the amplitude at 10^8 is sigma_sub times the ratio there
        npt.assert_allclose(mfunc.evaluate(1e8), 2e6 * 1e8 ** -1.9 * 10 ** 0.3, rtol=1e-8)
        np.random.seed(3)
        npt.assert_equal(len(mfunc.draw()), int(round(mfunc.n_mean)))

    def test_empty_mass_range(self):

        mfunc = NodeRatioSubhaloMassFunction(10.0, 10.0, -1.9, False, 2e6, log10_n_ratio_nodes=[0.3, 0.0, 0.0])
        npt.assert_equal(mfunc.n_mean, 0)
        npt.assert_equal(len(mfunc.draw()), 0)


if __name__ == '__main__':
    pytest.main()

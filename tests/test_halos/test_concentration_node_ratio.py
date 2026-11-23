import pytest
import numpy as np
import numpy.testing as npt
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.concentration import ConcentrationLudlow
from pyHalo.Halos.concentration_node_ratio import ConcentrationNodeRatio
from pyHalo.Halos.batch_halo_util import batch_nfw_concentration
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.node_ratios import Log10RatioSpline


class TestConcentrationNodeRatio(object):

    def setup_method(self):

        self.cosmo = Cosmology().astropy
        self.nodes = [0.5, 0.2, -0.1]
        self.model = ConcentrationNodeRatio(self.cosmo, ConcentrationLudlow, self.nodes, scatter=False)
        self.cdm = ConcentrationLudlow(self.cosmo, scatter=False)
        self.m = np.logspace(5.5, 10.5, 11)

    def test_ratio(self):

        for z in [0.2, 1.5]:
            c = self.model.nfw_concentration(self.m, z)
            expected = self.cdm.nfw_concentration(self.m, z) * 10 ** Log10RatioSpline(self.nodes)(np.log10(self.m))
            npt.assert_allclose(c, expected, rtol=1e-10)
        c_float = self.model.nfw_concentration(1e7, 0.5)
        npt.assert_equal(isinstance(c_float, float), True)
        npt.assert_allclose(c_float, self.cdm.nfw_concentration(1e7, 0.5) * 10 ** Log10RatioSpline(self.nodes)(7.0),
                            rtol=1e-10)

    def test_cdm(self):

        model = ConcentrationNodeRatio(self.cosmo, ConcentrationLudlow, [0.0, 0.0, 0.0], scatter=False)
        npt.assert_allclose(model.nfw_concentration(self.m, 0.5), self.cdm.nfw_concentration(self.m, 0.5), rtol=1e-12)

    def test_batch(self):

        # the batch path (used when rendering realizations) agrees with the per-halo path
        npt.assert_allclose(batch_nfw_concentration(self.model, self.m, 0.7),
                            [self.model.nfw_concentration(float(mi), 0.7) for mi in self.m], rtol=1e-10)

    def test_scatter(self):

        model = ConcentrationNodeRatio(self.cosmo, ConcentrationLudlow, self.nodes, scatter=True, scatter_dex=0.15)
        np.random.seed(4)
        m = np.full(20000, 1e8)
        log_ratio = np.log10(batch_nfw_concentration(model, m, 0.5) / self.model.nfw_concentration(1e8, 0.5))
        npt.assert_allclose(np.std(log_ratio), 0.15, rtol=0.03)
        npt.assert_allclose(np.mean(log_ratio), 0.0, atol=0.005)

    def test_universal_minimum(self):

        model = ConcentrationNodeRatio(self.cosmo, ConcentrationLudlow, [-2.0, -2.0, -2.0], scatter=False)
        npt.assert_allclose(model.nfw_concentration(self.m, 0.5), model._universal_minimum)

    def test_preset_concentration_models(self):

        model, kwargs = preset_concentration_models('NODE_RATIO', {'concentration_cdm_class': ConcentrationLudlow,
                                                                    'log10_c_ratio_nodes': self.nodes})
        npt.assert_equal(model is ConcentrationNodeRatio, True)
        npt.assert_raises(Exception, preset_concentration_models, 'NODE_RATIO', {'log10_c_ratio_nodes': self.nodes})


if __name__ == '__main__':
    pytest.main()

import pytest
import numpy as np
import numpy.testing as npt
from pyHalo.node_ratios import Log10RatioSpline, LOG10_M_NODES


class TestLog10RatioSpline(object):

    def setup_method(self):

        self.nodes = [0.4, -0.3, 0.2]
        self.spline = Log10RatioSpline(self.nodes)

    def test_nodes(self):

        npt.assert_allclose(self.spline(np.array(LOG10_M_NODES)), self.nodes, atol=1e-12)
        npt.assert_equal(isinstance(self.spline(7.0), float), True)
        npt.assert_allclose(self.spline.ratio(10 ** np.array(LOG10_M_NODES)), 10 ** np.array(self.nodes), rtol=1e-12)

    def test_held_outside_nodes(self):

        npt.assert_allclose(self.spline(np.array([4.0, 5.9])), self.nodes[0], atol=1e-12)
        npt.assert_allclose(self.spline(np.array([10.1, 13.0])), self.nodes[-1], atol=1e-12)

    def test_natural_spline(self):

        # a natural cubic spline has zero curvature at the end nodes
        eps = 1e-4
        for x0 in [LOG10_M_NODES[0] + 2 * eps, LOG10_M_NODES[-1] - 2 * eps]:
            curvature = (self.spline(x0 + eps) - 2 * self.spline(x0) + self.spline(x0 - eps)) / eps ** 2
            npt.assert_array_less(abs(curvature), 1e-2)
        # zero at every node is CDM everywhere
        npt.assert_allclose(Log10RatioSpline([0.0, 0.0, 0.0])(np.linspace(4, 12, 20)), 0.0, atol=1e-15)

    def test_floor(self):

        spline = Log10RatioSpline([-3.0, -1.0, 0.0], log10_floor=-2.0)
        npt.assert_allclose(spline(np.array([5.0, 6.0])), -2.0)
        npt.assert_array_less(-2.0 - 1e-12, spline(np.linspace(4, 12, 50)))
        npt.assert_allclose(spline(8.0), -1.0, atol=1e-12)

    def test_exceptions(self):

        npt.assert_raises(ValueError, Log10RatioSpline, [0.0, 0.1])
        npt.assert_raises(ValueError, Log10RatioSpline, [0.0, 0.1], [7.0, 6.0])
        npt.assert_raises(ValueError, Log10RatioSpline, [0.0, np.nan, 0.0])


if __name__ == '__main__':
    pytest.main()

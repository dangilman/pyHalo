import numpy as np
import numpy.testing as npt
import pytest
from pyHalo.Rendering.SpatialDistributions.correlated import Correlated2D

class GeometryDummy(object):

    def rendering_scale(self, *args, **kwargs):

        return 1.

    def kpc_per_arcsec(self, *args,  **kwargs):

        return 1.

class TestCorrelated(object):

    def setup_method(self):

        geometry = GeometryDummy()
        self.correlated = Correlated2D(geometry, 1.)

    def test_correlated(self):

        sigma = 0.8
        r_max = 3. * sigma
        n_points = 500000
        nbins = 200

        mean = [0., 0.]
        cov = [[sigma**2, 0], [0, sigma**2]]
        out = np.random.multivariate_normal(mean, cov, n_points)
        x_gaussian, y_gaussian = out[:,0], out[:, 1]
        r_theory = np.hypot(x_gaussian, y_gaussian)
        density, _,  _ = np.histogram2d(x_gaussian, y_gaussian, bins=nbins,
                                        range=((-r_max, r_max), (-r_max, r_max)))

        x, y = self.correlated.draw(n_points, r_max, density, 1.)
        r = np.hypot(x, y)
        h, _, _ = np.histogram2d(x, y, bins=100, range=((-r_max, r_max), (-r_max, r_max)))
        h_theory, _, _ = np.histogram2d(x_gaussian, y_gaussian, bins=100, range=((-r_max, r_max), (-r_max, r_max)))

        radii = np.linspace(0, 1. * r_max, 50)
        rendered = []
        theory = []
        for i in range(0, len(radii)-1):
            rad = radii[i + 1]**2 - radii[i]**2

            condition = np.logical_and(r >= radii[i], r < radii[i+1])
            n_rendered = np.sum(np.where(condition))
            rendered.append(n_rendered/rad)

            condition = np.logical_and(r_theory >= radii[i], r_theory < radii[i + 1])
            n_theory = np.sum(np.where(condition))
            theory.append(n_theory / rad)

        residuals = np.absolute(1-np.array(theory)/np.array(rendered))
        mean_error = np.mean(residuals)
        npt.assert_equal(len(x), len(y))
        npt.assert_equal(len(x), n_points)
        npt.assert_equal(True, mean_error < 0.05)


if __name__ == '__main__':

    pytest.main()

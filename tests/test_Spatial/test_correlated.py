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

    def setup(self):

        geometry = GeometryDummy()
        self.correlated = Correlated2D(geometry, 1.)

    def test_correlated(self):

        r_max = 1.5
        sigma = 0.5
        z_plane = 1.
        n_points = 20000
        nbins = 50
        r_gaussian = np.random.normal(0., sigma, n_points)
        theta = np.random.uniform(0, 2*np.pi, len(r_gaussian))
        x_gaussian, y_guassian = r_gaussian * np.cos(theta), r_gaussian * np.sin(theta)
        density, _,  _ = np.histogram2d(x_gaussian, y_guassian, bins=nbins,
                                        range=((-r_max, r_max), (-r_max, r_max)))

        n_sample = n_points * 10
        x, y = self.correlated.draw(n_points * 10, r_max, density, z_plane)

        npt.assert_equal(len(x), len(y))
        npt.assert_equal(len(x), n_sample)
        #
        # density_sampled, _, _ = np.histogram2d(x, y,
        #                                        bins=nbins,
        #                                        range=((-r_max, r_max), (-r_max, r_max)))


if __name__ == '__main__':

    pytest.main()

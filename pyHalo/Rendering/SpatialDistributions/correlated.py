import numpy as np
from pyHalo.Rendering.SpatialDistributions.base import SpatialDistributionBase

class Correlated2D(SpatialDistributionBase):
    """
    This class generates points from an arbitrary 2D probability distribution
    """
    def __init__(self, geometry, smooth_scale=1.):

        """
        :param geometry: an instance of Geometry
        :param smooth_scale: a smoothing scale that removes the regular grid
        pattern in the rendered points
        """
        self._geo = geometry
        self._smooth_scale = smooth_scale
        super(Correlated2D, self).__init__()

    def draw(self, n, r_max, density, z_plane, shift_x=0., shift_y=0.):

        """

        :param n: the number of points to draw
        :param r_max: the radius in arcsec of the rendering area, should correspond to
        the angular size of density
        :param density: the 2D probability density to sample from
        :param z_plane: the redshift of the lens plane
        :param shift_x: moves the center of rendered points
        from (0, 0) to (shift_x, shift_y)
        :param shift_y: moves the center of rendered points
        from (0, 0) to (shift_x, shift_y)
        :return: x and y samples
        """
        norm = np.sum(density)
        if norm == 0:
            raise Exception('2D probability distribution not normalizable')

        density = density / np.sum(density)

        s = density.shape[0]
        p = density.reshape(-1)

        values = np.arange(len(p))
        pairs = np.indices(dimensions=(s, s)).T

        x_coordinates_arcsec = np.linspace(-r_max, r_max, s)
        y_coordinates_arcsec = np.linspace(-r_max, r_max, s)

        inds = np.random.choice(values, p=p, size=n, replace=True)
        locations = pairs.reshape(-1, 2)[inds]
        x_sample_pixel, y_sample_pixel = locations[:, 0], locations[:, 1]

        x_sample_arcsec = x_coordinates_arcsec[x_sample_pixel]
        y_sample_arcsec = y_coordinates_arcsec[y_sample_pixel]

        smoothing = self._smooth_scale * r_max / s
        x_sample_arcsec += np.random.normal(0., smoothing, len(x_sample_arcsec))
        y_sample_arcsec += np.random.normal(0., smoothing, len(y_sample_arcsec))

        kpc_per_asec = self._geo.kpc_per_arcsec(z_plane)

        x_sample_arcsec += shift_x
        y_sample_arcsec += shift_y

        x_kpc, y_kpc = x_sample_arcsec * kpc_per_asec, y_sample_arcsec * kpc_per_asec

        return x_kpc, y_kpc

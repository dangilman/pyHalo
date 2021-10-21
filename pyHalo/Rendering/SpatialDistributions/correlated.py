import numpy as np
import matplotlib.pyplot as plt

class Correlated2D(object):

    def __init__(self, geometry, smooth_scale=1.):

        self._geo = geometry
        self._smooth_scale = smooth_scale

    def draw(self, n, r_max, density, z_plane, shift_x=0., shift_y=0.):

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
        #
        # plt.imshow(density.reshape(s, s), cmap='bwr')
        # plt.scatter(x_sample_pixel, y_sample_pixel, marker='+', color='k')
        # plt.show()

        x_sample_arcsec += shift_x
        y_sample_arcsec += shift_y

        x_kpc, y_kpc = x_sample_arcsec * kpc_per_asec, y_sample_arcsec * kpc_per_asec

        return x_kpc, y_kpc

import numpy as np


class LensConeUniform(object):

    def __init__(self, cone_opening_angle, geometry):

        self._cosmo_geometry = geometry

        self._uni = Uniform(0.5 * cone_opening_angle, geometry)

    def draw(self, N, z_plane, center_x=0, center_y=0):

        if N == 0:
            return np.array([]), np.array([])

        rescale = self._cosmo_geometry.rendering_scale(z_plane)

        x_kpc, y_kpc = self._uni.draw(N, z_plane, rescale=rescale,
                                        center_x=center_x, center_y=center_y)

        return x_kpc, y_kpc

class Uniform(object):

    def __init__(self, rmax2d_arcsec, geometry):

        self.rmax2d_arcsec = rmax2d_arcsec
        self._geo = geometry

    def draw(self, N, z_plane, rescale=1.0, center_x=0, center_y=0):

        if N == 0:
            return [], []

        angle = np.random.uniform(0, 2 * np.pi, int(N))

        rmax = self.rmax2d_arcsec * rescale

        r = np.random.uniform(0, rmax ** 2, int(N))

        x_arcsec = r ** .5 * np.cos(angle)
        y_arcsec = r ** .5 * np.sin(angle)

        x_arcsec += center_x
        y_arcsec += center_y
        kpc_per_asec = self._geo.kpc_per_arcsec(z_plane)
        x_kpc, y_kpc = x_arcsec * kpc_per_asec, y_arcsec * kpc_per_asec

        return np.array(x_kpc), np.array(y_kpc)

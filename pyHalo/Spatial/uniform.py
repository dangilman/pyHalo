import numpy as np

class LensConeUniform(object):

    def __init__(self, cone_opening_angle, cosmo_geometry):

        self._cosmo_geometry = cosmo_geometry

        self._uni = Uniform(cone_opening_angle * 0.5, cosmo_geometry)

    def draw(self, N, z_plane, center_x=0, center_y=0):

        if N == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        rescale = self._cosmo_geometry.rendering_scale(z_plane)

        x_kpc, y_kpc, r2d_kpc, r3d_kpc = self._uni.draw(N, z_plane, rescale=rescale,
                                        center_x=center_x, center_y=center_y)

        return x_kpc, y_kpc, r2d_kpc, r3d_kpc

class Uniform(object):

    def __init__(self, rmax2d_arcsec, geometry):

        self.rmax2d_arcsec = rmax2d_arcsec
        self._geo = geometry

    def draw(self, N, z_plane, rescale=1.0, center_x = 0, center_y = 0):

        if N == 0:
            return [], [], [], []

        angle = np.random.uniform(0, 2 * np.pi, int(N))

        rmax = self.rmax2d_arcsec * rescale

        r = np.random.uniform(0, rmax ** 2, int(N))

        x_arcsec = r ** .5 * np.cos(angle)
        y_arcsec = r ** .5 * np.sin(angle)

        x_arcsec += center_x
        y_arcsec += center_y
        kpc_per_asec = self._geo.kpc_per_arcsec(z_plane)
        x_kpc, y_kpc = x_arcsec * kpc_per_asec, y_arcsec * kpc_per_asec
        rmax_kpc = rmax * kpc_per_asec

        r2d_kpc = (x_kpc ** 2 + y_kpc ** 2) ** .5

        zcoord = np.random.uniform(0, (rmax_kpc ** 2 - r2d_kpc ** 2) ** 0.5)

        return np.array(x_kpc), np.array(y_kpc), np.array(r2d_kpc), np.sqrt(zcoord ** 2 + r2d_kpc ** 2)

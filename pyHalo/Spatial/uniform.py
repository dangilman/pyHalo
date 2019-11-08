import numpy as np

class LensConeUniform(object):

    def __init__(self, cone_opening_angle, cosmo_geometry):

        self._cosmo_geometry = cosmo_geometry

        self._dd_comoving = self._cosmo_geometry._cosmo.D_C_transverse(
            self._cosmo_geometry._lens_cosmo.z_lens
        )
        self._d_ds_comoving = self._cosmo_geometry._cosmo.D_C_transverse(
            self._cosmo_geometry._lens_cosmo.z_source) - self._dd_comoving

        self._uni = Uniform(cone_opening_angle * 0.5)

    def draw(self,N, z_plane, center_x = 0, center_y = 0):

        if N == 0:
            return [], [], [], []

        rescale = self._cosmo_geometry.rendering_scale(z_plane)

        x, y, r2d, r3d = self._uni.draw(N, rescale=rescale,
                                        center_x = center_x, center_y = center_y)

        return x, y, r2d, r3d

class Uniform(object):

    def __init__(self, rmax2d_arcsec=None):

        self.rmax2d_arcsec = rmax2d_arcsec

    def draw(self, N, rescale=1, center_x = 0, center_y = 0):

        if N == 0:
            return [], [], [], []
        angle = np.random.uniform(0, 2 * np.pi, int(N))
        r = np.random.uniform(0, (self.rmax2d_arcsec * rescale) ** 2, int(N))

        x = r ** .5 * np.cos(angle)
        y = r ** .5 * np.sin(angle)
        r2d = (x ** 2 + y ** 2) ** .5

        zcoord = np.random.uniform(0, ((self.rmax2d_arcsec*rescale) ** 2 - r2d ** 2) ** 0.5)

        x += center_x
        y += center_y

        return x, y, r2d, np.sqrt(zcoord ** 2 + r2d ** 2)

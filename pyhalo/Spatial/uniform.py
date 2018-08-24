import numpy as np

class LensConeUniform(object):

    def __init__(self, cone_opening_angle, cosmo_geometry):

        self._cosmo_geometry = cosmo_geometry

        self._uni = Uniform(cone_opening_angle)

    def draw(self,N,z_plane):

        if self.z_plane > self.z_lens:

            rmax_0 = self._uni.rmax2d_arcsec
            new_rmax = self._cosmo_geometry.lens_cone_angle(rmax_0, z_plane,
                                                            self._cosmo_geometry._lens_cosmo.z_lens)

            rescale = new_rmax / rmax_0

        else:
            rescale = 1

        x, y, r2d, r3d = self._uni.draw(N,rescale)

        return x, y, r2d, r3d

class Uniform(object):

    def __init__(self, rmax2d_arcsec=None):

        self.rmax2d_arcsec = rmax2d_arcsec

    def draw(self,N, rescale=1):

        angle = np.random.uniform(0, 2 * np.pi, int(N))
        r = np.random.uniform(0, (self.rmax2d_arcsec * rescale) ** 2, int(N))

        x = r ** .5 * np.cos(angle)
        y = r ** .5 * np.sin(angle)
        r2d = (x ** 2 + y ** 2) ** .5
        zcoord = np.random.uniform(0, (self.rmax2d_arcsec ** 2 - r2d ** 2) ** 0.5)

        return x, y, r2d, np.sqrt(zcoord ** 2 + r2d ** 2)


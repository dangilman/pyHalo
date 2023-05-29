import numpy as np
from pyHalo.Rendering.SpatialDistributions.base import SpatialDistributionBase

class LensConeUniform(SpatialDistributionBase):

    """
    This class generates samples drawn uniformly in two dimensions out to maximum radius
    r(z) = 0.5 * cone_opening_angle * f(z), where cone_opening_angle is the opening angle of the rendering volume
    specified when creating the realization, and f(z) is given by geometry.rendering_scale(z) and depends on the rendering
    geometry. For example, if the rendering volume is definied as DOUBLE_CONE (the default setting), then f(z) = 1
    for z < z_lens and decreases approaching the source redshift at z > z_lens. As the effect of the rendering geometry
    is built into this class, it is best used to generate halo positions distributed randomly in two dimensions
    along the line of sight.
    """

    def __init__(self, cone_opening_angle, geometry):

        """

        :param cone_opening_angle: the opening angle of the rendering volume [arcsec]

        If the rendering geometry is DOUBLE_CONE (default) then this is the opening angle of the cone

        If the rendering geometry is set to CYLINDER, then this is sets the comoving radius of the cylinder such that
        a ray that moves in a straight line from the observer to the lens plane at an angle cone_opening_angle inersects
        the edge of the cylinder at z_lens.

        :param geometry: an instance of Geometry (Cosmology.geometry)
        """
        self._cosmo_geometry = geometry
        self._uni = Uniform(0.5 * cone_opening_angle, geometry)
        super(LensConeUniform, self).__init__()

    @classmethod
    def from_Mhost(cls, *args, **kwargs):
        """

        :param rmax2d_arcsec:
        :param geometry:
        :return:
        """
        raise Exception('Spatial distribution class LensConeUniform not currently implemented for subhalos')

    def draw(self, N, z_plane, center_x=0, center_y=0):

        """
        Generates samples in two dimensions out to a maximum radius r = 0.5 * cone_opening_angle * f(z)
        where f(z) = geometry.rendering_scale(z)
        :param N: number of objects to generate
        :param z_plane: the redshift where the objects are being placed (used to compute the conversion to physical kpc)
        :param center_x: the x-center of the rendering area [arcsec]
        :param center_y: the y-center of the rendering area [arcsec]
        :return: the x and y coordinates sampled in 2D [kpc]
        """
        if N == 0:
            return np.array([]), np.array([])

        rescale = self._cosmo_geometry.rendering_scale(z_plane)

        x_kpc, y_kpc = self._uni.draw(N, z_plane, rescale=rescale,
                                        center_x=center_x, center_y=center_y)

        return x_kpc, y_kpc

class Uniform(SpatialDistributionBase):

    """
    This class generates samples distributed uniformly in two dimensions out to a radius 0.5 * cone_opening_angle
    """

    def __init__(self, rmax2d_arcsec, geometry):

        """

        :param rmax2d_arcsec: the maximum radius to render objects [arcsec]
        :param geometry: an instance of Geometry (Cosmology.geometry)
        """
        self.rmax2d_arcsec = rmax2d_arcsec
        self._geo = geometry
        super(Uniform, self).__init__()

    @classmethod
    def from_Mhost(cls, *args, **kwargs):
        """

        :param rmax2d_arcsec:
        :param geometry:
        :return:
        """
        raise Exception('Spatial distribution class Uniform not currently implemented for subhalos')

    def draw(self, N, z_plane, rescale=1.0, center_x=0, center_y=0):

        """
        Generate samples distributed uniformly in two dimensions
        :param N: number of objects to generate
        :param z_plane: the redshift where the objects are being placed (used to compute the conversion to physical kpc)
        :param rescale: rescales the maximum rendering radius
        :param center_x: the x-center of the rendering area [arcsec]
        :param center_y: the y-center of the rendering area [arcsec]
        :return: the x and y coordinates sampled in 2D [kpc]
        """
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

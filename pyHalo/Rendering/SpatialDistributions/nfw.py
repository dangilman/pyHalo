import numpy as np
from lenstronomy.LensModel.Profiles.cnfw import CNFW
from pyHalo.Rendering.SpatialDistributions.base import SpatialDistributionBase
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from pyHalo.utilities import inverse_transform_sampling
from scipy.interpolate import interp1d


class ProjectedNFW(SpatialDistributionBase):

    """
    This class approximates sampling from a full 3D NFW profile by
    sampling the projected mass of a cored NFW profile in 2D, and then sampling
    the z coordinate from an NFW profile. This method should approximate sampling from a cored NFW when the maximum
     2D radius is much less than the maximum 3D radius
    """

    def __init__(self, rmax2d_arcsec, rs_arcsec, r_core_arcsec, r_200_arcsec, arcsec_to_kpc):

        """

        :param Rs: the scale radius of the host dark matter halo [kpc]
        :param rmax2d: the maximum projected 2D radius where halos are rendered [arcsec]
        :param rmax3d: the virial radius of the host dark matter halo [kpc]
        :param r_core_parent: the core radius of the host dark matter halo [kpc]
        """

        self._rs_arcsec = rs_arcsec
        self._rmax2d = rmax2d_arcsec
        self._rmax3d = r_200_arcsec
        self._x3dmax = r_200_arcsec / rs_arcsec
        self._rcore_arcsec = r_core_arcsec
        self._cnfw_profile = CNFW()
        self._arcsec_to_kpc = arcsec_to_kpc
        super(ProjectedNFW, self).__init__()

    @classmethod
    def from_Mhost(cls, m_host, zlens, rmax2d_arcsec, r_core_units_rs, lens_cosmo):
        """

        :param m_host:
        :param zlens:
        :param rmax2d_arcsec:
        :param r_core_units_rs:
        :param lens_cosmo:
        :return:
        """
        c_model = ConcentrationDiemerJoyce(lens_cosmo)
        c = c_model.nfw_concentration(m_host, zlens)
        _, rs_mpc, r_200_mpc = lens_cosmo.nfwParam_physical(m_host, c, zlens)
        arcsec_to_kpc = lens_cosmo.cosmo.kpc_proper_per_asec(zlens)
        rs_arcsec = rs_mpc * 1000 / arcsec_to_kpc
        r_200_arcsec = r_200_mpc * 1000 / arcsec_to_kpc
        r_core_arcsec = r_core_units_rs * rs_arcsec
        return ProjectedNFW(rmax2d_arcsec, rs_arcsec, r_core_arcsec, r_200_arcsec, arcsec_to_kpc)

    def draw(self, N, z_plane=None):

        if N == 0:
            return [], [], []

        x2d = np.linspace(1e-3 * self._rmax2d, self._rmax2d, 20000)
        args = (0.0, self._rs_arcsec, 1.0, self._rcore_arcsec)
        rho2d = self._cnfw_profile.density_2d(x2d, *args)
        rho2d_integral = [rho2d[i] * (x2d[i+1]**2 - x2d[i]**2) for i in range(0, len(x2d)-1)]
        function_2d = interp1d(x2d[0:-1], rho2d_integral)
        r2d_arcsec = inverse_transform_sampling(x2d[0:-1], function_2d, (), N)
        r2d_kpc = r2d_arcsec * self._arcsec_to_kpc
        theta = np.random.uniform(0, 2*np.pi, N)
        x_kpc, y_kpc = r2d_kpc * np.cos(theta), r2d_kpc * np.sin(theta)

        x3d = np.linspace(1e-3 * self._rmax2d, self._rmax3d, 20000)
        args = (self._rs_arcsec, 1.0, self._rcore_arcsec)
        rho3d = self._cnfw_profile.density(x3d, *args)
        rho3d_integral = [rho3d[i] * (x3d[i+1]**3 - x3d[i]**3) for i in range(0, len(x3d)-1)]
        function_3d = interp1d(x3d[0:-1], rho3d_integral)
        r3d_arcsec = inverse_transform_sampling(x3d[0:-1], function_3d, (), N)
        r3d_kpc = r3d_arcsec * self._arcsec_to_kpc

        return x_kpc, y_kpc, r3d_kpc

    # def draw(self, N, z_plane=None):
    #
    #     if N == 0:
    #         return [], [], []
    #
    #     x2d = np.linspace(1e-3 * self._rmax2d, self._rmax2d, 50000)
    #     function_2d = self._cnfw_profile.density_2d
    #     args = (0.0, self._rs_arcsec, 1.0, self._rcore_arcsec)
    #     r2d_arcsec = inverse_transform_sampling(x2d, function_2d, args, N)
    #     r2d_kpc = r2d_arcsec * self._arcsec_to_kpc
    #     theta = np.random.uniform(0, 2*np.pi, N)
    #     x_kpc, y_kpc = r2d_kpc * np.cos(theta), r2d_kpc * np.sin(theta)
    #
    #     x3d = np.linspace(1e-3 * self._rmax2d, self._rmax3d, 50000)
    #     function_3d = self._cnfw_profile.density
    #     args = (self._rs_arcsec, 1.0, self._rcore_arcsec)
    #     r3d_arcsec = inverse_transform_sampling(x3d, function_3d, args, N)
    #     r3d_kpc = r3d_arcsec * self._arcsec_to_kpc
    #
    #     return x_kpc, y_kpc, r3d_kpc


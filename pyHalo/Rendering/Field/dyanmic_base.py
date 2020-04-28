import numpy as np

class DynamicBase(object):

    def __init__(self, lensing_mass_func, rendering_args, spatial_parameterization):

        self._lensing_mass_func = lensing_mass_func

        self._geometry = self._lensing_mass_func.geometry

        self._spatial_parameterization = spatial_parameterization

        self._parameterization_args = rendering_args

    def normalization_dNdM(self, ):

    def _volume_element_comoving(self, radius_arcsec, z, delta_z):

        cosmo = self._geometry._cosmo

        dr_comoving = self._geometry.delta_R_comoving(z, delta_z)
        radius_radian = radius_arcsec * cosmo.arcsec
        radius_comoving = radius_radian * cosmo.D_C(z)

        return np.pi * radius_comoving ** 2 * dr_comoving

    def _angle_scale(self, z, zref):

        if z <= zref:
            return 1.
        else:
            cosmo = self._geometry._cosmo
            return cosmo.D_C(zref)/cosmo.D_C(z)

    def render_positions_at_z(self, z, nhalos, rescale, xshift_arcsec, yshift_arcsec):

        if rescale is None:
            x_kpc, y_kpc, r2d_kpc, r3d_kpc = self._spatial_parameterization.draw(nhalos, z)
        else:
            x_kpc, y_kpc, r2d_kpc, r3d_kpc = self._spatial_parameterization.draw(nhalos, z, rescale=rescale)

        if len(x_kpc) > 0:

            kpc_per_asec = self._geometry.kpc_per_arcsec(z)

            x_arcsec = x_kpc * kpc_per_asec ** -1
            y_arcsec = y_kpc * kpc_per_asec ** -1

            x_arcsec += xshift_arcsec
            y_arcsec += yshift_arcsec

            return x_arcsec, y_arcsec, r2d_kpc, r3d_kpc

        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

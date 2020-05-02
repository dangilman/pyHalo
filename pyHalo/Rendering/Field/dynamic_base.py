import numpy as np

class LOSDynamicBase(object):

    def __init__(self, lensing_mass_func, rendering_args, spatial_parameterization):

        self._lensing_mass_func = lensing_mass_func

        self._geometry = self._lensing_mass_func.geometry

        self._spatial_parameterization = spatial_parameterization

        self._parameterization_args = rendering_args

    def rescale_angle(self, z, angle_pad=0.8):

        zlens = self._geometry._zlens
        return self._geometry.background_angle_rescale(z, zlens, angle_pad)

    def _volume_element_comoving(self, z, delta_z, radius_arcsec):

        return self._geometry.volume_element_comoving(z, delta_z, radius=radius_arcsec)

    def render_positions_at_z(self, z, nhalos, rescale, xshift_arcsec, yshift_arcsec):

        if rescale is None:
            x_kpc, y_kpc, r2d_kpc, r3d_kpc = self._spatial_parameterization.draw(nhalos, z)
        else:
            x_kpc, y_kpc, r2d_kpc, r3d_kpc = self._spatial_parameterization.draw(nhalos, z, rescale=rescale)

        if len(x_kpc) > 0:
            kpc_per_asec = self._geometry.kpc_per_arcsec(z)

            x_arcsec = x_kpc * kpc_per_asec ** -1 + xshift_arcsec
            y_arcsec = y_kpc * kpc_per_asec ** -1 + yshift_arcsec
            return x_arcsec, y_arcsec, r2d_kpc, r3d_kpc

        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

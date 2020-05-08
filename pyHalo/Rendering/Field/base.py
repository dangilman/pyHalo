import numpy as np
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Rendering.render_base import RenderingBase

class LOSBase(RenderingBase):

    def __init__(self, lensing_mass_func, geometry_render, rendering_args, spatial_parameterization, lens_plane_redshifts, delta_zs):

        self.halo_mass_function = lensing_mass_func

        self._spatial_parameterization = spatial_parameterization

        self.rendering_args = rendering_args

        self.lens_plane_redshifts, self.delta_zs = lens_plane_redshifts, delta_zs

        super(LOSBase, self).__init__(geometry_render)

    @staticmethod
    def two_halo_boost(z, delta_z, host_m200, zlens, lensing_mass_function_class):

        boost = 1.
        if lensing_mass_function_class._two_halo_term and z == zlens:
            rmax = lensing_mass_function_class._cosmo.T_xy(zlens - delta_z, zlens)
            boost = lensing_mass_function_class.two_halo_boost(host_m200, z, rmax=rmax)

        return boost

    def render_positions_at_z(self, z, nhalos, xshift_arcsec, yshift_arcsec):

        x_kpc, y_kpc, r2d_kpc, r3d_kpc = self._spatial_parameterization.draw(nhalos, z)

        if len(x_kpc) > 0:
            kpc_per_asec = self.geometry.kpc_per_arcsec(z)

            x_arcsec = x_kpc * kpc_per_asec ** -1 + xshift_arcsec
            y_arcsec = y_kpc * kpc_per_asec ** -1 + yshift_arcsec
            return x_arcsec, y_arcsec, r2d_kpc, r3d_kpc

        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

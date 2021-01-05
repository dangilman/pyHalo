import numpy as np
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Rendering.render_base import RenderingBase
from abc import ABC, abstractmethod

class LineOfSightBase(ABC):

    type = 'LOS'

    def __init__(self, rendering_type, halo_mass_function_class, geometry_class, lens_plane_redshifts, delta_zs):

        self.halo_mass_function = halo_mass_function_class
        self.lens_plane_redshifts = lens_plane_redshifts
        self.delta_zs = delta_zs
        self.geometry = geometry_class



    @staticmethod
    def two_halo_boost(z, delta_z, host_m200, zlens, lensing_mass_function_class):

        boost = 1.
        if z == zlens:
            rmax = lensing_mass_function_class.geometry._cosmo.D_C_transverse(zlens) - \
                   lensing_mass_function_class.geometry._cosmo.D_C_transverse(zlens - delta_z)
            boost = lensing_mass_function_class.two_halo_boost(host_m200, z, rmax=rmax)

        return boost

    def render_masses(self, *args, **kwargs):

    def render_positions_at_z(self, z, nhalos, xshift_arcsec, yshift_arcsec):

        x_kpc, y_kpc, r3d_kpc = self._spatial_parameterization.draw(nhalos, z)

        if len(x_kpc) > 0:
            kpc_per_asec = self.geometry.kpc_per_arcsec(z)

            x_arcsec = x_kpc * kpc_per_asec ** -1 + xshift_arcsec
            y_arcsec = y_kpc * kpc_per_asec ** -1 + yshift_arcsec
            return x_arcsec, y_arcsec, r3d_kpc

        else:
            return np.array([]), np.array([]), np.array([])

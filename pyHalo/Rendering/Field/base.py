class LOSBase(object):

    def __init__(self, lensing_mass_func, rendering_args, spatial_parameterization):

        self._lensing_mass_func = lensing_mass_func

        self._geometry = self._lensing_mass_func.geometry

        self._spatial_parameterization = spatial_parameterization

        self._parameterization_args = rendering_args

    def render_positions_at_z(self, z, nhalos):

        x_kpc, y_kpc, r2d_kpc, r3d_kpc = self._spatial_parameterization.draw(nhalos, z)

        kpc_per_asec = self._geometry.kpc_per_arcsec(z)
        x_arcsec = x_kpc * kpc_per_asec ** -1
        y_arcsec = y_kpc * kpc_per_asec ** -1

        return x_arcsec, y_arcsec, r2d_kpc, r3d_kpc

from pyHalo.Rendering.Field.PowerLaw.powerlaw_base import PowerLawBase
from pyHalo.Spatial.keywords import LOS_spatial_global
from pyHalo.Spatial.uniform import LensConeUniform

class LOSPowerLaw(PowerLawBase):

    def __init__(self, args, halo_mass_function_class, geometry_class, lens_plane_redshifts, delta_zs):

        rendering_args = self.keyword_parse(args, halo_mass_function_class)

        self._zlens = halo_mass_function_class.geometry._zlens

        spatial_args = LOS_spatial_global(args)

        spatial_parameterization = LensConeUniform(spatial_args['cone_opening_angle'], geometry_class)

        super(LOSPowerLaw, self).__init__(halo_mass_function_class, geometry_class, rendering_args,
                                          spatial_parameterization, lens_plane_redshifts, delta_zs)



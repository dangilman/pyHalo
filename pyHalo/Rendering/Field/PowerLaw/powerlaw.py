from pyHalo.Rendering.Field.PowerLaw.powerlaw_base import PowerLawBase
from pyHalo.Spatial.keywords import LOS_spatial_global
from pyHalo.Spatial.uniform import LensConeUniform

class LOSPowerLaw(PowerLawBase):

    def __init__(self, args, lensing_mass_func, geometry_render, lens_plane_redshifts, delta_zs):

        rendering_args = self.keyword_parse(args, lensing_mass_func)

        self._zlens = lensing_mass_func.geometry._zlens

        spatial_args = LOS_spatial_global(args)

        spatial_parameterization = LensConeUniform(spatial_args['cone_opening_angle'], geometry_render)

        super(LOSPowerLaw, self).__init__(lensing_mass_func, geometry_render, rendering_args,
                                                 spatial_parameterization, lens_plane_redshifts, delta_zs)

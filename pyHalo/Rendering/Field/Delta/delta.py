from pyHalo.Spatial.keywords import LOS_spatial_global
from pyHalo.Spatial.uniform import LensConeUniform
from pyHalo.Rendering.Field.Delta.delta_base import DeltaBase

class LOSDelta(DeltaBase):

    def __init__(self, args, lensing_mass_func, geometry_render, log_min_mass, lens_plane_redshifts, delta_zs):

        self._rendering_args = self.keyword_parse(args, lensing_mass_func)

        self._zlens = lensing_mass_func.geometry._zlens

        self._minimum_mass = 10**log_min_mass

        spatial_args = LOS_spatial_global(args)

        spatial_parameterization = LensConeUniform(spatial_args['cone_opening_angle'], geometry_render)

        super(LOSDelta, self).__init__(lensing_mass_func, geometry_render, self._rendering_args,
                                          spatial_parameterization, lens_plane_redshifts, delta_zs)


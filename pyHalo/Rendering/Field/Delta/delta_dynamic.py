from pyHalo.Rendering.Field.Delta.delta_base import DeltaBase

from pyHalo.Spatial.uniform import Uniform
from pyHalo.Spatial.uniform import LensConeUniform

class LOSDeltaDynamic(DeltaBase):

    def __init__(self, args, lensing_mass_func, geometry_render, aperture_radius, global_render,
                 log_min_mass, lens_plane_redshifts, delta_zs):

        self._rendering_args = self.keyword_parse(args, lensing_mass_func)

        minimum_mass = 10**log_min_mass

        if global_render:
            # factor of two because LensConeUniform expects cone_opening_angle
            spatial_parameterization = LensConeUniform(2*aperture_radius, geometry_render)
        else:
            spatial_parameterization = Uniform(aperture_radius, geometry_render)

        super(LOSDeltaDynamic, self).__init__(lensing_mass_func, geometry_render, self._rendering_args,
                                                 spatial_parameterization, minimum_mass,
                                              lens_plane_redshifts, delta_zs)



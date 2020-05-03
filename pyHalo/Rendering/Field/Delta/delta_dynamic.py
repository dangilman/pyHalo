from pyHalo.Rendering.Field.Delta.delta_base import DeltaBase

from pyHalo.Spatial.uniform import Uniform

class LOSDeltaDynamic(DeltaBase):

    def __init__(self, args, lensing_mass_func, aperture_size, log_min_mass, lens_plane_redshifts, delta_zs):

        self._rendering_args = self.keyword_parse(args, lensing_mass_func)

        minimum_mass = 10**log_min_mass

        spatial_parameterization = Uniform(aperture_size, lensing_mass_func.geometry)

        super(LOSDeltaDynamic, self).__init__(lensing_mass_func, self._rendering_args,
                                                 spatial_parameterization, minimum_mass,
                                              lens_plane_redshifts, delta_zs)



from pyHalo.Rendering.Field.PowerLaw.powerlaw_base import PowerLawBase
from pyHalo.Spatial.uniform import Uniform
from pyHalo.Spatial.uniform import LensConeUniform

class LOSPowerLawDynamic(PowerLawBase):

    def __init__(self, args, lensing_mass_func, geometry_render, aperture_size, global_render,
                 lens_plane_redshifts, delta_zs):

        self._rendering_args = self.keyword_parse(args, lensing_mass_func)
        self._args_base = args
        self._zlens = lensing_mass_func.geometry._zlens

        if global_render:
            # factor of two because LensConeUniform expects cone_opening_angle
            spatial_paramterization = LensConeUniform(2*aperture_size, geometry_render)

        else:
            spatial_paramterization = Uniform(aperture_size, geometry_render)

        super(LOSPowerLawDynamic, self).__init__(lensing_mass_func, geometry_render, self._rendering_args,
                                          spatial_paramterization, lens_plane_redshifts, delta_zs)


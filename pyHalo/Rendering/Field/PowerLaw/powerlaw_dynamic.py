from pyHalo.Rendering.Field.PowerLaw.powerlaw_base import PowerLawBase
from pyHalo.Spatial.uniform import Uniform

class LOSPowerLawDynamic(PowerLawBase):

    def __init__(self, args, lensing_mass_func, geometry_render, aperture_radius, lens_plane_redshifts, delta_zs):

        self._rendering_args = self.keyword_parse(args, lensing_mass_func)

        self._zlens = lensing_mass_func.geometry._zlens

        spatial_parameterization = Uniform(aperture_radius, geometry_render)

        super(LOSPowerLawDynamic, self).__init__(lensing_mass_func, geometry_render, self._rendering_args,
                                          spatial_parameterization, lens_plane_redshifts, delta_zs)


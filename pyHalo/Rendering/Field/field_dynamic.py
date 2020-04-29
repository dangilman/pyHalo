from copy import deepcopy

import numpy as np


from pyHalo.Rendering.Field.dynamic_base import LOSDynamicBase
from pyHalo.Rendering.parameterizations import BrokenPowerLaw
from pyHalo.Rendering.keywords import LOS_powerlaw_mfunc
from pyHalo.Rendering.Field.LOS_normalizations import powerlaw_normalization

from pyHalo.Spatial.uniform import Uniform

class LOSPowerLawDynamic(LOSDynamicBase):

    def __init__(self, args, lensing_mass_func,
                 aperture_x, aperture_y, aperture_size):

        self._rendering_args = LOS_powerlaw_mfunc(args, lensing_mass_func)

        self._zlens = lensing_mass_func.geometry._zlens
        spatial_parameterization = Uniform(aperture_size, lensing_mass_func.geometry)

        self._aperture_size, self._aperture_x, self._aperture_y = aperture_size, aperture_x, aperture_y

        super(LOSPowerLawDynamic, self).__init__(lensing_mass_func, self._rendering_args,
                                          spatial_parameterization)

    def render_masses(self, zi, delta_zi):

        rescale_angle = self.rescale_angle(zi, self._zlens)
        aperture_size = self._aperture_size * rescale_angle
        volume_element_comoving = self._volume_element_comoving(zi, delta_zi, aperture_size)

        norm = powerlaw_normalization(zi, delta_zi, self._zlens, self._lensing_mass_func, self._rendering_args,
                                      volume_element_comoving)

        pargs = deepcopy(self._rendering_args)
        plaw_index = self._lensing_mass_func.plaw_index_z(zi)

        pargs.update({'normalization': norm})
        pargs.update({'power_law_index': plaw_index})
        mfunc = BrokenPowerLaw(**pargs)

        m = mfunc.draw()

        return m

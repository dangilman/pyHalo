from copy import deepcopy

import numpy as np

from pyHalo.Rendering.Field.base import LOSBase
from pyHalo.Rendering.parameterizations import BrokenPowerLaw
from pyHalo.Rendering.keywords import LOS_powerlaw_mfunc
from pyHalo.Spatial.keywords import LOS_spatial_global
from pyHalo.Spatial.uniform import LensConeUniform
from pyHalo.Rendering.Field.LOS_normalizations import powerlaw_normalization

class LOSPowerLaw(LOSBase):

    def __init__(self, args, lensing_mass_func):

        self._rendering_args = LOS_powerlaw_mfunc(args, lensing_mass_func)
        self._zlens = lensing_mass_func.geometry._zlens
        spatial_args = LOS_spatial_global(args)
        spatial_parameterization = LensConeUniform(spatial_args['cone_opening_angle'], lensing_mass_func.geometry)
        super(LOSPowerLaw, self).__init__(lensing_mass_func, self._rendering_args,
                                                 spatial_parameterization)

    def render_masses(self, zi, delta_zi):

        volume_element_comoving = self._volume_element_comoving(zi, delta_zi)

        norm = powerlaw_normalization(zi, delta_zi, self._zlens, self._lensing_mass_func, self._rendering_args,
                                      volume_element_comoving)

        pargs = deepcopy(self._rendering_args)
        plaw_index = self._lensing_mass_func.plaw_index_z(zi)

        pargs.update({'normalization': norm})
        pargs.update({'power_law_index': plaw_index})
        mfunc = BrokenPowerLaw(**pargs)

        m = mfunc.draw()

        return m

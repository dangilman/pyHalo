from copy import copy

import numpy as np

from pyHalo.Rendering.Field.base import LOSBase
from pyHalo.Rendering.keywords import LOS_delta_mfunc
from pyHalo.Spatial.keywords import LOS_spatial_global
from pyHalo.Spatial.uniform import LensConeUniform
from pyHalo.Rendering.Field.LOS_normalizations import delta_function_normalization

class LOSDelta(LOSBase):

    def __init__(self, args, lensing_mass_func, log_min_mass):

        self._rendering_args = LOS_delta_mfunc(args, lensing_mass_func)
        self._zlens = lensing_mass_func.geometry._zlens
        self._minimum_mass = 10**log_min_mass
        spatial_args = LOS_spatial_global(args)
        spatial_parameterization = LensConeUniform(spatial_args['cone_opening_angle'], lensing_mass_func.geometry)
        super(LOSDelta, self).__init__(lensing_mass_func, self._rendering_args,
                                          spatial_parameterization)

    def render_masses(self, zi, delta_zi):

        object_mass = 10 ** self._parameterization_args['logM_delta']

        if object_mass < self._minimum_mass:
            component_fraction = 0.
        else:
            component_fraction = self._parameterization_args['mass_fraction']

        volume_element_comoving = self._volume_element_comoving(zi, delta_zi)
        nobjects = delta_function_normalization(zi, delta_zi, object_mass, component_fraction,
                                                self._zlens, self._lensing_mass_func, self._parameterization_args,
                                                volume_element_comoving)

        m = np.array([object_mass] * nobjects)

        return m

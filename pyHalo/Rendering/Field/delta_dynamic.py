from copy import deepcopy

import numpy as np

from pyHalo.Rendering.Field.dynamic_base import LOSDynamicBase
from pyHalo.Rendering.keywords import LOS_delta_mfunc
from pyHalo.Rendering.Field.LOS_normalizations import delta_function_normalization

from pyHalo.Spatial.uniform import Uniform

class LOSDeltaDynamic(LOSDynamicBase):

    def __init__(self, args, lensing_mass_func, aperture_size, log_min_mass):

        self._rendering_args = LOS_delta_mfunc(args, lensing_mass_func)
        self._minimum_mass = 10**log_min_mass
        self._zlens = lensing_mass_func.geometry._zlens
        spatial_parameterization = Uniform(aperture_size, lensing_mass_func.geometry)

        self._aperture_size = aperture_size

        super(LOSDeltaDynamic, self).__init__(lensing_mass_func, self._rendering_args,
                                                 spatial_parameterization)

    def render_masses(self, zi, delta_zi, aperture_radius):

        object_mass = 10 ** self._parameterization_args['logM_delta']

        if object_mass < self._minimum_mass:
            component_fraction = 0.
        else:
            component_fraction = self._parameterization_args['mass_fraction']

        volume_element_comoving = self._volume_element_comoving(zi, delta_zi, aperture_radius)

        nobjects = delta_function_normalization(zi, delta_zi, object_mass, component_fraction,
                                                self._zlens, self._lensing_mass_func, self._parameterization_args,
                                                volume_element_comoving)
        m = np.array([object_mass] * nobjects)

        return m

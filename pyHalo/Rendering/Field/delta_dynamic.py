from copy import deepcopy

import numpy as np

from pyHalo.Rendering.Field.dynamic_base import LOSDynamicBase
from pyHalo.Rendering.keywords import LOS_powerlaw_mfunc
from pyHalo.Rendering.Field.LOS_normalizations import delta_function_normalization

from pyHalo.Spatial.uniform import Uniform


class LOSDeltaDynamic(LOSDynamicBase):

    def __init__(self, args, lensing_mass_func,
                 aperture_x, aperture_y, aperture_size):

        self._rendering_args = LOS_powerlaw_mfunc(args, lensing_mass_func)

        self._zlens = lensing_mass_func.geometry._zlens
        spatial_parameterization = Uniform(aperture_size, lensing_mass_func.geometry)

        self._aperture_size, self._aperture_x, self._aperture_y = aperture_size, aperture_x, aperture_y

        super(LOSDeltaDynamic, self).__init__(lensing_mass_func, self._rendering_args,
                                                 spatial_parameterization)

    def render_masses(self, zi, delta_zi):

        object_mass = 10 ** self._parameterization_args['logM_delta']
        component_fraction = self._parameterization_args['mass_fraction']

        rescale_angle = self.rescale_angle(zi, self._zlens)
        aperture_size = self._aperture_size * rescale_angle
        volume_element_comoving = self._volume_element_comoving(zi, delta_zi, aperture_size)

        nobjects = delta_function_normalization(zi, delta_zi, object_mass, component_fraction,
                                                self._zlens, self._lensing_mass_func, self._parameterization_args,
                                                volume_element_comoving)
        m = np.array([object_mass] * nobjects)

        return m

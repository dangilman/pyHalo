from copy import copy

import numpy as np

from pyHalo.Rendering.Field.base import LOSBase
from pyHalo.Rendering.keywords import LOS_delta_mfunc
from pyHalo.Spatial.keywords import LOS_spatial_global
from pyHalo.Spatial.uniform import LensConeUniform

class LOSDelta(LOSBase):

    def __init__(self, args, lensing_mass_func, redshifts, delta_zs):

        rendering_args = LOS_delta_mfunc(args, lensing_mass_func)
        spatial_args = LOS_spatial_global(args, lensing_mass_func)
        spatial_parameterization = LensConeUniform(spatial_args['cone_opening_angle'],
                                                   lensing_mass_func.geometry)

        self._lensing_mass_func = lensing_mass_func
        self._redshifts, self._deltazs = redshifts, delta_zs
        super(LOSDelta, self).__init__(lensing_mass_func, rendering_args, spatial_parameterization)

    def __call__(self):

        redshifts = []
        for i, (zi, delta_zi) in enumerate(zip(self._redshifts, self._deltazs)):

            pargs = copy(self._parameterization_args)

            nobjects = self._lensing_mass_func.\
            dN_comoving_deltaFunc(10**pargs['logM_delta'], zi, delta_zi, pargs['mass_fraction'])
            nobjects *= pargs['LOS_normalization']
            nobjects = np.random.poisson(nobjects)

            mi = np.array([10 ** pargs['logM_delta']] * nobjects)
            xi, yi, r2di, r3di = self.render_positions_at_z(zi, len(masses))
            new_redshifts = [zi] * len(mi)
            if i == 0:
                masses = mi
                x, yi = xi, yi
                r2d = r2di
                r3d = r3di
                redshifts = new_redshifts

            else:
                masses = np.append(masses, mi)
                x = np.append(x, xi)
                y = np.append(y, yi)
                r2d = np.append(r2d, r2di)
                r3d = np.append(r3d, r3di)
                redshifts += new_redshifts

        return np.array(masses), np.array(x), np.array(y), np.array(r2d), \
               np.array(r3d), np.array(redshifts)

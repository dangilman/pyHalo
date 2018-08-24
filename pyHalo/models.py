import numpy as np
from pyHalo.massfunc import *
from pyHalo.Spatial.uniform import LensConeUniform
from pyHalo.Spatial.nfw import NFW_3D
from copy import copy

class MainLensPowerLaw(object):

    required_spatial_kwargs = []
    required_parameterization_args = []

    def __init__(self, parameterization_args, spatial_args):

        self._mass_func_parameterization = BrokenPowerLaw(**parameterization_args)
        self._spatial_parameterization = NFW_3D(**spatial_args)

    def __call__(self):

        masses = self._mass_func_parameterization.draw()
        x, y, r2d, r3d = self._spatial_parameterization.draw(len(masses))

        return np.array(masses), np.array(x), np.array(y), np.array(r2d), np.array(r3d), np.array([None]*len(masses))

class LOSPowerLaw(object):

    required_parameterization_args = []

    def __init__(self, lensing_mass_func, parameterization_args, zlens, zstep):

        zmin, zmax = parameterization_args['zmin'], parameterization_args['zmax']
        self._redshift_range, self._delta_z = _redshift_range_LOS(zmin,zmax,zlens,zstep)
        self._lensing_mass_func = lensing_mass_func

        self._spatial_parameterization = LensConeUniform(lensing_mass_func.geometry.cone_opening_angle,
                                                         lensing_mass_func.geometry)
        self._parameterization_args = parameterization_args

    def __call__(self):

        redshifts = []
        zstart = self._redshift_range[0]

        pargs = copy(self._parameterization_args)
        pargs['Nhalos_mean'] = self._lensing_mass_func.n_objects_at_z(zstart, self._delta_z)
        pargs['power_law_index'] = self._lensing_mass_func.plaw_index_z(zstart, self._delta_z)

        mfunc = BrokenPowerLaw(**pargs)

        masses = mfunc.draw()
        x, y, r2d, r3d = self._render_positions_atz(zstart, len(masses))
        if len(masses) > 0:
            redshifts += [zstart] * len(masses)

        for z in self._redshift_range[1:-1]:

            pargs = copy(self._parameterization_args)
            pargs['power_law_index'] = self._lensing_mass_func.plaw_index_z(z, self._delta_z)
            pargs['normalization'] = self._lensing_mass_func.norm_at_z(z, self._delta_z)

            mfunc = BrokenPowerLaw(**pargs)

            m = mfunc.draw()
            xi, yi, r2di, r3di = self._render_positions_atz(z, len(masses))

            masses = np.append(masses,m)
            x = np.append(x, xi)
            y = np.append(y, yi)
            r2d = np.append(r2d, r2di)
            r3d = np.append(r3d, r3di)
            if len(m) > 0:
                redshifts += [z]*len(m)


        return np.array(masses), np.array(x), np.array(y), np.array(r2d), np.array(r3d), np.array(redshifts)


    def _render_z(self,z):

        m = self._render_mass_atz(z)
        x, y, r2d, r3d = self._render_positions_atz(z, len(m))

        return m, x, y, r2d, r3d

    def _render_positions_atz(self, z, nhalos):

        x, y, r2d, r3d = self._spatial_parameterization.draw(nhalos, z)

        return x, y, r2d, r3d

def _redshift_range_LOS(zmin, zmax, zlens, zstep):

    nsteps = (zmax - zmin) * zstep**-1
    zvalues = np.linspace(zmin, zmax, nsteps)

    # remove anything too close to main lens plane
    zvalues = zvalues[np.where(np.absolute(zvalues - zlens) > zstep)]

    return zvalues, zstep






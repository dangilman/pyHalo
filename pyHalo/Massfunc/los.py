from copy import copy

import numpy as np
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw
from pyHalo.Spatial.uniform import LensConeUniform
from pyHalo.defaults import default_zstart, default_z_round

class LOSPowerLaw(object):

    def __init__(self, args, lensing_mass_func, zlens, zstep):

        self._lensing_mass_func = lensing_mass_func
        spatial_args, parameterization_args = self._set_kwargs(args)

        zmin, zmax = parameterization_args['zmin'], parameterization_args['zmax']
        self._redshift_range, self._delta_z = _redshift_range_LOS(zmin,zmax,zlens,zstep)

        self._spatial_parameterization = LensConeUniform(spatial_args['cone_opening_angle'],
                                                         lensing_mass_func.geometry)
        self._parameterization_args = parameterization_args

    def __call__(self):

        redshifts = []
        zstart = self._redshift_range[0]

        pargs = copy(self._parameterization_args)

        pargs['normalization'] = self._lensing_mass_func.norm_at_z_biased(zstart, self._parameterization_args['parent_m200'],
                                                                          self._delta_z)

        pargs['normalization'] *= pargs['LOS_normalization']

        #pargs['normalization'] = self._lensing_mass_func.norm_at_z(zstart, delta_z = self._delta_z)
        pargs['power_law_index'] = self._lensing_mass_func.plaw_index_z(zstart)

        mfunc = BrokenPowerLaw(**pargs)

        masses = mfunc.draw()
        x, y, r2d, r3d = self._render_positions_atz(zstart, len(masses))

        if len(masses) > 0:
            redshifts += [zstart] * len(masses)

        for z in self._redshift_range[1:-1]:

            pargs = copy(self._parameterization_args)
            pargs['power_law_index'] = self._lensing_mass_func.plaw_index_z(z)
            pargs['normalization'] = self._lensing_mass_func.norm_at_z_biased(z, self._parameterization_args[
                'parent_m200'], self._delta_z)
            pargs['normalization'] *= pargs['LOS_normalization']
            #pargs['normalization'] = self._lensing_mass_func.norm_at_z(z, delta_z=self._delta_z)
            mfunc = BrokenPowerLaw(**pargs)

            m = mfunc.draw()

            xi, yi, r2di, r3di = self._render_positions_atz(z, len(m))

            masses = np.append(masses,m)
            x = np.append(x, xi)
            y = np.append(y, yi)
            r2d = np.append(r2d, r2di)
            r3d = np.append(r3d, r3di)
            if len(m) > 0:
                redshifts += [z]*len(m)

        redshifts = self._round_redshifts(redshifts, self._lensing_mass_func.geometry._zlens)

        return np.array(masses), np.array(x), np.array(y), np.array(r2d), np.array(r3d), np.array(redshifts)

    def _round_redshifts(self,zvalues,zlens):

        zvalues = np.round(zvalues,default_z_round)
        zvalues[np.where(np.absolute(zvalues - zlens) <= 0.01)] = zlens

        return zvalues

    def _set_kwargs(self, args):

        args_mfunc = self._mfunc(args)
        args_spatial = self._spatial(args)

        return args_spatial, args_mfunc

    def _spatial(self,args):

        args_spatial ={}
        args_spatial['cone_opening_angle'] = args['cone_opening_angle']
        return args_spatial

    def _mfunc(self,args):

        args_mfunc = {}
        required_keys = ['zmin', 'zmax', 'log_m_break', 'log_mlow_los',
                         'log_mhigh_los', 'parent_m200', 'LOS_normalization']

        for key in required_keys:

            if key == 'LOS_normalization':

                if key in args.keys():
                    args_mfunc['LOS_normalization'] = args[key]
                else:
                    args_mfunc['LOS_normalization'] = 1
                continue

            if key == 'log_mlow_los':

                if key in args.keys():
                    args_mfunc['log_mlow'] = args[key]
                else:
                    args_mfunc['log_mlow'] = args['log_mlow']
                continue

            if key == 'log_mhigh_los':

                if key in args.keys():
                    args_mfunc['log_mhigh'] = args[key]
                else:
                    args_mfunc['log_mhigh'] = args['log_mhigh']
                continue

            try:
                args_mfunc[key] = args[key]
            except:
                if key == 'zmin':
                    args_mfunc['zmin'] = default_zstart
                else:
                    args_mfunc['zmax'] = self._lensing_mass_func.geometry._zsource - default_zstart

        if args_mfunc['log_m_break'] == 0:
            args_mfunc['break_index'] = 0
            args_mfunc['c_scale'] = 0
            args_mfunc['c_power'] = 0

        else:

            try:
                args_mfunc['break_index'] = args['break_index']
                args_mfunc['c_scale'] = args['c_scale']
                args_mfunc['c_power'] = args['c_power']
            except:
                raise ValueError('must specify a value for "break_index, c_scale, c_power" if log_m_break != 0 '
                                 '(because you are specifying a WDM scenario in which the concentration and mass function'
                                 'slope  of halos is affected')

        return args_mfunc

    def _render_z(self,z):

        m = self._render_mass_atz(z)
        x, y, r2d, r3d = self._render_positions_atz(z, len(m))

        return m, x, y, r2d, r3d

    def _render_positions_atz(self, z, nhalos):

        x, y, r2d, r3d = self._spatial_parameterization.draw(nhalos, z)

        return x, y, r2d, r3d

def _redshift_range_LOS(zmin, zmax, zlens, zstep):

    nsteps = int((zmax - zmin) * zstep**-1)
    zvalues = np.linspace(zmin, zmax, nsteps)

    return zvalues, zstep

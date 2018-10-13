from copy import copy

import numpy as np
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw
from pyHalo.Spatial.uniform import LensConeUniform
from pyHalo.defaults import default_zstart, default_z_round, default_z_step

class LOSDelta(object):

    def __init__(self, args, lensing_mass_func, zlens, zstep):

        self._lensing_mass_func = lensing_mass_func
        spatial_args, parameterization_args = self._set_kwargs(args)

        zmin, zmax = parameterization_args['zmin'], parameterization_args['zmax']
        self._redshift_range, self._delta_z = _redshift_range_LOS(zmin, zmax, zlens, zstep)

        self._spatial_parameterization = LensConeUniform(spatial_args['cone_opening_angle'],
                                                         lensing_mass_func.geometry)
        self._parameterization_args = parameterization_args

    def __call__(self):

        redshifts = []
        zstart = self._redshift_range[0]

        pargs = copy(self._parameterization_args)

        nobjects = self._lensing_mass_func.dN_dV_comoving_deltaFunc(pargs['M'], zstart, pargs['mass_fraction'])
        nobjects *= pargs['LOS_normalization']
        masses = np.array([pargs['M']]*nobjects)

        x, y, r2d, r3d = self._render_positions_atz(zstart, len(masses))

        if len(masses) > 0:
            redshifts += [zstart] * len(masses)

        for z in self._redshift_range[1:-1]:

            pargs = copy(self._parameterization_args)

            nobjects = self._lensing_mass_func.dN_dV_comoving_deltaFunc(pargs['M'], z,
                                                                        pargs['mass_fraction'])
            nobjects *= pargs['LOS_normalization']
            m = np.array([pargs['M']] * nobjects)

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


    def _render_z(self,z):

        m = self._render_mass_atz(z)
        x, y, r2d, r3d = self._render_positions_atz(z, len(m))

        return m, x, y, r2d, r3d

    def _render_positions_atz(self, z, nhalos):

        x, y, r2d, r3d = self._spatial_parameterization.draw(nhalos, z)

        return x, y, r2d, r3d

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
        required_keys = ['zmin', 'zmax', 'M', 'mass_fraction', 'LOS_normalization']

        for key in required_keys:

            if key == 'LOS_normalization':

                if key in args.keys():
                    args_mfunc['LOS_normalization'] = args[key]
                else:
                    args_mfunc['LOS_normalization'] = 1
                continue

            try:
                args_mfunc[key] = args[key]
            except:
                if key == 'zmin':
                    args_mfunc['zmin'] = default_zstart
                else:
                    args_mfunc['zmax'] = self._lensing_mass_func.geometry._zsource - default_zstart

        return args_mfunc

class LOSPowerLaw(object):

    def __init__(self, args, lensing_mass_func, zlens, zstep):

        self._lensing_mass_func = lensing_mass_func
        spatial_args, parameterization_args = self._set_kwargs(args)

        zmin, zmax = parameterization_args['zmin'], parameterization_args['zmax']
        self._redshift_range, self._delta_z = _redshift_range_LOS(zmin,zmax,zlens,
                                                                  default_z_step, zstep)

        self._spatial_parameterization = LensConeUniform(spatial_args['cone_opening_angle'],
                                                         lensing_mass_func.geometry)
        self._parameterization_args = parameterization_args

    def _draw(self, norm, plaw_index, pargs, z_current):

        pargs_new = copy(pargs)
        pargs_new.update({'normalization': norm})
        pargs_new.update({'power_law_index': plaw_index})

        mfunc = BrokenPowerLaw(**pargs_new)

        masses = mfunc.draw()

        x, y, r2d, r3d = self._render_positions_atz(z_current, len(masses))

        if len(masses) > 0:
            redshifts = [z_current] * len(masses)
        else:
            redshifts = []

        return np.array(masses), np.array(x), np.array(y), np.array(r2d), np.array(r3d), np.array(redshifts)

    def __call__(self):

        for idx, zcurrent in enumerate(self._redshift_range):

            norm = self._lensing_mass_func.norm_at_z_biased(zcurrent, self._parameterization_args['parent_m200'],
                                                         self._delta_z[idx])

            norm *= self._parameterization_args['LOS_normalization']

            plaw_idx = self._lensing_mass_func.plaw_index_z(zcurrent)

            if idx == 0:

                masses, x, y, r2d, r3d, z = self._draw(norm, plaw_idx,
                                                       self._parameterization_args, zcurrent)


            else:

                mi, xi, yi, r2di, r3di, zi = self._draw(norm, plaw_idx, self._parameterization_args, zcurrent)
                masses = np.append(masses, mi)
                x, y = np.append(x, xi), np.append(y, yi)
                r2d, r3d = np.append(r2d, r2di), np.append(r3d, r3di)
                z = np.append(z, zi)

        z = self._round_redshifts(z, self._lensing_mass_func.geometry._zlens)

        return masses, x, y, r2d, r3d, z

    def _round_redshifts(self,zvalues,zlens):

        #zvalues = np.array(zvalues)

        #front_idx = np.where(zvalues < zlens)
        #back_idx = np.where(zvalues > zlens)

        #zvals_front = np.round(zvalues[front_idx], 2)
        #last_z_front = zvals_front[np.where(zlens - zvals_front >0)][-1]
        #zvals_front[np.where(zvals_front == zlens)] = last_z_front

        #zvals_back = np.round(zvalues[back_idx], 2)
        #first_z_back = zvals_back[np.where(-zlens + zvals_back > 0)][0]
        #zvals_back[np.where(zvals_back == zlens)] = first_z_back

        #zvalues = np.append(zvals_front, zvals_back)

        zvalues[np.where(np.absolute(zvalues - zlens) <= 0.005)] = zlens

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

def _redshift_range_LOS(zmin, zmax, zlens, zstep, zstep_fine):

    twohalo_range = 0.005

    nsteps_front = np.round((zlens - twohalo_range - zmin) * zstep ** -1)
    zvals_front = np.linspace(zmin, zlens - twohalo_range, nsteps_front)[:-1]
    delta_z = [zvals_front[1] - zvals_front[0]] * len(zvals_front)

    nstep_back = np.round((zmax - twohalo_range - zlens) * delta_z[0] ** -1)
    zvals_back = np.linspace(zlens + twohalo_range, zmax, nstep_back)[1:]

    zvals_fine_front = np.linspace(zlens - twohalo_range, zlens - zstep_fine, np.round(0.1 * zstep_fine ** -1))
    zvals_fine_back = np.linspace(zlens + zstep_fine, zlens + twohalo_range, np.round(0.1 * zstep_fine ** -1))

    zvals_fine = np.append(zvals_fine_front, zvals_fine_back)

    delta_z += [zvals_fine_front[1] - zvals_fine_front[0]] * len(zvals_fine_front)
    delta_z += [zvals_fine_back[1] - zvals_fine_back[0]] * len(zvals_fine_back)
    delta_z += [zvals_back[1] - zvals_back[0]] * len(zvals_back)

    zvalues = np.append(np.append(zvals_front, zvals_fine), zvals_back)

    return zvalues, np.array(delta_z)


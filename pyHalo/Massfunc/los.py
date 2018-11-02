from copy import copy

import numpy as np
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw
from pyHalo.Spatial.uniform import LensConeUniform
from pyHalo.defaults import default_zstart, default_z_round, default_z_step

class LOS(object):

    def __init__(self, args, lensing_mass_func, lens_cosmo = None):

        self._lensing_mass_func = lensing_mass_func
        self._lens_cosmo = lens_cosmo

        spatial_args, parameterization_args = self._set_kwargs(args)

        zmin, zmax = parameterization_args['zmin'], parameterization_args['zmax']
        self._redshift_range = _redshift_range_LOS(zmin, zmax, default_z_step)

        self._spatial_parameterization = LensConeUniform(spatial_args['cone_opening_angle'],
                                                         lensing_mass_func.geometry)
        self._parameterization_args = parameterization_args

    def _spatial(self,args):

        args_spatial ={}
        args_spatial['cone_opening_angle'] = args['cone_opening_angle']
        return args_spatial


    def _set_kwargs(self, args):

        args_mfunc = self._mfunc(args)
        args_spatial = self._spatial(args)

        return args_spatial, args_mfunc

    def _render_z(self,z):

        m = self._render_mass_atz(z)
        x, y, r2d, r3d = self._render_positions_atz(z, len(m))

        return m, x, y, r2d, r3d

    def _render_positions_atz(self, z, nhalos):

        x, y, r2d, r3d = self._spatial_parameterization.draw(nhalos, z)

        return x, y, r2d, r3d

    def _render_positions_localized(self, z, nhalos):

        x, y, r2d, r3d = self._spatial_parameterization.draw(nhalos, z)

        return x, y, r2d, r3d

class LOSDelta(LOS):

    def __call__(self):

        redshifts = []
        zstart = self._redshift_range[0]

        delta_z = self._redshift_range[1] - zstart
        pargs = copy(self._parameterization_args)

        nobjects = self._lensing_mass_func.dN_comoving_deltaFunc(10**pargs['M'], zstart, delta_z, pargs['mass_fraction'])
        nobjects *= pargs['LOS_normalization']
        nobjects = np.random.poisson(nobjects)

        masses = np.array([10**pargs['M']]*nobjects)

        x, y, r2d, r3d = self._render_positions_atz(zstart, len(masses))

        if len(masses) > 0:
            redshifts += [zstart] * len(masses)

        for z in self._redshift_range[1:-1]:

            pargs = copy(self._parameterization_args)

            nobjects = self._lensing_mass_func.dN_comoving_deltaFunc(10**pargs['M'], z,
                                                                     delta_z, pargs['mass_fraction'])
            nobjects *= pargs['LOS_normalization']
            nobjects = np.random.poisson(nobjects)
            m = np.array([10**pargs['M']] * nobjects)

            xi, yi, r2di, r3di = self._render_positions_atz(z, len(m))

            masses = np.append(masses,m)
            x = np.append(x, xi)
            y = np.append(y, yi)
            r2d = np.append(r2d, r2di)
            r3d = np.append(r3d, r3di)
            if len(m) > 0:
                redshifts += [z]*len(m)

        return np.array(masses), np.array(x), np.array(y), np.array(r2d), np.array(r3d), np.array(redshifts)

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

class LOSPowerLaw(LOS):

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

        zlens = self._lensing_mass_func.geometry._zlens

        compute_two_halo = self._lensing_mass_func._two_halo_term

        if compute_two_halo:
            z_2halo_term = self._redshift_range[np.where(self._redshift_range<zlens)][-1]
        else:
            z_2halo_term = None

        init = True
        delta_z = self._redshift_range[1] - self._redshift_range[0]

        for idx, zcurrent in enumerate(self._redshift_range):

            if zcurrent == self._lensing_mass_func.geometry._zlens:
                continue

            plaw_idx = self._lensing_mass_func.plaw_index_z(zcurrent)
            norm = self._lensing_mass_func.norm_at_z(zcurrent, delta_z)

            if zcurrent == z_2halo_term:
                add_two_halo = True
                rmax = self._lensing_mass_func._cosmo.T_xy(zlens - delta_z, zlens)
                norm_2halo = self._lensing_mass_func.norm_at_z_biased(zcurrent, delta_z,
                                                                      self._parameterization_args['parent_m200'],
                                                                      rmax=rmax)

                ratio = norm * norm_2halo ** -1
                norm_2halo *= self._parameterization_args['LOS_normalization']

                mi, xi, yi, r2di, r3di, zi = self._draw(norm_2halo, plaw_idx, self._parameterization_args, zcurrent)
                N_boost = int(np.round(len(mi) * (1 - ratio)))

                mi_2halo, xi_2halo,_yi_2halo, r2di_2halo, r3di_2halo = mi[0:N_boost], xi[0:N_boost], \
                                                                       yi[0:N_boost], r2di[0:N_boost], \
                                                                                 r3di[0:N_boost]
                zi_2halo = np.array([zlens]*len(mi_2halo))

                mi, xi, yi, r2di, r3di, zi = mi[N_boost:], xi[N_boost:], yi[N_boost:], r2di[N_boost:], \
                                             r3di[N_boost:], zi[N_boost:]

            else:
                add_two_halo = False
                norm = self._lensing_mass_func.norm_at_z(zcurrent, delta_z)
                norm *= self._parameterization_args['LOS_normalization']

                mi, xi, yi, r2di, r3di, zi = self._draw(norm, plaw_idx, self._parameterization_args, zcurrent)

            if init:

                masses, x, y, r2d, r3d, z = mi, xi, yi, r2di, r3di, zi
                init = False

            else:

                masses = np.append(masses, mi)
                x, y = np.append(x, xi), np.append(y, yi)
                r2d, r3d = np.append(r2d, r2di), np.append(r3d, r3di)
                z = np.append(z, zi)

            if add_two_halo:
                masses = np.append(masses, mi_2halo)
                x, y = np.append(x, xi_2halo), np.append(y, _yi_2halo)
                r2d, r3d = np.append(r2d, r2di_2halo), np.append(r3d, r3di_2halo)
                z = np.append(z, zi_2halo)

        return masses, x, y, r2d, r3d, z


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


def _redshift_range_LOS(zmin, zmax, zstep):

    zvalues = np.arange(zmin, zmax, zstep)

    return zvalues


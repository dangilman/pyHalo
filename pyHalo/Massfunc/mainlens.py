import numpy as np
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw
from pyHalo.Spatial.nfw import NFW3D
from pyHalo.Halos.lens_cosmo import LensCosmo

class MainLensPowerLaw(object):

    def __init__(self, args, geometry):

        self._geometry = geometry

        spatial_args, parameterization_args = self._set_kwargs(args)

        self._mass_func_parameterization = BrokenPowerLaw(**parameterization_args)

        self._spatial_parameterization = NFW3D(**spatial_args)

    @property
    def _lenscosmo(self):

        if not hasattr(self, '_lens_cosmo'):
            self._lens_comso = LensCosmo(self._geometry._zlens, self._geometry._zsource, self._geometry._cosmo)

        return self._lens_comso

    def __call__(self):
        """

        :return: x coordinate, y coordinates, r3d, r3d
        NOTE: x and y are returned in arcsec, while r2d and r3d are expressed in kpc
        """
        masses = self._mass_func_parameterization.draw()

        # EVERYTHING EXPRESSED IN KPC
        x_kpc, y_kpc, r2d_kpc, r3d_kpc = self._spatial_parameterization.draw(len(masses))
        
        x_arcsec = x_kpc * self._geometry._kpc_per_arcsec_zlens ** -1
        y_arcsec = y_kpc * self._geometry._kpc_per_arcsec_zlens ** -1

        return np.array(masses), np.array(x_arcsec), np.array(y_arcsec), np.array(r2d_kpc), np.array(r3d_kpc), np.array(
            [self._geometry._zlens] * len(masses)), [True] * len(masses)

    def _spatial(self,args):

        args_spatial = {}

        # EVERYTHING EXPRESSED IN KPC
        args_spatial['rmax2d'] = 0.5*args['cone_opening_angle']*self._geometry._kpc_per_arcsec_zlens

        if 'parent_m200' in args.keys():
            # EVERYTHING EXPRESSED IN KPC
            if 'parent_c' not in args.keys():
                args['parent_c'] = self._lenscosmo.NFW_concentration(args['parent_m200'], self._geometry._zlens)

            if 'parent_Rs' not in args.keys():
                parent_Rs = self._lenscosmo.NFW_params_physical(args['parent_m200'],
                                                                args['parent_c'], self._geometry._zlens)[1]

            parent_r200 = parent_Rs * args['parent_c']

            args_spatial['Rs'] = parent_Rs
            args_spatial['rmax3d'] = parent_r200
        else:
            try:
                args_spatial['Rs'] = args['parent_Rs']
                args_spatial['rmax3d'] = args['parent_r200']
            except:
                raise ValueError('must specify either (parent_c, m200) for parent halo, or '
                                 '(parent_Rs, parent_r200) directly')

        if 'r_tidal' in args.keys():

            if isinstance(args['r_tidal'], str):
                if args['r_tidal'] == 'Rs':
                    args_spatial['r_core_parent'] = args_spatial['Rs']
                else:
                    if args['r_tidal'][-2:] != 'Rs':
                        raise ValueError('if specifying the tidal core radius as number*Rs, the last two '
                                         'letters in the string must be "Rs".')

                    scale = float(args['r_tidal'][:-2])
                    args_spatial['r_core_parent'] = scale * args_spatial['Rs']

            else:
                args_spatial['r_core_parent'] = args['r_tidal']

        return args_spatial

    def _mfunc(self,args):

        args_mfunc = {}

        required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_m_break']

        for key in required_keys:
            try:
                args_mfunc[key] = args[key]
            except:
                raise ValueError('must specify a value for ' + key)

        if 'sigma_sub' in args.keys() and 'mass_in_subhalos' not in args.keys():

            a0_area_parent_halo = args['sigma_sub'] * host_scaling_function(args['parent_m200'], self._geometry._zlens)

            args_mfunc['normalization'] = norm_A0_from_a0area(a0_area_parent_halo,
                                                                             self._geometry._kpc_per_arcsec_zlens, args['cone_opening_angle'],
                                                                             args_mfunc['power_law_index'], m_pivot=10**8)

        elif 'amp_at_8' in args.keys() and 'mass_in_subhalos' not in args.keys():

            args_mfunc['normalization'] = norm_A0_from_a0area(args['amp_at_8'],
                                                              self._geometry._kpc_per_arcsec_zlens,
                                                              args['cone_opening_angle'],
                                                              args_mfunc['power_law_index'], m_pivot=10 ** 8)

        elif 'f_sub' in args.keys() or 'log_f_sub' in args.keys():

            if 'log_f_sub' in args.keys():
                args['f_sub'] = 10**args['log_f_sub']

            a0_area_parent_halo = convert_fsub_to_norm(
                        args['f_sub'], args['parent_m200'], self._geometry._zlens, args['R_ein_main'], args['cone_opening_angle'], self._geometry._zlens,
                        args_mfunc['power_law_index'], 10**args_mfunc['log_mlow'],
                        10 ** args_mfunc['log_mhigh'], mpivot=10**8)

            args_mfunc['normalization'] = norm_A0_from_a0area(a0_area_parent_halo,
                                                              self._geometry._kpc_per_arcsec_zlens, args['cone_opening_angle'],
                                                              args_mfunc['power_law_index'], m_pivot=10**8)


        else:
            raise Exception('Must specify normalization of the subhalo '
                             'mass function in terms of sigma_sub or f_sub (fraction of total host mass).')


        return args_mfunc

    def _set_kwargs(self,args):

        args_mfunc = self._mfunc(args)
        args_spatial = self._spatial(args)

        for key in args.keys():
            if key not in args_mfunc.keys():
                args_mfunc[key] = args[key]

        return args_spatial, args_mfunc

def host_scaling_function(mhalo, z, k1 = 0.88, k2 = 1.7, k3 = -2):

    # interpolated from galacticus

    logscaling = k1 * np.log10(mhalo * 10**-13) + k2 * np.log10(z + 0.5)

    return 10**logscaling

def norm_A0_from_a0area(a0_per_kpc2, kpc_per_asec_zlens, cone_opening_angle, plaw_index, m_pivot = 10**8):

    R_kpc = kpc_per_asec_zlens * (0.5 * cone_opening_angle)

    area = np.pi * R_kpc ** 2

    return a0_per_kpc2 * m_pivot ** (-plaw_index-1) * area

def convert_fsub_to_norm(f_sub, m_host, zhost, rein_arcsec, cone_opening_angle, kpc_per_asec_zlens, plaw_index, mlow,
                         mhigh, mpivot=10**8):


    power = 2+plaw_index
    R_kpc = kpc_per_asec_zlens * (0.5 * cone_opening_angle)
    #R_kpc = kpc_per_asec_zlens * rein_arcsec

    area = np.pi * R_kpc ** 2

    integral = (mpivot/power) * ((mhigh/mpivot)**power - (mlow/mpivot)**power)

    m_sub_scaled = f_sub * m_host * host_scaling_function(m_host, zhost)

    sigma_sub = m_sub_scaled / integral / area

    return sigma_sub


import numpy as np
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw
from pyHalo.Spatial.nfw import NFW_3D
from pyHalo.Halos.Profiles.nfw import NFW

class MainLensPowerLaw(object):

    def __init__(self, args, lens_cosmo):

        self._lens_cosmo = lens_cosmo
        self._nfw_cosmo = NFW(lens_cosmo)
        spatial_args, parameterization_args = self._set_kwargs(args)

        self._mass_func_parameterization = BrokenPowerLaw(**parameterization_args)
        self._spatial_parameterization = NFW_3D(**spatial_args)

    def __call__(self):
        """

        :return: x coordinate, y coordinates, r3d, r3d
        NOTE: x and y are returned in arcsec, while r2d and r3d are expressed in kpc
        """
        masses = self._mass_func_parameterization.draw()

        # EVERYTHING EXPRESSED IN KPC
        x, y, r2d, r3d = self._spatial_parameterization.draw(len(masses))

        x_arcsec = x*self._lens_cosmo._kpc_per_asec_zlens ** -1
        y_arcsec = y*self._lens_cosmo._kpc_per_asec_zlens ** -1

        return np.array(masses), np.array(x_arcsec), np.array(y_arcsec), np.array(r2d), np.array(r3d), np.array(
            [self._lens_cosmo.z_lens] * len(masses))

    def _spatial(self,args):

        args_spatial = {}

        # EVERYTHING EXPRESSED IN KPC
        args_spatial['rmax2d'] = 0.5*args['cone_opening_angle']*\
                                 self._lens_cosmo.cosmo.kpc_per_asec(self._lens_cosmo.z_lens)

        if 'parent_m200' in args and 'parent_c' in args.keys():
            # EVERYTHING EXPRESSED IN KPC
            rho0_kpc, parent_Rs, parent_r200 = self._nfw_cosmo.NFW_params_physical(args['parent_m200'],
                                                                                    args['parent_c'],
                                                                                    self._lens_cosmo.z_lens)
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
                    args_spatial['r_core'] = args_spatial['Rs']
                else:
                    if args['r_tidal'][-2:] != 'Rs':
                        raise ValueError('if specifying the tidal core radius as number*Rs, the last two '
                                         'letters in the string must be "Rs".')

                    scale = float(args['r_tidal'][:-2])
                    args_spatial['r_core'] = scale * args_spatial['Rs']

            else:
                args_spatial['r_core'] = args['r_tidal_parent']

        return args_spatial

    def _mfunc(self,args):

        args_mfunc = {}

        required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_m_break']

        for key in required_keys:
            try:
                args_mfunc[key] = args[key]
            except:
                raise ValueError('must specify a value for ' + key)

        if 'norm_A0' in args.keys():

            args_mfunc['normalization'] = args['norm_A0']

        elif 'a0_area' in args.keys():
            print('enforcing cone_opening_angle = 6 * R_ein')
            r_ein_arcsec = args['cone_opening_angle'] * 0.5 * 3 ** -1
            a0_area_parent_halo = self._lens_cosmo.subhalo_mass_function_amplitude(args['a0_area'],
                                  r_ein_arcsec, self._lens_cosmo.z_lens)

            norm_0 = self._lens_cosmo.norm_A0_from_a0area(a0_area_parent_halo,
                           self._lens_cosmo.z_lens, args['cone_opening_angle'],
                           args_mfunc['power_law_index'], m_pivot = 10**8)

            args_mfunc['normalization'] = norm_0

        elif 'fsub' in args.keys():

            norm_0 = self._lens_cosmo.convert_fsub_to_norm(args['fsub'],args['cone_opening_angle'],
                                                                                       args_mfunc[
                                                                                           'power_law_index'],
                                                                                       10 ** args_mfunc[
                                                                                           'log_mlow'],
                                                                                       10 ** args_mfunc[
                                                                                           'log_mhigh'])
            ml, mh, index = 10 ** args_mfunc['log_mlow'], 10 ** args_mfunc['log_mhigh'], \
                            args_mfunc['power_law_index']
            denom = mh ** (2 + index) - ml ** (2 + index)
            denom_norm = (10 ** 10) ** (2 + index) - (10 ** 6) ** (2 + index)
            rescale = denom * denom_norm ** -1

            args_mfunc['normalization'] = norm_0 * rescale

        else:
            raise ValueError('must either specify the normalization "a0_area" direclty, or specify the mass fraction'
                             'in substructure at the Einstein radius "fsub".')


        return args_mfunc

    def _set_kwargs(self,args):

        args_mfunc = self._mfunc(args)
        args_spatial = self._spatial(args)

        for key in args.keys():
            if key not in args_mfunc.keys():
                args_mfunc[key] = args[key]

        return args_spatial, args_mfunc


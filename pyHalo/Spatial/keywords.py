
def LOS_spatial_global(args):

    args_spatial = {}
    args_spatial['cone_opening_angle'] = args['cone_opening_angle']
    return args_spatial

def subhalo_spatial_NFW(args, kpc_per_arcsec_zlens, zlens, lenscosmo):
    args_spatial = {}

    # EVERYTHING EXPRESSED IN KPC
    args_spatial['rmax2d'] = 0.5 * args['cone_opening_angle'] * kpc_per_arcsec_zlens

    if 'log_m_host' in args.keys():
        args['host_m200'] = 10 ** args['log_m_host']

    if 'host_m200' in args.keys():
        # EVERYTHING EXPRESSED IN KPC

        if 'host_c' not in args.keys():
            args['host_c'] = lenscosmo.NFW_concentration(args['host_m200'], zlens,
                  model='diemer19', mdef='200c', logmhm=args['log_mc'], scatter=True,
                 c_scale=args['c_scale'], c_power=args['c_power'], scatter_amplitude=args['c_scatter_dex'])

        if 'host_Rs' not in args.keys():
            host_Rs = lenscosmo.NFW_params_physical(args['host_m200'],
                                                            args['host_c'], zlens)[1]

        parent_r200 = host_Rs * args['host_c']

        args_spatial['Rs'] = host_Rs
        args_spatial['rmax3d'] = parent_r200
    else:
        raise Exception('Must specify the host halo mass when rendering subhalos')

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

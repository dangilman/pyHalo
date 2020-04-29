from pyHalo.defaults import *
from pyHalo.Rendering.Main.SHMF_normalizations import *

def LOS_powerlaw_mfunc(args, lens_mass_function):

    args_mfunc = {}
    required_keys = ['zmin', 'zmax', 'log_m_break', 'log_mlow_los',
                     'log_mhigh_los', 'parent_m200', 'LOS_normalization',
                     'draw_poission']

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
                args_mfunc['zmin'] = lenscone_default.default_zstart
            else:
                args_mfunc['zmax'] = lens_mass_function.geometry._zsource - lenscone_default.default_zstart

    if args_mfunc['log_m_break'] == 0:
        args_mfunc['break_index'] = 0
        args_mfunc['c_scale'] = 0
        args_mfunc['c_power'] = 0
        args_mfunc['break_scale'] = 1

    else:

        try:
            args_mfunc['break_index'] = args['break_index']
            args_mfunc['c_scale'] = args['c_scale']
            args_mfunc['c_power'] = args['c_power']
            args_mfunc['break_scale'] = args['break_scale']

        except:
            raise ValueError('must specify a value for "break_index, c_scale, c_power" if log_m_break != 0 '
                             '(because you are specifying a WDM scenario in which the concentration and mass function'
                             'slope  of halos is affected')

    return args_mfunc

def LOS_delta_mfunc(args, lensing_mass_func):

    args_mfunc = {}
    required_keys = ['zmin', 'zmax', 'logM_delta', 'mass_fraction', 'LOS_normalization', 'parent_m200']

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
                args_mfunc['zmin'] = lenscone_default.default_zstart
            else:
                args_mfunc['zmax'] = lensing_mass_func.geometry._zsource - lenscone_default.default_zstart

    return args_mfunc

def subhalo_mass_function(args, kpc_per_arcsec_zlens, zlens):

    args_mfunc = {}

    required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_m_break', 'break_index', 'break_scale']

    for key in required_keys:
        try:
            args_mfunc[key] = args[key]
        except:
            raise ValueError('must specify a value for ' + key)

    if 'sigma_sub' in args.keys():

        args_mfunc['normalization'] = norm_AO_from_sigmasub(args['sigma_sub'], args['parent_m200'],
                                                            zlens,
                                                            kpc_per_arcsec_zlens,
                                                            args['cone_opening_angle'],
                                                            args['power_law_index'])


    elif 'norm_kpc2' in args.keys():

        args_mfunc['normalization'] = norm_A0_from_a0area(args['norm_kpc2'],
                                                          zlens,
                                                          args['cone_opening_angle'],
                                                          args_mfunc['power_law_index'])

    elif 'norm_arcsec2' in args.keys():

        args_mfunc['normalization'] = norm_constant_per_squarearcsec(args['norm_arcsec2'],
                                                                     kpc_per_arcsec_zlens,
                                                                     args['cone_opening_angle'],
                                                                     args_mfunc['power_law_index'])

    elif 'f_sub' in args.keys() or 'log_f_sub' in args.keys():

        if 'log_f_sub' in args.keys():
            args['f_sub'] = 10 ** args['log_f_sub']

        a0_area_parent_halo = convert_fsub_to_norm(
            args['f_sub'], args['parent_m200'], zlens, args['R_ein_main'], args['cone_opening_angle'],
            zlens,
            args_mfunc['power_law_index'], 10 ** args_mfunc['log_mlow'],
                                           10 ** args_mfunc['log_mhigh'], mpivot=10 ** 8)

        args_mfunc['normalization'] = norm_A0_from_a0area(a0_area_parent_halo,
                                                          kpc_per_arcsec_zlens,
                                                          args['cone_opening_angle'],
                                                          args_mfunc['power_law_index'], m_pivot=10 ** 8)


    else:
        routines = 'sigma_sub: amplitude of differential mass function at 10^8 solar masses (d^2N / dmdA) in units [kpc^-2 M_sun^-1];\n' \
                   'automatically accounts for evolution of projected number density with halo mass and redshift (see Gilman et al. 2020)\n\n' \
                   'norm_kpc2: same as sigma_sub, but does not automatically account for evolution with halo mass and redshift\n\n' \
                   'norm_arcsec2: same as norm_kpc2, but in units (d^2N / dmdA) in units [arcsec^-2 M_sun^-1]\n\n' \
                   'f_sub or log_f_sub: projected mass fraction in substructure within the radius 0.5*cone_opening_angle'

        raise Exception('Must specify normalization of the subhalo '
                        'mass function. Recognized normalization routines are: \n' + routines)

    return args_mfunc

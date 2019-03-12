class CosmoDefaults(object):

    def __init__(self):

        default_mass_function = 'sheth99'

        if default_mass_function == 'despali16':
            self.default_mdef = '200c'
        if default_mass_function == 'reed07':
            self.default_mdef = 'fof'
        if default_mass_function == 'sheth99':
            self.default_mdef = 'fof'

        self.default_mass_function = default_mass_function

        # default from WMAP9
        self.H0 = 69.7
        self.omega_baryon = 0.0464
        self.omega_DM = 0.235
        self.sigma8 = 0.82

        self.default_args = {'H0': self.H0, 'omega_baryon': self.omega_baryon,
                             'omega_DM': self.omega_DM, 'sigma8': self.sigma8}

class LensConeDefaults(object):

    def __init__(self):
        self.default_zstart = 0.01
        self.distance_resolution_MPC = 1
        self.default_z_round = 2
        self.default_z_step = 0.02

class TruncationDefaults(object):

    def __init__(self):

        self.RocheNorm = 1.4
        self.RocheNu = 2
        self.LOS_truncation = 50 # truncate at 'r50'

class DMHaloDefaults(object):

    def __init__(self):

        self.mass_concentration_relation = 'diemer18'

        if self.mass_concentration_relation == 'diemer18':
            self.mc_mdef = '200c'
        else:
            self.mc_mdef = '200c'

        self.scatter = True

        # From Bose et al 2016
        self.c_scale = 60
        self.c_power = -0.17

class RealizationDefaults(object):

    def __init__(self):


        # opening angle = opening_anlge_factor * Rein
        self.opening_angle_factor = 5

        self.default_mhm = 0
        self.default_break_index = 1.3
        self.default_r_tidal = 0.5 # r_tidal = 'default_r_ridal * Rs'

        self.default_type = 'composite_powerlaw'


        self.default_include_subhalos = False
        self.default_LOS_normalization = 1

        self.log_mlow = 6
        self.log_mhigh = 10
        self.two_halo_term = True

        self.m_parent = 10**13


####################################################################################

cosmo_default = CosmoDefaults()
lenscone_default = LensConeDefaults()
truncation_default = TruncationDefaults()
halo_default = DMHaloDefaults()
realization_default = RealizationDefaults()

def set_default_kwargs(args):

    profile_params = {}

    if 'include_subhalos' in args.keys():
        profile_params.update({'include_subhalos': args['include_subhalos']})
        if args['include_subhalos'] is True:
            profile_params.update({'subhalo_args': args['subhalo_args']})
    else:
        profile_params.update({'include_subhalos':
                                   realization_default.default_include_subhalos})

    if 'log_m_break' in args.keys():
        if 'break_index' not in args.keys():
            raise Exception('If log_m_break is specified, must include break_index keyword.'
                            'default is '+str(realization_default.default_break_index))

        profile_params.update({'log_m_break': args['log_m_break'],
                               'break_index': args['break_index']})
    else:
        print('log_m_break not specified, assuming '+str(realization_default.default_mhm))
        profile_params.update({'log_m_break': realization_default.default_mhm,
                               'break_index': realization_default.default_break_index})

    if 'c_power' in args.keys():
        profile_params.update({'c_power': args['c_power']})
    else:
        print('c_power not specified, assuming -0.17 (only applies if log_m_break>0)')
        profile_params.update({'c_power': halo_default.c_power})
    if 'c_scale' in args.keys():
        profile_params.update({'c_scale': args['c_scale']})
    else:
        print('c_scale not specified, assuming 60')
        profile_params.update({'c_scale': halo_default.c_scale})

    if 'parent_m200' in args.keys():
        profile_params.update({'parent_m200': args['parent_m200']})
    else:
        print('Warning: halo mass not specified, assuming a parent halo mass of 10^13.')
        profile_params.update({'parent_m200': realization_default.m_parent})
    if 'LOS_normalization' in args.keys():
        profile_params.update({'LOS_normalization': args['LOS_normalization']})
    else:
        profile_params.update({'LOS_normalization':
                                   realization_default.default_LOS_normalization})

    if 'c_scatter' not in args.keys():
        profile_params.update({'c_scatter': halo_default.scatter})
    if 'include_subhalos' not in args.keys():
        profile_params.update({'include_subhalos':
                                   realization_default.default_include_subhalos})
        if realization_default.default_include_subhalos is True:
            raise Exception('not yet implemented.')

    if 'RocheNorm' not in args.keys():
        profile_params.update({'RocheNorm': truncation_default.RocheNorm})
    if 'RocheNu' not in args.keys():
        profile_params.update({'RocheNu': truncation_default.RocheNu})
    if 'LOS_truncation_factor' not in args.keys():
        profile_params.update({'LOS_truncation_factor': truncation_default.LOS_truncation})

    return profile_params






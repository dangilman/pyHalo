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
        self.Ob0 = 0.0464
        self.omega_DM = 0.235
        self.Om0 = self.Ob0 + self.omega_DM
        self.sigma8 = 0.82
        self.curvature = 'flat'
        self.ns = 0.9608
        self.power_law = False

        self._cosmo_param_dictionary = {'H0': self.H0, 'Ob0': self.Ob0, 'Om0': self.Om0,
                                        'Odm0': self.omega_DM, 'sigma8': self.sigma8, 'flat': self.curvature,
                                        'ns': self.ns, 'power_law': self.power_law}

    def __call__(self, key):

        try:
            return self._cosmo_param_dictionary[key]
        except:
            raise Exception(key + ' not a recognized cosmology key word argument.')


class LensConeDefaults(object):

    def __init__(self):
        self.default_zstart = 0.01
        self.distance_resolution_MPC = 1
        self.default_z_round = 2
        self.default_z_step = 0.02
        self.default_geometry = 'DOUBLE_CONE'
        # other possibilities:
        # CONE, CYLINDER

class TruncationDefaults(object):

    def __init__(self):

        self.RocheNorm = 1.4
        self.RocheNu = 2
        self.LOS_truncation = 50 # truncate at 'r50'

        trunc_routines = ['mean_ISOhost',
                          'mean_NFWhost',
                          'simple',
                          'constant']

        self.routine = trunc_routines[2]
        self.truncate_at_pericenter = False

class DMHaloDefaults(object):

    def __init__(self):

        self.mass_concentration_relation = 'diemer19'
        self.evaluate_mc_at_zlens = False

        if self.mass_concentration_relation == 'diemer19':
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
        self.opening_angle_factor = 6

        self.default_mhm = 0
        self.default_break_scale = 1
        self.default_break_index = -1.3
        self.default_r_tidal = 0.5 # r_tidal = 'default_r_ridal * Rs'

        self.default_type = 'composite_powerlaw'

        self.default_mass_function = 'sheth99'

        self.default_subhalos_of_field_halos = False
        self.default_LOS_normalization = 1

        self.log_mlow = 6
        self.log_mhigh = 10
        self.two_halo_term = True

        self.m_parent = 10**13

        self.subtract_exact_mass_sheets = False
        self.subhalo_mass_sheet_scale = 1.
        self.subtract_subhalo_mass_sheet = True
        self.draw_poisson = True

        self.subhalo_spatial_distribution = 'HOST_NFW'

        self.subhalo_convergence_correction_profile = 'NFW'

        self.kappa_scale = 1

####################################################################################

cosmo_default = CosmoDefaults()
lenscone_default = LensConeDefaults()
truncation_default = TruncationDefaults()
halo_default = DMHaloDefaults()
realization_default = RealizationDefaults()
print_defaults = False

def set_default_kwargs(profile_params, dynamic, zsource):

    if 'subhalos_of_field_halos' not in profile_params.keys():
        profile_params.update({'subhalos_of_field_halos':
                                   realization_default.default_subhalos_of_field_halos})
        if realization_default.default_subhalos_of_field_halos is True:
            raise Exception('not yet implemented.')

    if 'subtract_exact_mass_sheets' not in profile_params.keys():
        profile_params.update({'subtract_exact_mass_sheets': realization_default.subtract_exact_mass_sheets})

    if 'subtract_subhalo_mass_sheet' not in profile_params.keys():
        profile_params.update({'subtract_subhalo_mass_sheet': realization_default.subtract_subhalo_mass_sheet})

    if 'subhalo_convergence_correction_profile' not in profile_params.keys():
        profile_params.update({'subhalo_convergence_correction_profile': realization_default.subhalo_convergence_correction_profile})

    if 'nfw_kappa_centroid' not in profile_params.keys():
        profile_params.update({'nfw_kappa_centroid': [0., 0.]})

    if 'subhalo_mass_sheet_scale' not in profile_params.keys():
        profile_params.update({'subhalo_mass_sheet_scale': realization_default.subhalo_mass_sheet_scale})

    if 'kappa_scale' not in profile_params.keys():
        profile_params.update({'kappa_scale': realization_default.kappa_scale})

    if 'draw_poisson' not in profile_params.keys():
        profile_params.update({'draw_poisson': realization_default.draw_poisson})

    if 'log_m_break' in profile_params.keys():
        if 'break_index' not in profile_params.keys():
            raise Exception('If log_m_break is specified, must include break_index keyword.'
                            'default is '+str(realization_default.default_break_index))
        if 'break_scale' not in profile_params.keys():
            profile_params['break_scale'] = realization_default.default_break_scale

        profile_params.update({'log_m_break': profile_params['log_m_break'],
                               'break_index': profile_params['break_index'],
                               'break_scale': profile_params['break_scale']})
    else:
        profile_params.update({'log_m_break': realization_default.default_mhm,
                               'break_index': realization_default.default_break_index,
                              'break_scale': realization_default.default_break_scale})

    if 'c_power' in profile_params.keys():
        profile_params.update({'c_power': profile_params['c_power']})
    else:
        if print_defaults:
            print('c_power not specified, assuming -0.17 (only applies if log_m_break>0)')
        profile_params.update({'c_power': halo_default.c_power})

    if 'c_scale' in profile_params.keys():
        profile_params.update({'c_scale': profile_params['c_scale']})
    else:
        if print_defaults:
            print('c_scale not specified, assuming 60')
        profile_params.update({'c_scale': halo_default.c_scale})

    if 'parent_m200' in profile_params.keys():
        profile_params.update({'parent_m200': profile_params['parent_m200']})
    elif 'log_m_host' in profile_params.keys():
        profile_params.update({'parent_m200': 10**profile_params['log_m_host']})
    else:
        if print_defaults:
            print('Warning: halo mass not specified, assuming a parent halo mass of 10^13.')
        profile_params.update({'parent_m200': realization_default.m_parent})

    if 'subhalo_spatial_distribution' not in profile_params.keys():
        profile_params.update({'subhalo_spatial_distribution':
                               realization_default.subhalo_spatial_distribution})

    if 'r_tidal' not in profile_params.keys():
        profile_params.update({'r_tidal': realization_default.default_r_tidal})
    if 'LOS_normalization' in profile_params.keys():
        profile_params.update({'LOS_normalization': profile_params['LOS_normalization']})
    else:
        profile_params.update({'LOS_normalization':
                                   realization_default.default_LOS_normalization})

    if 'LOS_normalization_mass_sheet' in profile_params.keys():
        profile_params.update({'LOS_normalization_mass_sheet': profile_params['LOS_normalization_mass_sheet']})
    else:
        profile_params.update({'LOS_normalization_mass_sheet': profile_params['LOS_normalization']})

    if 'mc_model' not in profile_params.keys():
        profile_params.update({'mc_model': halo_default.mass_concentration_relation})

    if 'evaluate_mc_at_zlens' not in profile_params.keys():
        profile_params.update({'evaluate_mc_at_zlens': halo_default.evaluate_mc_at_zlens})

    if 'c_scatter' not in profile_params.keys():
        profile_params.update({'c_scatter': halo_default.scatter})

    if 'truncation_routine' not in profile_params.keys():
        profile_params.update({'truncation_routine': truncation_default.routine})
    if profile_params['truncation_routine'] == 'simple':
        if 'RocheNorm' not in profile_params.keys():
            profile_params.update({'RocheNorm': truncation_default.RocheNorm})
        if 'RocheNu' not in profile_params.keys():
            profile_params.update({'RocheNu': truncation_default.RocheNu})
    if 'truncate_at_pericenter' not in profile_params.keys():
        profile_params.update({'truncate_at_pericenter': truncation_default.truncate_at_pericenter})
    if 'LOS_truncation_factor' not in profile_params.keys():
        profile_params.update({'LOS_truncation_factor': truncation_default.LOS_truncation})

    if 'zmin' not in profile_params.keys():
        profile_params.update({'zmin': lenscone_default.default_zstart})
    if 'zmax' not in profile_params.keys():
        profile_params.update({'zmax': zsource - lenscone_default.default_zstart})

    if not dynamic:
        if 'cone_opening_angle' not in profile_params.keys():
            raise Exception('must specify cone_opening_angle in keyword arguments.')

    return profile_params






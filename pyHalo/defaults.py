"""
default parameters used to create realizations. This should be good for most applications
"""
from pyHalo.Halos.tidal_truncation import TruncationRN, TruncationRoche

class CosmoDefaults(object):

    def __init__(self):

        default_mass_function = 'sheth99'

        self.default_mass_function = default_mass_function

        # # default from WMAP9
        # self.H0 = 69.7
        # self.Ob0 = 0.0464
        # self.omega_DM = 0.235
        # self.Om0 = self.Ob0 + self.omega_DM
        # self.sigma8 = 0.82
        # self.curvature = 'flat'
        # self.ns = 0.9608
        # self.power_law = False

        # default from PLANCK2018
        self.H0 = 67.5
        self.Ob0 = 0.049
        self.omega_DM = 0.26
        self.Om0 = self.Ob0 + self.omega_DM
        self.sigma8 = 0.81
        self.curvature = 'flat'
        self.ns = 0.965
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
        self.RocheNu = 2. / 3
        self.LOS_truncation = 50  # truncate at 'r50'
        self.truncation_model_subhalos = TruncationRoche(self.RocheNorm, RocheNu=self.RocheNu)
        self.truncation_model_field_halos = TruncationRN(self.LOS_truncation)

class DMHaloDefaults(object):

    def __init__(self):

        self.mass_concentration_relation = 'diemer19'
        self.mass_concentration_mdef = '200c'
        self.evaluate_mc_at_zlens = False

        self.scatter = True
        self.c_scatter_dex = 0.2

        # From Bose et al 2016
        self.suppression_model = 'polynomial'
        self.kwargs_suppression = {'c_scale': 60., 'c_power': -0.17,
                                   'c_power_inner': 1.0, 'mc_suppression_redshift_evolution': True}

class RealizationDefaults(object):

    def __init__(self):

        # opening angle = opening_anlge_factor * Rein
        self.opening_angle_factor = 6

        self.default_r_tidal = '0.5Rs' # r_tidal = 'default_r_ridal * Rs'

        self.default_type = 'composite_powerlaw'

        self.default_mass_function = 'sheth99'

        self.default_subhalos_of_field_halos = False
        self.default_LOS_normalization = 1

        self.log_mlow = 6
        self.log_mhigh = 10

        self.host_m200 = 10 ** 13

        self.m_pivot = 10 ** 8

        self.delta_power_law_index = 0.
        self.delta_power_law_index_coupling = 1.

        self.subtract_exact_mass_sheets = False
        self.subhalo_mass_sheet_scale = 1.
        self.subtract_subhalo_mass_sheet = True
        self.draw_poisson = True

        self.subhalo_spatial_distribution = 'HOST_NFW'

        self.subhalo_convergence_correction_profile = 'NFW'

        self.kappa_scale = 1

        self.default_turnover_model = 'POLYNOMIAL'

####################################################################################

cosmo_default = CosmoDefaults()
lenscone_default = LensConeDefaults()
truncation_default = TruncationDefaults()
halo_default = DMHaloDefaults()
realization_default = RealizationDefaults()
print_defaults = False

def set_default_kwargs(profile_params, zsource):

    if 'm_pivot' not in profile_params.keys():
        profile_params.update({'m_pivot': realization_default.m_pivot})

    if 'delta_power_law_index' not in profile_params.keys():
        profile_params.update({'delta_power_law_index': realization_default.delta_power_law_index})
    if 'delta_power_law_index_coupling' not in profile_params.keys():
        profile_params.update({'delta_power_law_index_coupling': realization_default.delta_power_law_index_coupling})

    if 'subtract_exact_mass_sheets' not in profile_params.keys():
        profile_params.update({'subtract_exact_mass_sheets': realization_default.subtract_exact_mass_sheets})

    if 'subtract_subhalo_mass_sheet' not in profile_params.keys():
        profile_params.update({'subtract_subhalo_mass_sheet': realization_default.subtract_subhalo_mass_sheet})

    if 'subhalo_convergence_correction_profile' not in profile_params.keys():
        profile_params.update({'subhalo_convergence_correction_profile': realization_default.subhalo_convergence_correction_profile})

    if 'subhalo_mass_sheet_scale' not in profile_params.keys():
        profile_params.update({'subhalo_mass_sheet_scale': realization_default.subhalo_mass_sheet_scale})

    if 'kappa_scale' not in profile_params.keys():
        profile_params.update({'kappa_scale': realization_default.kappa_scale})

    if 'draw_poisson' not in profile_params.keys():
        profile_params.update({'draw_poisson': realization_default.draw_poisson})

    if 'log_mc' in profile_params.keys():

        if 'log_mc' is not None:

            if 'mass_function_turnover_model' not in profile_params.keys():
                profile_params.update({'mass_function_turnover_model': realization_default.default_turnover_model})

            if profile_params['mass_function_turnover_model'] in ['POLYNOMIAL', 'MIXED_DM']:
                for param_name in ['a_wdm', 'b_wdm', 'c_wdm']:
                    if param_name not in profile_params.keys():
                        raise Exception('If log_mc is specified, must include a_wdm, b_wdm, c_wdm keywords')

                if profile_params['mass_function_turnover_model'] in ['MIXED_DM']:
                    if 'mixed_DM_frac' not in profile_params.keys():
                        raise Exception('with MIXED_DM model, must specify mixed_DM_frac keyword')

        else:
            profile_params.update({'log_mc': None,
                                   'a_wdm': None,
                                   'b_wdm': None,
                                   'c_wdm': None})

    else:
        profile_params.update({'log_mc': None,
                               'a_wdm': None,
                              'b_wdm': None,
                               'c_wdm': None})

    if 'suppression_model' not in profile_params.keys():
        if print_defaults:
            print('suppression_model not specified, assuming ', halo_default.suppression_model)
        profile_params['suppression_model'] = halo_default.suppression_model
    if 'kwargs_suppression' not in profile_params.keys():
        if print_defaults:
            print('kwargs_suppression not specified, assuming ', halo_default.kwargs_suppression)
        profile_params.update({'kwargs_suppression': halo_default.kwargs_suppression})

    if 'host_m200' in profile_params.keys():
        profile_params.update({'host_m200': profile_params['host_m200']})
    elif 'log_m_host' in profile_params.keys():
        profile_params.update({'host_m200': 10**profile_params['log_m_host']})
    else:
        if print_defaults:
            print('Warning: halo mass not specified, assuming a parent halo mass of 10^13.')
        profile_params.update({'host_m200': realization_default.host_m200})

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
    if 'mc_mdef' not in profile_params.keys():
        profile_params.update({'mc_mdef': halo_default.mass_concentration_mdef})

    if 'evaluate_mc_at_zlens' not in profile_params.keys():
        profile_params.update({'evaluate_mc_at_zlens': halo_default.evaluate_mc_at_zlens})

    if 'c_scatter' not in profile_params.keys():
        profile_params.update({'c_scatter': halo_default.scatter})
    if 'c_scatter_dex' not in profile_params.keys():
        profile_params.update({'c_scatter_dex': halo_default.c_scatter_dex})

    if 'zmin' not in profile_params.keys():
        profile_params.update({'zmin': lenscone_default.default_zstart})
    if 'zmax' not in profile_params.keys():
        profile_params.update({'zmax': zsource - lenscone_default.default_zstart})

    if 'log_mass_sheet_min' not in profile_params.keys():
        profile_params.update({'log_mass_sheet_min': profile_params['log_mlow']})
    if 'log_mass_sheet_max' not in profile_params.keys():
        profile_params.update({'log_mass_sheet_max': profile_params['log_mhigh']})

    if 'mass_function_LOS_type' not in profile_params.keys():
        profile_params.update({'mass_function_LOS_type': 'POWER_LAW'})
    if 'mass_function_SUB_type' not in profile_params.keys():
        profile_params.update({'mass_function_SUB_type': 'POWER_LAW'})

    if 'truncation_model_subhalos' not in profile_params.keys():
        profile_params.update({'truncation_model_subhalos': truncation_default.truncation_model_subhalos})
    if 'truncation_model_field_halos' not in profile_params.keys():
        profile_params.update({'truncation_model_field_halos': truncation_default.truncation_model_field_halos})


    return profile_params

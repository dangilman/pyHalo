# """
# default parameters used to create realizations. This should be good for most applications
# """
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
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
        self.cosmo_param_dictionary = {'H0': self.H0, 'Ob0': self.Ob0, 'Om0': self.Om0,
                                        'Odm0': self.omega_DM, 'sigma8': self.sigma8, 'flat': self.curvature,
                                        'ns': self.ns, 'power_law': self.power_law}

    def __call__(self, key):

        try:
            return self.cosmo_param_dictionary[key]
        except:
            raise Exception(key + ' not a recognized cosmology key word argument.')


class LensConeDefaults(object):

    def __init__(self):
        self.default_zstart = 0.01
        self.distance_resolution_MPC = 1
        self.default_z_round = 2
        self.default_z_step = 0.02
        self.default_geometry = 'DOUBLE_CONE'
        self.kwargs_mass_sheet_default = {'kappa_scale': 1.0, 'log_mlow': 7.0,
                                          'log_mhigh': 10.0, 'subtract_exact_mass_sheets': False}
        # other possibilities:
        # CONE, CYLINDER

class TruncationDefaults(object):

    def __init__(self):

        self.RocheNorm = 1.4
        self.RocheNu = 2. / 3
        self.LOS_truncation = 50  # truncate at 'r50'
        self.truncation_class_subhalos = TruncationRoche(self.RocheNorm, RocheNu=self.RocheNu)
        self.truncation_class_field_halos = TruncationRN(self.LOS_truncation)

class DMHaloDefaults(object):

    def __init__(self):

        self.concentration_class_subhalos = ConcentrationDiemerJoyce
        self.concentration_class_field_halos = ConcentrationDiemerJoyce
        self.evaluate_mc_at_zlens = False

        self.scatter = True
        self.c_scatter_dex = 0.2

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

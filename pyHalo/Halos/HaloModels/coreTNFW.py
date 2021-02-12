from pyHalo.Halos.halo_base import Halo
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo, TNFWSubhalo
import numpy as np

class coreTNFWFieldHalo(Halo):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        self._tnfw = TNFWFieldHalo(mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag)
        self._lens_cosmo = lens_cosmo_instance
        self._concentration = self._tnfw._concentration
        super(coreTNFWFieldHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        return 'NUMERICAL_ALPHA'

    def central_density(self):

        function_rho = self._args['SIDM_rhocentral_function']
        profile_args_tnfw = self._tnfw.profile_args
        median_concentration = self._concentration.NFW_concentration(self.mass,
                                                              self.z,
                                                              self._args['mc_model'],
                                                              self._args['mc_mdef'],
                                                              self._args['log_mc'],
                                                              False,
                                                              self._args['c_scale'],
                                                              self._args['c_power'])

        delta_concentration_dex = np.log10(profile_args_tnfw[0]) - np.log10(median_concentration)
        cross_section_type = self._args['cross_section_type']
        kwargs_cross_section = self._args['kwargs_cross_section']
        args_function = (self.mass, self.z, delta_concentration_dex, cross_section_type, kwargs_cross_section)
        log10_rho = function_rho(*args_function)
        rho0 = 10 ** log10_rho
        return rho0

    @property
    def lenstronomy_params(self):

        if not hasattr(self, '_kwargs_lenstronomy'):

            [concentration, rt, rho_central] = self.profile_args
            rhos_kpc, rs_kpc, _ = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            rhos, rs = rhos_kpc * 1000 ** 3, rs_kpc / 1000
            Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle_fromNFWparams(rhos, rs, self.z)

            x, y = np.round(self.x, 4), np.round(self.y, 4)

            Rs_angle = np.round(Rs_angle, 10)
            theta_Rs = np.round(theta_Rs, 10)
            r_trunc_arcsec = rt / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            rc_over_rs = rhos_kpc / rho_central
            r_core = rc_over_rs * rs_kpc

            kwargs = {'alpha_Rs': theta_Rs, 'Rs': Rs_angle,
                      'center_x': x, 'center_y': y, 'r_trunc': r_trunc_arcsec, 'r_core': r_core}

            self._kwargs_lenstronomy = kwargs

        return self._kwargs_lenstronomy, self._args['numerical_deflection_angle_class']

    @property
    def profile_args(self):

        if not hasattr(self, '_profile_args'):

            profile_args_tnfw = self._tnfw.profile_args
            core_density = self.central_density()
            self._profile_args = (profile_args_tnfw[0], profile_args_tnfw[1], core_density)

        return self._profile_args


class coreTNFWSubhalo(Halo):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        self._tnfw = TNFWFieldHalo(mass, x, y, r3d, mdef, z,
                                   sub_flag, lens_cosmo_instance, args, unique_tag)
        self._lens_cosmo = lens_cosmo_instance
        self._concentration = self._tnfw._concentration
        super(coreTNFWSubhalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                                lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        return 'NUMERICAL_ALPHA'

    def central_density(self):

        if self._args['evaluate_mc_at_zlens']:
            z_eval = self.z
        else:
            z_eval = self.z_infall

        function_rho = self._args['SIDM_rhocentral_function']
        profile_args_tnfw = self._tnfw.profile_args
        median_concentration = self._concentration.NFW_concentration(self.mass,
                                                                     z_eval,
                                                                     self._args['mc_model'],
                                                                     self._args['mc_mdef'],
                                                                     self._args['log_mc'],
                                                                     False,
                                                                     self._args['c_scale'],
                                                                     self._args['c_power'])

        delta_concentration_dex = np.log10(profile_args_tnfw[0]) - np.log10(median_concentration)
        cross_section_type = self._args['cross_section_type']
        kwargs_cross_section = self._args['kwargs_cross_section']

        args_function = (self.mass, z_eval, delta_concentration_dex, cross_section_type, kwargs_cross_section)
        log10_rho = function_rho(*args_function)
        rho0 = 10 ** log10_rho
        return rho0

    @property
    def lenstronomy_params(self):

        if not hasattr(self, '_kwargs_lenstronomy'):

            [concentration, rt, rho_central] = self.profile_args
            rhos_kpc, rs_kpc, _ = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            rhos, rs = rhos_kpc * 1000 ** 3, rs_kpc / 1000
            Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle_fromNFWparams(rhos, rs, self.z)

            x, y = np.round(self.x, 4), np.round(self.y, 4)

            Rs_angle = np.round(Rs_angle, 10)
            theta_Rs = np.round(theta_Rs, 10)
            r_trunc_arcsec = rt / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            rc_over_rs = rhos_kpc / rho_central
            r_core = rc_over_rs * rs_kpc

            kwargs = {'alpha_Rs': theta_Rs, 'Rs': Rs_angle,
                      'center_x': x, 'center_y': y, 'r_trunc': r_trunc_arcsec, 'r_core': r_core}

            self._kwargs_lenstronomy = kwargs

        return self._kwargs_lenstronomy, self._args['numerical_deflection_angle_class']

    @property
    def profile_args(self):

        if not hasattr(self, '_profile_args'):
            profile_args_tnfw = self._tnfw.profile_args
            core_density = self.central_density()
            self._profile_args = (profile_args_tnfw[0], profile_args_tnfw[1], core_density)

        return self._profile_args

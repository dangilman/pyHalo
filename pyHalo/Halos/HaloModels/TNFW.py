from pyHalo.Halos.halo_base import Halo
from pyHalo.Halos.concentration import Concentration
import numpy as np

class TNFWFieldHalo(Halo):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        self._lens_cosmo = lens_cosmo_instance
        self._concentration = Concentration(lens_cosmo_instance)

        super(TNFWFieldHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                           lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        return 'TNFW'

    @property
    def lenstronomy_params(self):

        [concentration, rt] = self.profile_args
        Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)

        x, y = np.round(self.x, 4), np.round(self.y, 4)

        Rs_angle = np.round(Rs_angle, 10)
        theta_Rs = np.round(theta_Rs, 10)
        r_trunc_arcsec = rt / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)

        kwargs = {'alpha_Rs': theta_Rs, 'Rs': Rs_angle,
                  'center_x': x, 'center_y': y, 'r_trunc': r_trunc_arcsec}

        return kwargs, None

    @property
    def profile_args(self):

        if not hasattr(self, '_profile_args'):

            truncation_radius = self._lens_cosmo.LOS_truncation_rN(self.mass, self.z,
                                                             self._args['LOS_truncation_factor'])

            concentration = self._concentration.NFW_concentration(self.mass,
                                                                  self.z,
                                                                  self._args['mc_model'],
                                                                  self._args['mc_mdef'],
                                                                  self._args['log_mc'],
                                                                  self._args['c_scatter'],
                                                                  self._args['c_scale'],
                                                                  self._args['c_power'],
                                                                  self._args['c_scatter_dex'])

            self._profile_args = (concentration, truncation_radius)

        return self._profile_args

class TNFWSubhhalo(Halo):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        self._lens_cosmo = lens_cosmo_instance
        self._concentration = Concentration(lens_cosmo_instance)

        super(TNFWSubhhalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                       lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        return 'TNFW'

    @property
    def lenstronomy_params(self):

        (concentration, rt) = self.profile_args
        Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)

        x, y = np.round(self.x, 4), np.round(self.y, 4)

        Rs_angle = np.round(Rs_angle, 10)
        theta_Rs = np.round(theta_Rs, 10)
        r_trunc_arcsec = rt / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)

        kwargs = {'alpha_Rs': theta_Rs, 'Rs': Rs_angle,
                  'center_x': x, 'center_y': y, 'r_trunc': r_trunc_arcsec}

        return kwargs, None

    @property
    def profile_args(self):

        if not hasattr(self, '_profile_args'):
            if self._args['evaluate_mc_at_zlens']:
                z_eval = self.z
            else:
                z_eval = self.z_infall

            truncation_radius = self._lens_cosmo.truncation_roche(self.mass, self.r3d, self._args['RocheNorm'], self._args['RocheNu'])
            concentration = self._concentration.NFW_concentration(self.mass,
                                                                  z_eval,
                                                                  self._args['mc_model'],
                                                                  self._args['mc_mdef'],
                                                                  self._args['log_mc'],
                                                                  self._args['c_scatter'],
                                                                  self._args['c_scale'],
                                                                  self._args['c_power'],
                                                                  self._args['c_scatter_dex'])

            self._profile_args = (concentration, truncation_radius)

        return self._profile_args


from pyHalo.Halos.halo_base import Halo
from pyHalo.Halos.concentration import Concentration
import numpy as np

class NFWFieldHalo(Halo):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        self._lens_cosmo = lens_cosmo_instance
        self._concentration = Concentration(lens_cosmo_instance)

        super(NFWFieldHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    @classmethod
    def change_profile_definition(cls, halo, new_mdef):
        """

        :param halo: an instance of Halo with a certain mdef
        :param new_mdef: a new mass definition
        :return: a new instance of Halo with the same mass, redshift, and angular position,
        but a new mass definition.
        """

        mass = halo.mass
        x = halo.x
        y = halo.y
        r3d = halo.r3d
        z = halo.z
        sub_flag = halo.is_subhalo
        args = halo._args
        cosmo_m_prof = halo.lens_cosmo
        unique_tag = halo.unique_tag

        return NFWFieldHalo(mass, x, y, r3d, new_mdef, z,
                            sub_flag, cosmo_m_prof, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        return 'NFW'

    @property
    def lenstronomy_params(self):

        (concentration) = self.profile_args
        Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)

        x, y = np.round(self.x, 4), np.round(self.y, 4)

        Rs_angle = np.round(Rs_angle, 10)
        theta_Rs = np.round(theta_Rs, 10)

        kwargs = {'alpha_Rs': theta_Rs, 'Rs': Rs_angle,
                  'center_x': x, 'center_y': y}

        return kwargs, None

    @property
    def profile_args(self):

        if not hasattr(self, '_profile_args'):
            concentration = self._concentration.NFW_concentration(self.mass,
                                                                  self.z,
                                                                  self._args['mc_model'],
                                                                  self._args['mc_mdef'],
                                                                  self._args['log_mc'],
                                                                  self._args['c_scatter'],
                                                                  self._args['c_scale'],
                                                                  self._args['c_power'],
                                                                  self._args['c_scatter_dex'])

            self._profile_args = (concentration)

        return self._profile_args

class NFWSubhhalo(Halo):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        self._lens_cosmo = lens_cosmo_instance
        self._concentration = Concentration(lens_cosmo_instance)

        super(NFWSubhhalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                           lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        return 'NFW'

    @property
    def lenstronomy_params(self):

        (concentration) = self.profile_args
        Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)

        x, y = np.round(self.x, 4), np.round(self.y, 4)

        Rs_angle = np.round(Rs_angle, 10)
        theta_Rs = np.round(theta_Rs, 10)

        kwargs = {'alpha_Rs': theta_Rs, 'Rs': Rs_angle,
                  'center_x': x, 'center_y': y}

        return kwargs, None

    @property
    def profile_args(self):

        if not hasattr(self, '_profile_args'):
            if self._args['evaluate_mc_at_zlens']:
                z_eval = self.z
            else:
                z_eval = self.z_infall

            concentration = self._concentration.NFW_concentration(self.mass,
                                                                  z_eval,
                                                                  self._args['mc_model'],
                                                                  self._args['mc_mdef'],
                                                                  self._args['log_mc'],
                                                                  self._args['c_scatter'],
                                                                  self._args['c_scale'],
                                                                  self._args['c_power'],
                                                                  self._args['c_scatter_dex'])

            self._profile_args = (concentration)

        return self._profile_args


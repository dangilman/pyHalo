from pyHalo.Halos.halo_base import Halo
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo, TNFWSubhalo
from lenstronomy.LensModel.Profiles.tnfw import TNFW
import numpy as np


class coreTNFWBase(Halo):
    """
    The main class for a cored NFW field halo profile

    See the base class in Halos/halo_base.py for the required routines for any instance of a Halo class
    """

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class,
                 unique_tag, tnfw_class):

        """
        See documentation in base class (Halos/halo_base.py)

        """
        self._tnfw_lenstronomy = TNFW()
        self._tnfw = tnfw_class
        self._lens_cosmo = lens_cosmo_instance

        super(coreTNFWBase, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                                lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['NumericalAlpha']

    @property
    def params_physical(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_params_physical'):
            [concentration, rt, core_density] = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            self._params_physical = {'rhos': rhos, 'rs': rs, 'r200': r200, 'r_trunc': rt,
                                     'rc_over_rs': min(1., rhos/core_density)}

        return self._params_physical

    @property
    def central_density(self):
        """
        Computes the central density of the cored profile using the user-specified class "SIDM_rhocentral_function"
        """
        z_eval = self._tnfw.z_eval
        profile_args_tnfw = self._tnfw.profile_args
        median_concentration = self._lens_cosmo.NFW_concentration(self.mass,
                                                                     z_eval,
                                                                     self._args['mc_model'],
                                                                     self._args['mc_mdef'],
                                                                     self._args['log_mc'],
                                                                     False,
                                                                     0.,
                                                                self._args['kwargs_suppression'],
                                                                self._args['suppression_model'])

        c = profile_args_tnfw[0]
        delta_c_over_c = (c - median_concentration)/c
        cross_section_type = self._args['cross_section_type']
        kwargs_cross_section = self._args['kwargs_cross_section']
        args_function = (self.mass, self.z, delta_c_over_c, cross_section_type, kwargs_cross_section)
        function_rho = self._args['SIDM_rhocentral_function']
        rho_central = function_rho(*args_function)

        return rho_central

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_kwargs_lenstronomy'):

            [concentration, rt, rho_central] = self.profile_args
            rhos, rs, _ = self.lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)

            rhos_mpc, rs_mpc = rhos * 1000 ** 3, rs / 1000
            Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle_fromNFWparams(rhos_mpc, rs_mpc, self.z)

            numerical_deflection_class = self._args['numerical_deflection_angle_class']

            beta_norm = 0.0025 * Rs_angle
            x_match = 10.
            r_trunc_norm = x_match * Rs_angle

            alpha_norm, _ = numerical_deflection_class(x_match * Rs_angle, 0., Rs_angle, beta_norm, r_trunc_norm, norm=1.)
            alpha_tnfw, _ = self._tnfw_lenstronomy.derivatives(x_match * Rs_angle, 0., Rs=Rs_angle,
                                                               alpha_Rs=theta_Rs,
                                                               r_trunc=r_trunc_norm)

            norm = alpha_tnfw / alpha_norm
            Rs_angle = np.round(Rs_angle, 10)

            beta = min(1., rhos / rho_central)

            r_trunc_arcsec = rt / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)

            self._kwargs_lenstronomy = [{'Rs': self._rescale_norm * Rs_angle, 'r_core': beta * Rs_angle,
                                       'center_x': self.x, 'center_y': self.y, 'norm': norm,
                                        'r_trunc': r_trunc_arcsec}]

        return self._kwargs_lenstronomy, self._args['numerical_deflection_angle_class']

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            profile_args_tnfw = self._tnfw.profile_args
            core_density = self.central_density
            self._profile_args = (profile_args_tnfw[0], profile_args_tnfw[1], core_density)

        return self._profile_args

class coreTNFWFieldHalo(coreTNFWBase):
    """
    Describes a cored TNFW profile in the field
    """

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, unique_tag):

        tnfw_class = TNFWFieldHalo(mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, unique_tag)
        super(coreTNFWFieldHalo, self).__init__(mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag, tnfw_class)

    @classmethod
    def fromTNFW(cls, tnfw_halo, kwargs_new):
        """
        Creates the profile class from an instance of a TNFWSubhalo
        :param tnfw_halo: an instance of TNFWSubhalo
        :param kwargs_new: new keyword arguments required to constrct the coreTNFW profile
        :return: instance of coreTNFW
        """
        new_halo = coreTNFWFieldHalo(tnfw_halo.mass, tnfw_halo.x, tnfw_halo.y, tnfw_halo.r3d, 'coreTNFW',
                                 tnfw_halo.z, False, tnfw_halo.lens_cosmo, kwargs_new, tnfw_halo.unique_tag)
        profile_args = tnfw_halo.profile_args
        new_halo._profile_args = (profile_args[0], profile_args[1], new_halo.central_density)

        return new_halo

class coreTNFWSubhalo(coreTNFWBase):
    """
    Describes a cored TNFW subhalo
    """

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, unique_tag):

        tnfw_class = TNFWSubhalo(mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, unique_tag)
        super(coreTNFWSubhalo, self).__init__(mass, x, y, r3d, mdef, z,
                                                sub_flag, lens_cosmo_instance, args, unique_tag, tnfw_class)

    @classmethod
    def fromTNFW(cls, tnfw_halo, kwargs_new):
        """
        Creates the profile class from an instance of a TNFWFieldHalo
        :param tnfw_halo: an instance of TNFWFieldHalo
        :param kwargs_new: new keyword arguments required to constrct the coreTNFW profile
        :return: instance of coreTNFW
        """
        new_halo = coreTNFWSubhalo(tnfw_halo.mass, tnfw_halo.x, tnfw_halo.y, tnfw_halo.r3d, 'coreTNFW',
                                     tnfw_halo.z, True, tnfw_halo.lens_cosmo, kwargs_new, tnfw_halo.unique_tag)
        profile_args = tnfw_halo.profile_args
        new_halo._profile_args = (profile_args[0], profile_args[1], new_halo.central_density)

        return new_halo

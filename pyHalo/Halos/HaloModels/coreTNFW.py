from pyHalo.Halos.halo_base import Halo
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo, TNFWSubhalo
from lenstronomy.LensModel.Profiles.tnfw import TNFW

class coreTNFWBase(Halo):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag, tnfw_class):

        self._tnfw_lenstronomy = TNFW()
        self._tnfw = tnfw_class
        self._lens_cosmo = lens_cosmo_instance
        self._concentration = self._tnfw._concentration
        super(coreTNFWBase, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                                lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        return 'NumericalAlpha'

    @property
    def params_physical(self):

        if not hasattr(self, '_params_physical'):
            [concentration, rt, core_density] = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            self._params_physical = {'rhos': rhos, 'rs': rs, 'r200': r200, 'r_trunc': rt,
                                     'rc_over_rs': min(1., rhos/core_density)}

        return self._params_physical

    @property
    def central_density(self):

        z_eval = self._tnfw.z_eval
        profile_args_tnfw = self._tnfw.profile_args
        median_concentration = self._concentration.NFW_concentration(self.mass,
                                                                     z_eval,
                                                                     self._args['mc_model'],
                                                                     self._args['mc_mdef'],
                                                                     self._args['log_mc'],
                                                                     False,
                                                                     self._args['c_scale'],
                                                                     self._args['c_power'],
                                                                     0.)

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

        if not hasattr(self, '_kwargs_lenstronomy'):

            lenstronomy_kwargs_tnfw = self._tnfw.lenstronomy_params[0]
            [concentration, _, rho_central] = self.profile_args

            rhos, _, _ = self.lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            rc_over_rs = min(1., rhos / rho_central)
            r_core = rc_over_rs * lenstronomy_kwargs_tnfw['Rs']

            numerical_deflection_class = self._args['numerical_deflection_angle_class']

            rs = lenstronomy_kwargs_tnfw['Rs']
            beta = 0.0025 * lenstronomy_kwargs_tnfw['Rs']
            r_trunc = lenstronomy_kwargs_tnfw['r_trunc']
            alpha_norm = numerical_deflection_class(10 * rs, 0., rs, beta, r_trunc, norm=1.)
            alpha_tnfw, _ = self._tnfw_lenstronomy.derivatives(10 * rs, 0., Rs=rs,
                                                               alpha_Rs=lenstronomy_kwargs_tnfw['alpha_Rs'],
                                                               r_trunc=r_trunc)

            norm = alpha_tnfw / alpha_norm
            lenstronomy_kwargs_tnfw['r_core'] = r_core
            lenstronomy_kwargs_tnfw['norm'] = norm
            del lenstronomy_kwargs_tnfw['alpha_Rs']

            self._kwargs_lenstronomy = lenstronomy_kwargs_tnfw

        return self._kwargs_lenstronomy, self._args['numerical_deflection_angle_class']

    @property
    def profile_args(self):

        if not hasattr(self, '_profile_args'):
            profile_args_tnfw = self._tnfw.profile_args
            core_density = self.central_density
            self._profile_args = (profile_args_tnfw[0], profile_args_tnfw[1], core_density)

        return self._profile_args

class coreTNFWFieldHalo(coreTNFWBase):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        tnfw_class = TNFWFieldHalo(mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag)
        super(coreTNFWFieldHalo, self).__init__(mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag, tnfw_class)

    @classmethod
    def fromTNFW(cls, tnfw_halo, kwargs_new):

        new_halo = coreTNFWFieldHalo(tnfw_halo.mass, tnfw_halo.x, tnfw_halo.y, tnfw_halo.r3d, 'coreTNFW',
                                 tnfw_halo.z, False, tnfw_halo.lens_cosmo, kwargs_new, tnfw_halo.unique_tag)
        profile_args = tnfw_halo.profile_args
        new_halo._profile_args = (profile_args[0], profile_args[1], new_halo.central_density)

        return new_halo

class coreTNFWSubhalo(coreTNFWBase):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        tnfw_class = TNFWSubhalo(mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag)
        super(coreTNFWSubhalo, self).__init__(mass, x, y, r3d, mdef, z,
                                                sub_flag, lens_cosmo_instance, args, unique_tag, tnfw_class)

    @classmethod
    def fromTNFW(cls, tnfw_halo, kwargs_new):

        new_halo = coreTNFWSubhalo(tnfw_halo.mass, tnfw_halo.x, tnfw_halo.y, tnfw_halo.r3d, 'coreTNFW',
                                     tnfw_halo.z, True, tnfw_halo.lens_cosmo, kwargs_new, tnfw_halo.unique_tag)
        profile_args = tnfw_halo.profile_args
        new_halo._profile_args = (profile_args[0], profile_args[1], new_halo.central_density)

        return new_halo

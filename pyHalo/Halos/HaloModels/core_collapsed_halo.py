import numpy as np
from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import PseudoDoublePowerlaw
from pyHalo.Halos.halo_base import Halo
from scipy.optimize import minimize
from copy import deepcopy



class CoreCollapsedHalo(Halo):
    _pseudo_nfw = False
    """

    """

    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._profile_envelope = TNFW()
        self._profile_center = PseudoDoublePowerlaw()
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_class = concentration_class
        self._penalty_final_R200, self._penalty_final_R = None, None
        mdef = 'CORE_COLLAPSED'
        super(CoreCollapsedHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                                lens_cosmo_instance, args, unique_tag)

    def component_density_profile_3d(self, r, profile_args=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        kwargs_lenstronomy = deepcopy(self.lenstronomy_params_split[0])
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        sigma_crit_kpc = sigma_crit_mpc * 1e-6
        factor = sigma_crit_kpc / kpc_per_arcsec
        del kwargs_lenstronomy[0]['center_x']
        del kwargs_lenstronomy[0]['center_y']
        del kwargs_lenstronomy[1]['center_x']
        del kwargs_lenstronomy[1]['center_y']
        inner = factor * self._profile_center.density_lens(r / kpc_per_arcsec, **kwargs_lenstronomy[0])
        kwargs_lenstronomy[1]['rho0'] = self._profile_envelope.alpha2rho0(kwargs_lenstronomy[1]['alpha_Rs'],
                                                                          kwargs_lenstronomy[1]['Rs'])
        del kwargs_lenstronomy[1]['alpha_Rs']
        outer = factor * self._profile_envelope.density(r / kpc_per_arcsec, **kwargs_lenstronomy[1])
        return inner, outer

    def density_profile_3d(self, r, kwargs_lenstronomy=None):
        inner, outer = self.component_density_profile_3d(r, kwargs_lenstronomy)
        return inner + outer

    @property
    def c(self):
        """
        Computes the halo concentration (once)
        """

        if not hasattr(self, '_c'):
            self._c = self._concentration_class.nfw_concentration(self.mass, self.z_eval)
        return self._c

    @property
    def lenstronomy_params_split(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):
            (alpha_Rs_center, Rs_angle, Rs_angle_inner, gamma_inner,
             gamma_outer, alpha_Rs_envelope, Rs_angle, r_trunc_arcsec) = self.profile_args

            x, y = np.round(self.x, 4), np.round(self.y, 4)
            Rs_angle = np.round(Rs_angle, 10)
            alpha_Rs_center = np.round(alpha_Rs_center, 10)
            Rs_angle_inner = np.round(Rs_angle_inner, 10)
            kwargs_collpsed_center = {'alpha_Rs': alpha_Rs_center,
                                      'Rs': Rs_angle_inner,
                                      'center_x': x,
                                      'center_y': y,
                                      'gamma_inner': gamma_inner,
                                      'gamma_outer': gamma_outer}

            r_trunc_arcsec = np.round(r_trunc_arcsec, 10)
            alpha_Rs_envelope = np.round(alpha_Rs_envelope, 10)
            kwargs_envelope = {'alpha_Rs': alpha_Rs_envelope,
                               'Rs': Rs_angle,
                               'center_x': x,
                               'center_y': y,
                               'r_trunc': r_trunc_arcsec}
            self._lenstronomy_args = [kwargs_collpsed_center, kwargs_envelope]
        return self._lenstronomy_args, None

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):
            (alpha_Rs_center, Rs_angle, Rs_angle_inner, gamma_inner,
             gamma_outer, alpha_Rs_envelope, Rs_angle, r_trunc_arcsec) = self.profile_args

            x, y = np.round(self.x, 4), np.round(self.y, 4)
            Rs_angle = np.round(Rs_angle, 10)
            alpha_Rs_center = np.round(alpha_Rs_center, 10)
            Rs_angle_inner = np.round(Rs_angle_inner, 10)
            r_trunc_arcsec = np.round(r_trunc_arcsec, 10)
            alpha_Rs_envelope = np.round(alpha_Rs_envelope, 10)
            kwargs = {'center_x': x, 'center_y': y, 'Rs_inner': Rs_angle_inner, 'Rs_outer': Rs_angle,
                      'alpha_Rs_inner': alpha_Rs_center, 'alpha_Rs_outer': alpha_Rs_envelope,
                      'r_trunc': r_trunc_arcsec, 'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer}
            self._lenstronomy_args = [kwargs]
        return self._lenstronomy_args, None

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['CORE_COLLAPSED_HALO']

    @property
    def profile_matching(self):
        """

        :return:
        """
        return self._penalty_final_R200, self._penalty_final_R

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            # solve for lens model parameters that conserve mass inside rs and r200
            gamma_inner = self._args['gamma_inner']
            gamma_outer = self._args['gamma_outer']
            rt_kpc = self._args['rt_kpc']
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
            rhos, rs, r200 = self.nfw_params
            x0 = (0.001, 0.001)
            r200_arcsec = r200 / kpc_per_arcsec
            Rs_angle = rs / kpc_per_arcsec
            r_trunc_angle = rt_kpc / kpc_per_arcsec
            Rs_angle_inner = self._args['Rs_inner_kpc'] / kpc_per_arcsec
            args = (r200_arcsec, gamma_inner, gamma_outer, Rs_angle, Rs_angle_inner, r_trunc_angle, sigma_crit_arcsec,
                    self._args['m_target_r200'], self._args['m_target_R'], self._args['r_match_kpc'] / kpc_per_arcsec,
                    10, 10)
            opt = minimize(self.constraint_mass, x0=x0, args=args, method='Nelder-Mead')
            alpha_Rs_center, alpha_Rs_envelope = opt['x']
            args = (r200_arcsec, gamma_inner, gamma_outer, Rs_angle, Rs_angle_inner, r_trunc_angle, sigma_crit_arcsec,
                    self._args['m_target_r200'], self._args['m_target_R'], self._args['r_match_kpc'] / kpc_per_arcsec,
                    1.0, 0.0)
            penalty_final_m200 = self.constraint_mass(opt['x'], *args)
            self._penalty_final_R200 = penalty_final_m200
            args = (r200_arcsec, gamma_inner, gamma_outer, Rs_angle, Rs_angle_inner, r_trunc_angle, sigma_crit_arcsec,
                    self._args['m_target_r200'], self._args['m_target_R'], self._args['r_match_kpc'] / kpc_per_arcsec,
                    0.0, 1.0)
            penalty_final_R = self.constraint_mass(opt['x'], *args)
            self._penalty_final_R = penalty_final_R
            self._profile_args = (alpha_Rs_center, Rs_angle, Rs_angle_inner, gamma_inner,
                                  gamma_outer, alpha_Rs_envelope, Rs_angle, r_trunc_angle)

        return self._profile_args

    @property
    def halo_effective_age(self):
        if not hasattr(self, '_halo_effective_age'):
            if self.is_subhalo:
                self._halo_effective_age = self.lens_cosmo.sidm_halo_effective_age(self.z,
                                                                                   self.z_infall,
                                                                                   self._args['lambda_t'])
            else:
                self._halo_effective_age = self.halo_age
        return self._halo_effective_age

    @property
    def sidm_timescale(self):
        """
        Computes the timescale given by Equation 2.2 in https://arxiv.org/pdf/2305.16176.pdf
        :return:
        """
        return self._args['sidm_timescale']

    @property
    def t_over_tc(self):
        """
        Computes the dimensionless timescale for the halo evolution
        :return:
        """
        return self.halo_effective_age / self.sidm_timescale

    def constraint_mass(self, x, r200_arcsec, gamma_inner, gamma_outer, Rs_angle, Rs_angle_inner, r_trunc_angle,
                        sigma_crit_arcsec,
                        m_target_r200,
                        m_target_r,
                        r_match_arcsec,
                        weight_r200=1.0,
                        weight_R=1.0):

        (alpha_Rs_center, alpha_Rs_envelope) = x
        if alpha_Rs_center < 0:
            return np.inf
        if alpha_Rs_envelope < 0:
            return np.inf
        rho0 = self._profile_envelope.alpha2rho0(alpha_Rs_envelope, Rs_angle)
        envelope_mass = self._profile_envelope.mass_3d(r200_arcsec, Rs_angle, rho0, r_trunc_angle / Rs_angle)
        central_mass = self._profile_center.mass_3d_lens(r200_arcsec, Rs_angle_inner, alpha_Rs_center, gamma_inner,
                                                         gamma_outer)
        total_mass_r200 = sigma_crit_arcsec * (envelope_mass + central_mass)
        penalty_r200 = abs(total_mass_r200 / m_target_r200 - 1)

        envelope_mass = self._profile_envelope.mass_3d(r_match_arcsec, Rs_angle, rho0, r_trunc_angle / Rs_angle)
        central_mass = self._profile_center.mass_3d_lens(r_match_arcsec, Rs_angle_inner, alpha_Rs_center, gamma_inner,
                                                         gamma_outer)
        total_mass_R = sigma_crit_arcsec * (envelope_mass + central_mass)
        penalty_R = abs(total_mass_R / m_target_r - 1)
        return penalty_r200 * weight_r200 + penalty_R * weight_R

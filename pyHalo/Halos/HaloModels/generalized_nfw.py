from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import PseudoDoublePowerlaw
import numpy as np
from scipy.optimize import minimize

class GeneralNFWSubhalo(Halo):
    _pseudo_nfw = True
    """
    The base class for a halo with a logarithmic inner slope gamma_inner and a logarithmic outer profile
    slope gamma_outer. The normalization is defined in terms of a parameter x_match, defined as the multiple of
    rs where this profile enclodes the same mass as an NFW profile. The scale radius is assume to be the same as that of
    an NFW profile with the specified halo mass
    """
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._prof = PseudoDoublePowerlaw()
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_class = concentration_class
        mdef = 'PSEUDO_DPL'
        super(GeneralNFWSubhalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                              lens_cosmo_instance, args, unique_tag)

    def density_profile_3d_lenstronomy(self, r, profile_args=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        prof = self._prof
        kwargs_lenstronomy = self.lenstronomy_params[0][0]
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        sigma_crit_kpc = sigma_crit_mpc * 1e-6
        factor = sigma_crit_kpc / kpc_per_arcsec
        return factor*prof.density(r / kpc_per_arcsec,
                                   kwargs_lenstronomy['Rs'],
                                   kwargs_lenstronomy['alpha_Rs'],
                                   kwargs_lenstronomy['gamma_inner'],
                                   kwargs_lenstronomy['gamma_outer'])

    @property
    def c(self):
        """
        Computes the halo concentration (once)
        """

        if not hasattr(self, '_c'):
            self._c = self._concentration_class.nfw_concentration(self.mass, self.z_eval)
        return self._c

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):
            (concentration, gamma_inner, gamma_outer) = self.profile_args
            _, rs, r200 = self.nfw_params
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            rs_arcsec = rs / kpc_per_arcsec
            rho0 = self._density_norm()
            alpha_Rs = self._prof.rho02alpha(rho0, rs_arcsec, gamma_inner, gamma_outer)
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            rs_arcsec = np.round(rs_arcsec, 10)
            alpha_Rs = np.round(alpha_Rs, 10)
            self._lenstronomy_args = [{'alpha_Rs': alpha_Rs, 'Rs': rs_arcsec, 'gamma_inner': gamma_inner, 'center_x': x, 'center_y': y,
                                      'gamma_outer': gamma_outer}]
        return self._lenstronomy_args, None

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['PSEUDO_DPL']

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):

            concentration = self._concentration_class.nfw_concentration(self.mass, self.z_eval)
            gamma_inner = self._args['gamma_inner']
            gamma_outer = self._args['gamma_outer']
            self._profile_args = (concentration, gamma_inner, gamma_outer)

        return self._profile_args

    def _density_norm(self):
        """
        Calculates the density normalization that conserves mass within x_match
        :return:
        """
        if not hasattr(self, '_rho0_norm'):
            (concentration, gamma_inner, gamma_outer) = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z_eval)
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            if 'x_match' in self._args.keys():
                if self._args['x_match'] == 'c':
                    x_match = concentration
                else:
                    x_match = self._args['x_match']
            else:
                # r_vmax = 2.16 * rs
                x_match = 2.16
            rs_arcsec = rs / kpc_per_arcsec
            r_match_arcsec = x_match * rs / kpc_per_arcsec
            fx = np.log(1 + x_match) - x_match / (1 + x_match)
            m_nfw = 4 * np.pi * rs ** 3 * rhos * fx
            sigma_crit_arcsec = self._lens_cosmo.sigma_crit_arcsecond_interp(self.z)
            self._rho0_norm = m_nfw / self._prof.mass_3d(r_match_arcsec, rs_arcsec, sigma_crit_arcsec, gamma_inner, gamma_outer)
        return self._rho0_norm

class GeneralNFWFieldHalo(GeneralNFWSubhalo):
    """
    Class that defines a power law halo in the field
    """
    pass

class GeneralNFWHaloFromMass(Halo):
    """
    Initialize the profile for a set of parameters (M, R, r, gamma_inner, gamma_outer), where M is the mass enclosed
    inside R, r [kpc] is the transition radius, and gamma_inner/gamma_outer is the log-slope inside/outside r [kpc]
    """

    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._prof = PseudoDoublePowerlaw()
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_class = concentration_class
        mdef = 'PSEUDO_DPL'
        super(GeneralNFWHaloFromMass, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                              lens_cosmo_instance, args, unique_tag)

    @property
    def c(self):
        """
        Computes the halo concentration (once)
        """
        if not hasattr(self, '_c'):
            raise Exception('concentration not defined for this halo')
        return self._c

    def density_profile_3d(self, r, profile_args):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :param profile_args: override internally-computed profile args for the halo (M, R, r, gamma_inner, gamma_outer)
        :return: the density profile in units M_sun / kpc^3
        """
        return self.density_profile_3d_lenstronomy(r, profile_args)

    def density_profile_3d_lenstronomy(self, r, profile_args=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :param profile_args: override internally-computed profile args for the halo (M, R, r, gamma_inner, gamma_outer)
        :return: the density profile in units M_sun / kpc^3
        """
        prof = self._prof
        kwargs_lenstronomy = self.lenstronomy_params[0][0]
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        sigma_crit_kpc = sigma_crit_mpc * 1e-6
        factor = sigma_crit_kpc / kpc_per_arcsec
        return factor*prof.density_lens(r / kpc_per_arcsec,
                                   kwargs_lenstronomy['Rs'],
                                   kwargs_lenstronomy['alpha_Rs'],
                                   kwargs_lenstronomy['gamma_inner'],
                                   kwargs_lenstronomy['gamma_outer'])

    def mass_2d(self, rmax):
        """
        Computes the 2-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        if rmax == 'r200':
            rmax = self.nfw_params[-1]
        kwargs_lenstronomy = self.lenstronomy_params[0][0]
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        x = np.logspace(-3.5, 0.0, 5000) * rmax / kpc_per_arcsec
        y = 0
        fxx, _, _, fyy = self._prof.hessian(x, y,
                                            kwargs_lenstronomy['Rs'],
                                            kwargs_lenstronomy['alpha_Rs'],
                                            kwargs_lenstronomy['gamma_inner'],
                                            kwargs_lenstronomy['gamma_outer'])
        kappa = 0.5 * (fxx + fyy)
        sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        sigma_crit_kpc = sigma_crit_mpc * 1e-6
        sigma_crit_arcsec = sigma_crit_kpc * kpc_per_arcsec ** 2
        return np.trapz(kappa * 2 * np.pi * x, x) * sigma_crit_arcsec
        # if rmax == 'r200':
        #     rmax = self.nfw_params[-1]
        # rho0_norm = self._density_norm()
        # r_transition = self._args['r']
        # gamma_inner = self._args['gamma_inner']
        # gamma_outer = self._args['gamma_outer']
        # sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        # kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        # sigma_crit_kpc = sigma_crit_mpc * 1e-6 / kpc_per_arcsec
        # return self._prof.mass_2d(rmax, r_transition, sigma_crit_kpc * rho0_norm, gamma_inner, gamma_outer)

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['PSEUDO_DPL']

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):
            (rho0, r_transition, gamma_inner, gamma_outer) = self.profile_args
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            r_transition_arcsec = r_transition / kpc_per_arcsec
            alpha_Rs = self._prof.rho02alpha(rho0, r_transition_arcsec, gamma_inner, gamma_outer)
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            r_transition_arcsec = np.round(r_transition_arcsec, 10)
            alpha_Rs = np.round(alpha_Rs, 10)
            self._lenstronomy_args = [
                {'alpha_Rs': alpha_Rs,
                 'Rs': r_transition_arcsec,
                 'gamma_inner': gamma_inner,
                 'center_x': x,
                 'center_y': y,
                 'gamma_outer': gamma_outer}]
        return self._lenstronomy_args, None

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            rho0 = self._density_norm()
            gamma_inner = self._args['gamma_inner']
            gamma_outer = self._args['gamma_outer']
            self._profile_args = (rho0, self._args['r'], gamma_inner, gamma_outer)

        return self._profile_args

    def _density_norm(self):
        """
        Calculates the density normalization that conserves mass M within R
        :return:
        """
        if not hasattr(self, '_rho0_norm'):

            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            sigma_crit_arcsec = self._lens_cosmo.sigma_crit_arcsecond_interp(self.z)

            M_enclosed = self._args['M']
            R = self._args['R']
            r_transition = self._args['r']
            gamma_inner = self._args['gamma_inner']
            gamma_outer = self._args['gamma_outer']

            R_arcsec = R / kpc_per_arcsec
            r_transition_arcsec = r_transition / kpc_per_arcsec

            profile_mass = self._prof.mass_3d(R_arcsec,
                                              r_transition_arcsec,
                                              sigma_crit_arcsec,
                                              gamma_inner, gamma_outer)
            self._rho0_norm = M_enclosed / profile_mass

        return self._rho0_norm

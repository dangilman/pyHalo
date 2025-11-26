from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import PseudoDoublePowerlaw
import numpy as np

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

    def density_profile_3d(self, r, profile_args=None):
        """
        Compute the 3D density profile
        :param r: radius in 3D [kpc]
        :param profile_args:
        :return: density profile in solar masses / kpc^3
        """
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        sigma_crit_arcsec = self._lens_cosmo.sigma_crit_arcsecond_interp(self.z)
        sigma_crit_kpc = sigma_crit_arcsec / kpc_per_arcsec ** 2
        factor = sigma_crit_kpc / kpc_per_arcsec
        kwargs_lenstronomy = self.lenstronomy_params[0][0]
        return factor * self._prof.density_lens(r / kpc_per_arcsec,
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
            rhos, rs_nfw, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z_eval)
            if 'r_transition' not in self._args.keys():
                self._args['r_transition'] = rs_nfw
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            if 'x_match' in self._args.keys():
                if self._args['x_match'] == 'c':
                    x_match = concentration
                else:
                    x_match = self._args['x_match']
            else:
                # r_vmax = 2.16 * rs
                x_match = 2.16
            rs_arcsec = self._args['r_transition'] / kpc_per_arcsec
            r_match_arcsec = x_match * self._args['r_transition'] / kpc_per_arcsec
            if 'mass_conservation' in self._args.keys():
                m_match = self._args['mass_conservation']
            else:
                fx = np.log(1 + x_match) - x_match / (1 + x_match)
                m_match = 4 * np.pi * self._args['r_transition'] ** 3 * rhos * fx
            sigma_crit_arcsec = self._lens_cosmo.sigma_crit_arcsecond_interp(self.z)
            self._rho0_norm = m_match / self._prof.mass_3d(r_match_arcsec, rs_arcsec, sigma_crit_arcsec, gamma_inner, gamma_outer)
        return self._rho0_norm

    @staticmethod
    def log_profile_slope(r, rs, gamma_inner, gamma_outer):
        """
        Compute the logarithmic profile slope for an array of radial (in 3D) coordinates r
        :param r: distance from center of halo [kpc]
        :param profile_args: keyword arguments for the density profile; if not specified, uses the ones computed inside
        each halo class
        :return: the density profile in units M_sun / kpc^3
        """
        x = r/rs
        d_log_rho_d_log_r = -(gamma_inner + gamma_outer * x ** 2) / (1 + x ** 2)
        return d_log_rho_d_log_r

class GeneralNFWFieldHalo(GeneralNFWSubhalo):
    """
    Class that defines a power law halo in the field
    """
    pass

from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.general_nfw import GNFW
import numpy as np

class GeneralNFWSubhalo(Halo):

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
        self._prof = GNFW()
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_class = concentration_class
        mdef = 'GNFW'
        super(GeneralNFWSubhalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                              lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):

            (concentration, gamma_inner, gamma_outer) = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
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

            sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2

            rho0 = m_nfw / self._prof.mass_3d(r_match_arcsec, rs_arcsec, sigma_crit_arcsec, gamma_inner, gamma_outer)
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
        return ['GNFW']

    @property
    def z_eval(self):
        """
        Returns the redshift at which to evalate the concentration-mass relation
        """
        if not hasattr(self, '_zeval'):

            if 'evaluate_mc_at_zlens' in self._args.keys() and self._args['evaluate_mc_at_zlens']:
                self._zeval = self.z
            else:
                self._zeval = self.z_infall

        return self._zeval

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


class GeneralNFWFieldHalo(GeneralNFWSubhalo):
    """
    Class that defines a power law halo in the field
    """
    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):

            concentration = self._concentration_class.nfw_concentration(self.mass, self.z)
            gamma_inner = self._args['gamma_inner']
            gamma_outer = self._args['gamma_outer']
            self._profile_args = (concentration, gamma_inner, gamma_outer)

        return self._profile_args

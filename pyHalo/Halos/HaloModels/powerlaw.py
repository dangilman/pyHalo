from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.splcore import SPLCORE
import numpy as np

class PowerLawSubhalo(Halo):

    """
    The base class for a halo modeled as a power law profile
    """
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._prof = SPLCORE()
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_class = concentration_class
        mdef = 'SPL_CORE'
        super(PowerLawSubhalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                              lens_cosmo_instance, args, unique_tag)

    @property
    def params_physical(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        if not hasattr(self, '_params_physical'):
            (concentration, gamma, x_core_halo) = self.profile_args
            rhos, rs, _ = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)

            if 'x_match' in self._args.keys():
                if self._args['x_match'] == 'c':
                    x_match = concentration
                else:
                    x_match = self._args['x_match']
            else:
                # r_vmax = 2.16 * rs
                x_match = 2.16

            r_match_arcsec = x_match * rs / kpc_per_arcsec
            fx = np.log(1 + x_match) - x_match / (1 + x_match)
            m = 4 * np.pi * rs ** 3 * rhos * fx
            r_core_arcsec = x_core_halo * r_match_arcsec / x_match

            sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2

            rho0 = m / self._prof.mass_3d(r_match_arcsec, sigma_crit_arcsec, r_core_arcsec, gamma)
            # units 1 / arcsec

            rho0 *= sigma_crit_arcsec * kpc_per_arcsec ** -3
            self._params_physical = {'rho0': rho0, 'r_core': x_core_halo * rs}

        return self._params_physical

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):

            (concentration, gamma, x_core_halo) = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)

            if 'x_match' in self._args.keys():
                if self._args['x_match'] == 'c':
                    x_match = concentration
                else:
                    x_match = self._args['x_match']
            else:
                # r_vmax = 2.16 * rs
                x_match = 2.16 # r_vmax

            r_match_arcsec = x_match * rs / kpc_per_arcsec
            fx = np.log(1+x_match) - x_match/(1 + x_match)
            m = 4 * np.pi * rs ** 3 * rhos * fx
            r_core_arcsec = x_core_halo * r_match_arcsec / x_match

            sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2

            rho0 = m/self._prof.mass_3d(r_match_arcsec, sigma_crit_arcsec, r_core_arcsec, gamma)
            sigma0 = rho0 * r_core_arcsec

            self._lenstronomy_args = [{'sigma0': sigma0, 'gamma': gamma, 'center_x': self.x, 'center_y': self.y,
                                      'r_core': r_core_arcsec}]

        return self._lenstronomy_args, None

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['SPL_CORE']

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
            gamma = self._args['log_slope_halo']
            x_core_halo = self._args['x_core_halo']
            self._profile_args = (concentration, gamma, x_core_halo)

        return self._profile_args


class PowerLawFieldHalo(PowerLawSubhalo):
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
            gamma = self._args['log_slope_halo']
            x_core_halo = self._args['x_core_halo']
            self._profile_args = (concentration, gamma, x_core_halo)

        return self._profile_args

    @property
    def z_eval(self):
        """
        Returns the redshift at which to evaluate the concentration-mass relation
        """
        return self.z

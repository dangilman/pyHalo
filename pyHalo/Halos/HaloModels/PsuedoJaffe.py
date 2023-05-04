from pyHalo.Halos.halo_base import Halo
import numpy as np
from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe

class PJaffeSubhalo(Halo):
    """
    Class that defines a halo modeled as a Psuedo-Jaffe profile.

    The profile is normalized such that the Psuedo-Jaffe profile has the same mass within r200 as an NFW profile with mass
    M200. The scale radius of the Psuedo-Jafee profile is the same as the NFW profile

    """
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)

        This profile is defined to have the same total mass as an NFW profile; rs = rs_pjaffe
        """
        self._lens_cosmo = lens_cosmo_instance
        self._prof = PJaffe()
        self._concentration_class = concentration_class
        self._truncation_class = truncation_class
        mdef = 'PJAFFE'
        super(PJaffeSubhalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    @property
    def params_physical(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_params_physical'):

            concentration = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            _, rs_kpc, r200_kpc = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            ra_kpc = 0.01 * rs_kpc

            rho = self._rho(self.mass, rs_kpc, ra_kpc, r200)

            self._params_physical = {'rhos': rhos, 'rs': rs, 'r200': r200, 'rho': rho}

        return self._params_physical

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):

            kpc_to_arcsec = 1 / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            (concentration) = self.profile_args
            rhos_kpc, rs_kpc, r200_kpc = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            ra_kpc = 0.01 * rs_kpc
            ra_arcsec = ra_kpc * kpc_to_arcsec
            rs_arcsec = rs_kpc * kpc_to_arcsec

            rmatch = r200_kpc
            rmatch_arcsec = rmatch * kpc_to_arcsec

            xmatch = rmatch/rs_kpc
            fx = np.log(1+xmatch) - xmatch / (1 + xmatch)
            m_nfw = 4 * np.pi * rs_kpc ** 3 * rhos_kpc * fx

            rho = self._rho(m_nfw, rs_arcsec, ra_arcsec, rmatch_arcsec)
            sigma0 = self._prof.rho2sigma(rho, ra_arcsec, rs_arcsec)
            sigma_crit_kpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source) * 0.001 ** 2
            sigma0 *= (sigma_crit_kpc/kpc_to_arcsec**2) ** -1

            self._lenstronomy_args = [{'center_x': self.x,
                                      'center_y': self.y,
                                      'Ra': ra_arcsec,
                                      'Rs': rs_arcsec,
                                      'sigma0': sigma0}]

        return self._lenstronomy_args, None

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

    def _rho(self, m, rs, ra, r_match):

        """
        returns the central density of a PJaffe halo such that the resulting halo has the mass m within r_match
        :param m:
        :param ra:
        :param r_match:
        :return:
        """
        f = (ra * np.arctan(r_match / ra) - rs * np.arctan(r_match / rs)) / (ra ** 2 - rs ** 2)
        rho = m / (4 * np.pi * ra ** 2 * rs ** 2 * f)
        return rho

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['PJAFFE']

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):

            concentration = self._concentration_class.nfw_concentration(self.mass, self.z_eval)
            self._profile_args = (concentration)

        return self._profile_args

class PJaffeFieldhalo(PJaffeSubhalo):
    """
    Defines a field halo modeled as a Psuedo-Jaffe profile
    """
    @property
    def profile_args(self):

        if not hasattr(self, '_profile_args'):

            concentration = self._concentration_class.nfw_concentration(self.mass, self.z)
            self._profile_args = (concentration)

        return self._profile_args





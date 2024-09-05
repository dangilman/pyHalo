from pyHalo.Halos.halo_base import Halo
import numpy as np
from pyHalo.Halos.tnfw_halo_util import tnfw_mass_fraction
from scipy.integrate import quad

class TNFWFieldHalo(Halo):

    """
    The base class for a truncated NFW halo
    """
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args,
                 truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        self._concentration_class = concentration_class
        self._truncation_class = truncation_class
        mdef = 'TNFW'
        super(TNFWFieldHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                           lens_cosmo_instance, args, unique_tag)

    def density_profile_3d(self, r, profile_args=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        if profile_args is None:
            c, rt = self.profile_args
        else:
            c, rt = profile_args
        rhos, rs, _ = self.lens_cosmo.NFW_params_physical(self.mass, self.c, self.z_eval)
        tau = rt/rs
        x = r / rs
        rho_nfw = rhos / x / (1 + x) ** 2
        return rho_nfw * tau ** 2 / (tau ** 2 + x ** 2)

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        return ['TNFW']

    @property
    def c(self):
        """
        Computes the halo concentration (once)
        """

        if not hasattr(self, '_c'):
            self._c = self._concentration_class.nfw_concentration(self.mass, self.z_eval)
        return self._c

    @property
    def params_physical(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        if not hasattr(self, '_params_physical'):
            [_, rt] = self.profile_args
            rhos, rs, r200 = self.nfw_params
            self._params_physical = {'rhos': rhos * self._rescale_norm, 'rs': rs, 'r200': r200, 'r_trunc_kpc': rt}

        return self._params_physical

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_kwargs_lenstronomy'):

            [concentration, rt] = self.profile_args
            Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)

            x, y = np.round(self.x, 4), np.round(self.y, 4)

            Rs_angle = np.round(Rs_angle, 10)
            theta_Rs = np.round(theta_Rs, 10)
            r_trunc_arcsec = rt / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)

            kwargs = [{'alpha_Rs': self._rescale_norm * theta_Rs, 'Rs': Rs_angle,
                      'center_x': x, 'center_y': y, 'r_trunc': r_trunc_arcsec}]

            self._kwargs_lenstronomy = kwargs

        return self._kwargs_lenstronomy, None

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        if not hasattr(self, '_profile_args'):
            truncation_radius_kpc = self._truncation_class.truncation_radius_halo(self)

            self._profile_args = (self.c, truncation_radius_kpc)
        return self._profile_args

class TNFWSubhalo(TNFWFieldHalo):
    """
    Defines a truncated NFW halo that is a subhalo of the host dark matter halo
    """

    @property
    def bound_mass(self):
        """
        Computes the mass inside the virial radius (with truncation effects included)
        :return: the mass inside r = c * r_s
        """
        if hasattr(self, '_kwargs_lenstronomy'):
            tau = self._kwargs_lenstronomy[0]['r_trunc'] / self._kwargs_lenstronomy[0]['Rs']
        else:
            params_physical = self.params_physical
            tau = params_physical['r_trunc_kpc'] / params_physical['rs']
        f = tnfw_mass_fraction(tau, self.c)
        return f * self.mass

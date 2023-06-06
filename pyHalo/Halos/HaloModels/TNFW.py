from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.tnfw import TNFW as TNFWLenstronomy
import numpy as np

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

            [concentration, rt] = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z_eval)
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
    def z_eval(self):
        """
        Returns the halo redshift
        """
        return self.z

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):

            truncation_radius_kpc = self._truncation_class.truncation_radius_halo(self)
            self._profile_args = (self.c, truncation_radius_kpc)

        return self._profile_args

    @property
    def bound_mass(self):
        """
        Computes the mass inside the virial radius (with truncation effects included)
        :return: the mass inside r = c * r_s
        """
        prof = TNFWLenstronomy()
        kwargs_profile = self.lenstronomy_params[0][0]
        alpha_rs = kwargs_profile['alpha_Rs']
        rs = kwargs_profile['Rs']
        r_trunc = kwargs_profile['r_trunc']
        r = self.c * rs
        rho0 = prof.alpha2rho0(alpha_rs, rs)
        mass_3d = prof.mass_3d(r, rs, rho0, r_trunc)
        mass_3d_infall = prof.mass_3d(r, rs, rho0, 1000*rs)
        return (mass_3d / mass_3d_infall) * self.mass


class TNFWSubhalo(TNFWFieldHalo):
    """
    Defines a truncated NFW halo that is a subhalo of the host dark matter halo
    """

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
    def params_physical(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_params_physical'):
            [concentration, rt] = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z_eval)
            self._params_physical = {'rhos': rhos, 'rs': rs, 'r200': r200, 'r_trunc_kpc': rt}

        return self._params_physical

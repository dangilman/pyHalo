from pyHalo.Halos.halo_base import Halo
import numpy as np

class NFWFieldHalo(Halo):

    """
    The main class for an NFW field halo profile without truncation

    See the base class in Halos/halo_base.py for the required routines for any instance of a Halo class
    """

    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class, unique_tag):

        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_class = concentration_class
        mdef = 'NFW'
        super(NFWFieldHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    def density_profile_3d(self, r, profile_args=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        if profile_args is None:
            c = self.profile_args
        else:
            c = profile_args
        rhos, rs, _ = self.lens_cosmo.NFW_params_physical(self.mass, c, self.z_eval)
        x = r/rs
        return rhos / x / (1 + x) ** 2

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        return ['NFW']

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        (concentration) = self.profile_args
        Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)
        x, y = np.round(self.x, 4), np.round(self.y, 4)
        Rs_angle = np.round(Rs_angle, 10)
        theta_Rs = np.round(theta_Rs, 10)
        kwargs = [{'alpha_Rs': self._rescale_norm * theta_Rs, 'Rs': Rs_angle,
                  'center_x': x, 'center_y': y}]
        return kwargs, None

    @property
    def c(self):
        """
        Computes the halo concentration (once)
        """

        if not hasattr(self, '_c'):
            self._c = self._concentration_class.nfw_concentration(self.mass, self.z_eval)
        return self._c

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            concentration = self.c
            self._profile_args = (concentration)

        return self._profile_args

class NFWSubhhalo(NFWFieldHalo):

    """
    The main class for an NFW subhalo profile without truncation

    See the base class in Halos/halo_base.py for the required routines for any instance of a Halo class
    """

from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.nfw import NFW as NFWLenstronomy
import numpy as np

class NFWFieldHalo(Halo):
    # we use the pseudo nfw methods to normalize profile
    _pseudo_nfw = False
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
        self._nfw_lenstronomy = NFWLenstronomy()
        mdef = 'NFW'
        super(NFWFieldHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    def density_profile_3d(self, r, profile_args=None, scaling=1.0):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        rhos, rs, _ = self.lens_cosmo.NFW_params_physical(self.mass, self.c, self.z_eval)
        x = r/rs
        return scaling * rhos / x / (1 + x) ** 2

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

        rhos_kpc, rs_kpc, _ = self.nfw_params
        rhos_mpc = rhos_kpc * 1e9
        rs_mpc = rs_kpc * 1e-3
        Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle_fromNFWparams(rhos_mpc,
                                                                               rs_mpc,
                                                                               self.z)
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

    @property
    def vmax_nfw(self):
        """
        Returns the maximum circular velocity in km/sec
        :return:
        """
        if not hasattr(self, '_vmax'):
            rhos, rs, _ = self.nfw_params
            _ = self.profile_args
            self._vmax = self._lens_cosmo.nfw_vmax(self._rescale_norm * rhos, rs)
        return self._vmax

    def mass_3d(self, rmax, profile_args=None):
        """
        Calculate the enclosed mass in 3D
        :param rmax:
        :param profile_args:
        :return:
        """
        if rmax == 'r200':
            rmax = self.nfw_params[1] * self.c
        rs = self.nfw_params[1]
        rho0 = self.nfw_params[0] * self._rescale_norm
        return self._nfw_lenstronomy.mass_3d(rmax, rs, rho0)

class NFWSubhhalo(NFWFieldHalo):

    """
    The main class for an NFW subhalo profile without truncation

    See the base class in Halos/halo_base.py for the required routines for any instance of a Halo class
    """

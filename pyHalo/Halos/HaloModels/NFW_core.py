from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.cnfw import CNFW as CNFWLenstronomy
import numpy as np

class CoreNFWHalo(Halo):
    # we use the pseudo nfw methods to normalize profile
    _pseudo_nfw = False
    """
    The main class for an NFW field halo profile without truncation

    See the base class in Halos/halo_base.py for the required routines for any instance of a Halo class
    """
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class, unique_tag,
                 conserve_m200=True):

        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_class = concentration_class
        self._cnfw_lenstronomy = CNFWLenstronomy()
        mdef = 'CNFW'
        super(CoreNFWHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    def density_profile_3d(self, r, profile_args=None, scaling=1.0):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        rho0, Rs, _ = self.nfw_params
        M0 = 4 * np.pi * rho0 * Rs ** 3
        r_core = self._args['beta'] * Rs
        return (M0 / 4 / np.pi) * ((r_core + r) * (r + Rs) ** 2) ** -1

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['CNFW']

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
        beta = self._args['beta']
        x, y = np.round(self.x, 4), np.round(self.y, 4)
        Rs_angle = np.round(Rs_angle, 10)
        theta_Rs = np.round(theta_Rs, 10)
        kwargs = [{'alpha_Rs': self._rescale_norm * theta_Rs, 'Rs': Rs_angle, 'r_core': beta * Rs_angle,
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
            concentration, beta = self.c, self._args['beta']
            self._profile_args = (concentration, beta)
        return self._profile_args

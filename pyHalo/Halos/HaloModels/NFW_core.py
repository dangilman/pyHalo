from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.nfw import NFW as NFWLenstronomy
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
        self._cnfw_lenstronomy = NFWLenstronomy()
        mdef = 'CNFW'
        if 'conserve_m200' in list(args.keys()):
            pass
        else:
            args['conserve_m200'] = conserve_m200
        super(CoreNFWHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    def density_profile_3d(self, r, profile_args=None, scaling=1.0):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        rhos, rs, _ = self.nfw_params
        x = r/rs
        beta = self._args['beta']
        return scaling * rhos / (x + beta) / (1 + x) ** 2

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
        if self._args['conserve_m200']:
            c = self.c
            m200 = 4 * np.pi * rhos_kpc * rs_kpc ** 3 * (np.log(1+c) - c/(1+c))
            r = np.logspace(-2.5, np.log10(self.c), 2000) * rs_kpc
            rho = self.density_profile_3d(r)
            m = np.trapz(4 * np.pi * rho * r ** 2, r)
            mass_rescale = m200 / m
        else:
            mass_rescale = 1.0
        kwargs = [{'alpha_Rs': mass_rescale * self._rescale_norm * theta_Rs, 'Rs': Rs_angle, 'r_core': beta * Rs_angle,
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

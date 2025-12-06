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

    def density_profile_3d(self, r, kwargs_lenstronomy=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        if kwargs_lenstronomy is None:
            kwargs_lenstronomy = self.lenstronomy_params[0][0]
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        sigma_crit_kpc = sigma_crit_mpc * 1e-6
        factor = sigma_crit_kpc / kpc_per_arcsec
        return factor * self._cnfw_lenstronomy.density_lens(r / kpc_per_arcsec,
                                                            kwargs_lenstronomy['Rs'],
                                                            kwargs_lenstronomy['alpha_Rs'],
                                                            kwargs_lenstronomy['r_core'])

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
        rhos_kpc, rs_kpc, r200_kpc = self.nfw_params
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        Rs_angle = rs_kpc / kpc_per_arcsec
        r_core_angle = Rs_angle * self._args['beta']
        _, alpha_Rs_nfw = self._lens_cosmo.nfw_physical2angle_fromNFWparams(rhos_kpc * 1e9,
                                                                         rs_kpc * 1e-3,
                                                                         self.z,
                                                                            pseudo_nfw=False)
        mass_2d_core = self._cnfw_lenstronomy.mass_2d(rs_kpc, rs_kpc, 1.0, self._args['beta'] * rs_kpc)
        mass_2d_nfw = self._cnfw_lenstronomy.mass_2d(rs_kpc, rs_kpc, 1.0, 1e-4 * rs_kpc)
        theta_Rs = alpha_Rs_nfw * mass_2d_core / mass_2d_nfw

        x, y = np.round(self.x, 4), np.round(self.y, 4)
        Rs_angle = np.round(Rs_angle, 10)
        theta_Rs = np.round(theta_Rs, 10)
        r_core_angle = np.round(r_core_angle, 10)
        kwargs = [{'alpha_Rs': self._rescale_norm * theta_Rs, 'Rs': Rs_angle, 'r_core': r_core_angle,
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

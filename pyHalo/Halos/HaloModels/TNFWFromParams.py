from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.tnfw import TNFW as TNFWLenstronomy
import numpy as np
from pyHalo.Halos.tnfw_halo_util import tnfw_mass_fraction

class TNFWFromParams(Halo):


    KEY_RT = "r_trunc_kpc"
    KEY_RS = "rs"
    KEY_RHO_S = "rho_s"
    KEY_RV = "rv"

    """
    The base class for a truncated NFW halo
    """
    def __init__(self, mass, x_kpc, y_kpc, r3d, z,
                 sub_flag, lens_cosmo_instance,params_physical, args, unique_tag=None):
        """
        Denfines a TNFW subhalo with physical params r_trunc, rs, rhos passed in the args argument
        """
        #Rename args to params_physical ?

        self._lens_cosmo = lens_cosmo_instance

        self._kpc_per_arcsec_at_z = self._lens_cosmo.cosmo.kpc_proper_per_asec(z)

        x = x_kpc / self._kpc_per_arcsec_at_z

        y = y_kpc / self._kpc_per_arcsec_at_z

        self._params_physical = params_physical

        mdef = 'TNFW'

        super(TNFWFromParams, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                           lens_cosmo_instance, args, unique_tag)

    def density_profile_3d(self, r, params_physical=None):
        """
        Computes the 3-D density profile of the halo
        :param r: distance from center of halo [kpc]
        :return: the density profile in units M_sun / kpc^3
        """
        
        _params = self._params_physical if params_physical is None else params_physical

        r_t = _params["rt"]
        r_s = _params["rs"]
        rho_s = _params["rhos"]

        n = 1


        x = r / r_s
        tau = r_t / r_s

        n = 1

        return (rho_s / ((x)*(1+x)**2))(tau**2 / (tau**2 + x**2))**n 


    @property
    def nfw_params(self):
        pass

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
        return self._params_physical[self.KEY_RV] / self._params_physical[self.KEY_RS]
    
    @property
    def params_physical(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        return self._params_physical

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_kwargs_lenstronomy'):

            #TODO change this
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
        #TODO Change this
        if not hasattr(self, '_profile_args'):
            truncation_radius_kpc = self._params_physical[truncation_radius_kpc]
            self.profile_args = (self.c, truncation_radius_kpc)
        return self.profile_args
    
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


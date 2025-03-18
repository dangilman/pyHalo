from pyHalo.Halos.halo_base import Halo
import numpy as np
from pyHalo.Halos.tnfw_halo_util import tnfw_mass_fraction
from lenstronomy.LensModel.Profiles.tnfw import TNFW

class SIS(Halo):
    """
    The base class for an SIS object; this is intended to represent LMC-like objects with enough luminous material
    that they do not have NFW profiles

    This class is intended instantiated from an NFW halo class
    """
    def __init__(self, nfw_halo):
        """

        :param nfw_halo: an instance of TNFW, or TNFWCHalo halo classes
        """
        self._lens_cosmo = nfw_halo.lens_cosmo
        mdef = 'SIS'
        self._nfw_halo = nfw_halo
        super(SIS, self).__init__(mass=nfw_halo.mass,
                                  x=nfw_halo.x,
                                  y=nfw_halo.y,
                                  r3d=nfw_halo.r3d,
                                  mdef=mdef,
                                  z=nfw_halo.z,
                                  sub_flag=nfw_halo.is_subhalo,
                                  lens_cosmo_instance=nfw_halo.lens_cosmo,
                                  args=nfw_halo._args,
                                  unique_tag=nfw_halo.unique_tag,
                                  fixed_position=nfw_halo.fixed_position)

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            vmax = self._nfw_halo.vmax_nfw
            velocity_dispersion = 1.0 * vmax
            self._profile_args = (velocity_dispersion)
        return self._profile_args

    @property
    def bound_mass(self):
        if self.is_subhalo:
            return self._nfw_halo.bound_mass
        else:
            raise Exception('field halos do not have bound mass attribute')

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['TNFW']

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_kwargs_lenstronomy'):
            sigma_v = self.profile_args
            theta_E = self._lens_cosmo.thetaE_from_sigma(self.z, sigma_v)
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            kwargs = [{'theta_E': theta_E, 'center_x': x,  'center_y': y}]
            self._kwargs_lenstronomy = kwargs
        return self._kwargs_lenstronomy, None



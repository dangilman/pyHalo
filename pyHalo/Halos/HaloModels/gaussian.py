from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
import numpy as np

class Gaussian(Halo):
    """
    The base class for a Gaussian fluctuation
    """
    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance

        super(Gaussian, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)
                                            
    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):

            self._profile_args=None

        return self._profile_args

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['GAUSSIAN_KAPPA']

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        kwargs=[{'amp':self._args['amp'],
                'sigma':self._args['sigma'],
                'center_x':self._args['center_x'],
                'center_y':self._args['center_y']}]

        kwargs[0]['amp'] = self._optimize_amplitude(self.mass,kwargs) #scale amplitude to enforce mass definition

        return kwargs,None
    
    @property
    def z_eval(self):
        """
        Returns the halo redshift
        """
        return self.z

    def _optimize_amplitude(self,M,kwargs):
        """
        Total mass of the Gaussian is defined as the mass within a 5sigma radius, returns scaled amplitude to enforce this definition
        """
        M_trial = GaussianKappa().mass_3d_lens(5*kwargs[0]['sigma'],kwargs[0]['amp'],kwargs[0]['sigma'])
        factor = M/M_trial
        return factor*kwargs[0]['amp']


from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
import numpy as np

class Gaussian(Halo):
    """
    The base class for a Gaussian fluctuation

    # kappa0 = amp / (2 * np.pi * sigma ** 2)
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
        mdef = 'GAUSSIAN_KAPPA'
        super(Gaussian, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag, fixed_position=True)

    @property
    def profile_args(self):
        return None

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
        return kwargs, None

    @property
    def z_eval(self):
        """
        Returns the halo redshift
        """
        return self.z

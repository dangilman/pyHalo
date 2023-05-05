import numpy as np
from pyHalo.Halos.halo_base import Halo

class PTMass(Halo):
    """
    Class that defines a point mass object in the lens model
    """
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_clss = concentration_class
        mdef = 'PT_MASS'
        super(PTMass, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['POINT_MASS']

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_params'):

            factor = self._lens_cosmo.point_mass_factor_z(self.z)

            theta_E = factor * np.sqrt(self.mass)

            kwargs = [{'center_x': self.x, 'center_y': self.y, 'theta_E': theta_E}]

            self._lenstronomy_params = kwargs

        return self._lenstronomy_params, None


    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ()

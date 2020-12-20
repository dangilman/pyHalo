import numpy as np
from pyHalo.Halos.halo_base import Halo

class PTMass(Halo):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        self._lens_cosmo = lens_cosmo_instance

        super(PTMass, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        return 'POINT_MASS'

    @property
    def lenstronomy_params(self):

        factor = self._lens_cosmo.point_mass_factor_z(self.z)

        theta_E = factor * np.sqrt(self.mass)

        return {'center_x': self.x, 'center_y': self.y, 'theta_E': theta_E}, None

    @property
    def profile_args(self):

        return ()

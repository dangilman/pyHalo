import numpy.testing as npt
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.cosmology import Cosmology
import pytest
from time import time
import numpy as np
from scipy.integrate import quad

class TestConeGeometry(object):

    def setup(self):

        self.arcsec = 2 * np.pi / 360 / 3600
        self.zlens = 1
        self.zsource = 2
        self.zsrc = 2
        self.angle_diameter = 2 * self.arcsec ** -1
        self.angle_radius = 0.5*self.angle_diameter
        self.cosmo = Cosmology(H0=70, omega_baryon = 0, omega_DM = 0.3, sigma8 = 0.82)
        self.geometry = Geometry(self.cosmo, self.zlens, self.zsource, self.angle_diameter,
                                 rendering_volume='cone')

    def test_distance_computations(self):

        z = 1
        true_comvoing = 3303 # MPC at z=1
        npt.assert_almost_equal(true_comvoing/self.geometry.angle_to_comovingradius(z), 1, 3)

        npt.assert_almost_equal(true_comvoing /
                                self.geometry.angle_to_radius_comoving(self.angle_radius, z), 1, 3)

        true_physical = true_comvoing / (1 + z)
        npt.assert_almost_equal(true_physical / self.geometry.angle_to_physicalradius(z), 1, 3)

    #def test_lensing_distances(self):


test = TestConeGeometry()
test.setup()
test.test_distance_computations()

#if __name__ == '__main__':
#    pytest.main()

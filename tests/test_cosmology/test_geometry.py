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
        self.angle_diameter = 2/self.arcsec
        self.angle_radius = 0.5*self.angle_diameter
        self.cosmo = Cosmology(H0=70, omega_baryon = 0, omega_DM = 0.3, sigma8 = 0.82)
        self.geometry = Geometry(self.cosmo, self.zlens, self.zsource, self.angle_diameter,
                                 rendering_volume='cone')

    def test_distances(self):

        z = 0.3
        radius_physical = self.geometry.angle_to_physicalradius(z)
        npt.assert_almost_equal(radius_physical, 919, 0)

        comoving_radius = self.geometry.angle_to_comovingradius(z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1 + z), 3)

        z = 1
        radius_physical = self.geometry.angle_to_physicalradius(z)
        npt.assert_almost_equal(radius_physical, 1651, 0)

        comoving_radius = self.geometry.angle_to_comovingradius(z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1+z), 3)

    def test_lenscone_distances(self):

        angle_at_zlens = self.geometry.ray_angle_atz(self.angle_diameter, self.zlens, self.zlens)
        npt.assert_almost_equal(angle_at_zlens, self.angle_diameter)

        angle_at_zsource = self.geometry.ray_angle_atz(self.angle_diameter, self.zsource, self.zlens)
        npt.assert_almost_equal(0, angle_at_zsource)

    #def test_lensing_distances(self):


test = TestConeGeometry()
test.setup()
test.test_distances()
test.test_lenscone_distances()

#if __name__ == '__main__':
#    pytest.main()

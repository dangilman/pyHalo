import numpy.testing as npt
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.cosmology import Cosmology
import pytest
from time import time
import numpy as np
from scipy.integrate import quad

class TestGeometry(object):

    def setup(self):

        self.arcsec = 2 * np.pi / 360 / 3600
        self.zlens = 0.9
        self.zsrc = 2
        self.angle = 6
        self.cosmo = Cosmology()
        self.geometry = Geometry(self.cosmo, self.zlens, self.zsrc, self.angle)

    def test_angle_to_comoving_radius(self):

        # test foreground
        z = 0.2
        rad_phys = self.geometry.angle_to_physicalradius(z, self.zlens)
        angle_radian = self.angle * self.arcsec
        rad_phys_true = 0.5 * angle_radian * self.cosmo.D_A(0, z)
        npt.assert_almost_equal(rad_phys, rad_phys_true)

        comoving_radius = rad_phys_true * (1 + z)
        npt.assert_almost_equal(comoving_radius, self.cosmo.T_xy(0, z) * angle_radian * 0.5)
        npt.assert_almost_equal(comoving_radius, self.geometry.angle_to_comovingradius(z, self.zlens))

        asec_per_comoving_kpc = self.cosmo.astropy.arcsec_per_kpc_comoving(z).value
        comoving_kpc = 2 * self.geometry.angle_to_comovingradius(z, self.zlens) * 1000

        npt.assert_almost_equal(comoving_kpc, self.angle / asec_per_comoving_kpc)

        # test background
        z = self.zsrc
        rad_phys = self.geometry.angle_to_physicalradius(z, self.zlens)
        npt.assert_almost_equal(rad_phys, 0)

    def test_dr_comoving(self):

        z = 0.4
        delta_z = 0.0001
        dr = self.geometry._delta_R_comoving(z, delta_z)
        npt.assert_almost_equal(dr, self.cosmo.T_xy(z, z+delta_z), decimal= 3)

    def test_volume(self):

        delta_z = 0.0001
        steradian = (self.angle * self.arcsec**2)
        z = 1

        volume = self.geometry.volume_element_comoving(z, self.zlens, delta_z)

        area = np.pi*self.geometry.angle_to_comovingradius(z, self.zlens) ** 2
        dr = self.geometry._delta_R_comoving(z, delta_z)

        npt.assert_almost_equal(area * dr, volume, decimal = 4)

        def integrand(zi):
            return self.cosmo.astropy.differential_comoving_volume(zi).value

        def volume_astro():
            v1 = quad(integrand, 0, z+delta_z)[0] - quad(integrand, 0, z)[0]
            return v1 * steradian

        vol = volume_astro()

        npt.assert_almost_equal(vol / (volume / np.pi)-1, 0, decimal=1)

    def test_ray_angle_z(self):

        z = 0.4
        angle = self.geometry.ray_angle_atz(self.angle, z, self.zlens)
        npt.assert_almost_equal(angle, self.angle)

        z = self.zsrc
        angle = self.geometry.ray_angle_atz(self.angle, z, self.zlens)
        npt.assert_almost_equal(angle, 0)

        angle = self.geometry.ray_angle_atz(1, z, self.zlens)
        npt.assert_almost_equal(angle, 0)

if __name__ == '__main__':
    pytest.main()

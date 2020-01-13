import numpy.testing as npt
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.cosmology import Cosmology
import pytest
import numpy as np

class TestCone(object):

    def setup(self):

        self.arcsec = 2 * np.pi / 360 / 3600
        self.zlens = 1
        self.zsource = 2
        self.angle_diameter = 2/self.arcsec
        self.angle_radius = 0.5*self.angle_diameter

        H0 = 70
        omega_baryon = 0.0
        omega_DM = 0.3
        sigma8 = 0.82
        curvature = 'flat'
        ns = 0.9608
        cosmo_params = {'H0': H0, 'Om0': omega_baryon + omega_DM, 'Ob0': omega_baryon,
                      'sigma8': sigma8, 'ns': ns, 'curvature': curvature}
        self.cosmo = Cosmology(cosmo_kwargs=cosmo_params)
        self.geometry = Geometry(self.cosmo, self.zlens, self.zsource, self.angle_diameter, 'CONE')

    def test_distances_lensing(self):

        z = 0.3

        radius_physical = self.geometry.angle_to_physicalradius(self.angle_radius, z)
        npt.assert_almost_equal(radius_physical, 919, 0)

        comoving_radius = self.geometry.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1 + z), 3)

        z = 1
        radius_physical = self.geometry.angle_to_physicalradius(self.angle_radius, z)
        npt.assert_almost_equal(radius_physical, 1651, 0)

        comoving_radius = self.geometry.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1+z), 3)

        z = 0.3

    def test_lenscone_angles(self):

        angle_at_zlens = self.geometry.ray_angle_atz(self.angle_radius, self.zlens, 0)
        npt.assert_almost_equal(angle_at_zlens, self.angle_radius)

        angle_at_zsource = self.geometry.ray_angle_atz(self.angle_radius, self.zsource, 0)
        npt.assert_almost_equal(angle_at_zlens, angle_at_zsource)

        angle_at_zsource = self.geometry.ray_angle_atz(self.angle_radius, self.zsource, 0.1)
        npt.assert_almost_equal(angle_at_zlens, angle_at_zsource)

    def test_volume(self):

        cone_arcsec = 3
        radius = cone_arcsec*0.5
        geo = Geometry(self.cosmo, 1, 1.5, cone_arcsec, 'DOUBLE_CONE')
        astropy = geo._cosmo.astropy

        delta_z = 1e-3
        steradian = (radius * self.arcsec) ** 2
        dV_pyhalo = geo.volume_element_comoving(0.6, delta_z)
        dV = astropy.differential_comoving_volume(0.6).value

        # astropy steradian 'area' is d^2, whereas I define area as pi*d^2
        dV_persteradian = dV * delta_z
        npt.assert_almost_equal(dV_persteradian * np.pi * steradian, dV_pyhalo, 5)

t = TestCone()
t.setup()
t.test_lenscone_angles()
#
# if __name__ == '__main__':
#     pytest.main()

import numpy.testing as npt
from pyHalo.Cosmology.geometry import GeometryBase
from pyHalo.Cosmology.cosmology import Cosmology
import pytest
import numpy as np

class TestGeometryCylinder(object):

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
        self.geometry = GeometryBase(self.cosmo, self.zlens, self.zsource, self.angle_diameter, 'CYLINDER')
        self.comoving_radius_at_zlens = self.geometry._geometrytype.comoving_radius_cylinder
        a_zlens = 1/(1+self.zlens)
        self.physical_radius_at_zlens = a_zlens * self.comoving_radius_at_zlens

    def test_distances_lensing(self):

        z = 0.3

        radius_comoving = self.geometry.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(radius_comoving, self.comoving_radius_at_zlens, 4)

        radius_physical = self.geometry.angle_to_physicalradius(self.angle_radius, z)
        radius_physical_2 = radius_comoving * (1 + z) ** -1
        npt.assert_almost_equal(radius_physical, radius_physical_2, 4)

        z = 1
        radius_physical = self.geometry.angle_to_physicalradius(self.angle_radius, z)
        npt.assert_almost_equal(radius_physical, 1651.9144, 4)

        comoving_radius = self.geometry.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1+z), 3)

        z = 1.4
        radius_physical = self.geometry.angle_to_physicalradius(self.angle_radius, z)
        comoving_radius = self.geometry.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1 + z), 3)

    def test_lenscone_angles(self):

        pass

    def test_volume(self):

        cone_arcsec = 3
        radius = cone_arcsec*0.5
        geo = GeometryBase(self.cosmo, 1, 1.5, cone_arcsec, 'CYLINDER')
        astropy = geo._cosmo.astropy

        delta_z = 1e-3
        steradian = (radius * self.arcsec) ** 2
        dV_pyhalo = geo.volume_element_comoving(0.6, delta_z)
        dV = astropy.differential_comoving_volume(0.6).value

        # astropy steradian 'area' is d^2, whereas I define area as pi*d^2
        dV_persteradian = dV * delta_z
        npt.assert_almost_equal(dV_persteradian * np.pi * steradian, dV_pyhalo, 5)

t = TestGeometryCylinder()
t.setup()
t.test_lenscone_angles()
# if __name__ == '__main__':
#     pytest.main()

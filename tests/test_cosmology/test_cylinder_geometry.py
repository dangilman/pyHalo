import numpy.testing as npt
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.cosmology import Cosmology
import pytest
import numpy as np

class TestConeGeometry(object):

    def setup_method(self):

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

        self.geometry_cylinder = Geometry(self.cosmo, self.zlens, self.zsource, self.angle_diameter,
                                             'CYLINDER')

    def test_angle_scale(self):

        scale = self.geometry_cylinder.rendering_scale(self.zlens)
        npt.assert_almost_equal(scale, 1.)

        scale = self.geometry_cylinder.rendering_scale(self.zlens + 0.35)
        dratio = self.geometry_cylinder.cosmo.D_C_z(self.zlens) / self.geometry_cylinder.cosmo.D_C_z(self.zlens + 0.35)
        val = 0.5 * self.geometry_cylinder.cone_opening_angle * self.arcsec * dratio
        npt.assert_almost_equal(scale, val, 3)

        dratio = self.geometry_cylinder.cosmo.D_C_z(self.zlens) / self.geometry_cylinder.cosmo.D_C_z(self.zsource)
        val = 0.5 * self.geometry_cylinder.cone_opening_angle * self.arcsec * dratio
        npt.assert_almost_equal(self.geometry_cylinder.rendering_scale(self.zsource), val)

    def test_distances_lensing(self):

        z = 0.3
        radius_physical = self.geometry_cylinder.angle_to_physicalradius(self.angle_radius, z)
        dratio = self.geometry_cylinder.cosmo.D_C_z(self.zlens) / self.geometry_cylinder.cosmo.D_C_z(z)
        radius = self.geometry_cylinder.cosmo.D_A_z(z) * self.angle_radius * self.arcsec * dratio
        npt.assert_almost_equal(radius_physical, radius, 0)

        comoving_radius = self.geometry_cylinder.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1 + z), 3)

        z = 1
        radius_physical = self.geometry_cylinder.angle_to_physicalradius(self.angle_radius, z)
        dratio = self.geometry_cylinder.cosmo.D_C_z(self.zlens) / self.geometry_cylinder.cosmo.D_C_z(z)
        radius = self.geometry_cylinder.cosmo.D_A_z(z) * self.angle_radius * self.arcsec * dratio
        npt.assert_almost_equal(radius_physical, radius, 0)

        comoving_radius = self.geometry_cylinder.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1+z), 3)

        z = 1.25
        radius_physical = self.geometry_cylinder.angle_to_physicalradius(self.angle_radius, z)
        dratio = self.geometry_cylinder.cosmo.D_C_z(self.zlens) / self.geometry_cylinder.cosmo.D_C_z(z)
        radius = self.geometry_cylinder.cosmo.D_A_z(z) * self.angle_radius * self.arcsec * dratio
        npt.assert_almost_equal(radius_physical, radius, 0)

        comoving_radius = self.geometry_cylinder.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1 + z), 3)

    def test_total_volume(self):

        delta_z = 0.001
        volume_pyhalo = 0
        for zi in np.arange(0, self.zsource+delta_z, delta_z):
            dV_pyhalo = self.geometry_cylinder.volume_element_comoving(zi, delta_z)
            volume_pyhalo += dV_pyhalo

        r = self.geometry_cylinder.cosmo.D_C_z(self.zlens) * self.geometry_cylinder.cone_opening_angle * 0.5 * self.arcsec
        d = self.geometry_cylinder.cosmo.D_C_z(self.zsource)
        volume = np.pi * r ** 2 * d
        npt.assert_almost_equal(volume/volume_pyhalo, 1, 3)

if __name__ == '__main__':
      pytest.main()

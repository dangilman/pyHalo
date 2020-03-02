import numpy.testing as npt
from pyHalo.Cosmology.geometry import Geometry, Cylinder
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.pyhalo import pyHalo
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
        self.geometry = Geometry(self.cosmo, self.zlens, self.zsource, self.angle_diameter, 'CYLINDER')
        self.comoving_radius_at_zlens = self.geometry._geometrytype.comoving_radius_cylinder
        self.comoving_radius_at_zsource = self.angle_radius * self.cosmo.D_C_transverse(self.zsource)
        a_zlens = 1/(1+self.zlens)
        self.physical_radius_at_zlens = a_zlens * self.comoving_radius_at_zlens

        cosmo_params_instance = {'cosmo_kwargs': cosmo_params}
        pyhalo_instance = pyHalo(self.zlens, self.zsource, cosmology_kwargs=cosmo_params_instance,
                                 kwargs_halo_mass_function={'geometry_type': 'CYLINDER'})
        args = {'cone_opening_angle': self.angle_diameter}
        mfunc = pyhalo_instance._build_LOS_mass_function(args)
        geometry_from_instance = mfunc.geometry

        assert isinstance(geometry_from_instance._geometrytype, Cylinder)

    def test_distances_lensing(self):

        z = 0.3

        radius_comoving_0 = self.geometry.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(radius_comoving_0, self.comoving_radius_at_zlens, 4)

        radius_physical = self.geometry.angle_to_physicalradius(self.angle_radius, z)
        radius_physical_2 = radius_comoving_0 * (1 + z) ** -1
        npt.assert_almost_equal(radius_physical, radius_physical_2, 4)

        z = 1
        radius_physical = self.geometry.angle_to_physicalradius(self.angle_radius, z)
        npt.assert_almost_equal(radius_physical, radius_comoving_0 * (1+z) ** -1)

        comoving_radius = self.geometry.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1+z), 3)

        z = 1.4
        radius_physical = self.geometry.angle_to_physicalradius(self.angle_radius, z)
        comoving_radius = self.geometry.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1 + z), 3)
        npt.assert_almost_equal(comoving_radius, radius_comoving_0)

    def test_rendering_scale(self):

        z = self.zlens
        scale = self.geometry.rendering_scale(z)
        npt.assert_almost_equal(scale, 1)

        z = 0.35
        scale = self.geometry.rendering_scale(z)
        scale_theory = self.cosmo.D_C_transverse(self.zlens)/self.cosmo.D_C_transverse(z)
        npt.assert_almost_equal(scale, scale_theory)

        z = 1.4
        scale = self.geometry.rendering_scale(z)
        scale_theory = self.cosmo.D_C_transverse(self.zlens) / self.cosmo.D_C_transverse(z)
        npt.assert_almost_equal(scale, scale_theory)

    def test_lenscone_angles(self):

        z = self.zlens
        angle = self.geometry.ray_angle_atz(self.angle_radius, z)
        npt.assert_almost_equal(angle, self.angle_radius)

        z = 0.5
        angle = self.geometry.ray_angle_atz(self.angle_radius, z)
        npt.assert_almost_equal(angle, self.angle_radius)

        z = 1.5
        angle = self.geometry.ray_angle_atz(self.angle_radius, z)
        npt.assert_almost_equal(angle, self.angle_radius)

    def test_volume(self):

        radius_arcsec = 1

        z = 0.2
        radius_comoving = self.geometry.angle_to_comovingradius(radius_arcsec, z)
        area = np.pi * radius_comoving ** 2
        npt.assert_almost_equal(area, self.geometry.angle_to_comoving_area(radius_arcsec, z))

        z = self.zlens
        radius_comoving = self.geometry.angle_to_comovingradius(radius_arcsec, z)
        area = np.pi * radius_comoving ** 2
        npt.assert_almost_equal(area, self.geometry.angle_to_comoving_area(radius_arcsec, z))

        z = self.zlens+0.25
        radius_comoving = self.geometry.angle_to_comovingradius(radius_arcsec, z)
        area = np.pi * radius_comoving ** 2
        npt.assert_almost_equal(area, self.geometry.angle_to_comoving_area(radius_arcsec, z))

        z, delta_z = 0.5, 1e-4
        area = self.geometry.angle_to_comoving_area(self.angle_radius, z)
        dV_pyhalo = self.geometry.volume_element_comoving(z, delta_z)
        dR = self.geometry.delta_R_comoving(z, delta_z)
        dV_theory = area * dR
        npt.assert_almost_equal(dV_pyhalo, dV_theory)

        z = 0.3
        area = self.geometry.angle_to_comoving_area(self.angle_radius, z)
        dV_theory_2 = area * dR
        npt.assert_almost_equal(dV_theory_2, dV_theory)

if __name__ == '__main__':
    pytest.main()

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

        self._angle_pad = 0.75
        self.geometry_double_cone = Geometry(self.cosmo, self.zlens, self.zsource, self.angle_diameter,
                                             'DOUBLE_CONE', self._angle_pad)

    def test_angle_scale(self):

        reduced_to_phys = self.geometry_double_cone.cosmo.D_A(0, self.zsource) / \
                          self.geometry_double_cone.cosmo.D_A(self.zlens, self.zsource)

        ratio_double_cone = reduced_to_phys * \
                            self.cosmo.D_A(self.zlens, self.zsource)/self.cosmo.D_A_z(self.zsource)
        angle_scale_zsource = 1 - self._angle_pad*ratio_double_cone

        npt.assert_almost_equal(self.geometry_double_cone.rendering_scale(self.zlens), 1.)
        npt.assert_almost_equal(self.geometry_double_cone.rendering_scale(self.zsource), angle_scale_zsource)

        npt.assert_almost_equal(self.geometry_double_cone.rendering_scale(self.zlens), 1)

    def test_distances_lensing(self):

        z = 0.3
        radius_physical = self.geometry_double_cone.angle_to_physicalradius(self.angle_radius, z)
        radius = self.geometry_double_cone.cosmo.D_A(0., z) * self.angle_radius * self.arcsec
        npt.assert_almost_equal(radius_physical, radius, 0)

        comoving_radius = self.geometry_double_cone.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1 + z), 3)

        z = 1
        radius_physical = self.geometry_double_cone.angle_to_physicalradius(self.angle_radius, z)
        radius = self.geometry_double_cone.cosmo.D_A(0., z) * self.angle_radius * self.arcsec
        npt.assert_almost_equal(radius_physical, radius, 0)

        comoving_radius = self.geometry_double_cone.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1+z), 3)

        z = 1.25
        radius_physical = self.geometry_double_cone.angle_to_physicalradius(self.angle_radius, z)
        D_dz = self.geometry_double_cone.cosmo.D_A(self.zlens, z)
        D_s = self.geometry_double_cone.cosmo.D_A(0, self.zsource)
        D_z = self.geometry_double_cone.cosmo.D_A(0, z)
        D_ds = self.geometry_double_cone.cosmo.D_A(self.zlens, self.zsource)

        rescale = 1 - self._angle_pad * D_dz * D_s / (D_z * D_ds)
        radius = self.geometry_double_cone.cosmo.D_A(0., z) * self.angle_radius * self.arcsec * rescale

        npt.assert_almost_equal(radius_physical, radius, 0)

        comoving_radius = self.geometry_double_cone.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1 + z), 3)


    def test_volume(self):

        cone_arcsec = 3
        radius = cone_arcsec*0.5
        angle_pad = 0.7
        zlens = 1.
        zsrc = 1.8
        geo = Geometry(self.cosmo, zlens, zsrc, cone_arcsec, 'DOUBLE_CONE', angle_pad=angle_pad)
        astropy = geo.cosmo.astropy

        delta_z = 1e-3

        dV_pyhalo = geo.volume_element_comoving(0.6, delta_z)
        dV = astropy.differential_comoving_volume(0.6).value

        dV_astropy = dV * delta_z
        steradian = np.pi * (radius * self.arcsec) ** 2
        npt.assert_almost_equal(dV_astropy * steradian, dV_pyhalo, 5)

        angle_scale = geo.rendering_scale(1.3)
        dV_pyhalo = geo.volume_element_comoving(1.3, delta_z)
        dV = astropy.differential_comoving_volume(1.3).value
        dV_astropy = dV * delta_z
        steradian = np.pi * (radius * angle_scale * self.arcsec) ** 2
        npt.assert_almost_equal(dV_astropy * steradian, dV_pyhalo, 5)

    def test_total_volume(self):

        cone_arcsec = 4
        radius_radians = cone_arcsec * 0.5 * self.cosmo.arcsec
        geo = Geometry(self.cosmo, 0.5, 1.5, cone_arcsec, 'DOUBLE_CONE', angle_pad=1.)
        volume_pyhalo = 0
        z = np.linspace(0.0, 1.5, 200)
        for i in range(0, len(z)-1):
            delta_z = z[i+1] - z[i]
            volume_pyhalo += geo.volume_element_comoving(z[i], delta_z)

        ds = self.cosmo.D_C_z(1.5)
        dz = self.cosmo.D_C_z(0.5)
        volume_true = 1./3 * np.pi * radius_radians ** 2 * dz ** 2 * ds
        npt.assert_almost_equal(volume_true, volume_pyhalo, 3)


if __name__ == '__main__':
      pytest.main()

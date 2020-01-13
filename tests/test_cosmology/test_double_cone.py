import numpy.testing as npt
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.pyhalo import pyHalo
import pytest
import numpy as np

class TestDoubleCone(object):

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
        self.geometry = Geometry(self.cosmo, self.zlens, self.zsource, self.angle_diameter, 'DOUBLE_CONE')

        cosmo_params_instance = {'cosmo_kwargs': cosmo_params}
        pyhalo_instance = pyHalo(self.zlens, self.zsource, cosmology_kwargs=cosmo_params_instance,
                                 kwargs_halo_mass_function={'geometry_type': 'DOUBLE_CONE'})
        args = {'cone_opening_angle': self.angle_diameter}
        mfunc = pyhalo_instance._build_LOS_mass_function(args)
        self.geometry_from_instance = mfunc.geometry

    def test_distances_lensing(self):

        z = 0.3

        radius_physical = self.geometry.angle_to_physicalradius(self.angle_radius, z)
        npt.assert_almost_equal(radius_physical, 919, 0)
        radius_physical_instance = self.geometry_from_instance.angle_to_physicalradius(self.angle_radius, z)
        npt.assert_almost_equal(radius_physical_instance, radius_physical, 4)

        comoving_radius = self.geometry.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1 + z), 3)
        comoving_radius_from_instance = self.geometry_from_instance.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius_from_instance, comoving_radius)

        z = 1.
        radius_physical = self.geometry.angle_to_physicalradius(self.angle_radius, z)
        npt.assert_almost_equal(radius_physical, 1651, 0)

        comoving_radius = self.geometry.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius, radius_physical * (1+z), 3)
        comoving_radius_from_instance = self.geometry_from_instance.angle_to_comovingradius(self.angle_radius, z)
        npt.assert_almost_equal(comoving_radius_from_instance, comoving_radius)

    def test_lenscone_angles(self):

        angle_at_zlens = self.geometry.ray_angle_atz(self.angle_diameter, self.zlens, 0)
        npt.assert_almost_equal(angle_at_zlens, self.angle_diameter)
        angle_at_zlens_from_instance = self.geometry_from_instance.ray_angle_atz(self.angle_diameter, self.zlens, 0)
        npt.assert_almost_equal(angle_at_zlens_from_instance, angle_at_zlens)

        angle_at_zsource = self.geometry.ray_angle_atz(self.angle_radius, self.zsource, 0)
        npt.assert_almost_equal(0, angle_at_zsource)
        angle_at_zsource_from_instance = self.geometry_from_instance.ray_angle_atz(self.angle_radius, self.zsource, 0)
        npt.assert_almost_equal(angle_at_zsource_from_instance, angle_at_zsource)

        angle_at_zsource = self.geometry.ray_angle_atz(self.angle_radius, self.zsource, 0.1)
        npt.assert_almost_equal(0.1, angle_at_zsource)
        angle_at_zsource_from_instance = self.geometry_from_instance.ray_angle_atz(self.angle_radius, self.zsource, 0.1)
        npt.assert_almost_equal(angle_at_zsource_from_instance, angle_at_zsource)

        z1, z2, z3 = 0.3, 0.7, 0.48
        theta = self.geometry._cosmo.arcsec ** -1
        xlow = theta * self.geometry._cosmo.D_C(z1)
        xhigh = theta * self.geometry._cosmo.D_C(z2)
        ylow = 0.5*theta * self.geometry._cosmo.D_C(z1)
        yhigh = 0.5*theta * self.geometry._cosmo.D_C(z2)

        Tzlow = self.geometry._cosmo.D_C(z1)
        Tzhigh = self.geometry._cosmo.D_C(z2)
        Tzcurrent = self.geometry._cosmo.D_C(z3)

        x_interp, y_interp = self.geometry.interp_ray_angle(xlow, xhigh, ylow, yhigh, Tzlow, Tzhigh, Tzcurrent)

        npt.assert_almost_equal(x_interp, theta)
        npt.assert_almost_equal(y_interp, 0.5 * theta)

        z1, z2, z3 = 1, 2, 1.6
        xlow = self.geometry.angle_to_comovingradius(theta, z1)
        xhigh = self.geometry.angle_to_comovingradius(theta, z2)
        ylow = self.geometry.angle_to_comovingradius(0.5 * theta, z1)
        yhigh = self.geometry.angle_to_comovingradius(0.5 * theta, z2)
        Tzlow = self.geometry._cosmo.D_C(z1)
        Tzhigh = self.geometry._cosmo.D_C(z2)
        Tzcurrent = self.geometry._cosmo.D_C(z3)
        x_interp, y_interp = self.geometry.interp_ray_angle(xlow, xhigh, ylow, yhigh, Tzlow, Tzhigh, Tzcurrent)

        run = Tzhigh - Tzlow
        risex, risey = 0 - xlow, 0 - ylow
        delta = Tzcurrent - Tzlow

        x_true = (delta * risex / run + xlow) / Tzcurrent
        y_true = (delta * risey / run + ylow) / Tzcurrent

        npt.assert_almost_equal(x_true, x_interp)
        npt.assert_almost_equal(y_true, y_interp)

        z1, z2, z3 = 1, 2, 1.99999
        xlow = self.geometry.angle_to_comovingradius(theta, z1)
        xhigh = self.geometry.angle_to_comovingradius(theta, z2)
        ylow = self.geometry.angle_to_comovingradius(0.5 * theta, z1)
        yhigh = self.geometry.angle_to_comovingradius(0.5 * theta, z2)
        Tzlow = self.geometry._cosmo.D_C(z1)
        Tzhigh = self.geometry._cosmo.D_C(z2)
        Tzcurrent = self.geometry._cosmo.D_C(z3)
        x_interp, y_interp = self.geometry.interp_ray_angle(xlow, xhigh, ylow, yhigh, Tzlow, Tzhigh, Tzcurrent)

        x_true = 0
        y_true = 0

        npt.assert_almost_equal(x_true, x_interp, 5)
        npt.assert_almost_equal(y_true, y_interp, 5)

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

if __name__ == '__main__':
    pytest.main()

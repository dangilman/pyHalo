import numpy as np
from pyHalo.Cosmology.lens_cosmo import LensCosmo
from scipy.integrate import quad
#from pyHalo.defaults import *

class Geometry(object):

    _delta_z_min = 1e-4

    def __init__(self, cosmology, z_lens, z_source, opening_angle):

        self._cosmo = cosmology
        self._lens_cosmo = LensCosmo(z_lens, z_source, cosmology)
        self._reduced_to_phys = self._lens_cosmo.D_s * self._lens_cosmo.D_ds ** -1

        self._zlens, self._zsource = z_lens, z_source

        self._d_comoving_zlens = self._cosmo.D_C_transverse(z_lens)

        self.cone_opening_angle = opening_angle
        self._arcsec = self._cosmo.arcsec

    def angle_to_physicalradius(self, radius_arcsec, z):

        angle_radian = radius_arcsec * self._arcsec

        return angle_radian * self.rendering_scale(z) * self._cosmo.D_A(0, z)

    def angle_to_comovingradius(self, radius_arcsec, z):

        """
        This is specific to the geometry, e.g. if you pick cone if will close behind the main
        deflector
        """
        a_z = self._cosmo.scale_factor(z)
        return self.angle_to_physicalradius(radius_arcsec, z) / a_z

    def rendering_scale(self, z):

        if z <= self._zlens:
            return 1
        else:
            D_dz = self._cosmo.D_A(self._zlens, z)
            D_z = self._cosmo.D_A(0, z)
            ratio = D_dz / D_z

            return 1 - self._reduced_to_phys * ratio

    def ray_angle_atz(self, theta_arcsec, z, source_pos=0):

        if z <= self._zlens:
            return theta_arcsec
        else:

            D_dz = self._cosmo.D_A(self._zlens, z)
            D_z = self._cosmo.D_A(0, z)

            delta_theta = theta_arcsec - source_pos
            subtract_angle = delta_theta * (D_dz / D_z) * self._reduced_to_phys

            return theta_arcsec - subtract_angle

    def interp_ray_angle(self, xlow, xhigh, ylow, yhigh, Tzlow, Tzhigh, Tz_current):

        delta = Tz_current - Tzlow
        rise_x, rise_y = xhigh - xlow, yhigh - ylow
        run = Tzhigh - Tzlow
        slope_x, slope_y = rise_x / run, rise_y / run

        comoving_x, comoving_y = delta*slope_x + xlow, delta*slope_y + ylow

        return comoving_x/Tz_current, comoving_y/Tz_current

    def volume_element_comoving(self, z, delta_z):
        """

        :param theta:
        :param z_lens:e
        :param z:
        :param delta_z:
        :return: volume element in comoving Mpc for small delta_z
        """

        cone_radius = 0.5*self.cone_opening_angle
        if delta_z > self._delta_z_min:
            func = self._volume_integrand_comoving
            volume_element = quad(func, z, z+delta_z, args=(cone_radius))[0]

        else:

            volume_element = self._volume_integrand_comoving(z, cone_radius) * delta_z

        return volume_element

    def _volume_integrand_comoving(self, z, radius_arcsec):
        """

        :param theta:
        :param z_lens:
        :param z:
        :return: integrand element in comoving Mpc (for use in scipy.integrate quad)
        """
        area_comoving = self._angle_to_comoving_area(radius_arcsec, z)

        dR = self._cosmo.astropy.hubble_distance.value * self._cosmo.astropy.efunc(z) ** -1

        return area_comoving * dR

    def _delta_R_comoving(self, z, delta_z):

        d_h = self._cosmo.astropy.hubble_distance.value
        Ez_inv = self._cosmo.astropy.efunc(z)**-1

        return d_h * Ez_inv * delta_z

    def _angle_to_arcsec_area(self, radius_arcsec, z):

        theta = self._angle_to_arcsec_radius(radius_arcsec, z)

        return np.pi * theta ** 2

    def _angle_to_comoving_area(self, radius_arcsec, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        r = self.angle_to_comovingradius(radius_arcsec, z)

        return np.pi * r ** 2

    def _angle_to_physical_area(self, radius_arcsec, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        area_comoving = self._angle_to_comoving_area(radius_arcsec, z)

        a_z = self._cosmo.scale_factor(z)

        return area_comoving * a_z ** 2

    def _angle_to_arcsec_radius(self, radius_arcsec, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        r_co = self.angle_to_comovingradius(radius_arcsec, z)
        asec_per_mpc = self._cosmo.astropy.arcsec_per_kpc_comoving(z).value * 1000

        return r_co * asec_per_mpc




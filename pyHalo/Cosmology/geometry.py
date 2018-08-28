import numpy as np
from pyHalo.Cosmology.lens_cosmo import LensCosmo
from scipy.integrate import quad

class Geometry(object):

    _delta_z_min = 0.02

    def __init__(self,cosmology,z_lens,z_source,delta_theta_lens,cone_opening_angle):

        self._cosmo = cosmology
        self._lens_cosmo = LensCosmo(z_lens,z_source)
        self._reduced_to_phys = self._lens_cosmo.D_s * self._lens_cosmo.D_ds**-1
        if delta_theta_lens is None:
            delta_theta_lens = -cone_opening_angle
        self.delta_theta_lens = delta_theta_lens
        self.cone_opening_angle = cone_opening_angle

        self._zlens, self._zsource = z_lens, z_source

    def lens_cone_angle(self, z, z_lens):
        """

        :param z: redshift
        :param z_lens: lens redshift
        :param delta_theta_lens: lens deflection angle (reduced)
        :return: angular size (in arcsec) of lens cone at redshift z
        """

        if z <= z_lens:
            return self.cone_opening_angle

        R_physical = self.angle_to_physicalradius(z, z_lens)

        R_radians = R_physical*self._cosmo.D_A(0,z)**-1

        R_arcsec = R_radians * self._cosmo.arcsec**-1

        return R_arcsec

    def volume_element_comoving(self, z, z_lens, delta_z):
        """

        :param theta:
        :param z_lens:
        :param z:
        :param delta_z:
        :return: volume element in comoving Mpc for small delta_z
        """

        if delta_z > self._delta_z_min:
            func = self._volume_integrand_comoving
            args = (z_lens)
            volume_element = quad(func, z, z+delta_z, args=args)[0]

        else:
            area_comoving = self._angle_to_comoving_area(z_lens, z)
            volume_element = area_comoving * self._delta_R_comoving(z, delta_z)

        return volume_element

    def volume_element_physical(self, z, z_lens, delta_z):

        volume_comoving = self.volume_element_comoving(z, z_lens, delta_z)

        return self._cosmo.scale_factor(z) ** 3 * volume_comoving

    def angle_to_physicalradius(self, z, z_lens):

        """

        :param angle: cone opening angle
        :param z: redshift
        :param z_lens: lens redshift
        :return: physical radius corresponding to the path of a light ray at redshift z
        """

        # convert to radians
        angle_radian = self.cone_opening_angle * self._cosmo.arcsec
        R_in = angle_radian * self._cosmo.D_A(0, z)

        if z <= z_lens:
            # in front of main deflector
            R = R_in
        else:
            # convert reduced deflection angle to physical deflection angle
            angle_deflection_reduced = self.delta_theta_lens * self._cosmo.arcsec
            angle_deflection = angle_deflection_reduced * self._reduced_to_phys

            # subtract the main deflector deflection
            R = R_in - angle_deflection * self._cosmo.D_A(z_lens, z)

        return R

    def angle_to_comovingradius(self, z, z_lens):

        return self._cosmo.scale_factor(z)**-1 * self.angle_to_physicalradius(z, z_lens)

    def _volume_integrand_comoving(self, z, z_lens):
        """

        :param theta:
        :param z_lens:
        :param z:
        :return: integrand element in comoving Mpc (for use in scipy.integrate quad)
        """
        area_comoving = self._angle_to_comoving_area(z_lens, z)

        return area_comoving * self._cosmo.astropy.hubble_distance.value * self._cosmo.astropy.efunc(z)

    def _delta_R_comoving(self, z, delta_z):

        delta_comoving = self._cosmo.astropy.hubble_distance.value * self._cosmo.astropy.efunc(z) * delta_z

        return delta_comoving

    def _angle_to_arcsec_area(self, z_lens, z):

        if z==0:
            return 0

        r = self.angle_to_physicalradius(z, z_lens)

        area_radians = np.pi * r ** 2 * self._cosmo.D_A(0,z) ** -2

        return area_radians * self._cosmo.arcsec ** -2

    def _angle_to_comoving_area(self, z_lens, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        R_comoving = self.angle_to_comovingradius(z, z_lens)

        return np.pi*R_comoving**2

    def _angle_to_physical_area(self, z_lens, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        area_comoving = self._angle_to_comoving_area(z, z_lens)

        return np.pi * area_comoving * (self._cosmo.scale_factor(z)) ** 2

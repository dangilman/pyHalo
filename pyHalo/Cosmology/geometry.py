import numpy as np
from pyHalo.Cosmology.lens_cosmo import LensCosmo
from scipy.integrate import quad
from pyHalo.defaults import distance_resolution_MPC

class Geometry(object):

    _delta_z_min = 1e-4

    def __init__(self,cosmology,z_lens,z_source,cone_opening_angle):

        self._cosmo = cosmology
        self._lens_cosmo = LensCosmo(z_lens,z_source)
        self._reduced_to_phys = self._lens_cosmo.D_s * self._lens_cosmo.D_ds**-1
        self.delta_theta_lens = cone_opening_angle
        #self.delta_theta_lens = delta_theta_lens
        self.cone_opening_angle = cone_opening_angle

        self._zlens, self._zsource = z_lens, z_source

    def ray_angle_atz(self, theta, z, z_lens):

        if z <= z_lens:
            return theta

        delta_DA_z = self._cosmo.D_A(0, z)
        delta_DA_zlens_z = self._cosmo.D_A(z_lens, z)

        # convert reduced deflection angle to physical deflection angle
        angle_deflection_reduced = theta
        angle_deflection = angle_deflection_reduced * self._reduced_to_phys

        # subtract the main deflector deflection
        theta = (theta - angle_deflection * delta_DA_zlens_z * delta_DA_z**-1)

        return theta

    def angle_between_planes(self, z1, z2, z3, theta1, theta2):

        x12 = self._cosmo.T_xy(z1, z2)
        x2 = self._cosmo.T_xy(0, z2)
        x13 = self._cosmo.T_xy(z1, z3)
        x3 = self._cosmo.T_xy(0, z3)

        d_theta = (theta1 - theta2) * (x2 / x12)

        return theta1 - d_theta * (x13 / x3)

    def _volume_integrand_comoving(self, z, z_lens):
        """

        :param theta:
        :param z_lens:
        :param z:
        :return: integrand element in comoving Mpc (for use in scipy.integrate quad)
        """
        area_comoving = self._angle_to_comoving_area(z_lens, z)

        return area_comoving * self._cosmo.astropy.hubble_distance.value * self._cosmo.astropy.efunc(z) ** -1

    def volume_element_comoving(self, z, z_lens, delta_z):
        """

        :param theta:
        :param z_lens:e
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

    def _delta_R_comoving(self, z, delta_z):

        d_h = self._cosmo.astropy.hubble_distance.value
        Ez_inv = self._cosmo.astropy.efunc(z)**-1

        return d_h * Ez_inv * delta_z

    def angle_to_physicalradius(self, z, z_lens):

        """

        :param z: redshift
        :param z_lens: lens redshift
        :return: physical radius of the cone
        corresponding to the path of a light ray at redshift z
        """

        # convert to radians

        angle_radian = self.cone_opening_angle * self._cosmo.arcsec

        R_in = angle_radian * self._cosmo.D_A(0, z)

        if z <= z_lens:
            # in front of main deflector
            R = R_in
        else:

            angle_deflection_reduced = self.delta_theta_lens * self._cosmo.arcsec
            angle_deflection = angle_deflection_reduced * self._reduced_to_phys

            # subtract the main deflector deflection
            R = R_in - angle_deflection * self._cosmo.D_A(z_lens, z)

        return 0.5 * R

    def solid_angle_z(self, z, z_lens):

        angle = self.angle_to_physicalradius(z, z_lens)

        omega = angle / self._cosmo.D_A(0, z)

        return omega ** 2

    def angle_to_comovingradius(self, z, z_lens):

        a_z = self._cosmo.scale_factor(z)
        return self.angle_to_physicalradius(z, z_lens) / a_z

    def _angle_to_arcsec_area(self, z_lens, z):

        omega = self.solid_angle_z(z, z_lens)

        return omega * self._cosmo.arcsec ** -2

    def _angle_to_comoving_area(self, z_lens, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        sigma = self.solid_angle_z(z, z_lens) * self._cosmo.T_xy(0, z) ** 2

        return sigma

    def _angle_to_physical_area(self, z_lens, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        area_comoving = self._angle_to_comoving_area(z, z_lens)

        a_z = self._cosmo.scale_factor(z)

        return area_comoving * a_z ** 2


import numpy as np
from pyHalo.Cosmology.lens_cosmo import LensCosmo
from scipy.integrate import quad
#from pyHalo.defaults import *

class Geometry(object):

    _delta_z_min = 1e-4

    def __init__(self, cosmology, z_lens, z_source, opening_size,
                 rendering_volume='cone'):

        self._lens_cosmo = LensCosmo(z_lens, z_source)
        self._reduced_to_phys = self._lens_cosmo.D_s * self._lens_cosmo.D_ds ** -1
        self._zlens, self._zsource = z_lens, z_source

        if rendering_volume == 'cylinder':
            self._rendering_volume = 'cylinder'
            raise Exception('cylinder not yet implemented')
            self._geometry = CylinderGeometry(cosmology, z_lens, z_source, opening_size,
                                              self._reduced_to_phys)

        elif rendering_volume == 'cone':
            self._rendering_volume = 'cone'
            self._geometry = ConeGeometry(cosmology, z_lens, z_source, opening_size,
                                          self._reduced_to_phys)
        else:
            raise Exception('volume type ' + str(rendering_volume) + ' not recognized.')

        self._cosmo = self._geometry._cosmo
        self._d_comoving_zlens = self._cosmo.D_C_transverse(z_lens)

    def angle_to_radius_comoving(self, radius, z):
        """
        This is just geometry, not the 'lensing' angle!
        angle in arcseconds
        """
        angle_radian = radius * self._cosmo.arcsec
        return angle_radian*self._cosmo.D_C_transverse(z)

    def angle_to_comovingradius(self, z):

        """
        This is specific to the geometry, e.g. if you pick cone if will close behind the main
        deflector
        """
        a_z = self._cosmo.scale_factor(z)
        return self._geometry._angle_to_physicalradius(z) / a_z

    def angle_to_physicalradius(self, z):

        return self._geometry._angle_to_physicalradius(z)

    def ray_angle_atz(self, theta, z, z_lens):

        if z <= z_lens:
            return theta * self._cosmo.D_C_transverse(z) / self._d_comoving_zlens
        else:
            # tp * Dz = tE * Dz - alpha * D_dz
            ratio = (1 - self._d_comoving_zlens / self._cosmo.D_C_transverse(z))
            alpha = theta * self._reduced_to_phys

            return theta * (1 - ratio*alpha)

    def rendering_scale(self, z):

        return self._geometry._rendering_scale(z)

    def volume_element_comoving(self, z, delta_z):
        """

        :param theta:
        :param z_lens:e
        :param z:
        :param delta_z:
        :return: volume element in comoving Mpc for small delta_z
        """

        if delta_z > self._delta_z_min:
            func = self._volume_integrand_comoving
            volume_element = quad(func, z, z+delta_z)[0]

        else:

            volume_element = self._volume_integrand_comoving(z) * delta_z

        return volume_element

    def _volume_integrand_comoving(self, z):
        """

        :param theta:
        :param z_lens:
        :param z:
        :return: integrand element in comoving Mpc (for use in scipy.integrate quad)
        """
        area_comoving = self._angle_to_comoving_area(z)

        dR = self._cosmo.astropy.hubble_distance.value * self._cosmo.astropy.efunc(z) ** -1

        return area_comoving * dR

    def _delta_R_comoving(self, z, delta_z):

        d_h = self._cosmo.astropy.hubble_distance.value
        Ez_inv = self._cosmo.astropy.efunc(z)**-1

        return d_h * Ez_inv * delta_z

    def _angle_to_arcsec_area(self, z):

        theta = self._angle_to_arcsec_radius(z)

        return np.pi * theta ** 2

    def _angle_to_comoving_area(self, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        r = self.angle_to_comovingradius(z)

        return np.pi * r ** 2

    def _angle_to_physical_area(self, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        area_comoving = self._angle_to_comoving_area(z)

        a_z = self._cosmo.scale_factor(z)

        return area_comoving * a_z ** 2

    def _angle_to_arcsec_radius(self, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        r_co = self.angle_to_comovingradius(z)
        asec_per_mpc = self._cosmo.astropy.arcsec_per_kpc_comoving(z).value * 1000

        return r_co * asec_per_mpc


class ConeGeometry(object):

    _delta_z_min = 1e-4

    def __init__(self, cosmo, zlens, zsource, opening_radius, reduced_to_phys):

        self._zlens = zlens
        self._zsource = zsource
        self._cosmo = cosmo
        self._reduced_to_phys = reduced_to_phys
        self.cone_opening_angle = opening_radius
        self._dd_comoving = self._cosmo.D_C_transverse(self._zlens)
        self._d_ds_comoving = self._cosmo.D_C_transverse(self._zsource) - self._dd_comoving

    def _angle_to_physicalradius(self, z):

        # convert to radians

        angle_radian = self.cone_opening_angle * self._cosmo.arcsec

        R_in = angle_radian * self._cosmo.D_A(0, z)

        if z <= self._zlens:
            # in front of main deflector
            R = R_in
        else:

            angle_deflection_reduced = self.cone_opening_angle * self._cosmo.arcsec
            angle_deflection = angle_deflection_reduced * self._reduced_to_phys

            # subtract the main deflector deflection
            R = R_in - angle_deflection * self._cosmo.D_A(self._zlens, z)

        return 0.5 * R

    def _rendering_scale(self, z):

        if z <= self._zlens:

            return 1
        else:
            delta = self._cosmo.D_C_transverse(z) - self._dd_comoving
            dratio = delta / self._d_ds_comoving
            return 1 - dratio

class CylinderGeometry(object):

    _delta_z_min = 1e-4

    def __init__(self, cosmo, zlens, zsource, opening_radius, reduced_to_phys):

        self._zlens = zlens
        self._zsource = zsource
        self._cosmo = cosmo
        self._reduced_to_phys = reduced_to_phys
        self.cone_opening_angle = opening_radius
        self._dd_comoving = self._cosmo.D_C_transverse(self._zlens)

    def _angle_to_physicalradius(self, z):

        # convert to radians

        angle_radian = self.cone_opening_angle * self._cosmo.arcsec

        R = angle_radian * self._dd_comoving

        return 0.5 * R

    def _rendering_scale(self, z):

        return self._dd_comoving/self._cosmo.D_C_transverse(z)

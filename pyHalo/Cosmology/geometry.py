import numpy as np
from scipy.integrate import quad
#from pyHalo.defaults import *

class Geometry(object):

    _delta_z_min = 1e-4

    def __init__(self, cosmology, z_lens, z_source, opening_angle, geometry_type,
                 angle_pad=0.95):

        """
        This class handles geometric calculations for rendering halos along the line of sight

        :param cosmology: an instance of the Cosmology() class in pyHalo
        :param z_lens: lens redshift
        :param z_source: source redshift
        :param opening_angle: the opening angle of the rendering area in arcsec.
        For a double cone configuration, for example, the angular size of the area in which halos are rendered
        is equal to 2 * cone_opening_angle for z < z_lens, and closes towards the source redshift
        :param geometry_type: the type of geometry
        DOUBLE_CONE opens towards the lens redshift and closes behind it
        CYLINDER generates objects in a cylinder; the physical radius of the cylinder is the angular diameter distance
        to the lens times 2 * cone_opening_angle (in radians)
        :param angle_pad: modifies the closure of the rendering volume behind the main deflector, only relevant for
        DOUBLE_CONE geometries. Values less than one will result in a non-zero rendering area at the source redshift
        """

        if geometry_type == 'DOUBLE_CONE':
            self._geometrytype = DoubleCone(cosmology, z_lens, z_source, opening_angle, angle_pad)
            self.volume_type = 'DOUBLE_CONE'
        elif geometry_type == 'CYLINDER':
            self._geometrytype = Cylinder(cosmology, z_lens, z_source, opening_angle)
            self.volume_type = 'CYLINDER'
        elif geometry_type == 'CONE':
            self._geometrytype = Cone(cosmology, z_lens, z_source, opening_angle, angle_pad)
        #elif geometry_type == 'DOUBLE_CONE_CYLINDER':
        #    self._geometrytype = DoubleConeCylindner(cosmology, z_lens, z_source, opening_angle, angle_pad)
        else:
            raise Exception('geometry type '+str(geometry_type) + ' not recognized.')

        self.cosmo = cosmology
        self._zlens, self._zsource = z_lens, z_source
        self.cone_opening_angle = opening_angle
        self._arcsec = self.cosmo.arcsec
        self.kpc_per_arcsec_zlens = self.cosmo.kpc_proper_per_asec(self._zlens)
        self._reduced_to_phys = self._geometrytype._reduced_to_phys

    def rendering_scale(self, z):

        return self._geometrytype.rendering_scale(z)

    def kpc_per_arcsec(self, z):

        return self.cosmo.kpc_proper_per_asec(z)

    def angle_to_physicalradius(self, radius_arcsec, z):

        angle_radian = radius_arcsec * self._arcsec

        return angle_radian * self.rendering_scale(z) * self.cosmo.D_A_z(z)

    def angle_to_comovingradius(self, radius_arcsec, z):

        """
        This is specific to the geometry, e.g. if you pick cone if will close behind the main
        deflector
        """
        a_z = self.cosmo.scale_factor(z)
        return self.angle_to_physicalradius(radius_arcsec, z) / a_z

    def volume_element_comoving(self, z, delta_z, radius=None):
        """

        :param z: redshift
        :param delta_z: thickness of the redshift pancake
        :param radius: angular radius of the rendering area in arcseconds
        :return: volume element in comoving Mpc for small delta_z
        """

        if radius is None:
            radius = 0.5*self.cone_opening_angle

        if delta_z > self._delta_z_min:
            func = self._volume_integrand_comoving
            volume_element = quad(func, z, z+delta_z, args=(radius))[0]
        else:
            volume_element = self._volume_integrand_comoving(z, radius) * delta_z

        return volume_element

    def _volume_integrand_comoving(self, z, radius_arcsec):
        """

        :param theta:
        :param z_lens:
        :param z:
        :return: integrand element in comoving Mpc (for use in scipy.integrate quad)
        """
        area_comoving = self.angle_to_comoving_area(radius_arcsec, z)
        dR = self.cosmo.astropy.hubble_distance.value * self.cosmo.astropy.efunc(z) ** -1
        return area_comoving * dR

    def _angle_to_arcsec_area(self, radius_arcsec, z):

        theta = self._angle_to_arcsec_radius(radius_arcsec, z)

        return np.pi * theta ** 2

    def angle_to_comoving_area(self, radius_arcsec, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param radius_arcsec: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        r = self.angle_to_comovingradius(radius_arcsec, z)

        return np.pi * r ** 2

    def angle_to_physical_area(self, radius_arcsec, z):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param theta: lens cone opening angle in arcsec
        :param z: redshift of plane
        :param z_lens: redshift of main lens
        :return: comoving area
        """

        area_comoving = self.angle_to_comoving_area(radius_arcsec, z)

        a_z = self.cosmo.scale_factor(z)

        return area_comoving * a_z ** 2

    def _angle_to_arcsec_radius(self, radius_arcsec, z):

        r_co_mpc = self.angle_to_comovingradius(radius_arcsec, z)
        r_co_kpc = r_co_mpc * 1000
        asec_per_kpc = self.cosmo.astropy.arcsec_per_kpc_comoving(z).value

        return r_co_kpc * asec_per_kpc

class Cone(object):

    def __init__(self, cosmology, z_lens, z_source, opening_angle, angle_pad):

        self.cosmo = cosmology

        self.opening_angle_radians = opening_angle * cosmology.arcsec

        self.d_c_lens = cosmology.D_C_transverse(z_lens)

        self._reduced_to_phys = self.cosmo.D_A(0, z_source) / self.cosmo.D_A(z_lens, z_source)

        self.comoving_radius_cylinder = 0.5 * self.opening_angle_radians * self.d_c_lens

        self._zlens, self._zsource = z_lens, z_source

        self._total_volume = np.pi * self.comoving_radius_cylinder ** 2 * \
                             cosmology.D_C_transverse(self._zsource)

    def rendering_scale(self, z):
        """
        Determines the angular size of the area at redshift z where halos are rendered;
        for this geometry, the physical size of the cylinder remains constant, and is set by
        cone_opening_angle / 2 * D_z, where D_z is the distance to the main deflector

        :param z: redshift
        :return: a factor that scales the size of the rendering area
        """
        return 1.0

class Cylinder(object):

    def __init__(self, cosmology, z_lens, z_source, opening_angle):

        self.cosmo = cosmology

        self.opening_angle_radians = opening_angle * cosmology.arcsec

        self.d_c_lens = cosmology.D_C_transverse(z_lens)

        self._reduced_to_phys = self.cosmo.D_A(0, z_source) / self.cosmo.D_A(z_lens, z_source)

        self.comoving_radius_cylinder = 0.5 * self.opening_angle_radians * self.d_c_lens

        self._zlens, self._zsource = z_lens, z_source

        self._total_volume = np.pi * self.comoving_radius_cylinder ** 2 * \
                             cosmology.D_C_transverse(self._zsource)

    def rendering_scale(self, z):
        """
        Determines the angular size of the area at redshift z where halos are rendered;
        for this geometry, the physical size of the cylinder remains constant, and is set by
        cone_opening_angle / 2 * D_z, where D_z is the distance to the main deflector

        :param z: redshift
        :return: a factor that scales the size of the rendering area
        """
        d_c = self.cosmo.D_C_transverse(z)
        xi = self.d_c_lens/d_c

        return xi

class DoubleCone(object):

    def __init__(self, cosmology, z_lens, z_source, opening_angle, angle_pad):

        self.cosmo = cosmology

        self._angle_pad = angle_pad

        d_c_lens = self.cosmo.D_C_transverse(z_lens)
        d_c_lens_source = self.cosmo.D_C_transverse(z_source) - d_c_lens

        comoving_radius_zlens = 0.5 * opening_angle * self.cosmo.arcsec * d_c_lens

        self._reduced_to_phys = self.cosmo.D_A(0, z_source) / self.cosmo.D_A(z_lens, z_source)

        self._zlens, self._zsource = z_lens, z_source

        volume_foreground = (np.pi/3) * comoving_radius_zlens ** 2 * d_c_lens
        volume_background = (np.pi/3) * comoving_radius_zlens ** 2 * d_c_lens_source
        self._total_volume = volume_background + volume_foreground

    def rendering_scale(self, z):
        """
        Determines the angular size of the area at redshift z where halos are rendered;
        for this geometry, the angle stays constant for z < z_lens, and closes towards the source
        angle_pad < 1.0 ensures that the rendering area doesn't close exactly at the source, so that halos are
        rendered up to z = z_source
        :param z: redshift
        :return: a factor that scales the size of the rendering area
        """
        if z <= self._zlens:
            return 1.
        else:
            D_dz = self.cosmo.D_A(self._zlens, z)
            D_z = self.cosmo.D_A_z(z)
            ratio = D_dz / D_z

            return 1 - self._angle_pad * self._reduced_to_phys * ratio

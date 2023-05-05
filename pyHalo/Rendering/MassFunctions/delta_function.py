from copy import deepcopy
import numpy as np


class DeltaFunction(object):
    """
    This class generates masses from a delta function normalized with respect to a
    background density, a mass, and a volume

    number of objects = density * volume / mass
    """
    name = 'DELTA_FUNCTION'
    def __init__(self, mass, volume, rho, draw_poisson, *args, **kwargs):
        """

        :param mass: mass of objects to render
        :param volume: rendering volume
        :param rho: a density
        :param draw_poisson: whether or not to draw from a poisson distribution
        """
        self.volume = volume
        self.mass = mass
        self.rho = rho
        self.draw_poisson = draw_poisson
        self.n_mean = self.rho * self.volume / self.mass
        self.first_moment = self.mass

    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, kwargs_model):
        """
        Creates the class from arbitrary functions that specify the amplitude and slope of the
         halo mass function at different redshifts

        """
        astropy = geometry_class.cosmo.astropy
        density_to_MsunperMpc = geometry_class.cosmo.density_to_MsunperMpc
        rho_dm_crit = astropy.Odm(0.) * astropy.critical_density(z).value * density_to_MsunperMpc
        volume = geometry_class.volume_element_comoving(z, delta_z)
        rho = kwargs_model['LOS_normalization'] * kwargs_model['mass_fraction'] * rho_dm_crit
        return DeltaFunction(kwargs_model['mass'], volume, rho, kwargs_model['draw_poisson'])

    def draw(self):

        """
        :return: an array of masses
        """

        if self.draw_poisson:
            n = int(np.random.poisson(self.n_mean))
        else:
            n = int(np.round(self.n_mean))

        if n > 0:
            return np.array([self.mass] * n)
        else:
            return np.array([])

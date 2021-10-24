import numpy as np

class DeltaFunction(object):

    """
    This class generates masses from a delta function normalized with respect to a
    background density, a mass, and a volume

    number of objects = density * volume / mass
    """

    def __init__(self, mass, volume, rho, draw_poisson=True):
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

    def draw(self):

        """
        :return: an array of masses
        """
        n = self.rho * self.volume / self.mass
        if self.draw_poisson:
            n = int(np.random.poisson(n))
        else:
            n = int(np.round(n))

        if n > 0:
            return np.array([self.mass] * n)
        else:
            return np.array([])




from copy import deepcopy
import numpy as np


class Gaussian(object):
    """
    This class generates masses from a delta function normalized with respect to a
    background density, a mass, and a volume

    """
    name = 'GAUSSIAN'
    def __init__(self, n, mean, sigma, draw_poisson=True, *args, **kwargs):
        """
        :param n: normalization, also equal to the number of objects
        :param mean: mass of objects to render
        :param sigma: rendering volume
        :param draw poisson: whether or not to draw from a poisson distribution
        """
        self.mean = mean
        self.sigma = sigma
        self.n_mean = n
        self.first_moment = self.mean
        self.draw_poisson = draw_poisson

    def draw(self):

        """
        :return: an array of masses
        """

        if self.draw_poisson:
            n = int(np.random.poisson(self.n_mean))
        else:
            n = int(np.round(self.n_mean))

        if n > 0:
            log10_mass = np.random.normal(self.mean, self.sigma, n)
            m = 10 ** log10_mass
            return np.array(m)
        else:
            return np.array([])

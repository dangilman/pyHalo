import numpy as np

class BackgroundDensityDelta(object):

    def __init__(self, mass, volume, rho, draw_poisson=True):

        self.volume = volume
        self.mass = mass
        self.rho = rho
        self.draw_poisson = draw_poisson

    def draw(self):

        n = self.rho * self.volume / self.mass
        if self.draw_poisson:
            n = int(np.random.poisson(n))
        else:
            n = int(np.round(n))

        if n > 0:
            return np.array([self.mass] * n)
        else:
            return np.array([])




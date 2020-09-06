import numpy as np
from pyHalo.Rendering.MassFunctions.PowerLaw.broken_powerlaw import BrokenPowerLaw

class PiecewisePowerLaw(object):

    def __init__(self, plaws):

        self.plaws = plaws

    def theory_mass(self, mlow, mhigh):

        mass = 0
        for plaw in self.plaws:
            mass += plaw.theory_mass(mlow, mhigh)
        return mass

    def draw(self):

        masses = []
        for plaw in self.plaws:
            if len(masses) == 0:
                masses = plaw.draw()
            else:
                masses = np.append(masses, plaw.draw())

        return np.array(masses)

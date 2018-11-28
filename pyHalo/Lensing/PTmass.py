import numpy as np

class PTmassLensing(object):

    hybrid = False

    def __init__(self, lens_cosmo):
        self.lens_cosmo = lens_cosmo

    def params(self, x, y, mass, redshift):

        center_x = x
        center_y = y
        factor = self.lens_cosmo.point_mass_fac(redshift)
        theta_E = factor * np.sqrt(mass)

        return {'center_x':center_x, 'center_y': center_y, 'theta_E': theta_E}

import numpy as np

class PJaffeLensing(object):

    def __init__(self, lens_cosmo):
        self.lens_cosmo = lens_cosmo

    def params(self, x, y, mass, r_trunc):

        center_x = x
        center_y = y
        theta_E = self.lens_cosmo.PJaffe_norm(mass, r_trunc)

        return {'center_x':center_x, 'center_y': center_y,
                'sigma0': theta_E, 'Ra':0, 'Rs': r_trunc}

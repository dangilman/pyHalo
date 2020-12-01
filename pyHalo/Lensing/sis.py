import numpy as np

class SISLensing(object):

    hybrid = False

    lenstronomy_ID = 'SIS'

    def __init__(self, lens_cosmo):

        self._lens_cosmo = lens_cosmo

    def params(self, x, y, mass, redshift):

        center_x = x
        center_y = y

        factor_ref = self._lens_cosmo.point_mass_factor_z(0.5)
        factor = self._lens_cosmo.point_mass_factor
        factor_scale = factor / factor_ref
        theta_E_ref = 1.
        mref = 10**11

        theta_E = theta_E_ref * np.sqrt(mass / mref)
        theta_E *= factor_scale

        return {'center_x':center_x, 'center_y': center_y, 'theta_E': theta_E}, None

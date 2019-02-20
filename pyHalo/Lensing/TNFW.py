import numpy as np
from pyHalo.Halos.Profiles.nfw import NFW

class TNFWLensing(object):

    hybrid = False

    def __init__(self,lens_cosmo):

        self.lens_cosmo = NFW(lens_cosmo)

    def params(self, x, y, mass, concentration, redshift, r_trunc):

        Rs_angle, theta_Rs = self.lens_cosmo.nfw_physical2angle(mass, concentration, redshift)

        kwargs = {'theta_Rs':theta_Rs, 'Rs': Rs_angle,
                  'center_x':x, 'center_y':y, 'r_trunc':r_trunc}

        return kwargs

    def mass_finite(self, m200, c, z, tau):

        rho, Rs, r200 = self.lens_cosmo.NFW_params_physical(m200, c, z)

        t2 = tau ** 2

        return 4 * np.pi * Rs ** 3 * rho * t2 * (t2 + 1) ** -2 * (
                (t2 - 1) * np.log(tau) + np.pi * tau - (t2 + 1))


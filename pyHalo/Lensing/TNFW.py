from pyHalo.Lensing.NFW import NFWLensing
import numpy as np

class TNFWLensing(object):

    def __init__(self,lens_cosmo):

        self.lens_cosmo = lens_cosmo

    def params(self, x, y, mass, concentration, redshift, r_trunc):

        Rs_angle, theta_Rs = self.lens_cosmo.nfw_physical2angle(mass, concentration, redshift)

        kwargs = {'theta_Rs':theta_Rs, 'Rs': Rs_angle,
                  'center_x':x, 'center_y':y, 'r_trunc':r_trunc}

        return kwargs

    def mass_finite(self, m200, c, z, r_trunc):

        rho, Rs, r200 = self.lens_cosmo.NFW_params_physical(m200, c, z)

        Rs_arcsec = Rs * self.lens_cosmo.cosmo.kpc_per_asec(z) ** -1
        tau = r_trunc * Rs_arcsec ** -1
        t2 = tau ** 2

        return 4 * np.pi * Rs ** 3 * rho * t2 * (t2 + 1) ** -2 * (
                (t2 - 1) * np.log(tau) + np.pi * tau - (t2 + 1))

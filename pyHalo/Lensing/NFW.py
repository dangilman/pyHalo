import numpy as np

class NFWLensing(object):

    hybrid = False

    lenstronomy_ID = 'NFW'

    def __init__(self, lens_cosmo):

        self._lens_cosmo = lens_cosmo

    def params(self, x, y, mass, redshift, concentration):

        Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(mass, concentration, redshift)

        x, y = np.round(x, 4), np.round(y, 4)

        Rs_angle = np.round(Rs_angle, 6)
        theta_Rs = np.round(theta_Rs, 6)

        kwargs = {'alpha_Rs':theta_Rs, 'Rs': Rs_angle,
                  'center_x':x, 'center_y':y}

        return kwargs, None

class TNFWLensing(object):

    lenstronomy_ID = 'TNFW'

    def __init__(self, lens_cosmo):

        self._lens_cosmo = lens_cosmo

    def params(self, x, y, mass, redshift, concentration, r_trunc_kpc):

        Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(mass, concentration, redshift)

        x, y = np.round(x, 4), np.round(y, 4)

        Rs_angle = np.round(Rs_angle, 4)
        theta_Rs = np.round(theta_Rs, 6)
        r_trunc = r_trunc_kpc * self._lens_cosmo.cosmo.kpc_per_asec(redshift) ** -1

        kwargs = {'alpha_Rs': theta_Rs, 'Rs': Rs_angle,
                  'center_x': x, 'center_y': y, 'r_trunc': r_trunc}

        return kwargs, None
